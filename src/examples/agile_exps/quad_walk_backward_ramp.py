"""
Go2 **backward** walk with:

- **Swing foot clearance** 12 cm above terrain when not in contact.
- **Speed ramp**: backward speed magnitude grows **linearly in time** from 0 to ``V_MAX_M_S``
  (default 1.0 m/s) over the horizon; base **x** target uses the corresponding displacement
  ``Δx = -V_MAX * T / 2`` (integral of ``v(t) = (t/T) * V_MAX`` along ``-x``).
- **Horizon** ``HORIZON_S = 3`` s, fixed ``sum(dt) = 3`` via ``TimeConstraint``.

Gait: same diagonal trot rhythm as ``quad_walk_forward``, phase durations chosen so
``int(phase/dt)`` steps sum to ``HORIZON_S / DT`` (60 nodes at ``DT = 0.05``).
"""
import time

import numpy as np

from trajectory_optimization import NLTrajOpt
from contact_scheduler import ContactScheduler
from node import Node
from constraint_models import *
from cost_models import *
import utils as reprutils

from terrain.terrain_grid import TerrainGrid
from robots.go2.Go2Wrapper import Go2
from visualiser.visualiser import TrajoptVisualiser

import params as pars

from _export_go2_datasets import ensure_repo_root, export_go2_agile_trajectory

_REPO_ROOT = ensure_repo_root()

VIS = pars.VIS
DT = 0.05
HORIZON_S = 3.0
V_MAX_M_S = 1.0
SWING_CLEARANCE_M = 0.12

HIP_JOINT_NAMES = ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint")
HIP_CFG_WEIGHT = 2e-3
JOINT_CFG_WEIGHT = 5e-6

terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()

robot = Go2()
q = robot.go_neutral()

nj = robot.model.nv - 6
W_cfg = np.eye(nj) * JOINT_CFG_WEIGHT
for jn in HIP_JOINT_NAMES:
    iq = robot.model.joints[robot.model.getJointId(jn)].idx_q
    jtan = iq - 7
    W_cfg[jtan, jtan] = HIP_CFG_WEIGHT

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
    "l_gripper": robot.left_gripper_frames,
    "r_gripper": robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)

# 3.0 s / 0.05 = 60 nodes: 4 + 48 + 4 + 4
contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)
for _ in range(6):
    contact_scheduler.add_phase(["l_foot", "r_gripper"], 0.2)
    contact_scheduler.add_phase(["r_foot", "l_gripper"], 0.2)
contact_scheduler.add_phase(["l_foot", "r_gripper"], 0.2)
contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)

frame_contact_seq = contact_scheduler.contact_sequence_fnames
contact_frame_names = (
    robot.left_foot_frames
    + robot.right_foot_frames
    + robot.left_gripper_frames
    + robot.right_gripper_frames
)

n_nodes = len(frame_contact_seq)
_expected_k = int(round(HORIZON_S / DT))
if n_nodes != _expected_k:
    raise RuntimeError(
        f"Contact phases sum to {n_nodes} nodes, need {_expected_k} for T={HORIZON_S}s, DT={DT}"
    )

stages = []
for contact_phase_fnames in frame_contact_seq:
    stage_node = Node(
        nv=robot.model.nv,
        contact_phase_fnames=contact_phase_fnames,
        contact_fnames=contact_frame_names,
    )

    dyn_const = WholeBodyDynamics()
    stage_node.dynamics_type = dyn_const.name

    stage_node.constraints_list.extend(
        [
            dyn_const,
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=HORIZON_S),
            SemiEulerIntegration(),
            TerrainGridFrictionConstraints(terrain, max_delta_force=80.0),
            TerrainGridContactConstraints(terrain, swing_min_clearance=SWING_CLEARANCE_M),
        ]
    )

    q_ref = reprutils.pin2rep(q)
    stage_node.costs_list.extend([ConfigurationCost(q_ref[6:], W_cfg)])

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

# Backward displacement for v(t) = (t/T)*V_MAX along -x:  ∫_0^T v dt = V_MAX*T/2
dist_x = V_MAX_M_S * HORIZON_S / 2.0

qf = np.copy(q)
qf[0] = q[0] - dist_x
qf[2] += terrain.height(qf[0], qf[1]) - terrain.height(q[0], q[1])

opti.set_initial_pose(q)
opti.set_target_pose(qf)

for k, node in enumerate(opti.nodes):
    alpha = k / max(n_nodes - 1, 1)
    t_eff = alpha * HORIZON_S
    x_t = q[0] - V_MAX_M_S * t_eff**2 / (2.0 * HORIZON_S)
    qk = np.copy(q)
    qk[0] = x_t
    qk[2] += terrain.height(qk[0], qk[1]) - terrain.height(q[0], q[1])
    opti.x0[node.q_id] = reprutils.pin2rep(qk)

result = opti.solve(250, 1e-3, False, print_level=0)
print(
    f"[quad_walk_backward_ramp] T={HORIZON_S}s, swing={SWING_CLEARANCE_M * 100:.0f} cm, "
    f"backward speed ramp 0→{V_MAX_M_S} m/s ⇒ Δx≈{-dist_x:.2f} m"
)
print(
    f"[quad_walk_backward_ramp] Planning time: {result['solve_time']:.4f} s "
    f"(IPOPT iterations: {result['iter_count']})"
)
opti.save_solution("quad_walk_backward_ramp_3s")

export_go2_agile_trajectory(
    _REPO_ROOT,
    result,
    robot.model,
    "quad_walk_backward_ramp_3s",
    extra_meta={
        "source_script": "quad_walk_backward_ramp.py",
        "horizon_s": HORIZON_S,
        "v_max_m_s": V_MAX_M_S,
        "swing_clearance_m": SWING_CLEARANCE_M,
        "dt_nominal": DT,
    },
    log_prefix="quad_walk_backward_ramp",
)

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    tvis.load_terrain(terrain)

    time.sleep(1)
    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)
