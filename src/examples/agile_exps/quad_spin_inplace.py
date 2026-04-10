"""
Go2 in-place turning: target average yaw rate about world +Z (default 0.5 rad/s).

Uses the same diagonal trot contact rhythm as ``quad_walk_forward``. Initial and final base
position (x, y, z) match; final yaw is initial yaw + YAW_RATE_RAD_S * horizon_s.

Gait / costs mirror the walk demo (swing clearance, hip regularization).

After a successful solve, exports datasets like other ``agile_exps`` Go2 scripts (see
``_export_go2_datasets``). AMP file keeps the name ``spin_inplace_25hz.txt`` for compatibility.
"""
import time

import numpy as np
import pinocchio as pin

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
# World-frame yaw rate (rad/s); total heading change over the horizon = rate * (N * DT)
YAW_RATE_RAD_S = 0.5
SWING_CLEARANCE_M = 0.10

HIP_JOINT_NAMES = ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint")
HIP_CFG_WEIGHT = 2e-3
JOINT_CFG_WEIGHT = 5e-6


def pin_q_apply_world_yaw(
    model: pin.Model,
    data: pin.Data,
    q_init_pin: np.ndarray,
    q_pos_pin: np.ndarray,
    dyaw: float,
) -> np.ndarray:
    """Base rotation ``R = R_z(dyaw) @ R_init`` with ``R_init`` from ``q_init_pin``; translation from ``q_pos_pin``."""
    qf = np.copy(q_pos_pin)
    pin.forwardKinematics(model, data, q_init_pin)
    jid = model.getJointId("floating_base_joint")
    R_init = data.oMi[jid].rotation
    R_new = pin.rpy.rpyToMatrix(0.0, 0.0, dyaw) @ R_init
    Q = pin.Quaternion(R_new)
    if hasattr(Q, "coeffs"):
        coef = np.asarray(Q.coeffs()).reshape(-1)
    elif hasattr(Q, "vector"):
        coef = np.asarray(Q.vector).reshape(-1)
    else:
        coef = np.array([Q.x, Q.y, Q.z, Q.w], dtype=float)
    qf[3:7] = coef
    return qf


terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()

robot = Go2()
q = robot.go_neutral()
data = robot.data

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

contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)
for _ in range(4):
    contact_scheduler.add_phase(["l_foot", "r_gripper"], 0.2)
    contact_scheduler.add_phase(["r_foot", "l_gripper"], 0.2)
contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)

frame_contact_seq = contact_scheduler.contact_sequence_fnames
contact_frame_names = (
    robot.left_foot_frames
    + robot.right_foot_frames
    + robot.left_gripper_frames
    + robot.right_gripper_frames
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
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
            SemiEulerIntegration(),
            TerrainGridFrictionConstraints(terrain, max_delta_force=80.0),
            TerrainGridContactConstraints(terrain, swing_min_clearance=SWING_CLEARANCE_M),
        ]
    )

    q_ref = reprutils.pin2rep(q)
    stage_node.costs_list.extend([ConfigurationCost(q_ref[6:], W_cfg)])

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

n_nodes = len(stages)
horizon_s = n_nodes * DT
delta_yaw = YAW_RATE_RAD_S * horizon_s

qf = pin_q_apply_world_yaw(robot.model, data, q, q, delta_yaw)
qf[0:3] = q[0:3].copy()
qf[2] += terrain.height(qf[0], qf[1]) - terrain.height(q[0], q[1])

opti.set_initial_pose(q)
opti.set_target_pose(qf)

for k, node in enumerate(opti.nodes):
    alpha = k / max(n_nodes - 1, 1)
    qk = pin_q_apply_world_yaw(robot.model, data, q, q, alpha * delta_yaw)
    qk[0:3] = q[0:3].copy()
    qk[2] += terrain.height(qk[0], qk[1]) - terrain.height(q[0], q[1])
    opti.x0[node.q_id] = reprutils.pin2rep(qk)

result = opti.solve(200, 1e-3, False, print_level=0)
print(
    f"[quad_spin_inplace] horizon={horizon_s:.2f}s, Δyaw≈{delta_yaw:.3f} rad "
    f"(target avg {YAW_RATE_RAD_S} rad/s about +Z), swing clearance {SWING_CLEARANCE_M * 100:.0f} cm"
)
print(
    f"[quad_spin_inplace] Planning time: {result['solve_time']:.4f} s "
    f"(IPOPT iterations: {result['iter_count']})"
)
_rate_tag = f"{YAW_RATE_RAD_S:.2f}".rstrip("0").rstrip(".").replace(".", "p")
opti.save_solution(f"quad_spin_inplace_{_rate_tag}rads")

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

export_go2_agile_trajectory(
    _REPO_ROOT,
    result,
    robot.model,
    "quad_spin_inplace",
    extra_meta={
        "source_script": "quad_spin_inplace.py",
        "yaw_rate_rad_s": YAW_RATE_RAD_S,
        "horizon_s": float(n_nodes * DT),
        "dt_nominal": DT,
    },
    mocap_filename="spin_inplace_25hz.txt",
    log_prefix="quad_spin_inplace",
)

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
