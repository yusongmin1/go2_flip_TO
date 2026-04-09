"""
Go2 forward locomotion with approximate average horizontal speed and swing-foot clearance.

- Target average base +x speed: TARGET_SPEED_M_S (default 1.0 m/s) over the planned horizon.
- Swing feet (not in contact): z >= terrain + SWING_CLEARANCE_M (default 0.10 m).
- Hip joints (FL/FR/RL/RR hip): extra weight in ConfigurationCost so they stay near 0 rad.

Gait: diagonal trot — stance on (RL+FR) then (RR+FL), using contact groups l_foot+r_gripper /
r_foot+l_gripper (see Go2Wrapper).
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

VIS = pars.VIS
DT = 0.05
TARGET_SPEED_M_S = 1.0
SWING_CLEARANCE_M = 0.10

# Pinocchio idx_q for *_hip_joint; joint part of tangent q is q_pin[idx_q] == q_tan[6 + (idx_q - 7)]
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

# 2 s horizon: 4 + 32 + 4 nodes at DT=0.05 → average |Δx|/T ≈ TARGET_SPEED if we set Δx = TARGET_SPEED * 2
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
dist_x = TARGET_SPEED_M_S * horizon_s

qf = np.copy(q)
qf[0] = q[0] + dist_x
qf[2] += terrain.height(qf[0], qf[1]) - terrain.height(q[0], q[1])

opti.set_initial_pose(q)
opti.set_target_pose(qf)

for k, node in enumerate(opti.nodes):
    alpha = k / max(n_nodes - 1, 1)
    qk = np.copy(q)
    qk[0] = q[0] + alpha * (qf[0] - q[0])
    qk[2] += terrain.height(qk[0], qk[1]) - terrain.height(q[0], q[1])
    opti.x0[node.q_id] = reprutils.pin2rep(qk)

result = opti.solve(200, 1e-3, False, print_level=0)
print(
    f"[quad_walk_forward] horizon={horizon_s:.2f}s, Δx≈{dist_x:.2f}m "
    f"(target avg {TARGET_SPEED_M_S} m/s), swing clearance {SWING_CLEARANCE_M * 100:.0f} cm, "
    f"hip joints penalized toward 0 (weight {HIP_CFG_WEIGHT})"
)
print(
    f"[quad_walk_forward] Planning time: {result['solve_time']:.4f} s "
    f"(IPOPT iterations: {result['iter_count']})"
)
opti.save_solution("quad_walk_forward_1mps")

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
