"""
Go2 backflip. Ground clearance + Go2 left/right leg **joint angle** symmetry + capped joint
velocities ``vq[6:]`` — see ``_go2_flip_ground_clearance``.
"""
import numpy as np
import time

from trajectory_optimization import NLTrajOpt
from contact_scheduler import ContactScheduler
from node import Node
from constraint_models import *
import utils as reprutils

from terrain.terrain_grid import TerrainGrid
from robots.go2.Go2Wrapper import Go2
from visualiser.visualiser import TrajoptVisualiser

from _go2_flip_ground_clearance import (
    FOOT_SWING_CLEARANCE_M,
    JOINT_VEL_ABS_MAX_RAD_S,
    STANCE_FOOT_CLEARANCE_M,
    apply_joint_velocity_cap,
    terrain_body_clearance_dict,
)

np.set_printoptions(precision=2, suppress=False)

import params as pars

VIS = pars.VIS
DT = 0.02

terrain_body_clearance = terrain_body_clearance_dict()

robot = Go2()
q = robot.go_neutral()

terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()


contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
    "l_gripper": robot.left_gripper_frames,
    "r_gripper": robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)


contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 1.0)
k1 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase([], 0.4)
k2 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 1.0)


frame_contact_seq = contact_scheduler.contact_sequence_fnames


contact_frame_names = robot.left_foot_frames + robot.right_foot_frames + robot.left_gripper_frames + robot.right_gripper_frames


stages = []
K = len(frame_contact_seq)
print("K = ", K)
for k, contact_phase_fnames in enumerate(frame_contact_seq):
    stage_node = Node(
        nv=robot.model.nv,
        contact_phase_fnames=contact_phase_fnames,
        contact_fnames=contact_frame_names,
        terrain_body_clearance=terrain_body_clearance,
        go2_lr_leg_symmetry=True,
    )

    dyn_const = WholeBodyDynamics()
    stage_node.dynamics_type = dyn_const.name

    stage_node.constraints_list.extend(
        [
            dyn_const,
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
            SemiEulerIntegration(),
            TerrainGridContactConstraints(
                terrain,
                swing_min_clearance=FOOT_SWING_CLEARANCE_M,
                stance_min_clearance=STANCE_FOOT_CLEARANCE_M,
            ),
            TerrainBodyClearanceConstraints(terrain),
            Go2LeftRightLegSymmetryConstraints(robot.model),
            TerrainGridFrictionConstraints(terrain),
        ]
    )

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q)
qf = np.copy(q)
# slight sagittal offset for landing target (same idea as sideflip’s qf[1] tweak)
qf[0] = -0.35
opti.set_target_pose(qf)
apply_joint_velocity_cap(opti, JOINT_VEL_ABS_MAX_RAD_S)

# warm start: full 2*pi rotation in pitch (backflip); sideflip uses [theta, 0, 0] on roll
for k, node in enumerate(opti.nodes):
    if k1 <= k <= k2:
        theta = -2 * np.pi * (k - k1) / (k2 - k1)
        opti.x0[node.q_id] = reprutils.rpy2rep(q, [0.0, theta, 0.0])


result = opti.solve(
    200,
    7e-3,
    False,
    print_level=0,
    accept_max_iter_exceeded=True,
)
if result.get("warning"):
    print(f"[quad_backflip] WARNING: {result['warning']}")
print(f"[quad_backflip] Planning time: {result['solve_time']:.4f} s (IPOPT iterations: {result['iter_count']})")
opti.save_solution("quad_backflip")


K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])

    time.sleep(1)
    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)
