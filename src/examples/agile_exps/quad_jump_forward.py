"""
Go2 forward jump along world +x (see JUMP_FORWARD_M). Default 1.0 m.
Contact pattern: four feet (+ “grippers” as front feet) compressed, flight, then landing.
"""
import time

import numpy as np

from trajectory_optimization import NLTrajOpt
from contact_scheduler import ContactScheduler
from node import Node
from constraint_models import *
from cost_models import *

from terrain.terrain_grid import TerrainGrid
from robots.go2.Go2Wrapper import Go2
from visualiser.visualiser import TrajoptVisualiser

import params as pars

VIS = pars.VIS
DT = 0.02
JUMP_FORWARD_M = 1.0

terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()

robot = Go2()
q = robot.go_neutral()

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
    "l_gripper": robot.left_gripper_frames,
    "r_gripper": robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)

# Prep on ground -> flight -> land (longer flight helps larger horizontal jumps)
contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 1.0)
contact_scheduler.add_phase([], 0.4 if JUMP_FORWARD_M >= 1.0 else 0.3)
contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 1.0)

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
            TerrainGridFrictionConstraints(terrain, max_delta_force=100.0),
            TerrainGridContactConstraints(terrain),
        ]
    )

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q)
qf = np.copy(q)
qf[0] = q[0] + JUMP_FORWARD_M
qf[2] += terrain.height(qf[0], qf[1]) - terrain.height(q[0], q[1])
opti.set_target_pose(qf)

result = opti.solve(600, 1e-3, False, print_level=0)
_save_tag = f"{JUMP_FORWARD_M:.2f}".rstrip("0").rstrip(".").replace(".", "p")
print(
    f"[quad_jump_forward {JUMP_FORWARD_M:g} m] Planning time: {result['solve_time']:.4f} s "
    f"(IPOPT iterations: {result['iter_count']})"
)
opti.save_solution(f"quad_jump_forward_{_save_tag}m")

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
