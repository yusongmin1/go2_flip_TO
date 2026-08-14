"""
Go2 front flip (pitch +2π in flight). Same contact layout as quad_backflip; pitch sign and
qf[0] differ.

Ground avoidance:
- Swing feet: ``swing_min_clearance`` (flight).
- Stance feet: pin foot *frames* at the measured mesh-standoff height so the foot meshes
  (which sit below the URDF foot frames) rest on — not through — the plane.
- **Mesh collision** (``Go2MeshCollisionConstraints``): qhull convex hulls of the visual
  meshes; hard constraints on hpp-fcl signed distances (link-link and link-ground) at every
  node, with margins auto-calibrated from the fixed start pose.
- Go2 **left/right leg symmetry** on leg ``q`` (FL/FR, RL/RR), plus **joint velocity** cap on
  ``vq[6:]`` (same module).

The solve runs in an **isolated child process with automatic retry** (``_flip_solve_isolated``);
see that module for why. ``GO2_NO_ISOLATED=1`` reverts to an in-process solve.
"""
import os
import time

import numpy as np

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
    FOOT_STANCE_MESH_EXTRA_M,
    JOINT_VEL_ABS_MAX_RAD_S,
    MESH_AUTO_SLACK_M,
    MESH_GROUND_MARGIN_M,
    MESH_SELF_COLLISION_MARGIN_M,
    go2_foot_mesh_standoff_m,
    apply_joint_velocity_cap,
    terrain_body_clearance_dict,
)
from _flip_solve_isolated import solve_isolated

import params as pars

from _export_go2_datasets import ensure_repo_root, export_go2_agile_trajectory

_REPO_ROOT = ensure_repo_root()

VIS = pars.VIS
DT = 0.02


def build_and_solve():
    """Build the front-flip NLP (with mesh collision constraints) and solve it."""
    robot = Go2()
    q = robot.go_neutral()

    # Pin stance foot *frames* high enough that the foot meshes rest on the plane
    # (the mesh sole sits below the URDF foot frame).
    stance_clearance_m = go2_foot_mesh_standoff_m() + FOOT_STANCE_MESH_EXTRA_M
    q[2] += stance_clearance_m

    terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
    terrain.set_zero()

    # Mesh self-collision + mesh-vs-ground hard constraints; margins auto-calibrated from the
    # (fixed) start pose so the endpoints stay feasible. Set GO2_NO_MESH=1 to disable.
    if os.environ.get("GO2_NO_MESH", "").lower() not in ("1", "true", "yes"):
        mesh_collision = Go2MeshCollisionConstraints(
            robot,
            q,
            self_margin=MESH_SELF_COLLISION_MARGIN_M,
            ground_margin=MESH_GROUND_MARGIN_M,
            auto_slack=MESH_AUTO_SLACK_M,
            ground_z=0.0,
        )
        mesh_rows = mesh_collision.n_rows
    else:
        mesh_collision, mesh_rows = None, 0

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

    contact_frame_names = (
        robot.left_foot_frames
        + robot.right_foot_frames
        + robot.left_gripper_frames
        + robot.right_gripper_frames
    )

    terrain_body_clearance = terrain_body_clearance_dict()

    stages = []
    for contact_phase_fnames in frame_contact_seq:
        stage_node = Node(
            nv=robot.model.nv,
            contact_phase_fnames=contact_phase_fnames,
            contact_fnames=contact_frame_names,
            terrain_body_clearance=terrain_body_clearance,
            go2_lr_leg_symmetry=True,
            go2_mesh_collision_rows=mesh_rows,
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
                    stance_min_clearance=stance_clearance_m,
                ),
                TerrainBodyClearanceConstraints(terrain),
                Go2LeftRightLegSymmetryConstraints(robot.model),
                TerrainGridFrictionConstraints(terrain),
            ]
        )
        if mesh_collision is not None:
            stage_node.constraints_list.append(mesh_collision)

        stages.append(stage_node)

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

    opti.set_initial_pose(q)
    qf = np.copy(q)
    qf[0] = 0.35
    opti.set_target_pose(qf)
    apply_joint_velocity_cap(opti, JOINT_VEL_ABS_MAX_RAD_S)

    for k, node in enumerate(opti.nodes):
        if k1 <= k <= k2:
            theta = 2 * np.pi * (k - k1) / (k2 - k1)
            opti.x0[node.q_id] = reprutils.rpy2rep(q, [0.0, theta, 0.0])

    # +pitch flip is a harder NLP than backflip for Go2: more iterations, looser tol, optional last-iterate fallback
    result = opti.solve(
        200,
        7e-3,
        False,
        print_level=0,
        accept_max_iter_exceeded=True,
    )
    if result.get("warning"):
        print(f"[quad_frontflip] WARNING: {result['warning']}")
    print(
        f"[quad_frontflip] Planning time: {result['solve_time']:.4f} s "
        f"(IPOPT iterations: {result['iter_count']})"
    )
    return result


result = solve_isolated(build_and_solve, max_attempts=4, tag="quad_frontflip")

export_go2_agile_trajectory(
    _REPO_ROOT,
    result,
    Go2().model,
    "quad_frontflip",
    extra_meta={"source_script": "quad_frontflip.py", "dt_nominal": DT},
    log_prefix="quad_frontflip",
)

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    robot = Go2()
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])

    time.sleep(1)
    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)
