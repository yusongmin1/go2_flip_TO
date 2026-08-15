"""
Go2 **right** sideflip (roll +2π in flight) — the mirror of ``quad_sideflip``
(left roll -2π, landing offset +y): warm start rolls about +x, target offsets -y.

Same constraint stack as the front/back flips (see ``_go2_flip_ground_clearance`` and the
``quad_backflip`` docstring): mesh self-collision + mesh-vs-ground hard constraints, base
angular velocity cap, stance feet pinned at the measured mesh standoff — **but no
left/right leg symmetry constraint**: a roll is a lateral motion, the leading and trailing
legs do different things, so forcing FL==FR / RL==RR would be wrong here.

Flight-only ``ConfigurationCost`` pulls the legs toward a tucked aerial reference — the
soft tuck works *with* the mesh-collision hard constraints (pull legs in, but never
through another link).

The solve runs in an isolated child process with retry (``_flip_solve_isolated``).
"""
import os
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

from _go2_flip_ground_clearance import (
    BASE_ANGULAR_VEL_MAX_RAD_S,
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

np.set_printoptions(precision=2, suppress=False)

import params as pars

from _export_go2_datasets import ensure_repo_root, export_go2_agile_trajectory

_REPO_ROOT = ensure_repo_root()

VIS = pars.VIS
DT = 0.02

# --- Flight segment only (no contacts); stronger = legs pulled harder toward tuck reference ---
_FLIGHT_HIP_CFG_WEIGHT = 2.5e-2
_FLIGHT_THIGH_CFG_WEIGHT = 2.0e-2
_FLIGHT_CALF_CFG_WEIGHT = 2.0e-2
_FLIGHT_OTHER_JOINT_WEIGHT = 8e-5


def _air_tuck_pin_configuration(model, q_stand_pin: np.ndarray) -> np.ndarray:
    """Pin ``q`` with legs more tucked under body than ``go_neutral`` (for aerial reference only)."""
    q = np.array(q_stand_pin, dtype=np.float64).copy()
    for jn in ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"):
        iq = model.joints[model.getJointId(jn)].idx_q
        q[iq] = 0.0
    for jn in ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"):
        iq = model.joints[model.getJointId(jn)].idx_q
        q[iq] = 1.22
    for jn in ("FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"):
        iq = model.joints[model.getJointId(jn)].idx_q
        q[iq] = -2.28
    return q


def _flight_leg_tuck_configuration_cost(model, q_tuck_pin: np.ndarray) -> ConfigurationCost:
    """Soft pull of leg joints toward tucked aerial pose during flight only."""
    q_ref = reprutils.pin2rep(q_tuck_pin)
    nj = model.nv - 6
    W = np.eye(nj) * _FLIGHT_OTHER_JOINT_WEIGHT
    for jn in ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"):
        iq = model.joints[model.getJointId(jn)].idx_q
        jtan = iq - 7
        W[jtan, jtan] = _FLIGHT_HIP_CFG_WEIGHT
    for jn in ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"):
        iq = model.joints[model.getJointId(jn)].idx_q
        jtan = iq - 7
        W[jtan, jtan] = _FLIGHT_THIGH_CFG_WEIGHT
    for jn in ("FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"):
        iq = model.joints[model.getJointId(jn)].idx_q
        jtan = iq - 7
        W[jtan, jtan] = _FLIGHT_CALF_CFG_WEIGHT
    return ConfigurationCost(q_ref[6:], W)


def build_and_solve():
    """Build the side-flip NLP and solve it."""
    terrain_body_clearance = terrain_body_clearance_dict()

    robot = Go2()
    q = robot.go_neutral()
    q_air_tuck = _air_tuck_pin_configuration(robot.model, q)

    # Pin stance foot *frames* high enough that the foot meshes rest on the plane.
    stance_clearance_m = go2_foot_mesh_standoff_m() + FOOT_STANCE_MESH_EXTRA_M
    q[2] += stance_clearance_m

    terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
    terrain.set_zero()

    # Mesh self-collision + mesh-vs-ground hard constraints. GO2_NO_MESH=1 disables.
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
    # 0.55 s flight: with the 12 rad/s base-rotation cap a full 2*pi needs >= 0.524 s of
    # rotation, so the flip cannot unroll inside the old 0.4 s flight
    contact_scheduler.add_phase([], 0.55)
    k2 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 1.0)

    frame_contact_seq = contact_scheduler.contact_sequence_fnames
    contact_frame_names = robot.left_foot_frames + robot.right_foot_frames + robot.left_gripper_frames + robot.right_gripper_frames

    stages = []
    for contact_phase_fnames in frame_contact_seq:
        # NOTE: no go2_lr_leg_symmetry here — a roll is a lateral motion, the left and
        # right legs legitimately do different things during a sideflip.
        stage_node = Node(
            nv=robot.model.nv,
            contact_phase_fnames=contact_phase_fnames,
            contact_fnames=contact_frame_names,
            terrain_body_clearance=terrain_body_clearance,
            go2_mesh_collision_rows=mesh_rows,
            base_angular_velocity_limit=True,
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
                TerrainGridFrictionConstraints(terrain),
            ]
        )
        if mesh_collision is not None:
            stage_node.constraints_list.append(mesh_collision)
        # ||base angular velocity|| <= 12 rad/s: with the cap the flip needs a longer
        # flight phase (0.55 s) to unroll the full 2*pi
        stage_node.constraints_list.append(
            BaseAngularVelocityLimitConstraints(BASE_ANGULAR_VEL_MAX_RAD_S)
        )

        if len(contact_phase_fnames) == 0:
            stage_node.costs_list.append(_flight_leg_tuck_configuration_cost(robot.model, q_air_tuck))

        stages.append(stage_node)

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

    opti.set_initial_pose(q)
    apply_joint_velocity_cap(opti, JOINT_VEL_ABS_MAX_RAD_S)
    qf = np.copy(q)
    qf[1] = -0.3
    opti.set_target_pose(qf)

    # warm start: full +2*pi rotation in roll (right sideflip)
    for k, node in enumerate(opti.nodes):
        if k1 <= k <= k2:
            theta = 2 * np.pi * (k - k1) / (k2 - k1)
            opti.x0[node.q_id] = reprutils.rpy2rep(q, [theta, 0.0, 0.0])

    result = opti.solve(
        300,
        7e-3,
        False,
        print_level=0,
        accept_max_iter_exceeded=True,
    )
    if result.get("warning"):
        print(f"[quad_sideflip_right] WARNING: {result['warning']}")
    print(f"[quad_sideflip_right] Planning time: {result['solve_time']:.4f} s (IPOPT iterations: {result['iter_count']})")
    return result


result = solve_isolated(build_and_solve, max_attempts=4, tag="quad_sideflip_right")

export_go2_agile_trajectory(
    _REPO_ROOT,
    result,
    Go2().model,
    "quad_sideflip_right",
    extra_meta={"source_script": "quad_sideflip_right.py", "dt_nominal": DT},
    log_prefix="quad_sideflip_right",
    # the solver already pins stance feet at the measured mesh standoff — the legacy
    # +0.022 export offset would double-count it and leave the feet floating
    base_z_offset=0.0,
)

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
vs = [np.asarray(result["nodes"][k]["v"], dtype=np.float64).ravel() for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

# Go2 ``JointModelFreeFlyer``: v[0:3] = 机体系**线速度** (m/s), v[3:6] = 机体系**角速度** (rad/s)。
omega_b = np.stack([v[3:6] for v in vs], axis=0)
print(
    f"[quad_sideflip_right] 基座角速度范数 max={np.linalg.norm(omega_b, axis=1).max():.4f} rad/s "
    f"(上限 {BASE_ANGULAR_VEL_MAX_RAD_S}), {K} 个节点"
)

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
