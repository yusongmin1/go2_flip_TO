"""Core logic for 3 s in-place spin with yaw rate ramp 0 → ``OMEGA_MAX_RAD_S`` (sign = direction)."""
import time

import numpy as np
import pinocchio as pin

from trajectory_optimization import NLTrajOpt
from node import Node
from constraint_models import *
import utils as reprutils

from terrain.terrain_grid import TerrainGrid
from robots.go2.Go2Wrapper import Go2
from visualiser.visualiser import TrajoptVisualiser

import params as pars

from _export_go2_datasets import ensure_repo_root, export_go2_agile_trajectory
from _go2_ramp_3s_common import (
    DT,
    HORIZON_S,
    OMEGA_MAX_RAD_S,
    SWING_CLEARANCE_M,
    configuration_cost_legs,
    make_trot_contact_scheduler_3s,
)


def _pin_q_apply_world_yaw(
    model: pin.Model,
    data: pin.Data,
    q_init_pin: np.ndarray,
    q_pos_pin: np.ndarray,
    dyaw: float,
) -> np.ndarray:
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


def run_spin_inplace_ramp(
    yaw_sign: float,
    *,
    save_solution_basename: str,
    export_run_name: str,
    mocap_filename: str,
    log_prefix: str,
) -> None:
    """
    ``yaw_sign = +1`` or ``-1``: world +Z yaw integration direction.
    Total heading change ``= yaw_sign * OMEGA_MAX_RAD_S * HORIZON_S / 2``.
    """
    repo_root = ensure_repo_root()
    VIS = pars.VIS

    terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
    terrain.set_zero()

    robot = Go2()
    q = robot.go_neutral()
    data = robot.data

    contacts_dict = {
        "l_foot": robot.left_foot_frames,
        "r_foot": robot.right_foot_frames,
        "l_gripper": robot.left_gripper_frames,
        "r_gripper": robot.right_gripper_frames,
    }

    contact_scheduler = make_trot_contact_scheduler_3s(robot.model, contacts_dict, dt=DT)
    frame_contact_seq = contact_scheduler.contact_sequence_fnames
    contact_frame_names = (
        robot.left_foot_frames
        + robot.right_foot_frames
        + robot.left_gripper_frames
        + robot.right_gripper_frames
    )

    n_nodes = len(frame_contact_seq)
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
        stage_node.costs_list.extend([configuration_cost_legs(robot.model, q)])
        stages.append(stage_node)

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

    delta_yaw = yaw_sign * OMEGA_MAX_RAD_S * HORIZON_S / 2.0
    qf = _pin_q_apply_world_yaw(robot.model, data, q, q, delta_yaw)
    qf[0:3] = q[0:3].copy()
    qf[2] += terrain.height(qf[0], qf[1]) - terrain.height(q[0], q[1])

    opti.set_initial_pose(q)
    opti.set_target_pose(qf)

    for k, node in enumerate(opti.nodes):
        alpha = k / max(n_nodes - 1, 1)
        t_eff = alpha * HORIZON_S
        dyaw_k = yaw_sign * OMEGA_MAX_RAD_S * t_eff**2 / (2.0 * HORIZON_S)
        qk = _pin_q_apply_world_yaw(robot.model, data, q, q, dyaw_k)
        qk[0:3] = q[0:3].copy()
        qk[2] += terrain.height(qk[0], qk[1]) - terrain.height(q[0], q[1])
        opti.x0[node.q_id] = reprutils.pin2rep(qk)

    result = opti.solve(250, 1e-3, False, print_level=0)
    print(
        f"[{log_prefix}] T={HORIZON_S}s, swing={SWING_CLEARANCE_M * 100:.0f} cm, "
        f"yaw rate ramp 0→{yaw_sign * OMEGA_MAX_RAD_S:+.3f} rad/s ⇒ Δyaw≈{delta_yaw:+.3f} rad"
    )
    print(
        f"[{log_prefix}] Planning time: {result['solve_time']:.4f} s "
        f"(IPOPT iterations: {result['iter_count']})"
    )
    opti.save_solution(save_solution_basename)

    K = len(result["nodes"])
    dts = [result["nodes"][k]["dt"] for k in range(K)]
    qs = [result["nodes"][k]["q"] for k in range(K)]
    forces = [result["nodes"][k]["forces"] for k in range(K)]

    export_go2_agile_trajectory(
        repo_root,
        result,
        robot.model,
        export_run_name,
        extra_meta={
            "source_script": log_prefix + ".py",
            "yaw_sign": yaw_sign,
            "omega_max_rad_s": OMEGA_MAX_RAD_S,
            "horizon_s": HORIZON_S,
            "dt_nominal": DT,
            "swing_clearance_m": SWING_CLEARANCE_M,
        },
        mocap_filename=mocap_filename,
        log_prefix=log_prefix,
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
