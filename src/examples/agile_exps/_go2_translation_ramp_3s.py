"""3 s world-frame translation with speed ramp 0 → ``V_MAX_M_S`` along ``axis`` (0=x, 1=y)."""
import time

import numpy as np

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
    V_MAX_M_S,
    SWING_CLEARANCE_M,
    configuration_cost_legs,
    make_trot_contact_scheduler_3s,
)


def run_translation_ramp_3s(
    axis: int,
    direction: float,
    *,
    save_solution_basename: str,
    export_run_name: str,
    log_prefix: str,
    motion_label: str,
    swing_clearance_m: float | None = None,
) -> None:
    """
    ``axis`` 0 = world x, 1 = world y. ``direction`` ±1: sign of displacement along that axis.

    Integrated distance ``|direction| * V_MAX_M_S * HORIZON_S / 2`` (speed ``(t/T)*V_MAX``).

    For ``axis == 1`` (strafe), only **lateral** base motion is specified: initial/final and warm
    start keep the other horizontal coordinate fixed; the speed ramp applies only along ``y``.

    ``swing_clearance_m``: if ``None``, uses ``SWING_CLEARANCE_M`` (12 cm). Strafe scripts pass a
    lower value (see ``SWING_CLEARANCE_STRAFE_M`` in ``_go2_ramp_3s_common``).
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (x) or 1 (y)")
    direction = float(direction)
    if abs(abs(direction) - 1.0) > 1e-9:
        raise ValueError("direction must be +1.0 or -1.0")

    swing_m = SWING_CLEARANCE_M if swing_clearance_m is None else float(swing_clearance_m)

    repo_root = ensure_repo_root()
    VIS = pars.VIS

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
                TerrainGridContactConstraints(terrain, swing_min_clearance=swing_m),
            ]
        )
        stage_node.costs_list.extend([configuration_cost_legs(robot.model, q)])
        stages.append(stage_node)

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

    dist = direction * V_MAX_M_S * HORIZON_S / 2.0
    qf = np.copy(q)
    qf[axis] = q[axis] + dist
    qf[2] += terrain.height(qf[0], qf[1]) - terrain.height(q[0], q[1])

    opti.set_initial_pose(q)
    opti.set_target_pose(qf)

    for k, node in enumerate(opti.nodes):
        alpha = k / max(n_nodes - 1, 1)
        t_eff = alpha * HORIZON_S
        qk = np.copy(q)
        qk[axis] = q[axis] + direction * V_MAX_M_S * t_eff**2 / (2.0 * HORIZON_S)
        qk[2] += terrain.height(qk[0], qk[1]) - terrain.height(q[0], q[1])
        opti.x0[node.q_id] = reprutils.pin2rep(qk)

    result = opti.solve(250, 1e-3, False, print_level=0)
    ax_name = "x" if axis == 0 else "y"
    print(
        f"[{log_prefix}] T={HORIZON_S}s, swing={swing_m * 100:.0f} cm, "
        f"{motion_label} along {ax_name} ⇒ Δ{ax_name}≈{dist:+.3f} m "
        f"(|v| ramp 0→{V_MAX_M_S} m/s along {ax_name})"
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
            "axis": axis,
            "direction": direction,
            "horizon_s": HORIZON_S,
            "v_max_m_s": V_MAX_M_S,
            "swing_clearance_m": swing_m,
            "dt_nominal": DT,
        },
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
