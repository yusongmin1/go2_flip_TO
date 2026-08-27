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
    STAND_DWELL_TIME_S,
    stand_dwell_costs,
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

    # 起始站稳 0.4 s → 下蹲 0.5 s（只许蹲）→ 蓄力 0.1 s（最低点）→ 腾空 0.5 s
    # （向前上方翻：起跳带前向水平速度，绕质心角速度比原地翻低，需要更长腾空）
    # → 落地缓冲 0.6 s → 结束站稳 0.4 s
    contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], STAND_DWELL_TIME_S)
    k0 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.5)
    k_load = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.1)
    k1 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase([], 0.5)
    k2 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 1.0 - STAND_DWELL_TIME_S)
    k3 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], STAND_DWELL_TIME_S)

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

    # 首尾站稳段：构型拉向默认姿态 + 关节速度惩罚
    _cfg_cost, _vel_cost = stand_dwell_costs(robot.model, q)
    for _k in list(range(0, k0)) + list(range(k3, len(stages))):
        stages[_k].costs_list.extend([_cfg_cost, _vel_cost])

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

    opti.set_initial_pose(q)
    qf = np.copy(q)
    # 向前上方翻：起跳带前向速度，落点前移
    qf[0] = 0.4
    opti.set_target_pose(qf)
    apply_joint_velocity_cap(opti, JOINT_VEL_ABS_MAX_RAD_S)

    # warm start（阶段 A 用）：俯仰角线性展开 +2π（frontflip），叠加前向平移
    for k, node in enumerate(opti.nodes):
        if k1 <= k <= k2:
            theta = 2 * np.pi * (k - k1) / (k2 - k1)
            opti.x0[node.q_id] = reprutils.rpy2rep(q, [0.0, theta, 0.0])
            opti.x0[node.q_id][0] = qf[0] * (k - k1) / (k2 - k1)

    def _solve_chained(tag, max_rounds=3):
        """≤300 轮/次的续算链：未收敛则 x0=当前解接着算"""
        res = None
        for _round in range(max_rounds):
            res = opti.solve(
                300,
                7e-3,
                False,
                print_level=0,
                accept_max_iter_exceeded=True,
            )
            print(
                f"[quad_frontflip] {tag} round {_round + 1}: {res['solve_time']:.1f} s "
                f"({res['iter_count']} iters, warning={res.get('warning')})"
            )
            if not res.get("warning"):
                return res
            opti.x0 = np.asarray(opti.sol, dtype=float).copy()
        return res

    # ---- 阶段 A：不启用下蹲时序约束，解"原"问题 ----
    result_a = _solve_chained("stage A")

    # ---- 阶段 B：下蹲时序约束（thigh 单调递增坡道下界 0.8→1.3）重解 ----
    _n_crouch = max(k_load - k0, 1)
    _THIGH_END = 1.3
    for _k in range(k0, k_load):
        _node = opti.nodes[_k]
        _s = (_k - k0 + 1) / _n_crouch
        _th_lo = 0.8 + (_THIGH_END - 0.8) * _s
        for _jn in ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"):
            _iq = robot.model.joints[robot.model.getJointId(_jn)].idx_q
            _ti = _node.q_id.start + 6 + (_iq - 7)
            opti.lb[_ti] = max(opti.lb[_ti] if opti.lb[_ti] is not None else -np.inf, _th_lo)

    opti.x0 = np.asarray(opti.sol, dtype=float).copy()
    for _k in range(0, k_load):
        _node = opti.nodes[_k]
        for _jn in ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"):
            _iq = robot.model.joints[robot.model.getJointId(_jn)].idx_q
            _ti = _node.q_id.start + 6 + (_iq - 7)
            opti.x0[_ti] = np.clip(opti.x0[_ti], opti.lb[_ti], opti.ub[_ti])

    result = _solve_chained("stage B")
    if result.get("warning"):
        print(f"[quad_frontflip] WARNING: {result['warning']}")
    return result


result = solve_isolated(build_and_solve, max_attempts=4, tag="quad_frontflip")

export_go2_agile_trajectory(
    _REPO_ROOT,
    result,
    Go2().model,
    "quad_frontflip",
    extra_meta={"source_script": "quad_frontflip.py", "dt_nominal": DT},
    log_prefix="quad_frontflip",
    # the solver already pins stance feet at the measured mesh standoff — the legacy
    # +0.022 export offset would double-count it and leave the feet floating
    base_z_offset=0.0,
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
