"""
Shared ground-avoidance parameters for Go2 flip scripts (front/back/side).

Constraints are **point checks** on Pinocchio link frames vs analytic terrain height.
Collision meshes can extend beyond these frames; margins are set conservatively so the
optimized trajectory stays visually above z=0 in MeshCat for typical landing poses.
"""
from __future__ import annotations

from typing import Any, Dict

# Cap joint-rate magnitude on ``vq[6:]`` (rad/s), intersected with URDF limits.
JOINT_VEL_ABS_MAX_RAD_S = 18.0

# Hard cap on the **base** angular velocity norm ||vq[3:6]|| (rad/s) — the flip rotation
# rate. Without it the whole 2*pi unrolls inside the flight phase (~15.7 rad/s for a 0.4 s
# flight). A full flip at 12 rad/s needs >= 0.524 s of rotation, so flight phases must be
# ~0.55 s (see the flip scripts).
BASE_ANGULAR_VEL_MAX_RAD_S = 12.0

# Flight: non-contact feet
FOOT_SWING_CLEARANCE_M = 0.14
# Stance: contact z = terrain + offset (sole / mesh below foot frame)
STANCE_FOOT_CLEARANCE_M = 0.00
# Base link origin above ground (body box extends below origin ~5–6 cm in URDF)
BASE_MIN_CLEARANCE_M = 0.14
# Hip / thigh / calf origins above ground (deep knee bend on landing)
LEG_LINK_CLEARANCE_M = 0.09

# All leg kinematic frames that should stay above the plane when crouched
LEG_SEGMENT_FRAMES = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)


def terrain_body_clearance_dict() -> Dict[str, float]:
    d = {"base": BASE_MIN_CLEARANCE_M}
    for fn in LEG_SEGMENT_FRAMES:
        d[fn] = LEG_LINK_CLEARANCE_M
    return d


# --- Mesh collision (Go2MeshCollisionConstraints) parameters -----------------------------
# Required separation between mesh convex hulls (self collision) and hull-vs-ground.
MESH_SELF_COLLISION_MARGIN_M = 0.002
MESH_GROUND_MARGIN_M = 0.001
# Slack used when auto-lowering margins from the reference pose (see constraint module).
MESH_AUTO_SLACK_M = 0.0005
# Extra height (beyond the measured mesh depth) at which stance foot *frames* are pinned,
# so the foot meshes rest on — not below — the ground plane.
FOOT_STANCE_MESH_EXTRA_M = 0.002

_FOOT_MESH_STANDOFF_CACHE: Dict[str, float] = {}


def go2_foot_mesh_standoff_m() -> float:
    """Height of the lowest foot-mesh point below the foot frames at the neutral pose.

    ``STANCE_FOOT_CLEARANCE_M`` should be at least this value (+ extra): the URDF foot
    frame sits above the mesh sole, so pinning frames to the terrain pushes the visible
    foot meshes through the ground.
    """
    if "standoff" in _FOOT_MESH_STANDOFF_CACHE:
        return _FOOT_MESH_STANDOFF_CACHE["standoff"]

    import os
    import sys

    examples_dir = os.path.dirname(os.path.abspath(__file__))
    for p in (examples_dir, os.path.join(os.path.dirname(examples_dir), "nltrajopt")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    import pinocchio as pin
    from robots.go2.Go2Wrapper import Go2

    robot = Go2()
    q = robot.go_neutral()
    pin.forwardKinematics(robot.model, robot.data, q)

    worst = 0.0
    for go in robot.visual_model.geometryObjects:
        if "_foot_" not in go.name:
            continue
        frame_name = go.name.rsplit("_", 1)[0]
        frame_z = robot.data.oMf[robot.model.getFrameId(frame_name)].translation[2]
        go.geometry.buildConvexHull(True, "Qt")
        conv = go.geometry.convex
        M = robot.data.oMi[go.parentJoint] * go.placement
        lowest = min(
            (M.rotation @ np.asarray(conv.point(i)))[2] + M.translation[2]
            for i in range(conv.num_points)
        )
        worst = max(worst, frame_z - lowest)
    _FOOT_MESH_STANDOFF_CACHE["standoff"] = worst
    return worst


def apply_joint_velocity_cap(opti: Any, cap_rad_s: float) -> None:
    """Intersect ``SemiEulerIntegration`` joint velocity box with ``±cap_rad_s`` on ``vq[6:]``."""
    for node in opti.nodes:
        for i in range(node.vq_id.start + 6, node.vq_id.stop):
            lo, hi = opti.lb[i], opti.ub[i]
            nlo = -cap_rad_s if lo is None else max(float(lo), -cap_rad_s)
            nhi = cap_rad_s if hi is None else min(float(hi), cap_rad_s)
            opti.lb[i] = nlo
            opti.ub[i] = nhi
