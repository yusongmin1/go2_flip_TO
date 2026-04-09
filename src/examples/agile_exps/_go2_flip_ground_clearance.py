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

# Flight: non-contact feet
FOOT_SWING_CLEARANCE_M = 0.14
# Stance: contact z = terrain + offset (sole / mesh below foot frame)
STANCE_FOOT_CLEARANCE_M = 0.05
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


def apply_joint_velocity_cap(opti: Any, cap_rad_s: float) -> None:
    """Intersect ``SemiEulerIntegration`` joint velocity box with ``±cap_rad_s`` on ``vq[6:]``."""
    for node in opti.nodes:
        for i in range(node.vq_id.start + 6, node.vq_id.stop):
            lo, hi = opti.lb[i], opti.ub[i]
            nlo = -cap_rad_s if lo is None else max(float(lo), -cap_rad_s)
            nhi = cap_rad_s if hi is None else min(float(hi), cap_rad_s)
            opti.lb[i] = nlo
            opti.ub[i] = nhi
