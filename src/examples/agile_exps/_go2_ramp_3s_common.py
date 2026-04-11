"""
Shared settings for Go2 **3 s** locomotion with **0 → 1 m/s** (linear) or **0 → 1 rad/s** (yaw)
linear ramps, **12 cm** swing clearance, diagonal trot (60 nodes @ DT=0.05).
"""
import numpy as np

import utils as reprutils
from contact_scheduler import ContactScheduler
from cost_models import ConfigurationCost

HORIZON_S = 3.0
DT = 0.05
V_MAX_M_S = 1.0
OMEGA_MAX_RAD_S = 1.0
SWING_CLEARANCE_M = 0.12
# Strafe-only: lower swing (motion is purely ±y; no need for extra foot height).
SWING_CLEARANCE_STRAFE_M = 0.08

HIP_JOINT_NAMES = ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint")
HIP_CFG_WEIGHT = 2e-3
JOINT_CFG_WEIGHT = 5e-6


def make_trot_contact_scheduler_3s(model, contact_frame_dict, dt: float = DT) -> ContactScheduler:
    """Exactly ``round(HORIZON_S/dt)`` nodes (60 for 3 s @ 0.05 s)."""
    cs = ContactScheduler(model, dt=dt, contact_frame_dict=contact_frame_dict)
    cs.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)
    for _ in range(6):
        cs.add_phase(["l_foot", "r_gripper"], 0.2)
        cs.add_phase(["r_foot", "l_gripper"], 0.2)
    cs.add_phase(["l_foot", "r_gripper"], 0.2)
    cs.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)
    n = len(cs.contact_sequence_fnames)
    exp = int(round(HORIZON_S / dt))
    if n != exp:
        raise RuntimeError(
            f"Contact phases → {n} nodes, need {exp} for T={HORIZON_S}s, dt={dt}"
        )
    return cs


def configuration_cost_legs(model, q_pin) -> ConfigurationCost:
    nj = model.nv - 6
    W = np.eye(nj) * JOINT_CFG_WEIGHT
    for jn in HIP_JOINT_NAMES:
        iq = model.joints[model.getJointId(jn)].idx_q
        jtan = iq - 7
        W[jtan, jtan] = HIP_CFG_WEIGHT
    q_ref = reprutils.pin2rep(q_pin)
    return ConfigurationCost(q_ref[6:], W)
