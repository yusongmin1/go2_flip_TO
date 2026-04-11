"""
Go2 **right strafe** (world **−y**) over 3 s:

- **Pure lateral** motion: **0 → 1 m/s** along **y** only; **x** fixed in target / warm start.
- **Swing clearance** 8 cm (see ``SWING_CLEARANCE_STRAFE_M``).
"""
from _go2_ramp_3s_common import SWING_CLEARANCE_STRAFE_M
from _go2_translation_ramp_3s import run_translation_ramp_3s

if __name__ == "__main__":
    run_translation_ramp_3s(
        1,
        -1.0,
        save_solution_basename="quad_strafe_right_ramp_3s",
        export_run_name="quad_strafe_right_ramp_3s",
        log_prefix="quad_strafe_right_ramp",
        motion_label="strafe right (−y)",
        swing_clearance_m=SWING_CLEARANCE_STRAFE_M,
    )
