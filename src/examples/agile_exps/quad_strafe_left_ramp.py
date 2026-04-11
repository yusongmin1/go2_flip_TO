"""
Go2 **left strafe** (world **+y**, REP-103 style) over 3 s:

- **Pure lateral** motion: speed ramp **0 → 1 m/s** only along **y**; base **x** unchanged in
  target and warm start (no forward “line speed” component in the task).
- **Swing clearance** lower than forward/back (8 cm) — enough for side-step, not as high as 12 cm.
"""
from _go2_ramp_3s_common import SWING_CLEARANCE_STRAFE_M
from _go2_translation_ramp_3s import run_translation_ramp_3s

if __name__ == "__main__":
    run_translation_ramp_3s(
        1,
        1.0,
        save_solution_basename="quad_strafe_left_ramp_3s",
        export_run_name="quad_strafe_left_ramp_3s",
        log_prefix="quad_strafe_left_ramp",
        motion_label="strafe left (+y)",
        swing_clearance_m=SWING_CLEARANCE_STRAFE_M,
    )
