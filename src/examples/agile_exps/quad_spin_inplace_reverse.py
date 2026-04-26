"""
Go2 in-place turn, **opposite yaw direction** to ``quad_spin_inplace.py``:

3 s horizon, yaw rate **0 → −1 rad/s** (world +Z), 12 cm swing clearance.

Exports ``spin_inplace_ccw_50hz.txt`` (50 Hz) and trajectory under
``quad_spin_inplace_reverse_ramp_3s``.
"""
from _spin_inplace_ramp import run_spin_inplace_ramp

if __name__ == "__main__":
    run_spin_inplace_ramp(
        -1.0,
        save_solution_basename="quad_spin_inplace_reverse_ramp_3s",
        export_run_name="quad_spin_inplace_reverse_ramp_3s",
        mocap_filename="spin_inplace_ccw_50hz.txt",
        log_prefix="quad_spin_inplace_reverse",
    )
