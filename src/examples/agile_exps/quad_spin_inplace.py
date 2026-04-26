"""
Go2 in-place turn over **3 s**: yaw rate ramps **linearly 0 → 1 rad/s** (world +Z, one direction).

**Swing clearance** 12 cm; diagonal trot rhythm matches other 3 s ramp demos.

Exports like other ``agile_exps`` Go2 scripts; mocap uses ``spin_inplace_50hz.txt``.
Opposite direction: ``quad_spin_inplace_reverse.py``.
"""
from _spin_inplace_ramp import run_spin_inplace_ramp

if __name__ == "__main__":
    run_spin_inplace_ramp(
        1.0,
        save_solution_basename="quad_spin_inplace_ramp_3s",
        export_run_name="quad_spin_inplace_ramp_3s",
        mocap_filename="spin_inplace_50hz.txt",
        log_prefix="quad_spin_inplace",
    )
