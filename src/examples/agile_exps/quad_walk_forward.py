"""
Go2 **forward** walk (world **+x**) with:

- **Swing clearance** 12 cm; **horizon** 3 s; **|v|** ramp **0 → 1 m/s** (linear in time).
- Displacement ``Δx = +V_max·T/2``; warm start ``x(t) = x₀ + V_max·t²/(2T)``.

Same gait and timings as ``quad_walk_backward_ramp.py`` (``_go2_ramp_3s_common``).
"""
from _go2_translation_ramp_3s import run_translation_ramp_3s

if __name__ == "__main__":
    run_translation_ramp_3s(
        0,
        1.0,
        save_solution_basename="quad_walk_forward_ramp_3s",
        export_run_name="quad_walk_forward_ramp_3s",
        log_prefix="quad_walk_forward",
        motion_label="forward",
    )
