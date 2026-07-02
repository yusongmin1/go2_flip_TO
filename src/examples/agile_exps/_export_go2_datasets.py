"""
Export Go2 trajectories after ``NLTrajOpt`` solves:

- ``datasets/go2/mocap_motions_go2/<run>_50hz.txt`` — **only** output: JSON ``Frames`` (49-D rows).

Disable with ``GO2_NO_DATASET=1`` (or legacy ``QUAD_SPIN_NO_DATASET=1``).

Root world height: ``GO2_EXPORT_BASE_Z_OFFSET`` (metres, default **0.022**) is
added to each knot ``q[2]`` before export (set ``0`` to disable).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_repo_root() -> Path:
    """Repository root (parent of ``src/``); ensures ``datasets`` package is importable."""
    root = Path(__file__).resolve().parents[3]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def go2_dataset_export_disabled() -> bool:
    return any(
        os.environ.get(k, "").lower() in ("1", "true", "yes")
        for k in ("GO2_NO_DATASET", "QUAD_SPIN_NO_DATASET")
    )


def export_go2_agile_trajectory(
    repo_root: Path,
    result: dict,
    model,
    run_name: str,
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
    mocap_filename: Optional[str] = None,
    fps_amp: float = 50.0,
    log_prefix: Optional[str] = None,
    isaac_frame_layout: str = "default",
) -> None:
    if go2_dataset_export_disabled():
        return

    from datasets.go2_amp_export import export_go2_isaac_motion_txt
    from datasets.go2_base_z_offset import apply_go2_base_z_offset_to_qs, go2_export_base_z_offset_m

    nodes = result["nodes"]
    K = len(nodes)
    qs = [nodes[k]["q"] for k in range(K)]
    vs = [nodes[k]["v"] for k in range(K)]
    dts = [nodes[k]["dt"] for k in range(K)]

    qs = apply_go2_base_z_offset_to_qs(qs, go2_export_base_z_offset_m())

    mocap_stem = mocap_filename if mocap_filename else f"{run_name}_50hz"
    if mocap_stem.endswith(".txt"):
        mocap_stem = mocap_stem[: -len(".txt")]

    mocap_txt = repo_root / "datasets" / "go2" / "mocap_motions_go2" / f"{mocap_stem}.txt"
    export_go2_isaac_motion_txt(
        model, qs, vs, dts, mocap_txt, fps=fps_amp, frame_layout=isaac_frame_layout
    )
    tag = log_prefix or run_name
    print(f"[{tag}] Mocap txt {fps_amp:g} Hz -> {mocap_txt}")
