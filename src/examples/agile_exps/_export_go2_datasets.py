"""
Export Go2 trajectories after ``NLTrajOpt`` solves:

- ``datasets/go2/trajectories/<run_name>/`` — NPZ + CSV + ``meta.json`` (Pinocchio/URDF ``q``/``v``).
- ``datasets/go2/mocap_motions_go2/<mocap_file>`` — 25 Hz AMP JSON (``datasets/ai.py``-style 49-D frames).

Disable with ``GO2_NO_DATASET=1`` (or legacy ``QUAD_SPIN_NO_DATASET=1``).
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
    fps_amp: float = 25.0,
    log_prefix: Optional[str] = None,
) -> None:
    if go2_dataset_export_disabled():
        return

    from datasets.go2_amp_export import export_amp_mocap_txt
    from datasets.go2_pin_trajectory import save_go2_pin_trajectory_dataset

    nodes = result["nodes"]
    K = len(nodes)
    qs = [nodes[k]["q"] for k in range(K)]
    vs = [nodes[k]["v"] for k in range(K)]
    dts = [nodes[k]["dt"] for k in range(K)]

    meta: Dict[str, Any] = {"run_name": run_name, **(extra_meta or {})}

    traj_dir = repo_root / "datasets" / "go2" / "trajectories" / run_name
    save_go2_pin_trajectory_dataset(traj_dir, qs, vs, dts, model, extra_meta=meta)

    tag = log_prefix or run_name
    print(f"[{tag}] Dataset (NPZ/CSV, URDF joint order) -> {traj_dir}")

    mocap_path = repo_root / "datasets" / "go2" / "mocap_motions_go2" / (
        mocap_filename if mocap_filename else f"{run_name}_25hz.txt"
    )
    export_amp_mocap_txt(model, qs, vs, dts, mocap_path, fps=fps_amp)
    print(f"[{tag}] AMP mocap {fps_amp:g} Hz -> {mocap_path}")
