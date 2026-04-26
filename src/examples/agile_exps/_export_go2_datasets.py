"""
Export Go2 trajectories after ``NLTrajOpt`` solves:

- ``datasets/go2/trajectories/<run_name>/`` — NPZ + CSV + ``meta.json`` (Pinocchio/URDF ``q``/``v``).
- ``datasets/go2/mocap_motions_go2/<mocap_file>`` — **50 Hz** Isaac-style JSON flat frames
  (``datasets/go2_amp_export.compose_go2_isaac_motion_row``): root pos/quat, base twist,
  dof_pos/vel, then ``(p_\text{foot}-p_\text{base})`` in **world** frame for the four foot frames only).

Disable with ``GO2_NO_DATASET=1`` (or legacy ``QUAD_SPIN_NO_DATASET=1``).

Root world height: ``GO2_EXPORT_BASE_Z_OFFSET`` (metres, default **0.022**) is
added to each knot ``q[2]`` before writing NPZ/AMP (set ``0`` to disable).
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
    from datasets.go2_pin_trajectory import save_go2_pin_trajectory_dataset

    nodes = result["nodes"]
    K = len(nodes)
    qs = [nodes[k]["q"] for k in range(K)]
    vs = [nodes[k]["v"] for k in range(K)]
    dts = [nodes[k]["dt"] for k in range(K)]

    dz = go2_export_base_z_offset_m()
    qs = apply_go2_base_z_offset_to_qs(qs, dz)

    meta: Dict[str, Any] = {"run_name": run_name, **(extra_meta or {})}
    meta["base_z_offset_applied_m"] = dz
    if isaac_frame_layout != "default":
        meta["isaac_frame_layout"] = isaac_frame_layout

    traj_dir = repo_root / "datasets" / "go2" / "trajectories" / run_name
    save_go2_pin_trajectory_dataset(traj_dir, qs, vs, dts, model, extra_meta=meta)

    tag = log_prefix or run_name
    print(f"[{tag}] Dataset (NPZ/CSV, URDF joint order) -> {traj_dir}")

    mocap_path = repo_root / "datasets" / "go2" / "mocap_motions_go2" / (
        mocap_filename if mocap_filename else f"{run_name}_50hz.txt"
    )
    export_go2_isaac_motion_txt(
        model, qs, vs, dts, mocap_path, fps=fps_amp, frame_layout=isaac_frame_layout
    )
    print(f"[{tag}] Isaac-style mocap JSON {fps_amp:g} Hz -> {mocap_path}")
