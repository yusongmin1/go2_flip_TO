"""
Visualize Go2 trajectories from ``trajectory.npz`` (solver export) or mocap JSON txt
(``save_as_txt_with_metadata`` / ``go2_amp_export``).

Supports **50 Hz** Isaac-style flat ``Frames`` (default agile export: **49-D** = 37 + four
``(p_\text{foot}-p_\text{base})_w`` triples)
and legacy **49-D** AMP49 rows (first 19 columns are full ``q``);
see ``datasets/go2_amp_export.py``. Run from repo root with
``PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"`` (root prefix required for ``import datasets``).

Uses MeshCat + URDF under ``src/nltrajopt/robots/go2`` — loads with ``buildModelsFromUrdf`` like
``Go2Wrapper`` (URDF already has a floating base; **do not** add a second ``JointModelFreeFlyer``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

try:
    import meshcat.geometry as g
    import meshcat.transformations as tf
except ImportError as e:
    raise SystemExit("meshcat is required: conda install -c conda-forge meshcat-python") from e

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_URDF = _REPO_ROOT / "src" / "nltrajopt" / "robots" / "go2" / "go2" / "urdf" / "go2.urdf"
_DEFAULT_PKG = _REPO_ROOT / "src" / "nltrajopt" / "robots" / "go2" / "go2"

FOOT_FRAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def _load_robot(urdf_path: Path, package_dir: Path):
    """Match ``robots/go2/Go2Wrapper.py``: one floating base from URDF only → ``nq=19``."""
    return pin.buildModelsFromUrdf(
        str(urdf_path),
        package_dirs=[str(package_dir)],
    )


def _qs_from_npz(path: Path) -> tuple:
    z = np.load(path, allow_pickle=True)
    Q = np.asarray(z["q"], dtype=np.float64)
    dts = np.asarray(z["dt"], dtype=np.float64).ravel()
    return Q, dts


def _load_amp_sidecar_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _qs_from_amp_json(path: Path) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = 1.0 / float(data["FrameDuration"])
    frames = np.asarray(data["Frames"], dtype=np.float64)
    n_col = frames.shape[1]
    dt = 1.0 / fps
    meta = _load_amp_sidecar_meta(path)
    fmt = meta.get("format")
    foot_names = (
        meta.get("foot_frame_names")
        or meta.get("foot_world_frame_names")
        or meta.get("key_body_frame_names")
    )

    def _from_go2_isaac_motion() -> tuple:
        pos = frames[:, :3]
        quat = frames[:, 3:7]
        # Legacy ``frame_layout: sideflip`` mocap: dof/key/twist blocks were permuted; default is below.
        if meta.get("frame_layout") == "sideflip":
            dof = frames[:, 7:19]
        else:
            dof = frames[:, 13:25]
        Q = np.hstack([pos, quat, dof])
        n = np.linalg.norm(Q[:, 3:7], axis=1, keepdims=True)
        Q[:, 3:7] /= np.maximum(n, 1e-12)
        dts = np.full(Q.shape[0], dt, dtype=np.float64)
        return Q, dts

    # 49 columns: either legacy AMP49 (q + feet_b + twist + qdot) or Go2 Isaac (37 + 4 feet × 3).
    if n_col == 49:
        _tail = meta.get("tail_kind")
        isaac49 = (
            fmt == "go2_isaac_motion"
            or _tail in ("foot_relative_base_world", "foot_translation_world")
            or (
                isinstance(foot_names, list)
                and len(foot_names) == 4
                and int(meta.get("frame_dim", 0)) == 49
            )
        )
        if isaac49:
            return _from_go2_isaac_motion()
        # Legacy 49-D: full Pinocchio q in first 19 columns.
        Q = frames[:, :19]
        dts = np.full(Q.shape[0], dt, dtype=np.float64)
        return Q, dts

    # Isaac-style (agile export): pos(3)+quat(4)+lin_b(3)+ang_b(3)+dof_pos(12)+dof_vel(12)+keys…
    if n_col >= 37 and (n_col - 37) % 3 == 0:
        return _from_go2_isaac_motion()
    raise ValueError(f"Unsupported Frames width {n_col} (expected 49 legacy or 37+3K Isaac export)")


def _foot_markers(viz: MeshcatVisualizer, model: pin.Model, data: pin.Data, q: np.ndarray) -> None:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    radius = 0.02
    sphere = g.Sphere(radius)
    mat = g.MeshLambertMaterial(color=0xFF0000)
    for idx, name in enumerate(FOOT_FRAMES):
        fid = model.getFrameId(name)
        p = data.oMf[fid].translation
        node = f"/foot_marker_{idx}"
        viz.viewer[node].set_object(sphere, mat)
        viz.viewer[node].set_transform(tf.translation_matrix(p))


def main():
    p = argparse.ArgumentParser(description="Play back Go2 trajectory in MeshCat")
    p.add_argument("--npz", type=str, default=None, help="trajectory.npz under datasets/go2/trajectories/<run>/")
    p.add_argument(
        "--amp",
        type=str,
        default=None,
        help="Mocap JSON txt (50 Hz Isaac-style default, or legacy 49-D *_25hz.txt)",
    )
    p.add_argument("--urdf", type=str, default=str(_DEFAULT_URDF))
    p.add_argument("--package-dir", type=str, default=str(_DEFAULT_PKG))
    p.add_argument("--loop", action="store_true", help="Repeat playback")
    args = p.parse_args()

    if bool(args.npz) == bool(args.amp):
        p.error("Provide exactly one of --npz or --amp")

    if args.npz:
        Q, dts = _qs_from_npz(Path(args.npz))
    else:
        Q, dts = _qs_from_amp_json(Path(args.amp))

    model, cmod, vmod = _load_robot(Path(args.urdf), Path(args.package_dir))
    if Q.shape[1] != model.nq:
        raise SystemExit(f"q columns {Q.shape[1]} != model.nq {model.nq}")

    viz = MeshcatVisualizer(model, cmod, vmod)
    viz.initViewer(open=True)
    viz.loadViewerModel(rootNodeName="go2")
    ground = g.Box([10.0, 10.0, 0.1])
    viz.viewer["/Ground"].set_object(ground)
    viz.viewer["/Ground"].set_transform(tf.translation_matrix([0, 0, -0.05]))

    data = model.createData()
    try:
        viz.viewer["go2/trunk_0"].set_property("color", [0.7, 0.7, 0.7, 1.0])
    except Exception:
        pass

    print(f"Playing {Q.shape[0]} configs, dt stats: min={dts.min():.4f} max={dts.max():.4f} s")

    while True:
        for i in range(Q.shape[0]):
            viz.display(Q[i])
            _foot_markers(viz, model, data, Q[i])
            time.sleep(float(dts[i]))
        if not args.loop:
            break


if __name__ == "__main__":
    # Allow running without PYTHONPATH if repo root is cwd
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    main()
