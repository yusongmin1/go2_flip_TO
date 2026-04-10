"""
Visualize Go2 trajectories from ``trajectory.npz`` (quad_spin export) or AMP JSON txt
(``save_as_txt_with_metadata`` / ``go2_amp_export``).

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


def _qs_from_amp_json(path: Path) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = 1.0 / float(data["FrameDuration"])
    frames = np.asarray(data["Frames"], dtype=np.float64)
    n_col = frames.shape[1]
    if n_col < 19:
        raise ValueError(f"Frames need at least 19 columns (full q), got {n_col}")
    Q = frames[:, :19]
    dt = 1.0 / fps
    dts = np.full(Q.shape[0], dt, dtype=np.float64)
    return Q, dts


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
    p.add_argument("--npz", type=str, default=None, help="trajectory.npz from quad_spin export")
    p.add_argument("--amp", type=str, default=None, help="AMP JSON txt from go2_amp_export")
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
