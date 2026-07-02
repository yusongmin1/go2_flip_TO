"""
Play back Go2 mocap JSON txt (``go2_amp_export.export_go2_isaac_motion_txt``) in MuJoCo.

Each **43-D** row: ``root_pos(3)``, ``root_rot`` xyzw ``(4)``, ``dof_pos(12)``,
four feet in base frame ``(12)``, ``dof_vel(12)``. Rendering uses ``root_pos`` /
``root_rot`` / ``dof_pos`` only.

Run from repo root::

    export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"
    python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_backflip_50hz.txt
    # default: loop until you close the window; --no-loop to play once

If the interactive viewer segfaults (GLX / GPU driver), try::

    export MUJOCO_GL=egl
    python datasets/viz_go2_amp_trajectory.py --amp ... --video out.mp4

默认加载 **`datasets/go2/go2/scene.xml`**（含 `go2.xml` 的 OBJ 网格、地面、天空盒），不再用 Pinocchio URDF（URDF 在 MuJoCo 里往往只剩碰撞盒、且无天空/地面）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCENE = _REPO_ROOT / "datasets" / "go2" / "go2" / "scene.xml"
_DEFAULT_MODEL_DIR = _DEFAULT_SCENE.parent
# Legacy URDF (collision boxes only if mesh fails to load):
_DEFAULT_URDF = _REPO_ROOT / "src" / "nltrajopt" / "robots" / "go2" / "go2" / "urdf" / "go2.urdf"
_DEFAULT_PKG = _REPO_ROOT / "src" / "nltrajopt" / "robots" / "go2" / "go2"

# go2.xml: visual geoms group 2, collision group 3; scene floor/sky in group 0.
ROBOT_VISUAL_GEOM_GROUP = 2
ROBOT_COLLISION_GEOM_GROUP = 3
ROBOT_BASE_BODY = "base_link"

GO2_DOF_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


def _import_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as e:
        raise SystemExit(
            "mujoco is required: pip install mujoco  (or conda install -c conda-forge mujoco)"
        ) from e
    return mujoco, mujoco.viewer


def _load_mujoco_model(mujoco: Any, model_path: Path, model_dir: Path | None = None):
    """Load MJCF/URDF; relative mesh paths resolve from ``model_dir`` (or the XML parent)."""
    model_path = model_path.resolve()
    base = (model_dir if model_dir is not None else model_path.parent).resolve()
    cwd = os.getcwd()
    try:
        os.chdir(base)
        return mujoco.MjModel.from_xml_path(str(model_path))
    finally:
        os.chdir(cwd)


def _make_scene_option(mujoco: Any, show_collision: bool = False):
    """Show MJCF visual meshes + scene; hide robot collision unless ``show_collision``."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # floor, sky-linked geoms, scene props
    opt.geomgroup[ROBOT_VISUAL_GEOM_GROUP] = 1
    if show_collision:
        opt.geomgroup[ROBOT_COLLISION_GEOM_GROUP] = 1
    return opt


def _load_frames_from_txt(path: Path) -> tuple[np.ndarray, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = 1.0 / float(data["FrameDuration"])
    frames = np.asarray(data["Frames"], dtype=np.float64)
    if frames.ndim != 2:
        raise ValueError(f"Frames must be 2-D, got shape {frames.shape}")
    n_col = frames.shape[1]
    if n_col == 43:
        return frames, fps
    if n_col == 49:
        out = np.hstack(
            [
                frames[:, :7],
                frames[:, 13:25],
                frames[:, 37:49],
                frames[:, 25:37],
            ]
        )
        return out, fps
    if n_col >= 19 and n_col < 43:
        padded = np.zeros((frames.shape[0], 43), dtype=np.float64)
        padded[:, :19] = frames[:, :19]
        return padded, fps
    raise ValueError(f"Unsupported frame width {n_col} (expected 43)")


def _frame_to_qpos(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float64).ravel()
    if frame.shape[0] < 19:
        raise ValueError(f"Need at least 19 values for root+dof, got {frame.shape[0]}")
    qx, qy, qz, qw = frame[3:7]
    qpos = np.zeros(19, dtype=np.float64)
    qpos[0:3] = frame[0:3]
    qpos[3:7] = np.array([qw, qx, qy, qz], dtype=np.float64)
    qpos[7:19] = frame[7:19]
    n = np.linalg.norm(qpos[3:7])
    if n > 1e-12:
        qpos[3:7] /= n
    return qpos


def _check_joint_order(mujoco: Any, model) -> None:
    for i, name in enumerate(GO2_DOF_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"Joint {name!r} not found in MuJoCo model")
        adr = int(model.jnt_qposadr[jid])
        if adr != 7 + i:
            raise SystemExit(
                f"Joint {name!r} qposadr={adr}, expected {7 + i}; "
                "dof_pos column order may not match this URDF."
            )


def _apply_frame(mujoco: Any, model, data, frame: np.ndarray) -> None:
    data.qpos[:] = _frame_to_qpos(frame)
    mujoco.mj_forward(model, data)


def _setup_camera(mujoco: Any, model, data, viewer_cam) -> None:
    mujoco.mj_forward(model, data)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)
    if base_bid < 0:
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_bid >= 0:
        viewer_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_cam.trackbodyid = base_bid
        viewer_cam.lookat[:] = data.xpos[base_bid]
    else:
        viewer_cam.lookat[:] = data.subtree_com[0]
    viewer_cam.distance = 2.5
    viewer_cam.elevation = -15.0
    viewer_cam.azimuth = 90.0


def _export_video(
    mujoco: Any,
    model,
    frames: np.ndarray,
    fps: float,
    output_path: Path,
    width: int,
    height: int,
    show_collision: bool,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise SystemExit("Video export needs imageio: pip install imageio imageio-ffmpeg") from e

    data = mujoco.MjData(model)
    _apply_frame(mujoco, model, data, frames[0])
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    scene_option = _make_scene_option(mujoco, show_collision=show_collision)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        pixels = []
        for i in range(frames.shape[0]):
            _apply_frame(mujoco, model, data, frames[i])
            renderer.update_scene(data, scene_option=scene_option)
            pixels.append(renderer.render())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), pixels, fps=fps)
    finally:
        renderer.close()
    print(f"Saved video ({frames.shape[0]} frames @ {fps:g} Hz) -> {output_path}")


def _play_interactive(
    mujoco: Any,
    mujoco_viewer: Any,
    model,
    frames: np.ndarray,
    dt: float,
    loop: bool,
    show_collision: bool,
) -> None:
    data = mujoco.MjData(model)
    _apply_frame(mujoco, model, data, frames[0])

    with mujoco_viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.opt.geomgroup[:] = _make_scene_option(mujoco, show_collision).geomgroup
        _setup_camera(mujoco, model, data, viewer.cam)
        idx = 0
        n = frames.shape[0]
        while viewer.is_running():
            _apply_frame(mujoco, model, data, frames[idx])
            viewer.sync()
            time.sleep(dt)
            idx += 1
            if idx >= n:
                if loop:
                    idx = 0
                else:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    break


def _configure_gl(gl_backend: str | None, video_path: Path | None) -> None:
    if gl_backend:
        os.environ["MUJOCO_GL"] = gl_backend
    elif video_path is not None and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    elif "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"


def main() -> None:
    p = argparse.ArgumentParser(description="Play back Go2 mocap txt in MuJoCo")
    p.add_argument(
        "--amp",
        type=str,
        required=True,
        help="Mocap JSON txt, e.g. datasets/go2/mocap_motions_go2/quad_backflip_50hz.txt",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(_DEFAULT_SCENE),
        help="MuJoCo scene XML (default: datasets/go2/go2/scene.xml with mesh + ground + sky)",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=str(_DEFAULT_MODEL_DIR),
        help="Directory for resolving relative mesh paths in the MJCF",
    )
    p.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="(deprecated) same as --model; use --model instead",
    )
    p.add_argument("--package-dir", type=str, default=str(_DEFAULT_PKG))
    p.add_argument(
        "--show-collision",
        action="store_true",
        help="Also draw robot collision geoms (group 3); default is visual mesh only",
    )
    p.add_argument(
        "--no-loop",
        action="store_true",
        help="Play once then hold last frame (default: loop until window is closed)",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Export MP4 instead of opening a window (uses offscreen render; sets MUJOCO_GL=egl)",
    )
    p.add_argument(
        "--gl",
        type=str,
        default=None,
        choices=("glfw", "egl", "osmesa"),
        help="MuJoCo GL backend (default glfw for viewer; egl for --video)",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    video_path = Path(args.video) if args.video else None
    _configure_gl(args.gl, video_path)

    mujoco, mujoco_viewer = _import_mujoco()

    model_path = Path(args.urdf if args.urdf else args.model)
    model_dir = Path(args.package_dir if args.urdf else args.model_dir)

    txt_path = Path(args.amp)
    frames, fps = _load_frames_from_txt(txt_path)
    dt = 1.0 / fps

    model = _load_mujoco_model(mujoco, model_path, model_dir)
    if model.nq != 19:
        raise SystemExit(f"Expected Go2 nq=19, got {model.nq}")
    _check_joint_order(mujoco, model)

    print(f"Playing {frames.shape[0]} frames @ {fps:g} Hz from {txt_path}")
    loop = not args.no_loop
    if loop:
        print("Loop: on (close window to exit; use --no-loop to play once)")

    if video_path is not None:
        _export_video(
            mujoco, model, frames, fps, video_path, args.width, args.height, args.show_collision
        )
        return

    try:
        _play_interactive(mujoco, mujoco_viewer, model, frames, dt, loop, args.show_collision)
    except Exception as exc:
        raise SystemExit(
            f"Interactive viewer failed ({exc}). Try offscreen export:\n"
            f"  MUJOCO_GL=egl python {sys.argv[0]} --amp {txt_path} --video /tmp/go2.mp4\n"
            "On Linux/Wayland you can also try: export GLFW_USE_WAYLAND=0"
        ) from exc


if __name__ == "__main__":
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    main()
