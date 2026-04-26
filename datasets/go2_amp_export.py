"""
Export Go2 trajectories as JSON ``Frames`` (``save_as_txt_with_metadata``).

**Agile export (50 Hz):** Isaac-style flat rows (``compose_go2_isaac_motion_row``), **one** column order:

``pos(3)``, ``quat(4)`` xyzw, **机体系**线速度 ``v[0:3]``、角速度 ``v[3:6]``, ``dof_pos(12)``,
``dof_vel(12)``, **key_body(12)** = four triples ``(p_\text{foot}-p_\text{base})`` in **world** frame,
order ``["FL_foot","FR_foot","RL_foot","RR_foot"]`` (``foot_frame_names`` in ``*.meta.json``).

**Legacy 49-D AMP row** (``compose_amp49_row``): kept for older tooling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio as pin

# Foot frames (Go2 URDF used by nltrajopt + datasets/go2)
FOOT_FRAME_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

# Four foot frames; tail columns are ``p_foot_w - p_base_w`` (world-frame vector, m).
GO2_ISAAC_MOTION_FOOT_FRAME_NAMES: Tuple[str, ...] = FOOT_FRAME_NAMES

GO2_ISAAC_FRAME_LAYOUT_DEFAULT = "default"


def save_as_txt_with_metadata(frames: np.ndarray, fps: float, output_path: Union[str, Path]) -> None:
    """
    Same structure as ``datasets/ai.py``: JSON with LoopMode, FrameDuration = 1/fps, Frames array.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration = 1.0 / float(fps)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write('  "LoopMode": "Wrap",\n')
        f.write(f"  \"FrameDuration\": {frame_duration},\n")
        f.write('  "EnableCycleOffsetPosition": true,\n')
        f.write('  "EnableCycleOffsetRotation": true,\n')
        f.write('  "MotionWeight": 0.5,\n')
        f.write('  "Frames": [\n')

        for i, frame in enumerate(frames):
            frame_str = ", ".join(f"{float(x):.6f}" for x in frame)
            if i < len(frames) - 1:
                f.write(f"    [{frame_str}],\n")
            else:
                f.write(f"    [{frame_str}]\n")

        f.write("  ]\n")
        f.write("}\n")

    print(f"[go2_amp_export] Saved {frames.shape[0]} frames @ {fps} Hz -> {output_path}")
    print(f"  FrameDuration={frame_duration:.6f} s, dim={frames.shape[1]}")


def _node_times(dts: Sequence[float]) -> np.ndarray:
    """Time stamp of each knot q[k]: t[k] = sum(dts[:k])."""
    dts = np.asarray(dts, dtype=np.float64).ravel()
    K = len(dts)
    t = np.zeros(K)
    for k in range(1, K):
        t[k] = t[k - 1] + dts[k - 1]
    return t


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).ravel()
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else q


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation; quaternions xyzw (Pinocchio ``q[3:7]``)."""
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return _quat_normalize(out)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return _quat_normalize(s0 * q0 + s1 * q1)


def resample_qv_to_fps(
    dts: Sequence[float],
    Q: np.ndarray,
    V: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniform-rate samples along time knots ``t[k]=sum(dts[:k])``.
    Position linear, quaternion slerp, joints & v linear.
    """
    Q = np.asarray(Q, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    t_ref = _node_times(dts)
    if Q.shape[0] != len(t_ref) or V.shape[0] != len(t_ref):
        raise ValueError("Q, V, dts length mismatch")
    if Q.shape[0] < 2:
        return Q.copy(), V.copy(), t_ref

    dt = 1.0 / float(fps)
    t0, t1 = float(t_ref[0]), float(t_ref[-1])
    t_new = np.arange(t0, t1 + 0.5 * dt, dt, dtype=np.float64)
    if len(t_new) == 0 or t_new[-1] < t1 - 1e-9:
        t_new = np.append(t_new, t1)

    n_out = len(t_new)
    nq, nv = Q.shape[1], V.shape[1]
    Qn = np.zeros((n_out, nq))
    Vn = np.zeros((n_out, nv))

    idx = np.searchsorted(t_ref, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(t_ref) - 2)

    for i, tn in enumerate(t_new):
        k = int(idx[i])
        tk, tk1 = t_ref[k], t_ref[k + 1]
        alpha = 0.0 if tk1 <= tk + 1e-15 else float((tn - tk) / (tk1 - tk))
        Qn[i, :3] = (1.0 - alpha) * Q[k, :3] + alpha * Q[k + 1, :3]
        Qn[i, 3:7] = _slerp(Q[k, 3:7], Q[k + 1, 3:7], alpha)
        Qn[i, 7:] = (1.0 - alpha) * Q[k, 7:] + alpha * Q[k + 1, 7:]
        Vn[i] = (1.0 - alpha) * V[k] + alpha * V[k + 1]

    Qn[:, 3:7] /= np.linalg.norm(Qn[:, 3:7], axis=1, keepdims=True)
    return Qn, Vn, t_new


def compose_amp49_row(model: pin.Model, data: pin.Data, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    One AMP-style row: full ``q`` (Pinocchio), foot positions in base frame, base twist in base
    (``lin_b=v[0:3]``, ``ang_b=v[3:6]``), joint rates ``v[6:]`` in **same order as** ``q[7:]``.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if q.shape[0] != model.nq or v.shape[0] != model.nv:
        raise ValueError(f"Bad q/v shape: nq={model.nq}, nv={model.nv}")

    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    base_id = model.getFrameId("base")
    oMb = data.oMf[base_id]
    foot_blocks: List[np.ndarray] = []
    for fname in FOOT_FRAME_NAMES:
        fid = model.getFrameId(fname)
        foot_blocks.append((oMb.inverse() * data.oMf[fid]).translation)
    foot_b = np.concatenate(foot_blocks)

    lin_b, ang_b = go2_freeflyer_lin_ang_body(v)
    qj = np.asarray(v[6:], dtype=np.float64).ravel()

    return np.concatenate([q, foot_b, lin_b, ang_b, qj])


def build_amp49_frames(
    model: pin.Model,
    Q: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Stack ``compose_amp49_row`` for each row of ``Q``, ``V``."""
    data = model.createData()
    rows: List[np.ndarray] = []
    for k in range(Q.shape[0]):
        rows.append(compose_amp49_row(model, data, Q[k], V[k]))
    return np.stack(rows, axis=0)


def go2_freeflyer_lin_ang_body(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Go2 URDF ``JointModelFreeFlyer``: ``v[0:3]`` base linear velocity in **base** frame (m/s),
    ``v[3:6]`` base angular velocity in **base** frame (rad/s). (Check with
    ``pin.computeAllTerms`` + ``pin.getFrameVelocity`` if the model changes.)
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    return v[0:3].copy(), v[3:6].copy()


def go2_isaac_motion_frame_dim(foot_frame_names: Optional[Sequence[str]] = None) -> int:
    kn = tuple(foot_frame_names) if foot_frame_names is not None else GO2_ISAAC_MOTION_FOOT_FRAME_NAMES
    return 37 + 3 * len(kn)


def compose_go2_isaac_motion_row(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    v: np.ndarray,
    foot_frame_names: Optional[Sequence[str]] = None,
    frame_layout: str = GO2_ISAAC_FRAME_LAYOUT_DEFAULT,
) -> np.ndarray:
    """
    One flat row (Isaac-style JSON ``Frames`` row, **49** floats for Go2):

    ``[root_pos_w(3), quat_xyzw(4), lin_b(3), ang_b(3), dof_pos(12), dof_vel(12),
       (p_foot - p_base)_w × 4]``

    Foot blocks: **only four foot frames** (``FOOT_FRAME_NAMES`` order); each triple is
    ``oMf[foot].translation - oMf[base].translation`` in **world** frame (**key_body**).

    Base twist: ``lin_b=v[0:3]``, ``ang_b=v[3:6]`` in **base/body frame** (Go2 free flyer).

    ``frame_layout`` is ignored except ``""`` / ``"default"``; other values raise (legacy
    ``sideflip`` rows must be reordered to this layout).
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if q.shape[0] != model.nq or v.shape[0] != model.nv:
        raise ValueError(f"Bad q/v shape: nq={model.nq}, nv={model.nv}")
    feet = tuple(foot_frame_names) if foot_frame_names is not None else GO2_ISAAC_MOTION_FOOT_FRAME_NAMES
    n_dof = model.nq - 7
    if n_dof != 12 or model.nv - 6 != 12:
        raise ValueError("This exporter assumes Go2-style 12-DoF legs (nq=19, nv=18).")

    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    root_p = q[:3].copy()
    root_quat = q[3:7].copy()
    lin_b, ang_b = go2_freeflyer_lin_ang_body(v)
    dof_pos = q[7:19].copy()
    dof_vel = v[6:18].copy()

    base_id = model.getFrameId("base")
    p_base = np.asarray(data.oMf[base_id].translation, dtype=np.float64).ravel()
    foot_rel_w: List[np.ndarray] = []
    for name in feet:
        fid = model.getFrameId(name)
        p_foot = np.asarray(data.oMf[fid].translation, dtype=np.float64).ravel()
        foot_rel_w.append(p_foot - p_base)

    foot_block = np.concatenate(foot_rel_w)
    if frame_layout not in ("", GO2_ISAAC_FRAME_LAYOUT_DEFAULT):
        raise ValueError(
            f"Unknown or unsupported frame_layout {frame_layout!r}; "
            "only default pos,quat,lin_b,ang_b,dof_pos,dof_vel,key_body is exported."
        )
    return np.concatenate([root_p, root_quat, lin_b, ang_b, dof_pos, dof_vel, foot_block])


def build_go2_isaac_motion_frames(
    model: pin.Model,
    Q: np.ndarray,
    V: np.ndarray,
    foot_frame_names: Optional[Sequence[str]] = None,
    frame_layout: str = GO2_ISAAC_FRAME_LAYOUT_DEFAULT,
) -> np.ndarray:
    data = model.createData()
    rows: List[np.ndarray] = []
    for k in range(Q.shape[0]):
        rows.append(
            compose_go2_isaac_motion_row(
                model, data, Q[k], V[k], foot_frame_names, frame_layout=frame_layout
            )
        )
    return np.stack(rows, axis=0)


def export_go2_isaac_motion_txt(
    model: pin.Model,
    qs: Sequence[np.ndarray],
    vs: Sequence[np.ndarray],
    dts: Sequence[float],
    output_path: Union[str, Path],
    fps: float = 50.0,
    foot_frame_names: Optional[Sequence[str]] = None,
    key_frame_names: Optional[Sequence[str]] = None,
    frame_layout: str = GO2_ISAAC_FRAME_LAYOUT_DEFAULT,
) -> Path:
    """
    Resample to ``fps`` Hz, build Isaac-style flat frames, write JSON txt (+ ``.meta.json``).

    ``key_frame_names`` is deprecated; use ``foot_frame_names`` (same meaning: foot URDF frame names).
    ``frame_layout``: only ``"default"`` / ``""`` (see ``compose_go2_isaac_motion_row``).
    """
    feet = foot_frame_names if foot_frame_names is not None else key_frame_names
    Q = np.stack([np.asarray(q, dtype=np.float64).ravel() for q in qs], axis=0)
    V = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vs], axis=0)
    Qr, Vr, _ = resample_qv_to_fps(dts, Q, V, fps)
    frames = build_go2_isaac_motion_frames(model, Qr, Vr, feet, frame_layout=frame_layout)
    dim = frames.shape[1]
    exp = go2_isaac_motion_frame_dim(feet)
    if dim != exp:
        raise RuntimeError(f"Expected {exp}-D frames, got {dim}")
    output_path = Path(output_path)
    names = tuple(feet) if feet is not None else GO2_ISAAC_MOTION_FOOT_FRAME_NAMES
    save_as_txt_with_metadata(frames, fps, output_path)
    print(
        f"[go2_isaac_motion] layout dim={dim}, frame_layout={frame_layout}, "
        f"foot_frames={len(names)} -> {output_path}"
    )
    meta_side = output_path.with_suffix(".meta.json")
    layout_cols = {
        "columns_0_2": "root position world (m)",
        "columns_3_6": "root quaternion xyzw world",
        "columns_7_9": "root linear velocity base/body frame (m/s)",
        "columns_10_12": "root angular velocity base/body frame (rad/s)",
        "columns_13_24": "dof_pos (12), URDF / Pinocchio q[7:] order",
        "columns_25_36": "dof_vel (12), Pinocchio v[6:] order",
        "columns_37_48": "key_body (p_foot - p_base) world (m), xyz × 4, order = foot_frame_names",
    }
    meta_side.write_text(
        json.dumps(
            {
                "format": "go2_isaac_motion",
                "fps": fps,
                "frame_dim": dim,
                "frame_layout": GO2_ISAAC_FRAME_LAYOUT_DEFAULT,
                "tail_kind": "foot_relative_base_world",
                "foot_frame_names": list(names),
                "layout": layout_cols,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path


def export_amp_mocap_txt(
    model: pin.Model,
    qs: Sequence[np.ndarray],
    vs: Sequence[np.ndarray],
    dts: Sequence[float],
    output_path: Union[str, Path],
    fps: float = 25.0,
) -> Path:
    """
    Resample to ``fps`` Hz, build 49-D frames (URDF/Pinocchio joint order), write JSON txt.
    """
    Q = np.stack([np.asarray(q, dtype=np.float64).ravel() for q in qs], axis=0)
    V = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vs], axis=0)
    Qr, Vr, _ = resample_qv_to_fps(dts, Q, V, fps)
    frames = build_amp49_frames(model, Qr, Vr)
    if frames.shape[1] != 49:
        raise RuntimeError(f"Expected 49-D frames, got {frames.shape[1]}")
    output_path = Path(output_path)
    save_as_txt_with_metadata(frames, fps, output_path)
    meta_side = output_path.with_suffix(".meta.json")
    meta_side.write_text(
        json.dumps(
            {
                "format": "amp49_legacy",
                "fps": fps,
                "frame_dim": 49,
                "layout": {
                    "q_pin_nq": model.nq,
                    "v_pin_nv": model.nv,
                    "columns_0_18": "q: base xyz, quat xyzw, joints in Pinocchio/URDF q[7:] order",
                    "columns_19_30": "foot FL,FR,RL,RR position in base frame (3 each)",
                    "columns_31_33": "base linear velocity in base frame (v[0:3])",
                    "columns_34_36": "base angular velocity in base frame (v[3:6])",
                    "columns_37_48": "joint velocities v[6:] same joint order as q[7:]",
                },
                "foot_frames": list(FOOT_FRAME_NAMES),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path
