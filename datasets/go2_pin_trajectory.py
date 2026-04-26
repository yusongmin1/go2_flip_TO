"""
Export Go2 trajectories in **Pinocchio configuration order** (same as the URDF used to build
``pin.Model``): ``q`` matches ``model.nq``, actuated coordinates are ``q[j0:j0+n_j]`` with
``joint_names[i]`` the URDF joint name for column ``i``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pinocchio as pin


def revolute_joint_names_in_pin_q_order(model: pin.Model) -> List[str]:
    """1-DoF joints sorted by ``idx_q`` (Pinocchio / URDF configuration order)."""
    pairs: List[tuple] = []
    for jid in range(1, model.njoints):
        jm = model.joints[jid]
        if jm.nq == 1:
            pairs.append((int(jm.idx_q), model.names[jid]))
    pairs.sort(key=lambda x: x[0])
    return [name for _, name in pairs]


def actuated_q_index_range(model: pin.Model) -> tuple:
    """(start, n_joints) for the contiguous actuated block in ``q`` (after floating base)."""
    names = revolute_joint_names_in_pin_q_order(model)
    if not names:
        raise ValueError("No 1-DoF joints found on model")
    start = None
    for jid in range(1, model.njoints):
        jm = model.joints[jid]
        if jm.nq == 1:
            s = int(jm.idx_q)
            start = s if start is None else min(start, s)
    n_j = len(names)
    idxs = []
    for jid in range(1, model.njoints):
        jm = model.joints[jid]
        if jm.nq == 1:
            idxs.append(int(jm.idx_q))
    if sorted(idxs) != list(range(start, start + n_j)):
        raise ValueError(f"Actuated q indices not contiguous: {sorted(idxs)}")
    return start, n_j


def save_go2_pin_trajectory_dataset(
    output_dir: Union[str, Path],
    qs: Sequence[np.ndarray],
    vs: Sequence[np.ndarray],
    dts: Sequence[float],
    model: pin.Model,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write under ``output_dir``:

    - ``trajectory.npz``: ``q`` (T,nq), ``v`` (T,nv), ``dt`` (T,), ``joint_names`` (n_j,)
    - ``meta.json``: layout + ``actuated_joint_names`` (URDF order = Pinocchio q order)
    - ``joints_only.csv``: only actuated ``q``, one row per time, header = URDF joint names
    - ``trajectory_full.csv``: ``t`` + full ``q`` (base + joints), header documents columns
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Q = np.stack([np.asarray(q, dtype=np.float64).ravel() for q in qs], axis=0)
    V = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vs], axis=0)
    dt_arr = np.asarray(dts, dtype=np.float64)

    if Q.shape[1] != model.nq:
        raise ValueError(f"q dim {Q.shape[1]} != model.nq {model.nq}")
    if V.shape[1] != model.nv:
        raise ValueError(f"v dim {V.shape[1]} != model.nv {model.nv}")
    if dt_arr.shape[0] != Q.shape[0]:
        raise ValueError("len(dts) must match len(qs)")

    joint_names = revolute_joint_names_in_pin_q_order(model)
    j0, n_j = actuated_q_index_range(model)
    if n_j != len(joint_names):
        raise RuntimeError("joint name count mismatch")
    if j0 + n_j != model.nq:
        raise ValueError(
            f"Expected actuated block q[{j0}:{j0+n_j}] to fill to nq={model.nq}, "
            f"got j0={j0}, n_j={n_j}"
        )

    jname_arr = np.array(joint_names, dtype=object)
    np.savez_compressed(
        output_dir / "trajectory.npz",
        q=Q,
        v=V,
        dt=dt_arr,
        joint_names=jname_arr,
        actuated_q_start=j0,
    )

    t = np.concatenate([[0.0], np.cumsum(dt_arr[:-1])])
    meta_full: Dict[str, Any] = {
        "robot_model_name": str(model.name),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "actuated_joint_names": joint_names,
        "actuated_q_indices": list(range(j0, j0 + n_j)),
        "q_pinocchio_layout": {
            "note": "Same order as Pinocchio model built from URDF (floating base + joints).",
            "translation_world_xyz": [0, 1, 2],
            "quaternion_joint_frame_xyzw": [3, 4, 5, 6],
            "actuated_urdf_joint_order": joint_names,
        },
        "v_pinocchio_layout": {
            "note": "Generalized velocity v aligned with model.nv (6-d base + joint rates).",
            "floating_base_v0_5_go2": "v[0:3] base linear vel in base frame (m/s); v[3:6] base angular vel in base frame (rad/s); v[6:] joint rates.",
        },
    }
    if extra_meta:
        meta_full.update(extra_meta)

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_full, f, indent=2, ensure_ascii=False)

    J = Q[:, j0 : j0 + n_j]
    joints_csv = output_dir / "joints_only.csv"
    with open(joints_csv, "w", encoding="utf-8") as f:
        f.write(",".join(joint_names) + "\n")
        for row in J:
            f.write(",".join(f"{x:.18e}" for x in row) + "\n")

    base_hdr = ["base_tx", "base_ty", "base_tz", "quat_x", "quat_y", "quat_z", "quat_w"]
    full_hdr = ["t"] + base_hdr + joint_names
    full_csv = output_dir / "trajectory_full.csv"
    with open(full_csv, "w", encoding="utf-8") as f:
        f.write(",".join(full_hdr) + "\n")
        for ti, row in zip(t, Q):
            f.write(f"{ti:.18e}," + ",".join(f"{x:.18e}" for x in row) + "\n")

    return output_dir
