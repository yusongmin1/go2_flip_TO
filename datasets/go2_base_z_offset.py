"""World-frame root height offset for Go2 (Pinocchio free-flyer ``q[2]``)."""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np


def go2_export_base_z_offset_m() -> float:
    """
    Extra metres added to ``q[2]`` when exporting NPZ/CSV and Isaac-style AMP txt.

    Environment variable ``GO2_EXPORT_BASE_Z_OFFSET`` (float string). If unset,
    defaults to **0.022**. Set to ``0`` to disable.
    """
    raw = os.environ.get("GO2_EXPORT_BASE_Z_OFFSET")
    if raw is None:
        return 0.022
    s = str(raw).strip()
    if s == "":
        return 0.022
    return float(s)


def apply_go2_base_z_offset_to_qs(qs: Sequence[np.ndarray], dz: float) -> List[np.ndarray]:
    if abs(float(dz)) < 1e-15:
        return list(qs)
    out: List[np.ndarray] = []
    for q in qs:
        qc = np.asarray(q, dtype=np.float64, copy=True)
        qc[2] += dz
        out.append(qc)
    return out
