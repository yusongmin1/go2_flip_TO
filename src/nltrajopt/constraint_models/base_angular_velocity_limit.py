"""Hard cap on the floating-base angular velocity norm ``||vq[3:6]|| <= max_omega``.

``vq[3:6]`` are the base angular velocity components in the base frame (Go2 free-flyer);
their norm is frame-invariant, so a single quadratic inequality per node bounds the true
rotation rate. Squared form ``max_w^2 - ||w||^2 >= 0`` keeps the gradient (``-2w``) well
defined at ``w = 0``.

The per-component variable-bound alternative would silently allow ``sqrt(3) * max_omega``.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from constraint_models.abstract_constraint import *
from node import Node


class BaseAngularVelocityLimitConstraints(AbstractConstraint):
    """One inequality row per node: ``max_omega^2 - ||vq[3:6]||^2 >= 0``."""

    def __init__(self, max_omega: float):
        if max_omega <= 0.0:
            raise ValueError(f"max_omega must be positive, got {max_omega}")
        self.max_w = float(max_omega)

    @property
    def name(self) -> str:
        return "base_angular_velocity_limit"

    def compute_constraints(self, node_curr: Node, node_next, state_vars, c, model, data):
        sl = node_curr.c_base_w_id
        if sl is None:
            return
        w = state_vars[node_curr.vq_id][3:6]
        c[sl] = self.max_w**2 - float(w @ w)

    def compute_jacobians(self, node_curr: Node, node_next, w_, jac, model, data):
        sl = node_curr.c_base_w_id
        if sl is None:
            return
        w = w_[node_curr.vq_id][3:6]
        row = np.zeros(node_curr.nv)
        row[3:6] = -2.0 * w
        jac[sl, node_curr.vq_id] = row

    def get_structure_ids(self, node_curr: Node, node_next, row_ids, col_ids):
        if node_curr.c_base_w_id is None:
            return
        for col in range(node_curr.vq_id.start + 3, node_curr.vq_id.start + 6):
            row_ids.append(node_curr.c_base_w_id.start)
            col_ids.append(col)

    def get_bounds(self, node: Node, lb, ub, clb, cub, model: pin.Model):
        if node.c_base_w_id is None:
            return
        clb[node.c_base_w_id] = [0.0]
        cub[node.c_base_w_id] = [None]
