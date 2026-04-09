import numpy as np
import pinocchio as pin

from constraint_models.abstract_constraint import *
from node import Node

# Go2 (Go2Wrapper / URDF): nv=18 → base 0:6, FL 6:9, FR 9:12, RL 12:15, RR 15:18
_GO2_NV = 18
_Q_PAIRS = ((6, 9), (7, 10), (8, 11), (12, 15), (13, 16), (14, 17))


class Go2LeftRightLegSymmetryConstraints(AbstractConstraint):
    """Equality: left and right legs have the same joint coordinates (sagittal symmetry).

    Six residuals per node: FL vs FR and RL vs RR for {hip, thigh, calf}:
    ``q[left] - q[right] = 0``. Joint speed limits are handled separately (variable bounds on
    ``vq[6:]``).

    Indices follow Pinocchio ``nv`` layout for the bundled Go2 model (``nv == 18``).
    """

    def __init__(self, model: pin.Model):
        if model.nv != _GO2_NV:
            raise ValueError(
                f"Go2LeftRightLegSymmetryConstraints expects nv={_GO2_NV}, got {model.nv}"
            )

    @property
    def name(self) -> str:
        return "go2_lr_leg_symmetry"

    def compute_constraints(self, node_curr: Node, node_next, state_vars, c, model, data):
        sid = node_curr.c_go2_lr_sym_id
        if sid is None:
            return
        q = state_vars[node_curr.q_id]
        row = sid.start
        for i, j in _Q_PAIRS:
            c[row] = q[i] - q[j]
            row += 1

    def compute_jacobians(self, node_curr: Node, node_next, w, jac, model, data):
        sid = node_curr.c_go2_lr_sym_id
        if sid is None:
            return
        row = sid.start
        qb = node_curr.q_id.start
        for i, j in _Q_PAIRS:
            jac[row, qb + i] = 1.0
            jac[row, qb + j] = -1.0
            row += 1

    def get_structure_ids(self, node_curr: Node, node_next, row_ids, col_ids):
        if node_curr.c_go2_lr_sym_id is None:
            return
        sid = node_curr.c_go2_lr_sym_id
        qb = node_curr.q_id.start
        row = sid.start
        for i, j in _Q_PAIRS:
            row_ids.extend([row, row])
            col_ids.extend([qb + i, qb + j])
            row += 1

    def get_bounds(self, node: Node, lb, ub, clb, cub, model: pin.Model):
        if node.c_go2_lr_sym_id is None:
            return
        sl = node.c_go2_lr_sym_id
        n = sl.stop - sl.start
        clb[sl] = np.zeros(n)
        cub[sl] = np.zeros(n)
