"""Mesh-based collision avoidance for Go2 (visual DAE meshes convexified via qhull, hpp-fcl).

Replaces the old frame-point separation idea (``go2_leg_separation.py``): every link is
represented by the convex hull of its **visual mesh**, and the constraints are hard
inequalities on the hpp-fcl **signed distance**:

- self collision:   ``dist(hull_i, hull_j) - margin_ij >= 0``   (all non-adjacent pairs)
- ground clearance: ``dist(hull_i, ground plane) - margin_i  >= 0``  (every mesh, every node)

Because ``hull ⊇ mesh``, a non-negative hull distance guarantees the meshes you see in
MeshCat do not interpenetrate — the strict (conservative) standard.

Convexification inflates concave meshes (e.g. base <-> front-thigh hulls already overlap in
the standing pose). Margins are therefore auto-calibrated from a ``reference_q`` so a pair
is never required to be more separated than in that certified-safe pose:

    margin_ij = min(self_margin, dist_ref_ij - auto_slack)

Derivatives: hpp-fcl witness points ``p1`` (on hull i) and ``p2`` (on hull j) are material
points of the parent links, so with ``n = res.normal = (p2 - p1) / d``:
``dd/dq = -n @ J_p1 + n @ J_p2`` where ``J_p`` is the world-frame point Jacobian at ``p``.
For the ground plane ``d = p1_z - ground_z`` and ``dd/dq = e_z @ J_p1``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
import hppfcl

from constraint_models.abstract_constraint import *
from node import Node


class _MeshGeom:
    """One convexified link geometry: hppfcl hull rigidly attached to a parent joint."""

    __slots__ = ("name", "joint", "placement", "convex")

    def __init__(self, go: pin.GeometryObject):
        go.geometry.buildConvexHull(True, "Qt")
        convex = go.geometry.convex
        if convex is None:
            raise RuntimeError(f"qhull convexification failed for {go.name}")
        self.name = go.name
        self.joint = go.parentJoint
        self.placement = go.placement.copy()
        self.convex = convex


class Go2MeshCollisionConstraints(AbstractConstraint):
    """Hard inequality constraints on mesh (convex hull) signed distances.

    One instance is shared by all nodes of a problem; each node needs
    ``go2_mesh_collision_rows=mc.n_rows`` at construction so the row slice
    ``c_go2_mesh_col_id`` is allocated.
    """

    def __init__(
        self,
        robot,
        reference_q: Optional[np.ndarray] = None,
        *,
        self_margin: float = 2e-3,
        ground_margin: float = 1e-3,
        auto_slack: float = 5e-4,
        ground_z: float = 0.0,
        check_ground: bool = True,
        exclude_pairs: Tuple[Tuple[str, str], ...] = (),
        extra_pair_margins: Optional[Dict[Tuple[str, str], float]] = None,
        verbose: bool = True,
    ):
        """
        Args:
            robot: Go2 wrapper (needs ``.model`` and ``.visual_model``).
            reference_q: certified-safe Pinocchio configuration (usually the fixed start
                pose) used to auto-lower margins that convexification would otherwise
                make infeasible at the fixed endpoints.
            self_margin: default required hull separation (m).
            ground_margin: default required hull–ground separation (m).
            auto_slack: slack subtracted from reference distances when auto-lowering.
            ground_z: height of the (flat) ground plane.
            check_ground: also constrain every hull against the ground plane.
            exclude_pairs: extra ``(name_a, name_b)`` pairs (geom base names, order
                independent) to skip entirely.
            extra_pair_margins: optional ``{("a", "b"): margin}`` overrides applied
                before auto-calibration.
        """
        model: pin.Model = robot.model
        self._model = model
        self._data = model.createData()
        self._ground_z = float(ground_z)
        self._check_ground = check_ground
        self._shift = np.array([0.0, 0.0, -float(ground_z)])

        self._geoms: List[_MeshGeom] = [_MeshGeom(go) for go in robot.visual_model.geometryObjects]
        self._index = {g.name: k for k, g in enumerate(self._geoms)}
        # JOINT-type frame of each parent joint: computeFrameJacobian (the API every other
        # constraint here uses inside the cyipopt callback) gives the point Jacobian at the
        # joint origin; getJointJacobian/computeJointJacobians corrupt the interpreter heap
        # in this cyipopt/numpy-2 context and must not be used.
        self._joint_frame_ids: List[int] = []
        for g in self._geoms:
            fid = model.getFrameId(model.names[g.joint])
            if not model.existFrame(model.names[g.joint]) or model.frames[fid].type != pin.FrameType.JOINT:
                raise RuntimeError(f"no JOINT frame for joint {model.names[g.joint]}")
            self._joint_frame_ids.append(fid)
        self._ground = hppfcl.Halfspace(np.array([0.0, 0.0, 1.0]), 0.0)

        excl = {frozenset(p) for p in exclude_pairs}

        # All non-adjacent pairs (skip same joint and parent-child joints): (idx_a, idx_b, margin)
        self._pairs: List[Tuple[int, int, float]] = []
        for i in range(len(self._geoms)):
            for j in range(i + 1, len(self._geoms)):
                ga, gb = self._geoms[i], self._geoms[j]
                if ga.joint == gb.joint:
                    continue
                if model.parents[ga.joint] == gb.joint or model.parents[gb.joint] == ga.joint:
                    continue
                if frozenset((_base(ga.name), _base(gb.name))) in excl:
                    continue
                self._pairs.append((i, j, float(self_margin)))

        if extra_pair_margins:
            for (na, nb), m in extra_pair_margins.items():
                for pi, (ia, ib, _m) in enumerate(self._pairs):
                    if {_base(self._geoms[ia].name), _base(self._geoms[ib].name)} == {na, nb}:
                        self._pairs[pi] = (ia, ib, float(m))

        self._ground_margins: List[float] = [float(ground_margin)] * len(self._geoms) if check_ground else []

        if reference_q is not None:
            self.auto_calibrate_margins(reference_q, auto_slack=auto_slack, verbose=verbose)

    # ------------------------------------------------------------------ helpers

    @property
    def n_rows(self) -> int:
        """Constraint rows per node: one per pair (+ one per geom if ground is checked)."""
        return len(self._pairs) + len(self._ground_margins)

    @property
    def name(self) -> str:
        return "go2_mesh_collision"

    def _placements(self, q: np.ndarray, model=None, data=None):
        """FK on the caller's model/data (same instances the other constraints use)."""
        model = self._model if model is None else model
        data = self._data if data is None else data
        pin.forwardKinematics(model, data, q)
        return [data.oMi[g.joint] * g.placement for g in self._geoms]

    def _pair_dist(self, ka: int, kb: int, oMg):
        ga, gb = self._geoms[ka], self._geoms[kb]
        # Fresh request/result per call — a request shared across calls corrupted the heap
        # inside the cyipopt callbacks in this environment.
        req = hppfcl.DistanceRequest()
        req.enable_signed_distance = True
        res = hppfcl.DistanceResult()
        # np.array(..., copy=True): Transform3s may alias the numpy buffers; never hand it
        # short-lived temporaries (heap corruption / intermittent segfaults otherwise).
        Ra = np.array(oMg[ka].rotation, dtype=float, copy=True)
        ta = np.array(oMg[ka].translation, dtype=float, copy=True)
        Rb = np.array(oMg[kb].rotation, dtype=float, copy=True)
        tb = np.array(oMg[kb].translation, dtype=float, copy=True)
        d = hppfcl.distance(
            ga.convex,
            hppfcl.Transform3s(Ra, ta),
            gb.convex,
            hppfcl.Transform3s(Rb, tb),
            req,
            res,
        )
        return d, res

    def _ground_dist(self, k: int, oMg):
        """Hull vs ground plane; shapes are shifted down by ``ground_z`` so the plane is z=0."""
        g = self._geoms[k]
        M = oMg[k]
        req = hppfcl.DistanceRequest()
        req.enable_signed_distance = True
        res = hppfcl.DistanceResult()
        R = np.array(M.rotation, dtype=float, copy=True)
        t = np.array(M.translation, dtype=float, copy=True) + self._shift
        d = hppfcl.distance(
            g.convex,
            hppfcl.Transform3s(R, t),
            self._ground,
            hppfcl.Transform3s(),
            req,
            res,
        )
        return d, res

    def pair_distance(self, q: np.ndarray, name_a: str, name_b: str) -> float:
        """Signed hull distance of one pair at Pinocchio configuration ``q`` (audit helper)."""
        oMg = self._placements(q)
        d, _ = self._pair_dist(self._index[name_a], self._index[name_b], oMg)
        return d

    def ground_distance(self, q: np.ndarray, name: str) -> float:
        """Signed hull-vs-ground distance of one geom at ``q`` (audit helper)."""
        oMg = self._placements(q)
        d, _ = self._ground_dist(self._index[name], oMg)
        return d

    def audit(self, q: np.ndarray) -> Tuple[float, str]:
        """Worst ``d - margin`` over all rows at ``q``; negative means constraint violation."""
        oMg = self._placements(q)
        worst, worst_name = np.inf, ""
        for ia, ib, m in self._pairs:
            d, _ = self._pair_dist(ia, ib, oMg)
            if d - m < worst:
                worst = d - m
                worst_name = f"{_base(self._geoms[ia].name)}/{_base(self._geoms[ib].name)}"
        if self._ground_margins:
            for k in range(len(self._geoms)):
                d, _ = self._ground_dist(k, oMg)
                if d - self._ground_margins[k] < worst:
                    worst = d - self._ground_margins[k]
                    worst_name = f"{_base(self._geoms[k].name)}/ground"
        return worst, worst_name

    def auto_calibrate_margins(self, reference_q: np.ndarray, *, auto_slack: float = 5e-4, verbose: bool = True) -> None:
        """Lower per-pair / per-geom margins so the reference pose satisfies all rows.

        ``margin = min(desired, dist(reference) - auto_slack)`` — the problem then stays
        feasible at configurations fixed to the reference (start/end poses).
        """
        oMg = self._placements(np.asarray(reference_q, dtype=float))
        n_low = 0
        for pi, (ia, ib, desired) in enumerate(self._pairs):
            d_ref, _ = self._pair_dist(ia, ib, oMg)
            m = min(desired, d_ref - auto_slack)
            n_low += int(m < desired)
            self._pairs[pi] = (ia, ib, m)
        if self._ground_margins:
            for k in range(len(self._geoms)):
                d_ref, _ = self._ground_dist(k, oMg)
                self._ground_margins[k] = min(self._ground_margins[k], d_ref - auto_slack)
        if verbose:
            print(
                f"[go2_mesh_collision] {len(self._pairs)} self pairs (+{len(self._ground_margins)} ground rows), "
                f"{n_low} margins auto-lowered from reference pose"
            )

    # ------------------------------------------------------- AbstractConstraint

    def compute_constraints(self, node_curr: Node, node_next, state_vars, c, model, data):
        sl = node_curr.c_go2_mesh_col_id
        if sl is None:
            return
        q = q_tan2pin(state_vars[node_curr.q_id])
        oMg = self._placements(q, model, data)

        row = sl.start
        for ia, ib, margin in self._pairs:
            d, _ = self._pair_dist(ia, ib, oMg)
            c[row] = d - margin
            row += 1
        if self._ground_margins:
            for k in range(len(self._geoms)):
                d, _ = self._ground_dist(k, oMg)
                c[row] = d - self._ground_margins[k]
                row += 1

    def compute_jacobians(self, node_curr: Node, node_next, w, jac, model, data):
        sl = node_curr.c_go2_mesh_col_id
        if sl is None:
            return
        w_node = w[node_curr.q_id]
        q = q_tan2pin(w_node)
        oMg = self._placements(q, model, data)
        Jexp = pin.Jexp6(w_node[:6])

        # One world-aligned frame Jacobian per geom (at its joint origin), then rigid shifts
        # to each witness point.
        n_geoms = len(self._geoms)
        J_lin = [None] * n_geoms
        J_ang = [None] * n_geoms
        joint_origin = [None] * n_geoms
        for k in range(n_geoms):
            Jk = pin.computeFrameJacobian(
                model, data, q, self._joint_frame_ids[k], pin.LOCAL_WORLD_ALIGNED
            )
            Jk[:, :6] = Jk[:, :6] @ Jexp
            J_lin[k] = Jk[:3, :]
            J_ang[k] = Jk[3:, :]
            joint_origin[k] = np.array(data.oMi[self._geoms[k].joint].translation, dtype=float, copy=True)

        row = sl.start
        for ia, ib, _margin in self._pairs:
            d, res = self._pair_dist(ia, ib, oMg)
            p1 = np.array(res.getNearestPoint1(), dtype=float, copy=True)
            p2 = np.array(res.getNearestPoint2(), dtype=float, copy=True)
            diff = p1 - p2
            if abs(d) > 1e-9 and np.linalg.norm(diff) > 1e-9:
                n = -diff / d  # == res.normal, recomputed for numerical safety
            else:
                n = np.array(res.normal, dtype=float, copy=True)
                if np.linalg.norm(n) < 1e-9:
                    row += 1
                    continue
            Jp1 = J_lin[ia] - pin.skew(p1 - joint_origin[ia]) @ J_ang[ia]
            Jp2 = J_lin[ib] - pin.skew(p2 - joint_origin[ib]) @ J_ang[ib]
            jac[row, node_curr.q_id] = (n @ Jp2) - (n @ Jp1)
            row += 1

        ez = np.array([0.0, 0.0, 1.0])
        if self._ground_margins:
            for k in range(n_geoms):
                _d, res = self._ground_dist(k, oMg)
                p1 = np.array(res.getNearestPoint1(), dtype=float, copy=True) + [0.0, 0.0, self._ground_z]
                Jp1 = J_lin[k] - pin.skew(p1 - joint_origin[k]) @ J_ang[k]
                jac[row, node_curr.q_id] = ez @ Jp1
                row += 1

    def get_structure_ids(self, node_curr: Node, node_next, row_ids, col_ids):
        sl = node_curr.c_go2_mesh_col_id
        if sl is None:
            return
        for row in range(sl.start, sl.stop):
            for col in range(node_curr.q_id.start, node_curr.q_id.stop):
                row_ids.append(row)
                col_ids.append(col)

    def get_bounds(self, node: Node, lb, ub, clb, cub, model: pin.Model):
        sl = node.c_go2_mesh_col_id
        if sl is None:
            return
        n = sl.stop - sl.start
        clb[sl] = [0.0] * n  # d - margin >= 0
        cub[sl] = [None] * n


def _base(geom_name: str) -> str:
    """``FL_thigh_0`` -> ``FL_thigh``."""
    return geom_name.rsplit("_", 1)[0]
