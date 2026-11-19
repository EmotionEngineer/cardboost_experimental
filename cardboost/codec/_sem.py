"""Intermediate representation of a fitted card for exact SQL/C99 export.

The IR captures the *piecewise-constant additive structure* of a
:class:`r76_container31.DecodedCard`:

    score(x) = b0 + sum_j Vj[bj(x)] + sum_atoms atom(x)   (+ IsoLink on top)

Routing semantics (engine ``bin_idx``): right-closed intervals
``(-inf, c0], (c0, c1], ..., (c_last, +inf)`` plus a NaN shelf as the trailing
bin ``len(cuts)+1``.

Atom coverage (verified by ``tests/test_sql_c99.py`` equivalence tests):

* mains                          - analytic, exact
* IsoLink (knot_x, knot_v)       - clamped piecewise-linear, exact
* ProjRayAtom  (g4a..g4d)        - rank interpolation + u-binning, exact
* P1Atom       (8x8 grid)        - cell table + support box, exact
* M2Atom       (bilinear gamma)  - per-cell constants (cell-evaluated), exact
* MSumAtom     (main-sum ray)    - cell-evaluated u table, exact
* QMissAtom    (q_miss)          - analytic NaN shelves, exact
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------- IR nodes
@dataclass
class MainIR:
    j: int
    name: str
    cuts: List[float]
    levels: Any            # list len (m+2) of float, or (m+2) x K for multiclass


@dataclass
class RayIR:
    j: int
    k: int
    a: float
    b: float
    gj: List[float]; rgj: List[float]
    gk: List[float]; rgk: List[float]
    xmin_j: float; xmax_j: float
    xmin_k: float; xmax_k: float
    ucuts: List[float]
    levels: Any            # len(ucuts)+2 floats or (.., K)


@dataclass
class GridIR:
    """P1Atom: zero outside the [xmin, xmax] support box; NaN -> 0 as well."""
    j: int
    k: int
    cj: List[float]
    ck: List[float]
    xmin_j: float; xmax_j: float
    xmin_k: float; xmax_k: float
    table: Any             # (Aj, Bk) or (Aj, Bk, K) constants


@dataclass
class CellIR:
    """M2/MSum: per-(bj,bk)-cell constants (NaN cells included)."""
    j: int
    k: int
    cj: List[float]
    ck: List[float]
    table: Any             # (Aj, Bk) or (.., K); Aj = len(cj)+2


@dataclass
class QMissIR:
    shelves: List[Tuple[int, float]]      # (feature j, weight |V_j[b_NaN]|)
    qcuts: List[float]
    levels: Any


@dataclass
class IsoIR:
    kx: List[float]
    kv: List[float]


@dataclass
class CardIR:
    kind: str
    n_classes: int
    b0: Any
    mains: List[MainIR] = field(default_factory=list)
    rays: List[RayIR] = field(default_factory=list)
    grids: List[GridIR] = field(default_factory=list)
    cells: List[CellIR] = field(default_factory=list)
    qmiss: List[QMissIR] = field(default_factory=list)
    iso: Optional[IsoIR] = None
    names: Optional[list] = None


class UnsupportedAtomError(RuntimeError):
    pass


# --------------------------------------------------------------- helpers
def _f(x) -> float:
    return float(np.asarray(x, float).reshape(-1)[0])


def _levels(V, dim: int):
    V = np.asarray(V, float)
    return V.reshape(len(V), dim) if dim > 1 else V.reshape(len(V))


def _bin_sem(x: float, cuts: List[float]) -> int:
    """Scalar bin_idx semantics (right-open [c_i, c_i+1), NaN shelf)."""
    if x != x:
        return len(cuts) + 1
    b = 0
    while b < len(cuts) and x >= cuts[b]:
        b += 1
    return b


def _cell_table(atom, cj, ck, dim: int, reps_fn=None):
    """Exact per-cell constants of a bin-lookup atom, by evaluation."""
    Aj, Bk = len(cj) + 2, len(ck) + 2
    reps = []
    idx = []
    def rep_cell(cuts, b):
        if b == len(cuts) + 1:
            return np.nan
        if b == 0:
            return (cuts[0] - 1.0) if len(cuts) else -1.0
        if b == len(cuts):
            return cuts[-1] + 1.0 if len(cuts) else 1.0
        return 0.5 * (cuts[b - 1] + cuts[b])
    for a in range(Aj):
        for b in range(Bk):
            xj = rep_cell(cj, a)
            xk = rep_cell(ck, b)
            reps.append((xj, xk))
            idx.append((a, b))
    X = np.asarray(reps, float)
    out = np.asarray(atom.contrib(X), float)
    out = out.reshape(len(X), dim) if dim > 1 else out.reshape(len(X), 1)
    tab = np.zeros((Aj, Bk, dim))
    for r, (a, b) in enumerate(idx):
        tab[a, b] = out[r]
    return tab[:, :, 0] if dim == 1 else tab


# --------------------------------------------------------------- builder
def card_to_ir(card, names=None) -> CardIR:
    import r76_container31 as C

    kind = card.kind
    dim = int(len(card.b0)) if kind == "multiclass" else 1
    b0 = np.asarray(card.b0, float).reshape(-1)
    ir = CardIR(kind=kind, n_classes=dim, b0=b0.tolist() if dim > 1 else _f(b0),
                names=names)

    for j, cuts in enumerate(card.cuts):
        cuts = [float(c) for c in np.asarray(cuts, float).reshape(-1)]
        V = np.asarray(card.V[j], float)
        ir.mains.append(MainIR(j=j, name=(names[j] if names else f"x{j}"),
                               cuts=cuts, levels=_levels(V, dim)))

    for at in card.atoms:
        cls = type(at).__name__
        if cls == "ProjRayAtom":
            gj, rgj, gk, rgk = at.rank_interp
            ir.rays.append(RayIR(
                j=at.j, k=at.k, a=at.a, b=at.b,
                gj=[_f(v) for v in np.asarray(gj, float)],
                rgj=[_f(v) for v in np.asarray(rgj, float)],
                gk=[_f(v) for v in np.asarray(gk, float)],
                rgk=[_f(v) for v in np.asarray(rgk, float)],
                xmin_j=_f(at.xmin_j), xmax_j=_f(at.xmax_j),
                xmin_k=_f(at.xmin_k), xmax_k=_f(at.xmax_k),
                ucuts=[_f(v) for v in np.asarray(at.ucuts, float)] if at.ucuts is not None else [],
                levels=_levels(at.V if at.V is not None else at.Vc, dim)))
        elif cls == "P1Atom":
            cj = [_f(v) for v in np.asarray(at.cj, float)]
            ck = [_f(v) for v in np.asarray(at.ck, float)]
            tab = np.asarray(at.V if at.V is not None else at.Vc, float)
            if dim > 1:
                tab = tab.reshape(len(cj) + 2, len(ck) + 2, dim)
            else:
                tab = tab.reshape(len(cj) + 2, len(ck) + 2)
            ir.grids.append(GridIR(j=at.j, k=at.k, cj=cj, ck=ck,
                                   xmin_j=_f(at.xmin_j), xmax_j=_f(at.xmax_j),
                                   xmin_k=_f(at.xmin_k), xmax_k=_f(at.xmax_k),
                                   table=tab.tolist()))
        elif cls == "M2Atom":
            cj = [_f(v) for v in np.asarray(at.cj, float)]
            ck = [_f(v) for v in np.asarray(at.ck, float)]
            ir.cells.append(CellIR(at.j, at.k, cj, ck,
                                   _cell_table(at, cj, ck, dim).tolist()))
        elif cls == "MSumAtom":
            cj = [_f(v) for v in np.asarray(at.cj, float)]
            ck = [_f(v) for v in np.asarray(at.ck, float)]
            # u is bin-lookup constant per (bj, bk) cell -> cell-eval exact
            ir.cells.append(CellIR(at.j, at.k, cj, ck,
                                   _cell_table(at, cj, ck, dim).tolist()))
        elif cls == "QMissAtom":
            shelves = []
            w = np.asarray(getattr(at, "w", getattr(at, "weights", [])), float)
            js = getattr(at, "js", getattr(at, "feat_idx", range(len(w))))
            for jj, ww in zip(js, w):
                shelves.append((int(jj), _f(ww)))
            qcuts = [_f(v) for v in np.asarray(getattr(at, "ucuts", getattr(at, "qcuts", [])), float)]
            ir.qmiss.append(QMissIR(shelves, qcuts,
                                    _levels(at.V if at.V is not None else at.Vc, dim)))
        else:
            raise UnsupportedAtomError(
                f"atom class {cls!r} is not exportable in cardboost 1.0; "
                "export supports ProjRayAtom/P1Atom/M2Atom/MSumAtom/QMissAtom")

    iso = getattr(card, "isolink", None)
    if iso is not None:
        kx, kv = iso
        ir.iso = IsoIR([_f(v) for v in np.asarray(kx, float)],
                       [_f(v) for v in np.asarray(kv, float)])
    return ir
