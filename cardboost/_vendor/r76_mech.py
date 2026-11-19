#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r76_mech.py — Round 76 Stage-3/3.1 mechanism library (built on the frozen r71 engine).

Components (the audit's Stage-3 arbitration vocabulary):
  p1  Product-Potts Rectangles   : un-gated BxB cell surface per feature pair, 2-D TV-Potts fusion
                                   (alternating row/column exact 1-D Potts DPs, 2 cycles)
  g4  Projection Grammar         : u1 = r_j + r_k, u2 = r_j - r_k over empirical midranks, 1-D step
                                   functions, INTERIOR-GATED (rows touching a parent boundary bin get 0;
                                   the trailing boundary level is zeroed and re-centred) — the audit's
                                   Stage-3.1 fix for fold-4 boundary-ray aliasing
  m2  Bilinear scalar            : gamma_jk * z_j[bin_j] * z_k[bin_k], z = centred main levels
  sw14 Switch Card               : depth-1 global split; two full sub-models per side; charged
                                   ln(M_splits) + dq*kappa_child on the prequential ledger (the audit's fix)
Admission: 5-way Toof-style prequential ledger (seed 109 splits), per-atom EBIC entry price, >=4/5
sub-folds positive (the pre-registered stability bar).

All fitting funnels through r71 solvers (solve_bin_* / potts_dp / _segment_levels) so the base
mechanics are bit-compatible with the certified engine.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home/user/google_drive")
import r71_universal as r71
from r71_universal import (
    bin_idx, cuts_of, onehot, potts_dp, sig, softmax, log_softmax,
    solve_bin_logistic, solve_bin_regression, solve_bin_multiclass, _segment_levels,
)


# ----------------------------------------------------------------------------
# weighted bin solves (Bayesian-bootstrap bags need row weights)
# ----------------------------------------------------------------------------
# R107 (frozen 2026-11-16): exact binomial deviance in 1D potts_dp for binary
# tables (mains + g4 rays + msum).  One shared switch: mains (fit_joint) and
# atom refits read it together; off => byte-identical quadratic path.
EXACT_DEV = {"on": False}


def wsolve_bin_regression(off, y, nb, idx, w):
    cnt = np.bincount(idx, weights=w, minlength=nb)
    s = np.bincount(idx, weights=w * (y - off), minlength=nb)
    z = np.where(cnt > 0, s / np.maximum(cnt, 1e-12), 0.0)
    H = np.where(cnt > 0, cnt, 0.0)
    return z, H


def wsolve_bin_logistic(off, y, nb, idx, w, alpha=0.5, iters=60):
    gy = np.bincount(idx, weights=w * y, minlength=nb)
    cw = np.bincount(idx, weights=w, minlength=nb)
    lo = np.full(nb, -80.0)
    hi = np.full(nb, 80.0)
    v = np.zeros(nb)
    for _ in range(iters):
        p = sig(off + v[idx])
        gp = np.bincount(idx, weights=w * p, minlength=nb)
        g = gy - gp - alpha * v
        pos = g > 0
        lo = np.where(pos, v, lo)
        hi = np.where(pos, hi, v)
        v = 0.5 * (lo + hi)
    p = sig(off + v[idx])
    H = np.bincount(idx, weights=w * p * (1.0 - p), minlength=nb) + alpha
    H[cw == 0] = 0.0
    z = np.where(cw > 0, v, 0.0)
    return z, H


def wsolve_bin_multiclass(off, Y, nb, idx, K, w, alpha=0.5, iters=40):
    C = np.zeros((nb, K))
    for _ in range(iters):
        P = softmax(off + C[idx])
        g = np.zeros((nb, K))
        for k in range(K):
            g[:, k] = np.bincount(idx, weights=w * (Y[:, k] - P[:, k]), minlength=nb)
        g -= alpha * C
        Hd = np.bincount(idx, weights=w * (P * (1.0 - P)).sum(axis=1), minlength=nb) + alpha
        step = g / np.maximum(Hd, 1e-9)[:, None]
        C += 0.5 * step
        if np.max(np.abs(step)) < 1e-8:
            break
    C -= C.mean(axis=1, keepdims=True)
    P = softmax(off + C[idx])
    H = np.bincount(idx, weights=w * (P * (1.0 - P)).sum(axis=1), minlength=nb) + alpha
    cw = np.bincount(idx, weights=w, minlength=nb)
    H[cw == 0] = 0.0
    return C, H


def wsegment_levels(lab, z, H, kind, C2=None):
    return _segment_levels(lab, z, H, kind, C2=C2, shrink=0.0)


# ----------------------------------------------------------------------------
# product-Potts fusion on a grid (2-D TV-Potts by alternating exact 1-D DPs)
# ----------------------------------------------------------------------------
def product_potts(C, H, kappa, cycles=2):
    """C: (A, B[, K]) cell levels, H: (A, B) masses. Returns fused cell levels.

    Alternates exact 1-D potts_dp along rows (over columns) and columns (over rows),
    refitting levels as H-weighted segment means after each sweep. Deterministic.
    """
    r71.R84_COST["pair_ctx"] += 1        # R84-H3: 2-D grids may opt out of the currency
    C = np.asarray(C, float)
    H = np.asarray(H, float)
    one_d = C.ndim == 2
    if one_d:
        C = C[:, :, None]
    A, B, K = C.shape
    V = C.copy()
    for _ in range(cycles):
        for a in range(A):                      # fuse along the row (over b)
            if H[a].sum() <= 0:
                continue
            lab = potts_dp(np.where(H[a][:, None] > 0, V[a], 0.0), np.where(H[a] > 0, H[a], 1e-9), kappa)
            if K == 1:
                z = V[a][:, 0]
                V[a, :, 0] = wsegment_levels(lab, z, H[a], "regression")
            else:
                V[a] = wsegment_levels(lab, None, H[a], "multiclass", C2=V[a])
        for b in range(B):                      # fuse along the column (over a)
            if H[:, b].sum() <= 0:
                continue
            lab = potts_dp(np.where(H[:, b][:, None] > 0, V[:, b], 0.0), np.where(H[:, b] > 0, H[:, b], 1e-9), kappa)
            if K == 1:
                z = V[:, b, 0]
                V[:, b, 0] = wsegment_levels(lab, z, H[:, b], "regression")
            else:
                V[:, b] = wsegment_levels(lab, None, H[:, b], "multiclass", C2=V[:, b])
    V = np.where(H[:, :, None] > 0, V, 0.0)
    # re-centre on live cells (match the engine's convention)
    live = H > 0
    if live.any():
        if K == 1:
            m = (V[:, :, 0][live] * H[live]).sum() / max(H[live].sum(), 1e-12)
            V[:, :, 0] = np.where(live, V[:, :, 0] - m, 0.0)
        else:
            m = (V[live] * H[live][:, None]).sum(axis=0) / max(H[live].sum(), 1e-12)
            V = np.where(live[:, :, None], V - m[None, None, :], 0.0)
    r71.R84_COST["pair_ctx"] -= 1
    return V[:, :, 0] if one_d else V


# ----------------------------------------------------------------------------
# model components.  A Component owns its routing + level table and can refit on
# the current offset.  The Stage-3.1 model = Mains + admitted atoms (+ switch).
# ----------------------------------------------------------------------------
class Component:
    kind = "abstract"
    dim = 1

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1

    def snapshot(self):
        return np.zeros(1)

    def route(self, X):
        raise NotImplementedError

    def contrib(self, X):
        raise NotImplementedError

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        raise NotImplementedError

    def n_params(self):
        raise NotImplementedError


class Mains(Component):
    """The 1-D main effects: engine-fit_card semantics per feature (weighted variant)."""

    kind = "mains"

    def __init__(self, Xtr, m=32, names=None, cuts=None):
        self.n, self.p = Xtr.shape
        self.cuts = [cuts[j] if cuts is not None else cuts_of(Xtr[:, j], m) for j in range(self.p)]
        self.nb = [len(self.cuts[j]) + 2 for j in range(self.p)]
        self.idx = [bin_idx(Xtr[:, j], self.cuts[j]) for j in range(self.p)]
        self.V = [np.zeros(self.nb[j]) for j in range(self.p)]
        self.Vc = None                  # multiclass tables live here after first refit
        self.names = names

    def route(self, X):
        return [bin_idx(X[:, j], self.cuts[j]) for j in range(self.p)]

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1
        if self.dim > 1 and self.Vc is None:
            self.Vc = [np.zeros((self.nb[j], self.dim)) for j in range(self.p)]

    def contrib(self, X):
        idx = self.route(X)
        n = len(X)
        if self.Vc is not None:
            eta = np.zeros((n, self.dim))
            for j in range(self.p):
                eta = eta + self.Vc[j][idx[j]]
            return eta
        eta = np.zeros((n, self.dim)) if self.dim > 1 else np.zeros(n)
        for j in range(self.p):
            eta = eta + self.V[j][idx[j]]
        return eta

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        idx = self.route(X)
        if w is None:
            w = np.ones(len(y))
        K = n_classes if kind == "multiclass" else 0
        if kind == "multiclass":
            if self.Vc is None:
                self.Vc = [np.zeros((self.nb[j], K)) for j in range(self.p)]
            Y1 = onehot(y, K)
            for j in range(self.p):
                o = off - self.Vc[j][idx[j]]
                C2, H = wsolve_bin_multiclass(o, Y1, self.nb[j], idx[j], K, w)
                emp = H <= 1e-12
                kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
                lab = potts_dp(np.where(emp[:, None], 0.0, C2), np.where(emp, 1e-9, H), kap)
                V = wsegment_levels(lab, None, H, "multiclass", C2=np.where(emp[:, None], 0.0, C2))
                self.Vc[j] = V
                off = o + V[idx[j]]
        else:
            for j in range(self.p):
                o = off - self.V[j][idx[j]]
                if kind == "binary":
                    z, H = wsolve_bin_logistic(o, y, self.nb[j], idx[j], w)
                else:
                    z, H = wsolve_bin_regression(o, y, self.nb[j], idx[j], w)
                emp = H <= 1e-12
                zc = np.where(emp, 0.0, z)
                kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
                lab = potts_dp(zc, np.where(emp, 1e-9, H), kap)
                V = wsegment_levels(lab, zc, H, "binary" if kind == "binary" else "regression")
                self.V[j] = V
                off = o + V[idx[j]]
        return off

    def zero_levels(self):
        """Reset all level tables to zero (frozen cut structure, zeroed levels)."""
        if self.Vc is not None:
            self.Vc = [np.zeros_like(v) for v in self.Vc]
        self.V = [np.zeros(self.nb[j]) for j in range(self.p)]

    def n_params(self):
        if self.Vc is not None:
            return int(sum(int(np.sum(np.any(np.diff(v, axis=0) != 0, axis=1))) + 1 for v in self.Vc))
        return int(sum(int(np.sum(np.diff(v) != 0)) + 1 for v in self.V))


class RectAtom(Component):
    """p1: un-gated product-Potts rectangle on the cell grid of pair (j, k)."""

    kind = "p1"

    def __init__(self, j, k, cuts_j, cuts_k, B=8):
        self.j, self.k = j, k
        self.B = B
        self.cj = cuts_j
        self.ck = cuts_k
        self.A = len(cuts_j) + 2
        self.Bn = len(cuts_k) + 2
        self.V = np.zeros((self.A, self.Bn))
        self.Vc = None

    def cells(self, X):
        a = bin_idx(X[:, self.j], self.cj)
        b = bin_idx(X[:, self.k], self.ck)
        return a, b

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1
        if self.dim > 1 and self.Vc is None:
            self.Vc = np.zeros((self.A, self.Bn, self.dim))

    def contrib(self, X):
        a, b = self.cells(X)
        if self.Vc is not None:
            return self.Vc[a, b]
        if self.dim > 1:
            return np.zeros((len(X), self.dim))
        return self.V[a, b]

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        a, b = self.cells(X)
        n = len(y)
        if w is None:
            w = np.ones(n)
        flat = a * self.Bn + b
        nb = self.A * self.Bn
        cur = self.contrib(X)
        o = off - cur
        if kind == "multiclass":
            K = n_classes
            if self.Vc is None:
                self.Vc = np.zeros((self.A, self.Bn, K))
            C2, H = wsolve_bin_multiclass(o, onehot(y, K), nb, flat, K, w)
            grid = C2.reshape(self.A, self.Bn, K)
            Hg = H.reshape(self.A, self.Bn)
            kap = 0.5 * np.log(max(float(Hg.sum()), 1.0)) * kappa_mult
            self.Vc = product_potts(grid, Hg, kap)
            return o + self.Vc[a, b]
        if kind == "binary":
            z, H = wsolve_bin_logistic(o, y, nb, flat, w)
        else:
            z, H = wsolve_bin_regression(o, y, nb, flat, w)
        zg = z.reshape(self.A, self.Bn)
        Hg = H.reshape(self.A, self.Bn)
        kap = 0.5 * np.log(max(float(Hg.sum()), 1.0)) * kappa_mult
        self.V = product_potts(zg, Hg, kap)
        return o + self.V[a, b]

    def n_params(self):
        V = self.Vc if self.Vc is not None else self.V
        if V.ndim == 3:
            live = int(np.sum(np.any(np.diff(V.reshape(-1, V.shape[2]), axis=0) != 0, axis=1)))
        else:
            live = int(np.sum(np.diff(V.ravel()) != 0))
        return live + 1

    def snapshot(self):
        return (self.Vc if self.Vc is not None else self.V).ravel().copy()


class ProjAtom(Component):
    """g4: interior-gated 1-D projection atom u = r_j (+/-) r_k."""

    kind = "g4"

    def __init__(self, j, k, sign, cuts_j, cuts_k, nbins=16, interior_only=True):
        self.j, self.k, self.sign = j, k, sign
        self.cj, self.ck = cuts_j, cuts_k
        self.m = nbins
        self.interior_only = interior_only
        self.ucuts = None
        self.V = None
        self.Vc = None

    def _u(self, X):
        # empirical midrank coordinates via the training cut structure: r in [0,1]
        aj = bin_idx(X[:, self.j], self.cj)
        ak = bin_idx(X[:, self.k], self.ck)
        mj, mk = len(self.cj) + 1, len(self.ck) + 1
        rj = aj / mj
        rk = ak / mk
        interior = (aj > 0) & (aj < mj) & (ak > 0) & (ak < mk) if self.interior_only else np.ones(len(X), bool)
        u = (rj + rk) if self.sign > 0 else (rj - rk)
        return u, interior

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1

    def contrib(self, X):
        u, interior = self._u(X)
        if self.V is None and self.Vc is None:
            return np.zeros((len(X), self.dim)) if self.dim > 1 else np.zeros(len(X))
        b = bin_idx(u, self.ucuts) if self.ucuts is not None and len(self.ucuts) else np.zeros(len(X), int)
        b = np.where(interior, b, -1)
        if self.Vc is not None:
            out = np.zeros((len(X), self.Vc.shape[1]))
            m = b >= 0
            out[m] = self.Vc[b[m]]
            return out
        out = np.zeros(len(X))
        m = b >= 0
        out[m] = self.V[b[m]]
        return out

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        n = len(y)
        if w is None:
            w = np.ones(n)
        u, interior = self._u(X)
        cur = self.contrib(X)
        o = off - cur
        if self.ucuts is None:
            self.ucuts = cuts_of(u[interior], self.m) if interior.any() else np.array([])
        b = bin_idx(u, self.ucuts) if len(self.ucuts) else np.zeros(n, int)
        b = np.where(interior, b, -1)
        sel = b >= 0
        nb = (len(self.ucuts) + 2) if len(self.ucuts) else 2
        if kind == "multiclass":
            K = n_classes
            if self.Vc is None:
                self.Vc = np.zeros((nb, K))
            C2, H = wsolve_bin_multiclass(o[sel], onehot(y, K)[sel], nb, b[sel].astype(int), K, w[sel])
            emp = H <= 1e-12
            kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
            lab = potts_dp(np.where(emp[:, None], 0.0, C2), np.where(emp, 1e-9, H), kap)
            V = wsegment_levels(lab, None, H, "multiclass", C2=np.where(emp[:, None], 0.0, C2))
            self.Vc = V
            # boundary/NaN bin of the atom stays 0 (purified against mains + p1)
            return o + self.contrib(X)
        if kind == "binary":
            z, H = wsolve_bin_logistic(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        else:
            z, H = wsolve_bin_regression(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        emp = H <= 1e-12
        zc = np.where(emp, 0.0, z)
        kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
        lab = potts_dp(zc, np.where(emp, 1e-9, H), kap)
        V = wsegment_levels(lab, zc, H, "binary" if kind == "binary" else "regression")
        self.V = V
        return o + self.contrib(X)

    def n_params(self):
        V = self.Vc if self.Vc is not None else self.V
        if V is None:
            return 0
        if V.ndim == 2:
            return int(np.sum(np.any(np.diff(V, axis=0) != 0, axis=1))) + 1
        return int(np.sum(np.diff(V) != 0)) + 1

    def snapshot(self):
        V = self.Vc if self.Vc is not None else self.V
        return V.ravel().copy() if V is not None else np.zeros(1)


class GrammarAtom(Component):
    """g4: the projection grammar u1 = r_j + r_k, u2 = r_j - r_k, both interior-gated.

    The two step functions are refitted jointly (Gauss-Seidel) and count as one arbitration
    atom.  Rows touching a parent boundary bin route to the trailing bin whose level is zeroed
    (purification against the 1-D mains and the p1 rectangles) — the Stage-3.1 aliasing fix.
    """

    kind = "g4"

    def __init__(self, j, k, cuts_j, cuts_k, nbins=16):
        self.j, self.k = j, k
        self.u1 = ProjAtom(j, k, +1, cuts_j, cuts_k, nbins=nbins)
        self.u2 = ProjAtom(j, k, -1, cuts_j, cuts_k, nbins=nbins)

    def ensure(self, kind, n_classes=0):
        self.u1.ensure(kind, n_classes)
        self.u2.ensure(kind, n_classes)
        self.dim = self.u1.dim

    def contrib(self, X):
        return self.u1.contrib(X) + self.u2.contrib(X)

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        eta = self.u1.refit(X, y, off, kind, kappa_mult=kappa_mult, w=w, n_classes=n_classes)
        eta = self.u2.refit(X, y, eta, kind, kappa_mult=kappa_mult, w=w, n_classes=n_classes)
        return eta

    def n_params(self):
        return self.u1.n_params() + self.u2.n_params()

    def snapshot(self):
        return np.concatenate([self.u1.snapshot(), self.u2.snapshot()])


class BilinearAtom(Component):
    """m2: gamma_jk * z_j[b_j] * z_k[b_k], z = centred levels of the two mains at call time."""

    kind = "m2"

    def __init__(self, j, k, zj, zk):
        self.j, self.k = j, k
        self.zj, self.zk = zj, zk          # frozen centred level vectors (nb_j,), (nb_k,)
        self.gamma = 0.0

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1
        if self.dim > 1 and np.size(self.gamma) == 1:
            self.gamma = np.zeros(self.dim)

    def contrib(self, X):
        bj = bin_idx(X[:, self.j], self.cj)
        bk = bin_idx(X[:, self.k], self.ck)
        phi = self.zj[bj] * self.zk[bk]
        if self.dim > 1:
            return phi[:, None] * np.asarray(self.gamma)[None, :]
        return self.gamma * phi

    def set_cuts(self, cj, ck):
        self.cj, self.ck = cj, ck

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        n = len(y)
        if w is None:
            w = np.ones(n)
        bj = bin_idx(X[:, self.j], self.cj)
        bk = bin_idx(X[:, self.k], self.ck)
        phi = self.zj[bj] * self.zk[bk]
        cur = self.contrib(X)
        o = off - cur
        if kind == "regression":
            num = float((w * phi * (y - o)).sum())
            den = float((w * phi * phi).sum()) + 1e-9
            self.gamma = np.clip(num / den, -50.0, 50.0)
            return o + self.gamma * phi
        if kind == "binary":
            g = 0.0
            for _ in range(25):
                p = sig(o + g * phi)
                num = float((w * phi * (y - p)).sum())
                den = float((w * phi * phi * p * (1 - p)).sum()) + 1e-9
                step = np.clip(num / den, -2.0, 2.0)
                g += 0.5 * step
                if abs(step) < 1e-7:
                    break
            self.gamma = float(np.clip(g, -50.0, 50.0))
            return o + self.gamma * phi
        K = n_classes
        gam = np.zeros(K)
        Y1 = onehot(y, K)
        for _ in range(25):
            P = softmax(o + phi[:, None] * gam[None, :])
            num = (w[:, None] * phi[:, None] * (Y1 - P)).sum(axis=0)
            den = (w[:, None] * phi[:, None] ** 2 * P * (1 - P)).sum(axis=0) + 1e-9
            step = np.clip(num / den, -2.0, 2.0)
            gam += 0.5 * step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.gamma = np.clip(gam, -50.0, 50.0)
        return o + phi[:, None] * self.gamma[None, :]

    def n_params(self):
        return int(np.size(self.gamma))

    def snapshot(self):
        return np.atleast_1d(self.gamma).ravel().copy()
