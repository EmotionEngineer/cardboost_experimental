"""r81_lowrank.py -- R81 Part 3 (H3): low-rank pair surfaces, three forms
(user decision 2026-11-13):

  s1 : rank-1 separable surface   S(x) = u(x_j) v(x_k)
  s2 : rank-2 separable surface   S(x) = sum_{r=1..2} u_r(x_j) v_r(x_k)
  av : one-step H-weighted rectangle ANOVA contrast, purified against the
       mains (two-way H-weighted demeaning), support-shrunk.

All three fit the CURRENT chain residual on the pair grid (same binned solver
as p1: wsolve_bin_*), so they slot into the preq ledger unchanged; rank/form
selection happens by prequential admission of (pair, form) candidates.
Regularisation per triage #8: Haldane-style 0.5 pseudo-support in the factor
normal equations + 1-D Potts fusion of the factors (potts_dp) + support
shrink n/(n+25).  Name-blind (Rule 12.B-3).
"""
from __future__ import annotations

import numpy as np

from r71_universal import onehot, potts_dp
from r76_mech import (wsolve_bin_logistic, wsolve_bin_multiclass,
                      wsolve_bin_regression)
from r77_mech import P1Atom


def _fuse_factor(u, Hrow, kap):
    """1-D Potts fusion of a factor over its bins (support-aware)."""
    if len(u) < 2:
        return u
    lab = potts_dp(np.asarray(u, float).copy(),
                   np.where(Hrow > 0, Hrow, 1e-9), kap)
    out = np.zeros_like(u, dtype=float)
    for k in np.unique(lab):
        sel = lab == k
        w = float(Hrow[sel].sum())
        out[sel] = float((u[sel] * Hrow[sel]).sum() / w) if w > 0 else 0.0
    return out


class _LowRankBase(P1Atom):
    rank = 1

    def __init__(self, j, k, cuts_j, cuts_k, xs_j, xs_k):
        super().__init__(j, k, cuts_j, cuts_k, xs_j, xs_k)
        self.U = None      # (A, rank) or (A, rank, K)
        self.Vf = None     # (Bn, rank) or (Bn, rank, K)

    def contrib(self, X):
        a, b, ins = self._cells(X)
        if self.U is None:
            return np.zeros(len(X))
        if self.U.ndim == 3:
            out = np.zeros((len(X), self.U.shape[2]))
            for c in range(self.U.shape[2]):
                S = self.U[:, :, c] @ self.Vf[:, :, c].T    # (A, Bn)
                out[ins, c] = S[a[ins], b[ins]]
            return out
        S = self.U @ self.Vf.T
        out = np.zeros(len(X))
        out[ins] = S[a[ins], b[ins]]
        return out

    def _fit_factors(self, z, H, kappa_mult):
        """H-weighted rank-r ALS + per-factor Potts fusion + support shrink."""
        A, Bn = z.shape
        r = self.rank
        u, s, vt = np.linalg.svd(np.nan_to_num(z), full_matrices=False)
        U = np.zeros((A, r))
        Vf = np.zeros((Bn, r))
        for i in range(r):
            sc = np.sqrt(max(float(s[i]), 1e-12))
            U[:, i] = u[:, i] * sc
            Vf[:, i] = vt[i, :] * sc
        kap = 0.5 * np.log(max(float(H.sum()), 1.0)) * kappa_mult
        for _sweep in range(6):
            for i in range(r):
                R = z - (U @ Vf.T) + np.outer(U[:, i], Vf[:, i])
                den_u = (H * (Vf[:, i] ** 2)[None, :]).sum(axis=1) + 0.5
                U[:, i] = (H * R) @ Vf[:, i] / den_u
                den_v = (H * (U[:, i] ** 2)[:, None]).sum(axis=0) + 0.5
                Vf[:, i] = (H * R).T @ U[:, i] / den_v
        Hrow = H.sum(axis=1)
        Hcol = H.sum(axis=0)
        for i in range(r):
            U[:, i] = _fuse_factor(U[:, i], Hrow, kap)
            Vf[:, i] = _fuse_factor(Vf[:, i], Hcol, kap)
        U = U * (Hrow / (Hrow + 25.0))[:, None]
        Vf = Vf * (Hcol / (Hcol + 25.0))[:, None]
        U[Hrow <= 1e-12] = 0.0
        Vf[Hcol <= 1e-12] = 0.0
        return U, Vf

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        cur = self.contrib(X)
        o = off - cur
        a, b, ins = self._cells(X)
        n = len(y)
        if w is None:
            w = np.ones(n)
        flat = a * self.Bn + b
        nb = self.A * self.Bn
        if kind == "multiclass":
            K = n_classes
            C2, H = wsolve_bin_multiclass(o[ins], onehot(y, K)[ins], nb,
                                          flat[ins], K, w[ins])
            grid = C2.reshape(self.A, self.Bn, K)
            Hg = H.reshape(self.A, self.Bn)
            U = np.zeros((self.A, self.rank, K))
            Vf = np.zeros((self.Bn, self.rank, K))
            for c in range(K):
                U[:, :, c], Vf[:, :, c] = self._fit_factors(
                    grid[:, :, c], Hg, kappa_mult)
            self.U, self.Vf = U, Vf
            return o + self.contrib(X)
        if kind == "binary":
            z, H = wsolve_bin_logistic(o[ins], y[ins], nb, flat[ins], w[ins])
        else:
            z, H = wsolve_bin_regression(o[ins], y[ins], nb, flat[ins], w[ins])
        self.U, self.Vf = self._fit_factors(
            z.reshape(self.A, self.Bn), H.reshape(self.A, self.Bn),
            kappa_mult)
        return o + self.contrib(X)


class Rank1Atom(_LowRankBase):
    kind = "s1"
    rank = 1


class Rank2Atom(_LowRankBase):
    kind = "s2"
    rank = 2


class AnovaContrastAtom(P1Atom):
    """av: one-step H-weighted rectangle ANOVA contrast, purified against the
    mains (two-way H-weighted demeaning), demean-then-shrink."""

    kind = "av"

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        cur = self.contrib(X)
        o = off - cur
        a, b, ins = self._cells(X)
        n = len(y)
        if w is None:
            w = np.ones(n)
        flat = a * self.Bn + b
        nb = self.A * self.Bn
        if kind == "multiclass":
            K = n_classes
            C2, H = wsolve_bin_multiclass(o[ins], onehot(y, K)[ins], nb,
                                          flat[ins], K, w[ins])
            grid = C2.reshape(self.A, self.Bn, K)
            Hg = H.reshape(self.A, self.Bn)
            Vc = np.zeros_like(grid)
            for c in range(K):
                Vc[:, :, c] = self._contrast(grid[:, :, c], Hg)
            self.Vc = Vc
            return o + self.contrib(X)
        if kind == "binary":
            z, H = wsolve_bin_logistic(o[ins], y[ins], nb, flat[ins], w[ins])
        else:
            z, H = wsolve_bin_regression(o[ins], y[ins], nb, flat[ins], w[ins])
        self.V = self._contrast(z.reshape(self.A, self.Bn),
                                H.reshape(self.A, self.Bn))
        return o + self.contrib(X)

    @staticmethod
    def _contrast(z, H):
        W = np.maximum(H, 1e-9)
        T = np.asarray(z, float).copy()
        for _ in range(2):
            rm = (T * W).sum(axis=1) / W.sum(axis=1)
            T = T - rm[:, None]
            cm = (T * W).sum(axis=0) / W.sum(axis=0)
            T = T - cm[None, :]
        T = T * (H / (H + 25.0))
        T[H <= 1e-12] = 0.0
        return T
