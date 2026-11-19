"""r81_varcoeff.py -- R81 Part 2 (H2): VaryingCoeffAtom ('p1v', map variant).

User decision 2026-11-13: grade argument at deploy = fitted map
    g^(x) = E[ |r| | x ],  r = working residual of the chain state the atom
    corrects,  fitted leak-free (map built on the refit rows only).
Atom contribution:
    contrib(x) = alpha(g^(x)) * S(x),
    alpha(g)   = quadratic spline, 3 knots (truncated-power basis:
                 [1, t, t^2, (t-k1)+^2, (t-k2)+^2, (t-k3)+^2], t in [0,1]),
    S(x)       = the p1 pair surface (product-Potts + RSR, identical to P1Atom).

Deploy pieces per atom: surface (f16) + 6 spline coefs + 3 knots + lo/hi +
per-feature grade-map levels; grade-map cuts are shared per fold.  Everything
name-blind (Rule 12.B-3).  No test rows touch the fit.
"""
from __future__ import annotations

import numpy as np

from r76_stage31 import cuts_of
from r77_mech import P1Atom, bin_idx


class GradeMap:
    """Additive 1-D level map for E[|r| | x]: per-feature quantile-bin means of
    |r| via one backfitting round; cuts from the fit rows; NaN rows -> neutral
    (overall mean).  Positive output."""

    NBINS = 8

    def __init__(self, X, g):
        X = np.asarray(X, float)
        g = np.asarray(g, float)
        n, p = X.shape
        self.mu = float(np.mean(g)) if n else 1.0
        self.cuts = []
        self.lev = []
        pred = np.full(n, self.mu)
        for j in range(p):
            xj = X[:, j]
            fin = xj[np.isfinite(xj)]
            if len(fin) >= self.NBINS:
                cj = cuts_of(fin, self.NBINS)
            else:
                cj = np.array([], float)
            a = bin_idx(xj, cj)
            nb = len(cj) + 2
            lv = np.zeros(nb)
            for b in range(nb):
                m = a == b
                if m.any():
                    lv[b] = float(np.mean(g[m] - pred[m])) + self.mu
            # fallback: empty bins inherit the mean level
            lv[lv == 0.0] = self.mu
            self.cuts.append(cj)
            self.lev.append(lv)
            pred = pred + (lv[a] - self.mu)

    def predict(self, X):
        X = np.asarray(X, float)
        n, p = X.shape
        out = np.full(n, self.mu)
        for j in range(p):
            xj = X[:, j]
            a = bin_idx(xj, self.cuts[j])
            ok = a >= 0
            out[ok] += self.lev[j][a[ok]] - self.mu
        return np.maximum(out, 1e-9)


def spline_basis(g, knots, lo, hi):
    """(n, 6) truncated-power quadratic spline basis, 3 knots."""
    g = np.asarray(g, float)
    span = max(float(hi) - float(lo), 1e-9)
    t = np.clip((g - lo) / span, 0.0, 1.0)
    kn = np.clip((np.asarray(knots, float) - lo) / span, 0.0, 1.0)
    cols = [np.ones_like(t), t, t * t]
    for k in kn:
        d = np.maximum(t - k, 0.0)
        cols.append(d * d)
    return np.stack(cols, axis=1)


class VaryCoeffAtom(P1Atom):
    """p1 surface with a grade-modulated coefficient (see module doc)."""

    kind = "p1v"

    def __init__(self, j, k, cuts_j, cuts_k, xs_j, xs_k):
        super().__init__(j, k, cuts_j, cuts_k, xs_j, xs_k)
        self.gmap = None
        self.knots = np.array([0.25, 0.5, 0.75])
        self.lo, self.hi = 0.0, 1.0
        self.C = None          # (6,) or (K, 6) spline coefficients

    # alpha(g^(x)): (n,) or (n, K)
    def _alpha(self, X):
        g = self.gmap.predict(X) if self.gmap is not None else np.ones(len(X))
        B = spline_basis(g, self.knots, self.lo, self.hi)   # (n, 6)
        if self.C is None:
            return np.ones(len(X))
        if self.C.ndim == 1:
            return B @ self.C
        return B @ self.C.T

    def contrib(self, X):
        base = super().contrib(X)
        al = self._alpha(X)
        if base.ndim == 2:
            if al.ndim == 1:
                al = al[:, None]
            return base * al
        return base * al

    @staticmethod
    def _working_resid(y, o, kind, n_classes):
        if kind == "regression":
            return (y - o).astype(float)
        if kind == "binary":
            return y - 1.0 / (1.0 + np.exp(-o))
        P = np.exp(o - o.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        Y1 = np.zeros_like(P)
        Y1[np.arange(len(y)), y.astype(int)] = 1.0
        return Y1 - P

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        cur = self.contrib(X)
        o = off - cur
        # 1) surface on the chain residual (identical to P1Atom)
        super().refit(X, y, off, kind, kappa_mult=kappa_mult, w=w,
                      n_classes=n_classes)
        # 2) grade map g^(x) = E[|r| | x] on the SAME residual (fit rows only)
        r = self._working_resid(y, o, kind, n_classes)
        rag = np.abs(r).sum(axis=1) if r.ndim == 2 else np.abs(r)
        self.gmap = GradeMap(X, rag)
        gh = self.gmap.predict(X)
        # 3) knots + scale from the fit rows (leak-free), spline coef fit
        self.lo, self.hi = float(np.min(gh)), float(np.max(gh))
        if self.hi - self.lo < 1e-12:
            self.hi = self.lo + 1e-12
        self.knots = np.quantile(gh, [0.25, 0.5, 0.75])
        B = spline_basis(gh, self.knots, self.lo, self.hi)   # (n, 6)
        S = super().contrib(X)
        if S.ndim == 1:
            D = B * S[:, None]
            lam = 1e-6 * float(np.mean(D * D) + 1e-12)
            self.C = np.linalg.solve(D.T @ D + lam * np.eye(6), D.T @ r)
        else:
            K = S.shape[1]
            self.C = np.zeros((K, 6))
            for c in range(K):
                D = B * S[:, c][:, None]
                lam = 1e-6 * float(np.mean(D * D) + 1e-12)
                self.C[c] = np.linalg.solve(
                    D.T @ D + lam * np.eye(6), D.T @ r[:, c])
        return o + self.contrib(X)
