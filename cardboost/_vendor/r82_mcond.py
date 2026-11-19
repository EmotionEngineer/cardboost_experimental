"""r82_mcond.py -- R82 Part 1 (H8): missingness-conditioned response cards (#4).

Atom ``mcond`` (j | M_k), user formula 2026-11-13:

    a_{j|M_k}(x) = (M_k(x) - Mbar_{k, bin_j(x_j)}) * h[bin_j(x_j)],
    M_k(x) = 1[x_k is NaN],

with the in-bin missing rate
    Mbar_{k,b} = sum_{i in b} w_i M_ik / sum_{i in b} w_i.
Subtracting Mbar makes the atom STRICTLY H-ORTHOGONAL to the 1-D main
V_j[bin_j(x_j)] inside every bin b.  The vector h in R^{m_j} is obtained with
ONE potts_dp call over bins, weights
    Ht_b = sum_{i in b} w_i (M_ik - Mbar_{k,b})^2,
bin raw coefficients h_raw_b = num_b / Ht_b with
    num_b = sum_{i in b} w_i (M_ik - Mbar_{k,b}) r_i,
and Haldane shrink Ht_seg / (Ht_seg + 0.5) after the merge.

Candidate screen (pool admission, displacement guard from R81):
  * a candidate (j, k) exists only if feature k has >= min_rows rows with a
    NaN AND >= min_rows rows without one (both modes supported);
  * candidates are scored by the ORTHOGONAL score sum_b num_b^2/(Ht_b+0.5)
    (first-order deviance gain of the un-smoothed atom) at the base card, and
    only the top-K (K=12) enter the ledger pool.
  * panels without missings (California) produce an EMPTY pool (0 candidates,
    d(lnM)=0 -> results bit-exact).
Name-blind (Rule 12.B-3).
"""
from __future__ import annotations

import numpy as np

from r76_mech import bin_idx, cuts_of, onehot, potts_dp


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


def _stats(X, y, base_score, kind, j, k, cuts_j, w=None):
    """per-bin (Mbar, num, Ht) of the atom, on residual r = y - base_score."""
    n = len(y)
    if w is None:
        w = np.ones(n)
    r = _working_resid(y, base_score, kind,
                       0 if kind != "multiclass" else y.max() + 1)
    M = np.isnan(X[:, k]).astype(float)
    b = bin_idx(X[:, j], cuts_j)
    nb = len(cuts_j) + 2
    ok = b >= 0
    Wb = np.bincount(b[ok], weights=w[ok], minlength=nb)
    Mb = np.bincount(b[ok], weights=w[ok] * M[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        Mbar = np.where(Wb > 0, Mb / np.maximum(Wb, 1e-12), 0.0)
    Cen = M - Mbar[np.clip(b, 0, nb - 1)]
    Cen[~ok] = 0.0
    Ht = np.bincount(b[ok], weights=(w * Cen ** 2)[ok], minlength=nb)
    if r.ndim == 1:
        num = np.bincount(b[ok], weights=(w * Cen * r)[ok], minlength=nb)
    else:
        num = np.stack([np.bincount(b[ok], weights=(w * Cen * r[:, c])[ok],
                                    minlength=nb) for c in range(r.shape[1])], axis=1)
    return Mbar, num, Ht, b, Cen, ok


def screen_mcond(Xtr, ytr, base_score, kind, K=12, min_rows=25):
    """top-K (j, k) candidates by orthogonal score at the base residual."""
    n, p = Xtr.shape
    Xtr = np.asarray(Xtr, float)
    k_ok = []
    for k in range(p):
        nan = np.isnan(Xtr[:, k])
        if int(nan.sum()) >= min_rows and int((~nan).sum()) >= min_rows:
            k_ok.append(k)
    out = []
    for k in k_ok:
        for j in range(p):
            if j == k:
                continue
            fin = Xtr[:, j][np.isfinite(Xtr[:, j])]
            if len(fin) < 8:
                continue
            cj = cuts_of(fin, 8)
            _, num, Ht, _, _, _ = _stats(Xtr, ytr, base_score, kind, j, k, cj)
            if num.ndim == 1:
                sc = float(np.sum(num ** 2 / (Ht + 0.5)))
            else:
                sc = float(np.sum(num ** 2 / (Ht[:, None] + 0.5)))
            out.append((sc, j, k))
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(j, k, sc) for sc, j, k in out[:K]]


class MCondAtom:
    kind = "mcond"

    def __init__(self, j, k, cuts_j, xs_j):
        self.j, self.k = int(j), int(k)
        self.cj = np.asarray(cuts_j, float)
        self.m = len(self.cj) + 2
        self.Mbar = np.zeros(self.m)
        self.h = None              # (m,) or (m, K)
        self.dim = 1

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1
        if self.dim > 1 and (self.h is None or self.h.ndim == 1):
            self.h = np.zeros((self.m, self.dim))

    def contrib(self, X):
        X = np.asarray(X, float)
        n = len(X)
        M = np.isnan(X[:, self.k]).astype(float)
        b = bin_idx(X[:, self.j], self.cj)
        ok = b >= 0
        if self.h is None:
            return np.zeros(n) if self.dim == 1 else np.zeros((n, self.dim))
        Cen = M - self.Mbar[np.clip(b, 0, self.m - 1)]
        Cen[~ok] = 0.0
        if self.h.ndim == 1:
            out = np.zeros(n)
            out[ok] = (Cen * self.h[b[ok]])[ok]
            return out
        out = np.zeros((n, self.dim))
        out[ok] = Cen[ok][:, None] * self.h[b[ok]]
        return out

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        cur = self.contrib(X)
        o = off - cur
        X = np.asarray(X, float)
        n = len(y)
        if w is None:
            w = np.ones(n)
        Mbar, num, Ht, b, Cen, ok = _stats(X, y, o, kind, self.j, self.k,
                                           self.cj, w)
        self.Mbar = Mbar
        kap = 0.5 * np.log(max(float(Ht.sum()), 1.0)) * kappa_mult

        def fit_h(num_v):
            with np.errstate(invalid="ignore", divide="ignore"):
                hraw = np.where(Ht > 0, num_v / np.maximum(Ht, 1e-12), 0.0)
            lab = potts_dp(hraw.copy(), np.where(Ht > 0, Ht, 1e-9), kap)
            hv = np.zeros(self.m)
            for seg in np.unique(lab):
                sel = lab == seg
                Hs = float(Ht[sel].sum())
                Ns = float(num_v[sel].sum())
                val = Ns / max(Hs, 1e-12) * (Hs / (Hs + 0.5))   # Haldane shrink
                hv[sel] = val
            hv[Ht <= 1e-12] = 0.0
            return hv

        if num.ndim == 1:
            self.h = fit_h(num)
        else:
            self.h = np.stack([fit_h(num[:, c]) for c in range(num.shape[1])],
                              axis=1)
        return o + self.contrib(X)

    def snapshot(self):
        return np.concatenate([np.atleast_1d(np.asarray(self.Mbar, float).ravel()),
                               np.atleast_1d(np.asarray(self.h, float).ravel())]).copy()

    def n_params(self):
        return int(self.m + np.asarray(self.h).size)
