"""r81_screen12.py -- R81 Part 1 (H1): protected orthogonal pair screen (GPT #12).

All-pairs score over response-FREE orthonormal contrasts of ECDF ranks plus
missingness (the score itself is response-weighted; 0 deploy bytes).  Two
variants per the user plan:

  pure   : pairs sorted by S_12 desc, top-K.
  hybrid : first K0=10 pairs = the legacy 8x8 screen top-10 in the LEGACY order
           (W1-queue protection for cal_spat f4), remaining K-K0 slots by S_12
           among the rest.

Frozen constants (name-blind, Rule 12.B-3):
  * 4 contrasts per feature: r, r^2, r^3 (ECDF rank, tie-averaged) and the
    missingness indicator, each centred and Gram-Schmidt orthonormalised over
    rows (degenerate directions -> zero rows; basis stays response-free);
  * score  S_jk = sum_{a,b} N_ab^2 / (D_ab + 1.0),
        N_ab = sum_i g_i phi_ja(i) phi_kb(i)  (per-class g, squared over k),
        D_ab = sum_i h_i phi_ja(i)^2 phi_kb(i)^2;
    g/h are the working residual/weights at the base card (regression:
    g = y-eta, h = 1; binary: g = y-p, h = p(1-p); multiclass: g = Y1-P,
    h = sum_k P(1-P)).
"""
from __future__ import annotations

import numpy as np

import r76_stage31 as S
from r71_universal import softmax, sig


def _ecdf_ranks(col):
    nan = np.isnan(col)
    out = np.full(len(col), 0.5)
    obs = ~nan
    vals = col[obs]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals), float)
    ranks[order] = np.arange(len(vals), dtype=float)
    sv = vals[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    m = len(vals)
    out[obs] = ranks / max(m - 1, 1)
    return out, nan


def basis_matrix(X):
    """Phi: (p, 4, n) row-orthonormal contrasts (r, r^2, r^3, miss)."""
    n, p = X.shape
    Phi = np.zeros((p, 4, n))
    for j in range(p):
        r, nan = _ecdf_ranks(np.ascontiguousarray(X[:, j]))
        miss = nan.astype(float)
        cols = [r - r.mean(), r ** 2 - (r ** 2).mean(),
                r ** 3 - (r ** 3).mean(), miss - miss.mean()]
        basis = []
        for c in cols:
            v = c.astype(float).copy()
            for u in basis:
                v -= (u @ v) * u
            nv = float(np.linalg.norm(v))
            basis.append(v / nv if nv > 1e-9 else np.zeros_like(v))
        Phi[j] = np.stack(basis)
    return Phi


def _gw(Xtr, ytr, base_score, kind):
    """working residual g (n, Kc) and weights h (n,) at the base card."""
    if kind == "regression":
        return (ytr - base_score).astype(float)[:, None], np.ones(len(ytr))
    if kind == "binary":
        p = sig(base_score)
        return (ytr - p)[:, None], p * (1.0 - p)
    P = softmax(base_score)
    Y1 = np.zeros_like(P)
    Y1[np.arange(len(ytr)), ytr.astype(int)] = 1.0
    return Y1 - P, (P * (1.0 - P)).sum(axis=1)


def score12(Xtr, ytr, base_score, kind, alpha=1.0):
    """full p x p symmetric S_12 matrix."""
    n, p = Xtr.shape
    Phi = basis_matrix(Xtr)                       # (p, 4, n)
    g, h = _gw(Xtr, ytr, base_score, kind)        # (n, Kc), (n,)
    Phi2h = (Phi ** 2) * h[None, None, :]         # (p, 4, n)
    Sfull = np.zeros((p, p))
    for j in range(p):
        if j + 1 >= p:
            break
        rest = Phi[j + 1:]                        # (m, 4, n)
        N2 = np.zeros((p - j - 1, 4, 4))
        for c in range(g.shape[1]):
            Gj = Phi[j] * g[:, c][None, :]        # (4, n)
            M = np.einsum('an,kbn->kab', Gj, rest, optimize=True)
            N2 += M ** 2
        D = np.einsum('an,kbn->kab', Phi2h[j], rest ** 2, optimize=True)
        s = (N2 / (D + alpha)).sum(axis=(1, 2))   # (m,)
        Sfull[j, j + 1:] = s
        Sfull[j + 1:, j] = s
    return Sfull


def pair_screen12(Xtr, ytr, base_score, kind, K, mode="pure", K0=10):
    """mode 'pure': top-K by S_12.  mode 'hybrid': legacy top-K0 (legacy order)
    + top-(K-K0) by S_12 among the rest."""
    n, p = Xtr.shape
    K = min(K, p * (p - 1) // 2)
    if K <= 0:
        return []
    if mode == "hybrid":
        legacy = S.pair_screen(Xtr, ytr, base_score, kind, min(K0, K))
        if len(legacy) >= K:
            return legacy
        seen = set(legacy)
    Sf = score12(Xtr, ytr, base_score, kind)
    order = [((j, k), Sf[j, k]) for j in range(p) for k in range(j + 1, p)]
    order.sort(key=lambda t: (-t[1], t[0][0], t[0][1]))
    if mode == "pure":
        return [pair for pair, _ in order[:K]]
    out = list(legacy)
    for pair, _ in order:
        if len(out) >= K:
            break
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out
