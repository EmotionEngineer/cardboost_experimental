"""r82_merge.py -- R82 Part 2 (H11): Opus #5 prequential OOF segment merger.

After the 1-D fit, adjacent segments of every main are tested on 5 Toof
subfolds: removing a segment boundary (merging two adjacent shelves into their
H-weighted mean level) is ACCEPTED iff the aggregate OOF loss does not worsen
(dLoss_OOF <= 0, user convention).  Levels for the test are refit on the
sub-train rows with the segment structure fixed (kept vs merged); the deployed
card's merged shelf takes the H-weighted mean of the two existing levels.
Compresses overfit steps (anneal f2) and can only DECREASE segment counts.
Name-blind (Rule 12.B-3).
"""
from __future__ import annotations

import numpy as np

from r71_universal import (_segment_levels, bin_idx, onehot,
                          solve_bin_logistic, solve_bin_multiclass,
                          solve_bin_regression)


def _seg_labels(Vj):
    Vj = np.asarray(Vj, float)
    if Vj.ndim == 1:
        return np.concatenate([[0], np.cumsum(np.diff(Vj) != 0)]).astype(int)
    return np.concatenate([[0], np.cumsum(np.any(np.diff(Vj, axis=0) != 0,
                                                 axis=1))]).astype(int)


def _levels_for(lab, z, H, kind):
    if kind == "multiclass":
        return _segment_levels(lab, None, H, "multiclass", C2=z, shrink=0.0)
    return _segment_levels(lab, z, H,
                           "binary" if kind == "binary" else "regression",
                           shrink=0.0)


def _row_loss(kind, y, s):
    if kind == "regression":
        return float(np.sum((y - s) ** 2))
    if kind == "binary":
        p = 1.0 / (1.0 + np.exp(-s))
        return float(-np.sum(y * np.log(np.clip(p, 1e-12, 1))
                            + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    P = np.exp(s - s.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return float(-np.sum(np.log(np.clip(P[np.arange(len(y)), y.astype(int)],
                                        1e-12, 1.0))))


def merge_card(card, X, y, kind, n_classes=0, n_sub=5, seed=109):
    """greedy left-to-right OOF merge of interior segment boundaries, all mains."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    parts = [np.asarray(p, int) for p in
             np.array_split(rng.permutation(len(y)), n_sub)]
    p = X.shape[1]
    n_merged = 0
    for j in range(p):
        Vj = np.asarray(card.V[j], float).copy()
        lab = _seg_labels(Vj)
        if lab.max() < 1:
            continue
        cuts_j = np.asarray(card.cuts[j], float)
        Ij = bin_idx(X[:, j], cuts_j)
        nb = len(cuts_j) + 2
        eta = np.asarray(card.score(X), float)
        contrib_j = Vj[Ij]
        off = eta - contrib_j
        Hj = np.asarray(card.meta["H"][j], float)

        def refit_levels(lab_cur, F):
            idxF = Ij[F]
            offF = off[F]
            yF = y[F]
            if kind == "multiclass":
                C2, H = solve_bin_multiclass(offF, onehot(yF, n_classes),
                                             nb, idxF, n_classes, alpha=0.5)
                return _levels_for(lab_cur, C2, H, kind)
            if kind == "binary":
                z, H = solve_bin_logistic(offF, yF, nb, idxF, alpha=0.5)
            else:
                z, H = solve_bin_regression(offF, yF, nb, idxF, alpha=0.0)
            return _levels_for(lab_cur, z, H, kind)

        boundaries = [b for b in range(len(lab) - 1) if lab[b] != lab[b + 1]]
        for b0 in boundaries:
            if lab[b0] == lab[b0 + 1]:
                continue
            A, B = int(lab[b0]), int(lab[b0 + 1])
            lab_try = lab.copy()
            lab_try[lab == B] = A
            dloss = 0.0
            for s in range(n_sub):
                V_idx = parts[s]
                F = np.concatenate([parts[g] for g in range(n_sub) if g != s])
                Lk = refit_levels(lab, F)
                Lm = refit_levels(lab_try, F)
                s_keep = off[V_idx] + Lk[Ij[V_idx]]
                s_mrg = off[V_idx] + Lm[Ij[V_idx]]
                dloss += _row_loss(kind, y[V_idx], s_mrg) \
                    - _row_loss(kind, y[V_idx], s_keep)
            if dloss <= 0.0:
                selA, selB = lab == A, lab == B
                wA = float(Hj[selA].sum())
                wB = float(Hj[selB].sum())
                if Vj.ndim == 2:
                    mA = (Vj[selA] * Hj[selA][:, None]).sum(axis=0) / max(wA, 1e-12)
                    mB = (Vj[selB] * Hj[selB][:, None]).sum(axis=0) / max(wB, 1e-12)
                else:
                    mA = float((Vj[selA] * Hj[selA]).sum() / max(wA, 1e-12))
                    mB = float((Vj[selB] * Hj[selB]).sum() / max(wB, 1e-12))
                Vj[selA | selB] = (wA * mA + wB * mB) / max(wA + wB, 1e-12)
                lab = lab_try
                n_merged += 1
        card.V[j] = Vj
    if isinstance(getattr(card, "meta", None), dict):
        card.meta["oof_merges"] = n_merged
    return card
