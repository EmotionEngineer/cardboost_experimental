#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r76_stage31.py — Stage 3.1 arbitration machinery (Round 76), engine-faithful.

The mains loop is a verbatim reproduction of r71.fit_card (b0 Laplace intercept, Fisher
ordering, same solvers/potts_dp/_segment_levels, same convergence rule) with one extension:
after each mains sweep the admitted atoms (p1 rectangles / g4 interior-gated grammar /
m2 bilinear scalars) are refitted on the full current eta.  fit_joint with atoms=[] must
reproduce fit_card bit-close (verified in r76_verify31.py).

Arbitration per the audit blueprint:
  * candidate pairs: training-only residual-energy screen (multiclass-capable);
  * per pair, the three encodings arbitrate on a shared 5-way prequential ledger
    (seed-109 splits); entry price dQ*kappa_child + ln(3*K_screen) charged in nats;
    >=4/5 sub-folds positive (the pre-registered stability bar);
  * g4 atoms are interior-gated (rows touching a parent boundary bin route to the zero
    level — the Stage-3.1 fix for fold-4 boundary-ray aliasing);
  * sw14: levels-only screen over the shared cut structure, full per-side backfit on the
    winner, charged ln(M_splits) + dQ*kappa_child, >=4/5 positive (the Anneal-leak fix);
  * the admitted set is fitted jointly with the 1-D mains (fix #3).

Usage:  python3 r76_stage31.py --panel cal_rand [--k-pairs 16] [--no-sw14]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r76_lib as L
import r76_mech as M
import r71_universal as r71
from r71_universal import (
    bin_idx, cuts_of, onehot, softmax, sig, log_softmax, potts_dp,
    solve_bin_logistic, solve_bin_regression, solve_bin_multiclass, _segment_levels,
)


# ----------------------------------------------------------------------------
# engine-faithful joint fit
# ----------------------------------------------------------------------------
class JointCard:
    def __init__(self, kind, cuts, Fv, b0, atoms, meta=None):
        self.kind = kind
        self.cuts = cuts
        self.V = Fv
        self.b0 = b0
        self.atoms = list(atoms)
        self.meta = meta or {}

    def score(self, X):
        X = np.asarray(X, float)
        n = len(X)
        if self.kind == "multiclass":
            K = len(self.b0)
            eta = np.tile(self.b0, (n, 1))
        else:
            eta = np.full(n, float(self.b0))
        for j in range(len(self.cuts)):
            eta = eta + self.V[j][bin_idx(X[:, j], self.cuts[j])]
        for a in self.atoms:
            eta = eta + a.contrib(X)
        return eta


def fit_joint(X, y, kind, atoms=(), n_classes=0, kappa_mult=1.0, m=32, alpha=0.5,
              passes=24, max_passes=48, conv_tol=1e-5, cuts=None, order=None,
              w=None, kappa_sched=None, warm_card=None, lr=1.0,
              solver="gs", nom_perm=False, final_rot=None, exact_dev=False) -> JointCard:
    """fit_card's loop with an atom sweep after each mains sweep (full-eta convention).
    `w` = optional row weights (Bayesian-bootstrap bags; w=None reproduces fit_card exactly).
    `kappa_sched` = optional callable(pass_index) -> kappa_mult (Idea 11 continuation).
    `warm_card` = start from this card's mains tables + the atoms' current tables and run
    `passes` joint backfit passes (the audit's fix #3: 4-6 warm passes over
    [1D mains + admitted atoms])."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    n, p = X.shape
    # R85 Canary 1: lr / kappa_mult may be per-feature vectors (gram damping).
    # Scalars keep the original code path byte-identical.
    lr_vec = np.asarray(lr, float) if np.ndim(lr) else None
    km_vec = np.asarray(kappa_mult, float) if np.ndim(kappa_mult) else None
    if lr_vec is not None and lr_vec.shape != (p,):
        raise ValueError("lr vector must have length p")
    if km_vec is not None and km_vec.shape != (p,):
        raise ValueError("kappa_mult vector must have length p")
    atoms = list(atoms)
    # R82-2 (#5): unique-value counts for the nominal permutation guard
    if nom_perm:
        uq = [int(len(np.unique(X[np.isfinite(X[:, j]), j])))
              if np.isfinite(X[:, j]).any() else 0 for j in range(p)]
    else:
        uq = None
    if order is None:
        order = "fisher" if kind == "multiclass" else "cyclic"   # std_fitcfg contract
    E = [cuts[j] if cuts is not None else cuts_of(X[:, j], m) for j in range(p)]
    NB = [len(E[j]) + 2 for j in range(p)]
    Itr = [bin_idx(X[:, j], E[j]) for j in range(p)]
    K = (n_classes or (int(y.max()) + 1)) if kind == "multiclass" else 0

    if kind == "regression":
        b0 = np.array(float(np.average(y, weights=w)) if w is not None else float(np.mean(y)))
        eta = np.full(n, b0)
        Fv = [np.zeros(NB[j]) for j in range(p)]
    elif kind == "binary":
        pm = (float(np.average(y, weights=w)) if w is not None else float(np.mean(y)))
        pm = np.clip(pm, 1e-6, 1 - 1e-6)
        b0 = np.array(float(np.log(pm / (1 - pm))))
        eta = np.full(n, b0)
        Fv = [np.zeros(NB[j]) for j in range(p)]
    else:
        if w is not None:
            cnt = np.bincount(y.astype(int), weights=w, minlength=K).astype(float)
        else:
            cnt = np.bincount(y.astype(int), minlength=K).astype(float)
        prior = (cnt + 1.0) / (cnt.sum() + K)
        b0 = np.log(prior)
        eta = np.tile(b0, (n, 1))
        Fv = [np.zeros((NB[j], K)) for j in range(p)]

    if warm_card is not None:
        Fv = [np.asarray(v, float).copy() for v in warm_card.V]
        b0 = np.asarray(warm_card.b0, float).copy()
        if kind == "multiclass":
            eta = np.tile(b0, (n, 1))
        else:
            eta = np.full(n, float(b0))
        for j in range(p):
            eta = eta + Fv[j][Itr[j]]
    for a in atoms:
        a.ensure(kind, K)
        eta = eta + a.contrib(X)

    Y1 = onehot(y, K) if kind == "multiclass" else None
    V_prev = [np.zeros_like(v) for v in Fv]
    A_prev = [a.snapshot() for a in atoms]
    H_list = [np.zeros(NB[j]) for j in range(p)]
    delta = np.inf
    n_it = 0
    if solver != "gs":
        # R80 Phase-1 solver branches (fairshare / corrblock); "gs" keeps the
        # original fit_card loop byte-identical below.
        import r80_solvers as RS
        st = RS.init_state(solver, X, p)
    for it in range(max(passes, max_passes if conv_tol > 0 else passes)):
        n_it = it + 1
        km = float(kappa_sched(it)) if kappa_sched is not None else float(kappa_mult if km_vec is None else 1.0)
        km_atoms = km if km_vec is None else float(km_vec.min())
        # ---- ordering (verbatim fit_card: fisher for multiclass, cyclic else) ----
        if order == "fisher":
            if kind == "multiclass":
                P = softmax(eta)
                fw = (P * (1.0 - P)).sum(axis=1)
            elif kind == "binary":
                P = sig(eta)
                fw = P * (1.0 - P)
            else:
                fw = np.ones(n)
            mass = np.array([float(fw[np.clip(np.asarray(Itr[j], int), 0, n - 1)].sum())
                             * (1.0 + len(np.unique(Itr[j]))) for j in range(p)])
            sweep = list(np.argsort(-mass))
        else:
            sweep = list(range(p))
        # ---- mains sweep (verbatim fit_card) --------------------------------
        if solver != "gs":
            s2 = (max(float(np.mean((y - eta) ** 2)), 1e-12)
                  if kind == "regression" else 1.0)
            eta = RS.mains_sweep(solver, Itr, NB, Fv, eta, y, Y1, kind, w,
                                 alpha, km, H_list, st, n, p, s2)
        for j in (sweep if solver == "gs" else ()):
            idx = Itr[j]
            nb = NB[j]
            km_j = float(km_vec[j]) if km_vec is not None else km
            lr_j = float(lr_vec[j]) if lr_vec is not None else lr
            if kind == "multiclass":
                off = eta - Fv[j][idx]
                if r71.R84_FISHER["multiclass"]:
                    # R84 Canary 3B: KxK Fisher Newton + KxK Fisher potts cost
                    if w is not None:
                        raise NotImplementedError("R84 3B: weighted (w) path out of scope")
                    C2, Wb, Gb, Hs = r71.solve_bin_multiclass_fisher(off, Y1, nb, idx, K, alpha=alpha)
                    emp = Hs <= 1e-12
                    kap = 0.5 * np.log(max(float(Hs[~emp].sum()), 1.0)) * km_j
                    if nom_perm:
                        import r82_nomcon as NC
                        pm = NC.nom_perm(np.where(emp[:, None], 0.0, C2), Hs, uq[j])
                    else:
                        pm = None
                    if pm is not None:
                        lab_p = r71.potts_dp_fisher(np.where(emp[pm][:, None], 0.0, C2[pm]),
                                                    Wb[pm], Gb[pm], emp[pm], kap)
                        lab = np.zeros(len(Hs), int)
                        lab[pm] = lab_p
                    else:
                        lab = r71.potts_dp_fisher(np.where(emp[:, None], 0.0, C2),
                                                  Wb, Gb, emp, kap)
                    V = r71._segment_levels_fisher(lab, Wb, Gb, emp)
                    H = Hs        # bookkeeping: scalar Fisher mass tr(W_b) feeds H_list
                else:
                    if w is not None:
                        C2, H = M.wsolve_bin_multiclass(off, Y1, nb, idx, K, w, alpha=alpha)
                    else:
                        C2, H = solve_bin_multiclass(off, Y1, nb, idx, K, alpha=alpha)
                    emp = H <= 1e-12
                    kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * km_j
                    if nom_perm:
                        import r82_nomcon as NC
                        pm = NC.nom_perm(np.where(emp[:, None], 0.0, C2), H, uq[j])
                    else:
                        pm = None
                    if pm is not None:
                        lab_p = potts_dp(np.where(emp[pm][:, None], 0.0, C2[pm]),
                                         np.where(emp[pm], 1e-9, H[pm]), kap)
                        lab = np.zeros(len(H), int)
                        lab[pm] = lab_p
                    else:
                        lab = potts_dp(np.where(emp[:, None], 0.0, C2), np.where(emp, 1e-9, H), kap)
                    V = _segment_levels(lab, None, H, "multiclass",
                                        C2=np.where(emp[:, None], 0.0, C2), shrink=0.0)
            else:
                off = eta - Fv[j][idx]
                if kind == "binary":
                    if w is not None:
                        z, H = M.wsolve_bin_logistic(off, y, nb, idx, w, alpha=alpha)
                    else:
                        z, H = solve_bin_logistic(off, y, nb, idx, alpha=alpha)
                else:
                    if w is not None:
                        z, H = M.wsolve_bin_regression(off, y, nb, idx, w)
                    else:
                        z, H = solve_bin_regression(off, y, nb, idx, alpha=0.0)
                emp = H <= 1e-12
                zc = np.where(emp, 0.0, z)
                kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * km_j
                # R107 (frozen 2026-11-16): exact binomial deviance for binary
                # 1D mains (sufficient stats G=wy, N=w per bin); quadratic path
                # unchanged when exact_dev=False (bit-identity contract).
                exs = dict(exact=False)
                if kind == "binary" and (exact_dev or M.EXACT_DEV["on"]):
                    if w is not None:
                        Gb = np.bincount(idx, weights=w * y, minlength=nb)
                        Nb = np.bincount(idx, weights=w, minlength=nb)
                    else:
                        Gb = np.bincount(idx, weights=y, minlength=nb)
                        Nb = np.bincount(idx, weights=np.ones(len(y)), minlength=nb)
                    exs = dict(exact=True, Gs=Gb, Ns=Nb)
                if nom_perm:
                    import r82_nomcon as NC
                    pm = NC.nom_perm(zc, H, uq[j])
                else:
                    pm = None

                def _dp(exs_kw, kap_):
                    if pm is not None:
                        exs_p = (dict(exs_kw, Gs=exs_kw["Gs"][pm], Ns=exs_kw["Ns"][pm])
                                 if exs_kw.get("exact") else exs_kw)
                        lp = potts_dp(zc[pm], np.where(emp[pm], 1e-9, H[pm]), kap_, **exs_p)
                        lb = np.zeros(len(H), int)
                        lb[pm] = lp
                        return lb
                    return potts_dp(zc, np.where(emp, 1e-9, H), kap_, **exs_kw)

                # R111 (frozen R87 Finalist 3): exact deviance при ПОТОЛКЕ
                # сегментов S_j <= S_quad,j (теорема: бюджет квадратики -- встроенный
                # регуляризатор, L12; R107 без потолка раздул карту x2.7).
                # Минимальное kappa-умножение, при котором точная девианса не даёт
                # БОЛЬШЕ сегментов, чем квадратичный путь при том же kap.
                if exs.get("exact") and M.EXACT_DEV.get("cap"):
                    lab_q = _dp(dict(exact=False), kap)
                    S_q = int(len(np.unique(lab_q)))
                    lab_e = _dp(exs, kap)
                    if len(np.unique(lab_e)) <= S_q:
                        lab = lab_e
                    else:
                        lo_g, hi_g, lab_hi = 1.0, 2.0, None
                        for _ in range(12):
                            cand = _dp(exs, kap * hi_g)
                            if len(np.unique(cand)) <= S_q:
                                lab_hi = cand
                                break
                            lo_g, hi_g = hi_g, hi_g * 2.0
                        if lab_hi is None:
                            lab = lab_q
                        else:
                            for _ in range(6):
                                mid = 0.5 * (lo_g + hi_g)
                                cand = _dp(exs, kap * mid)
                                if len(np.unique(cand)) <= S_q:
                                    hi_g, lab_hi = mid, cand
                                else:
                                    lo_g = mid
                            lab = lab_hi
                else:
                    lab = _dp(exs, kap)
                V = _segment_levels(lab, zc, H, "binary" if kind == "binary" else "regression",
                                    shrink=0.0)
            if lr_j >= 1.0:
                Fv[j] = V
            else:
                # damped mains step (user Variant B): V <- (1-lr)*V_old + lr*V_new
                Fv[j] = (1.0 - lr_j) * Fv[j] + lr_j * V
            H_list[j] = H
            eta = off + Fv[j][idx]
        # ---- atom sweep (Stage-3 extension) ---------------------------------
        for a in atoms:
            eta = a.refit(X, y, eta, kind, kappa_mult=km_atoms, n_classes=K)
        # ---- fixed-point test on the level state ----------------------------
        d_main = max((float(np.max(np.abs(np.asarray(Fv[j], float) - V_prev[j]))) for j in range(p)),
                     default=0.0)
        d_atom = 0.0
        for i, a in enumerate(atoms):
            s = a.snapshot()
            if s.shape != A_prev[i].shape:
                d_atom = max(d_atom, np.inf)
            else:
                d_atom = max(d_atom, float(np.max(np.abs(s - A_prev[i]))))
        delta = max(d_main, d_atom)
        V_prev = [np.asarray(v, float).copy() for v in Fv]
        A_prev = [a.snapshot() for a in atoms]
        if conv_tol > 0 and delta < conv_tol:
            break
    if final_rot is not None and kind == "multiclass":
        # R82-2 (#6): the LAST 1-D update runs per-contrast independent potts_dp
        # in the OOF-error contrast basis Q (K-1 fused, common direction kept as
        # one global level), rotated back: V <- V_rot Q^T.
        import r82_nomcon as NC
        km_f = float(kappa_sched(n_it - 1)) if kappa_sched is not None else float(kappa_mult if km_vec is None else 1.0)
        for j in range(p):
            idx = Itr[j]
            nb = NB[j]
            km_j = float(km_vec[j]) if km_vec is not None else km_f
            lr_j = float(lr_vec[j]) if lr_vec is not None else lr
            off = eta - Fv[j][idx]
            if w is not None:
                C2, H = M.wsolve_bin_multiclass(off, Y1, nb, idx, K, w, alpha=alpha)
            else:
                C2, H = solve_bin_multiclass(off, Y1, nb, idx, K, alpha=alpha)
            emp = H <= 1e-12
            kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * km_j
            V = NC.contrast_fuse(np.where(emp[:, None], 0.0, C2), H, final_rot,
                                 kap, nom=bool(nom_perm and uq[j] <= 12))
            if lr_j >= 1.0:
                Fv[j] = V
            else:
                Fv[j] = (1.0 - lr_j) * Fv[j] + lr_j * V
            H_list[j] = H
            eta = off + Fv[j][idx]
    return JointCard(kind, E, Fv, b0, atoms,
                     meta=dict(passes=n_it, delta=float(delta), H=H_list))


# ----------------------------------------------------------------------------
# pair screen (training-only, multiclass-capable)
# ----------------------------------------------------------------------------
def pair_screen(Xtr, ytr, base_score, kind, K_pairs, bins=8):
    n, p = Xtr.shape
    if kind == "regression":
        resid = ytr - base_score
    elif kind == "binary":
        resid = ytr - sig(base_score)
    else:
        resid = onehot(ytr, base_score.shape[1]) - softmax(base_score)
    grid = [cuts_of(Xtr[:, j], bins) for j in range(p)]
    bidx = [bin_idx(Xtr[:, j], grid[j]) for j in range(p)]
    B = bins + 2
    score = []
    for j in range(p):
        for k in range(j + 1, p):
            cell = bidx[j] * B + bidx[k]
            if resid.ndim == 1:
                m = np.bincount(cell, weights=resid, minlength=B * B)
                c = np.bincount(cell, minlength=B * B)
                mu = np.where(c > 0, m / np.maximum(c, 1), 0.0)
                score.append((float((mu ** 2 * c).sum()), j, k))
            else:
                m = np.zeros((B * B, resid.shape[1]))
                for d in range(resid.shape[1]):
                    m[:, d] = np.bincount(cell, weights=resid[:, d], minlength=B * B)
                c = np.bincount(cell, minlength=B * B)
                mu = np.where(c[:, None] > 0, m / np.maximum(c, 1)[:, None], 0.0)
                score.append((float(((mu ** 2) * c[:, None]).sum()), j, k))
    score.sort(reverse=True)
    return [(j, k) for _, j, k in score[:K_pairs]]


def make_atom(atype, j, k, card, X=None, pair_bins=8):
    """Atoms for pair (j, k).  p1/g4 build their OWN BxB quantile grid (the audit's
    rectangle/interior box), NOT the mains' 32-cut grid; m2 rides the mains' levels.
    When X is given the pair grid is cut on its training rows (frozen thereafter)."""
    if atype in ("p1", "g4"):
        src = X if X is not None else None
        if src is None:
            cj, ck = card.cuts[j], card.cuts[k]        # fallback (legacy)
        else:
            cj = cuts_of(np.asarray(src[:, j], float), pair_bins)
            ck = cuts_of(np.asarray(src[:, k], float), pair_bins)
        if atype == "p1":
            return M.RectAtom(j, k, cj, ck)
        return M.GrammarAtom(j, k, cj, ck)
    if atype == "m2":
        zj = np.asarray(card.V[j], float)
        zk = np.asarray(card.V[k], float)
        if zj.ndim == 2:
            zj = np.linalg.norm(zj, axis=1)
        if zk.ndim == 2:
            zk = np.linalg.norm(zk, axis=1)
        zj = zj - zj.mean()
        zk = zk - zk.mean()
        a = M.BilinearAtom(j, k, zj, zk)
        a.set_cuts(card.cuts[j], card.cuts[k])
        return a
    raise ValueError(atype)


# ----------------------------------------------------------------------------
# losses + prequential ledger
# ----------------------------------------------------------------------------
def _nll(kind, y, s):
    if kind == "regression":
        return float(np.sum((y - s) ** 2) / 2.0)
    if kind == "binary":
        return float(np.sum(np.logaddexp(0.0, s) - y * s))
    P = softmax(s)
    Y = onehot(y, s.shape[1])
    return float(-np.sum(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))


def prequential_gains(candidates, Xtr, ytr, kind, n_classes, base_cuts=None,
                      seed=109, n_sub=5, base_passes=24, base_tol=1e-5, split="random",
                      space_col=6):
    """Shared 5-way prequential ledger.  Base fitted once per sub-fold (certified engine
    loop); each candidate atom fitted on the frozen base residual; per-sub-fold nats gains
    on the held-out rows, with a sub-fold T20 temperature for classification panels.

    split='spatial' (california_spatial): the sub-folds are 5 CONTIGUOUS blocks along
    `space_col` — the admission ledger must match the deployment split structure, else
    spatial-pair rectangles pass on random-subfold autocorrelation and then fail the
    held-out block catastrophically (measured: p1:6-7 +157 nats random vs -0.45 R2 block)."""
    rng = np.random.default_rng(seed)
    if split == "spatial":
        order = np.argsort(Xtr[:, space_col], kind="stable")
        parts = np.array_split(order, n_sub)
    else:
        parts = np.array_split(rng.permutation(len(ytr)), n_sub)
    gains = {tag: np.zeros(n_sub) for tag, *_ in candidates}
    for s in range(n_sub):
        va = parts[s]
        tr = np.concatenate([parts[g] for g in range(n_sub) if g != s])
        Xtr_s, ytr_s = Xtr[tr], ytr[tr]
        base = fit_joint(Xtr_s, ytr_s, kind, atoms=[], n_classes=n_classes, cuts=base_cuts,
                         passes=base_passes, max_passes=max(base_passes, 48), conv_tol=base_tol)
        eta_tr = base.score(Xtr_s)
        eta_va = base.score(Xtr[va])
        if kind == "regression":
            T = 1.0
        else:
            k2 = int(0.8 * len(tr))
            r2 = np.random.default_rng(65).permutation(len(tr))
            tail = r2[k2:]
            try:
                T = float(r71.fit_T(ytr_s[tail], eta_tr[tail], kind))
            except Exception:
                T = 1.0
        lb = _nll(kind, ytr[va], eta_va / T)
        for tag, atype, j, k in candidates:
            at = make_atom(atype, j, k, base, X=Xtr_s)
            at.ensure(kind, n_classes)
            # atoms arbitrate WITH their own fusion on (kappa_mult=1) exactly as deployed;
            # fitting them unfused would let free cell tables overfit the sub-train
            at.refit(Xtr_s, ytr_s, eta_tr, kind, kappa_mult=1.0, n_classes=n_classes)
            eta2_va = eta_va + at.contrib(Xtr[va])
            la = _nll(kind, ytr[va], eta2_va / T)
            gains[tag][s] = lb - la
    return gains


# ----------------------------------------------------------------------------
# sw14: levels-only screen + prequential ledger with the ln(M) gate
# ----------------------------------------------------------------------------
def sw14_screen(base, Xtr, ytr, kind, n_classes, max_splits=0):
    """Levels-only screen over the shared cut structure.  With max_splits>0 (aps fast
    protocol) the split space is pre-ranked by the closed-form constant-split residual
    gain and only the top max_splits go through the levels-only fits; the ln(M) charge
    downstream uses the FULL candidate count."""
    p = len(base.cuts)
    eta = base.score(Xtr)
    if kind == "regression":
        resid = ytr - eta
    elif kind == "binary":
        resid = ytr - sig(eta)
    else:
        resid = onehot(ytr, eta.shape[1]) - softmax(eta)
    prelim = []
    M_total = 0
    for j in range(p):
        for c in range(len(base.cuts[j])):
            left = Xtr[:, j] <= base.cuts[j][c]
            nL, nR = int(left.sum()), int((~left).sum())
            if nL < 30 or nR < 30:
                continue
            M_total += 1
            if resid.ndim == 1:
                d = float(resid[left].mean() - resid[~left].mean())
            else:
                d = float(np.linalg.norm(resid[left].mean(axis=0) - resid[~left].mean(axis=0)))
            prelim.append((d * d * nL * nR / max(nL + nR, 1), j, c))
    prelim.sort(reverse=True)
    if max_splits > 0:
        prelim = prelim[:max_splits]
    cands = []
    for _, j, c in prelim:
        left = Xtr[:, j] <= base.cuts[j][c]
        g = _levels_only_gain(base, Xtr, ytr, left, kind, n_classes)
        cands.append((float(g), j, c))
    cands.sort(reverse=True)
    return cands, M_total


def _levels_only_gain(base, Xtr, ytr, left, kind, n_classes):
    """Screen-grade levels-only gain (frozen cuts, kappa=0): capped passes + 3000-row subsample."""
    eta = base.score(Xtr)
    gain = 0.0
    for side in (left, ~left):
        if side.sum() < 10:
            continue
        rows = np.where(side)[0]
        if len(rows) > 3000:
            rows = rows[np.random.default_rng(7).permutation(len(rows))[:3000]]
        sub = fit_joint(Xtr[rows], ytr[rows], kind, atoms=[], n_classes=n_classes,
                        kappa_mult=0.0, cuts=base.cuts, passes=6, conv_tol=0.0)
        gain += _nll(kind, ytr[rows], eta[rows]) - _nll(kind, ytr[rows], sub.score(Xtr[rows]))
    return gain


def sw14_prequential(base, split, Xtr, ytr, kind, n_classes, seed=109, n_sub=5, side_passes=8):
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(ytr)), n_sub)
    gains = np.zeros(n_sub)
    _, j, c = split
    cut = base.cuts[j][c]
    for s in range(n_sub):
        va = parts[s]
        tr = np.concatenate([parts[g] for g in range(n_sub) if g != s])
        Xtr_s, ytr_s = Xtr[tr], ytr[tr]
        left_s = Xtr_s[:, j] <= cut
        left_v = Xtr[va][:, j] <= cut
        lb = _nll(kind, ytr[va], base.score(Xtr[va]))
        la = 0.0
        for side_s, side_v in ((left_s, left_v), (~left_s, ~left_v)):
            if side_v.sum() == 0:
                continue
            if side_s.sum() < 10:
                la += _nll(kind, ytr[va][side_v], base.score(Xtr[va][side_v]))
                continue
            sub = fit_joint(Xtr_s[side_s], ytr_s[side_s], kind, atoms=[],
                            n_classes=n_classes, kappa_mult=0.0, cuts=base.cuts,
                            passes=side_passes, conv_tol=0.0)
            la += _nll(kind, ytr[va][side_v], sub.score(Xtr[va][side_v]))
        gains[s] = lb - la
    return gains


def card_params(card):
    """Active level count of a JointCard: per feature, number of level changes + 1."""
    q = 0
    for V in card.V:
        V = np.asarray(V, float)
        if V.ndim == 2:
            q += int(np.sum(np.any(np.diff(V, axis=0) != 0, axis=1))) + 1
        else:
            q += int(np.sum(np.diff(V) != 0)) + 1
    return q


class SwitchModel:
    def __init__(self, j, cut, left, right):
        self.j, self.cut, self.left, self.right = j, cut, left, right

    def predict(self, X):
        m = X[:, self.j] <= self.cut
        s0 = self.left.score(X[:1])
        out = np.zeros((len(X),) + s0.shape[1:]) if s0.ndim > 1 else np.zeros(len(X))
        if m.any():
            out[m] = self.left.score(X[m])
        if (~m).any():
            out[~m] = self.right.score(X[~m])
        return out


def fit_switch(base, admitted, split, Xtr, ytr, kind, n_classes, passes=6):
    """Full per-side warm backfit (mains + admitted atoms) on the winning split."""
    _, j, c = split
    cut = base.cuts[j][c]
    sides = []
    for side in (Xtr[:, j] <= cut, Xtr[:, j] > cut):
        Xs, ys = Xtr[side], ytr[side]
        b = fit_joint(Xs, ys, kind, atoms=[], n_classes=n_classes, cuts=base.cuts)
        eta_b = b.score(Xs)
        ats = []
        for v in admitted.values():
            a = make_atom(v["type"], v["j"], v["k"], b, X=Xs)
            a.ensure(kind, n_classes)
            a.refit(Xs, ys, eta_b, kind, kappa_mult=1.0, n_classes=n_classes)
            ats.append(a)
        sides.append(fit_joint(Xs, ys, kind, atoms=ats, n_classes=n_classes, cuts=base.cuts,
                               warm_card=b, passes=passes, conv_tol=0.0))
    return SwitchModel(j, cut, sides[0], sides[1])


# ----------------------------------------------------------------------------
# one fold
# ----------------------------------------------------------------------------
def calib_metrics(kind, Xtr, ytr, Xte, yte, factory):
    """Registered calibrated rows for classification panels: Toof-T and 14b (per-class)."""
    out = {}
    if kind == "regression":
        return out
    eta = L.toof_eta(factory, Xtr, ytr, kind, folds=5, seed=109)
    card = factory(Xtr, ytr, kind)
    try:
        T = float(r71.fit_T(ytr, eta, kind))
    except Exception:
        T = 1.0
    s_toof = card.score(Xte) / T
    out["Toof"] = r71.metric(kind, yte, s_toof)[0]
    spec = L.fit_T14(ytr, eta, kind, True)
    s = L.apply_T14(spec, card.score(Xte))
    out["14b"] = r71.metric(kind, yte, s)[0]
    return out


def run_fold(panel, fold, tr, te, data, out, K_pairs=16, atoms_types=("p1", "g4", "m2"),
             do_sw14=True, seed=109, fast=False):
    kind = data["kind"]
    X, y = data["X"], data["y"]
    n_classes = int(np.max(y)) + 1 if kind == "multiclass" else 0
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    t0 = time.time()

    # ---- base card (engine-faithful mains-only, fused) ----------------------
    base = fit_joint(Xtr, ytr, kind, atoms=[], n_classes=n_classes)
    m0, _ = r71.metric(kind, yte, base.score(Xte))
    L.log_jsonl(out, dict(stage="3.1", panel=panel, fold=fold, arm="base", metric=m0,
                          passes=base.meta["passes"], secs=round(time.time() - t0, 1)))
    print("   base", round(m0, 5), {"passes": base.meta["passes"],
                                    "delta": round(base.meta["delta"], 6)}, flush=True)
    if kind != "regression":
        cm = calib_metrics(kind, Xtr, ytr, Xte, yte,
                           lambda X, y, k: fit_joint(X, y, k, atoms=[], n_classes=n_classes))
        for lbl, v in cm.items():
            L.log_jsonl(out, dict(stage="3.1", panel=panel, fold=fold, arm="base_" + lbl,
                                  metric=v, secs=round(time.time() - t0, 1)))
            print("   base_" + lbl, round(v, 5), flush=True)

    # ---- pair screen + arbitration ledger ----------------------------------
    eta_tr = base.score(Xtr)
    pairs = pair_screen(Xtr, ytr, eta_tr, kind, K_pairs)
    cands = [(f"{at}:{j}-{k}", at, j, k) for (j, k) in pairs for at in atoms_types]
    gains = prequential_gains(cands, Xtr, ytr, kind, n_classes, seed=seed,
                              base_passes=12 if fast else 24,
                              base_tol=1e-4 if fast else 1e-5,
                              split="spatial" if panel == "california_spatial" else "random")
    kappa_child = 0.5 * np.log(max(len(ytr), 2))
    lnM = np.log(3 * max(len(pairs), 1))
    dq_tab = {}
    for tag, at, j, k in cands:
        aq = make_atom(at, j, k, base, X=Xtr)
        aq.ensure(kind, n_classes)
        aq.refit(Xtr, ytr, eta_tr, kind, kappa_mult=1.0, n_classes=n_classes)
        dq_tab[tag] = max(1, aq.n_params())
    admitted = {}
    for (j, k) in pairs:
        best = None
        for at in atoms_types:
            tag = f"{at}:{j}-{k}"
            g = gains[tag]
            dq = dq_tab[tag]
            # entry price: one block kappa_child + ln(M) (the r74 pair-block convention;
            # the audit's p1/g4 arms at +0.019/+0.041 cannot clear a per-level BIC price of
            # dq*kappa_child ~ 50-230 nats — code and measurements win).  The per-level BIC
            # net is kept as a diagnostic column.
            net = float(g.sum()) - (kappa_child + lnM)
            net_bic = float(g.sum()) - (dq * kappa_child + lnM)
            n_pos = int((g > 0).sum())
            cand = dict(tag=tag, type=at, j=j, k=k, net=net, net_bic=net_bic,
                        gain=float(g.sum()), subs_pos=n_pos, dq=dq, subs=g.round(3).tolist())
            L.log_jsonl(out, dict(stage="3.1", panel=panel, fold=fold, arm="ledger", **cand))
            if n_pos >= int(np.ceil(0.8 * len(g))) and net > 0:
                if best is None or cand["net"] > best["net"]:
                    best = cand
        if best is not None:
            admitted[f"{j}-{k}"] = best
    for v in admitted.values():
        L.log_jsonl(out, dict(stage="3.1", panel=panel, fold=fold, arm="admit", **v))
        print("   admit", v, flush=True)

    # ---- admitted model: warm joint backfit over [mains + admitted atoms] ---
    # (fix #3: start from the base tables + atoms fitted on the base residual,
    # then 4-6 joint passes updating levels and Potts segments together)
    def arb_fit(Xf, yf, kf):
        b = fit_joint(Xf, yf, kf, atoms=[], n_classes=n_classes)
        eta_b = b.score(Xf)
        ats = []
        for v in admitted.values():
            a = make_atom(v["type"], v["j"], v["k"], b, X=Xf)
            a.ensure(kf, n_classes)
            a.refit(Xf, yf, eta_b, kf, kappa_mult=1.0, n_classes=n_classes)
            ats.append(a)
        return fit_joint(Xf, yf, kf, atoms=ats, n_classes=n_classes,
                         warm_card=b, passes=6, conv_tol=0.0)

    am = arb_fit(Xtr, ytr, kind)
    atoms = am.atoms
    m1, _ = r71.metric(kind, yte, am.score(Xte))
    L.log_jsonl(out, dict(stage="3.1", panel=panel, fold=fold, arm="arbitration", metric=m1,
                          n_admit=len(atoms), types=[v["type"] for v in admitted.values()],
                          passes=am.meta["passes"], secs=round(time.time() - t0, 1)))
    print("   arbitration", round(m1, 5), "n_admit", len(atoms), flush=True)
    if kind != "regression":
        cm = calib_metrics(kind, Xtr, ytr, Xte, yte, lambda X, y, k: arb_fit(X, y, k))
        for lbl, v in cm.items():
            L.log_jsonl(out, dict(stage="3.1", panel=panel, fold=fold, arm="arbitration_" + lbl,
                                  metric=v, n_admit=len(atoms), secs=round(time.time() - t0, 1)))
            print("   arbitration_" + lbl, round(v, 5), flush=True)

    # ---- sw14 --------------------------------------------------------------
    if do_sw14:
        cands_s, M_total = sw14_screen(base, Xtr, ytr, kind, n_classes,
                                       max_splits=200 if fast else 0)
        M_splits = max(M_total, 2)
        best_split = cands_s[0] if cands_s else None
        row2 = dict(stage="3.1", panel=panel, fold=fold, arm="sw14_screen",
                    M_splits=M_splits, best=list(best_split) if best_split else None,
                    secs=round(time.time() - t0, 1))
        if best_split is not None:
            sw_gain = sw14_prequential(base, best_split, Xtr, ytr, kind, n_classes, seed=seed,
                                       side_passes=4 if fast else 8)
            # EBIC entry price: ln(M_splits) + dQ_active * kappa_child, dQ from the
            # levels-only per-side cards on the full train (the audit's charge).
            jx, cx = best_split[1], best_split[2]
            cut = base.cuts[jx][cx]
            q_side = 0
            for side in (Xtr[:, jx] <= cut, Xtr[:, jx] > cut):
                if side.sum() < 10:
                    continue
                sub = fit_joint(Xtr[side], ytr[side], kind, atoms=[], n_classes=n_classes,
                                kappa_mult=0.0, cuts=base.cuts, passes=6, conv_tol=0.0)
                q_side += card_params(sub)
            dq_sw = max(1, q_side - card_params(base))
            net_sw = float(np.sum(sw_gain)) - (np.log(M_splits) + 0.5 * np.log(max(len(ytr), 2)) * dq_sw)
            n_pos = int((sw_gain > 0).sum())
            row2.update(gain=float(np.sum(sw_gain)), net=net_sw, subs_pos=n_pos,
                        subs=np.asarray(sw_gain).round(3).tolist())
            L.log_jsonl(out, row2)
            print("   sw14_ledger", row2, flush=True)
        else:
            n_pos, net_sw = 0, -np.inf
            L.log_jsonl(out, row2)
        if n_pos >= int(np.ceil(0.8 * 5)) and net_sw > 0 and best_split is not None:
            sw_model = fit_switch(base, admitted, best_split, Xtr, ytr, kind, n_classes,
                                  passes=6)
            m2, _ = r71.metric(kind, yte, sw_model.predict(Xte))
            row3 = dict(stage="3.1", panel=panel, fold=fold, arm="arbitration+sw14", metric=m2,
                        n_admit=len(atoms), split=list(best_split), secs=round(time.time() - t0, 1))
        else:
            row3 = dict(stage="3.1", panel=panel, fold=fold, arm="arbitration+sw14", metric=m1,
                        n_admit=len(atoms), split=None, note="sw14 rejected by lnM gate",
                        secs=round(time.time() - t0, 1))
        L.log_jsonl(out, row3)
        print("   arbitration+sw14", row3.get("metric"), row3.get("split"), flush=True)
    return m0, m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="stage31.jsonl")
    ap.add_argument("--k-pairs", type=int, default=16)
    ap.add_argument("--no-sw14", action="store_true")
    a = ap.parse_args()
    panels = L.build_panels()
    data, folds = panels[a.panel]
    print("== stage3.1 panel", a.panel, "n =", len(data["y"]), flush=True)
    for fold, (tr, te) in enumerate(folds):
        print("-- fold", fold, flush=True)
        run_fold(a.panel, fold, tr, te, data, a.out, K_pairs=a.k_pairs, do_sw14=not a.no_sw14)


if __name__ == "__main__":
    main()
