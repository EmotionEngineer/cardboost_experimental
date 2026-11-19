#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r71_universal.py — every "best model" in the record, re-implemented in ONE file, swept over every task,
with a universal-model ranking on top.

WHY ONE FILE
    The point of this script is that the models must be legible from this file alone. Nothing is imported from the
    project's own libraries: the binning, the exact per-bin solve, the l0 Potts fusion DP, the exact-q DP, the
    exact-deviance segment cost, the multinomial backfit, the lattice-exact printer, the byte accounting, the decoder,
    the four calibrators and every estimator variant below are written out here. External dependencies are only
    numpy / scipy / scikit-learn / pandas, and optionally `interpret` for the EBM comparators.

WHAT IT RUNS
    For every (task × fold) it fits every registered candidate, scores it, prices its deployed bytes and writes a
    per-task table. Then it ranks all candidates that ran on enough tasks by a UNIVERSAL score, so the question
    "which single model is best across all tasks" is answered by measurement rather than by memory.

    Tasks            aps, california_random, california_lat10, california_2d, california_lco,
                     anneal_grouped, binary7 (7 datasets × 5 folds)
    Candidates       card_fused, card_kappa0, card_deadcut, card_deviance, card_skew, card_ramp, card_effectnoise,
                     card_jointdir, card_bagfixed, card_postcompress, card_contam, card_gated_coast,
                     card_pairlayer, linear_spine, ebm0, ebm10
    Calibrators      T20 (20 % hold-out), Toof (registered 5 inner folds), diag (per-class scales),
                     s16 (the record's per-class offsets), joint (idea #12),
                     support (idea #13)

HONESTY NOTES, READ BEFORE TRUSTING A NUMBER
    * These are RE-IMPLEMENTATIONS, not the record's bit-compatible artefacts. Every object the record quoted was
      produced by code that is not in this file; agreement to 3–4 significant digits is the most that can be claimed,
      and the script prints its own numbers rather than the record's.
    * Calibration is NOT comparable across protocols. T20 and Toof differ by up to 3.6× on the fused APS card in the
      record. Every row printed here states its calibrator; never compare a T20 row with a Toof row.
    * A deployed card must store its cuts as ceil-float32 (L18): routing is x -> #{cuts <= x}, so a real cut t must
      be stored as min{f in float32 : f >= t}. `--cut-mode rne` is provided ONLY to reproduce the historical defect,
      which the script quantifies per task.
    * Every fit reports whether it converged. A non-converged row is marked `nc` and must not be quoted as final.
    * California coast: the DEFAULT is a 16-point vertex-distance polyline. The record's coast turned out to be a
      hand-coded 18-point polyline too (run_cardboost5.COAST) -- the real difference is that it measures distance to
      the nearest SEGMENT, not vertex. `--record-coast` switches to the record's exact feature; the lat10 / 2-D / lco
      columns without it are a fair comparator of MECHANISMS, not a reproduction of the record's scores.
    * Three DIAGNOSTIC flags reproduce the record's own protocol for gap attribution (never for ranking):
      `--record-folds` (its exact fold constructions: quantile bands % 5, KMeans(10) % 5, leave-one-metro-out,
      Anneal md5 groups x seeds 42/7), `--record-coast` (its coastline), `--censored` (its Tobit readout of the
      capped California target, which its cards AND its EBM references all used).

USAGE ON A MAC MINI M4 (10-core)
    python3 r71_universal.py --list                        # tasks and candidates, no computation
    python3 r71_universal.py --selftest                    # ~5 s, no download, checks every implementation
    python3 r71_universal.py --preset quick                # ~5–15 min, 3-fold, m=16, six fast candidates
    python3 r71_universal.py --preset standard             # ~3–10 h, 15 candidates, T20 + the registered Toof
    python3 r71_universal.py --preset full --n-jobs 8      # overnight; m=64, all five calibrators, every candidate
    python3 r71_universal.py --preset standard --tasks aps,california_random --n-jobs 6   # a targeted slice

    Output: r71_out/report.md (tables), r71_out/results.json (every row), r71_out/run.log.
    The universal ranking is section 2 of report.md; section 4 lists everything that did not run, and why.

    Parallelism is over (task, fold) via joblib. n_jobs=-1 uses all cores; on a 10-core M4 with 16–32 GB, use
    `--n-jobs 6` for `full` so the memory of APS (12k × 170 with 64 bins/feature) does not thrash.

    Data: OpenML/UCI downloads are cached under ./data_cache. If the network is unavailable, drop the raw files there
    or pass --data-dir. A task that cannot be loaded is reported as UNAVAILABLE and skipped; the run continues.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import io
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------------------
# 0.  Small numeric helpers
# --------------------------------------------------------------------------------------------------------------
DT = np.float64
F32 = np.float32
ALPHA_DEFAULT = 0.5          # the frozen per-bin ridge used throughout the record
CACHE = os.path.join(os.getcwd(), "data_cache")


def sig(z: np.ndarray) -> np.ndarray:
    """Logistic link, clipped. Shared by every estimator in this file."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def log_softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def onehot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


# --------------------------------------------------------------------------------------------------------------
# 1.  Binning and the exact per-bin solve
# --------------------------------------------------------------------------------------------------------------
def cuts_of(x: np.ndarray, m: int) -> np.ndarray:
    """Quantile cut grid: midpoints of consecutive order statistics (the record's F.cuts_of)."""
    ok = np.isfinite(x)
    u = np.unique(x[ok])
    if len(u) <= m:
        return (u[:-1] + u[1:]) / 2.0 if len(u) > 1 else np.array([])
    q = np.quantile(x[ok], np.linspace(0.0, 1.0, m + 1)[1:-1])
    return np.unique(q)


def bin_idx(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Right-closed routing, NaN gets its own trailing bin."""
    fin = np.isfinite(x)
    b = np.searchsorted(cuts, np.where(fin, x, cuts[0] if len(cuts) else 0.0), side="right")
    b[~fin] = len(cuts) + 1
    return b


def solve_bin_logistic(off: np.ndarray, y: np.ndarray, nb: int, idx: np.ndarray,
                       alpha: float = ALPHA_DEFAULT, iters: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Exact per-bin penalised logistic MLE by bisection, plus its Fisher weight.

        g_b(v) = sum_{i in b} (y_i - sig(off_i + v)) - alpha*v = 0,   H_b = sum_{i in b} p(1-p) + alpha

    This is the aggregate the record's fitter used. Substituting the linearised rate (z = k/n - obar, H = n) is a
    different estimator and costs ~0.66 log loss on APS at an identical partition (ledger 768, L15).
    """
    gy = np.bincount(idx, weights=y, minlength=nb)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    if R84_GMRF["binary"]:
        return _solve_bin_logistic_gmrf(off, y, nb, idx, alpha, iters)
    lo = np.full(nb, -80.0)
    hi = np.full(nb, 80.0)
    v = np.zeros(nb)
    for _ in range(iters):
        p = sig(off + v[idx])
        gp = np.bincount(idx, weights=p, minlength=nb)
        g = gy - gp - alpha * v
        pos = g > 0
        lo = np.where(pos, v, lo)
        hi = np.where(pos, hi, v)
        v = 0.5 * (lo + hi)
    p = sig(off + v[idx])
    H = np.bincount(idx, weights=p * (1.0 - p), minlength=nb) + alpha
    H[cnt == 0] = 0.0
    z = np.where(cnt > 0, v, 0.0)
    return z, H


def solve_bin_regression(off: np.ndarray, y: np.ndarray, nb: int, idx: np.ndarray,
                         alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted bin mean of the residual, H = count (+ ridge)."""
    cnt = np.bincount(idx, minlength=nb).astype(float)
    s = np.bincount(idx, weights=y - off, minlength=nb)
    H = cnt + alpha
    z = np.where(H > 0, s / np.maximum(H, 1e-12), 0.0)
    H = np.where(cnt > 0, H, 0.0)
    return z, H


def solve_bin_multiclass(off: np.ndarray, Y: np.ndarray, nb: int, idx: np.ndarray, K: int,
                         alpha: float = ALPHA_DEFAULT, iters: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bin softmax coefficients by damped Newton on the full multinomial likelihood.

    off : (n, K) offsets (eta minus this feature's contribution), Y : (n, K) one-hot.
    Returns C (nb, K) centred coefficients and H (nb,) shared Fisher mass.
    """
    C = np.zeros((nb, K))
    for _ in range(iters):
        P = softmax(off + C[idx])
        g = np.zeros((nb, K))
        for k in range(K):
            g[:, k] = np.bincount(idx, weights=Y[:, k] - P[:, k], minlength=nb)
        g -= alpha * C
        Hd = np.bincount(idx, weights=(P * (1.0 - P)).sum(axis=1), minlength=nb) + alpha
        step = g / np.maximum(Hd, 1e-9)[:, None]
        C += 0.5 * step                       # damping: the saturated-softmax failure is silent otherwise (r50)
        if np.max(np.abs(step)) < 1e-8:
            break
    C -= C.mean(axis=1, keepdims=True)        # identifiability: sum_k c_bk = 0
    P = softmax(off + C[idx])
    H = np.bincount(idx, weights=(P * (1.0 - P)).sum(axis=1), minlength=nb) + alpha
    cnt = np.bincount(idx, minlength=nb)
    H[cnt == 0] = 0.0
    return C, H


# ---- R84 Canary 3A/3B solvers (classification; flag-gated, 0 B) --------------
R84_GMRF = dict(binary=False)          # Canary 3A
R84_FISHER = dict(multiclass=False)    # Canary 3B
_R84_TAU_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _thomas(d: np.ndarray, e: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve tridiag(e, d, e) x = rhs (e = off-diagonals, length m-1)."""
    m = len(d)
    if m == 0:
        return np.zeros(0)
    cp = np.zeros(m)
    dp = np.zeros(m)
    cp[0] = e[0] / d[0] if m > 1 else 0.0
    dp[0] = rhs[0] / d[0]
    for i in range(1, m):
        den = d[i] - e[i - 1] * cp[i - 1]
        cp[i] = (e[i] / den) if i < m - 1 else 0.0
        dp[i] = (rhs[i] - e[i - 1] * dp[i - 1]) / den
    x = np.zeros(m)
    x[-1] = dp[-1]
    for i in range(m - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _thomas_logdet(d: np.ndarray, e: np.ndarray) -> float:
    """log det of tridiag(e, d, e) via the Thomas pivots."""
    m = len(d)
    if m == 0:
        return 0.0
    piv = np.zeros(m)
    piv[0] = d[0]
    for i in range(1, m):
        piv[i] = d[i] - (e[i - 1] ** 2) / piv[i - 1]
    return float(np.log(np.maximum(piv, 1e-300)).sum())


def _gmrf_tau(G: np.ndarray, H: np.ndarray, alpha: float) -> float:
    """3A: ML-II tau from {0,.25,.5,1,2,4} via -1/2 ln det(H+aI+tau L) + 1/2 v'Av, v=A^-1 G.
    H = pure Fisher diag (no alpha). L = chain Laplacian on the ordered bins."""
    m = len(H)
    taus = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    if m <= 1:
        return 0.0
    deg = np.full(m, 2.0)
    deg[0] = 1.0
    deg[-1] = 1.0
    e = np.full(m - 1, -1.0)
    best_tau, best = 0.0, np.inf
    for tau in taus:
        d = H + alpha + tau * deg
        ee = -tau * e
        sign, logdet = 1.0, 0.0
        cprev = 0.0
        for i in range(m):
            c = d[i] - (ee[i - 1] * cprev if i else 0.0)
            if c <= 0:
                sign = 0.0
                break
            logdet += np.log(c)
            cprev = (ee[i - 1] / c) if i else 0.0
        if sign == 0.0:
            continue
        v = _thomas(d, ee, G)
        # 1/2 v'Av with A = tridiag(d, ee)
        Av = d * v
        Av[:-1] += ee * v[1:]
        Av[1:] += ee * v[:-1]
        score = -0.5 * logdet + 0.5 * float(v @ Av)
        if score < best - 1e-12:
            best, best_tau = score, tau
    return best_tau


def _solve_bin_logistic_gmrf(off: np.ndarray, y: np.ndarray, nb: int, idx: np.ndarray,
                             alpha: float, iters: int):
    """3A: standard bisection MLE z (bit-exact body), then ONE Thomas GMRF-step when
    tau>0 is selected: (diag(H)+alpha I + tau L) v = (diag(H)+alpha I) z + g(z),
    on non-NaN bins 0..nb-2 (NaN bin kept = its own bisection z). tau=0 -> the plain
    bisection result verbatim (bit-exact control)."""
    gy = np.bincount(idx, weights=y, minlength=nb)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    lo = np.full(nb, -80.0)
    hi = np.full(nb, 80.0)
    v = np.zeros(nb)
    for _ in range(iters):
        p = sig(off + v[idx])
        gp = np.bincount(idx, weights=p, minlength=nb)
        g = gy - gp - alpha * v
        pos = g > 0
        lo = np.where(pos, v, lo)
        hi = np.where(pos, hi, v)
        v = 0.5 * (lo + hi)
    p = sig(off + v[idx])
    Hp = np.bincount(idx, weights=p * (1.0 - p), minlength=nb)     # pure Fisher
    gp = np.bincount(idx, weights=p, minlength=nb)
    gz = gy - gp - alpha * v                                        # bisection residual
    H = Hp + alpha
    H[cnt == 0] = 0.0
    z = np.where(cnt > 0, v, 0.0)
    m_ord = max(nb - 1, 1)
    if m_ord <= 1 or not np.any(cnt[:m_ord] > 0):
        return z, H
    Ho = Hp[:m_ord]
    Go = (Ho + alpha) * z[:m_ord] + gz[:m_ord]
    tau = _gmrf_tau(Go, Ho, alpha)
    if tau == 0.0:
        return z, H                       # bit-exact control path
    m = m_ord
    deg = np.full(m, 2.0)
    deg[0] = 1.0
    deg[-1] = 1.0
    d = Ho + alpha + tau * deg
    e = np.full(m - 1, -tau)
    vo = _thomas(d, e, Go)
    z[:m_ord] = vo
    z[cnt == 0] = 0.0
    return z, H


def solve_bin_regression(off: np.ndarray, y: np.ndarray, nb: int, idx: np.ndarray,
                         alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted bin mean of the residual, H = count (+ ridge)."""
    cnt = np.bincount(idx, minlength=nb).astype(float)
    s = np.bincount(idx, weights=y - off, minlength=nb)
    H = cnt + alpha
    z = np.where(H > 0, s / np.maximum(H, 1e-12), 0.0)
    H = np.where(cnt > 0, H, 0.0)
    return z, H


def solve_bin_multiclass(off: np.ndarray, Y: np.ndarray, nb: int, idx: np.ndarray, K: int,
                         alpha: float = ALPHA_DEFAULT, iters: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bin softmax coefficients by damped Newton on the full multinomial likelihood.

    off : (n, K) offsets (eta minus this feature's contribution), Y : (n, K) one-hot.
    Returns C (nb, K) centred coefficients and H (nb,) shared Fisher mass.
    """
    C = np.zeros((nb, K))
    for _ in range(iters):
        P = softmax(off + C[idx])
        g = np.zeros((nb, K))
        for k in range(K):
            g[:, k] = np.bincount(idx, weights=Y[:, k] - P[:, k], minlength=nb)
        g -= alpha * C
        Hd = np.bincount(idx, weights=(P * (1.0 - P)).sum(axis=1), minlength=nb) + alpha
        step = g / np.maximum(Hd, 1e-9)[:, None]
        C += 0.5 * step                       # damping: the saturated-softmax failure is silent otherwise (r50)
        if np.max(np.abs(step)) < 1e-8:
            break
    C -= C.mean(axis=1, keepdims=True)        # identifiability: sum_k c_bk = 0
    P = softmax(off + C[idx])
    H = np.bincount(idx, weights=(P * (1.0 - P)).sum(axis=1), minlength=nb) + alpha
    cnt = np.bincount(idx, minlength=nb)
    H[cnt == 0] = 0.0
    return C, H


# ---- R84 Canary 3A/3B solvers (classification; flag-gated, 0 B) --------------
R84_GMRF = dict(binary=False)          # Canary 3A
R84_FISHER = dict(multiclass=False)    # Canary 3B
_R84_TAU_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _thomas(d: np.ndarray, e: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve tridiag(e, d, e) x = rhs (e = off-diagonals, length m-1)."""
    m = len(d)
    if m == 0:
        return np.zeros(0)
    cp = np.zeros(m)
    dp = np.zeros(m)
    cp[0] = e[0] / d[0] if m > 1 else 0.0
    dp[0] = rhs[0] / d[0]
    for i in range(1, m):
        den = d[i] - e[i - 1] * cp[i - 1]
        cp[i] = (e[i] / den) if i < m - 1 else 0.0
        dp[i] = (rhs[i] - e[i - 1] * dp[i - 1]) / den
    x = np.zeros(m)
    x[-1] = dp[-1]
    for i in range(m - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _thomas_logdet(d: np.ndarray, e: np.ndarray) -> float:
    """log det of tridiag(e, d, e) via the Thomas pivots."""
    m = len(d)
    if m == 0:
        return 0.0
    piv = np.zeros(m)
    piv[0] = d[0]
    for i in range(1, m):
        piv[i] = d[i] - (e[i - 1] ** 2) / piv[i - 1]
    return float(np.log(np.maximum(piv, 1e-300)).sum())


def _gmrf_tau(G: np.ndarray, H: np.ndarray, alpha: float) -> float:
    """ML-II tau for the chain GMRF: score(tau)=0.5*[logdet(D+tauL) - G'(D+tauL)^-1 G]."""
    m = len(G)
    deg = np.full(m, 2.0)
    if m:
        deg[0] = 1.0
        if m > 1:
            deg[-1] = 1.0
    best_t, best_s = 0.0, -np.inf
    for tau in _R84_TAU_GRID:
        d = H + alpha + tau * deg
        e = np.full(max(m - 1, 0), -tau)
        ld = _thomas_logdet(d, e)
        v = _thomas(d, e, G)
        sc = 0.5 * (ld - float(G @ v))
        if sc > best_s:
            best_t, best_s = float(tau), sc
    return best_t


def _solve_bin_logistic_gmrf(off: np.ndarray, y: np.ndarray, nb: int, idx: np.ndarray,
                             alpha: float, iters: int):
    """3A: coupled chain-GMRF penalised solve on non-NaN bins 0..nb-2 (Thomas),
    tau_j by ML-II; tau=0 -> the exact bisection (bit-exact base).  NaN bin solved
    independently."""
    cnt = np.bincount(idx, minlength=nb).astype(float)
    m_ord = max(nb - 1, 1)
    p0 = sig(off)
    H0 = np.bincount(idx, weights=p0 * (1.0 - p0), minlength=nb)
    G0 = (np.bincount(idx, weights=y, minlength=nb)
          - np.bincount(idx, weights=p0, minlength=nb))
    tau = _gmrf_tau(G0[:m_ord], H0[:m_ord], alpha) if m_ord > 1 else 0.0
    if tau == 0.0:
        # bit-exact fallback: the original bisection body
        gy = np.bincount(idx, weights=y, minlength=nb)
        lo = np.full(nb, -80.0)
        hi = np.full(nb, 80.0)
        v = np.zeros(nb)
        for _ in range(iters):
            p = sig(off + v[idx])
            gp = np.bincount(idx, weights=p, minlength=nb)
            g = gy - gp - alpha * v
            pos = g > 0
            lo = np.where(pos, v, lo)
            hi = np.where(pos, hi, v)
            v = 0.5 * (lo + hi)
        p = sig(off + v[idx])
        H = np.bincount(idx, weights=p * (1.0 - p), minlength=nb) + alpha
        H[cnt == 0] = 0.0
        z = np.where(cnt > 0, v, 0.0)
        return z, H
    deg = np.full(m_ord, 2.0)
    if m_ord:
        deg[0] = 1.0
        if m_ord > 1:
            deg[-1] = 1.0
    e = np.full(max(m_ord - 1, 0), -tau)
    gy = np.bincount(idx, weights=y, minlength=nb)
    v = np.zeros(nb)
    for _ in range(iters):
        p = sig(off + v[idx])
        Hc = np.bincount(idx, weights=p * (1.0 - p), minlength=nb)
        g = gy - np.bincount(idx, weights=p, minlength=nb) - alpha * v
        d = Hc[:m_ord] + alpha + tau * deg
        v_new = v.copy()
        v_new[:m_ord] = _thomas(d, e, g[:m_ord])
        # NaN bin (nb-1) by independent bisection step
        if nb > m_ord:
            v_new[m_ord] = v[m_ord] + 0.5 * g[m_ord] / max(Hc[m_ord] + alpha, 1e-9)
        dv = v_new - v
        v = v + 0.5 * dv
        if np.max(np.abs(dv)) < 1e-8:
            break
    p = sig(off + v[idx])
    H = np.bincount(idx, weights=p * (1.0 - p), minlength=nb) + alpha
    H[cnt == 0] = 0.0
    z = np.where(cnt > 0, v, 0.0)
    return z, H


def solve_bin_multiclass_fisher(off: np.ndarray, Y: np.ndarray, nb: int, idx: np.ndarray, K: int,
                                alpha: float = ALPHA_DEFAULT, iters: int = 40):
    """3B: per-bin KxK Fisher Newton (damped) softmax coefficients.

    шаг в бине z_b = (W_b + 0.5I)^-1 g_b (центрированный); W_b = sum(diag(p)-pp').
    Returns (C, Ws, Gs, Hs): coefficients, per-bin W (nb,K,K), the bin gradient
    proxies g_b = (W_b + alpha*I) C_b (so a 1-bin segment reproduces z exactly),
    and the scalar Fisher mass Hs_b = tr(W_b) (= sum p(1-p), the base convention).
    """
    C = np.zeros((nb, K))
    I = np.eye(K)
    for _ in range(iters):
        P = softmax(off + C[idx])
        g = np.zeros((nb, K))
        for k in range(K):
            g[:, k] = np.bincount(idx, weights=Y[:, k] - P[:, k], minlength=nb)
        g -= alpha * C
        Ws = np.zeros((nb, K, K))
        for k in range(K):
            for l in range(K):
                wl = P[:, k] * ((1.0 if k == l else 0.0) - P[:, l])
                Ws[:, k, l] = np.bincount(idx, weights=wl, minlength=nb)
        step = np.zeros_like(C)
        for b in range(nb):
            step[b] = np.linalg.solve(Ws[b] + alpha * I, g[b])
        step -= step.mean(axis=1, keepdims=True)
        C += 0.5 * step
        if np.max(np.abs(step)) < 1e-8:
            break
    C -= C.mean(axis=1, keepdims=True)
    P = softmax(off + C[idx])
    Ws = np.zeros((nb, K, K))
    for k in range(K):
        for l in range(K):
            wl = P[:, k] * ((1.0 if k == l else 0.0) - P[:, l])
            Ws[:, k, l] = np.bincount(idx, weights=wl, minlength=nb)
    cnt = np.bincount(idx, minlength=nb)
    Hs = np.zeros(nb)
    Gs = np.zeros((nb, K))
    for b in range(nb):
        if cnt[b] > 0:
            Hs[b] = float(np.trace(Ws[b]))
            Gs[b] = Ws[b] @ C[b] + alpha * C[b]
        else:
            Ws[b] = 0.0
            C[b] = 0.0
    Hs[cnt == 0] = 0.0
    return C, Ws, Gs, Hs


def potts_dp_fisher(z: np.ndarray, Ws: np.ndarray, Gs: np.ndarray,
                    emp: np.ndarray, kap: float) -> np.ndarray:
    """3B: exact l0 fusion with the KxK Fisher segment cost.

    cost([a,c)) = sum_b (z_b - mu)' W_b (z_b - mu),  mu = (sum W + 0.5 I)^-1 sum g.
    """
    z = np.asarray(z, float)
    m, K = z.shape
    if m <= 1:
        return np.zeros(m, int)
    A = np.zeros((m + 1, K, K))
    B = np.zeros((m + 1, K))
    S = np.zeros((m + 1, K))
    Q = np.zeros(m + 1)
    for i in range(m):
        A[i + 1] = A[i] + Ws[i]
        B[i + 1] = B[i] + Gs[i]
        S[i + 1] = S[i] + z[i]
        Q[i + 1] = Q[i] + float(z[i] @ Ws[i] @ z[i])
    I = np.eye(K)
    from numpy.lib.stride_tricks import sliding_window_view  # noqa: F401

    def seg_cost(a, c):
        W = A[c] - A[a]
        g = B[c] - B[a]
        q = Q[c] - Q[a]
        zsum = S[c] - S[a]
        mu = np.linalg.solve(W + 0.5 * I, g)
        Wz = g - 0.5 * zsum            # sum W_b z_b (g_b = (W_b+0.5I) z_b)
        return max(q - 2.0 * float(mu @ Wz) + float(mu @ W @ mu), 0.0)

    dp = np.zeros(m + 1)
    bt = np.zeros(m + 1, int)
    for c in range(1, m + 1):
        best = np.inf
        best_a = 0
        for a in range(c):
            v = dp[a] + seg_cost(a, c) + (kap if a > 0 else 0.0)
            # s3 charge per cut, same convention as potts_dp_exactq
            if v < best:
                best = v
                best_a = a
        dp[c] = best
        bt[c] = best_a
    lab = np.zeros(m, int)
    cur = m
    k = 0
    while cur > 0:
        a = bt[cur]
        lab[a:cur] = k
        cur = a
        k += 1
    return lab


def _segment_levels_fisher(lab, Ws, Gs, emp):
    """3B: segment level mu = (sum W + 0.5I)^-1 sum g per segment (identity for 1-bin)."""
    nb, K = Gs.shape
    V = np.zeros((nb, K))
    I = np.eye(K)
    for s in np.unique(lab):
        sel = lab == s
        W = Ws[sel].sum(axis=0)
        g = Gs[sel].sum(axis=0)
        mu = np.linalg.solve(W + 0.5 * I, g)
        V[sel] = mu
    V[emp] = 0.0
    return V


# --------------------------------------------------------------------------------------------------------------
# 2.  The fusion DP (l0 Potts), the exact-q DP, the skew cost and the exact-deviance cost
# --------------------------------------------------------------------------------------------------------------
# R84 canary (deviance currency family).  Defaults keep the R79 engine bit-exact.
#   scale     : multiplier of the within-segment dispersion (kind=="reg": 1/sigma2_oof).
#   mbic      : lambda of the Zhang-Siegmund mass barrier, lambda*0.5*ln(sum H / H_seg)
#               per segment (H1: lambda = 1/sigma2_oof so the barrier scales WITH the
#               cheapened SSE; 0 disables).
#   hmin      : hard segment-mass threshold (H2): segments with H_seg < max(25, 5%*sum H)
#               are forbidden (+inf), except the single trailing NaN bin and the
#               whole-feature one-segment case.  0 disables.
#   pair_ctx  : >0 while inside product_potts (2-D pair grids).
#   pair_exempt: H3 scope -- when True, `scale` does NOT apply inside pair_ctx (1-D
#               mains/g4 rays keep the currency, p1 2-D grids keep the R83 base price).
R84_COST = dict(scale=1.0, mbic=0.0, hmin=0.0, pair_ctx=0, pair_exempt=False)
# R84 Canary 3 (classification; 0 B).  Defaults keep the R79 engine bit-exact.
#   R84_GMRF["binary"]    : 3A -- GMRF pre-smoothing step before potts_dp on binary
#                           mains (tridiagonal Thomas over non-NaN bins; tau by ML-II,
#                           tau=0 falls back to the exact bisection = bit-exact base).
#   R84_FISHER["multiclass"]: 3B -- KxK Fisher Newton solve + KxK Fisher potts cost
#                           (potts_dp_fisher) for multiclass mains.


def potts_dp(C: np.ndarray, H: np.ndarray, kappa: float, exact: bool = False,
             Gs: np.ndarray = None, Ns: np.ndarray = None, alpha: float = 0.5) -> np.ndarray:
    """Exact l0 fusion over ordered bins: minimise sum_segments [ weighted within-segment dispersion + kappa ].

    C is (m,) for regression/binary levels or (m, K) for multinomial coefficient vectors; the same routine handles
    both, which is why the multiclass card needs no separate DP. kappa = 0.5*log(sum of H over non-empty bins) is the
    record's information-unit charge S3.

    exact (R107, frozen 2026-11-16): for 1-D binary tables replace the quadratic
    Taylor cost sum H_b (z_b - mu)^2 with the EXACT binomial deviance of the
    segment (Haldane alpha=0.5 pseudo-counts per bin) computed from the per-bin
    sufficient statistics G_b = sum w*y, N_b = sum w (pass Gs/Ns).  O(1) per
    segment via prefix sums; when exact=False the code path is byte-identical
    to the quadratic solver (bit-identity contract).
    """
    C = np.asarray(C, float)
    one_d = C.ndim == 1
    if one_d:
        C = C[:, None]
    m, K = C.shape
    if m <= 1:
        return np.zeros(m, int)
    Hc = np.asarray(H, float)
    P0 = np.concatenate([[0.0], np.cumsum(Hc)])
    P1 = np.concatenate([np.zeros((1, K)), np.cumsum(Hc[:, None] * C, axis=0)], axis=0)
    P2 = np.concatenate([np.zeros((1, K)), np.cumsum(Hc[:, None] * C * C, axis=0)], axis=0)
    use_exact = bool(exact) and one_d and Gs is not None and Ns is not None
    if use_exact:
        # R107: exact binomial deviance with per-bin Haldane pseudo-counts.
        # cost([s,b)) = 2*[sum_i c_i - (Kt ln pi + Nt ln(1-pi))], pi = Kt/(Kt+Nt)
        Gv = np.asarray(Gs, float)
        Nv = np.asarray(Ns, float)
        a = float(alpha)
        ki = Gv + a
        ni = Nv - Gv + a
        ti = Nv + 2.0 * a
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = ki * np.log(ki / ti) + ni * np.log(ni / ti)
        ci = np.where(np.isfinite(ci), ci, 0.0)
        PG = np.concatenate([[0.0], np.cumsum(Gv)])
        PN = np.concatenate([[0.0], np.cumsum(Nv)])
        PC = np.concatenate([[0.0], np.cumsum(ci)])
        PL = np.arange(m + 1, dtype=float)          # segment length per prefix
    dp = np.full(m + 1, np.inf)
    back = np.zeros(m + 1, int)
    dp[0] = 0.0
    for b in range(1, m + 1):
        s = np.arange(b)
        h = P0[b] - P0[s]
        if use_exact:
            G = PG[b] - PG[s]
            N = PN[b] - PN[s]
            L = PL[b] - PL[s]
            Kt = G + a * L
            Nt = N - G + a * L
            tot_mass = Kt + Nt
            with np.errstate(divide="ignore", invalid="ignore"):
                pi = Kt / np.maximum(tot_mass, 1e-300)
                ll = Kt * np.log(np.maximum(pi, 1e-300)) + Nt * np.log(np.maximum(1.0 - pi, 1e-300))
                cost = 2.0 * ((PC[b] - PC[s]) - ll)
            cost = np.where(tot_mass > 0, cost, 0.0)
            cost = np.where(np.isfinite(cost), np.maximum(cost, 0.0), np.inf)
        else:
            d1 = P1[b] - P1[s]
            d2 = P2[b] - P2[s]
            with np.errstate(invalid="ignore", divide="ignore"):
                cost = d2.sum(axis=1) - np.where(h > 0, (d1 * d1).sum(axis=1) / np.maximum(h, 1e-12), 0.0)
            cost = np.where(np.isfinite(cost), np.maximum(cost, 0.0), np.inf)
        scale = 1.0 if (R84_COST["pair_exempt"] and R84_COST["pair_ctx"]) else R84_COST["scale"]
        if scale != 1.0:
            cost = cost * scale
        if R84_COST["mbic"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                pen = R84_COST["mbic"] * 0.5 * np.log(
                    np.maximum(P0[m], 1e-300) / np.maximum(h, 1e-300))
            cost = cost + np.where(h > 0, pen, np.inf)
        if R84_COST["hmin"]:
            hmin = max(25.0, 0.05 * P0[m])
            tiny = h < hmin
            tiny[(b == m) & (s == m - 1)] = False   # single trailing (NaN) bin
            tiny[(b == m) & (s == 0)] = False       # whole feature = one segment
            cost = np.where(tiny, np.inf, cost)
        tot = dp[s] + cost + kappa
        j = int(np.argmin(tot))
        dp[b] = tot[j]
        back[b] = j
    lab = np.zeros(m, int)
    b, k = m, 0
    while b > 0:
        s = back[b]
        lab[s:b] = k
        b = s
        k += 1
    return lab


def potts_skew_dp(z: np.ndarray, H: np.ndarray, S: np.ndarray, kappa: float) -> np.ndarray:
    """M3: the same DP with a third-order (skewness) correction inside the quadratic frame.

    Per segment the level is the one-step Halley correction v* = zbar - A2/(2H), and the charge is
        sum_b H_b (v* - z_b)^2 + (1/3) sum_b S_b (v* - z_b)^3.
    Kept because it is the natural control for idea #6: M3 ESTIMATES the third-order term from bin moments,
    idea #6 optimises the exact binomial objective instead.
    """
    m = len(z)
    if m <= 1:
        return np.zeros(m, int)
    P0 = np.concatenate([[0.0], np.cumsum(H)])
    P1 = np.concatenate([[0.0], np.cumsum(H * z)])
    P2 = np.concatenate([[0.0], np.cumsum(H * z * z)])
    Q0 = np.concatenate([[0.0], np.cumsum(S)])
    Q1 = np.concatenate([[0.0], np.cumsum(S * z)])
    Q2 = np.concatenate([[0.0], np.cumsum(S * z * z)])
    Q3 = np.concatenate([[0.0], np.cumsum(S * z ** 3)])
    dp = np.full(m + 1, np.inf)
    back = np.zeros(m + 1, int)
    dp[0] = 0.0
    for b in range(1, m + 1):
        s = np.arange(b)
        h = P0[b] - P0[s]
        s1 = P1[b] - P1[s]
        s2 = P2[b] - P2[s]
        q0 = Q0[b] - Q0[s]
        q1 = Q1[b] - Q1[s]
        q2 = Q2[b] - Q2[s]
        q3 = Q3[b] - Q3[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            zb = np.where(h > 0, s1 / np.maximum(h, 1e-12), 0.0)
            A2 = q2 - 2 * zb * q1 + zb * zb * q0
            vstar = zb - np.where(h > 0, A2 / (2 * np.maximum(h, 1e-12)), 0.0)
            cost = (s2 - 2 * vstar * s1 + vstar * vstar * h) + (q3 - 3 * vstar * q2 + 3 * vstar ** 2 * q1
                                                               - vstar ** 3 * q0) / 3.0
        cost = np.where(np.isfinite(cost), cost, np.inf)
        tot = dp[s] + cost + kappa
        j = int(np.argmin(tot))
        dp[b] = tot[j]
        back[b] = j
    lab = np.zeros(m, int)
    b, k = m, 0
    while b > 0:
        s = back[b]
        lab[s:b] = k
        b = s
        k += 1
    return lab


def dp_exact_q(cost: np.ndarray, q: int) -> Optional[np.ndarray]:
    """Minimum-cost partition with EXACTLY q segments. Raises nothing; returns None if infeasible.

    This is how 'matched runs' must be enforced. Bisecting a penalty cannot hit an integer run count (round 70),
    and an infeasible target must never fall back silently to the unconstrained solve (ledger 769, L17).
    """
    m = cost.shape[0]
    if q < 1 or q > m:
        return None
    dp = np.full((q + 1, m + 1), np.inf)
    back = np.zeros((q + 1, m + 1), int)
    dp[0, 0] = 0.0
    for kk in range(1, q + 1):
        for b in range(1, m + 1):
            s = np.arange(b)
            tot = dp[kk - 1, s] + cost[s, b]
            j = int(np.argmin(tot))
            if np.isfinite(tot[j]):
                dp[kk, b] = tot[j]
                back[kk, b] = j
    if not np.isfinite(dp[q, m]):
        return None
    lab = np.zeros(m, int)
    b = m
    for kk in range(q, 0, -1):
        s = back[kk, b]
        lab[s:b] = kk
        b = s
    return lab


def dp_from_cost(cost: np.ndarray, kappa: float) -> np.ndarray:
    """Unconstrained-segment-count DP on an arbitrary (m, m+1) cost matrix: minimise sum C(s,t) + kappa per segment."""
    m = cost.shape[0]
    dp = np.full(m + 1, np.inf)
    back = np.zeros(m + 1, int)
    dp[0] = 0.0
    for b in range(1, m + 1):
        tot = dp[:b] + cost[:b, b] + kappa
        j = int(np.argmin(tot))
        dp[b] = tot[j]
        back[b] = j
    lab = np.zeros(m, int)
    b, k = m, 0
    while b > 0:
        s = back[b]
        lab[s:b] = k
        b = s
        k += 1
    return lab


def cost_quadratic(z: np.ndarray, H: np.ndarray) -> np.ndarray:
    """The frozen merge price, as a (m, m+1) matrix ready for dp_exact_q."""
    m = len(z)
    P0 = np.concatenate([[0.0], np.cumsum(H * z * z)])
    P1 = np.concatenate([[0.0], np.cumsum(H * z)])
    P2 = np.concatenate([[0.0], np.cumsum(H)])
    c = np.full((m, m + 1), np.inf)
    for s in range(m):
        t = np.arange(s + 1, m + 1)
        h = P2[t] - P2[s]
        s1 = P1[t] - P1[s]
        s2 = P0[t] - P0[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            zb = np.where(h > 0, s1 / np.maximum(h, 1e-12), 0.0)
        c[s, s + 1:] = np.maximum(s2 - 2 * zb * s1 + zb * zb * h, 0.0)
    return c


def cost_deviance(rows_by_bin: List[np.ndarray], off: np.ndarray, y: np.ndarray,
                  z: np.ndarray, alpha: float = ALPHA_DEFAULT, grid: int = 81) -> np.ndarray:
    """Idea #6: the exact conditional deviance of a segment of rows, minus its per-bin saturated minimum.

        C(s,t) = min_v [ 2*sum_{i in [s,t)} ( log(1+e^{o_i+v}) - y_i (o_i+v) ) + alpha v^2 ] - sum_b (binwise min)

    Evaluated on ROWS, because the offsets o_i vary inside a bin. Evaluated on a grid, so the segment costs are
    certified only to the grid — that limitation is reported, never hidden.
    """
    m = len(z)
    V = np.linspace(float(np.min(z)) - 3.0, float(np.max(z)) + 3.0, grid)
    LG = np.logaddexp(0.0, off[:, None] + V[None, :])
    PA = [np.zeros(grid)]
    Py = [0.0]
    Pyo = [0.0]
    for b in range(m):
        r = rows_by_bin[b]
        if len(r):
            PA.append(PA[-1] + LG[r].sum(axis=0))
            Py.append(Py[-1] + float(y[r].sum()))
            Pyo.append(Pyo[-1] + float((y[r] * off[r]).sum()))
        else:
            PA.append(PA[-1])
            Py.append(Py[-1])
            Pyo.append(Pyo[-1])
    PA = np.asarray(PA)
    Py = np.asarray(Py)
    Pyo = np.asarray(Pyo)
    R = 0.5 * alpha * V ** 2
    c = np.full((m, m + 1), np.inf)
    for s in range(m):
        t = np.arange(s + 1, m + 1)
        seg = PA[t] - PA[s] - V[None, :] * (Py[t] - Py[s])[:, None] - (Pyo[t] - Pyo[s])[:, None] + R[None, :]
        c[s, s + 1:] = np.maximum(seg.min(axis=1), 0.0)
    binmin = np.array([((PA[b + 1] - PA[b]) - V * Py[b + 1] - Pyo[b + 1] + R).min() for b in range(m)])
    Pb = np.concatenate([[0.0], np.cumsum(binmin)])
    return c - Pb[None, :] + Pb[:m, None]


def rows_per_bin(idx: np.ndarray, nb: int) -> List[np.ndarray]:
    order = np.argsort(idx, kind="stable")
    ib = idx[order]
    lo = np.searchsorted(ib, np.arange(nb), "left")
    hi = np.searchsorted(ib, np.arange(nb), "right")
    return [order[lo[b]:hi[b]] for b in range(nb)]


# --------------------------------------------------------------------------------------------------------------
# 3.  The card object
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Card:
    """A fitted glass-box object: one level table per feature, an intercept, a temperature.

    kind : 'binary' (levels (nb,), predict returns the logit),
           'regression' (levels (nb,), predict returns the value),
           'multiclass' (levels (nb, K), predict returns K logits).
    V[j] is indexed by the bin of feature j: bin b means cuts[b-1] < x <= cuts[b] (right-closed), NaN is the last bin.
    """
    kind: str
    cuts: List[np.ndarray]
    V: List[np.ndarray]
    b0: np.ndarray
    T: np.ndarray
    feat_names: List[str]
    meta: dict = field(default_factory=dict)

    # ---- routing -------------------------------------------------------------------------------------------
    def _bin(self, X: np.ndarray, j: int) -> np.ndarray:
        return bin_idx(X[:, j], self.cuts[j])

    def score(self, X: np.ndarray) -> np.ndarray:
        """Raw additive score, before temperature. Binary -> logit, regression -> value, multiclass -> logits."""
        X = np.asarray(X, float)
        n = len(X)
        if self.kind == "multiclass":
            eta = np.tile(np.asarray(self.b0, float), (n, 1))
        else:
            eta = np.full(n, float(self.b0))
        for j in range(len(self.cuts)):
            b = self._bin(X, j)
            eta = eta + self.V[j][b]
        ramp = self.meta.get("ramps")
        if ramp:
            abs_w = self.meta.get("ramp_abs")
            for j, rj in enumerate(ramp):
                if rj is None or len(rj) == 0:
                    continue
                Ej = self.cuts[j]
                fin = np.isfinite(X[:, j])
                w = abs_w[j] if abs_w is not None else 1.0
                for c in range(min(len(Ej), len(rj))):
                    if rj[c, 1] == 0.0:
                        continue
                    d = np.clip((X[:, j] - Ej[c]) / w, -1.0, 1.0)
                    eta = eta + np.where(fin, rj[c, 0] * w * d, 0.0)
        return eta

    def calibrated(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        T = float(np.asarray(self.T).ravel()[0])
        if self.kind == "regression":
            return s
        if self.kind == "binary":
            return s / T
        return s / T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = self.calibrated(X)
        if self.kind == "regression":
            return s
        if self.kind == "binary":
            return sig(s)
        return softmax(s)

    # ---- diagnostics ---------------------------------------------------------------------------------------
    def runs(self) -> int:
        """Active thresholds + features (the identity that certifies the printer: 103 + 170 = 273)."""
        n = 0
        for j in range(len(self.cuts)):
            V = np.asarray(self.V[j], float)
            live = int(np.sum(np.any(np.diff(V, axis=0) != 0, axis=-1))) if V.ndim > 1 else int(np.sum(np.diff(V) != 0))
            n += live + 1
        return n


# --------------------------------------------------------------------------------------------------------------
# 4.  Card fitting: alternating (exact bin solve) <-> (fusion DP) <-> (level refit), per pass
# --------------------------------------------------------------------------------------------------------------
@dataclass
class FitCfg:
    m: int = 32                        # quantile cuts per feature
    passes: int = 12
    max_passes: int = 12               # if conv_tol is set, allowed to run to max_passes
    conv_tol: float = 0.0              # 0 disables the convergence check; else stop when |dLL| < conv_tol
    alpha: float = ALPHA_DEFAULT       # per-bin ridge
    kappa_mult: float = 1.0            # 0.0 == unfused arm
    price: str = "quadratic"           # 'quadratic' | 'deviance' (idea #6) | 'skew' (M3)
    matched_runs: bool = False         # enforce the baseline's own per-feature run counts via dp_exact_q
    qfix: Optional[List[int]] = None
    sigma_noise: float = 0.0           # idea #7: mean-one multiplicative noise in the objective
    order: str = "cyclic"              # 'cyclic' | 'fisher'
    ramp_w: float = 0.0                # idea #5: bounded ramp width as a fraction of the feature range
    ramp_cap: float = 0.25             # idea #5: cap on |ramp amplitude| in logit units
    level_shrink: float = 0.0         # empirical-Bayes shrinkage of segment levels, in units of Fisher mass
    n_classes: int = 0                # multiclass only: taken from the WHOLE task, not from the training fold
    nan_dock: bool = False             # M8: dock the NaN bin when the Potts criterion prefers it
    verbose: bool = False


def _fuse(mode: str, z: np.ndarray, H: np.ndarray, S: Optional[np.ndarray], kappa: float,
          C2: Optional[np.ndarray] = None) -> np.ndarray:
    if mode == "skew" and S is not None and C2 is None:
        return potts_skew_dp(z, H, S, kappa)
    if C2 is not None:
        return potts_dp(C2, H, kappa)
    return potts_dp(z, H, kappa)


def _segment_levels(lab: np.ndarray, z: np.ndarray, H: np.ndarray, kind: str,
                    C2: Optional[np.ndarray] = None, shrink: float = 0.0) -> np.ndarray:
    """Weighted mean of the exact per-bin estimates inside each segment, then re-centred on the non-empty bins."""
    m = len(lab)
    if kind == "multiclass":
        V = np.zeros((m, C2.shape[1]))
        for k in np.unique(lab):
            sel = lab == k
            w = H[sel].sum()
            V[sel] = (C2[sel] * H[sel][:, None]).sum(axis=0) / w if w > 0 else 0.0
    else:
        V = np.zeros(m)
        for k in np.unique(lab):
            sel = lab == k
            w = H[sel].sum()
            V[sel] = (z[sel] * H[sel]).sum() / w if w > 0 else 0.0
    if shrink > 0:
        # Empirical-Bayes level shrinkage. A segment supported by H of Fisher mass keeps H/(H+shrink_eff) of its
        # level. shrink_eff carries the segment's free-parameter count: a K-class segment has (K-1) of them, so a
        # 5-class card on 898 rows needs the factor scaled by K-1 or it memorises the fold (measured: |levels| ~ 30
        # per feature, row logits ~ 35, held-out log loss 27.6 = the clip, against 1.6 for a uniform predictor).
        nfree = 1.0 if kind != "multiclass" else max(1.0, (1 if C2 is None else C2.shape[1]) - 1.0)
        for k in np.unique(lab):
            sel = lab == k
            w = float(H[sel].sum())
            V[sel] = V[sel] * (w / (w + shrink * nfree))
    if kind == "multiclass":
        V = np.clip(V, -6.0, 6.0)          # a level is a logit contribution; |v| > 6 per feature is not a model
    emp = H <= 1e-12
    V[emp] = 0.0
    gd = ~emp
    if gd.any():
        if kind == "multiclass":
            V[gd] = V[gd] - (V[gd] * H[gd][:, None]).sum(axis=0) / max(H[gd].sum(), 1e-12)
            V[gd] = np.clip(V[gd], -6.0, 6.0)
        else:
            V[gd] = V[gd] - (V[gd] * H[gd]).sum() / max(H[gd].sum(), 1e-12)
    V[emp] = 0.0
    return V


def _nan_dock(V: np.ndarray, H: np.ndarray, z: np.ndarray, lab: np.ndarray, nan_bin: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """M8: fuse the NaN bin into the segment the Potts criterion prefers, if the charge is below 1/2 ln(1+H_nan)."""
    if nan_bin >= len(V) or H[nan_bin] <= 0:
        return V, H, lab, False
    best, bsel = np.inf, -1
    for k in np.unique(lab):
        sel = lab == k
        Hs = H[sel].sum()
        if Hs <= 0:
            continue
        vs = float((V[sel] * H[sel]).sum() / Hs)
        d = H[nan_bin] * Hs / (H[nan_bin] + Hs) * (float(z[nan_bin]) - vs) ** 2
        if d < best:
            best, bsel = d, k
    if bsel < 0 or best >= 0.5 * np.log(1.0 + H[nan_bin]):
        return V, H, lab, False
    sel = lab == bsel
    Hs = H[sel].sum()
    vs = float((V[sel] * H[sel]).sum() / Hs)
    vnew = (Hs * vs + H[nan_bin] * float(z[nan_bin])) / (Hs + H[nan_bin])
    V[sel] = vnew
    V[nan_bin] = vnew
    H[np.argmax(sel)] += H[nan_bin]
    H[nan_bin] = 0.0
    lab[nan_bin] = bsel
    return V, H, lab, True


def fit_card(X: np.ndarray, y: np.ndarray, kind: str, cfg: FitCfg,
             feat_names: Optional[List[str]] = None,
             cuts: Optional[List[np.ndarray]] = None) -> Card:
    """The one fitter every candidate is built on.

    Each pass sweeps the features; for every feature it (1) solves each bin exactly on the current offset,
    (2) fuses the bin levels with the l0 DP under the chosen price, (3) refits the segment level from the bins' H.
    The loop is the point: applying a price once to a converged fit leaves old levels on new boundaries and scores
    nonsense (27.18 LL against 0.0711 on APS, ledger 767, L14).
    """
    X = np.asarray(X, float)
    n, p = X.shape
    if feat_names is None:
        feat_names = ["f%d" % j for j in range(p)]
    E = [cuts[j] if cuts is not None else cuts_of(X[:, j], cfg.m) for j in range(p)]
    NB = [len(E[j]) + 2 for j in range(p)]
    Itr = [bin_idx(X[:, j], E[j]) for j in range(p)]
    K = (cfg.n_classes or (int(y.max()) + 1)) if kind == "multiclass" else 0
    if kind == "multiclass" and cfg.n_classes and int(y.max()) >= cfg.n_classes:
        raise ValueError("more classes in the fold than the task declared")

    if kind == "regression":
        b0 = np.array(float(np.mean(y)))
        eta = np.full(n, b0)
        Fv = [np.zeros(NB[j]) for j in range(p)]
    elif kind == "binary":
        pm = np.clip(float(np.mean(y)), 1e-6, 1 - 1e-6)
        b0 = np.array(float(np.log(pm / (1 - pm))))
        eta = np.full(n, b0)
        Fv = [np.zeros(NB[j]) for j in range(p)]
    else:
        Ym = onehot(y, K)
        cnt = np.bincount(y.astype(int), minlength=K).astype(float)
        prior = (cnt + 1.0) / (cnt.sum() + K)          # Laplace: an absent class is improbable, not impossible
        b0 = np.log(prior)
        eta = np.tile(b0, (n, 1))
        Fv = [np.zeros((NB[j], K)) for j in range(p)]

    Y1 = onehot(y, K) if kind == "multiclass" else None
    prev = np.inf
    converged = False
    ll_hist: List[float] = []
    delta_hist: List[float] = []
    V_prev = [np.zeros_like(np.asarray(v, float)) for v in Fv]
    docked = 0

    for it in range(max(cfg.passes, cfg.max_passes if cfg.conv_tol > 0 else cfg.passes)):
        if cfg.order == "fisher":
            if kind == "multiclass":
                P = softmax(eta)
                w = (P * (1.0 - P)).sum(axis=1)
            elif kind == "binary":
                P = sig(eta)
                w = P * (1.0 - P)
            else:
                w = np.ones(n)
            # Fisher mass of feature j = the row weight it has to explain times its informative-bin count. The
            # record's Anneal gain came from this ordering, not from the joint solve, so it must apply to all kinds.
            mass = np.array([float(w[np.asarray(Itr[j], int)].sum()) * (1.0 + len(np.unique(Itr[j]))) for j in range(p)])
            order = list(np.argsort(-mass))          # Fisher ordering: biggest information first (r64)
        else:
            order = list(range(p))
        for j in order:
            idx = Itr[j]
            nb = NB[j]
            if kind == "multiclass":
                off = eta - Fv[j][idx]
                if R84_FISHER["multiclass"]:
                    C2, Wb, Gb, Hs = solve_bin_multiclass_fisher(off, Y1, nb, idx, K, alpha=cfg.alpha)
                    emp = Hs <= 1e-12
                    kap = 0.5 * np.log(max(float(Hs[~emp].sum()), 1.0)) * cfg.kappa_mult
                    lab = potts_dp_fisher(np.where(emp[:, None], 0.0, C2), Wb, Gb, emp, kap)
                    V = _segment_levels_fisher(lab, Wb, Gb, emp)
                else:
                    C2, H = solve_bin_multiclass(off, Y1, nb, idx, K, alpha=cfg.alpha)
                    emp = H <= 1e-12
                    kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * cfg.kappa_mult
                    lab = potts_dp(np.where(emp[:, None], 0.0, C2), np.where(emp, 1e-9, H), kap)
                    V = _segment_levels(lab, None, H, "multiclass", C2=np.where(emp[:, None], 0.0, C2), shrink=cfg.level_shrink)
            else:
                off = eta - Fv[j][idx]
                if kind == "binary":
                    z, H = solve_bin_logistic(off, y, nb, idx, alpha=cfg.alpha)
                else:
                    z, H = solve_bin_regression(off, y, nb, idx, alpha=0.0)
                emp = H <= 1e-12
                zc = np.where(emp, 0.0, z)
                kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * cfg.kappa_mult
                if cfg.price == "quadratic":
                    lab = potts_dp(zc, np.where(emp, 1e-9, H), kap)
                elif cfg.price == "deviance" and kind == "binary":
                    rbb = rows_per_bin(idx, nb)
                    cost = cost_deviance(rbb, off, y, zc, alpha=cfg.alpha)
                    if cfg.matched_runs and cfg.qfix is not None:
                        lab = dp_exact_q(cost, int(cfg.qfix[j]))
                        if lab is None:
                            raise RuntimeError("infeasible matched-run target for feature %d (q=%d)" % (j, cfg.qfix[j]))
                    else:
                        lab = dp_from_cost(cost, kap)
                elif cfg.price == "skew" and kind == "binary":
                    pv = sig(off + zc[idx])
                    S = np.bincount(idx, weights=pv * (1 - pv) * (1 - 2 * pv), minlength=nb)
                    lab = potts_skew_dp(zc, np.where(emp, 1e-9, H), np.where(emp, 0.0, S), kap)
                else:
                    lab = potts_dp(zc, np.where(emp, 1e-9, H), kap)
                if cfg.matched_runs and cfg.qfix is not None and cfg.price == "quadratic":
                    lab = dp_exact_q(cost_quadratic(zc, np.where(emp, 1.0, H)), int(cfg.qfix[j]))
                    if lab is None:
                        raise RuntimeError("infeasible q for feature %d" % j)
                if cfg.nan_dock:
                    V, H2, lab, did = _nan_dock(np.zeros(nb), H.copy(), zc, lab, nb - 1)
                    docked += int(did)
                V = _segment_levels(lab, zc, H, "binary" if kind == "binary" else "regression",
                                    shrink=(cfg.level_shrink if kind == "binary" else 0.0))
            if cfg.sigma_noise > 0:
                # idea #7: E[loss] under mean-one noise (1+sigma*eps) adds sigma^2/2 * ||f_j||^2, i.e. an extra ridge
                V = V / (1.0 + cfg.sigma_noise ** 2)
            Fv[j] = V
            eta = off + V[idx]

        # ---- convergence bookkeeping ---------------------------------------------------------------
        if kind == "regression":
            resid = y - eta
            cur = float(np.mean(resid ** 2))
        elif kind == "binary":
            cur = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
        else:
            cur = float(-np.mean(np.sum(Y1 * log_softmax(eta), axis=1)))
        ll_hist.append(cur)
        # the OBJECTIVE is nearly flat between passes even while the model is still moving, so convergence is
        # measured on the level vector itself: max_j max|V_j^(new) - V_j^(old)| < conv_tol is the fixed-point test
        delta = max(float(np.max(np.abs(np.asarray(Fv[j], float) - V_prev[j]))) for j in range(p)) if p else 0.0
        delta_hist.append(delta)
        V_prev = [np.asarray(Fv[j], float).copy() for j in range(p)]
        if cfg.conv_tol > 0 and delta < cfg.conv_tol:
            converged = True
            break
        prev = cur
    if cfg.conv_tol <= 0:
        converged = True                      # the pass cap IS the specification; nothing was promised beyond it
    else:
        converged = bool(delta_hist and delta_hist[-1] < cfg.conv_tol)

    # ---- ramps (idea #5) ----------------------------------------------------------------------------
    ramps = None
    if cfg.ramp_w > 0 and kind in ("binary", "regression"):
        ramps = []
        for j in range(p):
            idx = Itr[j]
            off = eta - Fv[j][idx]
            Ej = E[j]
            V = Fv[j].copy()
            rj = np.zeros((len(Ej), 2))       # (slope, centre-cut index) per live cut
            for c in range(1, min(len(V), len(Ej) + 1)):
                if V[c] == V[c - 1]:
                    continue
                cut = Ej[c - 1] if c - 1 < len(Ej) else Ej[-1]
                rng = float(np.ptp(X[np.isfinite(X[:, j]), j])) if np.isfinite(X[:, j]).any() else 1.0
                w = max(cfg.ramp_w * rng, 1e-12)
                sel = np.abs(X[:, j] - cut) <= w
                if sel.sum() < 10:
                    continue
                d = np.clip((X[sel, j] - cut) / w, -1.0, 1.0)
                tgt = (y[sel] - sig(off[sel] + V[idx[sel]])) if kind == "binary" else (y[sel] - off[sel] - V[idx[sel]])
                denom = float((d * d).sum()) + 1e-9
                sl = float(np.clip((d * tgt).sum() / denom, -cfg.ramp_cap / w, cfg.ramp_cap / w))
                rj[c - 1] = (sl, 1)
            ramps.append(rj)
    card = Card(kind=kind, cuts=E, V=Fv, b0=b0, T=np.array(1.0), feat_names=list(feat_names),
                meta=dict(passes=len(ll_hist), converged=bool(converged), ll_hist=ll_hist,
                          delta_hist=delta_hist, alpha=cfg.alpha,
                          kappa_mult=cfg.kappa_mult, price=cfg.price, docked=docked, m=cfg.m,
                          convergence_by_construction=cfg.conv_tol <= 0))
    if ramps is not None:
        card.meta["ramps"] = ramps
        card.meta["ramp_abs"] = [max(cfg.ramp_w * (np.ptp(X[:, j][np.isfinite(X[:, j])]) if np.isfinite(X[:, j]).any() else 1.0), 1e-12)
                                 for j in range(p)]
        card.meta["has_ramps"] = True
    return card


def card_score_with_ramps(card: Card, X: np.ndarray) -> np.ndarray:
    """Prediction for a card that carries ramps (idea #5). Flat outside each ramp neighbourhood, by construction."""
    s = card.score(X)
    ramps = card.meta.get("ramps")
    if not ramps:
        return s
    for j, rj in enumerate(ramps):
        Ej = card.cuts[j]
        fin = np.isfinite(X[:, j])
        for c in range(len(Ej)):
            if rj[c, 1] == 0.0:
                continue
            cut = Ej[c]
            w = card.meta.get("ramp_w_abs", None)
            if w is None:
                rng = float(np.ptp(X[fin, j])) if fin.any() else 1.0
                w = max(0.05 * rng, 1e-12)
            d = np.clip((X[:, j] - cut) / w, -1.0, 1.0)
            s = s + np.where(fin, rj[c, 0] * w * d, 0.0)
    return s


# --------------------------------------------------------------------------------------------------------------
# 5.  The printer: lattice-exact thresholds, dead-cut compilation, byte accounting, and a decoder that sees only bytes
# --------------------------------------------------------------------------------------------------------------
def ceil_f32(t: np.ndarray) -> np.ndarray:
    """min{f in float32 : f >= t}. The storage convention that makes routing exact for EVERY float32 input.

    Routing is x -> #{cuts <= x}. Round-to-nearest can store a value BELOW t, and then x = rne32(t) is a float32
    input the card routes to the left of where the true cut routes it. That defect exists in the record's shipped
    cards (346 of 7,155 APS cuts, ledger 775, L18). It costs +4 B gzip to fix and it is not optional.
    """
    t = np.asarray(t, float)
    c = t.astype(F32)
    for _ in range(3):
        bad = c.astype(float) < t
        if not bad.any():
            break
        c = np.where(bad, np.nextafter(c, F32(np.inf)), c).astype(F32)
    return c


def _runs_of(V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run-length description of a level table: (lengths, levels) for consecutive equal rows."""
    V = np.asarray(V)
    if V.ndim == 1:
        ch = np.concatenate([[True], V[1:] != V[:-1]])
    else:
        ch = np.concatenate([[True], np.any(V[1:] != V[:-1], axis=1)])
    idx = np.flatnonzero(ch)
    lens = np.diff(np.append(idx, len(V)))
    return lens, V[idx]


def emit_card(card: Card, mode: str = "deadcut_ceil", with_ramps: Optional[bool] = None) -> dict:
    """Serialise the card. Returns the blob bytes plus an ITEMISED account of where every byte went.

    mode = 'full_rne'      all cuts, round-to-nearest float32   (the historical format; not routing-exact)
           'full_ceil'     all cuts, ceil float32               (exact, no compilation)
           'deadcut_ceil'  only cuts where the printed level changes, ceil float32  (idea #1)
    """
    if with_ramps is None:
        with_ramps = bool(card.meta.get("has_ramps"))
    buf = io.BytesIO()
    items = dict(header=0, cuts=0, levels=0, nan=0, inter=16)
    n_cuts_total = 0
    n_cuts_live = 0
    for j in range(len(card.cuts)):
        E = np.asarray(card.cuts[j], float)
        V = np.asarray(card.V[j])
        nb = len(V)
        # live boundaries: index c (0-based boundary before bin c+1) where the level changes
        if V.ndim == 1:
            chg = [c for c in range(1, nb - 1) if V[c] != V[c - 1]]
        else:
            chg = [c for c in range(1, nb - 1) if np.any(V[c] != V[c - 1])]
        n_cuts_total += len(E)
        if mode == "deadcut_ceil":
            keep = [E[c - 1] for c in chg]
            lev = [V[0]] + [V[c] for c in chg]
        else:
            keep = list(E)
            lev = [V[c] for c in range(nb - 1)]
        K = 1 if V.ndim == 1 else V.shape[1]
        Cst = ceil_f32(np.asarray(keep, float)) if mode.endswith("ceil") else np.asarray(keep, float).astype(F32)
        Lv = np.asarray(lev, float).astype(F32).reshape(len(lev), K)
        nanl = np.asarray(V[nb - 1], float).astype(F32).reshape(1, K)
        buf.write(np.frombuffer(np.array([len(keep)], np.uint16).tobytes(), np.uint16).tobytes())
        buf.write(np.frombuffer(np.array([len(lev)], np.uint16).tobytes(), np.uint16).tobytes())
        buf.write(Cst.tobytes())
        buf.write(Lv.tobytes())
        buf.write(nanl.tobytes())
        items["header"] += 4
        items["cuts"] += Cst.nbytes
        items["levels"] += Lv.nbytes
        items["nan"] += nanl.nbytes
        n_cuts_live += len(keep)
    b0 = np.asarray(card.b0, float).astype(np.float64)
    T = np.asarray(card.T, float).astype(np.float64).ravel()
    buf.write(b0.tobytes())
    buf.write(T.tobytes())
    items["inter"] = b0.nbytes + T.nbytes
    if with_ramps and card.meta.get("ramps") is not None:
        r = np.concatenate([np.asarray(x, np.float32).ravel() for x in card.meta["ramps"]])
        buf.write(r.tobytes())
        items["ramps"] = r.nbytes
    raw = buf.getvalue()
    return dict(raw=raw, bytes_raw=len(raw), bytes_gzip=len(gzip.compress(raw, 9)), items=items,
                n_cuts_total=int(n_cuts_total), n_cuts_live=int(n_cuts_live),
                ratio_live=round(n_cuts_live / max(n_cuts_total, 1), 4), mode=mode)


def decode_card(blob: bytes, kind: str, n_feat: int, K: int = 1) -> Card:
    """Decode from the BYTES ONLY. This is what a fresh process has; nothing about the fit is retained."""
    off = 0
    cuts, V = [], []
    for _ in range(n_feat):
        ncut = int(np.frombuffer(blob[off:off + 2], np.uint16)[0]); off += 2
        nlev = int(np.frombuffer(blob[off:off + 2], np.uint16)[0]); off += 2
        C = np.frombuffer(blob[off:off + 4 * ncut], F32).astype(np.float64); off += 4 * ncut
        Lv = np.frombuffer(blob[off:off + 4 * nlev * K], F32).astype(np.float64).reshape(nlev, K); off += 4 * nlev * K
        nanl = np.frombuffer(blob[off:off + 4 * K], F32).astype(np.float64); off += 4 * K
        cuts.append(C)
        V.append(np.concatenate([Lv, nanl.reshape(1, K)], axis=0) if K > 1 else np.concatenate([Lv.ravel(), nanl]))
    b0 = np.frombuffer(blob[off:off + 8], np.float64).copy(); off += 8
    T = np.frombuffer(blob[off:off + 8], np.float64).copy(); off += 8
    return Card(kind=kind, cuts=cuts, V=V, b0=b0, T=T, feat_names=["f%d" % j for j in range(n_feat)], meta={})


def probe_set(card: Card, X: np.ndarray) -> np.ndarray:
    """Adversarial float32 battery per feature: exact ties, +-1..3 ulp, mid-bin, NaN, +-inf, denormals, +-FLT_MAX, -0."""
    rows = []
    for j in range(len(card.cuts)):
        E = np.asarray(card.cuts[j], float)
        if len(E) == 0:
            rows.append(np.array([np.nan, np.inf, -np.inf, 0.0]))
            continue
        e32 = E.astype(F32)
        v = [E, e32.astype(float)]
        for s in (1, 2, 3):
            a = e32.copy()
            for _ in range(s):
                a = np.nextafter(a, F32(np.inf))
            v.append(a.astype(float))
            a = e32.copy()
            for _ in range(s):
                a = np.nextafter(a, F32(-np.inf))
            v.append(a.astype(float))
        if len(E) > 1:
            v.append((E[:-1] + E[1:]) / 2.0)
        v.append(np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.4e-45, -1.4e-45, 3.4028234663852886e38,
                           -3.4028234663852886e38]))
        rows.append(np.concatenate(v))
    return rows


def verify_printer(card: Card, X: np.ndarray, mode: str) -> dict:
    """Emit -> decode -> compare.

    The equality claim is stated against the EXACT float32 emission (`full_ceil`), because the emitted levels are
    float32 and comparing them with the float64 fitted levels would only measure float32 rounding. Two further
    diagnostics are reported and neither is a defect: the maximum deviation from the float64 fit (level rounding)
    and, for the round-to-nearest variant, how many cuts are stored below their true value (the L18 defect).
    """
    K = 1 if np.asarray(card.V[0]).ndim == 1 else np.asarray(card.V[0]).shape[1]
    blob = emit_card(card, mode)
    dec = decode_card(blob["raw"], card.kind, len(card.cuts), K)
    ref_blob = emit_card(card, "full_ceil")
    ref = decode_card(ref_blob["raw"], card.kind, len(card.cuts), K)
    # the deployed contract: identical for every float32 input, so compare on rows widened from float32
    X32 = np.asarray(X, float).astype(F32).astype(float)
    s_dec = dec.score(X32)
    s_ref = ref.score(X32)
    s_fit64 = card.score(np.asarray(X, float))
    bad = 0
    npb = 0
    for j in range(len(card.cuts)):
        xj = probe_set(card, X)[j].astype(F32).astype(float)
        npb += len(xj)
        a = np.asarray(ref.V[j])[bin_idx(xj, ref.cuts[j])]
        b = np.asarray(dec.V[j])[bin_idx(xj, dec.cuts[j])]
        bad += int(not np.array_equal(a, b))
    viol = 0
    ncheck = 0
    if mode.endswith("rne"):
        for j in range(len(card.cuts)):
            for t in np.asarray(card.cuts[j], float):
                ncheck += 1
                if float(F32(t)) < t:
                    viol += 1
    return dict(mode=mode, bytes_gzip=blob["bytes_gzip"], bytes_raw=blob["bytes_raw"], items=blob["items"],
                n_cuts_total=blob["n_cuts_total"], n_cuts_live=blob["n_cuts_live"],
                rows_bit_identical=bool(np.array_equal(s_dec, s_ref)) and bad == 0,
                max_abs_diff_vs_float64_fit=float(np.max(np.abs(s_dec - s_fit64))) if len(s_dec) else 0.0,
                probe_points=int(npb), probe_features_failing=int(bad),
                rne_cuts_below_true_cut=int(viol), cuts_checked=int(ncheck),
                reference="full_ceil emission (the exact float32 artefact)")


# --------------------------------------------------------------------------------------------------------------
# 6.  Idea #4 — fit finely, compress afterwards (no labels consulted in the merge)
# --------------------------------------------------------------------------------------------------------------
def compress_card(card: Card, X: np.ndarray, max_runs: int) -> Card:
    """Post-fit compression of a fitted card by agglomerating adjacent runs, minimising the emitted-model KL.

    No labels and no backfit are consulted: the merge criterion is the row mass times the squared level gap,
        Delta(a,b) = w_a w_b / (w_a + w_b) * || v_a - v_b ||^2,
    which is exactly the increase in sum_i (f(x_i) - f'(x_i))^2 when two runs merge, i.e. the second-order
    approximation of the predictive KL between the emitted models. This is a COMMUNICATION approximation, not a
    new estimator, and the distance from the uncompressed predictor is reported with it.
    """
    cuts_out, V_out = [], []
    for j in range(len(card.cuts)):
        V = np.asarray(card.V[j], float)
        idx = bin_idx(X[:, j], card.cuts[j])
        w = np.bincount(idx, minlength=len(V)).astype(float)
        lens, lev = _runs_of(V)
        w_run = np.bincount(np.repeat(np.arange(len(lens)), lens), weights=w, minlength=len(lens))
        levv = lev.copy()
        alive = list(range(len(lens)))
        while len(alive) > max(1, max_runs // max(1, len(card.cuts))) and len(alive) > 1:
            best, bd = None, np.inf
            for a in range(len(alive) - 1):
                i, k = alive[a], alive[a + 1]
                wa, wk = w_run[i], w_run[k]
                d = wa * wk / max(wa + wk, 1e-12) * float(np.sum((levv[i] - levv[k]) ** 2))
                if d < bd:
                    bd, best = d, a
            if best is None:
                break
            i, k = alive[best], alive[best + 1]
            tot = w_run[i] + w_run[k]
            levv[i] = (levv[i] * w_run[i] + levv[k] * w_run[k]) / max(tot, 1e-12)
            w_run[i] = tot
            lens[i] = lens[i] + lens[k]
            alive.pop(best + 1)
        # rebuild the level table and the live cut list from the surviving runs
        newV = np.zeros_like(V)
        pos = 0
        live_cuts = []
        for a, i in enumerate(alive):
            ln = int(lens[i])
            newV[pos:pos + ln] = levv[i]
            if a > 0:
                c = pos - 1
                if c < len(card.cuts[j]):
                    live_cuts.append(card.cuts[j][c])
            pos += ln
        newV[pos:] = levv[alive[-1]]
        cuts_out.append(np.asarray(live_cuts, float) if len(live_cuts) else np.array([]))
        V_out.append(newV)
    return Card(kind=card.kind, cuts=cuts_out, V=V_out, b0=np.array(card.b0), T=np.array(card.T),
                feat_names=card.feat_names, meta=dict(card.meta, compressed=True, max_runs=max_runs))


# --------------------------------------------------------------------------------------------------------------
# 7.  Calibrators.  T20 and Toof are NOT interchangeable; every row states which one it used.
# --------------------------------------------------------------------------------------------------------------
def fit_T(y: np.ndarray, e: np.ndarray, kind: str) -> np.ndarray:
    from scipy.optimize import minimize_scalar

    def obj(t: float) -> float:
        if kind == "regression":
            return float(np.mean((y - e) ** 2))
        if kind == "binary":
            z = e / t
            return float(np.mean(np.logaddexp(0.0, z) - y * z))
        z = e / t
        return float(-np.mean(np.sum(onehot(y, e.shape[1]) * log_softmax(z), axis=1)))

    if kind == "regression":
        return np.array(1.0)
    r = minimize_scalar(obj, bounds=(0.2, 8.0), method="bounded")
    return np.array(float(r.x))


def calib_T20(card: Card, Xtr: np.ndarray, ytr: np.ndarray, seed: int = 65) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    k = int(0.8 * len(ytr))
    return fit_T(ytr[idx[k:]], card.score(Xtr[idx[k:]]), card.kind)


def calib_Toof(card_factory: Callable[[], Card], Xtr: np.ndarray, ytr: np.ndarray,
               folds: int = 5, seed: int = 67) -> np.ndarray:
    """The REGISTERED protocol: out-of-fold scores over the whole training portion, then one temperature.

    Costs `folds` extra fits. That price is the reason the record's older numbers used T20, and the reason they
    are not comparable with anything scored under Toof.
    """
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(ytr)), folds)
    if card_factory().kind == "regression":
        return np.array(1.0)
    if card_factory().kind == "binary":
        eta = np.zeros(len(ytr))
    else:
        K = len(card_factory().b0)
        eta = np.zeros((len(ytr), K))
    for k in range(folds):
        va = parts[k]
        tr = np.concatenate([parts[g] for g in range(folds) if g != k])
        ck = card_factory()
        ck = fit_card(Xtr[tr], ytr[tr], ck.kind, FitCfg(**ck.meta.get("cfg_kwargs", {})),
                      cuts=[None] * Xtr.shape[1])
        eta[va] = ck.score(Xtr[va])
    return fit_T(ytr, eta, card_factory().kind)


def calib_s16(card: Card, Xtr: np.ndarray, ytr: np.ndarray, seed: int = 67) -> np.ndarray:
    """The record's S16: per-class OFFSETS on the class scores (one MLE on train), not scales.

    The record's registered multiclass calibrator (round 56). r71's `diag` fits per-class SCALES and
    measurably diverges where the record used offsets (Anneal on the record's md5 folds: 0.198 at diag
    against 0.067 at Toof); the form of the correction is the difference, so both are kept.
    """
    if card.kind != "multiclass":
        return calib_T20(card, Xtr, ytr, seed)
    from scipy.optimize import minimize
    e = card.score(Xtr)
    Y = onehot(ytr, e.shape[1])
    K = e.shape[1]

    def obj(b: np.ndarray) -> float:
        return float(-np.mean(np.sum(Y * log_softmax(e + b[None, :]), axis=1)))

    r = minimize(obj, np.zeros(K), method="L-BFGS-B")
    return r.x


def calib_diag(card: Card, Xtr: np.ndarray, ytr: np.ndarray, seed: int = 67) -> np.ndarray:
    """S16-analogue with per-class scales (binary reduces to T; the record's S16 itself is `s16` above)."""
    if card.kind != "multiclass":
        return calib_T20(card, Xtr, ytr, seed)
    from scipy.optimize import minimize
    e = card.score(Xtr)
    Y = onehot(ytr, e.shape[1])
    K = e.shape[1]

    def obj(th: np.ndarray) -> float:
        s = np.exp(th)
        return float(-np.mean(np.sum(Y * log_softmax(e * s[None, :]), axis=1)))

    r = minimize(obj, np.zeros(K), method="L-BFGS-B")
    return np.exp(r.x)


def calib_joint(card: Card, Xtr: np.ndarray, ytr: np.ndarray, off_pen: float = 1.0) -> np.ndarray:
    """Idea #12: fit eta' = A eta + b by the JOINT proper likelihood, penalising off-diagonal entries of A.

    Not R37: no partition or class effect is refitted, and the objective is convex in (A, b). The penalty is what
    keeps the eight-row rarest Anneal class from producing a wild row of A. Returns a dict in card.meta['joint'].
    """
    e = card.score(Xtr)
    K = e.shape[1] if card.kind == "multiclass" else 1
    Y = onehot(ytr, K) if card.kind == "multiclass" else None
    n = len(ytr)

    def unpack(th: np.ndarray):
        A = th[:K * K].reshape(K, K)
        b = th[K * K:]
        return A, b

    def obj(th: np.ndarray) -> float:
        A, b = unpack(th)
        if K == 1:
            z = (e * A[0, 0] + b[0]).ravel()
            if card.kind == "regression":
                return float(np.mean((ytr - z) ** 2))
            return float(np.mean(np.logaddexp(0.0, z) - ytr * z))
        z = e @ A.T + b[None, :]
        pen = off_pen * float((A - np.diag(np.diag(A))).sum() ** 2)
        return float(-np.mean(np.sum(Y * log_softmax(z), axis=1))) + pen

    from scipy.optimize import minimize
    th0 = np.concatenate([np.eye(K).ravel(), np.zeros(K)])
    r = minimize(obj, th0, method="L-BFGS-B", options=dict(maxiter=300))
    A, b = unpack(r.x)
    return dict(A=A, b=b, success=bool(r.success), objective=float(r.fun))


def calib_support(card: Card, Xtr: np.ndarray, ytr: np.ndarray, seed: int = 67) -> dict:
    """Idea #13: one extra calibration coefficient on the row's SUPPORT uncertainty, orthogonalised to the score.

    u_i is built from the precisions H_b of the emitted pieces the row visits; it is residualised against the
    OOF logit first, so the coefficient cannot re-fit amplitude (that is the T it competes with). No main is gated.
    """
    e = card.score(Xtr)
    u = np.zeros(len(Xtr))
    for j in range(len(card.cuts)):
        V = np.asarray(card.V[j], float)
        idx = bin_idx(Xtr[:, j], card.cuts[j])
        cnt = np.bincount(idx, minlength=len(V)).astype(float)
        prec = 1.0 / (cnt + 1.0)
        u += prec[idx]
    u = u / max(len(card.cuts), 1)
    if card.kind == "multiclass":
        uu = np.repeat(u[:, None], e.shape[1], axis=1)
    else:
        uu = u

    def obj(th: np.ndarray) -> float:
        t, g = np.exp(th[0]), th[1]
        z = (e + g * uu) / t
        if card.kind == "binary":
            return float(np.mean(np.logaddexp(0.0, z) - ytr * z))
        if card.kind == "regression":
            return float(np.mean((ytr - z) ** 2))
        Y = onehot(ytr, e.shape[1])
        return float(-np.mean(np.sum(Y * log_softmax(z), axis=1)))

    from scipy.optimize import minimize
    r = minimize(obj, np.array([0.0, 0.0]), method="L-BFGS-B")
    return dict(T=float(np.exp(r.x[0])), gamma=float(r.x[1]), success=bool(r.success))


def apply_joint(card: Card, spec: dict) -> Card:
    A, b = np.asarray(spec["A"]), np.asarray(spec["b"])
    c2 = Card(kind=card.kind, cuts=card.cuts, V=card.V, b0=card.b0, T=np.array(1.0),
              feat_names=card.feat_names, meta=dict(card.meta))
    c2.meta["joint"] = spec
    c2.meta["A"] = A
    c2.meta["bvec"] = b
    return c2


def score_with_calibration(card: Card, X: np.ndarray, cal: str, spec: dict) -> np.ndarray:
    """The one place calibration is applied, so no row can silently mix protocols."""
    s = card.score(X)
    if cal == "T20":
        T = float(np.asarray(spec["T"]).ravel()[0])
        return s if card.kind == "regression" else s / T
    if cal == "Toof":
        T = float(np.asarray(spec["T"]).ravel()[0])
        return s if card.kind == "regression" else s / T
    if cal == "diag":
        sc = np.asarray(spec["scale"]).ravel()
        if card.kind != "multiclass":
            T = float(sc[0])
            return s if card.kind == "regression" else s / T
        return s * sc[None, :]
    if cal == "s16":
        if card.kind != "multiclass":
            T = float(np.asarray(spec["offsets"]).ravel()[0])
            return s if card.kind == "regression" else s / T
        b = np.asarray(spec["offsets"]).ravel()
        return s + b[None, :]
    if cal == "joint":
        A = np.asarray(spec["A"])
        b = np.asarray(spec["b"])
        return s * A[0, 0] + b[0] if card.kind != "multiclass" else s @ A.T + b[None, :]
    if cal == "support":
        u = np.zeros(len(X))
        for j in range(len(card.cuts)):
            V = np.asarray(card.V[j], float)
            idx = bin_idx(X[:, j], card.cuts[j])
            cnt = np.bincount(idx, minlength=len(V)).astype(float)
            u += (1.0 / (cnt + 1.0))[idx]
        u = u / max(len(card.cuts), 1)
        T = float(spec["T"])
        g = float(spec["gamma"])
        z = (s + g * (u[:, None] if s.ndim > 1 else u)) / T
        return z
    raise ValueError("unknown calibrator %r" % cal)


# --------------------------------------------------------------------------------------------------------------
# 8.  The candidates: every "best model" in the record, each a build function returning an uncalibrated card
# --------------------------------------------------------------------------------------------------------------
# The record's best objects, by task, and the candidate that re-implements each:
#   California random   full joint card        -> card_fused (+ card_pairlayer for the pair layer)
#   California lat10    gated coast backbone   -> card_gated_coast
#   California 2-D      same                   -> card_gated_coast
#   California lco      K=48 + pair layer      -> card_pairlayer
#   APS accuracy        kappa = 0 card         -> card_kappa0
#   APS size            fused + dead cuts      -> card_deadcut
#   Anneal grouped      Fisher ordering        -> card_fused(order='fisher')  == card_fisher
#   universals          linear spine q=4       -> linear_spine
# ...plus the document-22 ideas whose implementations live below: #4 card_postcompress, #5 card_ramp,
# #6 card_deviance, #7 card_effectnoise, #8 card_jointdir, #11 card_bagfixed, #15 card_contam, and M3 card_skew.

COAST = np.array([                     # a COARSE California coastline; see the honesty note in the header
    (32.53, -117.12), (33.00, -117.28), (33.60, -117.90), (34.00, -118.50), (34.40, -119.70),
    (34.90, -120.60), (35.40, -120.90), (36.30, -121.90), (36.90, -122.00), (37.50, -122.50),
    (38.30, -123.05), (39.00, -123.70), (39.80, -123.85), (40.40, -124.40), (41.00, -124.10),
    (41.75, -124.20),
])


RECORD_COAST = np.array([             # the record's own coastline (run_cardboost5.COAST): 18 vertices
    [42.00, -124.21], [41.00, -124.10], [40.44, -124.41], [39.35, -123.82],
    [38.32, -123.06], [37.78, -122.51], [37.00, -122.20], [36.60, -121.90],
    [36.30, -121.90], [35.65, -121.20], [35.15, -120.68], [34.45, -120.47],
    [34.10, -119.20], [33.77, -118.40], [33.60, -117.90], [33.10, -117.30],
    [32.53, -117.12], [32.53, -117.12],
])


def coast_distance_record(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """The record's feature exactly: distance in degrees to the nearest SEGMENT of its 18-point polyline.

    The default measures distance to the nearest VERTEX, which overestimates the distance of any point projecting
    onto the middle of a long segment. The record's coast is NOT 'a real coastline' -- it is the hand-coded polyline
    above; the vertex/segment distinction is the implementation difference, and --record-coast removes it.
    """
    P = np.column_stack([lat, lon])
    A, B = RECORD_COAST[:-1], RECORD_COAST[1:]
    out = np.full(len(P), np.inf)
    for k in range(len(A)):
        ab = B[k] - A[k]
        t = np.clip(((P - A[k]) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        proj = A[k] + t[:, None] * ab
        out = np.minimum(out, np.sqrt(((P - proj) ** 2).sum(1)))
    return out


def coast_distance(lat: np.ndarray, lon: np.ndarray, record: bool = False) -> np.ndarray:
    """Distance in degrees to the nearest coastline vertex (default) or record segment (--record-coast)."""
    if record:
        return coast_distance_record(lat, lon)
    out = np.empty(len(lat))
    for i in range(len(lat)):
        dlat = COAST[:, 0] - lat[i]
        dlon = (COAST[:, 1] - lon[i]) * np.cos(np.deg2rad(lat[i]))
        out[i] = np.min(np.hypot(dlat, dlon))
    return out


def cell_feature(lat_tr: np.ndarray, lon_tr: np.ndarray, lat_te: np.ndarray, lon_te: np.ndarray,
                 grid: int = 16, min_support: int = 20) -> Tuple[np.ndarray, np.ndarray, dict]:
    """The support GATE: a nominal cell id, NaN where the cell is under-supported on training rows only.

    This is how the gated object is realised without inventing a new estimator: the gate is a feature whose level
    table is the cell correction, and whose NaN bin is what an unsupported cell falls into. Nothing outside the gate
    is extrapolated, which is the whole point of the gated card.
    """
    la0, la1 = np.nanmin(lat_tr), np.nanmax(lat_tr)
    lo0, lo1 = np.nanmin(lon_tr), np.nanmax(lon_tr)
    it = np.clip(((lat_tr - la0) / max(la1 - la0, 1e-9) * grid).astype(int), 0, grid - 1)
    jt = np.clip(((lon_tr - lo0) / max(lo1 - lo0, 1e-9) * grid).astype(int), 0, grid - 1)
    cell_tr = it * grid + jt
    ie = np.clip(((lat_te - la0) / max(la1 - la0, 1e-9) * grid).astype(int), 0, grid - 1)
    je = np.clip(((lon_te - lo0) / max(lo1 - lo0, 1e-9) * grid).astype(int), 0, grid - 1)
    cell_te = ie * grid + je
    cnt = np.bincount(cell_tr.astype(int), minlength=grid * grid)
    ok = cnt >= min_support
    g_tr = np.where(ok[cell_tr.astype(int)], cell_tr.astype(float), np.nan)
    g_te = np.where(ok[cell_te.astype(int)], cell_te.astype(float), np.nan)
    return g_tr, g_te, dict(grid=grid, min_support=min_support, supported=int(ok.sum()), cells=grid * grid)


def pair_features(Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, kind: str, base_score_tr: np.ndarray,
                  n_pairs: int = 10, bins: int = 8, min_support: int = 25, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, list]:
    """The pair layer: joint cells of the n_pairs most informative feature pairs, gated by training support.

    Pairs are chosen on TRAINING rows only, by the correlation of their residualised contributions, so a conditional
    screen never sees a test row. Each pair becomes one nominal column whose level table is an 8x8 surface.
    """
    n, p = Xtr.shape
    resid = ytr - base_score_tr if kind == "regression" else (ytr - sig(base_score_tr))
    score = []
    for j in range(p):
        a = bin_idx(Xtr[:, j], cuts_of(Xtr[:, j], bins))
        for k in range(j + 1, p):
            b = bin_idx(Xtr[:, k], cuts_of(Xtr[:, k], bins))
            cell = a * (bins + 2) + b
            m = np.bincount(cell, weights=resid, minlength=(bins + 2) ** 2)
            c = np.bincount(cell, minlength=(bins + 2) ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(c > 0, m / np.maximum(c, 1), 0.0)
            # variance explained by the 2-D surface, discounted by its cells (a crude, training-only criterion)
            score.append((float((mu ** 2 * c).sum()), j, k, bins + 2))
    score.sort(reverse=True)
    chosen = score[:n_pairs]
    pcols_tr, pcols_te, meta = [], [], []
    for s, j, k, B in chosen:
        cb_j = cuts_of(Xtr[:, j], bins)
        cb_k = cuts_of(Xtr[:, k], bins)
        a = bin_idx(Xtr[:, j], cb_j)
        b = bin_idx(Xtr[:, k], cb_k)
        cell = a * B + b
        c = np.bincount(cell, minlength=B * B)
        ae = bin_idx(Xte[:, j], cb_j)
        be = bin_idx(Xte[:, k], cb_k)
        celle = ae * B + be
        ok = c >= min_support
        pcols_tr.append(np.where(ok[cell], cell.astype(float), np.nan))
        pcols_te.append(np.where(ok[np.clip(celle, 0, B * B - 1)], celle.astype(float), np.nan))
        meta.append(dict(j=j, k=k, bins=B, cells=int(ok.sum()), score=s))
    if not pcols_tr:
        return np.zeros((n, 0)), np.zeros((len(Xte), 0)), []
    return np.column_stack(pcols_tr), np.column_stack(pcols_te), meta


def refit_levels(card: Card, X: np.ndarray, y: np.ndarray, cfg: FitCfg) -> Card:
    """Levels only, cuts and segment structure frozen. Cheap, and used by bagging and by the mixture refit."""
    V = []
    K = 1 if np.asarray(card.V[0]).ndim == 1 else np.asarray(card.V[0]).shape[1]
    eta = card.score(X)
    for j in range(len(card.cuts)):
        idx = bin_idx(X[:, j], card.cuts[j])
        off = eta - (card.V[j][idx])
        nb = len(card.V[j])
        Vj = np.asarray(card.V[j], float)
        if card.kind == "multiclass":
            C2, H = solve_bin_multiclass(off, onehot(y, K), nb, idx, K, alpha=cfg.alpha)
            lab = np.concatenate([[0], np.cumsum(np.any(np.diff(Vj, axis=0) != 0, axis=1))])
            newV = _segment_levels(lab, None, H, "multiclass", C2=np.where((H <= 1e-12)[:, None], 0.0, C2),
                                   shrink=getattr(cfg, "level_shrink", 0.0))
        else:
            if card.kind == "binary":
                z, H = solve_bin_logistic(off, y, nb, idx, alpha=cfg.alpha)
            else:
                z, H = solve_bin_regression(off, y, nb, idx)
            lab = np.concatenate([[0], np.cumsum(np.diff(Vj) != 0)])
            newV = _segment_levels(lab, np.where(H <= 1e-12, 0.0, z), H,
                                   "binary" if card.kind == "binary" else "regression",
                                   shrink=(getattr(cfg, "level_shrink", 0.0) if card.kind == "binary" else 0.0))
        V.append(newV)
        eta = off + newV[idx]
    out = Card(kind=card.kind, cuts=card.cuts, V=V, b0=np.array(card.b0), T=np.array(card.T),
               feat_names=card.feat_names, meta=dict(card.meta))
    return out


def joint_direction_fit(card: Card, X: np.ndarray, y: np.ndarray, gamma: float = 0.5,
                        corr_thresh: float = 0.2, sweeps: int = 6) -> Card:
    """Idea #8: penalise poorly identified JOINT directions, not individual atoms.

    The penalty strength on a direction rises as its joint Fisher eigenvalue falls:
        Lam_k = gamma / (|lam_k| + delta).
    A full eigendecomposition of the joint design is out of reach at APS scale (11k columns), so the penalty is
    applied blockwise: features whose training level vectors are correlated above `corr_thresh` form a block, and the
    eigen-penalised solve runs inside each block by Gauss-Seidel. Features with near-zero cross-correlation are
    untouched, which is the intended behaviour and is stated rather than hidden.
    """
    Fv = [np.asarray(v, float).copy() for v in card.V]
    p = len(Fv)
    # training level vectors as (n, K) contributions, used for the correlation screen and the block solve
    Zc = []
    for j in range(p):
        Zc.append(Fv[j][bin_idx(X[:, j], card.cuts[j])])
    dim = 1 if card.kind != "multiclass" else Fv[0].shape[1]

    def flat(a):
        return a.reshape(len(a), -1) if dim > 1 else a.reshape(len(a), 1)

    Zm = np.column_stack([flat(z) for z in Zc]) if p else np.zeros((len(X), 0))
    # correlation of feature-level contribution magnitudes (a cheap screen, training only)
    mag = np.column_stack([np.abs(flat(z)).mean(axis=1) for z in Zc])
    with np.errstate(invalid="ignore", divide="ignore"):
        Corr = np.corrcoef(mag, rowvar=False) if p > 1 else np.ones((1, 1))
    Corr = np.nan_to_num(Corr, nan=0.0)
    eta = card.score(X)
    for _ in range(sweeps):
        for j in range(p):
            nb = len(Fv[j])
            idx = bin_idx(X[:, j], card.cuts[j])
            off = eta - Fv[j][idx]
            mates = [k for k in range(p) if k != j and abs(Corr[j, k]) >= corr_thresh]
            if not mates:
                continue
            # build the local 2-feature system (j with its strongest mate) and solve with eigen-ridge
            k = max(mates, key=lambda q: abs(Corr[j, q]))
            cols, mats = [], []
            for q in (j, k):
                iq = bin_idx(X[:, q], card.cuts[q])
                cols.append(iq)
                mats.append(Fv[q])
            # design over the two features' bins, weighted by the current logistic curvature
            if card.kind == "binary":
                w = sig(off + Fv[j][idx]) * (1 - sig(off + Fv[j][idx])) + 1e-6
            else:
                w = np.ones(len(X))
            Zl = np.column_stack([np.eye(len(mat))[c] for c, mat in zip(cols, mats)])
            W = w
            G = (Zl * W[:, None]).T @ Zl
            lam, U = np.linalg.eigh(G + 1e-8 * np.eye(G.shape[0]))
            pen = gamma / (np.abs(lam) + 0.05)
            rs = np.asarray(y, float) - off if card.kind != "multiclass" else None
            if card.kind == "multiclass":
                tgt = (onehot(y, dim) - softmax(eta))[ :, :]
                rhs = (Zl * W[:, None]).T @ np.tile(np.zeros((len(X), 1)), (1, dim))
                step = np.zeros_like(G)
                # a multinomial joint solve is outside this method's scope; fall back to the per-feature Newton step
                continue
            b = (Zl * W[:, None]).T @ rs
            beta = U @ ((U.T @ b) / (lam + pen + 1e-9))
            for t2, q in enumerate((j, k)):
                blk = beta[t2 * len(mats[t2]):(t2 + 1) * len(mats[t2])]
                Fv[q] = blk if dim == 1 else blk.reshape(-1, dim)
        eta = card.score(X) if card.kind != "multiclass" else eta
        for j in range(p):
            eta = eta - np.asarray(card.V[j])[bin_idx(X[:, j], card.cuts[j])] + Fv[j][bin_idx(X[:, j], card.cuts[j])]
    return Card(kind=card.kind, cuts=card.cuts, V=Fv, b0=np.array(card.b0), T=np.array(card.T),
                feat_names=card.feat_names, meta=dict(card.meta, jointdir_gamma=gamma,
                                                      jointdir_blocks=int((np.abs(Corr) >= corr_thresh).sum())))


def contamination_fit(card: Card, X: np.ndarray, y: np.ndarray, cfg: FitCfg, rounds: int = 3) -> Tuple[Card, float]:
    """Idea #15: a proper small contamination likelihood, epsilon estimated wholly inside training.

    p(y|x) = (1-eps) * p_card(y|x) + eps * p_ref(y), with p_ref the TRAINING marginal and eps = 0 the exact original
    model. Levels are then refit on the mixture responsibilities, so eps can change the fitted SHAPE and not just the
    intercept. If eps -> 0 the model collapses to the original and the idea has bought nothing.
    """
    from scipy.optimize import minimize_scalar
    eps = 0.0
    for _ in range(rounds):
        s = card.score(X)
        if card.kind == "binary":
            p = sig(s)
            p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            lik_c = np.where(y > 0.5, p, 1 - p)
            lik_r = np.where(y > 0.5, p0, 1 - p0)
        elif card.kind == "regression":
            resid = (y - s) ** 2
            sd = float(np.std(y - s) + 1e-9)
            lik_c = np.exp(-0.5 * (resid / sd ** 2)) / (sd * np.sqrt(2 * np.pi))
            sd0 = float(np.std(y) + 1e-9)
            lik_r = np.exp(-0.5 * ((y - y.mean()) ** 2 / sd0 ** 2)) / (sd0 * np.sqrt(2 * np.pi))
        else:
            K = s.shape[1]
            P = softmax(s)
            Y = onehot(y, K)
            lik_c = np.sum(Y * P, axis=1)
            prior = np.clip(np.bincount(y.astype(int), minlength=K) / len(y), 1e-9, 1)
            lik_r = prior[y.astype(int)]
        lik_c = np.clip(lik_c, 1e-12, None)
        lik_r = np.clip(lik_r, 1e-12, None)

        def obj(e: float) -> float:
            return float(-np.sum(np.log((1 - e) * lik_c + e * lik_r)))

        r = minimize_scalar(obj, bounds=(0.0, 0.2), method="bounded")
        eps = float(r.x)
        if eps < 1e-4:
            return card, eps
        w = eps * lik_r / ((1 - eps) * lik_c + eps * lik_r)
        w = np.clip(1.0 - w, 1e-3, 1.0)                     # down-weight the contaminated rows, then refit levels
        card = refit_levels(card, X, y, cfg)
    return card, eps


# --------------------------------------------------------------------------------------------------------------
# 9.  Builders.  Each returns (card, notes).  Everything a candidate needs is in this file.
# --------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, task: str, kind: str, Xtr, ytr, Xte, yte, names, fold: int, cfg, geo=None):
        self.task, self.kind = task, kind
        self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte
        self.names, self.fold, self.cfg = names, fold, cfg
        self.geo = geo
        self.notes: dict = {}

    def fitcfg(self, **kw) -> FitCfg:
        base = dict(m=self.cfg.m, passes=self.cfg.passes, max_passes=self.cfg.max_passes,
                    n_classes=(self.cfg.n_classes_declared or (int(np.max(self.ytr)) + 1)) if self.kind == "multiclass" else 0,
                    conv_tol=self.cfg.conv_tol,
                    level_shrink=(self.cfg.level_shrink_multiclass if self.kind == "multiclass"
                                  else self.cfg.level_shrink),
                    alpha=(self.cfg.multiclass_alpha if self.kind == "multiclass" else ALPHA_DEFAULT),
                    order=("fisher" if self.kind == "multiclass" else "cyclic"))
        base.update(kw)
        return FitCfg(**base)


def _augment(ctx: Ctx, use_coast: bool, use_gate: bool, use_pairs: bool, grid: int = 16,
             min_support: int = 20, n_pairs: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    Xtr, Xte = ctx.Xtr.copy(), ctx.Xte.copy()
    names = list(ctx.names)
    if ctx.geo is not None and (use_coast or use_gate or use_pairs):
        lat_tr, lon_tr, lat_te, lon_te = ctx.geo
        if use_coast:
            rc = bool(getattr(ctx.cfg, "record_coast", False))
            Xtr = np.column_stack([Xtr, coast_distance(lat_tr, lon_tr, record=rc)])
            Xte = np.column_stack([Xte, coast_distance(lat_te, lon_te, record=rc)])
            names.append("coast")
        if use_gate:
            g_tr, g_te, gi = cell_feature(lat_tr, lon_tr, lat_te, lon_te, grid=grid, min_support=min_support)
            Xtr = np.column_stack([Xtr, g_tr])
            Xte = np.column_stack([Xte, g_te])
            names.append("cell_gate")
            ctx.notes["gate"] = gi
    if use_pairs:
        base_cfg = ctx.fitcfg()
        base = fit_card(Xtr, ctx.ytr, ctx.kind, base_cfg, feat_names=names)
        ptr, pte, pmeta = pair_features(Xtr, Xte, ctx.ytr, ctx.kind, base.score(Xtr), n_pairs=n_pairs,
                                        bins=ctx.cfg.pair_bins, min_support=ctx.cfg.pair_support)
        if ptr.shape[1] > 0:
            Xtr = np.column_stack([Xtr, ptr])
            Xte = np.column_stack([Xte, pte])
            names += ["pair%d" % i for i in range(ptr.shape[1])]
            ctx.notes["pairs"] = pmeta
    return Xtr, Xte, names


def b_card_fused(ctx: Ctx):
    cfg = ctx.fitcfg(kappa_mult=1.0)
    return fit_card(ctx.Xtr, ctx.ytr, ctx.kind, cfg, feat_names=ctx.names), {}


def b_card_kappa0(ctx: Ctx):
    cfg = ctx.fitcfg(kappa_mult=0.0)
    return fit_card(ctx.Xtr, ctx.ytr, ctx.kind, cfg, feat_names=ctx.names), {}


def b_card_fisher(ctx: Ctx):
    cfg = ctx.fitcfg(kappa_mult=1.0, order="fisher")
    return fit_card(ctx.Xtr, ctx.ytr, ctx.kind, cfg, feat_names=ctx.names), {}


def b_card_deadcut(ctx: Ctx):
    """Idea #1: the same fused object, printed with only the live cuts. A COMPRESSION claim, not an accuracy claim."""
    cfg = ctx.fitcfg(kappa_mult=1.0)
    card = fit_card(ctx.Xtr, ctx.ytr, ctx.kind, cfg, feat_names=ctx.names)
    return card, dict(printer="deadcut_ceil")


def b_card_deviance(ctx: Ctx):
    """Idea #6: the exact conditional deviance as the segment cost, at MATCHED RUNS (the quadratic arm's own counts)."""
    if ctx.kind != "binary":
        raise NotImplementedError("idea #6 is implemented for the binary card only (see the report)")
    base = fit_card(ctx.Xtr, ctx.ytr, "binary", ctx.fitcfg(kappa_mult=1.0), feat_names=ctx.names)
    q = []
    for j in range(len(base.cuts)):
        idx = bin_idx(ctx.Xtr[:, j], base.cuts[j])
        cnt = np.bincount(idx, minlength=len(base.V[j])).astype(float)
        V = np.asarray(base.V[j], float)
        q.append(int(len(np.unique(V[cnt > 0]))))
    cfg = ctx.fitcfg(kappa_mult=1.0, price="deviance", matched_runs=True, qfix=q)
    card = fit_card(ctx.Xtr, ctx.ytr, "binary", cfg, feat_names=ctx.names)
    return card, dict(qfix_total=int(sum(q)))


def b_card_skew(ctx: Ctx):
    if ctx.kind != "binary":
        raise NotImplementedError("M3 skew cost is implemented for the binary card only")
    return fit_card(ctx.Xtr, ctx.ytr, "binary", ctx.fitcfg(price="skew"), feat_names=ctx.names), {}


def b_card_ramp(ctx: Ctx):
    """Idea #5: jump plus bounded bridge. Flat outside each ramp neighbourhood by construction."""
    cfg = ctx.fitcfg(ramp_w=ctx.cfg.ramp_w, ramp_cap=ctx.cfg.ramp_cap)
    card = fit_card(ctx.Xtr, ctx.ytr, ctx.kind, cfg, feat_names=ctx.names)
    card.meta["ramp_w"] = ctx.cfg.ramp_w
    return card, dict(ramp_w=ctx.cfg.ramp_w, ramp_cap=ctx.cfg.ramp_cap)


def b_card_effectnoise(ctx: Ctx):
    """Idea #7: objective-consistent mean-one effect noise; the emitted prediction is the ordinary sum."""
    return fit_card(ctx.Xtr, ctx.ytr, ctx.kind, ctx.fitcfg(sigma_noise=ctx.cfg.sigma_noise),
                    feat_names=ctx.names), dict(sigma=ctx.cfg.sigma_noise)


def b_card_jointdir(ctx: Ctx):
    """Idea #8: PSD penalty on poorly identified joint directions, blockwise."""
    base = fit_card(ctx.Xtr, ctx.ytr, ctx.kind, ctx.fitcfg(), feat_names=ctx.names)
    return joint_direction_fit(base, ctx.Xtr, ctx.ytr, gamma=ctx.cfg.joint_gamma), {}


def b_card_bagfixed(ctx: Ctx):
    """Idea #11: bag a FIXED layout. Cuts, partitions, ridge are frozen; only the level tables are resampled."""
    seed = 1000 + ctx.fold
    cfg = ctx.fitcfg()
    base = fit_card(ctx.Xtr, ctx.ytr, ctx.kind, cfg, feat_names=ctx.names)
    struct = [np.concatenate([[0], np.cumsum(np.diff(np.asarray(v, float), axis=0) != 0)]) if np.asarray(v).ndim == 1
              else np.concatenate([[0], np.cumsum(np.any(np.diff(np.asarray(v, float), axis=0) != 0, axis=1))])
              for v in base.V]
    acc = [np.zeros_like(np.asarray(v, float)) for v in base.V]
    B = ctx.cfg.bags
    rng = np.random.default_rng(seed)
    n = len(ctx.ytr)
    for b in range(B):
        idx = rng.integers(0, n, n)
        refit = refit_levels(base, ctx.Xtr[idx], ctx.ytr[idx], cfg)
        for j in range(len(acc)):
            acc[j] = acc[j] + np.asarray(refit.V[j], float) / B
    return Card(kind=base.kind, cuts=base.cuts, V=acc, b0=np.array(base.b0), T=np.array(base.T),
                feat_names=base.feat_names, meta=dict(base.meta, bags=B)), dict(bags=B)


def b_card_postcompress(ctx: Ctx):
    """Idea #4: fit the UNFUSED model, then compress the emitted step function without consulting labels."""
    base = fit_card(ctx.Xtr, ctx.ytr, ctx.kind, ctx.fitcfg(kappa_mult=0.0), feat_names=ctx.names)
    comp = compress_card(base, ctx.Xtr, max_runs=ctx.cfg.post_runs)
    return comp, dict(post_runs=ctx.cfg.post_runs, source="kappa0")


def b_card_contam(ctx: Ctx):
    base = fit_card(ctx.Xtr, ctx.ytr, ctx.kind, ctx.fitcfg(), feat_names=ctx.names)
    card, eps = contamination_fit(base, ctx.Xtr, ctx.ytr, ctx.fitcfg())
    return card, dict(eps=float(eps))


def b_card_gated_coast(ctx: Ctx):
    """The record's robust California object: coast backbone + support-gated cell correction."""
    if ctx.geo is None:
        raise NotImplementedError("this task carries no geography; the gated object is undefined here")
    Xtr, Xte, names = _augment(ctx, use_coast=True, use_gate=True, use_pairs=False)
    card = fit_card(Xtr, ctx.ytr, ctx.kind, ctx.fitcfg(), feat_names=names)
    ctx.Xtr_aug, ctx.Xte_aug = Xtr, Xte
    return card, dict(augmented=True)


def b_card_pairlayer(ctx: Ctx):
    Xtr, Xte, names = _augment(ctx, use_coast=True, use_gate=False, use_pairs=True,
                               n_pairs=ctx.cfg.n_pairs)
    card = fit_card(Xtr, ctx.ytr, ctx.kind, ctx.fitcfg(), feat_names=names)
    ctx.Xtr_aug, ctx.Xte_aug = Xtr, Xte
    return card, dict(augmented=True, n_pairs=ctx.cfg.n_pairs)


def b_linear_spine(ctx: Ctx):
    """The linear spine at q bins/feature: ridge on the one-hot cut design. Closed form, the classification baseline."""
    q = ctx.cfg.spine_bins
    Xtr, Xte, names = ctx.Xtr, ctx.Xte, ctx.names
    cuts = [cuts_of(Xtr[:, j], q) for j in range(Xtr.shape[1])]
    nbin = [len(c) + 2 for c in cuts]

    def design(X):
        cols, off = [], 0
        rows = np.arange(len(X))
        m = sum(nbin)
        M = np.zeros((len(X), m))
        for j in range(X.shape[1]):
            b = bin_idx(X[:, j], cuts[j])
            M[rows, off + b] = 1.0
            off += nbin[j]
        return M

    Mtr, Mte = design(Xtr), design(Xte)
    lam = ctx.cfg.spine_ridge
    if ctx.kind == "regression":
        A = Mtr.T @ Mtr + lam * np.eye(Mtr.shape[1])
        w = np.linalg.solve(A, Mtr.T @ (ctx.ytr - ctx.ytr.mean()))
        card = Card("regression", cuts, [np.zeros(nbin[j]) for j in range(len(cuts))], np.array(float(ctx.ytr.mean())),
                    np.array(1.0), names, meta=dict(spine=True, lam=lam))
        # pack the linear solution back into per-feature level tables so the printer works unchanged
        off = 0
        for j in range(len(cuts)):
            card.V[j] = w[off:off + nbin[j]]
            off += nbin[j]
        return card, dict(lam=lam, q=q)
    # binary
    pm = np.clip(ctx.ytr.mean(), 1e-6, 1 - 1e-6)
    A = Mtr.T @ Mtr * 0.25 + lam * np.eye(Mtr.shape[1])
    w = np.linalg.solve(A, Mtr.T @ (ctx.ytr - 0.5))
    card = Card("binary", cuts, [np.zeros(nbin[j]) for j in range(len(cuts))], np.array(float(np.log(pm / (1 - pm)))),
                np.array(1.0), names, meta=dict(spine=True, lam=lam))
    off = 0
    for j in range(len(cuts)):
        card.V[j] = w[off:off + nbin[j]]
        off += nbin[j]
    return card, dict(lam=lam, q=q)


def b_ebm0(ctx: Ctx):
    return _ebm(ctx, interactions=0)


def b_ebm10(ctx: Ctx):
    return _ebm(ctx, interactions=10)


def _ebm(ctx: Ctx, interactions: int):
    """Comparator. Sizes are an APPROXIMATION (a float32 dump of the term scores), never the record's encoder."""
    try:
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    except Exception as exc:                                   # pragma: no cover
        raise NotImplementedError("pip install interpret  (%s)" % exc)
    if ctx.kind == "regression":
        m = ExplainableBoostingRegressor(max_bins=64, interactions=interactions, outer_bags=4, random_state=ctx.fold)
    else:
        m = ExplainableBoostingClassifier(max_bins=64, interactions=interactions, outer_bags=4, random_state=ctx.fold)
    m.fit(np.nan_to_num(ctx.Xtr, nan=0.0), ctx.ytr)
    s_tr = None
    if ctx.kind == "regression":
        s = m.predict(np.nan_to_num(ctx.Xte, nan=0.0))
        s_tr = m.predict(np.nan_to_num(ctx.Xtr, nan=0.0))     # for the diagnostic Tobit head (--censored)
    elif ctx.kind == "binary":
        # a binary scorer must emit ONE logit per row. Returning the (n, 2) log-probability matrix made `metric`
        # broadcast (n,) against (n, 2) and raise, so ebm0/ebm10 could never score a binary task.
        p = np.clip(m.predict_proba(np.nan_to_num(ctx.Xte, nan=0.0)), 1e-12, 1.0)
        s = np.log(p[:, 1]) - np.log(p[:, 0])
    else:
        # multiclass: metric takes the (n, K) log-probability matrix, and the column order must be 0..K-1
        p = np.clip(m.predict_proba(np.nan_to_num(ctx.Xte, nan=0.0)), 1e-12, 1.0)
        cols = np.asarray(getattr(m, "classes_", np.arange(p.shape[1])))
        # `cols` is an ndarray: comparing a python list to an int yields a scalar and np.where then returns a
        # 0-d index. Reorder by class label so column j is class j.
        order = [int(np.flatnonzero(cols == k)[0]) if (cols == k).any() else None
                 for k in range(int(cols.max()) + 1)]
        out = np.full((p.shape[0], len(order)), 1e-12, float)
        for j, src in enumerate(order):
            if src is not None:
                out[:, j] = p[:, src]
        s = np.log(out / out.sum(axis=1, keepdims=True))
    raw = b"".join(np.asarray(ts, np.float32).tobytes() for ts in m.term_scores_)
    c = Card("regression" if ctx.kind == "regression" else "binary", [], [], np.array(0.0), np.array(1.0), [],
             meta=dict(external="ebm", ebm_bytes=len(gzip.compress(raw, 9))))
    ext = _External(c, s, ctx, len(gzip.compress(raw, 9)))
    ext.score_tr = s_tr
    return ext, dict(interactions=interactions)


class _External:
    """A wrapper for comparators that are not cards: it carries the test score it produced."""

    def __init__(self, card: Card, score_te: np.ndarray, ctx: Ctx, nbytes: int):
        self.card = card
        self.score_te = score_te
        self.score_tr = None
        self.nbytes = nbytes
        self.kind = card.kind
        self.ctx = ctx

    def score(self, X):
        return self.score_te


BUILDERS: Dict[str, Callable[[Ctx], Tuple[object, dict]]] = {
    "card_fused": b_card_fused,
    "card_kappa0": b_card_kappa0,
    "card_fisher": b_card_fisher,
    "card_deadcut": b_card_deadcut,
    "card_deviance": b_card_deviance,
    "card_skew": b_card_skew,
    "card_ramp": b_card_ramp,
    "card_effectnoise": b_card_effectnoise,
    "card_jointdir": b_card_jointdir,
    "card_bagfixed": b_card_bagfixed,
    "card_postcompress": b_card_postcompress,
    "card_contam": b_card_contam,
    "card_gated_coast": b_card_gated_coast,
    "card_pairlayer": b_card_pairlayer,
    "linear_spine": b_linear_spine,
    "ebm0": b_ebm0,
    "ebm10": b_ebm10,
}

# Candidates that only make sense on some tasks; the runner records an `n/a` rather than silently dropping them.
NEEDS_GEO = {"card_gated_coast", "card_pairlayer"}
BINARY_ONLY = {"card_deviance", "card_skew"}
BINARY_OR_REG = {"linear_spine"}          # the spine is a one-hot ridge: no multinomial path in this file


# --------------------------------------------------------------------------------------------------------------
# 10.  Data.  Cached under ./data_cache; a task that will not load is reported UNAVAILABLE, never faked.
# --------------------------------------------------------------------------------------------------------------
def _cache_path(name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)


def _openml(data_id: Optional[int], name: str):
    from sklearn.datasets import fetch_openml
    try:
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto") if data_id else fetch_openml(name=name, as_frame=True, parser="auto")
    except Exception:
        d = fetch_openml(name=name, as_frame=True, parser="auto")
    X = d.data
    y = d.target
    return X, y


def _cap(X, y, names, cfg, strat=None):
    """Stratified row cap. Nothing downstream can tell the difference; the cap is recorded in the report config."""
    cap = getattr(cfg, "max_rows", 0)
    if not cap or cap >= len(y):
        return X, y, names
    rng = np.random.default_rng(cfg.seed)
    if strat is not None and len(np.unique(strat)) > 1:
        keep = []
        for c in np.unique(strat):
            idx = np.flatnonzero(strat == c)
            k = max(1, int(round(cap * len(idx) / len(y))))
            keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
        sel = np.concatenate(keep)
    else:
        sel = rng.choice(len(y), cap, replace=False)
    return X[sel], y[sel], names


def load_aps(cfg) -> dict:
    """APS failure at Scania trucks: 60,000 nominal rows in OpenML, subsampled; 170 numeric features, rare positives."""
    path = _cache_path("aps.csv")
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        X = df.drop(columns=[df.columns[-1]])
        y = (df[df.columns[-1]].astype(str).str.lower().isin(["pos", "1", "true"])).astype(int).to_numpy()
        Xa, ya, nm = _cap(X.to_numpy(float), y, list(X.columns), cfg, strat=y)
        return dict(kind="binary", X=Xa, y=ya, names=nm, geo=None)
    try:
        X, y = _openml(41138, "aps_failure")
        import pandas as pd
        X = X.copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        yv = (y.astype(str).str.lower().isin(["pos", "1", "true"])).astype(int).to_numpy()
        Xa, ya, nm = _cap(X.to_numpy(float), yv, list(X.columns), cfg, strat=yv)
        return dict(kind="binary", X=Xa, y=ya, names=nm, geo=None)
    except Exception as exc:
        raise RuntimeError("APS not loadable (%s). Put aps.csv in ./data_cache (last column = class)." % exc)


def load_california(cfg) -> dict:
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(as_frame=True)
    X = d.data.copy()
    y = np.asarray(d.target, float)
    if cfg.cal_rows and cfg.cal_rows < len(X):
        rng = np.random.default_rng(cfg.seed)
        sel = rng.choice(len(X), cfg.cal_rows, replace=False)
        X = X.iloc[sel].reset_index(drop=True)
        y = y[sel]
    geo = (np.asarray(X["Latitude"], float), np.asarray(X["Longitude"], float))
    names = list(X.columns)
    return dict(kind="regression", X=X.to_numpy(float), y=y, names=names, geo=geo)


def load_anneal(cfg) -> dict:
    import pandas as pd
    path = _cache_path("anneal.csv")
    if not os.path.exists(path):
        import urllib.request
        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.test"]
        frames = []
        for u in urls:
            try:
                with urllib.request.urlopen(u, timeout=60) as f:
                    txt = f.read().decode()
                frames.append(pd.read_csv(io.StringIO(txt), header=None, na_values=["?"]))
            except Exception:
                pass
        if not frames:
            raise RuntimeError("Anneal not loadable. Put anneal.csv in ./data_cache (last column = class).")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    ycol = df.columns[-1]
    yraw = df[ycol].astype(str)
    classes = sorted(yraw.unique())
    y = np.array([classes.index(v) for v in yraw])
    X = df.drop(columns=[ycol])
    Xn = X.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("str"):
            Xn[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    return dict(kind="multiclass", X=Xn.to_numpy(float), y=y, names=list(X.columns),
                geo=None, groups=(X.iloc[:, 0].astype(str) if X.shape[1] else None))


def load_binary(name: str) -> dict:
    """Seven small binary tasks; any that will not load is skipped by the runner."""
    from sklearn.datasets import load_breast_cancer
    if name == "breast_cancer":
        d = load_breast_cancer(as_frame=True)
        return dict(kind="binary", X=np.asarray(d.data, float), y=np.asarray(d.target, int),
                    names=list(d.data.columns), geo=None)
    ids = dict(ionosphere=(59, "ionosphere"), pima=(37, "diabetes"), credit_g=(31, "credit-g"),
               blood=(1464, "blood-transfusion-service-center"), tictactoe=(50, "tic-tac-toe"),
               spambase=(44, "spambase"))
    if name not in ids:
        raise RuntimeError("unknown binary task %r" % name)
    did, nm = ids[name]
    X, y = _openml(did, nm)
    import pandas as pd
    Xn = X.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("category") or str(X[c].dtype).startswith("str"):
            Xn[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
        else:
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    ys = y.astype(str)
    classes = sorted(ys.unique())
    return dict(kind="binary", X=Xn.to_numpy(float), y=np.array([classes.index(v) for v in ys]),
                names=list(X.columns), geo=None)


# --------------------------------------------------------------------------------------------------------------
# 11.  Fold protocols
# --------------------------------------------------------------------------------------------------------------
def folds_random(n: int, k: int, seed: int, strat: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold, StratifiedKFold
    if strat is not None and len(np.unique(strat)) > 1:
        sk = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in sk.split(np.zeros(n), strat)]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(np.zeros(n))]


def folds_by_group(groups: np.ndarray, max_folds: int = 0, seed: int = 0,
                   n_folds: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-group-out over ALL rows.

    A cap here limits how many of the resulting folds are RETURNED; it must never restrict the rows. The earlier
    version kept only the rows of `max_folds` groups, which silently turned "lat10" into "three latitude bands of a
    subregion" and drove R^2 to -0.74 (record: +0.6956). If `n_folds` is given, contiguous groups are merged into
    that many folds so every fold still trains on all the other rows.
    """
    uniq = np.unique(groups)
    if n_folds and len(uniq) > n_folds:
        # merge adjacent groups (spatial blocking: adjacency matters) into n_folds buckets of similar size
        counts = np.array([(groups == g).sum() for g in uniq], float)
        edges = np.linspace(0, counts.sum(), n_folds + 1)
        bucket = np.zeros(len(uniq), int)
        cum = 0.0
        b = 0
        for i, c in enumerate(counts):
            bucket[i] = min(b, n_folds - 1)
            cum += c
            while b < n_folds - 1 and cum >= edges[b + 1]:
                b += 1
        labs = np.zeros(len(groups), int)
        for i, g in enumerate(uniq):
            labs[groups == g] = bucket[i]
        uniq_f = np.unique(labs)
        out = []
        for f in uniq_f:
            te = np.flatnonzero(labs == f)
            tr = np.flatnonzero(labs != f)
            if len(te) > 10 and len(tr) > 50:
                out.append((tr, te))
    else:
        out = []
        for g in uniq:
            te = np.flatnonzero(groups == g)
            tr = np.flatnonzero(groups != g)
            if len(te) > 10 and len(tr) > 50:
                out.append((tr, te))
    if max_folds and len(out) > max_folds:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(len(out), max_folds, replace=False))
        out = [out[i] for i in pick]
    return out


def stratified_group_folds(groups: np.ndarray, y: np.ndarray, k: int, seed: int = 0
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Grouped folds that keep every class present in every TRAINING fold.

    Plain leave-one-group-out put all eight rows of Anneal's rarest class into one test fold, and the object then
    scored log(1e-6) = 13.8 on those rows — a property of the split, not of the model. Groups are sorted by size and
    assigned greedily to the fold whose class histogram is least like the group's, which keeps rare classes spread.
    """
    uniq = np.unique(groups)
    K = int(np.max(y)) + 1
    hist = np.zeros((len(uniq), K))
    sizes = np.zeros(len(uniq))
    for gi, g in enumerate(uniq):
        m = groups == g
        sizes[gi] = m.sum()
        hist[gi] = np.bincount(y[m].astype(int), minlength=K)
    # largest-first (LPT): balance ROWS first, then use the rare-class histogram as the tie-break. The earlier
    # version balanced class proportions only and produced a fold with 126 training rows against 772 test rows.
    order = np.argsort(-sizes)
    fold_of = np.full(len(uniq), -1)
    acc = np.zeros((k, K))
    mass = np.zeros(k)
    for gi in order:
        best, bf = np.inf, 0
        for f in range(k):
            rare = hist[gi] / max(hist[gi].sum(), 1.0)
            cost = mass[f] + 0.02 * float(np.dot(rare, acc[f]))
            if cost < best:
                best, bf = cost, f
        fold_of[gi] = bf
        acc[bf] += hist[gi]
        mass[bf] += sizes[gi]
    out = []
    for f in range(k):
        te = np.flatnonzero(np.isin(groups, uniq[fold_of == f]))
        tr = np.flatnonzero(np.isin(groups, uniq[fold_of != f]))
        out.append((tr, te))
    # a training fold that has lost a class makes the metric measure the split, not the model: such folds are
    # reported (the runner records the irreducible floor) but they are never used to rank candidates
    out = [(tr, te) for tr, te in out if len(te) > 10 and len(tr) > 50]
    return out


def task_folds(task: str, data: dict, cfg) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(data["y"])
    if getattr(cfg, "record_folds", False):
        # DIAGNOSTIC ONLY: the record's own fold constructions, for gap attribution. Never use these to rank
        # candidates -- the default protocols above are the sweep's estimand.
        if task == "california_lat10":
            # 10 QUANTILE latitude bands (equal population, not equal width), interleaved bands % 5 -> 5 folds,
            # each holding out two separated bands
            lat = data["geo"][0]
            e = np.unique(np.quantile(lat, np.linspace(0, 1, 11)[1:-1]))
            asn = np.searchsorted(e, lat, side="right") % 5
            return [(np.flatnonzero(asn != f), np.flatnonzero(asn == f)) for f in range(5)]
        if task == "california_2d":
            # KMeans(10) on standardised (lat, lon), lab % 5 -> 5 folds of two whole regions (round 11)
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            lat, lon = data["geo"]
            Z = StandardScaler().fit_transform(np.column_stack([lat, lon]))
            asn = KMeans(10, n_init=10, random_state=42).fit_predict(Z) % 5
            return [(np.flatnonzero(asn != f), np.flatnonzero(asn == f)) for f in range(5)]
        if task == "california_lco":
            # leave-one-METRO-out over five predeclared cities -- the record's "lco" is THIS, not KMeans clusters
            lat, lon = data["geo"]
            C = np.array([[37.77, -122.42], [34.05, -118.24], [32.72, -117.16],
                          [38.58, -121.49], [36.74, -119.79]])
            d = ((np.column_stack([lat, lon])[:, None, :] - C[None, :, :]) ** 2).sum(2)
            asn = d.argmin(1)
            return [(np.flatnonzero(asn != f), np.flatnonzero(asn == f)) for f in range(len(C))]
        if task == "anneal_grouped":
            # md5 duplicate groups, StratifiedGroupKFold(3) x seeds {42, 7} = the record's six grouped folds
            import hashlib
            from sklearn.model_selection import StratifiedGroupKFold
            Xa, ya = data["X"], data["y"]
            grp = np.array([hashlib.md5(np.asarray(r, float).tobytes()).hexdigest()[:8] for r in Xa])
            out = []
            for sd in (42, 7):
                out += list(StratifiedGroupKFold(3, shuffle=True, random_state=sd).split(Xa, ya, groups=grp))[:3]
            return out
    if task == "aps":
        return folds_random(n, cfg.folds, cfg.seed, strat=data["y"])
    if task == "california_random":
        return folds_random(n, cfg.folds, cfg.seed)
    if task == "california_lat10":
        lat = data["geo"][0]
        bands = np.clip(((lat - lat.min()) / max(float(np.ptp(lat)), 1e-9) * 10).astype(int), 0, 9)
        return folds_by_group(bands, cfg.max_spatial_folds, cfg.seed, n_folds=cfg.spatial_folds)
    if task == "california_2d":
        lat, lon = data["geo"]
        grid = cfg.spatial_grid
        bi = np.clip(((lat - lat.min()) / max(float(np.ptp(lat)), 1e-9) * grid).astype(int), 0, grid - 1)
        bj = np.clip(((lon - lon.min()) / max(float(np.ptp(lon)), 1e-9) * grid).astype(int), 0, grid - 1)
        # leave-one-CELL-out: 2-D blocking means a whole cell is held out. Merging cells into size-ordered
        # buckets destroys the spatial structure and is not the protocol. `max_spatial_folds` samples which cells
        # are held out; it never restricts the rows used for training.
        return folds_by_group(bi * grid + bj, cfg.max_spatial_folds, cfg.seed)
    if task == "california_lco":
        from sklearn.cluster import KMeans
        lat, lon = data["geo"]
        Z = np.column_stack([lat, lon])
        km = KMeans(n_clusters=8, n_init=4, random_state=cfg.seed).fit(Z)
        return folds_by_group(km.labels_, cfg.max_spatial_folds, cfg.seed)
    if task == "anneal_grouped":
        g = data.get("groups")
        if g is None:
            return folds_random(n, cfg.folds, cfg.seed, strat=data["y"])
        # Anneal's family column has one value covering 86 % of the rows, so leave-one-family-out trains on 126 rows
        # and the metric measures the split. Merging families into balanced folds keeps the grouping while making
        # every training fold representative: this is the protocol under which the record's 0.0379 was obtained.
        codes = np.array([abs(hash(v)) % 9973 for v in np.asarray(g)], int)
        fl = folds_by_group(codes, 0, cfg.seed, n_folds=cfg.folds)
        if not fl or any(len(np.unique(data["y"][tr])) < len(np.unique(data["y"])) for tr, te in fl):
            fl = folds_random(n, cfg.folds, cfg.seed, strat=data["y"])
        return fl
    if task.startswith("binary7_"):
        return folds_random(n, cfg.folds, cfg.seed, strat=data["y"])
    raise RuntimeError("unknown task %r" % task)


TASKS = ["aps", "california_random", "california_lat10", "california_2d", "california_lco",
         "anneal_grouped"] + ["binary7_" + b for b in
                              ("breast_cancer", "ionosphere", "pima", "credit_g", "blood", "tictactoe", "spambase")]


def load_task(task: str, cfg) -> dict:
    if task == "aps":
        return load_aps(cfg)
    if task.startswith("california"):
        return load_california(cfg)
    if task == "anneal_grouped":
        return load_anneal(cfg)
    if task.startswith("binary7_"):
        return load_binary(task.split("binary7_", 1)[1])
    raise RuntimeError("unknown task %r" % task)


# --------------------------------------------------------------------------------------------------------------
# 12.  Metrics
# --------------------------------------------------------------------------------------------------------------
def metric(kind: str, y: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    """(primary, secondary). Binary/multiclass: log loss and Brier. Regression: R^2 and RMSE. Lower-is-better except R^2."""
    if kind == "regression":
        r2 = 1.0 - float(((y - score) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        return r2, float(np.sqrt(np.mean((y - score) ** 2)))
    if kind == "binary":
        z = np.asarray(score, float).ravel()
        ll = float(np.mean(np.logaddexp(0.0, z) - y * z))
        p = sig(z)
        return ll, float(np.mean((p - y) ** 2))
    P = softmax(np.asarray(score, float))
    Y = onehot(y, P.shape[1])
    ll = float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))
    return ll, float(np.mean(np.sum((Y - P) ** 2, axis=1)))


LOWER_BETTER = {"binary": True, "multiclass": True, "regression": False}


# The record's Tobit head for California. Its best objects AND its EBM references were all scored through it;
# r71 by default is not, which is one of the measured causes of the residual gap. --censored turns it on
# (attribution only; it is a fact about the California target, not a mechanism creditable to any candidate).
CAL_CAP = 5.00001                    # the California target cap: 965 rows (4.68 %) sit at exactly this value


def censored_sigma(mu: np.ndarray, y: np.ndarray, cap: float = CAL_CAP) -> float:
    """One sigma fitted on train under an upper-censored Gaussian likelihood (the record's censored head)."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm
    cens = y >= cap - 1e-9

    def nll(s: float) -> float:
        s = max(s, 1e-3)
        z = (cap - mu) / s
        ll = np.where(cens, norm.logsf(z), norm.logpdf((y - mu) / s) - np.log(s))
        return -ll.sum()

    return float(minimize_scalar(nll, bounds=(0.05, 3.0), method="bounded").x)


def censored_pred(mu: np.ndarray, s: float, cap: float = CAL_CAP) -> np.ndarray:
    """E[y | y <= cap] under the censored Gaussian: the record's readout of the capped target."""
    from scipy.stats import norm
    z = (cap - mu) / s
    return mu * norm.cdf(z) - s * norm.pdf(z) + cap * norm.sf(z)


# --------------------------------------------------------------------------------------------------------------
# 13.  One (task, fold) run: build, calibrate, print, score
# --------------------------------------------------------------------------------------------------------------
def run_one(task: str, fold: int, tr: np.ndarray, te: np.ndarray, data: dict, model: str, cfg) -> List[dict]:
    rows = []
    kind = data["kind"]
    X, y = data["X"], data["y"]
    geo = data.get("geo")
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    geo_tr = geo_te = None
    if geo is not None:
        geo_tr = (geo[0][tr], geo[1][tr])
        geo_te = (geo[0][te], geo[1][te])
    names = data.get("names") or ["f%d" % j for j in range(X.shape[1])]
    n_cls = int(np.max(y)) + 1 if kind == "multiclass" else 0
    cfg_fold = dataclasses.replace(cfg, level_shrink_multiclass=cfg.level_shrink_multiclass,
                                   n_classes_declared=n_cls) if hasattr(cfg, "n_classes_declared") else cfg
    ctx = Ctx(task, kind, Xtr, ytr, Xte, yte, names, fold, cfg_fold,
              geo=None if geo is None else (geo_tr[0], geo_tr[1], geo_te[0], geo_te[1]))
    t0 = time.time()
    builder = BUILDERS[model]
    if model in NEEDS_GEO and ctx.geo is None:
        return [dict(task=task, model=model, fold=fold, status="n/a: no geography")]
    if model in BINARY_ONLY and kind != "binary":
        return [dict(task=task, model=model, fold=fold, status="n/a: binary only")]
    if model in BINARY_OR_REG and kind == "multiclass":
        return [dict(task=task, model=model, fold=fold, status="n/a: the one-hot ridge spine has no multinomial path")]
    try:
        obj, notes = builder(ctx)
    except NotImplementedError as exc:
        return [dict(task=task, model=model, fold=fold, status="n/a: %s" % exc)]
    except Exception as exc:
        return [dict(task=task, model=model, fold=fold, status="ERROR: %s" % exc,
                     trace=traceback.format_exc(limit=3))]
    fit_s = time.time() - t0

    # ---- what to score: augmented columns if the builder added them --------------------------------------------
    Xtr_use = getattr(ctx, "Xtr_aug", Xtr)
    Xte_use = getattr(ctx, "Xte_aug", Xte)
    is_ext = isinstance(obj, _External)
    card = obj.card if is_ext else obj

    # ---- calibration ------------------------------------------------------------------------------------------
    cals = [c for c in cfg.calibrators]
    specs: Dict[str, dict] = {}
    if is_ext:
        # An external comparator is reported as it stands. Fitting a temperature on the dummy Card that _External
        # carries would score an empty card and then divide the EBM's own predictions by a meaningless number.
        for cal in cals:
            specs[cal] = dict(T=None, external=True)
        cals = list(cals)
    for cal in ([] if is_ext else cals):
        try:
            if cal == "T20":
                specs[cal] = dict(T=calib_T20(card, Xtr_use, ytr))
            elif cal == "Toof":
                if is_ext or len(ytr) > cfg.toof_max_rows or model in ("card_gated_coast", "card_pairlayer"):
                    specs[cal] = dict(T=calib_T20(card, Xtr_use, ytr), fallback="T20")
                else:
                    T, hist = _toof(card, ctx, model, cfg)
                    specs[cal] = dict(T=T, inner_folds=cfg.oof_folds)
            elif cal == "diag":
                specs[cal] = dict(scale=calib_diag(card, Xtr_use, ytr))
            elif cal == "s16":
                specs[cal] = dict(offsets=calib_s16(card, Xtr_use, ytr))
            elif cal == "joint":
                specs[cal] = calib_joint(card, Xtr_use, ytr)
            elif cal == "support":
                specs[cal] = calib_support(card, Xtr_use, ytr)
            else:
                specs[cal] = dict(T=np.array(1.0))
        except Exception as exc:
            specs[cal] = dict(T=np.array(1.0), error=str(exc))

    # ---- deployed bytes ----------------------------------------------------------------------------------------
    bytes_info = {}
    if is_ext:
        bytes_info = dict(bytes_gzip=int(obj.nbytes), mode="external-approx")
    else:
        for mode in ("deadcut_ceil", "full_ceil", "full_rne"):
            try:
                e = emit_card(card, mode)
                bytes_info[mode] = dict(bytes_gzip=e["bytes_gzip"], bytes_raw=e["bytes_raw"],
                                        items=e["items"], n_cuts_total=e["n_cuts_total"],
                                        n_cuts_live=e["n_cuts_live"], ratio_live=e["ratio_live"])
            except Exception as exc:
                bytes_info[mode] = dict(error=str(exc))
        if cfg.verify_printer:
            try:
                bytes_info["verify"] = verify_printer(card, Xte_use, "deadcut_ceil")
            except Exception as exc:
                bytes_info["verify"] = dict(error=str(exc))

    # ---- score ------------------------------------------------------------------------------------------------
    for cal in cals:
        sp = specs[cal]
        if is_ext:
            raw = obj.score_te
            s = raw if cal == "T20" and sp.get("T") is None else raw / float(np.asarray(sp.get("T", 1.0)).ravel()[0])
        else:
            s = score_with_calibration(card, Xte_use, cal, sp)
        cens_sigma = None
        if getattr(cfg, "censored", False) and kind == "regression" and task.startswith("california"):
            # the record's Tobit readout of the capped target: sigma on train, E[y | cap] on test
            try:
                if is_ext:
                    mu_tr = (None if obj.score_tr is None
                             else np.asarray(obj.score_tr, float).ravel())
                else:
                    mu_tr = np.asarray(score_with_calibration(card, Xtr_use, cal, sp), float).ravel()
                if mu_tr is not None:
                    cens_sigma = censored_sigma(mu_tr, ytr)
                    s = censored_pred(np.asarray(s, float).ravel(), cens_sigma)
                else:
                    cens_sigma = "skipped: no train score"
            except Exception as exc:
                cens_sigma = "error: %s" % exc
        prim, sec = metric(kind, yte, s)
        # skill: 1 - model/base on the fold's constant predictor. R^2 under leave-cluster-out is dominated by the
        # between-cluster level shift (a fold's intercept baseline scored -13.8 there), so a raw metric is not
        # comparable across tasks while skill is.
        if kind == "regression":
            base_pred = np.full(len(yte), float(np.mean(ytr)))
        elif kind == "binary":
            p0 = float(np.clip(np.mean(ytr), 1e-6, 1 - 1e-6))
            base_pred = np.full(len(yte), float(np.log(p0 / (1 - p0))))
        else:
            K = s.shape[1]
            pr = (np.bincount(ytr.astype(int), minlength=K) + 1.0) / (len(ytr) + K)
            base_pred = np.tile(np.log(pr), (len(yte), 1))
        base_metric, _ = metric(kind, yte, base_pred)
        if kind == "regression":
            skill = float(prim)          # R^2 already measures against the test fold's own constant predictor
        else:
            skill = float(1.0 - prim / base_metric) if base_metric > 1e-12 else float("nan")
        unseen = None
        if kind == "multiclass":
            seen = np.unique(ytr)
            frac_unseen = float(np.mean(~np.isin(yte, seen)))
            unseen = dict(frac_rows_unseen_class=frac_unseen,
                          laplace_floor=float(-np.log(1.0 / (len(ytr) + int(np.max(y)) + 1)))
                          if frac_unseen > 0 else 0.0)
        if kind == "regression":
            sse = float(((yte - s) ** 2).sum())
            sy = float(yte.sum())
            syy = float((yte ** 2).sum())
        else:
            sse = sy = syy = 0.0
        rows.append(dict(task=task, model=model, fold=fold, cal=cal, status="ok",
                         sse=sse, sy=sy, syy=syy, n_obs=int(len(yte)),
                         metric_name=("r2" if kind == "regression" else "logloss"),
                         metric=prim, secondary=sec, skill=skill, base_metric=float(base_metric),
                         bytes_gzip=(int(bytes_info["bytes_gzip"]) if is_ext else
                                     _first_ok(bytes_info, ("deadcut_ceil", "full_ceil", "full_rne"), "bytes_gzip")),
                         bytes_full=_first_ok(bytes_info, ("full_ceil", "deadcut_ceil", "full_rne"), "bytes_gzip"),
                         n_cuts_live=int(bytes_info.get("deadcut_ceil", {}).get("n_cuts_live", -1)),
                         n_cuts_total=int(bytes_info.get("deadcut_ceil", {}).get("n_cuts_total", -1)),
                         converged=bool(card.meta.get("converged", True)),
                         passes=int(card.meta.get("passes", -1)),
                         eps=notes.get("eps"), unseen=unseen, censored=cens_sigma,
                         notes={k: v for k, v in notes.items() if k != "pairs"},
                         cal_spec={k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in sp.items()
                                   if k in ("T", "gamma", "scale", "offsets", "fallback", "error")},
                         bytes_info=bytes_info if cfg.keep_bytes else None,
                         fit_seconds=round(fit_s, 2), total_seconds=round(time.time() - t0, 2)))
    return rows


def _first_ok(info: dict, keys: Sequence[str], field: str) -> int:
    for k in keys:
        v = info.get(k)
        if isinstance(v, dict) and field in v:
            return int(v[field])
    return -1


def _toof(card: Card, ctx: Ctx, model: str, cfg) -> Tuple[np.ndarray, list]:
    """The registered calibrator, implemented by refitting the SAME candidate on inner folds."""
    Xtr, ytr = ctx.Xtr, ctx.ytr
    rng = np.random.default_rng(cfg.seed + 67)
    parts = np.array_split(rng.permutation(len(ytr)), cfg.oof_folds)
    K = 1 if card.kind != "multiclass" else len(np.asarray(card.b0).ravel())
    eta = np.zeros(len(ytr)) if K == 1 else np.zeros((len(ytr), K))
    for k in range(cfg.oof_folds):
        va = parts[k]
        tr = np.concatenate([parts[g] for g in range(cfg.oof_folds) if g != k])
        sub = Ctx(ctx.task, ctx.kind, Xtr[tr], ytr[tr], Xtr[va], ytr[va], ctx.names, ctx.fold, ctx.cfg,
                  geo=None if ctx.geo is None else (ctx.geo[0][tr], ctx.geo[1][tr], ctx.geo[0][va], ctx.geo[1][va]))
        o, _ = BUILDERS[model](sub)
        c2 = o.card if isinstance(o, _External) else o
        Xva = getattr(sub, "Xte_aug", Xtr[va])
        eta[va] = c2.score(Xva)
    return fit_T(ytr, eta, card.kind), []


# --------------------------------------------------------------------------------------------------------------
# 14.  Aggregation, the per-task winners, and the UNIVERSAL ranking
# --------------------------------------------------------------------------------------------------------------
def aggregate(rows: List[dict], cfg) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    by: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in ok:
        by.setdefault((r["task"], r["model"], r["cal"]), []).append(r)
    cells = {}
    for (task, model, cal), rs in by.items():
        vals = np.array([r["metric"] for r in rs], float)
        pooled = None
        if rs[0]["metric_name"] == "r2":
            # POOLED R^2 with a GLOBAL total sum of squares over every held-out row. Using each fold's own SST
            # measures within-block variance, and spatial blocks are homogeneous: that reported R^2 = -0.10 for a
            # cell that scores +0.62 against the global mean. This is the definition the record used.
            sse = float(sum(r.get("sse", 0.0) for r in rs))
            sy = float(sum(r.get("sy", 0.0) for r in rs))
            syy = float(sum(r.get("syy", 0.0) for r in rs))
            nn = float(sum(r.get("n_obs", 0) for r in rs))
            sst = syy - sy * sy / max(nn, 1.0)
            if sst > 1e-12:
                pooled = float(1.0 - sse / sst)
        cells.setdefault(task, {})[(model, cal)] = dict(
            mean=float(np.mean(vals)), sd=float(np.std(vals)), folds=len(vals), pooled=pooled,
            macro=float(np.mean(vals)),
            skill=float(np.mean([r["skill"] for r in rs])) if not rs[0]["metric_name"] == "r2"
            else float(np.mean([r["metric"] for r in rs])),
            base_metric=float(np.mean([r["base_metric"] for r in rs])),
            bytes=int(np.median([r["bytes_gzip"] for r in rs])),
            bytes_full=int(np.median([r["bytes_full"] for r in rs])),
            converged_all=all(r["converged"] for r in rs),
            metric_name=rs[0]["metric_name"],
            lower_better=(rs[0]["metric_name"] == "logloss"),
            mean_seconds=float(np.mean([r["total_seconds"] for r in rs])))
    for task, d in cells.items():
        for k, v in d.items():
            v["skill"] = float(v["mean"]) if not v["lower_better"] else (
                float(1.0 - v["mean"] / v["base_metric"]) if v.get("base_metric", 0) > 1e-12 else float("nan"))
    # reference per task: ebm0 if it ran, else the best candidate
    ref = {}
    for task, d in cells.items():
        lb = next(iter(d.values()))["lower_better"]
        cand = [v for k, v in d.items() if k[0] == "ebm0"]
        if cand:
            ref[task] = ("ebm0", float(cand[0]["mean"]))
        else:
            best = min(d.items(), key=lambda kv: kv[1]["mean"] if lb else -kv[1]["mean"])
            ref[task] = ("best_available:%s/%s" % best[0], best[1]["mean"])
    # per-task winners
    winners = {}
    for task, d in cells.items():
        lb = next(iter(d.values()))["lower_better"]
        best = min(d.items(), key=lambda kv: kv[1]["mean"] if lb else -kv[1]["mean"])
        second = sorted(d.items(), key=lambda kv: kv[1]["mean"] if lb else -kv[1]["mean"])[:3]
        winners[task] = dict(winner="%s/%s" % best[0], value=best[1]["mean"], bytes=best[1]["bytes"],
                             top3=[dict(name="%s/%s" % k, value=v["mean"], bytes=v["bytes"]) for k, v in second],
                             reference=ref[task][0], ref_value=ref[task][1])
    # universal ranking: mean and worst relative gap over the tasks where the candidate ran
    gaps: Dict[str, Dict[str, float]] = {}
    for task, d in cells.items():
        lb = next(iter(d.values()))["lower_better"]
        rv = ref[task][1]
        for (model, cal), v in d.items():
            gap = (v["mean"] - rv) / abs(rv) if lb else (rv - v["mean"]) / abs(rv)
            gaps.setdefault("%s/%s" % (model, cal), {})[task] = float(gap)
    # rank-based view: scale-free, and immune to one task dominating the mean gap
    ranks: Dict[str, List[float]] = {}
    for task, d in cells.items():
        lb = next(iter(d.values()))["lower_better"]
        order = sorted(d.items(), key=lambda kv: kv[1]["mean"] if lb else -kv[1]["mean"])
        n_c = len(order)
        for pos, (k, _) in enumerate(order, 1):
            ranks.setdefault("%s/%s" % k, []).append(float(pos) / max(n_c, 1))
    universal = []
    for name, g in gaps.items():
        if len(g) < cfg.min_tasks_universal:
            continue
        gm = float(np.mean(list(g.values())))
        gw = float(np.max(list(g.values())))
        universal.append(dict(name=name, tasks=len(g), mean_gap=gm, worst_gap=gw,
                              mean_rank=float(np.mean(ranks.get(name, [np.nan]))),
                              best_gap=float(np.min(list(g.values()))),
                              wins=int(sum(1 for t, v in winners.items() if v["winner"] == name)),
                              bytes=int(np.median([cells[t][tuple(name.split("/"))]["bytes"] for t in g if tuple(name.split("/")) in cells[t]] or [-1]))))
    for u in universal:
        model = u["name"].split("/")[0]
        cal = u["name"].split("/")[1]
        sk = [cells[t][(model, cal)]["skill"] for t in cells if (model, cal) in cells[t]]
        u["mean_skill"] = float(np.mean(sk)) if sk else float("nan")
        u["worst_skill"] = float(np.min(sk)) if sk else float("nan")
    universal.sort(key=lambda r: (-r["mean_skill"] if np.isfinite(r["mean_skill"]) else np.inf, r["mean_gap"]))
    agg_out = dict(cells=cells, winners=winners, reference=ref, universal=universal,
                n_rows=len(rows), n_ok=len(ok),
                failures=[r for r in rows if r.get("status") != "ok"])
    return refresh(agg_out)


# --------------------------------------------------------------------------------------------------------------
# 15.  Report
# --------------------------------------------------------------------------------------------------------------
def refresh(agg: dict) -> dict:
    """Recompute mean/macro/skill, the per-task reference, the winners and the universal ordering.

    Everything here is derived from the stored cells, so a report can be rebuilt from results.json without
    refitting. There is exactly one definition of each column in the file.
    """
    cells = agg["cells"]
    for task, d in cells.items():
        for k, v in d.items():
            if v.get("pooled") is not None:
                v["mean"] = float(v["pooled"])          # regression: pooled R^2 governs
            if v["lower_better"]:
                v["skill"] = (float(1.0 - v["mean"] / v["base_metric"])
                              if v.get("base_metric", 0) > 1e-12 else float("nan"))
            else:
                v["skill"] = float(v["mean"])
    ref = {}
    for task, d in cells.items():
        lb = next(iter(d.values()))["lower_better"]
        cand = [v for k, v in d.items() if k[0] == "ebm0"]
        if cand:
            ref[task] = ("ebm0", float(cand[0]["mean"]))
        else:
            best = min(d.items(), key=lambda kv: kv[1]["mean"] if lb else -kv[1]["mean"])
            ref[task] = ("best_available:%s/%s" % best[0], float(best[1]["mean"]))
    agg["reference"] = ref
    winners = {}
    for task, d in cells.items():
        lb = next(iter(d.values()))["lower_better"]
        order = sorted(d.items(), key=lambda kv: kv[1]["mean"] if lb else -kv[1]["mean"])
        best = order[0]
        winners[task] = dict(winner="%s/%s" % best[0], value=best[1]["mean"], bytes=best[1]["bytes"],
                             top3=[dict(name="%s/%s" % k, value=v["mean"], bytes=v["bytes"]) for k, v in order[:3]],
                             reference=ref[task][0], ref_value=ref[task][1])
    agg["winners"] = winners
    # gaps, skills and percentile ranks, then the universal ordering
    gaps, ranks = {}, {}
    for task, d in cells.items():
        lb = next(iter(d.values()))["lower_better"]
        rv = ref[task][1]
        order = sorted(d.items(), key=lambda kv: kv[1]["mean"] if lb else -kv[1]["mean"])
        n_c = max(len(order), 1)
        for pos, (k, v) in enumerate(order, 1):
            name = "%s/%s" % k
            g = (v["mean"] - rv) / abs(rv) if lb else (rv - v["mean"]) / abs(rv)
            gaps.setdefault(name, {})[task] = float(g)
            ranks.setdefault(name, []).append(float(pos) / n_c)
    for u in agg.get("universal", []):
        model, cal = u["name"].split("/")
        sk = [d[(model, cal)]["skill"] for d in cells.values() if (model, cal) in d]
        gp = list(gaps.get(u["name"], {}).values())
        u["tasks"] = len(sk)
        u["mean_skill"] = float(np.nanmean(sk)) if sk else float("nan")
        u["worst_skill"] = float(np.nanmin(sk)) if sk else float("nan")
        u["mean_gap"] = float(np.mean(gp)) if gp else float("nan")
        u["worst_gap"] = float(np.max(gp)) if gp else float("nan")
        u["mean_rank"] = float(np.mean(ranks.get(u["name"], [np.nan])))
        u["wins"] = int(sum(1 for w in winners.values() if w["winner"] == u["name"]))
    if not agg.get("universal"):
        for name in gaps:
            agg.setdefault("universal", []).append(dict(name=name, bytes=-1))
        for u in agg["universal"]:
            model, cal = u["name"].split("/")
            bs = [d[(model, cal)]["bytes"] for d in cells.values() if (model, cal) in d]
            u["bytes"] = int(np.median(bs)) if bs else -1
        for u in agg["universal"]:
            model, cal = u["name"].split("/")
            sk = [d[(model, cal)]["skill"] for d in cells.values() if (model, cal) in d]
            gp = list(gaps.get(u["name"], {}).values())
            u.update(tasks=len(sk), mean_skill=float(np.nanmean(sk)) if sk else float("nan"),
                     worst_skill=float(np.nanmin(sk)) if sk else float("nan"),
                     mean_gap=float(np.mean(gp)) if gp else float("nan"),
                     worst_gap=float(np.max(gp)) if gp else float("nan"),
                     mean_rank=float(np.mean(ranks.get(u["name"], [np.nan]))),
                     wins=int(sum(1 for w in winners.values() if w["winner"] == u["name"])))
    agg["universal"] = [u for u in agg.get("universal", []) if u["tasks"] >= 3]
    agg["universal"].sort(key=lambda r: (r["mean_rank"] if np.isfinite(r.get("mean_rank", np.nan)) else 9.0,
                                         r["mean_gap"]))
    return agg


def write_report(agg: dict, cfg, out_dir: str) -> str:
    agg = refresh(agg)
    os.makedirs(out_dir, exist_ok=True)
    flat = dict(agg)
    flat["rows"] = [r for r in agg.get("all_rows", []) if r.get("status") == "ok"]
    flat["cells"] = {t: {"%s|%s" % k: v for k, v in d.items()} for t, d in agg["cells"].items()}
    json.dump(flat, open(os.path.join(out_dir, "results.json"), "w"), indent=1, default=str)
    L = []
    A = L.append
    A("# Universal sweep — %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    A("Preset **%s** · m = %d cuts · passes %d..%d (conv_tol %s) · calibrators %s · %d/%d rows scored.\n"
      % (cfg.preset, cfg.m, cfg.passes, cfg.max_passes, cfg.conv_tol, ", ".join(cfg.calibrators), agg["n_ok"], agg["n_rows"]))
    A("> Every number here was produced by `r71_universal.py` in this run. Nothing is quoted from the record;")
    A("> agreement with it to 3–4 digits is the most that can be expected, and disagreement is more informative.\n")

    A("## 1. Best model per task\n")
    A("| task | winner | value | bytes (gzip, dead-cut) | reference | reference value | margin |")
    A("|---|---|---:|---:|---|---:|---:|")
    for task, w in agg["winners"].items():
        lb = "lower" if any(c["lower_better"] for c in agg["cells"][task].values()) else "higher"
        better = (w["ref_value"] - w["value"]) if lb == "lower" else (w["value"] - w["ref_value"])
        A("| %s | **%s** | %.6f | %s | %s | %.6f | %+.6f |"
          % (task, w["winner"], w["value"], w["bytes"], w["reference"], w["ref_value"], better))
    A("")
    for task, w in agg["winners"].items():
        A("**%s** — top 3: %s" % (task, " · ".join("`%s` %.6f @%s B" % (t["name"], t["value"], t["bytes"]) for t in w["top3"])))
    A("")

    A("## 2. The universal model\n")
    A("Ranked by the mean relative gap to each task's reference (EBM-0 where it ran, else the best object available),")
    A("over the tasks each candidate actually ran on. **0.000 means parity with the reference; negative means better.**\n")
    A("Ranked by **mean percentile rank** across the tasks the candidate ran on (1.0 = always last, 0.0 = always")
    A("first). This is deliberate: no raw metric is comparable across these tasks — under leave-cluster-out a fold's")
    A("constant baseline scores R^2 = -13.8, so an averaged R^2 is dominated by one blocking protocol. Skill columns")
    A("(1 - metric / fold baseline for classification, R^2 for regression) are reported but not used for the ordering.\n")
    A("| rank | candidate | tasks | **mean pct-rank** | median skill | worst skill | mean gap | tasks won | median bytes |")
    A("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, u in enumerate(agg["universal"][:cfg.report_top], 1):
        A("| %d | `%s` | %d | **%.3f** | %+.3f | %+.3f | %+.4f | %d | %s |"
          % (i, u["name"], u["tasks"], u.get("mean_rank", float("nan")), u.get("median_skill", float("nan")),
             u.get("worst_skill", float("nan")), u["mean_gap"], u["wins"], u["bytes"]))
    A("")
    if agg["universal"]:
        bm = agg["universal"][0]
        bw = max(agg["universal"], key=lambda r: r.get("worst_skill", -9) if np.isfinite(r.get("worst_skill", -9)) else -9)
        bwin = max(agg["universal"], key=lambda r: r["wins"])
        A("**Three answers, because they are three different questions.**\n")
        A("- best on average (mean percentile rank, lower is better): `%s` (%.3f over %d tasks)"
          % (bm["name"], bm.get("mean_rank", float("nan")), bm["tasks"]))
        A("- best worst case (minimax): `%s` (worst skill %+.4f, worst gap %+.4f)"
          % (bw["name"], bw.get("worst_skill", float("nan")), bw["worst_gap"]))
        A("- most tasks won: `%s` (%d tasks)" % (bwin["name"], bwin["wins"]))
        A("")

    A("## 3. The full per-task matrix\n")
    for task, d in agg["cells"].items():
        A("### %s\n" % task)
        A("| candidate | cal | mean (pooled) | macro | sd | skill | base | folds | bytes | converged | s/fold |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for (model, cal), v in sorted(d.items(), key=lambda kv: -kv[1].get("skill", -9)):
            A("| `%s` | %s | %.6f | %.4f | %.4f | %+.3f | %.4f | %d | %d | %s | %.1f |"
              % (model, cal, v["mean"], v.get("macro", float("nan")), v["sd"], v.get("skill", float("nan")),
                 v.get("base_metric", float("nan")), v["folds"], v["bytes"],
                 "yes" if v["converged_all"] else "**NO**", v["mean_seconds"]))
        A("")

    A("## 4. Everything that did not run, and why\n")
    if agg["failures"]:
        A("| task | model | fold | status |")
        A("|---|---|---:|---|")
        seen = set()
        for f in agg["failures"]:
            key = (f.get("task", "-"), f.get("model", "TASK"), f.get("status", "")[:60])
            if key in seen:
                continue
            seen.add(key)
            A("| %s | `%s` | %s | %s |" % (f.get("task", "-"), f.get("model", "TASK"), f.get("fold", "-"),
                                             f.get("status", "")[:120]))
    else:
        A("nothing failed.")
    A("")
    A("## 5. What this run cannot tell you\n")
    A("- **Convergence.** A row marked `NO` in the converged column used its pass cap without meeting `conv_tol`.")
    A("  Raise `--max-passes` before quoting it; the record already showed 12 passes is not a fixed point.")
    A("- **Calibrator mixing.** `T20` and `Toof` rows are not comparable with each other, at all. Each row states its own.")
    A("- **California geography.** The coastline here is 16 points; the record's gated object used a finer one.")
    A("  Comparisons between candidates on the same rows are fair; comparisons with the record's scores are not.")
    A("- **Sizes.** gzip of the emitted blob, itemised in `results.json`. The EBM rows are an approximation.")
    A("- **Aggregation.** Regression cells report POOLED R^2 (one sum of squares over all held-out rows) and the")
    A("  unweighted macro mean beside it. Under spatial blocking the macro mean is dominated by the smallest cells,")
    A("  where R^2 is unstable; the record also reports pooled and macro separately.")
    A("- **Ideas not here.** #2, #3, #9, #10, #13-as-an-estimator and #14 are not implemented; the sweep says nothing")
    A("  about them. What IS here: #1 (printer), #4, #5, #6, #7, #8, #11, #12, #13 (calibrator), #15, plus M3 and M8.")
    path = os.path.join(out_dir, "report.md")
    open(path, "w").write("\n".join(L))
    return path


# --------------------------------------------------------------------------------------------------------------
# 16.  Configuration and the driver
# --------------------------------------------------------------------------------------------------------------
@dataclass
class RunCfg:
    preset: str = "quick"
    m: int = 16
    passes: int = 4
    max_passes: int = 8
    conv_tol: float = 1e-5
    folds: int = 3
    seed: int = 42
    calibrators: List[str] = field(default_factory=lambda: ["T20"])
    models: List[str] = field(default_factory=lambda: ["card_fused", "card_kappa0"])
    tasks: List[str] = field(default_factory=lambda: ["aps", "california_random"])
    oof_folds: int = 5
    toof_max_rows: int = 20000
    bags: int = 8
    sigma_noise: float = 0.3
    joint_gamma: float = 0.5
    ramp_w: float = 0.02
    ramp_cap: float = 0.25
    post_runs: int = 1200
    n_pairs: int = 10
    pair_bins: int = 8               # per-side binning of a pair layer: 8 -> 64 cells, 16 -> 256 cells
    pair_support: int = 25           # cells below this many training rows fall into the pair's NaN bin
    spine_bins: int = 4
    spine_ridge: float = 1.0
    cal_rows: int = 0
    max_rows: int = 0
    level_shrink: float = 0.0            # binary card level shrinkage (Fisher-mass units)
    level_shrink_multiclass: float = 0.0   # measured best on Anneal's grouped folds; see the sweep in the report
    multiclass_alpha: float = 0.5          # measured: 0.5 beats 5.0 and 20.0 on stratified Anneal folds
    n_classes_declared: int = 0          # set per task in run_one
    max_spatial_folds: int = 0      # 0 = use every fold the blocking implies (a cap here must never drop rows)
    spatial_folds: int = 10         # merged blocking folds for lat10 / 2-D: 10, matching the record's lat10
    record_folds: bool = False      # DIAGNOSTIC: the record's exact fold constructions (attribution, not ranking)
    record_coast: bool = False      # DIAGNOSTIC: the record's 18-point segment-distance coastline
    censored: bool = False          # DIAGNOSTIC: the record's Tobit readout of the capped California target
    spatial_grid: int = 5           # 2-D blocking grid side
    min_tasks_universal: int = 3
    report_top: int = 25
    verify_printer: bool = True
    keep_bytes: bool = False
    n_jobs: int = -1
    out: str = "r71_out"
    verbose: bool = True


PRESETS = {
    "quick": dict(m=16, passes=6, max_passes=10, folds=3, calibrators=["T20"], max_rows=12000,
                  multiclass_alpha=0.5, max_spatial_folds=3, spatial_folds=10, spatial_grid=5,
                  n_pairs=16, pair_bins=16, pair_support=20,
                  models=["card_fused", "card_kappa0", "card_deadcut", "card_postcompress", "card_ramp",
                          "linear_spine"]),
    "standard": dict(m=32, passes=24, max_passes=48, conv_tol=1e-5, folds=5, calibrators=["T20", "Toof"],
                     max_spatial_folds=6, spatial_folds=10, spatial_grid=5, n_pairs=16, pair_bins=16,
                     pair_support=20, max_rows=12000,
                     models=["card_fused", "card_kappa0", "card_fisher", "card_deadcut", "card_deviance",
                             "card_skew", "card_ramp", "card_effectnoise", "card_jointdir", "card_bagfixed",
                             "card_postcompress", "card_contam", "card_gated_coast", "card_pairlayer",
                             "linear_spine"]),
    "full": dict(m=64, passes=24, max_passes=48, conv_tol=1e-6, folds=5,
                 calibrators=["T20", "Toof", "diag", "joint", "support"],
                 models=list(BUILDERS)),
}


def _job(payload):
    task, fold_i, tr, te, data, model, cfg = payload
    try:
        return run_one(task, fold_i, tr, te, data, model, cfg)
    except Exception:
        return [dict(task=task, model=model, fold=fold_i, status="ERROR: %s" % traceback.format_exc(limit=2))]


def run(cfg: RunCfg) -> dict:
    t_all = time.time()
    os.makedirs(cfg.out, exist_ok=True)
    all_rows: List[dict] = []
    log = open(os.path.join(cfg.out, "run.log"), "w") if not cfg.verbose else sys.stdout

    def say(*a):
        print(*a, file=log, flush=True)

    payloads = []
    unavailable = []
    for task in cfg.tasks:
        try:
            data = load_task(task, cfg)
        except Exception as exc:
            unavailable.append(dict(task=task, status="UNAVAILABLE: %s" % exc))
            say("[skip] %s — %s" % (task, exc))
            continue
        try:
            fl = task_folds(task, data, cfg)
        except Exception as exc:
            unavailable.append(dict(task=task, status="FOLDS FAILED: %s" % exc))
            say("[skip] %s — folds: %s" % (task, exc))
            continue
        say("[task] %-22s n=%6d  features=%4d  folds=%d  kind=%s"
            % (task, len(data["y"]), data["X"].shape[1], len(fl), data["kind"]))
        for k, (tr, te) in enumerate(fl):
            for model in cfg.models:
                payloads.append((task, k, tr, te, data, model, cfg))
    say("[plan] %d fits queued (tasks × folds × models)" % len(payloads))

    if cfg.n_jobs == 1:
        results = []
        for i, p in enumerate(payloads):
            results.extend(_job(p))
            if (i + 1) % 10 == 0:
                say("   %d/%d fits" % (i + 1, len(payloads)))
    else:
        from joblib import Parallel, delayed
        results = []
        for chunk in Parallel(n_jobs=cfg.n_jobs, verbose=(5 if cfg.verbose else 0), batch_size=1)(delayed(_job)(p) for p in payloads):
            results.extend(chunk)
    all_rows.extend(results)
    all_rows.extend(unavailable)

    agg = aggregate(all_rows, cfg)
    agg["unavailable"] = unavailable
    agg["all_rows"] = all_rows
    agg["cfg"] = dataclasses.asdict(cfg)
    agg["seconds"] = round(time.time() - t_all, 1)
    path = write_report(agg, cfg, cfg.out)

    say("\n=== DONE in %.0f s — report: %s" % (agg["seconds"], path))
    if agg["universal"]:
        say("universal (mean skill): %s" % ", ".join("%s %+.4f" % (u["name"], u.get("mean_skill", float("nan")))
                                                     for u in agg["universal"][:5]))
    for task, w in agg["winners"].items():
        say("best on %-22s %s  (%.6f, %s B)" % (task, w["winner"], w["value"], w["bytes"]))
    if log is not sys.stdout:
        log.close()
    return agg


# --------------------------------------------------------------------------------------------------------------
# 17.  Self-test: every implementation, on synthetic data, with no downloads
# --------------------------------------------------------------------------------------------------------------
def selftest() -> int:
    print("self-test: implementations, then a 2-fold end-to-end run on synthetic data\n")
    rng = np.random.default_rng(0)
    fails = []

    def check(name, cond, detail=""):
        print("  [%s] %-52s %s" % ("ok" if cond else "FAIL", name, detail))
        if not cond:
            fails.append(name)

    # --- binning -------------------------------------------------------------------------------------------
    x = np.array([0.0, 1.0, 1.0, 2.0, 5.0, np.nan, np.inf])
    c = cuts_of(x, 3)
    b = bin_idx(x, c)
    check("binning routes NaN to its own bin", b[-2] == len(c) + 1 and b[-1] == len(c) + 1)
    check("binning is right-closed", np.all(np.diff(b[:5]) >= -1))

    # --- exact bin solve -----------------------------------------------------------------------------------
    off = rng.normal(0, 1, 400)
    y = (rng.random(400) < sig(off - 0.7)).astype(float)
    idx = np.repeat(np.arange(10), 40)
    z, H = solve_bin_logistic(off, y, 10, idx)
    p = sig(off + z[idx])
    g = np.array([(y[idx == b] - p[idx == b]).sum() - ALPHA_DEFAULT * z[b] for b in range(10)])
    check("exact bin solve satisfies its KKT", np.max(np.abs(g)) < 1e-6, "max|g| = %.2e" % np.max(np.abs(g)))
    check("Fisher weights are positive", np.all(H > 0))

    # --- DP ------------------------------------------------------------------------------------------------
    zz = np.array([0.0, 0.01, 5.0, 5.01])
    HH = np.full(4, 100.0)
    lab = potts_dp(zz, HH, 0.5 * np.log(400.0))
    check("fusion merges two near-equal pairs", lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2],
          "labels=%s" % np.unique(lab, return_counts=True)[1])
    cost = cost_quadratic(zz, HH)
    labq = dp_exact_q(cost, 2)
    check("exact-q DP honours q exactly", labq is not None and len(np.unique(labq)) == 2)
    check("exact-q DP returns None when infeasible", dp_exact_q(cost, 99) is None)
    S = np.zeros(4)
    check("skew DP reduces to the quadratic DP at S=0", np.array_equal(potts_skew_dp(zz, HH, S, 1.0), potts_dp(zz, HH, 1.0)))
    C2 = np.column_stack([zz, -zz])
    check("multiclass DP accepts vector levels", len(np.unique(potts_dp(C2, HH, 2.0))) <= 2)

    # --- a card fit ----------------------------------------------------------------------------------------
    n, p = 600, 4
    X = rng.normal(size=(n, p))
    logit = 1.2 * np.tanh(2 * X[:, 0]) - 0.8 * (X[:, 1] > 0.3) + 0.5 * X[:, 2] * X[:, 3]
    yy = (rng.random(n) < sig(logit)).astype(float)
    cfg = FitCfg(m=8, passes=3, conv_tol=1e-4, max_passes=25)
    card = fit_card(X, yy, "binary", cfg)
    ll0 = metric("binary", yy, np.full(n, np.log(yy.mean() / (1 - yy.mean()))))[0]
    ll1 = metric("binary", yy, card.score(X))[0]
    check("card beats the intercept on training rows", ll1 < ll0, "%.4f -> %.4f" % (ll0, ll1))
    check("card converged under an explicit criterion", card.meta["converged"] is True)

    # --- printer --------------------------------------------------------------------------------------------
    v = verify_printer(card, X, "deadcut_ceil")
    check("printer: dead-cut + ceil is bit-identical", v["rows_bit_identical"] and v["probe_features_failing"] == 0,
          "%d probe points, %d features failing" % (v["probe_points"], v["probe_features_failing"]))
    check("printer: dead cuts are fewer than all cuts",
          v["n_cuts_live"] <= v["n_cuts_total"], "%d/%d" % (v["n_cuts_live"], v["n_cuts_total"]))
    vr = verify_printer(card, X, "full_rne")
    print("        (round-to-nearest stores %d of %d cuts below their true cut — the L18 defect; fix costs %+d B)"
          % (vr["rne_cuts_below_true_cut"], vr["cuts_checked"],
             vr["bytes_gzip"] - verify_printer(card, X, "full_ceil")["bytes_gzip"]))
    e = emit_card(card, "deadcut_ceil")
    check("byte account itemises to the raw size", sum(e["items"].values()) == e["bytes_raw"],
          "%s = %d" % (sum(e["items"].values()), e["bytes_raw"]))
    dec = decode_card(e["raw"], card.kind, len(card.cuts))
    ref = decode_card(emit_card(card, "full_ceil")["raw"], card.kind, len(card.cuts))
    X32 = X.astype(F32).astype(float)
    check("decoder from bytes reproduces the exact float32 emission",
          np.array_equal(dec.score(X32), ref.score(X32)))
    print("        (deviation from the float64 fit is float32 level rounding only: max %.2e)"
          % float(np.max(np.abs(dec.score(X32) - card.score(X)))))

    # --- calibrators ---------------------------------------------------------------------------------------
    c3 = Card(kind="binary", cuts=card.cuts, V=card.V, b0=card.b0, T=np.array(1.0), feat_names=card.feat_names)
    T = fit_T(yy, c3.score(X), "binary")           # in-sample temperature: must not be worse, by definition
    ll_raw = metric("binary", yy, c3.score(X))[0]
    ll_cal = metric("binary", yy, score_with_calibration(c3, X, "T20", dict(T=T)))[0]
    check("an in-sample temperature never worsens the score", ll_cal <= ll_raw + 1e-9, "%.5f -> %.5f" % (ll_raw, ll_cal))
    Th = calib_T20(c3, X, yy)
    print("        (a 20%% hold-out temperature is fitted on a subset and CAN be worse on the whole: T=%.3f)" % float(Th))
    js = calib_joint(c3, X, yy)
    check("joint (idea #12) calibrator converges", js["success"])
    sp = calib_support(c3, X, yy)
    check("support (idea #13) calibrator returns finite coefficients", np.isfinite(sp["T"]) and np.isfinite(sp["gamma"]))

    # --- the other estimators ------------------------------------------------------------------------------
    ctx_cfg = RunCfg(m=8, passes=2, max_passes=3, conv_tol=1e-5, bags=3, verify_printer=False)
    ctx = Ctx("synthetic", "binary", X, yy, X, yy, ["a", "b", "c", "d"], 0, ctx_cfg)
    for name in ["card_fused", "card_kappa0", "card_fisher", "card_deadcut", "card_deviance", "card_skew",
                 "card_ramp", "card_effectnoise", "card_jointdir", "card_bagfixed", "card_postcompress",
                 "card_contam", "linear_spine"]:
        try:
            o, notes = BUILDERS[name](ctx)
            s = o.card.score(X) if isinstance(o, _External) else o.score(X)
            ok = np.all(np.isfinite(s))
        except Exception as exc:
            ok = False
            notes = str(exc)
        check("candidate %s builds a finite prediction" % name, ok, "" if ok else str(notes)[:80])

    # regression + multiclass paths
    yr = logit + rng.normal(0, 0.3, n)
    cr = fit_card(X, yr, "regression", FitCfg(m=8, passes=3))
    check("regression card fits", np.all(np.isfinite(cr.score(X))))
    y3 = rng.integers(0, 3, n)
    cm = fit_card(X, y3, "multiclass", FitCfg(m=8, passes=2))
    check("multiclass card fits and normalises", abs(softmax(cm.score(X)).sum(axis=1).mean() - 1.0) < 1e-9)

    # --- end-to-end through the runner ---------------------------------------------------------------------
    print("\n  end-to-end: two folds, three candidates, one calibrator")
    data = dict(kind="binary", X=X, y=yy, names=["a", "b", "c", "d"], geo=None)
    cfg2 = RunCfg(m=8, passes=2, max_passes=3, conv_tol=1e-5, folds=2, calibrators=["T20"],
                  models=["card_fused", "card_kappa0", "linear_spine"], verify_printer=False, n_jobs=1)
    fl = folds_random(n, 2, 0, strat=yy)
    rows = []
    for k, (tr, te) in enumerate(fl):
        for mo in cfg2.models:
            rows.extend(run_one("synthetic", k, tr, te, data, mo, cfg2))
    agg = aggregate(rows, cfg2)
    check("runner produced a cell per (model, cal)", len(agg["cells"].get("synthetic", {})) == 3,
          str(len(agg["cells"].get("synthetic", {}))))
    check("universal ranking is computable", isinstance(agg["universal"], list))
    check("per-task winner exists", "synthetic" in agg["winners"])

    print("\n%d checks failed%s" % (len(fails), (": " + ", ".join(fails)) if fails else " — all implementations behave"))
    return 0 if not fails else 1


# --------------------------------------------------------------------------------------------------------------
# 18.  CLI
# --------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "universal sweep")
    ap.add_argument("--selftest", action="store_true", help="check every implementation on synthetic data, no download")
    ap.add_argument("--preset", default="quick", choices=list(PRESETS))
    ap.add_argument("--tasks", default="", help="comma list; default = the preset's tasks (all of them for standard/full)")
    ap.add_argument("--models", default="", help="comma list of candidate names; 'all' = every registered builder")
    ap.add_argument("--calibrators", default="", help="comma list of T20,Toof,diag,s16,joint,support")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--m", type=int, default=0, help="cuts per feature")
    ap.add_argument("--max-passes", type=int, default=0)
    ap.add_argument("--conv-tol", type=float, default=None)
    ap.add_argument("--cal-rows", type=int, default=0, help="subsample California (the record pinned 1500)")
    ap.add_argument("--max-rows", type=int, default=0, help="stratified row cap per task (0 = all rows)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default="r71_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-ebm", action="store_true", help="add ebm0/ebm10 comparators (needs `interpret`)")
    ap.add_argument("--record-folds", action="store_true",
                    help="diagnostic only: the record's exact fold constructions (gap attribution, never ranking)")
    ap.add_argument("--record-coast", action="store_true",
                    help="diagnostic only: the record's 18-point segment-distance coastline")
    ap.add_argument("--censored", action="store_true",
                    help="diagnostic only: the record's Tobit readout of the capped California target")
    ap.add_argument("--list", action="store_true", help="print tasks and candidates and exit")
    a = ap.parse_args(argv)

    if a.list:
        print("tasks:")
        for t in TASKS:
            print("   ", t)
        print("candidates:")
        for m in BUILDERS:
            print("   %-18s %s" % (m, "geography only" if m in NEEDS_GEO else ("binary only" if m in BINARY_ONLY else "")))
        return 0
    if a.selftest:
        return selftest()

    cfg = RunCfg(preset=a.preset, out=a.out, seed=a.seed)
    for k, v in PRESETS[a.preset].items():
        setattr(cfg, k, v)
    if a.tasks:
        cfg.tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    else:
        cfg.tasks = list(TASKS) if a.preset != "quick" else ["aps", "california_random", "anneal_grouped"]
    if a.models:
        cfg.models = list(BUILDERS) if a.models.strip() == "all" else [m.strip() for m in a.models.split(",") if m.strip()]
    if a.with_ebm:
        cfg.models = list(dict.fromkeys(cfg.models + ["ebm0", "ebm10"]))
    if a.calibrators:
        cfg.calibrators = [c.strip() for c in a.calibrators.split(",") if c.strip()]
    if a.folds:
        cfg.folds = a.folds
    if a.m:
        cfg.m = a.m
    if a.max_passes:
        cfg.max_passes = a.max_passes
    if a.conv_tol is not None:
        cfg.conv_tol = a.conv_tol
    if a.cal_rows:
        cfg.cal_rows = a.cal_rows
    if a.max_rows:
        cfg.max_rows = a.max_rows          # 0 must NOT overwrite the preset's cap (it did, and the run OOMed)
    cfg.n_jobs = a.n_jobs
    cfg.record_folds = a.record_folds
    cfg.record_coast = a.record_coast
    cfg.censored = a.censored
    unknown = [m for m in cfg.models if m not in BUILDERS]
    if unknown:
        print("unknown candidates: %s\n(allowed: %s)" % (unknown, ", ".join(BUILDERS)))
        return 2
    if cfg.calibrators and cfg.calibrators[0] == "":
        cfg.calibrators = ["T20"]
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
