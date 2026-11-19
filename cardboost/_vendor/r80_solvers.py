"""r80_solvers.py -- R80 Phase-1 competing solver branches (user-approved scope:
#1 fair-share microboosted Potts, #2 correlation-block simultaneous updates;
#3 antithetic sweeps DROPPED per user decision: too close to refuted R32).

Both branches keep the R79 p-law schedule (passes/conv_tol/kappa_mult come from
the unchanged _fitj injection); they replace ONLY the mains update rule:

  solver="gs"        -- untouched R71/R76/R79 Gauss-Seidel overwrite (default;
                        fit_joint's original code path, bit-exact).
  solver="fairshare" -- #1: per pass, all Potts proposals d_j = V_j^Potts - V_j
                        are computed and scored with EXACT deviance gains
                        (nats); visits tracked; j* = argmax
                        (gain_j - codeprice_j)/sqrt(1+visits_j) among
                        positive-gain, not-yet-visited features; the winner's
                        proposal is refreshed from the current offset and
                        re-checked; EXACT clipped line search s* in [0, 1] on
                        the true deviance (golden-section, analytic for
                        regression SSE); V_j += s* d_j.  A feature cannot win
                        twice in a pass; a non-positive fresh gain or s*=0
                        rejects it for the pass (spec: "visited or rejected").
  solver="corrblock" -- #2: response-free blocks from Spearman |rho| >= 0.3 or
                        missing-mask Jaccard >= 0.5 (Kruskal single-linkage,
                        |B| <= 8, deterministic); per pass, cyclic over blocks;
                        all block proposals computed from the SAME pre-block
                        offset; bounded amplitudes a in [0,1]^|B| minimise the
                        TRUE L(eta + D_B a) + (alpha/2)||a||^2 (coordinate
                        ascent, exact 1-D clipped searches per coordinate,
                        alpha = 0.5 nats frozen); commit simultaneously.

Conventions (frozen, name-blind, Rule 12.B-3):
  * codeprice_j = 0.5*ln(n) + ln(M_j), M_j = Potts segments of the proposal,
    in nats, compared against exact deviance gains in nats.
  * L is the data deviance (regression: SSE/(2*sigma2), sigma2 = current
    mean(y-eta)^2 >= 1e-12 per pass; binary: binomial deviance; multiclass:
    multinomial NLL).  The Haldane/alpha penalty lives inside the proposal
    construction (solve_bin_*), not in the amplitude model.
"""
from __future__ import annotations

import numpy as np

import r76_mech as M
from r71_universal import (
    solve_bin_logistic, solve_bin_regression, solve_bin_multiclass,
    potts_dp, _segment_levels, softmax, sig,
)


# ---------------------------------------------------------------- proposals
def propose(off, y, Y1, kind, idx, nb, w, alpha, km):
    """One Potts proposal V (as in fit_joint's per-feature block) from `off`."""
    if kind == "multiclass":
        K = Y1.shape[1]
        if w is not None:
            C2, H = M.wsolve_bin_multiclass(off, Y1, nb, idx, K, w, alpha=alpha)
        else:
            C2, H = solve_bin_multiclass(off, Y1, nb, idx, K, alpha=alpha)
        emp = H <= 1e-12
        kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * km
        lab = potts_dp(np.where(emp[:, None], 0.0, C2), np.where(emp, 1e-9, H), kap)
        V = _segment_levels(lab, None, H, "multiclass",
                            C2=np.where(emp[:, None], 0.0, C2), shrink=0.0)
        return V, H
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
    kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * km
    lab = potts_dp(zc, np.where(emp, 1e-9, H), kap)
    V = _segment_levels(lab, zc, H, "binary" if kind == "binary" else "regression",
                        shrink=0.0)
    return V, H


# ------------------------------------------------------- exact deviance loss
def loss_eval(eta, y, Y1, kind, sigma2=1.0):
    """data deviance in nats (regression: SSE/(2*sigma2))."""
    if kind == "regression":
        r = y - eta
        return float(np.sum(r * r) / (2.0 * sigma2))
    if kind == "binary":
        # -sum[y*eta - log(1+exp(eta))] = sum(softplus(eta) - y*eta)
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))
    # multiclass NLL
    Z = eta - eta.max(axis=1, keepdims=True)
    lse = np.log(np.exp(Z).sum(axis=1)) + eta.max(axis=1)
    return float(np.sum(lse - eta[np.arange(len(y)), y.astype(int)]))


def gold_search(f, lo=0.0, hi=1.0, tol=1e-4):
    """golden-section minimiser of convex f on [lo, hi]; deterministic."""
    gr = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b), f(0.5 * (a + b))


def line_step_exact(eta, d_idx, y, Y1, kind, sigma2=1.0, w=None):
    """exact clipped s* in [0,1] minimising L(eta + s*d_idx); (s*, L(s*))."""
    if kind == "regression":
        r = y - eta
        if w is not None:
            r = r * w
            den = float(np.sum(np.asarray(w, float) * d_idx * d_idx))
        else:
            den = float(np.sum(d_idx * d_idx))
        num = float(np.sum(r * d_idx))
        s = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0
        return s, loss_eval(eta + s * d_idx, y, Y1, kind, sigma2)
    if kind == "binary":
        f = lambda s: loss_eval(eta + s * d_idx, y, Y1, kind)
    else:
        f = lambda s: loss_eval(eta + s * d_idx, y, Y1, kind)
    best_s, best_l = 0.0, f(0.0)
    g_s, g_l = gold_search(f)
    if g_l < best_l:
        best_s, best_l = g_s, g_l
    l1 = f(1.0)
    if l1 < best_l:
        best_s, best_l = 1.0, l1
    return best_s, best_l


def count_segments(V):
    V = np.asarray(V)
    if V.ndim == 1:
        return int(1 + np.sum(V[1:] != V[:-1]))
    diff = np.any(V[1:] != V[:-1], axis=1)
    return int(1 + np.sum(diff))


# ------------------------------------------------------- #1 fair-share pass
def fairshare_pass(st, Itr, NB, Fv, eta, y, Y1, kind, w, alpha, km, n, p,
                   H_list, sigma2):
    L0 = loss_eval(eta, y, Y1, kind, sigma2)
    props, gain0, segs = [], [], []
    for j in range(p):
        off = eta - Fv[j][Itr[j]]
        V, H = propose(off, y, Y1, kind, Itr[j], NB[j], w, alpha, km)
        d = V - Fv[j]
        props.append(d)
        H_list[j] = H
        segs.append(count_segments(V))
        gain0.append(L0 - loss_eval(eta + d[Itr[j]], y, Y1, kind, sigma2))
    visited, rejected = set(), set()
    for _ in range(p):
        best, best_score = -1, -np.inf
        for j in range(p):
            if j in visited or j in rejected or gain0[j] <= 0.0:
                continue
            price = 0.5 * np.log(n) + np.log(max(segs[j], 1))
            score = (gain0[j] - price) / np.sqrt(1.0 + st["visits"][j])
            if score > best_score:
                best_score, best = score, j
        if best < 0:
            break
        j = best
        # fresh proposal + fresh check for the winner
        off = eta - Fv[j][Itr[j]]
        V, H = propose(off, y, Y1, kind, Itr[j], NB[j], w, alpha, km)
        H_list[j] = H
        d = V - Fv[j]
        segs[j] = count_segments(V)
        if L0 - loss_eval(eta + d[Itr[j]], y, Y1, kind, sigma2) <= 0.0:
            rejected.add(j)
            continue
        s, _ = line_step_exact(eta, d[Itr[j]], y, Y1, kind, sigma2, w)
        if s <= 0.0:
            rejected.add(j)
            continue
        Fv[j] = Fv[j] + s * d
        eta = eta + s * d[Itr[j]]
        L0 = loss_eval(eta, y, Y1, kind, sigma2)
        st["visits"][j] += 1
        visited.add(j)
    return eta


# ---------------------------------------------------- #2 correlation blocks
def _avg_ranks(col):
    order = np.argsort(col, kind="mergesort")
    ranks = np.empty(len(col), float)
    ranks[order] = np.arange(len(col), dtype=float)
    scol = col[order]
    i = 0
    while i < len(col):
        j = i
        while j + 1 < len(col) and scol[j + 1] == scol[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def corr_blocks(X, max_size=8, rho0=0.3, jacc0=0.5):
    """response-free feature blocks: |Spearman| >= rho0 OR mask-Jaccard >= jacc0,
    single-linkage via Kruskal (descending weight), hard size cap `max_size`."""
    n, p = X.shape
    nan = np.isnan(X)
    if nan.any():
        fill = np.nanmax(np.where(nan, -np.inf, X), axis=0) + 1.0
        Xr = np.where(nan, fill, X)
    else:
        Xr = X
    R = np.column_stack([_avg_ranks(np.ascontiguousarray(Xr[:, j])) for j in range(p)])
    R -= R.mean(axis=0)
    sd = np.sqrt((R * R).sum(axis=0))
    sd[sd <= 0] = 1.0
    C = (R.T @ R) / np.outer(sd, sd)
    edges = []
    for j in range(p):
        mj = nan[:, j]
        for k in range(j + 1, p):
            w_rho = abs(float(C[j, k]))
            mk = nan[:, k]
            inter = float(np.sum(mj & mk))
            uni = float(np.sum(mj | mk))
            w_jac = inter / uni if uni > 0 else 0.0
            if w_rho >= rho0 or w_jac >= jacc0:
                edges.append((max(w_rho, w_jac), j, k))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    parent = list(range(p))
    size = [1] * p

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, j, k in edges:
        rj, rk = find(j), find(k)
        if rj == rk:
            continue
        if size[rj] + size[rk] > max_size:
            continue
        if rj > rk:
            rj, rk = rk, rj
        parent[rk] = rj
        size[rj] += size[rk]
    blocks = {}
    for j in range(p):
        blocks.setdefault(find(j), []).append(j)
    return [sorted(v) for _, v in sorted(blocks.items(), key=lambda kv: min(kv[1]))]


def corrblock_pass(st, Itr, NB, Fv, eta, y, Y1, kind, w, alpha, km, n, p,
                   H_list, sigma2, lam=0.5):
    for B in st["blocks"]:
        props = []
        for j in B:
            off = eta - Fv[j][Itr[j]]          # SAME pre-block offset for all j
            V, H = propose(off, y, Y1, kind, Itr[j], NB[j], w, alpha, km)
            H_list[j] = H
            props.append(V - Fv[j])
        m = len(B)
        a = np.zeros(m)
        # coordinate ascent on L(eta_pre + sum_j a_j d_j) + (lam/2)||a||^2,
        # exact 1-D clipped searches (f convex in each coordinate).
        if m == 1:
            j = B[0]
            s, _ = line_step_exact(eta, props[0][Itr[j]], y, Y1, kind, sigma2, w)
            if s > 0.0:
                Fv[j] = Fv[j] + s * props[0]
                eta = eta + s * props[0][Itr[j]]
            continue
        base = eta.copy()
        cur = base.copy()

        def fj(jj, val):
            trial = cur + (val - a[jj]) * props[jj][Itr[B[jj]]]
            return loss_eval(trial, y, Y1, kind, sigma2) + 0.5 * lam * (
                float(np.sum(a * a)) - a[jj] * a[jj] + val * val)

        for _ in range(10):
            da = 0.0
            for jj in range(m):
                f = lambda val: fj(jj, val)
                s, _ = gold_search(f, 0.0, 1.0, tol=1e-3)
                if fj(jj, 1.0) < f(s):
                    s = 1.0
                if f(0.0) < f(s):
                    s = 0.0
                delta = s - a[jj]
                if delta != 0.0:
                    cur = cur + delta * props[jj][Itr[B[jj]]]
                    a[jj] = s
                da = max(da, abs(delta))
            if da < 1e-3:
                break
        for jj, j in enumerate(B):
            if a[jj] > 0.0:
                Fv[j] = Fv[j] + a[jj] * props[jj]
        eta = cur
    return eta


# ------------------------------------------------------------- pass driver
def mains_sweep(solver, Itr, NB, Fv, eta, y, Y1, kind, w, alpha, km,
                H_list, st, n, p, sigma2):
    if solver == "fairshare":
        return fairshare_pass(st, Itr, NB, Fv, eta, y, Y1, kind, w, alpha, km,
                              n, p, H_list, sigma2)
    if solver == "corrblock":
        return corrblock_pass(st, Itr, NB, Fv, eta, y, Y1, kind, w, alpha, km,
                              n, p, H_list, sigma2)
    raise ValueError(solver)


def init_state(solver, X, p):
    st = {"visits": np.zeros(p, float)}
    if solver == "corrblock":
        st["blocks"] = corr_blocks(X)
    return st
