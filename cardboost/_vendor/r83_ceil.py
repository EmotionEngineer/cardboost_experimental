"""r83_ceil.py -- R83 Part 1 (H7): ceiling / output-link arms for kind == "reg",
evaluated as TWO isolated arms on top of the frozen Stage-1 card.

Arm 1A  IsoLink-OOF (<= 64 B deploy): monotone map g(eta) fit by mass-pooled
        PAVA on 16 groups (each >= n/50 rows) of OOF predictions, weak ridge
        pull to the identity (lambda_id = 0.1), 16 piecewise-linear knots in
        float16.  Prequential gate: per-subfold fit on 4 parts / evaluate on
        the 5th; admitted only if aggregate OOF gain > 0.5*ln(n) nats,
        else g(eta) == eta.
Arm 1B  Point-Mass Hurdle (Opus #13, data-triggered): if mass at exact
        max(y) (or min(y)) >= 1.5% of training rows -> compact binary card
        C(x) (m=16, 1D mains only, passes=24) on z = 1[y == y_extreme],
        Toof-calibrated probability pi_hat(x); yhat = pi*y_extreme +
        (1-pi)*mu(x) with mu = Stage-1 output.  Prequential gate as in 1A.
"""
from __future__ import annotations

import numpy as np

from r71_universal import fit_T


def _parts(n, seed=109, k=5):
    rng = np.random.default_rng(seed)
    return [np.asarray(p, int) for p in np.array_split(rng.permutation(n), k)]


def nats_gain(y, eta_b, eta_a, sigma2):
    """total nats gain of eta_a over eta_b (regression, user convention)."""
    return float(np.sum((y - eta_b) ** 2 - (y - eta_a) ** 2) / (2.0 * max(sigma2, 1e-9)))


# ---------------------------------------------------------------- 1A IsoLink
def iso_fit(eta, y, n_nodes=16, ridge=0.1):
    """mass-pooled PAVA map; returns (knot_x, knot_v) monotone piecewise-linear."""
    eta = np.asarray(eta, float)
    y = np.asarray(y, float)
    n = len(y)
    order = np.argsort(eta, kind="stable")
    groups = [g for g in np.array_split(order, n_nodes) if len(g)]
    kx, kv, w = [], [], []
    for g in groups:
        kx.append(float(eta[g].mean()))
        w.append(float(len(g)))
        kv.append(float(y[g].mean()))
    kx = np.asarray(kx)
    w = np.asarray(w)
    # weak ridge to identity before PAVA (exact for the combined objective)
    tgt = (w * np.asarray(kv) + ridge * w * kx) / (w * (1.0 + ridge))
    ww = w * (1.0 + ridge)
    # PAVA (isotonic increasing): blocks carry (sum_w, sum_wy, members)
    blocks = [[ww[i], ww[i] * tgt[i], [i]] for i in range(len(tgt))]
    merged = True
    while merged:
        merged = False
        for b in range(len(blocks) - 1):
            m1 = blocks[b][1] / blocks[b][0]
            m2 = blocks[b + 1][1] / blocks[b + 1][0]
            if m1 > m2:
                blocks[b] = [blocks[b][0] + blocks[b + 1][0],
                             blocks[b][1] + blocks[b + 1][1],
                             blocks[b][2] + blocks[b + 1][2]]
                del blocks[b + 1]
                merged = True
                break
    keep = np.array(sorted(b[2][0] for b in blocks))
    vout = np.array([b[1] / b[0] for b in blocks])
    # R83 spec: quantize knots to float16 AT FIT so in-memory and decoded
    # computations run on identical f16 nodes (deploy == fitted g, exactly).
    return np.asarray(kx[keep], np.float16), np.asarray(vout, np.float16)


def iso_apply(knots, eta):
    from r76_container31 import apply_isolink
    return apply_isolink(knots, eta)


def iso_arm(eta_oof, y, eta_te, sigma2, n_nodes=16, ridge=0.1, seed=109):
    """returns (yhat_te, admitted, gain_nats)."""
    n = len(y)
    parts = _parts(n, seed)
    tot = 0.0
    for val in parts:
        tr = np.setdiff1d(np.arange(n), val)
        knots = iso_fit(eta_oof[tr], y[tr], n_nodes, ridge)
        tot += nats_gain(y[val], eta_oof[val], iso_apply(knots, eta_oof[val]), sigma2)
    admitted = bool(tot > 0.5 * np.log(n))
    if not admitted:
        return np.asarray(eta_te, float).copy(), False, tot
    knots = iso_fit(eta_oof, y, n_nodes, ridge)
    return iso_apply(knots, eta_te), True, tot


# ---------------------------------------------------------------- 1B Hurdle
def hurdle_trigger(y):
    y = np.asarray(y, float)
    n = len(y)
    for extreme in ("max", "min"):
        ye = y.max() if extreme == "max" else y.min()
        m = float(np.mean(y == ye))
        if m >= 0.015:
            return True, ye, m, extreme
    return False, 0.0, 0.0, ""


def _head_factory(m, passes):
    import r77_stage as T

    def fit(Xf, zf, kf):
        return T._fitj(Xf, zf, kf, atoms=[], n_classes=0, m=m, passes=passes,
                       max_passes=passes, conv_tol=0.0)
    return fit


def hurdle_head(Xtr, z, m=16, passes=24, seed=109):
    """compact binary card + Toof temperature; returns (card, T)."""
    import r76_lib as L
    fit = _head_factory(m, passes)
    card = fit(Xtr, z, "binary")
    eta_oof = L.toof_eta(fit, Xtr, z, "binary", folds=5, seed=seed)
    try:
        Tcal = float(fit_T(z, eta_oof, "binary"))
    except Exception:
        Tcal = 1.0
    return card, Tcal


def hurdle_arm(Xtr, z, Xte, ytr, mu_oof, mu_te, y_extreme, sigma2, seed=109):
    """5-subfold preq gate on the combined prediction; returns
    (yhat_te, admitted, gain_nats, card, Tcal)."""
    n = len(ytr)
    parts = _parts(n, seed)
    fit = _head_factory(16, 24)
    tot = 0.0
    for val in parts:
        tr = np.setdiff1d(np.arange(n), val)
        c = fit(Xtr[tr], z[tr], "binary")
        s = np.asarray(c.score(Xtr[val]), float)
        pi = 1.0 / (1.0 + np.exp(-np.clip(s, -80, 80)))
        comb = pi * y_extreme + (1.0 - pi) * mu_oof[val]
        tot += nats_gain(ytr[val], mu_oof[val], comb, sigma2)
    admitted = bool(tot > 0.5 * np.log(n))
    if not admitted:
        return mu_te.copy(), False, tot, None, 1.0
    card, Tcal = hurdle_head(Xtr, z, 16, 24, seed)
    s_te = np.asarray(card.score(Xte), float) / Tcal
    pi_te = 1.0 / (1.0 + np.exp(-np.clip(s_te, -80, 80)))
    yhat = pi_te * y_extreme + (1.0 - pi_te) * mu_te
    return yhat, True, tot, card, Tcal
