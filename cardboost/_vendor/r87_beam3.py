"""r87_beam3.py -- R87 Canary 6 (R114): beam3_ledger (Beam-Search преквенциальный леджер).

ЗАМОРОЖЕННАЯ СПЕЦИФИКАЦИЯ (план юзера 2026-11-18):
  На глубине d = 1..D храним B = 3 лучших РАЗЛИЧНЫХ набора атомов -- канонизация
  ключа состояния = frozenset индексов принятых пар (перестановки одного набора
  {A1,A2}/{A2,A1} не плодятся).  Каждый из <= 3 финальных кандидатов проходит
  штатный warm6 (совместный backfit с базовыми атомами карты), и побеждает тот,
  у кого максимален честный преквенциальный счёт
      Score(S) = dL_OOF(S) - 0.5*d_eff(S)*ln n - |S|*ln M - ln B.
  Барьер приёма: Score > 0.  d_eff(S) = суммарные live-ячейки атомов + |S|;
  M = размер пула кандидатов; плата за пучок ln B (B=3).
  На aps и anneal, где парные атомы не набираются, механизм БИТ-В-БИТ нейтрален
  (beam пуст -> карта не тронута).  multiclass (anneal) молчит структурно.
"""
from __future__ import annotations
import numpy as np

import r71_universal as r71
import r77_mech as R
import r77_stage as T

from r87_safetier import (nats_gain, oof_base, _mk_p1, tier_candidates,
                          d_eff, P1_BINS_SAFE, MIN_COUNT)


K_POOL = 24
D_DEPTH = 3
B_BEAM = 3
SUBFOLDS = 5
SEED = 109


def set_cheap_score(pairs_set, X, y, kind, eta_oof, parts, sigma2=None):
    """Аддитивный честный gain набора (crossfit): атомы по очереди на остатке."""
    n = len(y)
    tot = 0.0
    for val in parts:
        ftr = np.setdiff1d(np.arange(n), val)
        o_tr = np.asarray(eta_oof[ftr], float).copy()
        o_val = np.asarray(eta_oof[val], float).copy()
        for (j, k) in sorted(pairs_set):
            at = _mk_p1(j, k, X)
            at.ensure(kind, n_classes=0)
            at.refit(X[ftr], y[ftr], o_tr, kind, kappa_mult=1.0)
            o_tr = o_tr + at.contrib(X[ftr])
            o_val = o_val + at.contrib(X[val])
        tot += nats_gain(kind, y[val], eta_oof[val], o_val, sigma2)
    return float(tot)


def beam_search(card, X, y, kind, n_classes, K=K_POOL, D=D_DEPTH, B=B_BEAM, cache=None):
    """Пучок B=3 по различным наборам (frozenset-канонизация)."""
    meta = dict(tier="beam3", kind=kind, cand=0, nodes=[], admitted=[],
                suppressed=[], gain=0.0)
    if kind == "multiclass":
        meta["note"] = "skip multiclass (молчит)"
        return [], meta
    pool, base = tier_candidates(card, X, y, kind, K)
    meta["cand"] = len(pool)
    if not pool:
        return [], meta
    n = len(y)
    eta_oof, parts = oof_base(X, y, kind, n_classes, folds=SUBFOLDS, seed=SEED, cache=cache)
    sigma2 = float(np.mean((y - eta_oof) ** 2)) if kind == "regression" else None
    M = max(len(pool), 1)
    lnM, lnB, lnbar = float(np.log(M)), float(np.log(B)), 0.5 * float(np.log(n))
    de_bound = (P1_BINS_SAFE * P1_BINS_SAFE + 1)

    seen = {frozenset(): 0.0}
    beam = {frozenset(): 0.0}
    for d in range(1, D + 1):
        cand = {}
        for s in beam:
            for a in pool:
                if a in s:
                    continue
                ns = s | {a}
                if ns in seen:
                    continue
                g = set_cheap_score(ns, X, y, kind, eta_oof, parts, sigma2)
                de = de_bound * len(ns)
                score = g - 0.5 * de * lnbar - len(ns) * lnM - lnB
                cand[ns] = (float(score), float(g))
        if not cand:
            break
        top = sorted(cand.items(), key=lambda kv: -kv[1][0])[:B]
        beam = {k: v[0] for k, v in top}
        for k, v in top:
            seen[k] = v[0]
        meta["nodes"].append(dict(depth=d, top=[(sorted(map(list, k)), round(v[0], 2))
                                                for k, v in top]))
        print(f"  beam d={d}: " + " | ".join(
            f"{sorted(k)} s={v[0]:.2f} g={v[1]:.1f}" for k, v in top), flush=True)

    # финалисты: топ-3 различных набора по дешёвому счёту
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:B]
    return [(list(s), sc) for s, sc in ranked if s], meta


def honest_warm6_score(pairs_set, X, y, kind, n_classes, parts, sigma2=None):
    """Честный Δℓ_OOF набора: субфолд-карты mains + атомы + ШТАТНЫЙ warm6."""
    n = len(y)
    tot = 0.0
    for val in parts:
        ftr = np.setdiff1d(np.arange(n), val)
        b = T._fitj(X[ftr], y[ftr], kind, atoms=[], n_classes=n_classes)
        o = np.asarray(b.score(X[ftr]), float)
        aa = []
        for (j, k) in sorted(pairs_set):
            at = _mk_p1(j, k, X)
            at.ensure(kind, n_classes=n_classes)
            at.refit(X[ftr], y[ftr], o, kind, kappa_mult=1.0)
            o = o + at.contrib(X[ftr])
            aa.append(at)
        c = T._fitj(X[ftr], y[ftr], kind, atoms=aa, warm_card=b,
                    passes=T.FITCFG["warm_passes"], conv_tol=0.0)
        tot += nats_gain(kind, y[val],
                         np.asarray(b.score(X[val]), float),
                         np.asarray(c.score(X[val]), float), sigma2)
    return float(tot)


def final_select(finalists, X, y, kind, n_classes, K, B=B_BEAM, cache=None):
    """Штатный warm6 для <=B финалистов + честный счёт + барьер Score>0."""
    n = len(y)
    rng = np.random.default_rng(SEED)
    parts = [np.asarray(p, int) for p in
             np.array_split(rng.permutation(n), SUBFOLDS)]
    sigma2 = None
    if kind == "regression":
        e0, _ = oof_base(X, y, kind, n_classes, folds=SUBFOLDS, seed=SEED,
                         cache=cache)
        sigma2 = float(np.mean((y - e0) ** 2))
    M = max(K, 1)
    lnM, lnB, lnbar = float(np.log(M)), float(np.log(B)), 0.5 * float(np.log(n))
    out = []
    for (pairs, cheap) in finalists:
        g = honest_warm6_score(pairs, X, y, kind, n_classes, parts, sigma2)
        de = 0
        for (j, k) in pairs:
            de += (P1_BINS_SAFE * P1_BINS_SAFE + 1)
        score = g - 0.5 * de * lnbar - len(pairs) * lnM - lnB
        out.append(dict(pairs=[list(p) for p in sorted(pairs)], cheap=round(cheap, 2),
                        honest_gain=round(g, 2), d_eff=de, score=round(score, 2)))
        print(f"  finalist {sorted(pairs)}: honest_gain={g:.2f} score={score:.2f}",
              flush=True)
    out.sort(key=lambda z: -z["score"])
    return out


def deploy_with_set(card, admitted, Xtr, ytr, kind, n_classes, pairs):
    """Штатный warm6 деплой: базовые атомы карты + набор S, warm_card=мейны."""
    mains_only = card
    # атомы: переобучение базовых (конвенция calib_factory) + новые пары
    o = np.asarray(mains_only.score(Xtr), float)
    aa = []
    for (at_, j, k) in admitted:
        try:
            at = R.make_atom77(at_, j, k, mains_only, X=Xtr)
        except (IndexError, ValueError):
            continue
        at.ensure(kind, n_classes)
        at.refit(Xtr, ytr, o, kind, kappa_mult=1.0, n_classes=n_classes)
        o = o + at.contrib(Xtr)
        aa.append(at)
    for (j, k) in sorted(pairs):
        at = _mk_p1(j, k, Xtr)
        at.ensure(kind, n_classes=n_classes)
        at.refit(Xtr, ytr, o, kind, kappa_mult=1.0)
        o = o + at.contrib(Xtr)
        aa.append(at)
    cm2 = T._fitj(Xtr, ytr, kind, atoms=aa, warm_card=mains_only,
                  passes=T.FITCFG["warm_passes"], conv_tol=0.0)
    return cm2
