"""r87_safetier.py -- R87 Canary 5 (R113): safe_value_tier (Atom-Owned Marginals).

ЗАМОРОЖЕННАЯ СПЕЦИФИКАЦИЯ (план юзера 2026-11-18, Post-warm6 Safe Tier):
  Поверхности S_j x S_k (p1, сетка <=8x8, L10) учатся ПОСЛЕ warm6 на
  ЗАМОРОЖЕННЫХ мейнах (off = score готовой карты card_v8_base).  После potts_dp
  таблица проецируется DYKSTRA-центрированием в точные нулевые маргинали
  (sum_bk H[b_j,b_k] A = 0 по каждой строке/столбцу) -- атом НЕ владеет
  маргиналями и не сдвигает края V_j.  Пустые/редкие ячейки (<MIN_COUNT) = 0:
  за пределами бокса 1[(x_j,x_k) in B_jk] вклад ТОЖДЕСТВЕННО нулевой
  (spat f0/f4 не могут быть уронены по построению).  bias-safe.
Приём (post-fit standalone, честный OOF):
  Score = gain(dL_OOF в натсах) - 0.5*d_eff*ln n - ln M ;  d_eff считается
  консервативно (полная решётка+1) при отборе и точно при логе;
  M = размер пула кандидатов (плата за поиск).  5 субфолдов seed=109
  (конвенция IsoLink); regression nats = dSSE/(2*sigma2_oof), binary =
  логит-прирост.  K_MAX=4, greedy по убыванию честного gain.
  multiclass (anneal) -- молчит (бит-в-бит нейтрален).
Гейты (frozen): cal_rand dec >= .8100; cal_spat dec >= .5034; f4 >= +.061;
  anneal бит-в-бит (механизм молчит); кошелёк <= 4096 (запас card_v8_base
  2888->4096 = 1208 B); moved=0, diff==0.
"""
from __future__ import annotations
import numpy as np

import r71_universal as r71
import r77_mech as R


P1_BINS_SAFE = 8      # L10: решётки >8x8 убивают spat f4
K_POOL = 24           # платим ln(M) за поиск по пулу
K_MAX = 4             # максимум атомов на фолд
SUBFOLDS = 5
SEED = 109
MIN_COUNT = 5         # ячейка без массы -> 0 (бокс-гейт)


# ------------------------------------------------------------------ метрики
def nats_gain(kind, y, z0, z1, sigma2=None):
    """Прирост правдоподобия z1 над z0 в натсах (конвенция CE.nats_gain)."""
    y = np.asarray(y, float)
    z0 = np.asarray(z0, float)
    z1 = np.asarray(z1, float)
    if kind == "regression":
        s2 = float(sigma2) if sigma2 else max(float(np.mean((y - z0) ** 2)), 1e-9)
        return float(np.sum((y - z0) ** 2 - (y - z1) ** 2) / (2.0 * max(s2, 1e-9)))
    if kind == "binary":
        return float(np.sum(y * (z1 - z0)
                            - (np.logaddexp(0.0, z1) - np.logaddexp(0.0, z0))))
    return float("-inf")   # multiclass: механизм молчит


# ------------------------------------------------------- Dykstra-центрирование
def dykstra_zero_marginals(V, H, iters=300, tol=1e-13):
    """Проекция V в {строковые маргиналы = 0, столбцовые = 0} по весам H.

    Dykstra для двух аффинных множеств -> метрическая проекция: атом не владеет
    ни одной маргиналью, края мейнов неподвижны.
    """
    A = np.asarray(V, float).copy()
    W = np.maximum(np.asarray(H, float), 0.0)
    p = np.zeros_like(A)
    q = np.zeros_like(A)
    wr = np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    wc = np.maximum(W.sum(axis=0, keepdims=True), 1e-12)
    for _ in range(iters):
        t = A + p
        yv = t - (t * W).sum(axis=1, keepdims=True) / wr      # P_row
        p = t - yv
        t2 = yv + q
        An = t2 - (t2 * W).sum(axis=0, keepdims=True) / wc    # P_col
        q = t2 - An
        if float(np.max(np.abs(An - A))) < tol:
            A = An
            break
        A = An
    return A


def cell_counts(X, at):
    a, b, _ins = at._cells(X)
    H = np.bincount(a * at.Bn + b, minlength=at.A * at.Bn).astype(float)
    return H.reshape(at.A, at.Bn)


def safe_project(at, X, min_count=MIN_COUNT):
    """Dykstra-обнуление маргиналей + бокс-гейт пустых ячеек (после refit)."""
    H = cell_counts(X, at)
    holes = H < min_count
    if getattr(at, "Vc", None) is not None:
        for c in range(at.Vc.shape[2]):
            V = dykstra_zero_marginals(at.Vc[:, :, c], H)
            V[holes] = 0.0
            at.Vc[:, :, c] = V
    else:
        V = dykstra_zero_marginals(at.V, H)
        V[holes] = 0.0
        at.V = V
    return at


def d_eff(at):
    """Эффективное число параметров (live-ячейки + 1) -- для преq-платы."""
    V = at.Vc if getattr(at, "Vc", None) is not None else at.V
    live = int(np.count_nonzero(np.abs(V) > 1e-12))
    return live + 1


def _mk_p1(j, k, X):
    """P1Atom с собственной квантильной сеткой 8x8 (make_atom77 без карты)."""
    X = np.asarray(X, float)
    xs_j = np.sort(X[:, j][np.isfinite(X[:, j])])
    xs_k = np.sort(X[:, k][np.isfinite(X[:, k])])
    cj = R.cuts_of(np.asarray(X[:, j], float), P1_BINS_SAFE)
    ck = R.cuts_of(np.asarray(X[:, k], float), P1_BINS_SAFE)
    at = R.P1Atom(j, k, cj, ck, xs_j, xs_k)
    at.pair_bins = P1_BINS_SAFE
    return at


# ------------------------------------------------------------- честный OOF
def oof_base(Xtr, ytr, kind, n_classes, folds=SUBFOLDS, seed=SEED, cache=None):
    """OOF-прогноз базы (конвенция IsoLink r83_3: субфолды seed=109, мейны).

    cache: каталог для eta_oof_{kind}.npy -- пересчёт 5 субфолд-фитов не
    дёргается повторно (смоук/механики/повторные прогоны одного фолда).
    """
    import os
    import r77_stage as T
    n = len(ytr)
    rng = np.random.default_rng(seed)
    parts = [np.asarray(p, int) for p in np.array_split(rng.permutation(n), folds)]
    pth = os.path.join(cache, f"eta_oof_{kind}.npy") if cache else None
    if pth and os.path.exists(pth):
        eta_oof = np.load(pth)
        if len(eta_oof) == n:
            print(f"  oof_base cache hit {pth}", flush=True)
            return eta_oof, parts
    eta_oof = np.zeros(n)
    for val in parts:
        ftr = np.setdiff1d(np.arange(n), val)
        m = T._fitj(Xtr[ftr], ytr[ftr], kind, atoms=[], n_classes=n_classes)
        eta_oof[val] = np.asarray(m.score(Xtr[val]), float)
    if pth:
        os.makedirs(cache, exist_ok=True)
        np.save(pth, eta_oof)
    return eta_oof, parts


def crossfit_gain(j, k, Xtr, ytr, kind, eta_oof, parts, sigma2=None):
    """Честный gain поверхности: A_s обучается на вне-s, считается на s."""
    n = len(ytr)
    tot = 0.0
    for val in parts:
        ftr = np.setdiff1d(np.arange(n), val)
        at2 = _mk_p1(j, k, Xtr)              # сетка заморожена по всему Xtr
        at2.ensure(kind, n_classes=0)
        at2.refit(Xtr[ftr], ytr[ftr], np.asarray(eta_oof[ftr], float), kind,
                  kappa_mult=1.0)
        safe_project(at2, Xtr[ftr])
        z1 = np.asarray(eta_oof[val], float) + at2.contrib(Xtr[val])
        tot += nats_gain(kind, ytr[val], eta_oof[val], z1, sigma2)
    return float(tot)


# --------------------------------------------------------------- кандидаты
def tier_candidates(card, X, y, kind, K=K_POOL):
    """Пул кандидатов: pair_screen по остатку ГОТОВОЙ карты минус существующие."""
    import r76_stage31 as S31
    base = np.asarray(card.score(X), float)
    pairs = S31.pair_screen(X, y, base, kind, K * 2, bins=P1_BINS_SAFE)
    have = set()
    for a in card.atoms:
        if hasattr(a, "j") and hasattr(a, "k"):
            have.add((int(a.j), int(a.k)))
            have.add((int(a.k), int(a.j)))
    out = [(j, k) for (j, k) in pairs if (j, k) not in have]
    return out[:K], base


# ------------------------------------------------------------------ сборка
def build_safe_tier(card, X, y, kind, n_classes, K=K_POOL, k_max=K_MAX, cache=None):
    """Post-warm6 Safe Tier: обучение, честный отбор, refit.

    Возвращает (atoms, meta); atoms НЕ добавляются в карту автоматически.
    """
    meta = dict(tier="safe_value", kind=kind, admitted=[], suppressed=[],
                cand=0, gain=0.0)
    if kind == "multiclass":
        meta["note"] = "skip multiclass (молчит)"
        return [], meta
    pairs, base = tier_candidates(card, X, y, kind, K)
    meta["cand"] = len(pairs)
    if not pairs:
        return [], meta
    n = len(y)
    eta_oof, parts = oof_base(X, y, kind, n_classes, cache=cache)
    sigma2 = float(np.mean((y - eta_oof) ** 2)) if kind == "regression" else None
    M = max(len(pairs), 1)
    lnM = float(np.log(M))
    lnbar = 0.5 * float(np.log(n))
    de_bound = P1_BINS_SAFE * P1_BINS_SAFE + 1    # консервативная верхняя d_eff

    scored = []
    for (j, k) in pairs:
        g = crossfit_gain(j, k, X, y, kind, eta_oof, parts, sigma2)
        score = g - 0.5 * de_bound * lnbar - lnM
        scored.append((float(score), float(g), int(j), int(k)))
        print(f"  tier cand ({j},{k}): gain={g:.2f} score={score:.2f}", flush=True)
    scored.sort(reverse=True)

    atoms, o = [], base.copy()
    for (score, g, j, k) in scored:
        if score <= 0 or len(atoms) >= k_max:
            meta["suppressed"].append(dict(j=j, k=k, gain=round(g, 2),
                                           score=round(score, 2)))
            continue
        at = _mk_p1(j, k, X)
        at.ensure(kind, n_classes=n_classes)
        at.refit(X, y, o, kind, kappa_mult=1.0)
        safe_project(at, X)
        o = o + at.contrib(X)
        atoms.append(at)
        meta["admitted"].append(dict(j=j, k=k, gain=round(g, 2),
                                     score=round(score, 2), d_eff=d_eff(at)))
        meta["gain"] += g
    meta["gain"] = round(float(meta["gain"]), 2)
    return atoms, meta
