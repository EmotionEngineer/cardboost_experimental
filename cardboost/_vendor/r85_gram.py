#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r85_gram.py -- R85 Canary 1: Gram-Spectral Collinearity Damping (Пакет 1 #7 / Пакет 3 #1).

Замена слепого p-law (lr=km=min(1,sqrt(10/p))) на индивидуальные шаг/цену признака.
Спецификация ЗАМОРОЖЕНА (2026-11-15, голосование пользователя):
  * rho_jk = |Spearman| по совместным не-NaN строкам (X-only, БЕЗ y);
  * d_j    = 1 + SUM_{k!=j} max(0, rho_jk - rho0)^2,  rho0 = 0.2;
  * lr_j = km_j = min(1.0, sqrt(2.0/d_j));
  * passes = min(48, ceil(24 / min_j lr_j)), conv_tol = 1e-5 * min_j lr_j
    (обобщение R78/R79 через самый медленный признак; при min_j lr_j == 1.0
    ветка не срабатывает -- путь кода бит-в-бит как при lr=1.0).

Измеренные lr_j (диагностика r85_probe_corr.py):
  cal  (p=8):   8/8 == 1.0 при ЛЮБОМ rho0  -> бит-идентичность card_v7_H3 структурна;
  anneal (p=38): min .905 / 35 из 38 == 1.0 (rho0=0.2);
  aps (p=170):   min .218, p25 .240 (rho0=0.2) -- плотные гистограммные блоки.
MEAN-вариант d_j признан инертным (d_j <= 1.25 -> lr_j == 1.0 везде) и исключён.
"""
import numpy as np

RHO0 = 0.2


def _rank_cols(X):
    """Average ranks per column over finite entries (NaN -> nan)."""
    n, p = X.shape
    R = np.full((n, p), np.nan)
    for j in range(p):
        col = X[:, j]
        fin = np.isfinite(col)
        if not fin.any():
            continue
        c = col[fin]
        order = np.argsort(c, kind="stable")
        ranks = np.empty(len(c), float)
        ranks[order] = np.arange(len(c), dtype=float)
        sc = c[order]
        i = 0
        while i < len(sc):
            k = i + 1
            while k < len(sc) and sc[k] == sc[i]:
                k += 1
            if k > i + 1:
                avg = 0.5 * (i + k - 1)
                for t in range(i, k):
                    ranks[order[t]] = avg
            i = k
        R[fin, j] = ranks + 1.0
    return R


def gram_lr_vec(X, rho0=RHO0):
    """(lr_vec, km_vec) per feature; X-only.  len == X.shape[1]."""
    X = np.asarray(X, float)
    n, p = X.shape
    if p < 2:
        return np.ones(p), np.ones(p)
    R = _rank_cols(X)
    rho = np.zeros((p, p))
    for j in range(p):
        for k in range(j + 1, p):
            m = np.isfinite(R[:, j]) & np.isfinite(R[:, k])
            if m.sum() < 3:
                r = 0.0
            else:
                a = R[m, j] - R[m, j].mean()
                b = R[m, k] - R[m, k].mean()
                den = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
                r = float((a * b).sum() / den) if den > 0 else 0.0
            rho[j, k] = rho[k, j] = abs(r)
    d = np.ones(p)
    for j in range(p):
        exc = np.maximum(0.0, rho[j] - rho0)
        exc[j] = 0.0
        d[j] = 1.0 + float((exc ** 2).sum())
    lr = np.minimum(1.0, np.sqrt(2.0 / d))
    return lr, lr.copy()   # km_j == lr_j по спеке
