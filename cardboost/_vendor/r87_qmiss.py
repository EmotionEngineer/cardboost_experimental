"""r87_qmiss.py -- R87 Finalist 4 (R112) qmiss_coord: атом координации пропусков.

ЗАМОРОЖЕННАЯ СПЕЦИФИКАЦИЯ (голосование 2026-11-17, R87 «Точка невозврата»):
  q_miss(x) = Σ_j 1[x_j = NaN] · |V_j[b_NaN]|  -- «сила реакции модели на
  пропуски» строки.  16 квантилей q_miss -> 1D ступень через potts_dp (как
  msum).  Приём: preq-барьер ½d ln n + ln 2 при d=1 (gain по Δℓ_OOF на
  toof_eta-логитах, те же 5 субфолдов seed=109, что для Tcal).  Молчит без NaN
  (вся панель без пропусков -> атом не строится -> бит-в-бит).  ~32 B на вайре
  (tag 10: u_lo/u_hi f32 + активные разрезы f32 + полки f32; w_j = |V_j[b_NaN]|
  восстанавливается декодером из мейнов -> 0 B на w).

  Конструкция self-contained (паттерн MSum/R83 IsoLink): на входе атома --
  квантизованная (int8-mode) реконструкция NaN-полок мейнов, ровно то, что
  декодер увидит в мейнах -> u бит-в-бит совпадает на train и deploy.
  Binary/regression-only (anneal multiclass молчит -> бит-в-бит guard).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/user/google_drive")

import r71_universal as r71
from r71_universal import bin_idx, onehot, softmax
from r76_mech import wsegment_levels, wsolve_bin_multiclass
from r77_mech import wsolve_bin_logistic, wsolve_bin_regression


class QMissAtom:
    """qmiss: 1D таблица на q_miss(x) = isnan(x)·w, w_j = |V_j[b_NaN]| (frozen)."""
    kind = "qmiss"

    def __init__(self, w, u_lo, u_hi, nbins=16):
        self.w = np.asarray(w, float)
        self.u_lo = float(np.float32(u_lo))
        self.u_hi = float(np.float32(u_hi))
        self.nbins = int(nbins)
        self.ucuts = None
        self.V = None
        self.Vc = None
        self.dim = 1

    def _u(self, X):
        X = np.asarray(X, float)
        u = np.isnan(X) @ self.w
        u = np.clip(u, self.u_lo, self.u_hi)   # value-space clamp (R105-паттерн)
        return u, np.ones(len(X), bool)

    def _snap_cuts(self, uu):
        du = np.unique(uu)
        if du.size < 4:
            return np.zeros(0)
        qs = np.quantile(uu, np.linspace(0.0, 1.0, self.nbins + 2)[1:-1])
        cuts = []
        for c in qs:
            lo = du[du < c].max() if (du < c).any() else None
            hi = du[du > c].min() if (du > c).any() else None
            if lo is None or hi is None or hi <= lo:
                continue
            cuts.append(0.5 * (float(lo) + float(hi)))
        cuts = sorted(set(cuts))
        return np.asarray(np.float32(cuts), float) if cuts else np.zeros(0)

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1

    def contrib(self, X):
        u, ins = self._u(X)
        if self.V is None and self.Vc is None:
            return np.zeros((len(X), self.dim)) if self.dim > 1 else np.zeros(len(X))
        b = bin_idx(u, self.ucuts) if self.ucuts is not None and len(self.ucuts) \
            else np.zeros(len(X), int)
        b = np.where(ins, b, -1)
        m = b >= 0
        if self.Vc is not None:
            out = np.zeros((len(X), self.Vc.shape[1]))
            out[m] = self.Vc[b[m]]
            return out
        out = np.zeros(len(X))
        out[m] = self.V[b[m]]
        return out

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        if kind == "multiclass":
            return off + self.contrib(X)      # не поддержан (anneal молчит)
        n = len(y)
        if w is None:
            w = np.ones(n)
        u, ins = self._u(X)
        cur = self.contrib(X)
        o = off - cur
        if self.ucuts is None:
            self.ucuts = self._snap_cuts(u[ins]) if ins.any() else np.zeros(0)
        b = bin_idx(u, self.ucuts) if len(self.ucuts) else np.zeros(n, int)
        b = np.where(ins, b, -1)
        sel = b >= 0
        if sel.sum() < 10:
            return o + self.contrib(X)
        nb = (len(self.ucuts) + 2) if len(self.ucuts) else 2
        if kind == "binary":
            z, H = wsolve_bin_logistic(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        else:
            z, H = wsolve_bin_regression(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        emp = H <= 1e-12
        zc = np.where(emp, 0.0, z)
        kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
        lab = r71.potts_dp(zc, np.where(emp, 1e-9, H), kap)
        self.V = wsegment_levels(lab, zc, H, "binary" if kind == "binary" else "regression")
        return o + self.contrib(X)

    def n_params(self):
        V = self.V if self.V is not None else np.zeros(0)
        return int(np.sum(np.diff(V) != 0)) + 1 if len(V) else 0

    def snapshot(self):
        V = self.V if self.V is not None else np.zeros(1)
        return np.asarray(V, float).ravel().copy()


def quantized_nan_weights(card, mode="int8"):
    """w_j = |квантизованная NaN-полка мейна j| -- РОВНО то, что декодер увидит
    в мейнах (emit quantize_levels(mode) реконструкция).  Для j без мейнов/NaN-
    полки -- 0.  Возвращает (w, any_nan_flag)."""
    import r76_stage5 as T
    p = len(card.cuts)
    w = np.zeros(p)
    for j in range(p):
        V = np.asarray(card.V[j], float)
        if V.ndim == 2:                      # multiclass -- не используется
            continue
        nb = len(V)
        if nb != len(card.cuts[j]) + 2:
            continue
        q, scale, _q0, _recon = T.quantize_levels(V, mode)
        # Декодер умножает коды на scale, РАСПАКОВАННЫЙ из f32-вайра; считаем
        # ровно так же -> w бит-в-бит совпадает по обе стороны.
        w[j] = abs(float(int(q[nb - 1])) * float(np.float32(scale)))
    return w


def build_atom(card, Xtr, ytr, kind, n_classes=0):
    """Строит qmiss-атом по карточке; None, если панель без NaN или q_miss
    вырожден (молчит без NaN -- заморожено)."""
    Xtr = np.asarray(Xtr, float)
    if not np.isnan(Xtr).any():
        return None, dict(admitted=False, why="no-nan")
    w = quantized_nan_weights(card)
    if float(np.abs(w).sum()) <= 0.0:
        return None, dict(admitted=False, why="zero-w")
    u = np.isnan(Xtr) @ w
    u_lo, u_hi = float(u.min()), float(u.max())
    at = QMissAtom(w, u_lo, u_hi, nbins=16)
    at.ensure(kind, n_classes)
    return at, dict(w_sum=float(np.abs(w).sum()), u_lo=u_lo, u_hi=u_hi)
