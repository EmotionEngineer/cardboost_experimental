"""r84_p1swap.py -- R84 Canary 2: 1-к-1 замена слота p1 в W1 поверх card_v7_H3 (Law L10).

Рука 2A  P1Swap2A  "Nested Partition-Conditional Atom S_j x S_k":
  * ось полок  -- склеенные полки мейна (границы = H3-пороги), хранятся f16-значениями
    границ (self-contained: мёртвые каты контейнера вырезаются, ссылка на мейн ненадёжна);
  * ось столбцов -- существующая 32-квантильная сетка E[k], векторный potts_dp вдоль неё
    ВНУТРИ pair_ctx (базовая цена kappa, L10); направление j->k vs k->j выбирается
    in-fold OOF (цена ln2 берётся в леджере драйвером);
  * ячейки: Haldane z = G/(H+0.5); 8 проходов H-взвешенного RSR (row/col demean);
    ячейки с поддержкой n < 25 -> 0; строгий бокс-гейт.
Рука 2B  P1Swap2B  "Exact Guillotine-DP 16x16":
  * квантильная сетка 16x16 (собственные каты, f16); точное DP по всем гильотинным
    разбиениям, dp(R) = min(cost_leaf(R), min_split dp+dp+kappa_pair), kappa_pair =
    0.5*ln N (базовая цена, pair_ctx); любой дочерний прямоугольник с n < 25 запрещён
    (+inf) -- пустые полосы не отщепляются (иммунитет к R76);
  * листья -> таблица 16x16 -> 8 проходов H-взвешенного RSR -> перепроекция на листья,
    бокс-гейт.
Сериализация (r76_container31, kind 7=2A / 8=2B): границы/каты f16 (квантизация на фите),
уровни f16, боксы f32-floor.  kind != 'regression' не поддержан (канарейка cal-only).
"""
from __future__ import annotations

import numpy as np

import r71_universal as U
from r71_universal import bin_idx
from r77_mech import _support, EPS

F16 = np.float16


def _f16(a):
    return np.asarray(a, F16).astype(float)


def _f32(v):
    return float(np.float32(v))


def _main_shelf_bounds(card, f):
    """Границы склеенных полок мейна f: позиции смены уровня (квантизованы f16)."""
    Vf = np.asarray(card.V[f], float)
    cuts = np.asarray(card.cuts[f], float)
    nreal = len(cuts) + 1
    if Vf.ndim != 1:
        raise NotImplementedError("p1swap: multiclass mains not supported")
    if nreal < 2 or len(Vf) < nreal:
        return np.zeros(0)
    brk = Vf[:nreal - 1] != Vf[1:nreal]
    vals = cuts[:nreal - 1][brk]
    return _f16(vals)


def _cell_stats(X, y, o, w, n, rows, cols, R, C):
    """G, H, N на сетке (R, C): суммы w*(y-o), w и счётчики."""
    flat = rows * C + cols
    nb = R * C
    if w is None:
        w = np.ones(n)
    G = np.bincount(flat, weights=w * (y - o), minlength=nb).reshape(R, C)
    H = np.bincount(flat, weights=w, minlength=nb).reshape(R, C)
    N = np.bincount(flat, minlength=nb).reshape(R, C)
    return G, H, N


class P1Swap2A:
    """Partition-conditional S_a x S_b atom; полки мейна x склеенная 32-сетка."""

    kind = "p1s2a"

    def __init__(self, j, k, card, X):
        self.j, self.k = int(j), int(k)
        self.dir = 0                      # 0: полки по j, столбцы по k; 1: наоборот
        self.dir_chosen = False
        self.V = None
        self.bnd_a = np.zeros(0)          # границы полок (ось полок)
        self.bnd_b = np.zeros(0)          # склеенные границы столбцов
        self.shelf_from_main = {self.j: _main_shelf_bounds(card, self.j),
                                self.k: _main_shelf_bounds(card, self.k)}
        # 32-сетки столбцов (существующие каты мейнов, f16)
        self.grid32 = {}
        for f in (self.j, self.k):
            cf = np.asarray(card.cuts[f], float)
            self.grid32[f] = _f16(cf)
        xsj = np.sort(X[:, self.j][np.isfinite(X[:, self.j])])
        xsk = np.sort(X[:, self.k][np.isfinite(X[:, self.k])])
        self.xmin_j = _f32(xsj[0]) if len(xsj) else 0.0
        self.xmax_j = _f32(xsj[-1]) if len(xsj) else 0.0
        self.xmin_k = _f32(xsk[0]) if len(xsk) else 0.0
        self.xmax_k = _f32(xsk[-1]) if len(xsk) else 0.0

    # ---- protocol ----------------------------------------------------------
    def ensure(self, kind, n_classes=0):
        if kind != "regression":
            raise NotImplementedError("p1swap: canary 2 is kind=='regression' only")
        self.dim = 1

    def n_params(self):
        if self.V is None:
            return 1
        return int(np.sum(np.any(np.diff(self.V, axis=0) != 0, axis=1))
                   + np.sum(np.any(np.diff(self.V, axis=1) != 0, axis=0))) + 1

    def snapshot(self):
        return self.V.ravel().copy() if self.V is not None else np.zeros(1)

    def _axes(self):
        return (self.j, self.k) if self.dir == 0 else (self.k, self.j)

    def _lookup(self, X, dir_, bnd_a, bnd_b, V):
        fa, fc = (self.j, self.k) if dir_ == 0 else (self.k, self.j)
        xj = X[:, self.j]
        xk = X[:, self.k]
        ins = _support(xj, self.xmin_j, self.xmax_j) & _support(xk, self.xmin_k, self.xmax_k)
        out = np.zeros(len(X))
        if V is None or not ins.any():
            return out
        xa = X[:, fa]
        xc = X[:, fc]
        sa = np.searchsorted(bnd_a, xa, side="right")
        sc = np.searchsorted(bnd_b, xc, side="right")
        sa = np.clip(sa, 0, V.shape[0] - 1)
        sc = np.clip(sc, 0, V.shape[1] - 1)
        out[ins] = V[sa[ins], sc[ins]]
        return out

    def contrib(self, X):
        return self._lookup(X, self.dir, self.bnd_a, self.bnd_b, self.V)

    # ---- fitting -----------------------------------------------------------
    def _fit_table(self, X, y, o, w, dir_, kappa_mult):
        fa, fc = (self.j, self.k) if dir_ == 0 else (self.k, self.j)
        bnd_shelf = self.shelf_from_main[fa]
        grid = self.grid32[fc]
        xa = X[:, fa]
        xc = X[:, fc]
        ins = (_support(X[:, self.j], self.xmin_j, self.xmax_j)
               & _support(X[:, self.k], self.xmin_k, self.xmax_k))
        rows = np.searchsorted(bnd_shelf, xa, side="right")
        cols = np.searchsorted(grid, xc, side="right")
        R = len(bnd_shelf) + 1
        Cc = len(grid) + 1
        m = ins
        if w is None:
            w = np.ones(len(y))
        G, H, N = _cell_stats(X[m], y[m], o[m], w[m], int(m.sum()),
                              rows[m], cols[m], R, Cc)
        z = G / (H + 0.5)
        # векторный potts вдоль оси столбцов ВНУТРИ pair_ctx (базовая цена, L10)
        U.R84_COST["pair_ctx"] += 1
        try:
            Cmat = z.T.copy()                       # (Cc, R)
            Hc = H.sum(axis=0)                      # (Cc,)
            kap = 0.5 * np.log(max(float(H.sum()), 1.0)) * kappa_mult
            lab = U.potts_dp(np.where(Hc[:, None] > 0, Cmat, 0.0),
                             np.where(Hc > 0, Hc, 1e-9), kap)
        finally:
            U.R84_COST["pair_ctx"] -= 1
        # склейка: границы = каты, где метки меняются
        keep = np.zeros(max(len(grid) - 1, 0), bool)
        for c in range(len(grid) - 1):
            keep[c] = lab[c] != lab[c + 1]
        bnd_b = grid[:len(keep)][keep] if len(keep) else np.zeros(0)
        seg = np.searchsorted(np.where(keep)[0], np.arange(Cc), side="left")
        Sb = int(seg.max()) + 1 if Cc else 1
        Sa = R
        # финальные значения: H-взвешенное среднее ячеек по (полка, сегмент)
        T = np.zeros((Sa, Sb))
        Wt = np.zeros((Sa, Sb))
        Nt = np.zeros((Sa, Sb))
        for s in range(Sb):
            sel = seg == s
            Wt[:, s] = H[:, sel].sum(axis=1)
            Nt[:, s] = N[:, sel].sum(axis=1)
            hh = np.maximum(Wt[:, s], 1e-12)
            T[:, s] = (H[:, sel] * z[:, sel]).sum(axis=1) / hh
        T[Wt <= 0] = 0.0
        # 8 проходов H-взвешенного RSR (row/col demean против мейнов)
        for _ in range(8):
            rm = (T * Wt).sum(axis=1) / np.maximum(Wt.sum(axis=1), 1e-12)
            T = T - rm[:, None]
            cm = (T * Wt).sum(axis=0) / np.maximum(Wt.sum(axis=0), 1e-12)
            T = T - cm[None, :]
        T[Nt < 25] = 0.0                            # жёсткая поддержка ячейки
        T = _f16(T)
        return bnd_shelf, bnd_b, T

    def _choose_dir(self, X, y, o, w):
        n = len(y)
        rng = np.random.default_rng(109)
        parts = [np.asarray(p, int) for p in np.array_split(rng.permutation(n), 5)]
        best_dir, best_sse = 0, np.inf
        for d in (0, 1):
            sse = 0.0
            for s in range(5):
                va = parts[s]
                tr = np.concatenate([parts[g] for g in range(5) if g != s])
                try:
                    ba, bb, V = self._fit_table(X[tr], y[tr], o[tr],
                                                None if w is None else w[tr], d, 1.0)
                except (IndexError, ValueError):
                    sse = np.inf
                    break
                pred = self._lookup(X[va], d, ba, bb, V)
                r = y[va] - o[va] - pred
                sse += float((r * r).sum() if w is None else (w[va] * r * r).sum())
            if sse < best_sse:
                best_dir, best_sse = d, sse
        return best_dir

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        o = np.asarray(off, float) - self.contrib(X)
        if not self.dir_chosen:
            self.dir = self._choose_dir(X, y, o, w)
            self.dir_chosen = True
        ba, bb, T = self._fit_table(X, y, o, w, self.dir, kappa_mult)
        self.bnd_a, self.bnd_b, self.V = ba, bb, T
        return o + self.contrib(X)

    # ---- wire --------------------------------------------------------------
    def wire(self):
        out = bytearray()
        out += np.uint8(self.dir).tobytes()
        out += np.uint8(self.V.shape[0]).tobytes() + np.uint8(self.V.shape[1]).tobytes()
        out += np.asarray([self.xmin_j, self.xmax_j, self.xmin_k, self.xmax_k],
                          np.float32).tobytes()
        out += np.uint8(len(self.bnd_a)).tobytes() + np.uint8(len(self.bnd_b)).tobytes()
        out += np.asarray(self.bnd_a, F16).tobytes()
        out += np.asarray(self.bnd_b, F16).tobytes()
        out += np.asarray(self.V, F16).tobytes()
        return bytes(out)

    @classmethod
    def unwire(cls, j, k, buf):
        import struct
        off = 0

        def take(n):
            nonlocal off
            b = bytes(buf[off:off + n])
            off += n
            return b
        a = cls.__new__(cls)
        a.j, a.k = int(j), int(k)
        a.dir = take(1)[0]
        a.dir_chosen = True
        na, nb = take(1)[0], take(1)[0]
        a.xmin_j, a.xmax_j, a.xmin_k, a.xmax_k = struct.unpack("<ffff", take(16))
        nba, nbb = take(1)[0], take(1)[0]
        a.bnd_a = np.frombuffer(take(2 * nba), F16).astype(float).copy()
        a.bnd_b = np.frombuffer(take(2 * nbb), F16).astype(float).copy()
        a.V = np.frombuffer(take(2 * na * nb), F16).astype(float).reshape(na, nb).copy()
        a.dim = 1
        return a, off


class P1Swap2B:
    """Guillotine-DP 16x16 с листьями n >= 25 (полный иммунитет к R76 empty-strip)."""

    kind = "p1s2b"

    def __init__(self, j, k, card, X):
        self.j, self.k = int(j), int(k)
        xsj = np.sort(X[:, self.j][np.isfinite(X[:, self.j])])
        xsk = np.sort(X[:, self.k][np.isfinite(X[:, self.k])])
        self.xmin_j = _f32(xsj[0]) if len(xsj) else 0.0
        self.xmax_j = _f32(xsj[-1]) if len(xsj) else 0.0
        self.xmin_k = _f32(xsk[0]) if len(xsk) else 0.0
        self.xmax_k = _f32(xsk[-1]) if len(xsk) else 0.0
        self.cj = _f16(U.cuts_of(X[:, self.j], 16))
        self.ck = _f16(U.cuts_of(X[:, self.k], 16))
        self.leaves = []                   # [(r1, r2, c1, c2, value)]
        self.lmap = None

    def ensure(self, kind, n_classes=0):
        if kind != "regression":
            raise NotImplementedError("p1swap: canary 2 is kind=='regression' only")
        self.dim = 1

    def n_params(self):
        return len(self.leaves) + 1

    def snapshot(self):
        if not self.leaves:
            return np.zeros(1)
        return np.asarray([lv[4] for lv in self.leaves], float)

    def _build_lmap(self):
        R = len(self.cj) + 1
        C = len(ck := self.ck) + 1
        lmap = np.full((R, C), -1, int)
        for i, (r1, r2, c1, c2, _v) in enumerate(self.leaves):
            lmap[r1:r2, c1:c2] = i
        self.lmap = lmap

    def contrib(self, X):
        out = np.zeros(len(X))
        if not self.leaves:
            return out
        if self.lmap is None:
            self._build_lmap()
        xj = X[:, self.j]
        xk = X[:, self.k]
        ins = _support(xj, self.xmin_j, self.xmax_j) & _support(xk, self.xmin_k, self.xmax_k)
        if not ins.any():
            return out
        r = np.searchsorted(self.cj, xj[ins], side="right")
        c = np.searchsorted(self.ck, xk[ins], side="right")
        li = self.lmap[np.clip(r, 0, self.lmap.shape[0] - 1),
                       np.clip(c, 0, self.lmap.shape[1] - 1)]
        vals = np.zeros(len(li))
        ok = li >= 0
        if ok.any():
            vals[ok] = [self.leaves[t][4] for t in li[ok]]
        out[ins] = vals
        return out

    # ---- guillotine DP -----------------------------------------------------
    @staticmethod
    def _guillotine(G, H, N, kap):
        """dp по гильотинным разбиениям сетки (R, C).  Возвращает (leaves, T, Wm)."""
        R, C = N.shape
        pref = lambda A: np.pad(np.cumsum(np.cumsum(A, 0), 1), ((1, 0), (1, 0)))
        PN, PH, PG, PGG = pref(N), pref(H), pref(G), pref(G * G)

        def rect(P, r1, r2, c1, c2):
            return (P[r2, c2] - P[r1, c2] - P[r2, c1] + P[r1, c1])

        def leaf_cost(r1, r2, c1, c2):
            n = rect(PN, r1, r2, c1, c2)
            if n < 25:
                return np.inf
            h = rect(PH, r1, r2, c1, c2)
            g = rect(PG, r1, r2, c1, c2)
            gg = rect(PGG, r1, r2, c1, c2)
            mu = g / (h + 0.5)
            return max(gg - 2.0 * mu * g + mu * mu * h, 0.0)

        # dp[dr][dc] -> 2D массив по (r1, c1); arg splits параллельно
        dp = [[None] * (C + 1) for _ in range(R + 1)]
        ar = [[None] * (C + 1) for _ in range(R + 1)]
        ac = [[None] * (C + 1) for _ in range(R + 1)]
        for dr in range(1, R + 1):
            for dc in range(1, C + 1):
                best = np.full((R - dr + 1, C - dc + 1), np.inf)
                abr = np.zeros_like(best, np.int16)
                abc = np.zeros_like(best, np.int16)
                for r1 in range(R - dr + 1):
                    for c1 in range(C - dc + 1):
                        best[r1, c1] = leaf_cost(r1, r1 + dr, c1, c1 + dc)
                for s in range(1, dr):
                    cand = dp[s][dc][:R - dr + 1, :] + dp[dr - s][dc][s:, :]
                    upd = cand < best
                    if upd.any():
                        best = np.where(upd, cand, best)
                        abr = np.where(upd, s, abr)
                for s in range(1, dc):
                    cand = dp[dr][s][:, :C - dc + 1] + dp[dr][dc - s][:, s:]
                    upd = cand < best
                    if upd.any():
                        best = np.where(upd, cand, best)
                        abc = np.where(upd, s, abc)
                # формула пользователя: капа платится только за разрез
                dp[dr][dc] = best + kap * ((abr > 0) | (abc > 0))
                ar[dr][dc] = abr
                ac[dr][dc] = abc
        # вычет kap: в каждой вершине с разрезом ровно один kap (формула пользователя)
        leaves = []

        def rec(r1, r2, c1, c2):
            dr, dc = r2 - r1, c2 - c1
            if dr == 0 or dc == 0:
                return
            ab = ar[dr][dc][r1, c1]
            ab2 = ac[dr][dc][r1, c1]
            if ab > 0:
                rec(r1, r1 + ab, c1, c2)
                rec(r1 + ab, r2, c1, c2)
            elif ab2 > 0:
                rec(r1, r2, c1, c1 + ab2)
                rec(r1, r2, c1 + ab2, c2)
            else:
                n = rect(PN, r1, r2, c1, c2)
                h = rect(PH, r1, r2, c1, c2)
                g = rect(PG, r1, r2, c1, c2)
                mu = g / (h + 0.5) if n >= 25 else 0.0
                leaves.append((r1, r2, c1, c2, float(mu)))

        rec(0, R, 0, C)
        return leaves

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        o = np.asarray(off, float) - self.contrib(X)
        R, C = len(self.cj) + 1, len(self.ck) + 1
        xj = X[:, self.j]
        xk = X[:, self.k]
        ins = _support(xj, self.xmin_j, self.xmax_j) & _support(xk, self.xmin_k, self.xmax_k)
        rows = np.searchsorted(self.cj, xj, side="right")
        cols = np.searchsorted(self.ck, xk, side="right")
        if w is None:
            w = np.ones(len(y))
        G, H, N = _cell_stats(X, y, o, w, len(y), rows, cols, R, C)
        kap = 0.5 * np.log(max(float(N.sum()), 1.0)) * kappa_mult
        leaves = self._guillotine(G, H, N, kap)
        if not leaves:
            self.leaves = []
            self.lmap = None
            return o + self.contrib(X)
        # таблица листьев -> 8 проходов H-взвешенного RSR -> перепроекция на листья
        T = np.zeros((R, C))
        Wm = np.zeros((R, C))
        z = G / (H + 0.5)
        for (r1, r2, c1, c2, mu) in leaves:
            T[r1:r2, c1:c2] = mu
            Wm[r1:r2, c1:c2] = H[r1:r2, c1:c2]
        Tt = T.copy()
        for _ in range(8):
            rm = (Tt * Wm).sum(axis=1) / np.maximum(Wm.sum(axis=1), 1e-12)
            Tt = Tt - rm[:, None]
            cm = (Tt * Wm).sum(axis=0) / np.maximum(Wm.sum(axis=0), 1e-12)
            Tt = Tt - cm[None, :]
        out_leaves = []
        for (r1, r2, c1, c2, mu) in leaves:
            hh = Wm[r1:r2, c1:c2]
            v = float((Tt[r1:r2, c1:c2] * hh).sum() / max(hh.sum(), 1e-12))
            out_leaves.append((r1, r2, c1, c2, float(np.float16(v))))
        self.leaves = out_leaves
        self.lmap = None
        return o + self.contrib(X)

    # ---- wire --------------------------------------------------------------
    def wire(self):
        if len(self.leaves) > 250:
            raise ValueError("p1swap 2B: too many leaves for the u8 length")
        out = bytearray()
        out += np.uint8(len(self.leaves)).tobytes()
        out += np.asarray([self.xmin_j, self.xmax_j, self.xmin_k, self.xmax_k],
                          np.float32).tobytes()
        out += np.uint8(len(self.cj)).tobytes() + np.uint8(len(self.ck)).tobytes()
        out += np.asarray(self.cj, F16).tobytes()
        out += np.asarray(self.ck, F16).tobytes()
        for (r1, r2, c1, c2, v) in self.leaves:
            out += np.asarray([r1, r2, c1, c2], np.uint8).tobytes()
            out += np.float16(v).tobytes()
        return bytes(out)

    @classmethod
    def unwire(cls, j, k, buf):
        import struct
        off = 0

        def take(n):
            nonlocal off
            b = bytes(buf[off:off + n])
            off += n
            return b
        a = cls.__new__(cls)
        a.j, a.k = int(j), int(k)
        L = take(1)[0]
        a.xmin_j, a.xmax_j, a.xmin_k, a.xmax_k = struct.unpack("<ffff", take(16))
        ncj, nck = take(1)[0], take(1)[0]
        a.cj = np.frombuffer(take(2 * ncj), F16).astype(float).copy()
        a.ck = np.frombuffer(take(2 * nck), F16).astype(float).copy()
        leaves = []
        for _ in range(L):
            r1, r2, c1, c2 = take(4)
            v = struct.unpack("<e", take(2))[0]
            leaves.append((r1, r2, c1, c2, float(v)))
        a.leaves = leaves
        a.lmap = None
        a.dim = 1
        return a, off


def make_swap(arm, j, k, card, X):
    if arm == "2A":
        return P1Swap2A(j, k, card, X)
    if arm == "2B":
        return P1Swap2B(j, k, card, X)
    raise ValueError(arm)
