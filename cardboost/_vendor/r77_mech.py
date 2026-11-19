"""r77_mech.py — R77 atom machinery (R77 blueprint §1-2 fixes).

Fixes over r76_mech.py:
  * Out-of-Support Clamping Gate replaces the boundary-bin gate: a row is zeroed
    only when a parent value leaves the TRAIN range [xmin, xmax] (eps=1e-7), NOT
    when it lands in boundary quantile bins 0 / m-1 (which are interior train
    mass: bottom/top 3.125% of the sample).  cal_spat f2 keeps its tails; f4
    (98.5% rows beyond train Lat max) is still clamped to 0 -> rescue preserved.
  * g4 rays use continuous ECDF ranks r = searchsorted(sorted_train, x)/n
    (blueprint cause B: r76 used 8-level coarse bin ranks) and a 32-quantile
    grid on u.  4-ray star per pair: u1=rj+rk, u2=rj-rk, u3=2rj+rk, u4=rj+2rk.
  * p1: cell support shrinkage n/(n+25) + RSR two-way H-weighted demeaning
    (marginals stay in the mains) + out-of-support clamp (blueprint step 2.1).
  * m2 unchanged (bilinear on mains' levels) + optional pos-tail mode
    gamma * zj+ * zk+ for aps (step 3, wired but used later).
"""
import numpy as np

import r76_mech as M
from r76_mech import (bin_idx, cuts_of, onehot, product_potts, potts_dp,
                      sig, softmax, wsegment_levels, wsolve_bin_logistic,
                      wsolve_bin_multiclass, wsolve_bin_regression)

EPS = 1e-7

# ---- R77 step-2-bis Handle A (user-confirmed) ------------------------------
# 1) g4 RSR-marginal centering: two alternating H-weighted passes over the 8
#    quantile bins of each parent -> atom mean is 0 on ANY parent band
#    (kills the +0.02..+0.21 hole/band shift that exploded f1).
# 2) robust out-of-support gate [q_lo, q_hi] of TRAIN instead of [min, max]
#    (zeroes f4's 1.5% near-max edge noise; keeps interior folds' tails).
# Both are toggled together as Handle A of the W1 ablation.
# R79: default OFF permanently (R77 isolate confound, see r78_report.md S0.1).
# Enable only inside explicit Handle-A experiments.
HANDLE_A = False
QGATE = None
MARG_BINS = 8
P1_BINS = 8             # p1 grid size (w1_p1_12only sets 12)
P1_SHRINK_POST = False  # user 2.2: shrink AFTER Potts+RSR
R84_FIT_RANK = False    # R84 canary 1: fit g4 atoms against the quantized 96-knot
P1_SWAP = None          # R84 canary 2: '2A' | '2B' -> slot p1 builds the swap atom (r84_p1swap)
                        # rank grid (make_rank_grid) so fit == deploy at score level.


def _support(x, xmin, xmax):
    """in-support mask (NaN-safe): xmin-eps <= x <= xmax+eps."""
    return np.isfinite(x) & (x >= xmin - EPS) & (x <= xmax + EPS)


def _wmean(v, w):
    sw = float(w.sum())
    return float((v * w).sum() / sw) if sw > 0 else 0.0


def capped_dp(C, H, kap, exs):
    """R111 (frozen R87 Finalist 3): potts_dp с ПОТОЛКОМ сегментов S <= S_quad.
    exact-девианса берётся только если её решение не длиннее квадратичного при
    том же kap; иначе минимальное kappa-умножение, ужимающее её до S_quad
    (бюджет квадратики -- встроенный регуляризатор, L12; R107 без потолка -> x2.7).
    Без exact/cap -- бит-в-бит исходный вызов."""
    if not (exs.get("exact") and M.EXACT_DEV.get("cap")):
        return potts_dp(C, H, kap, **exs)
    lab_q = potts_dp(C, H, kap)
    S_q = int(len(np.unique(lab_q)))
    lab_e = potts_dp(C, H, kap, **exs)
    if len(np.unique(lab_e)) <= S_q:
        return lab_e
    lo_g, hi_g, lab_hi = 1.0, 2.0, None
    for _ in range(12):
        cand = potts_dp(C, H, kap * hi_g, **exs)
        if len(np.unique(cand)) <= S_q:
            lab_hi = cand
            break
        lo_g, hi_g = hi_g, hi_g * 2.0
    if lab_hi is None:
        return lab_q
    for _ in range(6):
        mid = 0.5 * (lo_g + hi_g)
        cand = potts_dp(C, H, kap * mid, **exs)
        if len(np.unique(cand)) <= S_q:
            hi_g, lab_hi = mid, cand
        else:
            lo_g = mid
    return lab_hi


class ProjRayAtom:
    """One projection ray u = a*rj + b*rk on continuous ECDF ranks.

    sign convention kept for logging: (a, b) in {(1,1), (1,-1), (2,1), (1,2)}.
    Out-of-support rows (any parent outside its train range) contribute 0.
    """

    kind = "g4"

    def __init__(self, j, k, a, b, xs_j, xs_k, nbins=32):
        self.j, self.k, self.a, self.b = j, k, float(a), float(b)
        self.xs_j = np.asarray(xs_j, float)   # sorted train values (ECDF support)
        self.xs_k = np.asarray(xs_k, float)
        self.xmin_j, self.xmax_j = float(self.xs_j[0]), float(self.xs_j[-1])
        self.xmin_k, self.xmax_k = float(self.xs_k[0]), float(self.xs_k[-1])
        if HANDLE_A and QGATE is not None:
            qlo, qhi = QGATE
            self.xmin_j = float(np.quantile(self.xs_j, qlo))
            self.xmax_j = float(np.quantile(self.xs_j, qhi))
            self.xmin_k = float(np.quantile(self.xs_k, qlo))
            self.xmax_k = float(np.quantile(self.xs_k, qhi))
        self.m = nbins
        self.ucuts = None
        self.V = None
        self.Vc = None
        self.rank_interp = None
        if R84_FIT_RANK:
            from r76_container31 import make_rank_grid
            gj, rgj = make_rank_grid(self.xs_j)
            gk, rgk = make_rank_grid(self.xs_k)
            self.rank_interp = (gj, rgj, gk, rgk)
        self.marg = HANDLE_A
        if self.marg:
            self.qcuts_j = cuts_of(self.xs_j, MARG_BINS)
            self.qcuts_k = cuts_of(self.xs_k, MARG_BINS)
            self.mj = None
            self.mk = None

    def _r(self, X):
        xj = X[:, self.j]
        xk = X[:, self.k]
        if getattr(self, "rank_interp", None) is not None:
            gj, rgj, gk, rgk = self.rank_interp
            rj = np.interp(np.where(np.isfinite(xj), xj, self.xmin_j), gj, rgj)
            rk = np.interp(np.where(np.isfinite(xk), xk, self.xmin_k), gk, rgk)
        else:
            rj = np.searchsorted(self.xs_j, np.where(np.isfinite(xj), xj, self.xmin_j), "right") / len(self.xs_j)
            rk = np.searchsorted(self.xs_k, np.where(np.isfinite(xk), xk, self.xmin_k), "right") / len(self.xs_k)
        ins = _support(xj, self.xmin_j, self.xmax_j) & _support(xk, self.xmin_k, self.xmax_k)
        return rj, rk, ins

    def _u(self, X):
        rj, rk, ins = self._r(X)
        u = self.a * rj + self.b * rk
        return u, ins

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1

    def contrib(self, X):
        u, ins = self._u(X)
        if self.V is None and self.Vc is None:
            return np.zeros((len(X), self.dim)) if self.dim > 1 else np.zeros(len(X))
        b = bin_idx(u, self.ucuts) if self.ucuts is not None and len(self.ucuts) else np.zeros(len(X), int)
        b = np.where(ins, b, -1)
        m = b >= 0
        if self.Vc is not None:
            out = np.zeros((len(X), self.Vc.shape[1]))
            out[m] = self.Vc[b[m]]
            if self.marg and self.mj is not None:
                bj = bin_idx(np.where(np.isfinite(X[:, self.j]), X[:, self.j], self.xmin_j), self.qcuts_j)
                bk = bin_idx(np.where(np.isfinite(X[:, self.k]), X[:, self.k], self.xmin_k), self.qcuts_k)
                out[m] -= self.mj[bj[m]] + self.mk[bk[m]]
            return out
        out = np.zeros(len(X))
        out[m] = self.V[b[m]]
        if self.marg and self.mj is not None:
            bj = bin_idx(np.where(np.isfinite(X[:, self.j]), X[:, self.j], self.xmin_j), self.qcuts_j)
            bk = bin_idx(np.where(np.isfinite(X[:, self.k]), X[:, self.k], self.xmin_k), self.qcuts_k)
            out[m] -= self.mj[bj[m]] + self.mk[bk[m]]
        return out

    def _fit_marginals(self, X, w, sel, b):
        """two alternating H-weighted passes of 1D marginal subtraction over the
        MARG_BINS quantile bins of each parent (Handle A)."""
        xj = np.where(np.isfinite(X[:, self.j]), X[:, self.j], self.xmin_j)
        xk = np.where(np.isfinite(X[:, self.k]), X[:, self.k], self.xmin_k)
        bj = bin_idx(xj, self.qcuts_j)
        bk = bin_idx(xk, self.qcuts_k)
        nj, nk = len(self.qcuts_j) + 2, len(self.qcuts_k) + 2
        if self.Vc is not None:
            raw = np.zeros((len(X), self.Vc.shape[1]))
            raw[sel] = self.Vc[b[sel]]
            self.mj = np.zeros((nj, raw.shape[1]))
            self.mk = np.zeros((nk, raw.shape[1]))
            for _ in range(2):
                for c in range(raw.shape[1]):
                    for q in range(nj):
                        s2 = sel & (bj == q)
                        if s2.any():
                            self.mj[q, c] += _wmean((raw[:, c] - self.mj[:, c][bj] - self.mk[:, c][bk])[s2], w[s2])
                    for q in range(nk):
                        s2 = sel & (bk == q)
                        if s2.any():
                            self.mk[q, c] += _wmean((raw[:, c] - self.mj[:, c][bj] - self.mk[:, c][bk])[s2], w[s2])
            return
        raw = np.zeros(len(X))
        raw[sel] = self.V[b[sel]]
        self.mj = np.zeros(nj)
        self.mk = np.zeros(nk)
        for _ in range(2):
            for q in range(nj):
                s2 = sel & (bj == q)
                if s2.any():
                    self.mj[q] += _wmean((raw - self.mj[bj] - self.mk[bk])[s2], w[s2])
            for q in range(nk):
                s2 = sel & (bk == q)
                if s2.any():
                    self.mk[q] += _wmean((raw - self.mj[bj] - self.mk[bk])[s2], w[s2])

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        n = len(y)
        if w is None:
            w = np.ones(n)
        u, ins = self._u(X)
        cur = self.contrib(X)
        o = off - cur
        if self.ucuts is None:
            self.ucuts = cuts_of(u[ins], self.m) if ins.any() else np.array([])
        b = bin_idx(u, self.ucuts) if len(self.ucuts) else np.zeros(n, int)
        b = np.where(ins, b, -1)
        sel = b >= 0
        if sel.sum() < 10:
            return o + self.contrib(X)
        nb = (len(self.ucuts) + 2) if len(self.ucuts) else 2
        if kind == "multiclass":
            K = n_classes
            if self.Vc is None:
                self.Vc = np.zeros((nb, K))
            C2, H = wsolve_bin_multiclass(o[sel], onehot(y, K)[sel], nb, b[sel].astype(int), K, w[sel])
            emp = H <= 1e-12
            kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
            lab = potts_dp(np.where(emp[:, None], 0.0, C2), np.where(emp, 1e-9, H), kap)
            self.Vc = wsegment_levels(lab, None, H, "multiclass", C2=np.where(emp[:, None], 0.0, C2))
            if self.marg:
                self._fit_marginals(X, w, sel, b)
            return o + self.contrib(X)
        if kind == "binary":
            z, H = wsolve_bin_logistic(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        else:
            z, H = wsolve_bin_regression(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        emp = H <= 1e-12
        zc = np.where(emp, 0.0, z)
        kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
        # R107: exact binomial deviance for binary 1D rays (flag-shared with mains)
        exs = dict(exact=False)
        if kind == "binary" and M.EXACT_DEV["on"]:
            exs = dict(exact=True,
                       Gs=np.bincount(b[sel].astype(int), weights=w[sel] * y[sel], minlength=nb),
                       Ns=np.bincount(b[sel].astype(int), weights=w[sel], minlength=nb))
        lab = capped_dp(zc, np.where(emp, 1e-9, H), kap, exs)
        self.V = wsegment_levels(lab, zc, H, "binary" if kind == "binary" else "regression")
        if self.marg:
            self._fit_marginals(X, w, sel, b)
        return o + self.contrib(X)

    def n_params(self):
        V = self.Vc if self.Vc is not None else self.V
        if V is None:
            return 0
        if V.ndim == 2:
            return int(np.sum(np.any(np.diff(V, axis=0) != 0, axis=1))) + 1
        return int(np.sum(np.diff(V) != 0)) + 1

    def snapshot(self):
        V = self.Vc if self.Vc is not None else self.V
        return V.ravel().copy() if V is not None else np.zeros(1)


class P1Atom:
    """p1: product-Potts rectangle on the pair grid, RSR-purified + support-shrunk.

    After the 2-D TV-Potts merge: cells shrink by n/(n+25) (support filter,
    blueprint 2.1.1) and the table is two-way H-weighted demeaned (rows then
    columns, 2 passes) so row/column marginals stay in the mains (2.1.2).
    Out-of-support rows contribute 0 (2.1.3).
    """

    kind = "p1"

    def __init__(self, j, k, cuts_j, cuts_k, xs_j, xs_k):
        self.j, self.k = j, k
        self.cj = np.asarray(cuts_j, float)
        self.ck = np.asarray(cuts_k, float)
        self.xs_j = np.asarray(xs_j, float)
        self.xs_k = np.asarray(xs_k, float)
        self.xmin_j, self.xmax_j = float(self.xs_j[0]), float(self.xs_j[-1])
        self.xmin_k, self.xmax_k = float(self.xs_k[0]), float(self.xs_k[-1])
        self.A = len(self.cj) + 2
        self.Bn = len(self.ck) + 2
        self.V = np.zeros((self.A, self.Bn))
        self.Vc = None

    def _cells(self, X):
        xj = X[:, self.j]
        xk = X[:, self.k]
        a = bin_idx(xj, self.cj)
        b = bin_idx(xk, self.ck)
        ins = _support(xj, self.xmin_j, self.xmax_j) & _support(xk, self.xmin_k, self.xmax_k)
        return a, b, ins

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1
        if self.dim > 1 and self.Vc is None:
            self.Vc = np.zeros((self.A, self.Bn, self.dim))

    def contrib(self, X):
        a, b, ins = self._cells(X)
        if self.Vc is not None:
            out = np.zeros((len(X), self.Vc.shape[2]))
            m = ins
            out[m] = self.Vc[a[m], b[m]]
            return out
        if self.dim > 1:
            return np.zeros((len(X), self.dim))
        out = np.zeros(len(X))
        out[ins] = self.V[a[ins], b[ins]]
        return out

    @staticmethod
    def _rsr_shrink(T, Hn, shrink_n=25.0, post=False):
        """post=False (W1 legacy): shrink N/(N+shrink_n) THEN two-way H-weighted
        demeaning (2 passes).  post=True (user §2.2 fix): demean FIRST and shrink
        AFTER RSR, so hole cells (N~0) collapse to exactly 0 instead of carrying a
        Potts-interpolated offset invisible to the demeaning."""
        if not post:
            T = T * (Hn / (Hn + shrink_n))
        W = np.maximum(Hn, 1e-9)
        for _ in range(2):
            rm = (T * W).sum(axis=1) / W.sum(axis=1)
            T = T - rm[:, None]
            cm = (T * W).sum(axis=0) / W.sum(axis=0)
            T = T - cm[None, :]
        if post:
            T = T * (Hn / (Hn + shrink_n))
        return T

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        a, b, ins = self._cells(X)
        n = len(y)
        if w is None:
            w = np.ones(n)
        flat = a * self.Bn + b
        nb = self.A * self.Bn
        cur = self.contrib(X)
        o = off - cur
        if kind == "multiclass":
            K = n_classes
            if self.Vc is None:
                self.Vc = np.zeros((self.A, self.Bn, K))
            C2, H = wsolve_bin_multiclass(o[ins], onehot(y, K)[ins], nb, flat[ins], K, w[ins])
            grid = C2.reshape(self.A, self.Bn, K)
            Hg = H.reshape(self.A, self.Bn)
            kap = 0.5 * np.log(max(float(Hg.sum()), 1.0)) * kappa_mult
            Vc = np.stack([product_potts(grid[:, :, c], Hg, kap) for c in range(K)], axis=2)
            for c in range(K):
                Vc[:, :, c] = self._rsr_shrink(Vc[:, :, c], Hg, post=P1_SHRINK_POST)
            self.Vc = Vc
            return o + self.contrib(X)
        if kind == "binary":
            z, H = wsolve_bin_logistic(o[ins], y[ins], nb, flat[ins], w[ins])
        else:
            z, H = wsolve_bin_regression(o[ins], y[ins], nb, flat[ins], w[ins])
        zg = z.reshape(self.A, self.Bn)
        Hg = H.reshape(self.A, self.Bn)
        kap = 0.5 * np.log(max(float(Hg.sum()), 1.0)) * kappa_mult
        V = product_potts(zg, Hg, kap)
        self.V = self._rsr_shrink(V, Hg, post=P1_SHRINK_POST)
        return o + self.contrib(X)

    def n_params(self):
        V = self.Vc if self.Vc is not None else self.V
        if V.ndim == 3:
            live = int(np.sum(np.any(np.diff(V.reshape(-1, V.shape[2]), axis=0) != 0, axis=1)))
        else:
            live = int(np.sum(np.diff(V.ravel()) != 0))
        return live + 1

    def snapshot(self):
        return (self.Vc if self.Vc is not None else self.V).ravel().copy()


class M2Atom:
    """m2: gamma * zj[bj] * zk[bk]  (pos=True: gamma * zj+ * zk+, aps tail synergy)."""

    kind = "m2"

    def __init__(self, j, k, zj, zk, pos=False):
        self.j, self.k = j, k
        self.zj = np.asarray(zj, float)
        self.zk = np.asarray(zk, float)
        self.pos = pos
        self.gamma = 0.0

    def set_cuts(self, cj, ck):
        self.cj, self.ck = cj, ck

    def ensure(self, kind, n_classes=0):
        self.dim = n_classes if kind == "multiclass" else 1
        if self.dim > 1 and np.size(self.gamma) == 1:
            self.gamma = np.zeros(self.dim)

    def contrib(self, X):
        bj = bin_idx(X[:, self.j], self.cj)
        bk = bin_idx(X[:, self.k], self.ck)
        zj = np.maximum(self.zj[bj], 0.0) if self.pos else self.zj[bj]
        zk = np.maximum(self.zk[bk], 0.0) if self.pos else self.zk[bk]
        phi = zj * zk
        if self.dim > 1:
            return phi[:, None] * np.asarray(self.gamma)[None, :]
        return self.gamma * phi

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        n = len(y)
        if w is None:
            w = np.ones(n)
        bj = bin_idx(X[:, self.j], self.cj)
        bk = bin_idx(X[:, self.k], self.ck)
        zj = np.maximum(self.zj[bj], 0.0) if self.pos else self.zj[bj]
        zk = np.maximum(self.zk[bk], 0.0) if self.pos else self.zk[bk]
        phi = zj * zk
        cur = self.contrib(X)
        o = off - cur
        if kind == "regression":
            num = float((w * phi * (y - o)).sum())
            den = float((w * phi * phi).sum()) + 1e-9
            self.gamma = np.clip(num / den, -50.0, 50.0)
            return o + self.gamma * phi
        if kind == "binary":
            g = 0.0
            for _ in range(25):
                p = sig(o + g * phi)
                num = float((w * phi * (y - p)).sum())
                den = float((w * phi * phi * p * (1 - p)).sum()) + 1e-9
                step = np.clip(num / den, -2.0, 2.0)
                g += 0.5 * step
                if abs(step) < 1e-7:
                    break
            self.gamma = float(np.clip(g, -50.0, 50.0))
            return o + self.gamma * phi
        K = n_classes
        gam = np.zeros(K)
        Y1 = onehot(y, K)
        for _ in range(25):
            P = softmax(o + phi[:, None] * gam[None, :])
            num = (w[:, None] * phi[:, None] * (Y1 - P)).sum(axis=0)
            den = (w[:, None] * phi[:, None] ** 2 * P * (1 - P)).sum(axis=0) + 1e-9
            step = np.clip(num / den, -2.0, 2.0)
            gam += 0.5 * step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.gamma = np.clip(gam, -50.0, 50.0)
        return o + phi[:, None] * self.gamma[None, :]

    def n_params(self):
        return int(np.size(self.gamma))

    def snapshot(self):
        return np.atleast_1d(np.asarray(self.gamma, float)).copy()


class MSumAtom:
    """msum (R85 Canary 3 / R105 value-space clamp): Main-Sum Ray -- 1D table
    on u = zj[bj] + zk[bk].

    zj/zk = parent mains shape at PAIR bins (f16-frozen at construction, R83
    IsoLink pattern) -- 0 B ранговой решётки, никаких родительских таблиц в
    payload.  u-cuts snapped to midpoints of distinct u values: ни одно значение
    u не равно резу -- emit-квантование f32 не двигает ни одной строки.

    R105 (frozen vote 2026-11-16): box-гейт [xmin,xmax]^2 ЗАМЕНЁН value-space
    clamp -- аргумент зажимается в [u_lo, u_hi] (диапазон u на train-строках),
    крайний уровень 1D-таблицы продолжается наружу; конечные строки вне box
    НЕ обнуляются (Инвариант III, Факт 3: ступенька -A_bar у msum токсична,
    аргумент и так ограничен полками мейнов).  NaN-строки вносят 0.
    """
    kind = "msum"

    def __init__(self, j, k, zj, zk, cj, ck, u_lo, u_hi, nbins=8):
        self.j, self.k = j, k
        self.zj = np.asarray(np.float16(np.asarray(zj, float)), float)
        self.zk = np.asarray(np.float16(np.asarray(zk, float)), float)
        self.cj = np.asarray(np.float32(np.asarray(cj, float)), float)
        self.ck = np.asarray(np.float32(np.asarray(ck, float)), float)
        self.u_lo = float(np.float32(u_lo))
        self.u_hi = float(np.float32(u_hi))
        self.nbins = int(nbins)
        self.ucuts = None
        self.V = None
        self.Vc = None

    def _u(self, X):
        xj = X[:, self.j]
        xk = X[:, self.k]
        f_j = float(self.cj[0]) if len(self.cj) else 0.0
        f_k = float(self.ck[0]) if len(self.ck) else 0.0
        bj = bin_idx(np.where(np.isfinite(xj), xj, f_j), self.cj)
        bk = bin_idx(np.where(np.isfinite(xk), xk, f_k), self.ck)
        u = self.zj[bj] + self.zk[bk]
        # R105: value-space clamp вместо box-обнуления (крайний уровень
        # 1D-таблицы продолжается за пределами train-диапазона u).
        u = np.clip(u, self.u_lo, self.u_hi)
        fin = np.isfinite(xj) & np.isfinite(xk)
        return u, fin

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
        b = bin_idx(u, self.ucuts) if self.ucuts is not None and len(self.ucuts) else np.zeros(len(X), int)
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
        if kind == "multiclass":
            K = n_classes
            if self.Vc is None:
                self.Vc = np.zeros((nb, K))
            C2, H = wsolve_bin_multiclass(o[sel], onehot(y, K)[sel], nb, b[sel].astype(int), K, w[sel])
            emp = H <= 1e-12
            kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
            lab = potts_dp(np.where(emp[:, None], 0.0, C2), np.where(emp, 1e-9, H), kap)
            self.Vc = wsegment_levels(lab, None, H, "multiclass", C2=np.where(emp[:, None], 0.0, C2))
            return o + self.contrib(X)
        if kind == "binary":
            z, H = wsolve_bin_logistic(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        else:
            z, H = wsolve_bin_regression(o[sel], y[sel], nb, b[sel].astype(int), w[sel])
        emp = H <= 1e-12
        zc = np.where(emp, 0.0, z)
        kap = 0.5 * np.log(max(float(H[~emp].sum()), 1.0)) * kappa_mult
        # R107: exact binomial deviance for binary 1D msum tables (flag-shared)
        exs = dict(exact=False)
        if kind == "binary" and M.EXACT_DEV["on"]:
            exs = dict(exact=True,
                       Gs=np.bincount(b[sel].astype(int), weights=w[sel] * y[sel], minlength=nb),
                       Ns=np.bincount(b[sel].astype(int), weights=w[sel], minlength=nb))
        lab = capped_dp(zc, np.where(emp, 1e-9, H), kap, exs)
        self.V = wsegment_levels(lab, zc, H, "binary" if kind == "binary" else "regression")
        return o + self.contrib(X)

    def n_params(self):
        V = self.Vc if self.Vc is not None else self.V
        if V is None:
            return 0
        if V.ndim == 2:
            return int(np.sum(np.any(np.diff(V, axis=0) != 0, axis=1))) + 1
        return int(np.sum(np.diff(V) != 0)) + 1

    def snapshot(self):
        V = self.Vc if self.Vc is not None else self.V
        return V.ravel().copy() if V is not None else np.zeros(1)


# ----------------------------------------------------------------------------
# two-scale product-Potts (step-2 '12x12 two-scale p1')
# ----------------------------------------------------------------------------
class P1TwoScale:
    """Fine 12x12 RSR product-Potts surface + coarse 4x4 RSR surface fitted on
    the fine's residual (the two-scale reading of the step-2 blueprint line)."""

    def __init__(self, j, k, cj, ck, xs_j, xs_k, fine_bins=12, coarse_bins=4):
        # builds its own cuts at each scale from the empirical xs
        self.fine = P1Atom(j, k, cuts_of(xs_j, fine_bins), cuts_of(xs_k, fine_bins),
                           xs_j, xs_k)
        self.coarse = P1Atom(j, k, cuts_of(xs_j, coarse_bins), cuts_of(xs_k, coarse_bins),
                             xs_j, xs_k)
        self.j, self.k = int(j), int(k)

    def ensure(self, kind, n_classes):
        self.fine.ensure(kind, n_classes)
        self.coarse.ensure(kind, n_classes)

    def contrib(self, X):
        return self.fine.contrib(X) + self.coarse.contrib(X)

    def refit(self, X, y, eta, kind, kappa_mult=1.0, w=None, n_classes=0):
        o = eta - self.contrib(X)
        self.fine.refit(X, y, o, kind, kappa_mult=kappa_mult, w=w, n_classes=n_classes)
        self.coarse.refit(X, y, o + self.fine.contrib(X), kind,
                          kappa_mult=kappa_mult, w=w, n_classes=n_classes)
        return o + self.contrib(X)

    def n_params(self):
        return self.fine.n_params() + self.coarse.n_params()

    def snapshot(self):
        # Component contract: 1-D numeric state vector (fit_joint's fixed point)
        return np.concatenate([np.atleast_1d(np.asarray(self.fine.snapshot(), float)).ravel(),
                               np.atleast_1d(np.asarray(self.coarse.snapshot(), float)).ravel()])

    def to_dict(self):
        return dict(type="p1ts", j=self.j, k=self.k,
                    fine=self.fine.snapshot(), coarse=self.coarse.snapshot())


# ----------------------------------------------------------------------------
# atom factory (R77): 4-ray star + p1 + p1ts + m2 (+ m2_pos when asked)
# ----------------------------------------------------------------------------
# R81-4 (H4): full sign palette, two tiers of slope --
#   tier 1: (1, +1), (1, -1)   -> g4a, g4b   (W1 legacy)
#   tier 2: (2, +1), (1, +2), (2, -1), (1, -2) -> g4c, g4d (W1) + g4e, g4f (new)
RAYS = (("g4a", 1.0, 1.0), ("g4b", 1.0, -1.0), ("g4c", 2.0, 1.0),
        ("g4d", 1.0, 2.0), ("g4e", 2.0, -1.0), ("g4f", 1.0, -2.0))


def make_atom77(atype, j, k, card, X, pair_bins=None):
    if pair_bins is None:
        pair_bins = P1_BINS
    X = np.asarray(X, float)
    xs_j = np.sort(X[:, j][np.isfinite(X[:, j])])
    xs_k = np.sort(X[:, k][np.isfinite(X[:, k])])
    if atype in ("g4a", "g4b", "g4c", "g4d", "g4e", "g4f"):
        a, b = dict((t, (aa, bb)) for t, aa, bb in RAYS)[atype]
        return ProjRayAtom(j, k, a, b, xs_j, xs_k)
    if atype == "p1":
        if P1_SWAP:
            import r84_p1swap as _PS
            a = _PS.make_swap(P1_SWAP, j, k, card, X)
            a.pair_bins = pair_bins
            return a
        cj = cuts_of(np.asarray(X[:, j], float), pair_bins)
        ck = cuts_of(np.asarray(X[:, k], float), pair_bins)
        return P1Atom(j, k, cj, ck, xs_j, xs_k)
    if atype == "mcond":
        import r82_mcond as MC
        cj = cuts_of(np.asarray(X[:, j], float), pair_bins)
        return MC.MCondAtom(j, k, cj, xs_j)
    if atype.startswith("mod"):
        import r81_mod as RM
        cj = cuts_of(np.asarray(X[:, j], float), pair_bins)
        ck = cuts_of(np.asarray(X[:, k], float), pair_bins)
        return RM.ModulationAtom(j, k, cj, ck, xs_j, xs_k, int(atype[3:]))
    if atype in ("s1", "s2", "av"):
        import r81_lowrank as LR
        cj = cuts_of(np.asarray(X[:, j], float), pair_bins)
        ck = cuts_of(np.asarray(X[:, k], float), pair_bins)
        cls = {"s1": LR.Rank1Atom, "s2": LR.Rank2Atom, "av": LR.AnovaContrastAtom}[atype]
        return cls(j, k, cj, ck, xs_j, xs_k)
    if atype == "p1v":
        import r81_varcoeff as VC
        cj = cuts_of(np.asarray(X[:, j], float), pair_bins)
        ck = cuts_of(np.asarray(X[:, k], float), pair_bins)
        return VC.VaryCoeffAtom(j, k, cj, ck, xs_j, xs_k)
    if atype == "p1ts":
        return P1TwoScale(j, k, None, None, xs_j, xs_k)
    if atype in ("m2", "m2_pos"):
        zj = np.asarray(card.V[j], float)
        zk = np.asarray(card.V[k], float)
        if zj.ndim == 2:
            zj = np.linalg.norm(zj, axis=1)
        if zk.ndim == 2:
            zk = np.linalg.norm(zk, axis=1)
        zj = zj - zj.mean()
        zk = zk - zk.mean()
        a = M2Atom(j, k, zj, zk, pos=(atype == "m2_pos"))
        a.set_cuts(card.cuts[j], card.cuts[k])
        return a
    if atype == "msum":
        # R85 Canary 3: Main-Sum Ray -- zj/zk = мейны родителя на pair-бинах
        # (f16-заморозка на конструировании), карта 1D по u = zj + zk.
        Xj = np.asarray(X[:, j], float)
        Xk = np.asarray(X[:, k], float)
        cj = cuts_of(Xj, pair_bins)
        ck = cuts_of(Xk, pair_bins)
        bj_p = bin_idx(Xj, cj)
        bk_p = bin_idx(Xk, ck)
        bj_m = bin_idx(Xj, np.asarray(card.cuts[j], float))
        bk_m = bin_idx(Xk, np.asarray(card.cuts[k], float))
        Vj = np.asarray(card.V[j], float)
        Vk = np.asarray(card.V[k], float)
        if Vj.ndim == 2:
            Vj = np.linalg.norm(Vj, axis=1)
        if Vk.ndim == 2:
            Vk = np.linalg.norm(Vk, axis=1)
        zj = np.zeros(len(cj) + 2)
        zk = np.zeros(len(ck) + 2)
        for b in range(len(cj) + 2):
            m = (bj_p == b) & np.isfinite(Xj)
            zj[b] = float(Vj[bj_m[m]].mean()) if m.any() else 0.0
            m = (bk_p == b) & np.isfinite(Xk)
            zk[b] = float(Vk[bk_m[m]].mean()) if m.any() else 0.0
        # R105: диапазон аргумента на train-строках -- границы value-space clamp
        u_tr = (zj[bj_p] + zk[bk_p])[np.isfinite(Xj) & np.isfinite(Xk)]
        if u_tr.size == 0:
            u_tr = np.zeros(1)
        return MSumAtom(j, k, zj, zk, cj, ck, float(u_tr.min()), float(u_tr.max()))
    raise ValueError(atype)
