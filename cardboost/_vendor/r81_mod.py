"""r81_mod.py -- R81 Part 5 (H6): contextual 3-way modulation ('mod<g>').

The pair surface S(x_j, x_k) (p1 machinery) is modulated by a 1-D function of a
third, context feature g:    contrib(x) = S(x_j, x_k) * m(x_g).
Heredity: the modulation cannot exist without the pair shape (S is fitted
first; admission additionally requires an admitted pair atom for the same
(j, k) -- strong heredity, patched in run_fold77).  Context price ln(p-2) is
charged on top of the barrier for mod candidates (run_fold77).
m is fitted leak-free on the refit rows: per context-bin
    m_b = sum_i S_i r_i / (sum_i S_i^2 + lam),  r = working residual,
Potts-fused over bins (potts_dp), then re-centred to S^2-weighted mean 1 so
the surface scale stays canonical.  Deploy: pair surface + 1-D context levels.
Name-blind (Rule 12.B-3).
"""
from __future__ import annotations

import numpy as np

from r71_universal import onehot, potts_dp
from r76_stage31 import cuts_of
from r77_mech import P1Atom, bin_idx


class ModulationAtom(P1Atom):
    kind = "mod"

    def __init__(self, j, k, cuts_j, cuts_k, xs_j, xs_k, g):
        super().__init__(j, k, cuts_j, cuts_k, xs_j, xs_k)
        self.g = int(g)
        self.gcuts = None
        self.M = None          # (nb,) or (nb, K)

    def _m(self, X, like):
        out = np.ones_like(like, dtype=float)
        if self.M is None or self.gcuts is None:
            return out
        xg = X[:, self.g]
        a = bin_idx(xg, self.gcuts)
        ok = a >= 0
        if self.M.ndim == 1:
            out[ok] = self.M[a[ok]]
        else:
            out[ok] = self.M[a[ok]]
        return out

    def contrib(self, X):
        base = super().contrib(X)
        return base * self._m(X, base)

    @staticmethod
    def _working_resid(y, o, kind, n_classes):
        if kind == "regression":
            return (y - o).astype(float)
        if kind == "binary":
            return y - 1.0 / (1.0 + np.exp(-o))
        P = np.exp(o - o.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        Y1 = np.zeros_like(P)
        Y1[np.arange(len(y)), y.astype(int)] = 1.0
        return Y1 - P

    def refit(self, X, y, off, kind, kappa_mult=1.0, w=None, n_classes=0):
        cur = self.contrib(X)
        o = off - cur
        super().refit(X, y, off, kind, kappa_mult=kappa_mult, w=w,
                      n_classes=n_classes)          # surface S on residual
        r = self._working_resid(y, o, kind, n_classes)
        S = super().contrib(X)
        xg = X[:, self.g]
        fin = xg[np.isfinite(xg)]
        self.gcuts = (cuts_of(fin, 8) if len(fin) >= 8 else np.array([], float))
        a = bin_idx(xg, self.gcuts)
        ok = a >= 0
        nb = len(self.gcuts) + 2

        def fit_m(Sv, rv):
            num = np.zeros(nb)
            den = np.zeros(nb)
            for b in range(nb):
                m = ok & (a == b)
                if m.any():
                    num[b] = float(np.sum(Sv[m] * rv[m]))
                    den[b] = float(np.sum(Sv[m] ** 2))
            lam = 1e-6 * float(den.sum() / max(nb, 1) + 1e-12)
            mval = num / (den + lam)
            # Potts fusion over context bins
            if nb >= 2:
                lab = potts_dp(mval.copy(), np.where(den > 0, den, 1e-9),
                               0.5 * np.log(max(float(den.sum()), 1.0))
                               * kappa_mult)
                for k in np.unique(lab):
                    sel = lab == k
                    wgt = float(den[sel].sum())
                    mval[sel] = float(num[sel].sum() / (wgt + lam)) if wgt > 0 else 1.0
            mval[den <= 1e-12] = 1.0
            # re-centre to S^2-weighted mean 1 over the fit rows
            wts = Sv[ok] ** 2
            scale = float(np.sum(wts * mval[a[ok]]) / max(float(np.sum(wts)), 1e-12))
            if abs(scale) > 1e-9:
                mval = mval / scale
            return mval

        if S.ndim == 2:
            K = S.shape[1]
            self.M = np.zeros((nb, K))
            for c in range(K):
                self.M[:, c] = fit_m(S[:, c], r[:, c])
        else:
            self.M = fit_m(S, r)
        return o + self.contrib(X)
