"""Cross-platform determinism helpers (`deterministic_ulp`).

Motivation (R87 cross-platform study): training on Linux x86-64 vs macOS ARM64
is bit-identical on 5 of 6 metric families; the knife-edge `cal_spat f4` fold
flips because a *decision* (pair ranking / DP argmax) compares float64 sums
that differ by 1 ULP across SIMD implementations (AVX2 vs NEON).

Mitigation implemented here: round decision scores to 12 significant decimals
and break exact ties lexicographically.  Applied by the engine wrapper at the
screening-score boundary (see ``engine.fit_fold``); the vendored DP accumulators
already sum in a fixed index order.

Golden anchors are defined with the canonical Linux x86-64 reference values in
``golden/canonical_r87.json``; with ``deterministic_ulp=True`` the anchors are
expected to hold on every platform (verified per-platform by running the golden
suite).
"""
from __future__ import annotations

import numpy as np

DIGITS = 12


def round_scores(S, digits: int = DIGITS):
    """Round decision scores to `digits` significant decimals (ties become exact)."""
    S = np.asarray(S, float)
    with np.errstate(invalid="ignore"):
        scale = np.where(S != 0, np.power(10.0, digits - 1 - np.floor(np.log10(np.abs(S + (S == 0))))), 1.0)
    return np.round(S * scale) / scale


def stable_desc_order(S, keys=None):
    """Descending order of scores S with deterministic tie-break on `keys`."""
    S = round_scores(S)
    if keys is None:
        keys = np.arange(len(S))
    keys = np.asarray(keys)
    order = np.lexsort((keys, -np.nan_to_num(S, nan=-np.inf)))
    return order


def dp_tie_break(cost_a, cost_b, prefer_larger: bool = True, eps: float = 1e-12):
    """Return True when candidate A wins over B under the deterministic tie rule.

    Costs within `eps` are considered equal; `prefer_larger` selects the
    coarser (larger-segment) candidate on ties, per the R87 spec.
    """
    if cost_a < cost_b - eps:
        return True
    if cost_b < cost_a - eps:
        return False
    return prefer_larger
