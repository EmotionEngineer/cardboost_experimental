"""ANOVA-style component importance over a reference sample.

Each additive component (main effect j, pair atom, q_miss) is scored by the
variance of its contribution over X (an exact functional-ANOVA term when the
sample is the empirical design distribution).  Shares sum to <= 1 with the
remainder attributed to b0/IsoLink.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def anova_importance(model, X: Optional[np.ndarray] = None, top: int = 20):
    """Return rows (component, var, share) sorted by variance desc."""
    card = getattr(model, "dec_card_", None) or getattr(model, "card_", None)
    if X is None:
        raise ValueError("pass the training design X for ANOVA importance")
    X = np.asarray(X, float)
    import r76_container31 as C
    import r71_universal as _L

    rows = []
    total = 0.0
    from ..core.binning import stable_desc_order
    for j, cuts in enumerate(card.cuts):
        V = np.asarray(card.V[j], float)
        b = _L.bin_idx(X[:, j], np.asarray(cuts, float)) if len(cuts) \
            else np.zeros(len(X), int)
        contrib = V[b] if V.ndim == 1 else V[b].sum(axis=1)
        v = float(np.var(contrib))
        rows.append((f"main:x{j}", v))
        total += v
    for i, at in enumerate(card.atoms):
        contrib = np.asarray(at.contrib(X), float)
        if contrib.ndim > 1:
            contrib = contrib.sum(axis=1)
        v = float(np.var(contrib))
        rows.append((f"atom{i}:{type(at).__name__}({getattr(at,'j','?')},{getattr(at,'k','?')})", v))
        total += v
    order = stable_desc_order(np.asarray([r[1] for r in rows], float))
    rows = [rows[i] for i in order]
    out = [(name, v, (v / total if total > 0 else 0.0)) for name, v in rows[:top]]
    return out
