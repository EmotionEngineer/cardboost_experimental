"""Exact local decomposition: score = b0 + sum contributions (+ IsoLink)."""
from __future__ import annotations

import numpy as np


def explain_local(model, x):
    """Exact additive waterfall for one row x.

    Returns dict: base, contributions [(component, value)], raw, prediction
    (after IsoLink when present).  The sum is bit-exact vs model.predict.
    """
    card = getattr(model, "dec_card_", None) or getattr(model, "card_", None)
    X = np.atleast_2d(np.asarray(x, float))
    import r76_container31 as C
    import r71_universal as _L

    names = getattr(model, "feature_names_", None)
    contribs = []
    base = float(np.asarray(card.b0, float).reshape(-1)[0])
    for j, cuts in enumerate(card.cuts):
        V = np.asarray(card.V[j], float)
        b = _L.bin_idx(X[:, j], np.asarray(cuts, float)) if len(cuts) \
            else np.zeros(len(X), int)
        v = V[b[0]] if V.ndim == 1 else float(np.sum(V[b[0]]))
        nm = names[j] if names else f"x{j}"
        contribs.append((f"main:{nm}", float(v)))
    for i, at in enumerate(card.atoms):
        v = float(np.asarray(at.contrib(X), float).reshape(len(X), -1)[0].sum())
        contribs.append((f"atom{i}:{type(at).__name__}({getattr(at,'j','?')},{getattr(at,'k','?')})", v))
    raw = base + sum(v for _, v in contribs)
    pred = float(np.asarray(C.model_score(card, X), float).reshape(-1)[0])
    return dict(base=base, contributions=contribs, raw=raw, prediction=pred)
