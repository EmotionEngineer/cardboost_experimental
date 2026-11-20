"""Scorecard table export (DataFrame) with optional PDO points (R68)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def scorecard_df(model, X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 pdo: float = 20.0, base_points: int = 600,
                 base_odds: float = 50.0, lo: float = 0.0):
    """Mains scorecard: Feature, bin, interval, [count, mass, event_rate], level, points.

    Points scaling (R68, only for 1D mains - the documented ÷sigma2_oof 1D
    currency): points = base_points - factor * level, factor = pdo / ln(2).
    """
    card = getattr(model, "dec_card_", None) or getattr(model, "card_", None)
    names = getattr(model, "feature_names_", None)
    factor = pdo / np.log(2.0)
    from ..core.binning import round_scores
    rows = []
    for j, cuts in enumerate(card.cuts):
        V = np.asarray(card.V[j], float)
        nm = names[j] if names else f"x{j}"
        m = len(cuts) + 2
        cnt = np.zeros(m)
        ssum = np.zeros(m)
        if X is not None and y is not None:
            import r76_container31 as C
            import r71_universal as _L
            b = _L.bin_idx(np.asarray(X)[:, j], np.asarray(cuts, float)) if len(cuts) \
                else np.zeros(len(X), int)
            for bi in range(m):
                sel = b == bi
                cnt[bi] = int(sel.sum())
                ssum[bi] = float(np.asarray(y, float)[sel].sum())
        edges = [None] + [float(c) for c in np.asarray(cuts, float).reshape(-1)] + [None]
        for bi in range(m):
            lvl = float(V[bi]) if V.ndim == 1 else float(np.sum(V[bi]))
            interval = ("NaN" if bi == m - 1 else
                        f"(.., {edges[bi + 1]:.6g}]" if bi == 0 else
                        f"({edges[bi]:.6g}, ..)" if bi == m - 2 else
                        f"({edges[bi]:.6g}, {edges[bi + 1]:.6g}]")
            row = dict(feature=nm, bin=bi, interval=interval,
                       level=float(round_scores(np.asarray([lvl]))[0]),
                       points=float(round_scores(
                           np.asarray([base_points - factor * lvl]))[0]))
            if X is not None and y is not None:
                row["count"] = int(cnt[bi])
                row["mass"] = float(cnt[bi] / max(len(X), 1))
                row["event_rate"] = float(ssum[bi] / cnt[bi]) if cnt[bi] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
