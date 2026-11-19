"""Self-contained HTML report (inline SVG step charts, zero external assets)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def _svg_step(cuts, levels, width=520, height=180, name=""):
    cuts = list(np.asarray(cuts, float).reshape(-1))
    lo = cuts[0] - 1.0 if cuts else -1.0
    hi = cuts[-1] + 1.0 if cuts else 1.0
    if not cuts:
        lo, hi = -1.0, 1.0
    edges = [lo] + cuts + [hi]
    vmin = min(levels) if len(levels) else 0.0
    vmax = max(levels) if len(levels) else 1.0
    pad = (vmax - vmin) * 0.1 + 1e-9
    vmin -= pad
    vmax += pad

    def sx(x):
        return 40 + (x - lo) / (hi - lo) * (width - 60)

    def sy(v):
        return height - 25 - (v - vmin) / (vmax - vmin) * (height - 40)

    pts = []
    for bi, lvl in enumerate(levels[:-1]):
        x0, x1 = edges[bi], edges[bi + 1]
        pts.append((sx(x0), sy(lvl)))
        pts.append((sx(x1), sy(lvl)))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background:#fbfbfd;border:1px solid #ddd;border-radius:6px">'
            f'<polyline points="{poly}" fill="none" stroke="#1f5fa8" stroke-width="2"/>'
            f'<text x="8" y="14" font-size="11" font-family="sans-serif">{name}</text></svg>')


def html_report(model, path: str, X: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None, title: str = "cardboost report") -> str:
    """Write the glass-box report to `path`; returns the path."""
    card = getattr(model, "dec_card_", None) or getattr(model, "card_", None)
    names = getattr(model, "feature_names_", None)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:-apple-system,Segoe UI,sans-serif;margin:2rem;"
        "color:#222}h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:1.6rem}"
        "table{border-collapse:collapse;font-size:.85rem}"
        "td,th{border:1px solid #ccc;padding:.25rem .6rem}"
        ".kpi{display:inline-block;margin:.4rem 1rem .4rem 0;padding:.6rem 1rem;"
        "background:#f2f5fa;border-radius:8px}.kpi b{display:block;font-size:1.25rem}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
    ]
    v31 = getattr(model, "v31_", None)
    kpis = []
    if v31 is not None:
        kpis.append(f"<div class='kpi'><b>{v31}</b>container bytes</div>")
    kpis.append(f"<div class='kpi'><b>{len(card.cuts)}</b>main effects</div>")
    kpis.append(f"<div class='kpi'><b>{len(card.atoms)}</b>pair atoms</div>")
    if getattr(model, "max_diff_", None) is not None:
        kpis.append(f"<div class='kpi'><b>{model.max_diff_}</b>decode max diff</div>")
    parts.extend(kpis)

    parts.append("<h2>Main effects</h2>")
    for j, cuts in enumerate(card.cuts):
        V = np.asarray(card.V[j], float)
        lvl = list(V if V.ndim == 1 else V.sum(axis=1))
        nm = names[j] if names else f"x{j}"
        parts.append(_svg_step(np.asarray(cuts, float).reshape(-1).tolist(),
                               [float(v) for v in lvl], name=nm))

    parts.append("<h2>Pair atoms</h2><table><tr><th>#</th><th>type</th><th>pair</th></tr>")
    for i, at in enumerate(card.atoms):
        parts.append(f"<tr><td>{i}</td><td>{type(at).__name__}</td>"
                     f"<td>({getattr(at,'j','?')},{getattr(at,'k','?')})</td></tr>")
    parts.append("</table>")

    if X is not None and y is not None:
        from ..explain.anova import anova_importance
        parts.append("<h2>ANOVA importance (top 12)</h2>"
                     "<table><tr><th>component</th><th>var</th><th>share</th></tr>")
        for name, v, share in anova_importance(model, X, top=12):
            parts.append(f"<tr><td>{name}</td><td>{v:.4g}</td><td>{share:.2%}</td></tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    with open(path, "w") as fh:
        fh.write("".join(parts))
    return path
