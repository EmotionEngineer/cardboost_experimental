"""Publication figures (matplotlib optional): 1D mains, 2D pair surfaces.

Both functions return the figure; they also work headless (``Agg``).  A
comparison helper plots a card against an EBM-style binned baseline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_1d(model, X=None, feature=0, ax=None, show_mass: bool = True):
    """Step-function main effect of `feature` (mass histogram optional)."""
    plt = _plt()
    card = getattr(model, "dec_card_", None) or getattr(model, "card_", None)
    names = getattr(model, "feature_names_", None)
    j = feature
    cuts = np.asarray(card.cuts[j], float).reshape(-1)
    V = np.asarray(card.V[j], float)
    lvl = V if V.ndim == 1 else V.sum(axis=1)
    m = len(cuts) + 2
    edges = np.concatenate([[-np.inf], cuts, [np.inf]])
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.4))
    for bi in range(m):
        lo, hi = edges[bi], edges[bi + 1]
        if not np.isfinite(lo):
            lo = cuts[0] - (cuts[-1] - cuts[0]) * 0.05 if len(cuts) else -1
        if not np.isfinite(hi):
            hi = cuts[-1] + (cuts[-1] - cuts[0]) * 0.05 if len(cuts) else 1
        ax.hlines(lvl[bi], lo, hi, lw=2, color="#1f5fa8")
    nm = names[j] if names else f"x{j}"
    ax.set_xlabel(nm)
    ax.set_ylabel("contribution")
    ax.set_title(f"Main effect: {nm}")
    if X is not None and show_mass:
        import r76_container31 as C
        import r71_universal as _L
        b = _L.bin_idx(np.asarray(X)[:, j], cuts) if len(cuts) \
            else np.zeros(len(X), int)
        mass = np.bincount(b, minlength=m) / max(len(X), 1)
        ax2 = ax.twinx()
        ax2.bar(edges[1:-1], mass[:-1], width=np.maximum(
            [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)] or [1.0], 1e-9),
            alpha=0.15, color="#888888", align="edge")
        ax2.set_ylabel("mass")
    return ax.get_figure()


def plot_2d(model, pair=0, n_grid: int = 60, ax=None):
    """Heatmap of the `pair`-th pair atom over its parent grid."""
    plt = _plt()
    card = getattr(model, "dec_card_", None) or getattr(model, "card_", None)
    atoms = [a for a in card.atoms if hasattr(a, "k")]
    at = atoms[pair]
    names = getattr(model, "feature_names_", None)
    j, k = at.j, at.k
    gx = np.linspace(getattr(at, "xmin_j", np.nanmin(card.cuts[j]) if len(card.cuts[j]) else -1),
                     getattr(at, "xmax_j", np.nanmax(card.cuts[j]) if len(card.cuts[j]) else 1), n_grid)
    gy = np.linspace(getattr(at, "xmin_k", np.nanmin(card.cuts[k]) if len(card.cuts[k]) else -1),
                     getattr(at, "xmax_k", np.nanmax(card.cuts[k]) if len(card.cuts[k]) else 1), n_grid)
    Xg = np.zeros((n_grid * n_grid, len(card.cuts)))
    Xg[:, j] = np.repeat(gx, n_grid)
    Xg[:, k] = np.tile(gy, n_grid)
    Z = np.asarray(at.contrib(Xg), float).reshape(n_grid, n_grid)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap="RdBu_r")
    ax.set_xlabel(names[j] if names else f"x{j}")
    ax.set_ylabel(names[k] if names else f"x{k}")
    ax.set_title(f"{type(at).__name__}({j},{k})")
    ax.get_figure().colorbar(im, ax=ax)
    return ax.get_figure()


def compare_with_ebm(model, X, y, ebm_pred, name_card="cardboost",
                     name_ebm="ebm"):
    """Parity scatter card vs EBM predictions (R69 comparison family)."""
    plt = _plt()
    pred = np.asarray(model.predict(X), float)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(ebm_pred, pred, s=6, alpha=0.4, color="#1f5fa8")
    lim = [min(pred.min(), np.min(ebm_pred)), max(pred.max(), np.max(ebm_pred))]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel(name_ebm)
    ax.set_ylabel(name_card)
    ax.set_title("card vs EBM parity")
    return fig
