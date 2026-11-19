"""Canonical panels and fold splits of the CardBoost research protocol.

`build_panels()` is a drop-in equivalent of the engine's
``r76_lib.build_panels()``: identical arrays, identical folds.  Equivalence is
asserted by ``tests/test_datasets_equivalence.py`` whenever scikit-learn can
fetch the California housing dataset.

Bundled raw data (``data_cache/`` next to this file's package root):

* ``california_housing.npz``  - X (20640x8 float64), y, feature names, Latitude,
  Longitude, extracted verbatim from ``sklearn.datasets.fetch_california_housing``
  so the panel loads with **no network access**;
* ``anneal.data`` + ``anneal.test`` - UCI anneal (898 rows);
* ``aps_failure_sample.csv`` - APS Failure Prediction sample (stratified
  subsample source, deterministic seed 20261021 -> 6000 rows).

Folds (frozen since round 72):

* ``california_random`` : KFold(5, shuffle=True, random_state=42)
* ``california_spatial``: rows sorted by Latitude, 5 contiguous blocks
* ``anneal``            : StratifiedKFold(3, shuffle=True, random_state=42)
* ``aps``               : StratifiedKFold(3, shuffle=True, random_state=42)
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("CARDBOOST_DATA", os.path.join(_PKG_ROOT, "data_cache"))

# Panel names -------------------------------------------------------------
CAL_RANDOM = "california_random"
CAL_SPATIAL = "california_spatial"
ANNEAL = "anneal"
APS = "aps"

PanelBundle = Tuple[dict, List[Tuple[np.ndarray, np.ndarray]]]


def _vendor():
    from . import _vendor  # noqa: F401  (sys.path side effect)
    import r76_lib as L
    return L


def load_california() -> dict:
    """California housing panel dict {kind, X, y, names, geo} from the bundled npz."""
    npz = os.path.join(DATA_DIR, "california_housing.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=False)
        X = np.asarray(z["X"], float)
        y = np.asarray(z["y"], float)
        names = [str(s) for s in z["names"]]
        lat = np.asarray(z["lat"], float)
        lon = np.asarray(z["lon"], float)
    else:  # pragma: no cover - network fallback
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        Xdf = d.data
        X = Xdf.to_numpy(float)
        y = np.asarray(d.target, float)
        names = list(Xdf.columns)
        lat = Xdf["Latitude"].to_numpy(float)
        lon = Xdf["Longitude"].to_numpy(float)
    return dict(kind="regression", X=X, y=y, names=names, geo=(lat, lon))


def load_anneal() -> dict:
    L = _vendor()
    L.DATA = DATA_DIR
    Xa_df, ya, K_classes = L.load_anneal_df()
    X = L.anneal_numeric(Xa_df)
    return dict(kind="multiclass", X=X, y=ya, names=list(Xa_df.columns),
                geo=None, K=K_classes)


def load_aps(max_rows: int = 6000) -> dict:
    L = _vendor()
    L.DATA = DATA_DIR
    Xp_df, yp, names = L.load_aps_df(max_rows, "auto")
    return dict(kind="binary", X=Xp_df.to_numpy(float), y=yp, names=names, geo=None)


def _folds_cal_random(X: np.ndarray):
    from sklearn.model_selection import KFold
    return list(KFold(5, shuffle=True, random_state=42).split(X))


def _folds_cal_spatial(lat: np.ndarray, n: int):
    order = np.argsort(lat)
    bs = n // 5
    return [(np.setdiff1d(order, order[b * bs:(b + 1) * bs if b < 4 else n]),
             order[b * bs:(b + 1) * bs if b < 4 else n]) for b in range(5)]


def _folds_strat3(X: np.ndarray, y: np.ndarray):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))


def build_panels(aps_rows: int = 6000) -> Dict[str, PanelBundle]:
    """All four panels with the frozen r72 folds. Identical to r76_lib.build_panels()."""
    cal = load_california()
    Xcal = cal["X"]
    lat, lon = cal["geo"]
    cal_data = dict(kind="regression", X=Xcal, y=cal["y"], names=cal["names"],
                    geo=(lat, lon))
    rand_folds = _folds_cal_random(Xcal)
    spat_folds = _folds_cal_spatial(lat, len(Xcal))

    ann = load_anneal()
    ann_folds = _folds_strat3(ann["X"], ann["y"])

    aps = load_aps(aps_rows)
    aps_folds = _folds_strat3(aps["X"], aps["y"])

    return {
        CAL_RANDOM: (cal_data, rand_folds),
        CAL_SPATIAL: (cal_data, spat_folds),
        ANNEAL: (ann, ann_folds),
        APS: (aps, aps_folds),
    }


CANONICAL_FOLDS: List[Tuple[str, int]] = (
    [("anneal", f) for f in range(3)]
    + [("aps", f) for f in range(3)]
    + [(CAL_RANDOM, f) for f in range(5)]
    + [(CAL_SPATIAL, f) for f in range(5)]
)
