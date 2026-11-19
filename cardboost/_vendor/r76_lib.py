#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r76_lib.py — Round 76 harness.

Rebuilds the exact r72 4-panel protocol (v6_benchmark_section.py) on top of the frozen
engine r71_universal.py, adds the r74/r75 registered calibrators (14a glob, 14b diag) and
provides the shared measurement / logging surface for the Stage 3.1 / 4 / 5 experiments.

Panels (identical to the r72 checkpoint in r72_rows.json):
  california_random  : KFold(5, shuffle=True, random_state=42), full 20,640 rows, R2
  california_spatial : rows sorted by Latitude, 5 contiguous blocks (bs = n//5, last absorbs), R2
  anneal             : anneal.arff (898 rows), StratifiedKFold(3, shuffle=True, rs=42), logloss
  aps                : aps_failure_sample.csv -> deterministic stratified subsample to N rows
                       (rng seed 20261021, per-class proportionate), StratifiedKFold(3, rs=42), logloss

Anchors certified by r76_0_controls.py (see that script for the table):
  card_fused T20  cal_rand 0.75155 / cal_spat 0.45277
  card_fused Toof aps 0.03717 / anneal 0.03284
  card_kappa0 Toof aps 0.03385 (14a 0.03370) / anneal 0.03402 (14a 0.03023, 14b 0.02863)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/google_drive")
import r71_universal as r71  # noqa: E402
from r71_universal import (  # noqa: E402
    Card, FitCfg, bin_idx, cuts_of, fit_card, fit_T, metric, onehot, log_softmax, sig, softmax,
)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r76_results")
os.makedirs(RESULTS, exist_ok=True)

# ----------------------------------------------------------------------------
# Panel loading — byte-for-byte the v6_benchmark_section protocol
# ----------------------------------------------------------------------------

ANNEAL_COLS = ["family", "product-type", "steel", "carbon", "hardness", "temper_rolling",
               "condition", "formability", "strength", "non-ageing", "surface-finish",
               "surface-quality", "enamelability", "bc", "bf", "bt", "bw%2Fme", "bl", "m",
               "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr",
               "blue%2Fbright%2Fvarn%2Fclean", "lustre", "jurofm", "s", "p", "shape",
               "thick", "width", "len", "oil", "bore", "packing"]


def load_anneal_df() -> pd.DataFrame:
    """glassbeams get_anneal_data() minus the fold split.

    Row order == OpenML anneal (id 2) == UCI anneal.data + anneal.test concatenated (verified
    row-by-row in the session log). The glassbeams pipeline is reproduced literally: '?' and '-'
    -> NaN, then a column is numeric iff >50% of its values parse as numbers, else category.
    """
    p1 = os.path.join(DATA, "anneal.data")
    p2 = os.path.join(DATA, "anneal.test")
    frames = [pd.read_csv(p, header=None, dtype=str) for p in (p1, p2)]
    df = pd.concat(frames, ignore_index=True)
    names = ANNEAL_COLS[:df.shape[1] - 1] + ["class"]
    if len(names) != df.shape[1]:
        raise RuntimeError("anneal column count mismatch: %d" % df.shape[1])
    df.columns = names
    y_raw = df["class"].astype(str)
    X = df.drop(columns=["class"]).copy()
    X = X.replace({"?": np.nan, "-": np.nan})
    for c in X.columns:
        converted = pd.to_numeric(X[c], errors="coerce")
        if converted.notna().sum() > len(X) * 0.5:
            X[c] = converted
        else:
            X[c] = X[c].astype("category")
    classes = sorted(y_raw.unique())
    y = np.array([classes.index(v) for v in y_raw], int)
    return X, y, classes


def anneal_numeric(X: pd.DataFrame) -> np.ndarray:
    """v6 bench encoding: category codes for non-numeric columns, -1 -> NaN."""
    Xn = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            Xn[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    return Xn.to_numpy(float)


def load_aps_df(max_rows: int, source: str = "auto") -> tuple:
    """glassbeams get_aps_failure_data(max_rows): deterministic stratified subsample, seed 20261021."""
    if source == "auto":
        path = os.path.join(DATA, "aps_failure_sample.csv")
    else:
        path = os.path.join(DATA, source)
    df = pd.read_csv(path, na_values="na")
    y_raw = (df["class"] == "pos").astype(int).to_numpy()
    X = df.drop(columns=["class"]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if len(X) > max_rows:
        rng = np.random.default_rng(20261021)
        pos_idx = np.where(y_raw == 1)[0]
        neg_idx = np.where(y_raw == 0)[0]
        n_pos = int(round(max_rows * len(pos_idx) / len(X)))
        n_neg = max_rows - n_pos
        take_pos = rng.choice(pos_idx, min(len(pos_idx), n_pos), replace=False)
        take_neg = rng.choice(neg_idx, min(len(neg_idx), n_neg), replace=False)
        take = np.sort(np.concatenate([take_pos, take_neg]))
        X = X.iloc[take].reset_index(drop=True)
        y = y_raw[take]
    else:
        y = y_raw
    return X, y, list(X.columns)


def build_panels(aps_rows: int = 6000, aps_source: str = "auto") -> dict:
    """All four panels with the r72 folds."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import KFold, StratifiedKFold

    d = fetch_california_housing(as_frame=True)
    Xdf, ycal = d.data, np.asarray(d.target, float)
    Xcal = Xdf.to_numpy(float)
    rand_folds = list(KFold(5, shuffle=True, random_state=42).split(Xcal))
    order = np.argsort(Xdf["Latitude"].to_numpy())
    bs = len(order) // 5
    spat_folds = [(np.setdiff1d(order, order[b * bs:(b + 1) * bs if b < 4 else len(order)]),
                   order[b * bs:(b + 1) * bs if b < 4 else len(order)]) for b in range(5)]
    cal_data = dict(kind="regression", X=Xcal, y=ycal, names=list(Xdf.columns),
                    geo=(Xdf["Latitude"].to_numpy(float), Xdf["Longitude"].to_numpy(float)))

    Xa_df, ya, K_classes = load_anneal_df()
    ann_X = anneal_numeric(Xa_df)
    ann_data = dict(kind="multiclass", X=ann_X, y=ya, names=list(Xa_df.columns), geo=None, K=K_classes)

    Xp_df, yp, aps_names = load_aps_df(aps_rows, aps_source)
    aps_data = dict(kind="binary", X=Xp_df.to_numpy(float), y=yp, names=aps_names, geo=None)

    ann_folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(ann_X, ya))
    aps_folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(aps_data["X"], yp))

    return dict(
        california_random=(cal_data, rand_folds),
        california_spatial=(cal_data, spat_folds),
        anneal=(ann_data, ann_folds),
        aps=(aps_data, aps_folds),
    )


# ----------------------------------------------------------------------------
# Fit configuration — the frozen "standard" card (r72 certified)
# ----------------------------------------------------------------------------

def std_fitcfg(kind: str, n_classes: int = 0, **kw) -> FitCfg:
    base = dict(m=32, passes=24, max_passes=48, conv_tol=1e-5, alpha=0.5,
                level_shrink=0.0, n_classes=0)
    if kind == "multiclass":
        base.update(order="fisher", n_classes=n_classes)
    else:
        base.update(order="cyclic")
    base.update(kw)
    return FitCfg(**base)


def fit_std(Xtr, ytr, kind, kappa_mult=1.0, n_classes=0, names=None, **kw) -> Card:
    cfg = std_fitcfg(kind, n_classes=n_classes, kappa_mult=kappa_mult, **kw)
    return fit_card(Xtr, ytr, kind, cfg, feat_names=names)


# ----------------------------------------------------------------------------
# Calibrators.  T20 / Toof-T exactly as r71; 14a (glob) and 14b (diag) per the
# r74 registry: eta' = beta_k * eta_k + delta_k, ridge 1e-3 toward (1, 0), fitted
# on Toof OOF logits.
# ----------------------------------------------------------------------------

def toof_eta(factory, Xtr, ytr, kind, folds: int = 5, seed: int = 109):
    """OOF raw scores by refitting `factory(Xtr, ytr, kind)` on r71's registered inner folds."""
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(ytr)), folds)
    if kind == "multiclass":
        # K known after first fit
        probe = factory(Xtr, ytr, kind)
        K = len(np.asarray(probe.b0).ravel())
        eta = np.zeros((len(ytr), K))
    else:
        eta = np.zeros(len(ytr))
    for k in range(folds):
        va = parts[k]
        tr = np.concatenate([parts[g] for g in range(folds) if g != k])
        ck = factory(Xtr[tr], ytr[tr], kind)
        eta[va] = ck.score(Xtr[va])
    return eta


def fit_T14(y, eta, kind, per_class: bool, ridge: float = 1e-3):
    """14a (per_class=False): one (beta, delta); 14b (per_class=True): (beta_k, delta_k).

    Fitted on OOF logits; objective = mean NLL + ridge * sum((beta-1)^2 + delta^2).
    Regression: identity (returns None).
    """
    if kind == "regression":
        return None
    from scipy.optimize import minimize
    if kind == "binary":
        e = np.asarray(eta, float).ravel()
        Y = np.asarray(y, float)

        def obj(th):
            b, d = th[0], th[1]
            z = b * e + d
            return float(np.mean(np.logaddexp(0.0, z) - Y * z) + ridge * ((b - 1.0) ** 2 + d ** 2))

        r = minimize(obj, np.array([1.0, 0.0]), method="L-BFGS-B")
        return dict(beta=np.array([r.x[0]]), delta=np.array([r.x[1]]), ok=bool(r.success))

    e = np.asarray(eta, float)
    Y = onehot(np.asarray(y, int), e.shape[1])
    K = e.shape[1]
    if not per_class:
        def obj(th):
            b, d = th[0], th[1]
            z = b * e + d
            return float(-np.mean(np.sum(Y * log_softmax(z), axis=1)) + ridge * ((b - 1.0) ** 2 + d ** 2))

        r = minimize(obj, np.array([1.0, 0.0]), method="L-BFGS-B")
        return dict(beta=np.full(K, r.x[0]), delta=np.full(K, r.x[1]), ok=bool(r.success))

    def obj2(th):
        b = th[:K]
        d = th[K:]
        z = e * b[None, :] + d[None, :]
        return float(-np.mean(np.sum(Y * log_softmax(z), axis=1))
                     + ridge * (np.sum((b - 1.0) ** 2) + np.sum(d ** 2)))

    r = minimize(obj2, np.concatenate([np.ones(K), np.zeros(K)]), method="L-BFGS-B")
    return dict(beta=r.x[:K], delta=r.x[K:], ok=bool(r.success))


def apply_T14(spec, score):
    if spec is None:
        return score
    b = np.asarray(spec["beta"], float).ravel()
    d = np.asarray(spec["delta"], float).ravel()
    if score.ndim == 1:
        return b[0] * score + d[0]
    return score * b[None, :] + d[None, :]


# ----------------------------------------------------------------------------
# Evaluation + logging
# ----------------------------------------------------------------------------

def score_metrics(kind, yte, s):
    return metric(kind, yte, s)


def log_jsonl(fname: str, row: dict):
    path = os.path.join(RESULTS, fname)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def mean_r2(fold_rows):
    """Mean of per-fold R2 (the record's headline convention) + pooled R2."""
    vals = [r["metric"] for r in fold_rows]
    return float(np.mean(vals)), float(np.std(vals))


def pooled_r2(fold_rows):
    sse = sum(r.get("sse", 0.0) for r in fold_rows)
    sy = sum(r.get("sy", 0.0) for r in fold_rows)
    syy = sum(r.get("syy", 0.0) for r in fold_rows)
    nn = sum(r.get("n_obs", 0.0) for r in fold_rows)
    sst = syy - sy * sy / max(nn, 1.0)
    return float(1.0 - sse / sst) if sst > 1e-12 else float("nan")


def fold_stats(kind, yte, s):
    if kind == "regression":
        return dict(sse=float(((yte - s) ** 2).sum()), sy=float(yte.sum()),
                    syy=float((yte ** 2).sum()), n_obs=int(len(yte)))
    return dict(sse=0.0, sy=0.0, syy=0.0, n_obs=int(len(yte)))
