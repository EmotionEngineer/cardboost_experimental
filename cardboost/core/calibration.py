"""Calibration wrappers: Toof binary, 14b multiclass, IsoLink-OOF regression.

Thin, documented access points over the certified calibration stack:
``r71_universal.fit_calib_spec / apply_metric / score_with_calibration``,
``r76_cal01.toof_eta`` and ``r83_3_wallet14.iso_deploy``.
"""
from __future__ import annotations

import numpy as np

from .. import _vendor  # noqa: F401
import r71_universal as L  # noqa: E402


def fit_calib_spec(admitted, kind, n_classes, Xtr, ytr, oof_merge=False):
    import r83_3_wallet14 as W
    return W.fit_calib_spec(admitted, kind, n_classes, Xtr, ytr,
                            oof_merge=oof_merge)


def apply_metric(kind, y, score, Tcal, spec14):
    import r83_3_wallet14 as W
    return W.apply_metric(kind, y, score, Tcal, spec14)


def metric_value(metric: dict) -> float:
    """Scalar optimisation direction: logloss (binary/multiclass) or -R2."""
    if "logloss" in metric:
        return float(metric["logloss"])
    if "R2" in metric:
        return -float(metric["R2"])
    vals = [float(v) for v in metric.values()]
    return vals[0]


def toof_eta(p):
    import r76_cal01 as K
    return K.toof_eta(p)


def iso_deploy(eta_oof, y_oof, sigma2=1.0, n_knots=16):
    import r83_3_wallet14 as W
    return W.iso_deploy(eta_oof, y_oof, sigma2, n_knots=n_knots)
