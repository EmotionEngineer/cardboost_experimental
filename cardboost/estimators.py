"""Scikit-learn compatible estimators: CardBoostRegressor / CardBoostClassifier.

``preset="v8"`` is the frozen R87 champion (``card_v8_base``); every knob maps
to a concrete engine flag (see ``presets.py``).  Full sklearn compatibility
(``clone``, ``Pipeline``, ``GridSearchCV``, ``cross_val_score``): all
constructor arguments are stored unmodified.

Deployment surface: ``to_bytes/from_bytes`` (Container v3.x),
``to_sql/to_c99/to_qr`` (codegen), ``certify`` (decode≡fit certification),
``to_scorecard_df``, ``plot_1d/plot_2d``, ``audit_ood_safety``,
``explain_local``, ``to_html_report``.
"""
from __future__ import annotations

import gzip
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from . import codec
from .presets import get_preset


def _as_names(X, feature_names):
    if feature_names is not None:
        return list(feature_names)
    if hasattr(X, "columns"):
        return [str(c) for c in X.columns]
    return [f"x{j}" for j in range(np.asarray(X).shape[1])]


def _temperature(spec):
    """Extract a scalar temperature from the calibration spec (1.0 fallback)."""
    if spec is None:
        return 1.0
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        for key in ("T", "temp", "temperature", "toof", "lambda_"):
            if key in spec:
                try:
                    return float(spec[key])
                except (TypeError, ValueError):
                    pass
    try:
        arr = np.asarray(spec, float).reshape(-1)
        if arr.size == 1:
            return float(arr[0])
    except (TypeError, ValueError):
        pass
    return 1.0


class _CardBoostBase(BaseEstimator):
    # ---- constructor: sklearn clone contract (store args verbatim) -------
    def __init__(self, preset="v8", max_bins=32, lr="formula", km="formula",
                 dead_col_law=None, currency="h3", enable_pairs=True,
                 atom_families=("g4", "p1", "m2"), ledger_mode="greedy",
                 warm_passes=6, edge_freeze=False, hmin_frac=0.0,
                 oof_merge=False, calibration="auto",
                 enable_log_selector=False, max_wallet_bytes=4096,
                 verify_bit_exact=True, deterministic_ulp=True, n_jobs=1,
                 random_state=109, potts_objective="quadratic",
                 include_qmiss=False, feature_names=None):
        self.preset = preset
        self.max_bins = max_bins
        self.lr = lr
        self.km = km
        self.dead_col_law = dead_col_law
        self.currency = currency
        self.enable_pairs = enable_pairs
        self.atom_families = atom_families
        self.ledger_mode = ledger_mode
        self.warm_passes = warm_passes
        self.edge_freeze = edge_freeze
        self.hmin_frac = hmin_frac
        self.oof_merge = oof_merge
        self.calibration = calibration
        self.enable_log_selector = enable_log_selector
        self.max_wallet_bytes = max_wallet_bytes
        self.verify_bit_exact = verify_bit_exact
        self.deterministic_ulp = deterministic_ulp
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.potts_objective = potts_objective
        self.include_qmiss = include_qmiss
        self.feature_names = feature_names

    # ---- preset resolution ---------------------------------------------
    def _overrides(self) -> dict:
        fam2star = {"g4": ("g4a", "g4b", "g4c", "g4d"), "p1": ("p1",),
                    "m2": ("m2",), "p1_2a": ("p1",), "safe_grid": (),
                    "msum": ("m2",), "q_miss": ()}
        star = []
        for f in self.atom_families:
            star.extend(fam2star.get(f, ()))
        ov = dict(
            max_bins=self.max_bins, lr=self.lr, km=self.km,
            warm_passes=self.warm_passes, edge_freeze=self.edge_freeze,
            hmin_frac=self.hmin_frac, oof_merge=self.oof_merge,
            enable_log_selector=self.enable_log_selector,
            max_wallet_bytes=self.max_wallet_bytes,
            ledger_mode=self.ledger_mode, n_jobs=self.n_jobs,
            random_state=self.random_state,
            potts_objective=self.potts_objective,
            include_qmiss=self.include_qmiss,
            verify_bit_exact=self.verify_bit_exact,
            deterministic_ulp=self.deterministic_ulp,
        )
        if star:
            ov["star_types"] = tuple(dict.fromkeys(
                ("g4a", "g4b", "g4c", "g4d", "p1", "m2") if self.enable_pairs
                else ("g4a", "g4b", "g4c", "g4d")))
        else:
            ov["star_types"] = (("g4a", "g4b", "g4c", "g4d", "p1", "m2")
                                if self.enable_pairs
                                else ("g4a", "g4b", "g4c", "g4d"))
        if self.dead_col_law is not None:
            # explicit mode override: "phase_v32" -> phase arm, "h3" -> control
            law = self.dead_col_law == "phase_v32"
            ov["dead_col_law"] = self.dead_col_law
            ov["p_phase_law"] = law
            ov["compress_ucuts"] = law
        get_preset(self.preset)  # validate early
        return ov

    def _fit_engine(self, X, y, kind, n_classes=0):
        from .core import engine
        names = _as_names(X, self.feature_names)
        Xa = np.asarray(X, float)
        data = dict(kind=kind, X=Xa, y=np.asarray(y), names=names,
                    geo=None, panel_name="custom")
        if n_classes:
            data["K"] = n_classes
        res = engine.fit_deploy(data, preset=self.preset,
                                overrides=self._overrides())
        if self.verify_bit_exact and (res.maxdiff != 0.0 or res.moved_train != 0):
            raise RuntimeError(
                f"bit-exact certification failed: maxdiff={res.maxdiff}, "
                f"moved_train={res.moved_train}")
        self.fit_card_ = res.card        # full-precision fit card (introspection)
        self.card_ = res.dec_card        # DEPLOY card: what predict/to_sql score with
        self.dec_card_ = res.dec_card
        self.payload_ = res.payload
        self.admitted_ = res.admitted
        self.Tcal_ = res.Tcal
        self.spec14_ = res.spec14
        self.v31_ = res.v31
        self.wallet_bytes_ = res.v31
        self.max_diff_ = res.maxdiff
        self.moved_train_ = res.moved_train
        self.kind_ = kind
        self.n_classes_ = res.n_classes
        self.feature_names_ = names
        return self

    # ---- serialization --------------------------------------------------
    def to_bytes(self) -> bytes:
        """Return the certified Container v3.x bytes produced at fit time.

        The payload is the exact artifact that was certified (``moved_train``,
        decode determinism); decoding it reproduces ``predict()`` bit-exactly.
        Re-encoding a decoded card programmatically (``codec.container_v32``)
        can move scores of rows lying exactly on a bin boundary (byte-quantised
        cut re-normalisation) - the estimator therefore serves the original
        certified artifact.
        """
        check_is_fitted(self, "card_")
        return self.payload_

    @classmethod
    def from_bytes(cls, blob: bytes, **kw):
        obj = cls(**kw)
        raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
        dec = codec.container_v32.from_bytes(raw)
        obj.dec_card_ = dec
        obj.card_ = dec
        obj.fit_card_ = None
        obj.payload_ = raw
        obj.kind_ = dec.kind
        obj.n_classes_ = len(dec.b0) if dec.kind == "multiclass" else 0
        obj.feature_names_ = None
        obj.Tcal_ = obj.spec14_ = None
        return obj

    def certify(self, X_train, X_test=None) -> dict:
        """Container certification: payload decode must be exact and stable.

        Returns wallet bytes, ``max_diff`` of ``from_bytes(to_bytes())`` vs
        ``predict()`` (must be 0.0), ``moved_train`` of the original fit, and
        the informational ``reencode_drift`` (fresh re-emit of the decoded
        card; nonzero only on rows exactly on quantised bin boundaries).
        """
        check_is_fitted(self, "card_")
        X = np.asarray(X_test if X_test is not None else X_train, float)
        import r76_container31 as C
        blob = self.to_bytes()
        dec2 = codec.container_v32.from_bytes(blob)
        s1 = np.asarray(C.model_score(self.card_, X), float)
        s2 = np.asarray(C.model_score(dec2, X), float)
        max_diff = float(np.max(np.abs(s1 - s2))) if len(s1) else 0.0
        reemit = codec.container_v32.from_bytes(
            codec.container_v32.to_bytes(self.card_,
                                         budget=self.max_wallet_bytes))
        s3 = np.asarray(C.model_score(reemit, X), float)
        drift = float(np.max(np.abs(s1 - s3))) if len(s1) else 0.0
        return dict(wallet_bytes=self.v31_, max_diff=max_diff,
                    moved_train=int(getattr(self, "moved_train_", 0)),
                    reencode_drift=drift,
                    decode_ok=bool(max_diff == 0.0))

    # ---- codegen --------------------------------------------------------
    def to_sql(self, table_name="scoring_view", dialect="postgres",
               score_col="card_score", pred_col="prediction") -> str:
        check_is_fitted(self, "card_")
        return codec.sql_export.to_sql(self.dec_card_, table_name=table_name,
                                       dialect=dialect,
                                       names=self.feature_names_,
                                       score_col=score_col, pred_col=pred_col)

    def to_c99(self, func_name="cardboost_score") -> str:
        check_is_fitted(self, "card_")
        return codec.c99_export.to_c99(self.dec_card_, func_name=func_name,
                                       names=self.feature_names_)

    def to_qr(self, path, **kw) -> str:
        check_is_fitted(self, "card_")
        return codec.qr_export.to_qr(self.card_, path,
                                     budget=self.max_wallet_bytes, **kw)

    def wallet(self, X=None):
        check_is_fitted(self, "card_")
        Xtr = np.zeros((1, 1)) if X is None else np.asarray(X, float)
        return codec.container_v32.wallet(self.card_, Xtr,
                                          budget=self.max_wallet_bytes)


class CardBoostRegressor(_CardBoostBase, RegressorMixin):
    """Glass-box additive scorecard regressor (IsoLink-OOF calibration)."""

    def fit(self, X, y, feature_names=None):
        X, y = check_X_y(X, y, dtype=None, force_all_finite="allow-nan",
                         y_numeric=True)
        if feature_names is not None:
            self.feature_names = feature_names
        return self._fit_engine(X, y, "regression")

    def predict(self, X):
        check_is_fitted(self, "card_")
        Xa = np.asarray(X, float)
        return np.asarray(codec.container_v32.C.model_score(self.card_, Xa), float)

    def to_scorecard_df(self):
        from .explain.scorecard_table import scorecard_df
        return scorecard_df(self)

    def plot_1d(self, X=None, feature=None, **kw):
        from .plot.figures import plot_1d
        return plot_1d(self, X=X, feature=feature, **kw)

    def plot_2d(self, pair=0, **kw):
        from .plot.figures import plot_2d
        return plot_2d(self, pair=pair, **kw)

    def audit_ood_safety(self, threshold=0.15):
        from .explain.audit import audit_ood_safety
        return audit_ood_safety(self, threshold=threshold)

    def explain_local(self, x):
        from .explain.local import explain_local
        return explain_local(self, x)

    def to_html_report(self, path, X=None, y=None):
        from .plot.report import html_report
        return html_report(self, path, X=X, y=y)


class CardBoostClassifier(_CardBoostBase, ClassifierMixin):
    """Glass-box additive scorecard classifier (Toof binary / 14b multiclass)."""

    def fit(self, X, y, feature_names=None):
        X, y = check_X_y(X, y, dtype=None, force_all_finite="allow-nan")
        if feature_names is not None:
            self.feature_names = feature_names
        classes = np.unique(y)
        self.classes_ = classes
        if len(classes) < 2:
            raise ValueError("need >= 2 classes")
        kind = "multiclass" if len(classes) > 2 else "binary"
        y_enc = np.searchsorted(classes, y).astype(int) if kind == "multiclass" \
            else np.asarray(y, float)
        n_classes = len(classes) if kind == "multiclass" else 0
        return self._fit_engine(X, y_enc, kind, n_classes=n_classes)

    def _scores(self, X):
        check_is_fitted(self, "card_")
        Xa = np.asarray(X, float)
        s = np.asarray(codec.container_v32.C.model_score(self.card_, Xa), float)
        return s

    def decision_function(self, X):
        return self._scores(X)

    def predict_proba(self, X):
        s = self._scores(X)
        if self.kind_ == "multiclass":
            s = np.atleast_2d(s.T) if s.ndim == 2 else s.reshape(1, -1)
            e = np.exp(s - s.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        T = _temperature(self.Tcal_)
        z = np.clip(s / max(T, 1e-9), -50, 50)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self.classes_[idx]

    def to_scorecard_df(self):
        from .explain.scorecard_table import scorecard_df
        return scorecard_df(self)

    def plot_1d(self, X=None, feature=None, **kw):
        from .plot.figures import plot_1d
        return plot_1d(self, X=X, feature=feature, **kw)

    def audit_ood_safety(self, threshold=0.15):
        from .explain.audit import audit_ood_safety
        return audit_ood_safety(self, threshold=threshold)

    def explain_local(self, x):
        from .explain.local import explain_local
        return explain_local(self, x)

    def to_html_report(self, path, X=None, y=None):
        from .plot.report import html_report
        return html_report(self, path, X=X, y=y)
