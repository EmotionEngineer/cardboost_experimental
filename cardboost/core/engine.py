"""Fitting engine: the exact research path of the canonical R87 pipeline.

``fit_fold`` mirrors ``r87_4_qmiss.fold_fit`` (arm='off') line by line: same
configuration block, same `run_fold77` call signatures, same calibration and
container emission.  That is why the golden tests can assert bit-exact equality
against the canonical 16-fold table recorded in ``golden/canonical_r87.json``.

Public surface
--------------
* ``fit_fold(panel, fold, data, splits, ...)`` -> :class:`FoldResult`
* ``fit_deploy(data, honest_split, ...)``      -> :class:`DeployResult` (production fit)
* ``FoldResult`` / ``DeployResult``            -> dataclasses with card, payload,
  metrics, wallet and certification numbers.
"""
from __future__ import annotations

import os
import pickle
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import _vendor  # noqa: F401  (sys.path side effect)
from ..presets import PresetConfig, get_preset

import r71_universal as U          # noqa: E402
import r76_container31 as C        # noqa: E402
import r76_lib as L                # noqa: E402
import r76_mech as M6              # noqa: E402
import r77_mech as R               # noqa: E402
import r77_stage as T              # noqa: E402
import r83_3_wallet14 as W         # noqa: E402
import r84_1_currency as C1        # noqa: E402


# ----------------------------------------------------------------- config
def apply_config(panel: str, cfg: PresetConfig) -> None:
    """Frozen `_cfg` block of the R87 pipeline (+ preset overrides).

    Every mutable engine global is set deterministically on each call so
    presets cannot leak into one another (the D1 class of bug).
    """
    nom_perm = cfg.nom_perm if cfg.nom_perm is not None else (not panel.startswith("california"))
    R.P1_SHRINK_POST = False
    R.P1_BINS = int(cfg.extra.get("p1_bins_k", 8))   # R98 S_jx32 grids (iid_max)
    R.HANDLE_A = False
    R.QGATE = None
    R.R84_FIT_RANK = False
    # R71/R73 card_kappa0 (unregularized_32): l0 merge price -> 0
    # (r77_stage: kappa_mult = law_use if km == "formula" else float(km)).
    km_eff = 0.0 if cfg.extra.get("kappa_zero") else cfg.km
    T.set_fitcfg(m=cfg.max_bins, kappa_sched=None, warm_passes=cfg.warm_passes,
                 lr=cfg.lr, km=km_eff, warm_lr=1.0, gram_lr=False,
                 base_passes=None, base_conv_tol=None,
                 solver=cfg.solver, nom_perm=nom_perm,
                 p_active_law=False, p_phase_law=cfg.p_phase_law)
    if cfg.potts_objective in ("exact_dev", "qfix_dev"):
        M6.EXACT_DEV["on"] = True
        M6.EXACT_DEV["cap"] = True
    else:
        M6.EXACT_DEV["on"] = False
        M6.EXACT_DEV["cap"] = False
    C.COMPRESS_UCUTS = cfg.compress_ucuts   # R109 Formula 2 (phase arm only)
    # R97 H1H2 hard segment-mass floor: any truthy value enables
    # h_min = max(25, 0.05 * bin_mass) in the potts DP (r71_universal).
    U.R84_COST["hmin"] = float(cfg.hmin_frac or 0.0)


def _run_kwargs(panel: str, cfg: PresetConfig) -> dict:
    """Exact run_fold77 kwargs of the canonical pipeline (per panel family)."""
    if panel.startswith("california"):
        kw = dict(K_pairs=20 if cfg.K_pairs is None else cfg.K_pairs,
                  K_refresh=cfg.K_refresh, max_admit=cfg.max_admit,
                  star_types=tuple(cfg.star_types), frag_mode=cfg.frag_mode,
                  screen_mode="s12_pure", oof_merge=cfg.oof_merge)
    else:
        if cfg.enable_pairs:
            default_stars = ("g4a", "g4b", "g4c", "g4d", "p1", "m2")
            stars = (W.STEP3_TYPES
                     if tuple(cfg.star_types) == default_stars
                     else tuple(cfg.star_types))
        else:
            stars = ("g4a", "g4b", "g4c", "g4d")
        kw = dict(K_pairs=0 if cfg.K_pairs is None else cfg.K_pairs,
                  K_refresh=cfg.K_refresh, max_admit=cfg.max_admit,
                  star_types=stars,
                  pool_mode="active", screen_mode="legacy",
                  oof_merge=cfg.oof_merge)
    return kw


# ------------------------------------------------------------- dataclasses
@dataclass
class FoldResult:
    panel: str
    fold: int
    kind: str
    card: Any
    dec_card: Any
    admitted: list
    Tcal: Any
    spec14: Any
    payload: bytes
    info: dict
    metric_fit: dict
    metric_dec: dict
    metric_mem: dict
    maxdiff: float
    moved_train: int
    v31: int
    mech: dict = field(default_factory=dict)
    quant_gap: float = 0.0

    def as_row(self) -> dict:
        return dict(panel=self.panel, fold=self.fold, kind=self.kind,
                    metric_fit=self.metric_fit, metric_dec=self.metric_dec,
                    metric_mem=self.metric_mem, maxdiff_mem_dec=self.maxdiff,
                    moved_train=self.moved_train, v31=self.v31, mech=self.mech,
                    quant_gap=self.quant_gap)


@dataclass
class DeployResult:
    card: Any
    dec_card: Any
    admitted: list
    Tcal: Any
    spec14: Any
    payload: bytes
    info: dict
    v31: int
    maxdiff: float
    moved_train: int
    kind: str
    n_classes: int
    quant_gap: float = 0.0
    mech: dict = field(default_factory=dict)


# ------------------------------------------------------------------ core
def _reset_cost():
    U.R84_COST["scale"] = 1.0
    U.R84_COST["pair_exempt"] = False


def fit_fold(panel: str, fold: int, data: dict,
             splits: List[Tuple[np.ndarray, np.ndarray]],
             preset: str = "v8", overrides: Optional[dict] = None,
             work_dir: Optional[str] = None,
             keep_work: bool = False) -> FoldResult:
    """Fit + certify one canonical fold.  Bit-exact path of the R87 pipeline."""
    cfg = get_preset(preset).with_overrides(**(overrides or {}))
    tr_idx, te_idx = splits[fold]
    kind = data["kind"]
    X, y = data["X"], data["y"]
    n_classes = int(np.max(y)) + 1 if kind == "multiclass" else 0
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    work = work_dir or os.path.join("cardboost_work", preset)
    save_dir = os.path.join(work, "cards", f"{panel}_f{fold}")
    os.makedirs(save_dir, exist_ok=True)

    apply_config(panel, cfg)
    sig2 = 1.0
    if panel.startswith("california"):
        sig2 = C1.sigma2_oof(Xtr, ytr)
        U.R84_COST["scale"] = 1.0 / sig2
        U.R84_COST["pair_exempt"] = True
    else:
        U.R84_COST["scale"] = 1.0
        U.R84_COST["pair_exempt"] = False

    out_jsonl = os.path.abspath(os.path.join(work, "telemetry.jsonl"))
    W.PANELS[panel] = (data, splits)
    kw = _run_kwargs(panel, cfg)
    try:
        T.run_fold77(panel, fold, tr_idx, te_idx, data, out_jsonl,
                     save_dir=save_dir, **kw)
        d = pickle.load(open(os.path.join(save_dir, f"{panel}_f{fold}.pkl"), "rb"))
        card = d["card"]
        admitted = list(d.get("admitted", []))

        # ---- optional post-fit experimental atoms (presets) --------------
        mech: Dict[str, Any] = {}
        if cfg.include_qmiss:
            import r87_qmiss as QM
            at, m0 = QM.build_atom(card, Xtr, ytr, kind, n_classes)
            if at is None:
                mech.update(m0)
            else:
                off_arr = np.asarray(card.score(Xtr), float)
                at.refit(Xtr, ytr, off_arr, kind, kappa_mult=1.0)
                fac = W.calib_factory(admitted, n_classes, oof_merge=False)
                eta_oof = L.toof_eta(fac, Xtr, ytr, kind, folds=5, seed=cfg.random_state)
                z1 = eta_oof + at.contrib(Xtr)
                dL = float(np.sum(ytr * (z1 - eta_oof)
                                  - (np.logaddexp(0.0, z1) - np.logaddexp(0.0, eta_oof))))
                barrier = 0.5 * np.log(max(len(ytr), 2)) + np.log(2.0)
                admit = bool(dL > barrier)
                mech.update(m0, gain=round(dL, 3), barrier=round(barrier, 3),
                            admitted=admit)
                if admit:
                    card.atoms.append(at)

        # ---- optional post-fit layers (presets, canary protocols) --------
        if cfg.include_safe_tier:
            # R113 safe_value_tier (r87_5 canary protocol): Dykstra-centred
            # atom-owned-marginal safe tier over the base card.
            import r87_safetier as ST
            tier_atoms, m0 = ST.build_safe_tier(card, Xtr, ytr, kind,
                                                n_classes,
                                                cache=os.path.join(save_dir,
                                                                   "tier"))
            card.atoms.extend(tier_atoms)
            mech["safe_tier"] = m0
        if cfg.ledger_mode == "beam3":
            # R114 beam3 prequential ledger (r87_6 canary protocol).
            import r87_beam3 as B3
            finalists, m0 = B3.beam_search(card, Xtr, ytr, kind, n_classes,
                                           cache=os.path.join(save_dir,
                                                              "beam"))
            win = []
            if finalists:
                fin = B3.final_select(finalists, Xtr, ytr, kind, n_classes,
                                      K=max(m0.get("cand", 1), 1),
                                      cache=os.path.join(save_dir, "beam"))
                m0["final"] = fin
                if fin and fin[0]["score"] > 0:
                    win = [tuple(p) for p in fin[0]["pairs"]]
            if win:
                card = B3.deploy_with_set(card, admitted, Xtr, ytr, kind,
                                          n_classes, win)
            m0["win"] = [list(w) for w in win]
            m0["admitted"] = bool(win)
            mech["beam3"] = m0

        # ---- IsoLink + calibration specs (canonical block) ---------------
        iso_meta = dict(admitted=False, gain=0.0)
        iso = None
        if kind == "regression":
            eta_tr = np.asarray(card.score(Xtr), float)
            sigma2 = float(np.mean((ytr - eta_tr) ** 2))
            iso, adm, gain = W.iso_deploy(*C1._iso_oof(Xtr, ytr), sigma2)
            iso_meta = dict(admitted=bool(adm), gain=round(float(gain), 2))
            card.isolink = iso
        Tcal, spec14 = W.fit_calib_spec(admitted, kind, n_classes, Xtr, ytr,
                                        oof_merge=cfg.oof_merge)

        budget = cfg.max_wallet_bytes or cfg.wallet_bytes
        payload, info = C.emit_v31(card, Xtr)
        v31 = info["bytes"]
        if budget is not None and v31 > budget:
            card, payload, info, _ = W.rd_squeeze(card, Xtr, budget)
            v31 = info["bytes"]

        dec_card = C.decode_card_from_bytes(payload)
        # canonical certify semantics: metric_fit = full-precision fit card;
        # metric_dec = metric_mem = DECODED card (deploy numbers);
        # maxdiff_mem_dec = decode-determinism (independent re-decode, must be 0);
        # quant_gap = |fit - decoded| (quantisation visibility, not gated).
        met_fit = W.apply_metric(kind, yte, C.model_score(card, Xte), Tcal, spec14)
        met_dec = W.apply_metric(kind, yte, C.model_score(dec_card, Xte), Tcal, spec14)
        met_mem = met_dec
        s_fit = np.asarray(C.model_score(card, Xte), float)
        s_dec = np.asarray(C.model_score(dec_card, Xte), float)
        s_dec2 = np.asarray(C.model_score(C.decode_card_from_bytes(payload), Xte),
                            float)
        maxdiff = float(np.max(np.abs(s_dec - s_dec2))) if len(s_dec) else 0.0
        quant_gap = float(np.max(np.abs(s_fit - s_dec))) if len(s_fit) else 0.0
    finally:
        _reset_cost()
        if not keep_work:
            shutil.rmtree(save_dir, ignore_errors=True)

    return FoldResult(panel=panel, fold=fold, kind=kind, card=card,
                      dec_card=dec_card, admitted=admitted, Tcal=Tcal,
                      spec14=spec14, payload=payload, info=info,
                      metric_fit=met_fit, metric_dec=met_dec, metric_mem=met_mem,
                      maxdiff=maxdiff, moved_train=int(info.get("moved", 0)),
                      v31=v31, mech=mech, quant_gap=quant_gap)


def fit_deploy(data: dict, preset: str = "v8", overrides: Optional[dict] = None,
               work_dir: Optional[str] = None,
               honest_stride: int = 5) -> DeployResult:
    """Production fit: atoms admitted on an honest stride split, final card
    refit on ALL rows through the deploy refit factory (calib_factory).

    The honest split selects rows ``np.arange(0, n, honest_stride)`` as the
    held-out scoring side; the final card = base + admitted atoms + warm joint
    backfit over the full data (the exact deploy-refit path of the research
    pipeline).  Golden metric anchors are defined on the canonical *folds*
    (see ``fit_fold``); ``certify()`` reports deploy metrics.
    """
    cfg = get_preset(preset).with_overrides(**(overrides or {}))
    kind = data["kind"]
    X, y = data["X"], data["y"]
    n = len(y)
    n_classes = int(np.max(y)) + 1 if kind == "multiclass" else 0
    te_idx = np.arange(0, n, honest_stride)
    tr_idx = np.setdiff1d(np.arange(n), te_idx)
    panel = data.get("panel_name", "custom")

    work = work_dir or os.path.join("cardboost_work", preset + "_deploy")
    save_dir = os.path.join(work, "cards", "deploy")
    os.makedirs(save_dir, exist_ok=True)

    apply_config(panel, cfg)
    if kind == "regression":
        sig2 = C1.sigma2_oof(X[tr_idx], y[tr_idx])
        U.R84_COST["scale"] = 1.0 / sig2
        U.R84_COST["pair_exempt"] = True
    else:
        U.R84_COST["scale"] = 1.0
        U.R84_COST["pair_exempt"] = False

    out_jsonl = os.path.abspath(os.path.join(work, "telemetry.jsonl"))
    W.PANELS[panel] = (data, [(tr_idx, te_idx)])
    kw = _run_kwargs(panel, cfg)
    try:
        T.run_fold77(panel, 0, tr_idx, te_idx, data, out_jsonl,
                     save_dir=save_dir, **kw)
        d = pickle.load(open(os.path.join(save_dir, f"{panel}_f0.pkl"), "rb"))
        admitted = list(d.get("admitted", []))
        # final deploy refit over ALL rows with the admitted atom set
        fac = W.calib_factory(admitted, n_classes, oof_merge=cfg.oof_merge,
                              seed=cfg.random_state)
        card = fac(X, y, kind)
        # ---- optional post-fit experimental atoms (presets) --------------
        mech: Dict[str, Any] = {}
        if cfg.include_qmiss:
            import r87_qmiss as QM
            at, m0 = QM.build_atom(card, X, y, kind, n_classes)
            if at is None:
                mech.update(m0)
            else:
                off_arr = np.asarray(card.score(X), float)
                at.refit(X, y, off_arr, kind, kappa_mult=1.0)
                fac2 = W.calib_factory(admitted, n_classes, oof_merge=False)
                eta_oof = L.toof_eta(fac2, X, y, kind, folds=5,
                                     seed=cfg.random_state)
                z1 = eta_oof + at.contrib(X)
                dL = float(np.sum(y * (z1 - eta_oof)
                                  - (np.logaddexp(0.0, z1)
                                     - np.logaddexp(0.0, eta_oof))))
                barrier = 0.5 * np.log(max(len(y), 2)) + np.log(2.0)
                admit = bool(dL > barrier)
                mech.update(m0, gain=round(dL, 3), barrier=round(barrier, 3),
                            admitted=admit)
                if admit:
                    card.atoms.append(at)
        # ---- optional post-fit layers (presets, canary protocols) --------
        if cfg.include_safe_tier:
            import r87_safetier as ST
            tier_atoms, m0 = ST.build_safe_tier(card, X, y, kind, n_classes,
                                                cache=os.path.join(save_dir,
                                                                   "tier"))
            card.atoms.extend(tier_atoms)
            mech["safe_tier"] = m0
        if cfg.ledger_mode == "beam3":
            import r87_beam3 as B3
            finalists, m0 = B3.beam_search(card, X, y, kind, n_classes,
                                           cache=os.path.join(save_dir,
                                                              "beam"))
            win = []
            if finalists:
                fin = B3.final_select(finalists, X, y, kind, n_classes,
                                      K=max(m0.get("cand", 1), 1),
                                      cache=os.path.join(save_dir, "beam"))
                m0["final"] = fin
                if fin and fin[0]["score"] > 0:
                    win = [tuple(p) for p in fin[0]["pairs"]]
            if win:
                card = B3.deploy_with_set(card, admitted, X, y, kind,
                                          n_classes, win)
            m0["win"] = [list(w) for w in win]
            m0["admitted"] = bool(win)
            mech["beam3"] = m0
        if kind == "regression":
            eta_tr = np.asarray(card.score(X), float)
            sigma2 = float(np.mean((y - eta_tr) ** 2))
            iso, _adm, _g = W.iso_deploy(*C1._iso_oof(X, y), sigma2)
            card.isolink = iso
        Tcal, spec14 = W.fit_calib_spec(admitted, kind, n_classes, X, y,
                                        oof_merge=cfg.oof_merge)
        budget = cfg.max_wallet_bytes or cfg.wallet_bytes
        payload, info = C.emit_v31(card, X)
        v31 = info["bytes"]
        if budget is not None and v31 > budget:
            card, payload, info, _ = W.rd_squeeze(card, X, budget)
            v31 = info["bytes"]
        dec_card = C.decode_card_from_bytes(payload)
        s_fit = np.asarray(C.model_score(card, X), float)
        s_dec = np.asarray(C.model_score(dec_card, X), float)
        s_dec2 = np.asarray(C.model_score(C.decode_card_from_bytes(payload), X),
                            float)
        maxdiff = float(np.max(np.abs(s_dec - s_dec2))) if len(s_dec) else 0.0
        quant_gap = float(np.max(np.abs(s_fit - s_dec))) if len(s_fit) else 0.0
    finally:
        _reset_cost()
        shutil.rmtree(save_dir, ignore_errors=True)

    return DeployResult(card=card, dec_card=dec_card, admitted=admitted,
                        Tcal=Tcal, spec14=spec14, payload=payload, info=info,
                        v31=v31, maxdiff=maxdiff, quant_gap=quant_gap,
                        mech=mech,
                        moved_train=int(info.get("moved", 0)),
                        kind=kind, n_classes=n_classes)


def certify_payload(payload: bytes, X, y, kind: str, Tcal, spec14) -> dict:
    """Fresh-decode certification: decode(encode) vs in-memory metrics."""
    dec = C.decode_card_from_bytes(payload)
    met = W.apply_metric(kind, y, C.model_score(dec, X), Tcal, spec14)
    return dict(metrics=met)
