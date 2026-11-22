"""Named presets of the CardBoost constructor.

Design principle (v1.0): **"Zero-Tuning Champion by Default, Full Arsenal on
Demand"**.  The default preset ``"v8"`` is the frozen champion ``card_v8_base``
of round R87; every other preset is a *verified milestone* of the research
campaign and maps to concrete engine knobs of the vendored R71--R87 engine.

Numeric anchors for each preset are declared in ``golden/canonical_r87.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


# Atom-family spellings understood by the engine star pools ---------------
FAMILIES = ("g4", "p1", "m2", "p1_2a", "safe_grid", "msum", "q_miss")

_STAR = {
    "v8": ("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
    "v7_h3": ("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
    "iid_max": ("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
    "ood_fortress": ("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
    "sensor_rare": ("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
    "scorecard_1d": ("g4a", "g4b", "g4c", "g4d"),
    "unregularized_32": ("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
}


@dataclass(frozen=True)
class PresetConfig:
    """Resolved constructor configuration for one preset."""

    name: str
    # -- engine knobs (bit-exact canonical values in parentheses) ----------
    max_bins: int = 32                       # (m=32)
    lr: object = "formula"                   # (lr="formula" -> min(1, sqrt(10/p)))
    km: object = "formula"                   # (km="formula")
    warm_passes: int = 6                     # (warm_passes=6)
    solver: str = "gs"                       # (solver="gs")
    dead_col_law: str = "phase_v32"          # "phase_v32" -> p_phase_law=True (R109);
                                             # "h3"        -> p_phase_law=False (H3 control)
    p_phase_law: bool = True                 # R109 deterministic phase thresholds (champion)
    compress_ucuts: bool = True              # R109 Formula 2: lossless active-cut packing
    currency: str = "h3"                     # (R84_COST scale=1/sigma2_oof on cal, pair_exempt)
    nom_perm: Optional[bool] = None          # None -> engine rule: not california
    enable_pairs: bool = True
    star_types: Tuple[str, ...] = _STAR["v8"]
    pool_mode: Optional[str] = None          # "active" on non-cal panels (canonical)
    screen_mode: Optional[str] = None        # "s12_pure" cal / "legacy" others (canonical)
    frag_mode: str = "w1"
    K_pairs: Optional[int] = None            # 20 cal / 0 others (canonical)
    K_refresh: int = 120
    max_admit: int = 40
    ledger_mode: str = "greedy"              # "beam3" (R114) available experimentally
    edge_freeze: bool = False                # R113 warm6 edge guard
    hmin_frac: float = 0.0                   # R97 5% mass threshold ("ood_fortress")
    oof_merge: bool = False                  # R82
    calibration: str = "isolink_oof"         # regressor; classifier: toof/14b auto
    enable_log_selector: bool = False        # R106 {lambda=1, ln y}
    include_qmiss: bool = False              # R112 QMissAtom (post-fit)
    include_safe_tier: bool = False          # R113 SafeGridAtom (post-fit, experimental)
    include_msum: bool = False               # R104/R105 MSumAtom in the star pool
    wallet_bytes: int = 4096                 # hard container contract
    potts_objective: str = "quadratic"       # "quadratic" | "exact_dev" | "qfix_dev"
    max_wallet_bytes: Optional[int] = 4096   # public alias -> wallet_bytes
    verify_bit_exact: bool = True
    deterministic_ulp: bool = True
    n_jobs: int = 1
    random_state: int = 109
    extra: dict = field(default_factory=dict)

    def with_overrides(self, **kw) -> "PresetConfig":
        clean = {k: v for k, v in kw.items() if v is not None or k in kw}
        return replace(self, **clean)


_PRESETS = {
    # Frozen champion of R87: H3 + phase_v32 + #5 nom_perm + Opus merge + IsoLink-OOF.
    # `card_v8_base` = the R109 PHASE arm (decision 2026-11-20): p_phase_law=True +
    # COMPRESS_UCUTS (Formula 2 live-cut packing).  Golden target: "folds" in
    # golden/canonical_r87.json (2548 B-family wallets, anneal 14b ~0.02605).
    "v8": PresetConfig(name="v8"),
    # R84--R85 canonical harness = the H3 control arm (r87_4 `--arm off` /
    # r87_1 `--arm off`): p_phase_law=False, COMPRESS_UCUTS=False.  Golden
    # target: "h3_folds" (3591 B-family wallets, anneal 14b ~0.02814).
    "v7_h3": PresetConfig(name="v7_h3", dead_col_law="h3",
                          p_phase_law=False, compress_ucuts=False),
    # R98 nested S_jx32 grids + R113 safe tier + beam-3, wallet up to 16384 B.
    # (Both budget fields must carry 16384: the fit budget resolves as
    # `max_wallet_bytes or wallet_bytes`, and the fat wallet IS this
    # preset's identity - it beat California by spending bytes.)
    "iid_max": PresetConfig(
        name="iid_max", include_safe_tier=True, ledger_mode="beam3",
        wallet_bytes=16384, max_wallet_bytes=16384,
        star_types=("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
        extra={"nested_grid": True, "p1_bins_k": 32}),
    # R97 H1H2 5% mass threshold + R113 edge freeze + Dykstra centering.
    "ood_fortress": PresetConfig(
        name="ood_fortress", hmin_frac=0.05, edge_freeze=True,
        include_safe_tier=True),
    # R107/R111 exact-deviance objective + R112 q_miss + R82 oof_merge.
    "sensor_rare": PresetConfig(
        name="sensor_rare", potts_objective="qfix_dev", include_qmiss=True,
        oof_merge=True),
    # R71/R82 1D-only card (bank-style scorecard): 400--900 B deployable.
    "scorecard_1d": PresetConfig(
        name="scorecard_1d", enable_pairs=False, oof_merge=True,
        star_types=("g4a", "g4b", "g4c", "g4d"), wallet_bytes=4096),
    # R71/R73 card_kappa0: 32 quantile shelves without l0 merging (kappa = 0).
    # Wallet budget 16384 B: the unregularised card's measured floor is
    # 12-16 KB (R73/R75: kappa0 = 12837 B); squeezing it to 4096 would
    # re-regularise exactly what this preset exists to preserve.
    "unregularized_32": PresetConfig(
        name="unregularized_32", extra={"kappa_zero": True},
        wallet_bytes=16384, max_wallet_bytes=16384),
}

PRESET_NAMES = tuple(_PRESETS)


def get_preset(name: str) -> PresetConfig:
    if name not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(PRESET_NAMES)}")
    return _PRESETS[name]
