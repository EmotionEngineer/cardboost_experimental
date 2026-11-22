"""Preset IDENTITY tests: every preset must be the historical model, not a stub.

Motivation (postmortem defect D1): a preset can look right while its knobs
are dead.  These tests assert two layers of identity per preset:

1. WIRING - the preset's unique knobs actually reach the engine (globals flip
   as the historical mechanism expects, and do not leak between presets);
2. FINGERPRINT - the unique mechanism changes measured behaviour in the
   direction the campaign documented (sources: MODEL_COMPENDIUM.md §12,
   INDEX.md round digests, r86/r87 canaries reports).

Historical fingerprints asserted here:
* unregularized_32 (R71/R73 card_kappa0, kappa=0): no l0 merging -> strictly
  more segments and a fatter card than v8 on the same data (measured floor
  12-16 KB, R73/R75).
* sensor_rare (R107/R112/R82): qfix_dev + q_miss attempt metadata present;
  q_miss is silent on data without a threshold-NaN effect (R112: admitted
  0/16 on canonical panels) and loud on the synthetic control (already
  covered by test_synthetic_controls: gain 49.93 > barrier 4.70).
* ood_fortress (R97 H1H2): hmin mass floor engaged (R84_COST["hmin"] > 0).
* iid_max (R98/R113/R114): safe-tier + beam3 layers engage and report their
  admission metadata; budget 16384 B is active (the fat-wallet identity).
* scorecard_1d (R71/R82): 1D-only card - no pair (p1/m2) atoms ever admitted.
* v8 / v7_h3: covered bit-exactly by the golden suites (folds / h3_folds).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.slow]


# ------------------------------------------------------------- wiring tests
def test_apply_config_wiring_and_no_leak():
    """Preset knobs must flip the engine globals - and reset for the rest."""
    from cardboost.core import engine
    from cardboost.presets import get_preset
    import r71_universal as U
    import r76_container31 as C
    import r76_mech as M6
    import r77_mech as R
    import r77_stage as T

    # v8: phase law on, ucuts on, no hmin, P1_BINS=8, km=formula
    engine.apply_config("anneal", get_preset("v8"))
    assert C.COMPRESS_UCUTS is True
    assert T.FITCFG["p_phase_law"] is True
    assert U.R84_COST["hmin"] == 0.0
    assert R.P1_BINS == 8
    assert M6.EXACT_DEV["on"] is False

    # ood_fortress: R97 mass floor engaged
    engine.apply_config("anneal", get_preset("ood_fortress"))
    assert U.R84_COST["hmin"] > 0.0

    # leak check: back to v8 must CLEAR the hmin floor
    engine.apply_config("anneal", get_preset("v8"))
    assert U.R84_COST["hmin"] == 0.0

    # v7_h3: phase law off, ucuts off
    engine.apply_config("anneal", get_preset("v7_h3"))
    assert T.FITCFG["p_phase_law"] is False
    assert C.COMPRESS_UCUTS is False

    # iid_max: R98 S_jx32 grids (P1_BINS=32)
    engine.apply_config("california_random", get_preset("iid_max"))
    assert R.P1_BINS == 32
    engine.apply_config("california_random", get_preset("v8"))
    assert R.P1_BINS == 8

    # sensor_rare: R107 exact-deviance objective on
    engine.apply_config("anneal", get_preset("sensor_rare"))
    assert M6.EXACT_DEV["on"] is True
    engine.apply_config("anneal", get_preset("v8"))
    assert M6.EXACT_DEV["on"] is False

    # unregularized_32: kappa=0 -> kappa_mult 0.0 in the fit kwargs
    engine.apply_config("anneal", get_preset("unregularized_32"))
    assert T.FITCFG["km"] == 0.0
    engine.apply_config("anneal", get_preset("v8"))
    assert T.FITCFG["km"] == "formula"


def test_budget_identity():
    """The fat budgets are part of the models' identities."""
    from cardboost.presets import get_preset
    for name, budget in (("v8", 4096), ("v7_h3", 4096), ("iid_max", 16384),
                         ("unregularized_32", 16384), ("ood_fortress", 4096),
                         ("sensor_rare", 4096), ("scorecard_1d", 4096)):
        cfg = get_preset(name)
        assert (cfg.max_wallet_bytes or cfg.wallet_bytes) == budget, name


# -------------------------------------------------------- fingerprint tests
def _fit(data, preset, overrides=None):
    import tempfile
    from cardboost.core import engine
    work = tempfile.mkdtemp(prefix="cb_ident_")
    return engine.fit_deploy(data, preset=preset, overrides=overrides or {},
                             work_dir=work)


def test_unregularized_32_kappa_zero_fingerprint(small_reg):
    """kappa=0: no l0 merging -> strictly fatter card than v8 (R71 card_k0).

    Vendor semantics (r71_universal): kappa_mult 0.0 == the "unfused arm" of
    card_kappa0 - every quantile shelf survives, so the wire must grow.
    """
    base = _fit(small_reg, "v8")
    k0 = _fit(small_reg, "unregularized_32",
              overrides={"max_wallet_bytes": 1 << 20, "wallet_bytes": 1 << 20})
    assert k0.v31 > base.v31 * 1.3, (
        f"kappa_zero is dead: wallet {k0.v31} vs v8 {base.v31}")


def test_sensor_rare_qmiss_engages(small_bin):
    """R112 q_miss layer must ATTEMPT the coordination atom and report it."""
    res = _fit(small_bin, "sensor_rare")
    assert "gain" in res.mech, "qmiss layer did not run"
    assert "admitted" in res.mech
    # R107 fingerprint: on data without extra signal the card stays in budget
    assert res.v31 <= 4096


def test_scorecard_1d_is_pairless(small_reg):
    """R71/R82 bank scorecard: 1D rays + mains only, never p1/m2 pairs."""
    res = _fit(small_reg, "scorecard_1d")
    for at in res.card.atoms:
        name = type(at).__name__.lower()
        for bad in ("p1", "m2", "pair", "swap", "grid"):
            assert bad not in name, (
                f"scorecard_1d admitted a pair atom: {name}")


def test_ood_fortress_and_iid_max_layers_engage(small_bin):
    """R113 safe tier + R114 beam3 must run and report admission metadata."""
    ood = _fit(small_bin, "ood_fortress")
    assert "safe_tier" in ood.mech, "safe tier layer did not run"

    iid = _fit(small_bin, "iid_max")
    assert "safe_tier" in iid.mech, "iid_max: safe tier layer did not run"
    assert "beam3" in iid.mech, "iid_max: beam3 ledger did not run"
    assert "admitted" in iid.mech["beam3"]
    assert iid.v31 <= 16384
