"""API smoke tests: sklearn contract, presets, binning semantics, codecs."""
import numpy as np
import pytest
from sklearn.base import clone

import cardboost
from cardboost import (CardBoostClassifier, CardBoostRegressor,
                       PRESET_NAMES, get_preset)
from cardboost.core.binning import round_scores, stable_desc_order


def test_version():
    assert cardboost.__version__ == "1.0.0"


def test_presets_registered():
    assert set(PRESET_NAMES) == {"v8", "v7_h3", "iid_max", "ood_fortress",
                                 "sensor_rare", "scorecard_1d",
                                 "unregularized_32"}
    for name in PRESET_NAMES:
        cfg = get_preset(name)
        assert cfg.name == name


def test_v8_frozen_defaults():
    cfg = get_preset("v8")
    assert cfg.max_bins == 32
    assert cfg.lr == "formula" and cfg.km == "formula"
    assert cfg.enable_pairs is True
    assert cfg.calibration == "isolink_oof"
    assert cfg.wallet_bytes == 4096
    assert cfg.deterministic_ulp is True
    assert cfg.star_types == ("g4a", "g4b", "g4c", "g4d", "p1", "m2")


def test_sklearn_clone_contract():
    m = CardBoostRegressor(preset="v8", random_state=5)
    m2 = clone(m)
    assert m2.get_params()["random_state"] == 5
    assert m2.get_params()["preset"] == "v8"
    c = clone(CardBoostClassifier())
    assert c.get_params()["preset"] == "v8"


def test_round_scores_ulp_determinism():
    x = np.array([0.1 + 0.2, 0.3, np.nan, -0.0])
    r = round_scores(x, digits=12)
    assert r[0] == round_scores(np.array([0.1 + 0.2]), digits=12)[0]
    assert np.isnan(r[2])


def test_stable_desc_order_ties():
    gains = np.array([1.0, 1.0, 2.0, 0.5])
    order = stable_desc_order(gains)
    assert order[0] == 2
    assert order[-1] == 3
    # equal gains keep their original relative order (lexicographic tie-break)
    assert list(order[1:3]) == [0, 1]


def test_bin_idx_semantics_match_reference():
    from cardboost.codec._sem import _bin_sem
    import r76_container31 as C
    import r71_universal as _L
    cuts = np.array([0.0, 1.0, 2.5])
    xs = np.array([-10.0, 0.0, 0.5, 1.0, 2.5, 3.0, np.nan])
    got = _L.bin_idx(xs, cuts)
    exp = [_bin_sem(float(x), [0.0, 1.0, 2.5]) for x in xs]
    assert list(got) == exp
    # right-open routing: x == cut goes to the higher bin
    assert _bin_sem(0.0, [0.0, 1.0, 2.5]) == 1
    assert _bin_sem(float("nan"), [0.0, 1.0, 2.5]) == 4


def test_estimator_constructs_and_fails_cleanly_before_fit():
    m = CardBoostRegressor()
    with pytest.raises(Exception):
        m.predict(np.zeros((2, 3)))
