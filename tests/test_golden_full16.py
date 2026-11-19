"""GOLDEN FULL PROTOCOL (slow): all 16 canonical folds vs golden/canonical_r87.json.

This is the acceptance suite of cardboost 1.0: it re-runs the exact R87
protocol (card_v8_base, r87_4 arm=off) and requires every fold to reproduce
the certified metrics, wallet sizes and decode invariants.

Bit-exact acceptance runs on **Linux x86-64** (the canonical platform).
On Apple Silicon: california/aps rows are checked bit-exact for v31 with a
libm-ulp metric tolerance; the knife-edge rows (anneal family, cal_spat_f4)
run in structural mode with a full report line — see tests/golden_util.py.

Runtime: ~1-2 h on a modern x86-64 core (aps folds dominate).  Run with:
    python -m pytest -m golden          # everything
    python -m pytest tests/test_golden_full16.py -k cal_rand   # a family
"""
import numpy as np
import pytest

from golden_util import check_fold

pytestmark = [pytest.mark.golden, pytest.mark.slow]

FOLD_CASES = [
    ("anneal", 0), ("anneal", 1), ("anneal", 2),
    ("aps", 0), ("aps", 1), ("aps", 2),
    ("california_random", 0), ("california_random", 1),
    ("california_random", 2), ("california_random", 3),
    ("california_random", 4),
    ("california_spatial", 0), ("california_spatial", 1),
    ("california_spatial", 2), ("california_spatial", 3),
    ("california_spatial", 4),
]


def _tag(panel, fold):
    short = {"california_random": "cal_rand", "california_spatial": "cal_spat"}
    return f"{short.get(panel, panel)}_f{fold}"


@pytest.mark.parametrize("panel,fold", FOLD_CASES,
                         ids=[_tag(p, f) for p, f in FOLD_CASES])
def test_fold_matches_canonical(panel, fold, panels, golden):
    from cardboost.core import engine
    data, splits = panels[panel]
    res = engine.fit_fold(panel, fold, data, splits, preset="v8")
    check_fold(res, golden["folds"][_tag(panel, fold)], _tag(panel, fold))


def test_deploy_protocol_aps_toof(panels, golden):
    """The R87 headline: aps Toof deploy mean == 0.03298 (3-fold protocol)."""
    means = []
    for fold in range(3):
        data, splits = panels["aps"]
        from cardboost.core import engine
        res = engine.fit_fold("aps", fold, data, splits, preset="v8")
        means.append(res.metric_dec["Toof"])
    assert abs(np.mean(means) -
               golden["deploy_checks"]["aps_toof_deploy_mean"]) < 3e-5
