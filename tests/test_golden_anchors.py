"""GOLDEN ANCHOR (fast): cal_rand f0 + anneal f0 vs the canonical R87 table.

    cal_rand_f0:  fit R2 == 0.8004754327779836, dec R2 == 0.7991800160444765,
                  v31 == 3591, moved_train == 0
    anneal_f0:    fit Toof == 0.015556201491460715, dec Toof == 0.015608297304323386,
                  v31 == 1491, moved_train == 0

Bit-exact on Linux x86-64 (the canonical platform).  On Apple Silicon the
anneal family runs in structural mode (knife-edge f16 quantisation) and
metric evaluation may drift at the libm ulp — see tests/golden_util.py and
README "Platform sensitivity".
"""
import pytest

from golden_util import check_fold

pytestmark = [pytest.mark.golden, pytest.mark.slow]


def test_cal_rand_f0_anchor(panels, golden):
    from cardboost.core import engine
    data, splits = panels["california_random"]
    res = engine.fit_fold("california_random", 0, data, splits, preset="v8")
    check_fold(res, golden["folds"]["cal_rand_f0"], "cal_rand_f0")


def test_anneal_f0_anchor(panels, golden):
    from cardboost.core import engine
    data, splits = panels["anneal"]
    res = engine.fit_fold("anneal", 0, data, splits, preset="v8")
    check_fold(res, golden["folds"]["anneal_f0"], "anneal_f0")
