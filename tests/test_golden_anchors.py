"""GOLDEN ANCHOR (fast): one cal fold + one anneal fold per mode family.

Mode map (decision 2026-11-20, "v8 = R109 phase"):

* ``preset="v8"``   -> phase arm (p_phase_law=True + COMPRESS_UCUTS):
    cal_rand_f0:  fit R2 == 0.8004754327779836, dec R2 == 0.7991800160444765,
                  v31 == 2548, moved_train == 0
    anneal_f0:    fit Toof == 0.017293655045400768, dec Toof == 0.01727664026372757,
                  v31 == 1297, moved_train == 0
* ``preset="v7_h3"`` -> H3 control arm (p_phase_law=False, m32 portrait):
    cal_rand_f0:  fit R2 == 0.8004754327779836, dec R2 == 0.7991800160444765,
                  v31 == 3591, moved_train == 0
    anneal_f0:    fit Toof == 0.015556201491460715, dec Toof == 0.015608297304323386,
                  v31 == 1491, moved_train == 0

Bit-exact on Linux x86-64 (the canonical platform).  On Apple Silicon the
anneal family runs in structural mode (knife-edge f16 quantisation) and metric
evaluation may drift at the libm ulp -- see tests/golden_util.py and README
"Platform sensitivity".
"""
import pytest

from golden_util import check_fold

pytestmark = [pytest.mark.golden, pytest.mark.slow, pytest.mark.timeout(10800)]


# ---- v8 = R109 phase arm -------------------------------------------------
def test_cal_rand_f0_anchor_v8(panels, golden):
    from cardboost.core import engine
    data, splits = panels["california_random"]
    res = engine.fit_fold("california_random", 0, data, splits, preset="v8")
    check_fold(res, golden["folds"]["cal_rand_f0"], "cal_rand_f0")


def test_anneal_f0_anchor_v8(panels, golden):
    from cardboost.core import engine
    data, splits = panels["anneal"]
    res = engine.fit_fold("anneal", 0, data, splits, preset="v8")
    check_fold(res, golden["folds"]["anneal_f0"], "anneal_f0")


# ---- v7_h3 = H3 control arm ----------------------------------------------
def test_cal_rand_f0_anchor_h3(panels, golden):
    from cardboost.core import engine
    data, splits = panels["california_random"]
    res = engine.fit_fold("california_random", 0, data, splits, preset="v7_h3")
    check_fold(res, golden["h3_folds"]["cal_rand_f0"], "cal_rand_f0")


def test_anneal_f0_anchor_h3(panels, golden):
    from cardboost.core import engine
    data, splits = panels["anneal"]
    res = engine.fit_fold("anneal", 0, data, splits, preset="v7_h3")
    check_fold(res, golden["h3_folds"]["anneal_f0"], "anneal_f0")
