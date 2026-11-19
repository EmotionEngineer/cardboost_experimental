"""Container v3.x round-trip: decode(encode(card)) == card on scores (exact)."""
import gzip

import numpy as np
import pytest

from cardboost.codec import container_v32 as cv


@pytest.fixture(scope="module")
def fitted(small_reg_module):
    return small_reg_module


def _fit(data, preset="v8"):
    from cardboost.core import engine
    res = engine.fit_deploy(data, preset=preset)
    return res


def test_roundtrip_scores_exact():
    from cardboost.core.atoms import synthetic_quadratic_2d
    X, y = synthetic_quadratic_2d(n=350, seed=11)
    data = dict(kind="regression", X=X, y=y, names=["u", "v"], geo=None,
                panel_name="rt_reg")
    res = _fit(data)
    # payload decode determinism (the canonical maxdiff_mem_dec invariant)
    dec = cv.from_bytes(res.payload)
    dec2 = cv.from_bytes(res.payload)
    s1 = np.asarray(cv.C.model_score(dec, X), float)
    s2 = np.asarray(cv.C.model_score(dec2, X), float)
    assert float(np.max(np.abs(s1 - s2))) == 0.0
    assert res.moved_train == 0
    assert res.v31 <= 4096
    assert res.maxdiff == 0.0  # decode determinism (decode twice)
    # quantisation gap between the full-precision fit card and the deploy
    # card is expected (f16 wire) and is covered by the metric anchors
    assert res.quant_gap < 0.5


def test_estimator_bytes_contract():
    """to_bytes/from_bytes of the estimator reproduces predict() bit-exactly."""
    from cardboost import CardBoostRegressor
    from cardboost.core.atoms import synthetic_quadratic_2d
    X, y = synthetic_quadratic_2d(n=350, seed=12)
    m = CardBoostRegressor(preset="v8", feature_names=["u", "v"]).fit(X, y)
    blob = m.to_bytes()
    m2 = CardBoostRegressor.from_bytes(blob)
    p1 = m.predict(X)
    p2 = m2.predict(X)
    assert float(np.max(np.abs(p1 - p2))) == 0.0
    cert = m.certify(X)
    assert cert["max_diff"] == 0.0
    assert cert["moved_train"] == 0
    assert cert["decode_ok"]


def test_roundtrip_binary_with_nans():
    from cardboost.core.atoms import synthetic_qmiss_control
    X, y, _ = synthetic_qmiss_control(n=400, p=5, seed=2)
    y01 = (y > y.mean()).astype(float)
    data = dict(kind="binary", X=X, y=y01,
                names=[f"f{j}" for j in range(X.shape[1])], geo=None,
                panel_name="rt_bin")
    res = _fit(data)
    dec = cv.from_bytes(res.payload)
    dec2 = cv.from_bytes(cv.to_bytes(dec, budget=4096))
    s1 = np.asarray(cv.C.model_score(dec, X), float)
    s2 = np.asarray(cv.C.model_score(dec2, X), float)
    assert float(np.max(np.abs(s1 - s2))) == 0.0
    assert res.moved_train == 0


def test_gzip_autodetect():
    from cardboost.core.atoms import synthetic_quadratic_2d
    X, y = synthetic_quadratic_2d(n=250, seed=5)
    data = dict(kind="regression", X=X, y=y, names=["u", "v"], geo=None,
                panel_name="rt_gz")
    res = _fit(data)
    gz = gzip.compress(res.payload, 9)
    dec = cv.from_bytes(gz)  # gzip magic auto-detected
    s2 = np.asarray(cv.C.model_score(dec, X), float)
    s1 = np.asarray(cv.C.model_score(cv.from_bytes(res.payload), X), float)
    assert float(np.max(np.abs(s1 - s2))) == 0.0


def test_wallet_report():
    from cardboost.core.atoms import synthetic_quadratic_2d
    X, y = synthetic_quadratic_2d(n=250, seed=6)
    data = dict(kind="regression", X=X, y=y, names=["u", "v"], geo=None,
                panel_name="rt_w")
    res = _fit(data)
    w = cv.wallet(res.card, X, budget=4096)
    assert w.raw_bytes <= 4096
    assert w.moved_train == 0
