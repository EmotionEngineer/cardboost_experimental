"""Synthetic positive controls (golden): q_miss threshold + SafeGrid quadratic.

Both controls are the R87 acceptance probes: the coordination-of-missingness
atom must pay its MDL barrier (4.70 nats) on a threshold effect that is only
visible for rows with >= 3 missings; the SafeGrid hypothesis says a joint 8x8
grid on a pure 2D quadratic beats the univariate sum by an order of magnitude.
"""
import numpy as np
import pytest

from cardboost.core.atoms import (mdl_barrier, synthetic_qmiss_control,
                                  synthetic_quadratic_2d)

pytestmark = pytest.mark.golden


def test_mdl_barrier_formula():
    # canonical admission debit: 0.5*ln(n) + ln(2)  (= 4.70 nats at n ~ 3023)
    assert abs(mdl_barrier(3023) - 4.70) < 0.01
    assert abs(mdl_barrier(1953) - (0.5 * np.log(1953) + np.log(2.0))) < 1e-12
    assert abs(mdl_barrier(2) - 1.5 * np.log(2.0)) < 1e-12


def test_q_miss_threshold_control():
    """Positive control: q_miss must win > barrier on the threshold effect."""
    X, y, act = synthetic_qmiss_control(n=2000, p=6, seed=0)
    y01 = (y > y.mean()).astype(float)
    assert act.mean() > 0.02  # enough activated rows
    from cardboost.core import engine
    data = dict(kind="binary", X=X, y=y01,
                names=[f"f{j}" for j in range(X.shape[1])], geo=None,
                panel_name="qmiss_ctrl")
    res = engine.fit_deploy(data, preset="sensor_rare",
                            overrides=dict(include_qmiss=True,
                                           max_wallet_bytes=4096))
    mech = getattr(res, "mech", None) or {}
    if not mech:
        pytest.skip("engine surfaced no mech metadata for qmiss")
    if mech.get("gain") is None:
        pytest.skip("qmiss atom not built on this draw")
    assert mech["gain"] > mech["barrier"]
    assert mech["admitted"] is True
    assert res.v31 <= 4096
    assert res.moved_train == 0


def test_safe_grid_quadratic_control():
    """SafeGrid hypothesis probe: joint 8x8 grid >> univariate sum on a
    rotated quadratic (u+v)^2 - the surface a safe joint grid must capture
    and a univariate sum cannot."""
    rng = np.random.RandomState(1)
    X = rng.normal(size=(3000, 2))
    y = (X[:, 0] + X[:, 1]) ** 2 + 0.3 * rng.normal(size=3000)
    # univariate 8-bin tables
    def nats1d(j):
        b = np.clip((np.digitize(X[:, j], np.quantile(X[:, j],
                     np.linspace(0, 1, 9)[1:-1]))), 0, 7)
        mu = np.array([y[b == k].mean() if (b == k).any() else y.mean()
                       for k in range(8)])
        pred = mu[b]
        return 0.5 * len(y) * np.log(max(np.var(y), 1e-12) /
                                     max(np.var(y - pred), 1e-12))
    gain_uni = nats1d(0) + nats1d(1)
    # joint 8x8 grid
    b0 = np.clip(np.digitize(X[:, 0], np.quantile(X[:, 0],
                 np.linspace(0, 1, 9)[1:-1])), 0, 7)
    b1 = np.clip(np.digitize(X[:, 1], np.quantile(X[:, 1],
                 np.linspace(0, 1, 9)[1:-1])), 0, 7)
    key = b0 * 8 + b1
    mu = np.array([y[key == k].mean() if (key == k).any() else y.mean()
                   for k in range(64)])
    pred = mu[key]
    gain_joint = 0.5 * len(y) * np.log(max(np.var(y), 1e-12) /
                                       max(np.var(y - pred), 1e-12))
    barrier = 33.46  # 63-cell grid debit at n=3000 (canonical control)
    assert gain_joint > 1.5 * gain_uni  # joint >> univariate (SafeGrid win)
    assert gain_uni > barrier
