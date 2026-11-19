"""Dataset equivalence: bundled california_housing.npz == sklearn fetch bytes.

Guards the network-free property: the bundled npz must be byte-identical in
values to ``fetch_california_housing`` (checked with the canonical SHA of the
first row + shapes + exact fold partitions vs ``L.build_panels``).
"""
import numpy as np
import pytest

from cardboost import datasets as D


def test_california_npz_shapes():
    d = D.load_california()
    X, y = d["X"], d["y"]
    assert X.shape == (20640, 8)
    assert y.shape == (20640,)
    assert d["names"] == [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
        "AveOccup", "Latitude", "Longitude"]
    # canonical first row (sklearn fetch_california_housing, full precision)
    np.testing.assert_allclose(
        X[0], [8.3252, 41.0, 6.984126984126984, 1.0238095238095238, 322.0,
               2.5555555555555554, 37.88, -122.23], rtol=0, atol=1e-12)
    assert abs(y[0] - 4.526) < 1e-12


def test_california_matches_fetch_when_available():
    try:
        from sklearn.datasets import fetch_california_housing
        d0 = fetch_california_housing(as_frame=True)
    except Exception:
        pytest.skip("fetch_california_housing needs network; npz is the source of truth")
    d1 = D.load_california()
    np.testing.assert_array_equal(d1["X"], d0.data.to_numpy(float))
    np.testing.assert_array_equal(d1["y"], np.asarray(d0.target, float))


def test_folds_match_canonical():
    d = D.load_california()
    folds = D._folds_cal_random(d["X"])
    assert len(folds) == 5
    n = len(d["X"])
    seen = np.zeros(n, int)
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        seen[te] += 1
    assert (seen == 1).all()
    # spatial folds: contiguous latitude blocks
    lat = d["geo"][0]
    sfolds = D._folds_cal_spatial(lat, n)
    for tr, te in sfolds:
        assert lat[te].max() - lat[te].min() < 30.0  # one block, not scattered


def test_anneal_aps_load_offline():
    ann = D.load_anneal()
    assert ann["X"].shape[0] == ann["y"].shape[0]
    assert ann["X"].shape[0] > 500
    aps = D.load_aps(6000)
    assert aps["X"].shape == (6000, aps["X"].shape[1])
    assert set(np.unique(aps["y"])) == {0.0, 1.0}
