"""Shared fixtures: canonical panels (bundled data, no network), golden table."""
import json
import os
import sys

# deterministic BLAS threading BEFORE numpy loads anywhere (golden bit-exactness)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "golden",
                           "canonical_r87.json")


@pytest.fixture(scope="session")
def golden():
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def panels():
    """Full canonical panels (california/anneal/aps) - slow to build once."""
    from cardboost import datasets
    return datasets.build_panels()   # {name: (data, splits)}


@pytest.fixture(scope="session")
def small_reg():
    """Tiny regression panel for fast functional tests."""
    import numpy as np
    from cardboost.core.atoms import synthetic_quadratic_2d
    X, y = synthetic_quadratic_2d(n=400, seed=7)
    return dict(kind="regression", X=X, y=y,
                names=["u", "v"], geo=None, panel_name="small_reg")


@pytest.fixture(scope="session")
def small_bin():
    """Tiny binary panel with NaNs for fast functional tests."""
    import numpy as np
    from cardboost.core.atoms import synthetic_qmiss_control
    X, y, _ = synthetic_qmiss_control(n=500, p=5, seed=3)
    y01 = (y > y.mean()).astype(float)
    return dict(kind="binary", X=X, y=y01,
                names=[f"f{j}" for j in range(X.shape[1])], geo=None,
                panel_name="small_bin")
