"""Atom registry and synthetic control generators.

The production atom classes live in the certified vendored mechanisms
(``r77_mech.ProjRayAtom/P1Atom/M2Atom/MSumAtom``, ``r87_qmiss.QMissAtom``,
``r87_safetier.SafeGrid``); this module documents their wire tags and provides
the two *synthetic controls* used by the golden suite:

* **q_miss threshold control** - a threshold effect that only shows up for
  rows with >= 3 missing features.  The coordination-of-missingness atom must
  pay its MDL barrier (4.70 nats) and still win (canonical gain 49.93 nats).
* **safe-grid quadratic control** - a pure 2D quadratic surface, where a joint
  8x8 grid explains 1147.67 nats vs 161.02 for the best univariate sum.

Wire tags (Container v3.1): 1=g4a, 2=g4b, 3=g4c, 4=g4d, 5=p1, 6=m2,
7=p1_swap2a, 8=p1_swap2b, 9=msum, 10=q_miss, 13/14=safe tiers.
"""
from __future__ import annotations

import numpy as np

ATOM_WIRE_TAGS = {
    "g4a": 1, "g4b": 2, "g4c": 3, "g4d": 4, "p1": 5, "m2": 6,
    "p1_2a": 7, "p1_2b": 8, "msum": 9, "q_miss": 10,
    "safe_tier1": 13, "safe_tier2": 14,
}

MDL_BARRIER_PAIR_NAT = 4.70   # 0.5*ln(n) + ln(2) at n = 1953 (cal panel)


def synthetic_qmiss_control(n: int = 3000, p: int = 6, seed: int = 0):
    """Threshold effect visible only for rows with >= 3 missing features.

    Returns (X, y, is_activated): y jumps by +1.5 sigma exactly for rows with
    >= 3 NaNs (threshold effect), independent of the observed feature values.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    mask = rng.uniform(size=(n, p)) < 0.25
    X[mask] = np.nan
    n_missing = mask.sum(axis=1)
    y = rng.normal(size=n)
    y = y + 1.5 * (n_missing >= 3)
    return X, y, n_missing >= 3


def synthetic_quadratic_2d(n: int = 4000, seed: int = 0):
    """Pure 2D quadratic surface (SafeGrid positive control)."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 + 0.3 * rng.normal(size=n))
    return X, y


def mdl_barrier(n: float, lo: float = 0.0) -> float:
    """Admission barrier 0.5*ln(n) + ln(2) nats (R75 debit accounting)."""
    return 0.5 * np.log(max(n, 2.0)) + np.log(2.0)
