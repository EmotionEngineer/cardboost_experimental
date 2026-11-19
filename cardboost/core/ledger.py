"""Admission ledger helpers: MDL barriers and beam arbitration constants.

Canonical admission rule (R75/R87): an atom enters the wallet iff its OOF
credit pays its MDL debit, i.e. ``gain_nats > 0.5*ln(n) + ln(2)`` for pair
atoms (4.70 nats at n = 1953), with beam arbitration (1 slot / 2 slots /
w1 frag) during refits.
"""
from __future__ import annotations

import numpy as np

from .atoms import mdl_barrier

PAIR_BARRIER_NAT = 4.70


def admission_ok(gain_nats: float, n: float, lo: float = 0.0,
                 cost_nats: float = 0.0) -> bool:
    """Debit-paid admission test used by the ledger."""
    return (float(gain_nats) - float(cost_nats)) > mdl_barrier(n, lo)


def beam_score(gain_nats: float, cost_nats: float) -> float:
    """Sort key for beam arbitration: highest net gain first."""
    return float(gain_nats) - float(cost_nats)


def sigma2_oof_scale(oof_var: float) -> float:
    """R84 credit currency: ÷sigma2_oof (pair-exempt), as used by r87_4."""
    if oof_var is None or not np.isfinite(oof_var) or oof_var <= 0:
        return 1.0
    return float(1.0 / oof_var)
