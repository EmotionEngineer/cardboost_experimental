"""Shared golden-suite helpers: platform tiers and metric tolerances.

Canonical acceptance platform: **Linux x86-64** - all 16 folds bit-exact
(v31, metrics, decode invariants).  Every other platform runs the same suite
in PLATFORM MODE, which is automatically tiered per row:

* row reproduces the canonical wallet size (v31) -> the card is bit-identical
  and metrics must agree within TIGHT_TOL (libm ulp in metric *evaluation*,
  4e-11 observed on the Apple Silicon canary);
* row's v31 flipped (f16 quantisation of cuts is knife-edge sensitive to
  libm/BLAS last-ulp deltas in the warm passes) -> structural invariants
  (moved_train == 0, decode determinism == 0, wallet <= 4096) plus quality
  within LOOSE bands, and a full report line - never a silent pass.

One summary line is printed per row in platform mode, so the log always
carries the full platform-vs-canonical matrix.

Environment knobs:
* CARDBOOST_GOLDEN_TOL    - tight metric tolerance override (default 1e-6)
* CARDBOOST_GOLDEN_STRICT - force the canonical bit-exact path on any platform
"""
from __future__ import annotations

import os
import platform
import sys

IS_CANONICAL = (sys.platform == "linux" and
                platform.machine().lower() in ("x86_64", "amd64"))
STRICT = os.environ.get("CARDBOOST_GOLDEN_STRICT", "") == "1"

TIGHT_TOL = float(os.environ.get("CARDBOOST_GOLDEN_TOL", "1e-6"))

# loose quality bands for rows whose f16 wire quantisation flipped
LOOSE_R2 = 0.15
LOOSE_LOGLoss = 0.05


def _fmt(d):
    return "{" + ", ".join(f"{k}: {v:.10g}" for k, v in d.items()) + "}"


def _delta(a, b):
    return "{" + ", ".join(f"{k}: {a.get(k, float('nan')) - b[k]:+.2e}"
                           for k in b) + "}"


def check_fold(res, entry, tag):
    """Compare one FoldResult against its golden entry, platform-aware."""
    # schema normalisation: rows carry either metric_fit/metric_dec or
    # metrics/dec (both spellings occur across golden tables)
    m_fit = entry.get("metric_fit", entry.get("metrics"))
    m_dec = entry.get("metric_dec", entry.get("dec"))
    assert m_fit and m_dec, f"{tag}: golden entry lacks metric dicts"
    if IS_CANONICAL or STRICT:
        tol = 0.0 if entry.get("exact") else 5e-7
        assert res.v31 == entry["v31"], f"{tag}: v31 {res.v31} != {entry['v31']}"
        assert res.moved_train == 0, f"{tag}: moved_train != 0"
        assert res.maxdiff == 0.0, f"{tag}: decode determinism broken"
        for key, val in m_fit.items():
            got = res.metric_fit[key]
            assert abs(got - val) <= max(tol, abs(val) * 1e-12), \
                f"{tag} metric_fit[{key}]: {got!r} != {val!r} (tol {tol})"
        for key, val in m_dec.items():
            got = res.metric_dec[key]
            assert abs(got - val) <= max(tol, abs(val) * 1e-12), \
                f"{tag} metric_dec[{key}]: {got!r} != {val!r} (tol {tol})"
        return

    # ---------------- platform mode: automatic tiers + report -------------
    assert res.moved_train == 0, f"{tag}: moved_train != 0"
    assert res.maxdiff == 0.0, f"{tag}: decode determinism broken"
    assert res.v31 <= 4096, f"{tag}: wallet {res.v31} > 4096"

    v31_ok = (res.v31 == entry["v31"])
    if v31_ok:
        for key, val in m_fit.items():
            got = res.metric_fit[key]
            assert abs(got - val) <= max(TIGHT_TOL, abs(val) * 1e-12), \
                (f"{tag} metric_fit[{key}]: {got!r} != {val!r} "
                 f"(bit-identical card but metric drift exceeds TIGHT_TOL "
                 f"{TIGHT_TOL})")
        for key, val in m_dec.items():
            got = res.metric_dec[key]
            assert abs(got - val) <= max(TIGHT_TOL, abs(val) * 1e-12), \
                (f"{tag} metric_dec[{key}]: {got!r} != {val!r} "
                 f"(bit-identical card but metric drift exceeds TIGHT_TOL "
                 f"{TIGHT_TOL})")
        grade = "A (bit-identical card, ulp metrics)"
    else:
        for key, val in m_dec.items():
            got = res.metric_dec[key]
            band = LOOSE_R2 if key.upper().startswith("R") else LOOSE_LOGLoss
            assert abs(got - val) <= band, (
                f"{tag} metric_dec[{key}]: platform value {got!r} vs canonical "
                f"{val!r} exceeds even the LOOSE band {band} - real breakage?")
        grade = "B (f16 quantisation flip, structural+quality band)"

    print(f"\n[golden:{tag}] platform-grade {grade}\n"
          f"    v31        {res.v31}  (canonical {entry['v31']})"
          f"{'' if v31_ok else '   <-- FLIP'}\n"
          f"    metric_fit {_fmt(res.metric_fit)}\n"
          f"    metric_dec {_fmt(res.metric_dec)}\n"
          f"    delta_fit  {_delta(res.metric_fit, m_fit)}\n"
          f"    delta_dec  {_delta(res.metric_dec, m_dec)}")
