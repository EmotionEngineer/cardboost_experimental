"""Shared golden-suite helpers: platform tiers and metric tolerances.

Canonical acceptance platform: **Linux x86-64** — all 16 folds bit-exact
(v31, metrics, decode invariants).

Other platforms (notably Apple Silicon Mac Mini M4) reproduce the *cards*
for the california and aps families exactly (v31 / moved_train / decode
determinism bit-exact) while Toof/14b metric *evaluation* drifts at the libm
ulp (4e-11 observed on the M4 canary).  The anneal family (tiny, discrete,
knife-edge warm passes at the ulp level) can flip f16 wire quantisation and
shift v31 by a few bytes — those rows run in **structural mode** off-canonical:
invariants + a loose metric band + a full report line.

Environment knobs:
* CARDBOOST_GOLDEN_TOL    - metric tolerance override (default 0.0 canonical,
                            1e-8 elsewhere)
* CARDBOOST_GOLDEN_STRICT - force bit-exact checks on any platform (debugging)
"""
from __future__ import annotations

import os
import platform
import sys

IS_CANONICAL = (sys.platform == "linux" and
                platform.machine().lower() in ("x86_64", "amd64"))
IS_APPLE_SILICON = (sys.platform == "darwin" and platform.machine() == "arm64")
STRICT = os.environ.get("CARDBOOST_GOLDEN_STRICT", "") == "1"

METRIC_TOL = float(os.environ.get(
    "CARDBOOST_GOLDEN_TOL", "0.0" if IS_CANONICAL else "1e-8"))

# Knife-edge rows: warm passes/fold decisions sit on ulp-level margins; the
# f16 wire quantisation can flip (R87 canary evidence: anneal family drifts
# on Apple Silicon; cal_spat_f4 is a knife-edge fold decision there).
DRIFT_ROWS = {"anneal_f0", "anneal_f1", "anneal_f2", "cal_spat_f4"}
DRIFT_TOL = 5e-3  # absolute metric sanity band used in structural mode


def check_fold(res, entry, tag):
    """Compare one FoldResult against its golden entry, platform-aware."""
    structural = (not IS_CANONICAL and not STRICT and tag in DRIFT_ROWS)
    if structural:
        assert res.moved_train == 0, f"{tag}: moved_train != 0"
        assert res.maxdiff == 0.0, f"{tag}: decode determinism broken"
        assert res.v31 <= 4096, f"{tag}: wallet {res.v31} > 4096"
        for key, val in entry["metric_dec"].items():
            got = res.metric_dec[key]
            assert abs(got - val) <= DRIFT_TOL, (
                f"{tag} metric_dec[{key}]: platform value {got!r} vs canonical "
                f"{val!r} exceeds even the structural band {DRIFT_TOL}")
        print(f"\n[golden:{tag}] PLATFORM-DRIFT STRUCTURAL MODE "
              f"(bit-exact acceptance: Linux x86-64)\n"
              f"    v31        {res.v31}   (canonical {entry['v31']})\n"
              f"    metric_fit {res.metric_fit}\n"
              f"    metric_dec {res.metric_dec}\n"
              f"    invariants OK: moved_train=0, decode determinism=0, "
              f"wallet<=4096")
        return

    tol = max(0.0 if entry.get("exact") else 5e-7, METRIC_TOL)
    assert res.v31 == entry["v31"], (
        f"{tag}: v31 {res.v31} != {entry['v31']}"
        + ("" if IS_CANONICAL else
           " (non-canonical platform: if this is a known f16 quantisation "
           "flip, add the tag to golden_util.DRIFT_ROWS)"))
    assert res.moved_train == 0, f"{tag}: moved_train != 0"
    assert res.maxdiff == 0.0, f"{tag}: decode determinism broken"
    for key, val in entry["metric_fit"].items():
        got = res.metric_fit[key]
        assert abs(got - val) <= max(tol, abs(val) * 1e-12), \
            f"{tag} metric_fit[{key}]: {got!r} != {val!r} (tol {tol})"
    for key, val in entry["metric_dec"].items():
        got = res.metric_dec[key]
        assert abs(got - val) <= max(tol, abs(val) * 1e-12), \
            f"{tag} metric_dec[{key}]: {got!r} != {val!r} (tol {tol})"
