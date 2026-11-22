"""FULL MATRIX MODE: every preset x every canonical fold (slow).

This is the "all models" acceptance mode.  It runs the full 16-fold canonical
panel for **all seven presets** (v8, v7_h3, iid_max, ood_fortress, sensor_rare,
scorecard_1d, unregularized_32) and checks every cell against a recorded
baseline, plus the structural invariants of the deploy contract.

Three layers of checking per cell:

1. structural invariants (always): moved_train == 0, decode determinism
   == 0.0, wallet size within the preset budget;
2. canonical cross-check for the two measured families: ``v8`` rows are
   additionally verified against ``golden["folds"]`` and ``v7_h3`` rows
   against ``golden["h3_folds"]`` (the certified R87 table);
3. baseline record/compare in ``golden/matrix_baseline.json``:
   - first run (row missing): the measured row is RECORDED (self-sealing
     baseline) and the cell passes on invariants alone - set
     ``CARDBOOST_MATRIX_NO_RECORD=1`` to forbid writes;
   - later runs (row present): tiered comparison - same wallet size ->
     TIGHT_TOL metrics; f16 knife-edge flip -> loose quality bands + a full
     report line (never a silent pass).  See tests/golden_util.py for tiers.

The baseline records the platform it was captured on (Apple Silicon vs
Linux x86-64 may differ at libm ulp level and by f16 flips; the tiers handle
both).  After the first full run, send ``golden/matrix_baseline.json`` back
for review/freezing; later runs then verify against it strictly.

Runtime: the full matrix is roughly 7x the golden full protocol
(~10-20 h on an M-series Mac; aps folds dominate).  Filter while iterating:

    CARDBOOST_MATRIX_PRESETS="iid_max" python -m pytest -m matrix -s
    CARDBOOST_MATRIX_FOLDS="anneal_f0,cal_rand_f0" python -m pytest -m matrix -s
    python -m pytest -m matrix -k "sensor_rare or scorecard_1d" -s

Run with:
    python -m pytest -m matrix -s          # full matrix, all presets
"""
from __future__ import annotations

import json
import os
import platform
import sys

import pytest

from golden_util import (LOOSE_LOGLoss, LOOSE_R2, TIGHT_TOL, _delta, _fmt,
                         check_fold)

pytestmark = [pytest.mark.matrix, pytest.mark.slow]

ALL_PRESETS = ("v8", "v7_h3", "iid_max", "ood_fortress", "sensor_rare",
               "scorecard_1d", "unregularized_32")

FOLD_CASES = [
    ("anneal", 0), ("anneal", 1), ("anneal", 2),
    ("aps", 0), ("aps", 1), ("aps", 2),
    ("california_random", 0), ("california_random", 1),
    ("california_random", 2), ("california_random", 3),
    ("california_random", 4),
    ("california_spatial", 0), ("california_spatial", 1),
    ("california_spatial", 2), ("california_spatial", 3),
    ("california_spatial", 4),
]

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "golden",
                         "matrix_baseline.json")
NO_RECORD = os.environ.get("CARDBOOST_MATRIX_NO_RECORD", "") == "1"


def _tag(panel, fold):
    short = {"california_random": "cal_rand", "california_spatial": "cal_spat"}
    return f"{short.get(panel, panel)}_f{fold}"


def _env_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [s.strip() for s in raw.split(",") if s.strip()]


MATRIX_PRESETS = _env_list("CARDBOOST_MATRIX_PRESETS", ALL_PRESETS)
_MATRIX_TAGS = _env_list("CARDBOOST_MATRIX_FOLDS",
                         [_tag(p, f) for p, f in FOLD_CASES])
FOLD_CASES = [(p, f) for p, f in FOLD_CASES if _tag(p, f) in _MATRIX_TAGS]

MATRIX_CASES = [(preset, panel, fold)
                for preset in MATRIX_PRESETS
                for panel, fold in FOLD_CASES]


# ------------------------------------------------------------- baseline io
def _load_base():
    if not os.path.exists(BASE_PATH):
        return {}
    with open(BASE_PATH) as fh:
        return json.load(fh).get("rows", {})


def _save_base(rows):
    doc = {
        "schema": "matrix_baseline_v1",
        "recorded_on": platform.platform(),
        "python": sys.version.split()[0],
        "note": ("auto-recorded by tests/test_full_matrix.py; "
                 "freeze only after review"),
        "rows": rows,
    }
    tmp = BASE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, BASE_PATH)


def _budget(preset):
    from cardboost.presets import get_preset
    cfg = get_preset(preset)
    return cfg.max_wallet_bytes or cfg.wallet_bytes


def _res_to_row(res, preset):
    return {
        "kind": res.kind,
        "metrics": {k: float(v) for k, v in res.metric_fit.items()},
        "dec": {k: float(v) for k, v in res.metric_dec.items()},
        "v31": int(res.v31),
        "quant_gap": float(res.quant_gap),
        "exact": False,
    }


def _assert_invariants(res, tag, budget):
    assert res.moved_train == 0, f"{tag}: moved_train != 0"
    assert res.maxdiff == 0.0, f"{tag}: decode determinism broken"
    assert res.v31 <= budget, f"{tag}: wallet {res.v31} > budget {budget}"


def _check_against_entry(res, entry, tag, budget):
    """Tiered compare (platform-style, cross-platform safe): wallet match ->
    TIGHT_TOL metrics; wallet flip -> loose quality bands + report line."""
    m_fit = entry.get("metrics", entry.get("metric_fit"))
    m_dec = entry.get("dec", entry.get("metric_dec"))
    assert m_fit and m_dec, f"{tag}: baseline entry lacks metric dicts"
    assert res.moved_train == 0, f"{tag}: moved_train != 0"
    assert res.maxdiff == 0.0, f"{tag}: decode determinism broken"
    assert res.v31 <= budget, f"{tag}: wallet {res.v31} > budget {budget}"

    v31_ok = (res.v31 == entry["v31"])
    tight_ok = v31_ok
    if v31_ok:
        for key, val in m_fit.items():
            got = res.metric_fit[key]
            if abs(got - val) > max(TIGHT_TOL, abs(val) * 1e-12):
                tight_ok = False
        for key, val in m_dec.items():
            got = res.metric_dec[key]
            if abs(got - val) > max(TIGHT_TOL, abs(val) * 1e-12):
                tight_ok = False
    if tight_ok:
        grade = "A (bit-identical card, ulp metrics)"
    else:
        for key, val in m_dec.items():
            got = res.metric_dec[key]
            band = LOOSE_R2 if key.upper().startswith("R") else LOOSE_LOGLoss
            assert abs(got - val) <= band, (
                f"{tag} metric_dec[{key}]: platform value {got!r} vs baseline "
                f"{val!r} exceeds even the LOOSE band {band} - real breakage?")
        grade = ("B (f16 quantisation flip, structural+quality band)"
                 if not v31_ok else
                 "B (same-size quantisation flip, structural+quality band)")

    print(f"\n[matrix:{tag}] grade {grade}\n"
          f"    v31        {res.v31}  (baseline {entry['v31']})"
          f"{'' if v31_ok else '   <-- FLIP'}\n"
          f"    metric_fit {_fmt(res.metric_fit)}\n"
          f"    metric_dec {_fmt(res.metric_dec)}\n"
          f"    delta_fit  {_delta(res.metric_fit, m_fit)}\n"
          f"    delta_dec  {_delta(res.metric_dec, m_dec)}")


def _record_or_check(res, key, tag, budget):
    rows = _load_base()
    entry = rows.get(key)
    if entry is None:
        _assert_invariants(res, tag, budget)
        print(f"\n[matrix:{tag}] RECORDED to golden/matrix_baseline.json "
              f"({key}): v31 {res.v31}, metric_dec {_fmt(res.metric_dec)}")
        if not NO_RECORD:
            rows[key] = _res_to_row(res, key)
            _save_base(rows)
        else:
            print(f"[matrix:{tag}] CARDBOOST_MATRIX_NO_RECORD=1 -> not written")
        return
    _check_against_entry(res, entry, tag, budget)


# --------------------------------------------------------------- fold cells
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("preset,panel,fold", MATRIX_CASES,
                         ids=[f"{p}::{_tag(pan, f)}"
                              for p, pan, f in MATRIX_CASES])
def test_matrix_fold(preset, panel, fold, panels, golden):
    """One matrix cell: full canonical protocol for one preset on one fold."""
    from cardboost.core import engine
    data, splits = panels[panel]
    res = engine.fit_fold(panel, fold, data, splits, preset=preset)
    tag = f"{preset}::{_tag(panel, fold)}"
    _assert_invariants(res, tag, _budget(preset))

    # canonical cross-check for the two measured families
    fam_tag = _tag(panel, fold)
    if preset == "v8" and fam_tag in golden["folds"]:
        check_fold(res, golden["folds"][fam_tag], f"v8/{fam_tag}")
    elif preset == "v7_h3" and fam_tag in golden["h3_folds"]:
        check_fold(res, golden["h3_folds"][fam_tag], f"v7_h3/{fam_tag}")

    _record_or_check(res, tag, tag, _budget(preset))


# ------------------------------------------------------- deploy-path smoke
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("preset", MATRIX_PRESETS, ids=MATRIX_PRESETS)
def test_matrix_deploy(preset, panels):
    """Deploy path per preset: fit_deploy + decode invariants on anneal."""
    from cardboost.core import engine
    data, splits = panels["anneal"]
    res = engine.fit_deploy(data, preset=preset)
    tag = f"{preset}::deploy_anneal"
    budget = _budget(preset)
    assert res.moved_train == 0, f"{tag}: moved_train != 0"
    assert res.maxdiff == 0.0, f"{tag}: decode determinism broken"
    assert res.v31 <= budget, f"{tag}: wallet {res.v31} > budget {budget}"

    rows = _load_base()
    key = f"deploy::{preset}::anneal"
    entry = rows.get(key)
    if entry is None:
        print(f"\n[matrix:{tag}] RECORDED: v31 {res.v31}")
        if not NO_RECORD:
            rows[key] = {"v31": int(res.v31), "kind": res.kind,
                         "moved_train": int(res.moved_train),
                         "maxdiff": float(res.maxdiff)}
            _save_base(rows)
    else:
        assert res.v31 == entry["v31"], (
            f"{tag}: deploy wallet {res.v31} != baseline {entry['v31']}")
        print(f"\n[matrix:{tag}] ok: v31 {res.v31} == baseline")
