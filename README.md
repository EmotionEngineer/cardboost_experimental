# cardboost — Rate-Distortion Optimal Glass-Box Scorecards

**Version 0.0.1** · BSD-3-Clause · Python >= 3.9 · numpy, pandas, scikit-learn

Ultra-compact (≤ 4 KB) piecewise-constant additive models — *scorecards with
pair interactions* — with **bit-exact** deployment to SQL, C99 and a binary
container.  `cardboost` freezes the certified R87 research pipeline
(`card_v8_base`, the Rate-Distortion champion) as a scikit-learn estimator and
ships every historical winner as a preset.

```
pip install cardboost            # core
pip install cardboost[plot]      # matplotlib figures
pip install cardboost[qr]        # QR-code export
pip install cardboost[fast]      # numba-accelerated Potts solver
pip install cardboost[dev]       # pytest + all extras
```

## 60-second tour

```python
import numpy as np
from cardboost import CardBoostRegressor

model = CardBoostRegressor(preset="v8").fit(X_train, y_train)   # frozen R87 champion
y_hat = model.predict(X_test)              # == deploy score, bit-exact

sql  = model.to_sql("scoring", dialect="postgres")   # postgres/clickhouse/snowflake/bigquery/sqlite
c99  = model.to_c99()                                # single self-contained header
blob = model.to_bytes()                              # Container v3.x, <= 4096 B
model.to_qr("model.png")                             # the model AS a QR code
df   = model.to_scorecard_df(pdo=20)                 # classic points table
rep  = model.to_html_report("report.html", X_train, y_train)
cert = model.certify(X_train, X_test)                # decode determinism audit
```

`CardBoostClassifier` mirrors the same surface (`predict_proba`, Toof binary /
14b multiclass calibration).

## Golden tests are the contract

The package's acceptance criterion is *exact reproduction of the certified R87
metrics*.  `tests/` re-runs the canonical protocol against bundled data
(**no network**):

```
python -m pytest -m "not golden"       # fast suite (~2 min)
python -m pytest -m golden             # acceptance suite (~1-2 h)
```

* **Anchors** (`test_golden_anchors.py`): `cal_rand f0` — fit R²
  **0.8004754327779836**, decoded R² **0.7991800160444765**, v31 **3591 B**;
  `anneal f0` — Toof 0.015556201491460715 → decoded 0.015608297304323386,
  v31 1491 B.
* **Full protocol** (`test_golden_full16.py`): all 16 folds
  (anneal ×3, aps ×3, california_random ×5, california_spatial ×5) against
  `golden/canonical_r87.json`, plus the R87 headline (aps Toof deploy
  mean **0.03298**).
* **Round-trip** (`test_codec_roundtrip.py`): `decode(encode(card)) == card`
  scores **max_diff == 0.0**, `moved_train == 0`, wallet ≤ 4096 B.
* **SQL/C99 parity** (`test_sql_c99.py`): generated code == model scoring
  within **1e-7**, including NaN shelves; SQLite executed in-test, C99
  compiled with gcc.
* **Synthetic controls** (`test_synthetic_controls.py`): q_miss threshold
  effect (gain 49.93 nats > barrier 4.70) and the SafeGrid quadratic probe
  (joint grid ≫ univariate sum).

Entries marked `"exact": true` in the golden table are byte-exact anchors
(tol = 0).  The few `"exact": false` rows carry 7-digit summary values with a
5e-7 tolerance; they are upgraded to exact as the streaming verification run
lands its jsonl rows.  **Linux x86-64 is the canonical platform** (the suite
sets single-thread BLAS itself).  On Apple Silicon 5/6 families are
bit-exact; the two libm-sensitive rows (`anneal_f0`, `cal_spat_f4`) are
skipped there, or run with `CARDBOOST_GOLDEN_TOL=1e-9` to accept last-ulp
drift.

## Presets — *Zero-Tuning Champion by Default, Full Arsenal on Demand*

| preset | provenance | what it turns on |
|---|---|---|
| `v8` *(default)* | R87 champion `card_v8_base` | frozen card: m=32, lr=km=formula, warm 6, gs solver, star g4a-d/p1/m2 (beam ladder on non-cal panels), w1 frags, h3 currency, IsoLink-OOF, 4096 B wallet |
| `v7_h3` | R84–R85 leader | pre-R87 frozen variant (identical numbers in v1.0 — kept for provenance) |
| `iid_max` | R98 | P1Swap2A as beam2 candidate + safe tier + beam3 + 16384 B wallet |
| `ood_fortress` | R97 | hmin_frac = 0.05 (5 % mass threshold) + edge freeze |
| `sensor_rare` | R107/R112/R82 | qfix_dev potts objective + q_miss coordination atom + OOF merge |
| `scorecard_1d` | R71/R82 | rays only, 400–900 B wallet, PDO table output |
| `unregularized_32` | R73 | κ = 0 (unregularised logistic refit) |

All constructor knobs map 1:1 to engine flags (`cardboost.presets.PresetConfig`,
documented with their canonical values).  Rejected research mechanisms are
**not** in the core path; they remain available as explicit knobs
(`include_qmiss`, `include_safe_tier`, `include_msum`, `edge_freeze`,
`hmin_frac`, `oof_merge`, `potts_objective="exact_dev"|"qfix_dev"`, …).

## What the model is

```
score(x) = b0 + Σ_j V_j[bin_j(x_j)] + Σ_atoms atom(x)     (+ IsoLink for regression)
```

* mains: 32-quantile bins per feature, NaN shelf = dedicated bin (right-open
  routing `(-inf, c0), [c0, c1), …` + NaN branch);
* pair atoms: ProjRay (g4a-d), P1 8×8 grid, bilinear M2, MSum, q_miss
  (coordination of missingness) — each pays its MDL debit
  (½·ln n + ln 2 ≈ 4.70 nats) from its OOF credit before admission;
* IsoLink: 16-knot isotonic recalibration of the regression score
  (trained OOF, deployed as piecewise-linear);
* binary: Toof temperature · multiclass: 14b diagonal calibration.

**Invariant III (OOD safety):** edge-bin shelf compensation is audited
(`audit_ood_safety`) — the safe tier is the R87 remedy when the audit fires.

## Deployment guarantees

| surface | guarantee |
|---|---|
| `to_bytes` / `from_bytes` | Container v3.x (gzip-9 aware); decode scores bit-exact vs memory |
| `to_sql(table, dialect)` | pure SQL scoring, ≤ 1e-7 vs memory; NULL/NaN shelf semantics preserved |
| `to_c99()` | dependency-free C99 header (`isnan`/`expf` only), ≤ 1e-7 vs memory |
| `to_qr(path)` | the container bytes as a QR image (model travels on paper) |
| `certify(X)` | wallet bytes, `moved_train`, decode-determinism `max_diff` |

## Repository layout

```
cardboost/
  estimators.py        CardBoostRegressor / CardBoostClassifier
  presets.py           7 frozen presets (canonical values documented)
  datasets.py          bundled data loaders (california npz, anneal, aps) + r72 folds
  core/                binning (deterministic_ulp), engine (fit_fold/fit_deploy),
                       atoms (registry + synthetic controls), calibration, ledger
  codec/               container_v32, sql_export (5 dialects), c99_export, qr_export
  explain/             ANOVA importance, local waterfall, scorecard/PDO table, OOD audit
  plot/                matplotlib figures + self-contained HTML report (inline SVG)
  _vendor/             the certified R71–R87 mechanism stack (unmodified)
data_cache/            california_housing.npz, anneal.data/test, aps_failure_sample.csv
golden/canonical_r87.json   the 16-fold certified metric table
tests/                 golden suite (see above)
```

## Provenance & platform notes

The R87 study (glass-box scorecards as a rate-distortion problem) certified
`card_v8_base` across six dataset families with wallets ≤ 3984 B and
decode ≡ memory scoring.  This package **replays that pipeline verbatim**
(`cardboost/core/engine.py` mirrors `r87_4_qmiss.py`'s `fold_fit` block;
the mechanism stack under `_vendor/` is unmodified).  Canonical numbers were
produced on Linux x86-64; a Mac Mini M4 reproduces 5/6 families bit-exact
(see the golden note above).  No network access is required — all datasets
are bundled in `data_cache/`.

## License

BSD-3-Clause (see `LICENSE`)
