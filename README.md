# cardboost — Rate-Distortion Optimal Glass-Box Scorecards

**Version 1.0.0** · **Status: CERTIFIED (`card_v8_base`, 2026-11-22)** · BSD-3-Clause · Python >= 3.9 · numpy, pandas, scikit-learn

Ultra-compact (<= 4 KB) piecewise-constant additive models — *scorecards with
pair interactions* — with **bit-exact** deployment to SQL, C99, QR codes, and a
binary container. `cardboost` freezes the certified R87/R109 research pipeline
(`card_v8_base`, the Rate-Distortion champion) as a scikit-learn estimator and
ships every historical milestone model as a named preset.

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

model = CardBoostRegressor(preset="v8").fit(X_train, y_train)   # certified R87/R109 champion
y_hat = model.predict(X_test)              # == deploy score, bit-exact

sql  = model.to_sql("scoring", dialect="postgres")   # postgres/clickhouse/snowflake/bigquery/sqlite
c99  = model.to_c99()                                # single self-contained header
blob = model.to_bytes()                              # Container v3.x, <= 4096 B
model.to_qr("model.png")                             # the model AS a QR code
df   = model.to_scorecard_df(pdo=20)                 # classic points table
rep  = model.to_html_report("report.html", X_train, y_train)
cert = model.certify(X_train, X_test)                # decode determinism audit
```

`CardBoostClassifier` mirrors the same surface (`predict_proba`, `Toof` binary /
`14b` multiclass calibration).

## Documentation Map

| Document | Contents |
|---|---|
| `MODEL_DESCRIPTIONS.md` | Mathematical formulation, pipeline stages, 7-preset parameter matrix, and detailed model cards |
| `METRICS.md` | Complete metrics across all 7 presets × 16 folds (119 matrix rows), 22 P2 gate rows, canonical Linux reference, and historical benchmarks |
| `PRESET_IDENTITY.md` | Proof that every preset reproduces its historical R71–R114 model (with wired-knob audit & identity tests) |
| `CERTIFICATION.md` | Formal certification record for `card_v8_base` (2026-11-22: full16 17/17 + P2 `phase`/`off` PASS + matrix 119/119) |
| `PROVENANCE.md` | Mode-to-number binding (`v8` 2548 B / 0.026055 vs `v7_h3` 3591 B / 0.028143), packaging notes, and platform tiers |

## Certified Acceptance Results (2026-11-22)

`card_v8_base` (`preset="v8"`) passed all formal certification suites (reference
platform: `macOS-15.3.1-arm64`, Python `3.9.6`):

```
python -m pytest -m "not golden and not matrix"   # fast + identity + codec + SQL/C99 suite (29/29 PASS)
python -m pytest -m golden                        # 16 canonical folds + APS deploy mean (17/17 PASS, 8415s)
python -m pytest -m matrix -s                     # all 7 presets x 16 folds + 7 deploy (119/119 PASS)
```

* **Anchors** (`test_golden_anchors.py`, both mode families):
  - `v8` (R109 phase = `card_v8_base`): `cal_rand_f0` — fit R² **0.8004754327779836**, decoded R² **0.7991800160444765**, `v31` **2548 B**; `anneal_f0` — Toof `0.017293655045400768 -> 0.01727664026372757`, `v31` **1297 B**.
  - `v7_h3` (H3 control): identical `cal_rand_f0` R² at `v31` **3591 B**; `anneal_f0` — Toof `0.015556201491460715 -> 0.015608297304323386`, `v31` **1491 B**.
* **Full protocol** (`test_golden_full16.py`): **17/17 PASSED** across all 16 folds (`anneal` ×3, `aps` ×3, `california_random` ×5, `california_spatial` ×5) plus the R87 headline (`aps` Toof deploy mean **0.03298**, exact **0.03297697147116147**).
* **P2 Certificate Gate (`r87_1_phase_v32.py`, kit v3):** Both `arm=phase` (`GATE_RESULT r87-1 phase PASS`) and `arm=off` (`GATE_RESULT r87-1 off PASS`) passed all 11+11 checks with `moved_train=0`, `maxdiff=0.0`, and **22/22 bit-exact agreement** with `golden/matrix_baseline.json`.
* **Round-trip** (`test_codec_roundtrip.py`): `decode(encode(card)) == card` scores `max_diff == 0.0`, `moved_train == 0`, wallet `<= 4096 B`.
* **SQL/C99 parity** (`test_sql_c99.py`): generated code == model scoring within `1e-7`, including NaN shelves; SQLite executed in-test, C99 compiled with gcc.
* **Synthetic controls** (`test_synthetic_controls.py`): `q_miss` threshold effect (gain `49.93 nats > barrier 4.70`) and the `SafeGrid` quadratic probe (joint grid >> univariate sum).
* **Full matrix** (`test_full_matrix.py`, `-m matrix`): **119/119 PASSED** across all 7 presets × 16 folds + 7 deploy checks, frozen in `golden/matrix_baseline.json`. Narrow runs while iterating:
  ```
  CARDBOOST_MATRIX_PRESETS="iid_max" python -m pytest -m matrix -s
  CARDBOOST_MATRIX_FOLDS="anneal_f0,cal_rand_f0" python -m pytest -m matrix -s
  ```

### Platform Tiering (`tests/golden_util.py`)

**Linux x86-64** is the canonical bit-exact platform (`tol = 0` for `exact` rows). Other platforms (such as **Apple Silicon**) run the same suite in tiered **platform mode**:

| Tier | Linux x86-64 | Apple Silicon (M-series) |
|---|---|---|
| `california` / `aps` cards (`v31`, `moved_train`, `decode`) | bit-exact | bit-exact (observed) |
| `Toof`/`14b` metric evaluation | bit-exact | libm ulp (`~4e-11`) -> `CARDBOOST_GOLDEN_TOL` default `1e-6` |
| f16 knife-edge rows (`anneal_*`, `cal_spat_*`) | bit-exact | structural mode: invariants + quality band (`5e-3` / `0.15` R²) + full report |

`CARDBOOST_GOLDEN_STRICT=1` forces bit-exact checks on any platform.

## Presets — *Zero-Tuning Champion by Default, Full Arsenal on Demand*

| Preset | Lineage | Key Mechanisms | Cal Rand R² mean | APS Toof mean | Anneal 14b mean | Wallet range (B) |
|---|---|---|---|---|---|---|
| `v8` *(default)* | R87 + R109 phase (`card_v8_base`) | `p_phase_law=True`, `compress_ucuts=True`, `m=32`, `gs`, `h3` currency, `IsoLink-OOF`, `4 KB` | 0.80538 | 0.03298 | 0.026055 | 1176 – 3074 |
| `v7_h3` | R84–R85 H3 control | `dead_col_law="h3"`, `p_phase_law=False`, `compress_ucuts=False` (`m32` portrait), `4 KB` | 0.80538 | 0.03298 | 0.028142 | 1323 – 4079 |
| `iid_max` | R98 + R113 + R114 | `p1_bins_k=32` (`S_j×32`), `include_safe_tier=True`, `ledger_mode="beam3"`, `16 KB` | 0.80684 | 0.03298 | 0.026055 | 1176 – 7028 |
| `ood_fortress` | R97 H1H2 + R113 | `hmin_frac=0.05` (5% mass floor), `edge_freeze=True`, `include_safe_tier=True`, `4 KB` | 0.75249 | 0.03425 | 0.254791 | 878 – 1276 |
| `sensor_rare` | R107/R111 + R112 + R82 | `potts_objective="qfix_dev"`, `include_qmiss=True`, `oof_merge=True`, `4 KB` | 0.80523 | 0.03284 | 0.026000 | 1188 – 3047 |
| `scorecard_1d` | R71/R82 1D scorecard | `enable_pairs=False` (1D mains + `g4` rays only), `oof_merge=True`, PDO table, `4 KB` | 0.78859 | 0.03297 | 0.026000 | 1188 – 2722 |
| `unregularized_32` | R71/R73 `card_kappa0` | `kappa_zero=True` (`kappa=0`, unfused 32 quantile shelves), `16 KB` | 0.80494 | 0.03320 | 0.031523 | 1945 – 15978 |

See `MODEL_DESCRIPTIONS.md` and `METRICS.md` for full parameter tables and all 119 per-fold rows.

## What the model is

```
score(x) = b0 + Σ_j V_j[bin_j(x_j)] + Σ_atoms atom(x)     (+ IsoLink for regression)
```

* **Mains:** 32-quantile bins per feature, NaN shelf = dedicated bin (right-open routing `(-inf, c0), [c0, c1), …` + NaN branch);
* **Pair atoms:** `ProjRay` (`g4a-d`), `P1` 8×8 grid (`32×32` in `iid_max`), bilinear `M2`, `MSum`, `q_miss` (coordination of missingness) — each pays its MDL debit (`½ ln n + ln 2 ≈ 4.70 nats`) from its OOF credit before admission;
* **IsoLink:** 16-knot isotonic recalibration of the regression score (trained OOF, deployed as piecewise-linear);
* **Binary / Multiclass:** `Toof` temperature scaling / `14b` diagonal calibration.

**Invariant III (OOD safety):** edge-bin shelf compensation is audited (`audit_ood_safety`) — the safe tier is the R87 remedy when the audit fires.

## Deployment guarantees

| Surface | Guarantee |
|---|---|
| `to_bytes` / `from_bytes` | Container v3.x (gzip-9 aware); decode scores bit-exact vs memory (`max_diff == 0.0`) |
| `to_sql(table, dialect)` | Pure SQL scoring, `<= 1e-7` vs memory; `NULL`/`NaN` shelf semantics preserved |
| `to_c99()` | Dependency-free C99 header (`isnan`/`expf` only), `<= 1e-7` vs memory |
| `to_qr(path)` | Container bytes encoded as a QR image (model travels on paper) |
| `certify(X)` | Wallet bytes, `moved_train`, decode-determinism `max_diff` |

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
golden/
  canonical_r87.json       16-fold Linux canonical table (v8 + v7_h3 + synthetic controls)
  matrix_baseline.json     119-row frozen Mac acceptance matrix (7 presets x 16 folds + 7 deploy)
  p2_certification_22.json 22 certified P2 gate rows (phase + off)
  historical_refs.json     r86_2 / r86_4_restore / r87_4 reference rows
tests/                 fast, identity, codec, SQL/C99, golden anchors, full16, full matrix
README.md              project overview
MODEL_DESCRIPTIONS.md  architecture & all 7 model cards
METRICS.md             complete metrics compendium (all 119 rows + canonical + P2 + historical)
PRESET_IDENTITY.md     historical model identity & wired-knob audit
CERTIFICATION.md       formal card_v8_base certification record (2026-11-22)
PROVENANCE.md          mode->number binding & packaging details
```

## License

BSD-3-Clause (see `LICENSE`).
