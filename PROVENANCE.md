# PROVENANCE — "mode -> number" Binding (2026-11-22, CERTIFIED)

Status: **CERTIFIED (`card_v8_base`, 2026-11-22).** The mode->number binding is
complete (`v8` = R109 PHASE family; `v7_h3` = H3 control family) and all formal
acceptance gates have passed: `pytest -m golden` (17/17 PASSED), P2 Cert Gate
(`GATE_RESULT r87-1 phase PASS` + `off PASS`, 22/22 bit-exact vs matrix
baseline), and `pytest -m matrix` (119/119 PASSED).

## The three contested numbers — resolved

| Number | What it is | Producing mode | Evidence |
|---|---|---|---|
| **3591 B** | wallet of `california_random_f0`, **m32 portrait** (uncompressed live-cut packing) | **H3 control** (`r87_4 --arm off` = `r87_1 --arm off`; preset `v7_h3`) | `r87_4.jsonl` arm=off row (`v31=3591`); `golden/matrix_baseline.json` `v7_h3::cal_rand_f0` |
| **2548 B** | wallet of `california_random_f0`, **Formula-2 live-cut packing** (`COMPRESS_UCUTS`) | **R109 phase** (`r87_1 --arm phase`; preset `v8`) | `r87_1.jsonl` arm=phase row `v31=2548`, metrics bit-exact vs H3 reference |
| **anneal 14b 0.0260495** | R109 phase **gate anchor** (round record, bit-exact `r86_4_restore`; R108 fold-mean recording 0.02607/0.02605 fit/dec) | **R109 phase** (preset `v8`) | anneal phase rows 14b dec `0.01712937656530835 / 0.018417769546663444 / 0.04261777306324809`, deploy-mean `0.026054973058406632` (gate `<= .0261` PASS) |

The `.0260495` (gate/round-record) vs `.0281425` (H3-family anneal deploy-mean
of the panel rows) pair uses **two different measurement bases AND two
different modes** — the per-fold rows of both families are the ground truth.

## Mode map (empirical & certified)

* **Phase family = `card_v8_base` = preset `v8`** — `r87_1_phase_v32.py --arm phase`:
  `p_phase_law=True` (deterministic phase thresholds, lattice nodes k=4,5,6)
  **+ `COMPRESS_UCUTS=True`** (Formula 2, lossless active-cut packing).
  Anneal wallets **1297/1293/1176** (spec bound <= 1297), anneal 14b deploy-mean
  **0.026055**, cal wallet list (Linux canonical / Mac P2):
  `2548 / 2232(2233) / 2758 / 2843 / 2621 / 2703(2702) / 2415(2818) / 2888(3074) / 2747(2904) / 1939`
  (all `<= 3500 B` gate), aps **2631/2916/2843** (spec bound <= 2916).
* **H3 control family = preset `v7_h3`** — `r87_4 --arm off` (= `r87_1 --arm off`,
  H3 control on all 16 folds): `p_phase_law=False`, m32 portrait. Anneal
  wallets **1491/1451/1323**, anneal 14b deploy-mean **0.0281425**, cal wallets
  **3591 / 2958(2960) / 3470 / 3632 / 3373 / 3984 / 3145(3559) / 3486(3860) / 3700(4079) / 2603(3792)**.
* **Shared (bit-identical across arms on Linux canonical):** aps Toof deploy mean
  **.03298** (`0.03297697`, wallets 2631/2916/2843), cal metrics `cal_rand .80716/.80538`,
  `cal_spat` `0.50336` (Linux) / `0.48104` (Mac), spat f4 `+0.06173` (Linux) / `-0.05801` (Mac).
  Cal/aps metrics stay identical across arms except on Apple Silicon `cal_spat_f3`
  knife-edge cut rounding (`0.73338` vs `0.71790`).

## Golden tables (`golden/`)

* `golden/canonical_r87.json` — historical Linux x86-64 reference:
  - `folds` = **phase family** (`v8` target)
  - `h3_folds` = **H3 control family** (`v7_h3` target, 16/16 rows from `r87_4.jsonl` `arm=off`)
  - `synthetic_controls` = `q_miss` (gain `49.93 > 4.70`) & `SafeGrid` quadratic probe
* `golden/matrix_baseline.json` — **frozen 119-row acceptance baseline**
  (all 7 presets × 16 folds + 7 deploy checks, captured on `macOS-15.3.1-arm64`,
  Python 3.9.6).
* `golden/p2_certification_22.json` — **22 certified P2 rows** (`phase` 11 +
  `off` 11) from Mac `r87_1.jsonl`, verified 22/22 bit-exact vs `matrix_baseline.json`.
* `golden/historical_refs.json` — extracted `r86_2`, `r86_4_restore`, and `r87_4`
  reference rows.

## Packaging note (live-cut packing)

The wallet is the **gzip-9 stream of the container**; `COMPRESS_UCUTS` changes
the container's cut packing, not the compression algorithm. `cal_rand_f0`: raw
container `5353 B` (m32 portrait) -> wallet `3591 B`; Formula-2 packing -> wallet
`2548 B`. CLI `gzip -9` on a stored `cert/<tag>/payload.bin` (uncompressed
container) = `wallet + 12 B` (CLI header/trailer delta). `cardboost.to_bytes()`
returns the certified container; `from_bytes().predict() == predict()` bit-exact.

## Source integrity

* `r87_4_qmiss.py` (md5 `6d037d1e618799a60b674e1de17713b1`) and
  `r87_1_phase_v32.py` (md5 `7f95c0f61574de3c2ad1e2a21facc40f`) verified
  byte-identical to `cardboost_r87_reproduce.zip`.
* Reference env: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1`.

## Determinism

`deterministic_ulp` helpers (`core/binning.py`: 12-significant-digit rounding +
lexicographic tie-break) are wired into the export/presentation surfaces
(`explain.scorecard_table`, `explain.anova`). The FIT path is intentionally
unrounded to stay bit-exact with the frozen reference rows; residual
cross-platform gaps (libm ulps, vendored screening sort ties — e.g. the anneal
and `cal_spat_f4` f16 knife-edges on Apple Silicon) are graded by the golden
tier system (`tests/golden_util.py`): bit-exact on Linux x86-64 (canonical),
tight `1e-6` tolerance when the wallet matches, structural mode + quality band
otherwise.
