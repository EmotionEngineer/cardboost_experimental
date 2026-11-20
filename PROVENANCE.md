# PROVENANCE — «mode → number» binding (2026-11-20, CONFIRMED)

Status: **PROVISIONAL candidate.** The mode→number binding is complete and the
user decision (2026-11-20) is implemented: **`card_v8_base` (preset `v8`) = the
R109 PHASE family**; the H3 control family lives under preset `v7_h3`. The
"certified" label stays withheld until (a) the full-protocol re-verify of the
inherited rows lands (`r76_work/r76_results/r87_1.jsonl`, 11 rows flagged
`pending P2`) and (b) the Linux x86-64 golden run passes against both families.

## The three contested numbers — resolved

| Number | What it is | Producing mode | Evidence |
|---|---|---|---|
| **3591 B** | wallet of `california_random_f0`, **m32 portrait** (uncompressed live-cut packing) | **H3 control** (`r87_4 --arm off` = `r87_1 --arm off`; preset `v7_h3`) | `r87_4.jsonl` arm=off row (`v31=3591`); report §5.1 panel rows |
| **2548 B** | wallet of `california_random_f0`, **Formula-2 live-cut packing** (`COMPRESS_UCUTS`) | **R109 phase** (`r87_1 --arm phase`; preset `v8`) | binding run 2026-11-20: `r87_1.jsonl` arm=phase row `v31=2548`, metrics bit-exact vs H3 reference |
| **anneal 14b 0.0260495** | R109 phase **gate anchor** (round record, bit-exact `r86_4_restore`; R108 fold-mean recording 0.02607/0.02605 fit/dec) | **R109 phase** (preset `v8`) | binding run: anneal phase rows 14b dec `0.01712937656530835 / 0.018417769546663444 / 0.04261777306324809`, deploy-mean `0.026054973058406632` (gate `<= .0261` PASS) |

The `.0260495` (gate/round-record) vs `.0281425` (H3-family anneal deploy-mean
of the §5.1 panel rows) pair uses **two different measurement bases AND two
different modes** — see the report §3.1 measurement-basis note. The per-fold
rows of both families are the ground truth.

## Mode map (empirical, 2026-11-20)

* **Phase family = `card_v8_base` = preset `v8`** — `r87_1_phase_v32.py --arm phase`:
  `p_phase_law=True` (deterministic phase thresholds, lattice nodes k=4,5,6)
  **+ `COMPRESS_UCUTS=True`** (Formula 2, lossless active-cut packing).
  Anneal wallets **1297/1293/1176** (ТЗ bound ≤1297), anneal 14b deploy-mean
  ≈ **0.02605**, cal wallet list **2548/2232/2758/2843/2621/2703/2415/2888/2747/1939**
  (max 2888 ≤ 3500 gate), aps **2631/2916/2843** (ТЗ bound ≤2916).
* **H3 control family = preset `v7_h3`** — `r87_4 --arm off` (= `r87_1 --arm off`,
  "bit-exact H3 on all 16 folds"): `p_phase_law=False`, m32 portrait. Anneal
  wallets **1491/1451/1323**, anneal 14b deploy-mean **0.0281425**, cal wallets
  **3591/2958/3470/3632/3373/3984/3145/3486/3700/2603**. These are the report
  §5.1 panel rows — "the ground truth for the deployed base on the canonical
  protocol".
* **Shared (bit-xarm, identical in both families):** aps Toof deploy mean
  `.03298` (wallets 2631/2916/2843), cal metrics `cal_rand .80716/.80538`,
  `cal_spat .51246/.50336`, spat f4 `+.06173`. Cal/aps **metrics never differ
  between the arms** — only anneal metrics (phase law) and wallet packing differ.

## Golden tables (`golden/canonical_r87.json`)

* `folds` = **phase family** (v8 target). 5 rows verified from `r87_1.jsonl`
  arm=phase (anneal f0–f2, cal_rand_f0, cal_spat_f0); 11 rows inherited
  (aps ×3 = bit-xarm; cal f1–f4 = metrics bit-xarm + wallets from the R87-1
  10-panel list) — each row carries a `source` flag; `pending P2` rows will be
  re-verified by the full-protocol run.
* `h3_folds` = **H3 family** (v7_h3 target): 16/16 exact rows from
  `r76_work/r76_results/r87_4.jsonl` (`arm=off`, `metric_fit` + `metric_dec` +
  `v31` — never from the rounded `base_`-summary, which carried wrong values).

## Packaging note ("упаковка живых разрезов")

The wallet is the **gzip-9 stream of the container**; `COMPRESS_UCUTS` changes
the container's cut packing, not the compression. cal_rand_f0: raw container
5353 B (m32 portrait) → wallet 3591 B; Formula-2 packing → wallet 2548 B.
CLI `gzip -9` on a stored `cert/<tag>/payload.bin` (uncompressed container) =
wallet + 12 B (CLI header/trailer delta). `cardboost.to_bytes()` returns the
certified container; `from_bytes().predict() ≡ predict()` bit-exact.

## Source integrity

* `r87_4_qmiss.py` (md5 `6d037d1e618799a60b674e1de17713b1`) and
  `r87_1_phase_v32.py` (md5 `7f95c0f61574de3c2ad1e2a21facc40f`) verified
  byte-identical to `cardboost_r87_reproduce.zip`.
* Reference env (`r76_work/r87_4_run.sh`, `r76_work/r87_1_run.sh`):
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1`.

## Determinism

`deterministic_ulp` helpers (`core/binning.py`: 12-significant-digit rounding +
lexicographic tie-break) are wired into the export/presentation surfaces
(`explain.scorecard_table`, `explain.anova`). The FIT path is intentionally
unrounded to stay bit-exact with the frozen reference rows; residual
cross-platform gaps (libm ulps, vendored screening sort ties — e.g. the anneal
f16 knife-edge on Apple Silicon) are graded by the golden tier system
(`tests/golden_util.py`): bit-exact on Linux x86-64 (canonical), tight 1e-6
tolerance when the wallet matches, structural mode + drift report otherwise.
