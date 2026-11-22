# Certification Record — `card_v8_base` (`cardboost 1.0`)

**Date:** 2026-11-22  
**Status:** **CERTIFIED** (all formal acceptance gates passed; pre-certification hold lifted).  
**Release Artifact:** `cardboost-1.0` (`v8` = `card_v8_base`, R109 phase arm; `v7_h3` = H3 control arm).  
**Reference Execution Platform:** `macOS-15.3.1-arm64` (Apple Silicon), Python `3.9.6`.

---

## 1. Certification Gate Results

| Gate | Command / Kit | Result | Timing |
|---|---|---|---|
| **Full 16-Fold Golden** (`pytest -m golden`) | `python3 -m pytest -m golden` | **17/17 PASSED** (16 folds + APS deploy mean) | 8415.13s (2:20:15) |
| **P2 Cert Gate (`phase` arm)** | `r87_1_phase_v32.py --arm phase` (kit v3) | **`GATE_RESULT r87-1 phase PASS`** (11 CHECK rows, `moved=0`, `diff=0`) | ~4250s |
| **P2 Cert Gate (`off` arm)** | `r87_1_phase_v32.py --arm off` (kit v3) | **`GATE_RESULT r87-1 off PASS`** (11 CHECK rows, `moved=0`, `diff=0`) | ~4390s |
| **Full Matrix (7×16 + 7 deploy)** | `python3 -m pytest -m matrix -s` | **119/119 PASSED** (114 initial + 5 Grade-B f16-flip re-run) | 15:20:39 + 0:42:50 |
| **Fast + Identity + API Suites** | `pytest -m "not golden and not matrix"` | **29/29 PASSED** (Mac), **13/13** (Linux fast) | ~3 min |

---

## 2. P2 Certification Gate Details (Kit v3, 22 Certified Rows)

Kit v3 fixes verified on Mac (`spawn`-safe `Pool(initializer=_set_arm, initargs=(ARM,))`, non-vacuous `check_gates([]) == False`, `logs/` directory creation, Apple Silicon tiering):
- Every `phase` row confirmed `FIT_DONE ... arm=phase compress=True`.
- Every `off` row confirmed `FIT_DONE ... arm=off compress=False`.
- **`phase` arm:**
  - APS wallets: **2631 / 2916 / 2843 B**; APS deploy-mean Toof: **`0.03297697147116147`** (rounds to **0.03298**, exact R87 champion anchor).
  - California Random `f1..f4` wallets: **2233 / 2758 / 2843 / 2621 B** (`dmax = 0.0` on `f1..f3` vs `r86_2`, 1 ulp `1.11e-16` on `f4`).
  - California Spatial `f1..f4` wallets: **2702 / 2818 / 3074 / 2904 B** (`max = 3074 <= 3500` gate); `f4` `R²_fit = -0.06042`, `R²_dec = -0.05801` (documented Apple Silicon M4 knife-edge side).
- **`off` arm (H3 control):**
  - APS wallets: **2631 / 2916 / 2843 B** (identical to `phase`).
  - California Random `f1..f4` wallets: **2960 / 3470 / 3632 / 3373 B**.
  - California Spatial `f1..f4` wallets: **3559 / 3860 / 4079 / 3792 B** (note: `3500 B` gate applies exclusively to `phase`).

### Cross-Verification: P2 (R71–R87 Protocol) vs Matrix Baseline (Package Engine)

Automated row-by-row comparison of all **22** Mac P2 rows (`v31`, `metric_fit`, `metric_dec`) against `golden/matrix_baseline.json` (`v8` for `phase`, `v7_h3` for `off`):
- **22/22 bit-exact match** (`Δv31 = 0`, `Δfit = 0.0`, `Δdec = 0.0`).

---

## 3. Acceptance Contracts Proven

1. **Metric & Anchor Reproduction:** All R87/R109 (`v8`) and R84 (`v7_h3`) anchors, plus all 7 preset mechanical fingerprints (`iid_max`, `ood_fortress`, `sensor_rare`, `scorecard_1d`, `unregularized_32`), are verified and frozen in `golden/matrix_baseline.json` (119 rows).
2. **Decode Determinism & Deployment Parity:** `moved_train == 0` and `maxdiff_mem_dec == 0.0` across all 119 matrix rows and all 22 P2 rows; SQL (5 dialects) and C99 parity `<= 1e-7`.
3. **Unmodified Research Core:** `cardboost/_vendor/` remains byte-identical to the certified R71–R87 mechanism stack.
