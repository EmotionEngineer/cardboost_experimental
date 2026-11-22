# Historical Results & Model Identity — CardBoost 1.0

Date: 2026-11-22 (originally compiled 2026-11-21, updated upon final certification).  
Purpose: Prove that every preset in `cardboost` reproduces **the actual historical models of the R71–R114 research campaign**, rather than stub approximations.  
Sources: `MODEL_COMPENDIUM.md`, `INDEX.md` round digests, `r86`/`r87` canary reports, `cardboost_full_report_en.md`, and primary JSONL logs (`r76_results/*.jsonl`).

---

## 1. Historical Benchmark Summary

### `v8` — `card_v8_base` (R87 Final + R109 Phase) — **CERTIFIED CHAMPION**
| Metric | Historical Value | Source |
|---|---|---|
| `aps` Toof (deploy-mean, 3 folds) | **0.03298** (wallets 2631/2916/2843 B) | R87 final |
| `anneal` 14b (deploy-mean) | **0.0260495** (gate anchor, bit-exact `r86_4_restore`) | R109 canary PASS |
| `cal_rand` / `cal_spat` (fold-mean) | 0.80538 / 0.50336, `f4` **+0.06173** | R87 final |
| Wallets 16/16 | `<= 3984 B`, `decode == mem` | R87 final |
| `anneal_f0` Toof (`fit -> dec`) | `0.017293655045400768 -> 0.01727664026372757`, `v31` 1297 B | canonical |
| `cal_rand_f0` R² (`fit -> dec`) | `0.8004754327779836 -> 0.7991800160444765`, `v31` 2548 B | canonical |

### `v7_h3` — H3 Control (R84 Phase-5 Winner, 4/4 Gates)
| Metric | Historical Value | Source |
|---|---|---|
| `cal_rand` `fit -> dec` (`f0`) | **0.80716 / 0.80538** family; `f0 = 0.8004754327779836 / 0.7991800160444765` | R84 Part 2 |
| `cal_spat` / `f4` | `0.51246 / 0.50336`, `f4` **+0.0610 / +0.0617** | R84 Part 2 |
| `anneal` 14b (deploy-mean) | **0.0281425** | R83/R84 |
| `anneal_f0` Toof `fit -> dec` | `0.015556201491460715 -> 0.015608297304323386`, `v31` 1491 B | canonical |
| Wallets | max **3984 B** / 10 folds (`rand` 3591/2958/3470/3632/3373) | R84 Part 2 |
| Currency | `÷ sigma2_oof` in 1D only; 2D `p1` pairs at base price (`p1` grids identified as source of `f4` fragility) | R84 diagnostics |

### `iid_max` — R98 (`nested S_j x 32` / `P1Swap2A`) + R113 Safe Tier + R114 Beam3
| Mechanism | Historical Behavior | Source |
|---|---|---|
| Budget | **up to 16384 B** ("fat-wallet" IID preset buying California R² with bytes) | R98/R114 |
| `P1Swap2A` / `S_j x 32` | Improves `cal_rand`, while 2D surfaces increase `f4` out-of-box sensitivity (`-0.03...-0.20`) | Full report Fact 2 |
| Safe tier (R113) | **0/16 admissions**; suppresses 20–24 candidates/fold (~290 total); honest gains `<= 48.5 nats` vs `~161 nats` cost | R87-5 canary FAIL |
| Beam3 (R114) | **0/16 admissions**, `honest_gain = 0.0`; `cal_random_f0` finalist `(5,7)` `37.27` vs score `-124.69` | R87-6 canary FAIL |

Historical role: **IID maximizer** (`cal_rand`-oriented) with an expanded 16 KB budget; its experimental post-fit layers (`safe_tier` / `beam3`) honestly abstain on canonical panels — that abstention is their documented signature, not a defect.

### `ood_fortress` — R97 (`H2`/`H1H2` 5% Mass Floor) + R113 Edge Freeze + Dykstra
| Mechanism | Historical Behavior | Source |
|---|---|---|
| `H2` 5% mass threshold | REFUTED as general winner: `cal_rand` `0.8072 -> 0.7819` (0/4 gates) — 5% floor suppresses legitimate 3.125% bins | `MODEL_COMPENDIUM` §12 (R97) |
| `H1H2` side fact | `cal_spat_f4` **+0.2138 (2.5x baseline)** — diagnostic reference for a purely spatial regime | R97 |
| Dykstra centering | Metric projection of marginals (atom does not own marginal) + empty-cell box gate | `r87_safetier` |

Historical role: **OOD distribution-shift regime** enforcing a strict 5% mass floor and edge protection, trading IID granularity for tail robustness.

### `sensor_rare` — R107 Exact Deviance + R112 `q_miss` + R82 `oof_merge`
| Mechanism | Historical Behavior | Source |
|---|---|---|
| R107 exact deviance | `anneal` + `cal` bit-identical neutral; `aps` `f0 -0.0015 / f1 +0.0036`; `qfix_dev` deviance splitting | R86 canaries |
| R112 `q_miss` | **Silent on canonical panels**: admitted 0/16, wire `+0 B` (MDL barrier `0.5 ln n + ln 2 ≈ 4.7 nats`); `aps` 5938/6000 NaN rows, `anneal` 898/898, `cal` 0 -> structural silence; **synthetic control: gain 49.93 > 4.70 ✓** | R87 canaries |
| R82 `oof_merge` (#5) | Segments `x0.79`, `anneal -0.00022`; nominal Potts #5: `anneal -0.00058, f2 -0.00286` | R82 report |

Historical role: **Rare-event / missingness sensor** combining `qfix_dev`, `q_miss` missingness coordination, and `oof_merge`. Its silence on canonical panels where main effects absorb linear missingness is an empirically measured property.

### `scorecard_1d` — R71/R82 Bank-Style 1D Scorecard
| Mechanism | Historical Behavior | Source |
|---|---|---|
| Card composition | **1D-only**: main effects + 1D `g4` rays; pair atoms disabled | R71/R82 |
| Budget / Output | Compact 1D scorecard (`400–900 B` on scorecard-scale features), PDO points table | presets provenance |
| `oof_merge` | Enabled (R82) | presets |

### `unregularized_32` — R71 `card_kappa0` (`kappa = 0`) / R73
| Mechanism | Historical Behavior | Source |
|---|---|---|
| Universal rank | **#1** (`TOP_UNIVERSAL 0.923 > xgboost 0.915`) | R72 head2head |
| `california_random` | **0.7553** (exact match on M4 checkpoint) | v6 checkpoint |
| `anneal` / `aps` | `anneal` Toof `0.0340`; `aps` glob `0.0337` | v6 checkpoint, R75 fact-check |
| Budget | **12–16 KB floor** (R73: `< 1.4 KB` target "arithmetically impossible"; R75: `12837 B`) | R73/R75 |
| Mechanism | 32 quantile shelves **without L0 merging** (`kappa = 0`, unfused arm) | compendium §2.2 |

---

## 2. Identity Audit: Eliminated Silent Stubs (Fixed)

Audit method: end-to-end "preset knob -> engine global" trace (lesson D1: an unwired knob produces a silent duplicate of `v8`). Six dead knobs were identified and wired in v1.0:

| Preset | Unwired Knob (Pre-Fix) | Consequence Before Fix | Fix Applied |
|---|---|---|---|
| `unregularized_32` | `extra.kappa_zero -> R84_COST["km_zero"]` (never read by vendor) | Card was an exact copy of `v8` (`anneal_f0: 1297 B`) | `km=0.0 -> kappa_mult=0.0` (unfused arm `card_kappa0`; card `~2.5x` larger on synthetic data) |
| `unregularized_32` | `max_wallet_bytes=4096` overrode `wallet_bytes=16384` | `kappa=0` card was re-compressed/re-regularized | Both budget fields set to `16384` |
| `iid_max` | `include_safe_tier`, `ledger_mode="beam3"` not wired | R113/R114 layers never executed | Post-fit layers wired per canary protocols `r87_5`/`r87_6` (`ST.build_safe_tier`, `B3.beam_search`/`final_select`/`deploy_with_set`) |
| `iid_max` | `extra.p1_bins_k=32` while `R.P1_BINS` hardcoded `8` | `S_j x 32` grids did not exist | `R.P1_BINS = p1_bins_k` (`8` default, `32` for `iid_max`) |
| `iid_max` | `16384 B` budget blocked by `max_wallet_bytes` | Fat-wallet regime inactive | Both budget fields set to `16384` |
| `ood_fortress` | `hmin_frac=0.05` not wired | R97 mass floor inactive | `U.R84_COST["hmin"] = hmin_frac` (vendor applies `max(25, 0.05 * sum(H))`) |
| `sensor_rare`, `scorecard_1d` | `oof_merge` hardcoded `False` in fold path | R82 OOF merge never fired | `run_fold77` / `fit_calib_spec` receive `cfg.oof_merge` |

Global state leakage across sequential preset calls in `apply_config` was also eliminated (`km_zero`, `hmin`, and `P1_BINS` are reset deterministically on every call).

### Documented Residual Notes
1. **`edge_freeze` (`ood_fortress`):** No standalone `freeze` switch exists in the vendored core. The implemented edge protection is Dykstra marginal centering + empty-cell box gating inside `include_safe_tier` (which `ood_fortress` enables).
2. **`P1Swap2A`:** Reachable via the safe-tier / `beam3` candidate pool (`_mk_p1`), matching historical canaries (R98 "as beam2 candidate").
3. **`scorecard_1d` 400–900 B:** Applies to lower-dimensional scorecard feature sets; on `anneal` (38 features), 1D mains occupy `~1293 B` (`deploy::scorecard_1d::anneal = 1345 B`), and on `aps` wallets drop to `2125–2320 B`.

---

## 3. Automated Identity & Acceptance Verification

* **`tests/test_preset_identity.py` (6/6 PASS):**
  - End-to-end knob wiring and absence of cross-preset state leakage;
  - Budget contracts (`4096` vs `16384`);
  - Mechanical fingerprints: `kappa=0` produces strictly more segments and larger cards than `v8`; `q_miss` attempt metadata present (silent on canonical, loud `49.93 > 4.70` on synthetic control); `scorecard_1d` admits zero 2D pair atoms; `ood_fortress`/`iid_max` report safe-tier and `beam3` metadata in `mech`.
* **`tests/test_full_matrix.py` (119/119 PASS):**
  - All 7 presets × 16 canonical folds (112 rows) + 7 deploy checks frozen in `golden/matrix_baseline.json` (macOS-15.3.1-arm64 / Python 3.9.6).
* **`tests/test_golden_full16.py` (17/17 PASS) & P2 Cert Gate (`PASS` both arms):**
  - Formal acceptance completed 2026-11-22 (see `CERTIFICATION.md` and `METRICS.md`).
