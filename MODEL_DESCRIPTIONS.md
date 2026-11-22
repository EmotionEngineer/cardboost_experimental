# CardBoost 1.0 — Model & Architecture Descriptions

`cardboost` implements **Rate-Distortion Optimal Glass-Box Scorecards**: ultra-compact piecewise-constant additive models with selective pairwise interaction atoms, serialized in a bit-exact binary container (`Container v3.x`, typically `<= 4096 B`) and deployable with `<= 1e-7` parity to SQL, C99, QR codes, and PDO points tables.

---

## 1. Mathematical Formulation & Pipeline Architecture

For an input vector $x \in (\mathbb{R} \cup \{\text{NaN}\})^p$, a CardBoost model computes:

$$\text{score}(x) = b_0 + \sum_{j=1}^{p} V_j\bigl[\text{bin}_j(x_j)\bigr] + \sum_{a \in \mathcal{A}} a(x)$$

followed by a calibrated link:
- **Regression (`CardBoostRegressor`):** `IsoLink-OOF` — a 16-knot piecewise-linear isotonic recalibration trained out-of-fold (`calibration="isolink_oof"`).
- **Binary Classification (`CardBoostClassifier`):** `Toof` temperature scaling on log-odds.
- **Multiclass Classification (`CardBoostClassifier`):** `14b` diagonal per-class calibration.

### Core Algorithmic Stages

1. **Quantile Binning & Dedicated NaN Shelf (`core/binning.py`):**
   - Each numeric feature is discretized into up to `max_bins = 32` quantile shelves using right-open routing `(-inf, c0), [c0, c1), ..., [c_{k-1}, +inf)`.
   - Missing values (`NaN` / `NULL`) are always routed to a dedicated NaN shelf (`bin == NaN_SHELF`) rather than imputed.
   - `deterministic_ulp` provides 12-significant-digit rounding and lexicographic tie-breaking on export/presentation surfaces while keeping the core fit path untouched for bit-exact reproduction.

2. **Main-Effect Fitting & L0 Potts Regularization (`core/engine.py`, `_vendor/`):**
   - Solved via Gauss-Seidel cyclic coordinate descent (`solver="gs"`, `warm_passes=6`) with adaptive step and merge penalty multipliers (`lr="formula" -> min(1, sqrt(10/p))`, `km="formula"`).
   - Adjacent bins with similar effects are fused via an L0 Potts merge solver (`kappa_mult` scales the merge penalty; setting `kappa_zero=True` disables L0 merging so all 32 shelves stay distinct).
   - **Phase Law (`p_phase_law=True`, `dead_col_law="phase_v32"`):** R109 deterministic phase thresholds (`k = 4, 5, 6`) governing dead-column and cut quantization phases (`v8` champion). In contrast, the H3 control (`p_phase_law=False`, `dead_col_law="h3"`) uses the pre-R109 R84–R85 law (`v7_h3`).
   - **Mass Floor (`hmin_frac`):** Enforces a minimum Hessian mass per bin (`max(25, hmin_frac * sum(H))`); `hmin_frac=0.05` activates the R97 5% OOD mass floor (`ood_fortress`).
   - **OOF Merging (`oof_merge`):** R82 out-of-fold bin merging (`sensor_rare`, `scorecard_1d`).
   - **Potts Objective (`potts_objective`):** `"quadratic"` (default L2 quadratic surrogate) or `"qfix_dev"` (R107/R111 exact-deviance / fixed-curvature deviance objective for rare-event sensors).

3. **Pairwise Interaction Atom Pool & MDL Ledger (`core/atoms.py`, `core/ledger.py`):**
   - Candidate pairwise atoms are screened (`screen_mode="s12_pure"` on California, `"legacy"` elsewhere) and scored in **H3 credit currency** (`R84_COST`, 1D credit scaled by `1 / sigma2_oof`, pair-exempt):
     - **`g4a`, `g4b`, `g4c`, `g4d` (`ProjRay`):** 1D projected ray interactions along diagonal/anti-diagonal axes.
     - **`p1`:** `8x8` (or `32x32` when `p1_bins_k=32` in `iid_max`) quantized 2D grid atom.
     - **`m2`:** Bilinear rank-1 multiplicative interaction atom.
     - **`msum` (optional):** Sum-projection atom (`include_msum=True`).
     - **`q_miss` (`QMissAtom`, R112, post-fit):** Coordination atom capturing joint missingness (`NaN` co-occurrence / threshold-NaN gates).
     - **`safe_grid` (`SafeGridAtom`, R113, post-fit):** Experimental safe-tier 2D grid with Dykstra marginal centering and empty-cell box gating.
   - **MDL Admission Barrier:** Every candidate atom must earn positive net OOF credit after paying its description-length debit (`0.5 * ln(n) + ln(2) ≈ 4.70 nats` per parameter).
   - **Ledger Arbitration (`ledger_mode`):** `"greedy"` (default R87 sequential admission) or `"beam3"` (R114 depth-3 beam search over candidate subsets).

4. **Container v3.x & Formula-2 Live-Cut Packing (`codec/container_v32.py`):**
   - Cuts are quantized to `float16` deltas and gzip-9 compressed into a self-contained binary stream (`v31` wallet size in bytes).
   - **`compress_ucuts=True` (Formula-2 packing, R109):** Losslessly packs only active/live cut boundaries, shrinking `cal_rand_f0` from `3591 B` (`m32` uncompressed portrait, `v7_h3`) to `2548 B` (`v8`) with **zero** change in predictions (`maxdiff == 0.0`, `moved_train == 0`).

5. **Deployment & Auditing Guarantees:**
   - `to_bytes()` / `from_bytes()`: Bit-exact round-trip (`maxdiff == 0.0`, `moved_train == 0`).
   - `to_sql(table, dialect)`: Pure SQL (`postgres`, `clickhouse`, `snowflake`, `bigquery`, `sqlite`), `<= 1e-7` parity including `NULL`/`NaN` shelf routing.
   - `to_c99()`: Dependency-free single-header C99 (`isnan`/`expf` only), `<= 1e-7` parity.
   - `to_qr(path)`: Encodes the full binary container into a scannable QR image.
   - `to_scorecard_df(pdo=20)`: Classic credit-scoring points table (base points + per-bin points).
   - `audit_ood_safety(X)`: Audits edge-bin shelf compensation for OOD exposure (Invariant III).

---

## 2. Preset Parameter Matrix (All 7 Models)

| Knob | `v8` *(default)* | `v7_h3` | `iid_max` | `ood_fortress` | `sensor_rare` | `scorecard_1d` | `unregularized_32` |
|---|---|---|---|---|---|---|---|
| **Historical Lineage** | R87 + R109 phase | R84–R85 H3 control | R98 + R113 + R114 | R97 H1H2 + R113 | R107 + R112 + R82 | R71/R82 1D | R71/R73 `card_kappa0` |
| **`dead_col_law`** | `"phase_v32"` | `"h3"` | `"phase_v32"` | `"phase_v32"` | `"phase_v32"` | `"phase_v32"` | `"phase_v32"` |
| **`p_phase_law`** | `True` | `False` | `True` | `True` | `True` | `True` | `True` |
| **`compress_ucuts`** | `True` | `False` | `True` | `True` | `True` | `True` | `True` |
| **`enable_pairs`** | `True` | `True` | `True` | `True` | `True` | **`False`** | `True` |
| **`star_types`** | `g4a-d, p1, m2` | `g4a-d, p1, m2` | `g4a-d, p1, m2` | `g4a-d, p1, m2` | `g4a-d, p1, m2` | **`g4a-d` (1D rays)** | `g4a-d, p1, m2` |
| **`p1_bins_k`** | `8` | `8` | **`32`** (`S_j×32`) | `8` | `8` | `8` | `8` |
| **`ledger_mode`** | `"greedy"` | `"greedy"` | **`"beam3"`** | `"greedy"` | `"greedy"` | `"greedy"` | `"greedy"` |
| **`include_safe_tier`** | `False` | `False` | **`True`** | **`True`** | `False` | `False` | `False` |
| **`hmin_frac`** | `0.0` | `0.0` | `0.0` | **`0.05`** (5%) | `0.0` | `0.0` | `0.0` |
| **`edge_freeze`** | `False` | `False` | `False` | **`True`** | `False` | `False` | `False` |
| **`potts_objective`** | `"quadratic"` | `"quadratic"` | `"quadratic"` | `"quadratic"` | **`"qfix_dev"`** | `"quadratic"` | `"quadratic"` |
| **`include_qmiss`** | `False` | `False` | `False` | `False` | **`True`** | `False` | `False` |
| **`oof_merge`** | `False` | `False` | `False` | `False` | **`True`** | **`True`** | `False` |
| **`kappa_zero`** | `False` | `False` | `False` | `False` | `False` | `False` | **`True`** (`κ=0`) |
| **`wallet_bytes`** | `4096` | `4096` | **`16384`** | `4096` | `4096` | `4096` | **`16384`** |

---

## 3. Detailed Model Cards

### 1. `v8` (`card_v8_base`) — **Certified Rate-Distortion Champion** *(Default)*
- **Origin:** R87 final model + R109 phase arm (`p_phase_law=True`, `COMPRESS_UCUTS=True`).
- **Role:** Primary production model balancing accuracy, OOD stability, and strict `<= 4 KB` footprint.
- **Verified Performance (Mac `card_v8_base` certificate, 2026-11-22):**
  - `aps` Toof deploy-mean: **0.03298** (`0.03297697147116147`, wallets **2631 / 2916 / 2843 B**).
  - `anneal` 14b deploy-mean: **0.026055** (wallets **1297 / 1293 / 1176 B**).
  - `california_random` mean R²: **0.80538** (`f0` R² **0.79918** at **2548 B**; wallets **2233–2843 B**).
  - `california_spatial` mean R²: **0.48104** (Mac matrix; `0.50336` Linux canonical; wallets **2415–3074 B**).

### 2. `v7_h3` — **H3 Control Arm (Pre-Phase Baseline)**
- **Origin:** R84–R85 Phase-5 winner (`r87_4 --arm off` / `r87_1 --arm off`), `p_phase_law=False`, `compress_ucuts=False` (`m32` uncompressed live-cut portrait).
- **Role:** Historical control proving the exact byte savings of Formula-2 cut packing (`3591 B -> 2548 B` on `cal_rand_f0` with identical R²) and the R109 phase-law improvement on `anneal` (`0.0281425 -> 0.026055`).
- **Verified Performance:** `aps` Toof **0.03298** (identical to `v8`); `cal_rand` R² **0.80538** (`2960–3632 B`); `anneal` 14b **0.0281425** (`1323–1491 B`); `cal_spat` wallets up to **4079 B** (Mac `cal_spat_f3`).

### 3. `iid_max` — **High-Budget IID Maximizer**
- **Origin:** R98 (`S_j x 32` fine 2D grids, `p1_bins_k=32`), R113 (`SafeGridAtom`), R114 (`beam3` ledger), and `16384 B` wallet budget.
- **Role:** Maximizes IID accuracy on dense continuous data by allowing higher-resolution 2D grids and larger container budgets (`cal_rand` mean R² **0.80684**, wallets **6498–7028 B** on California).
- **Mechanical Signature:** On `cal_rand_f0`, `beam3` inspects `(5,7)` with `honest_gain=37.27` vs `score=-124.78` and safe tier suppresses `20–24` candidates/fold at `gain=0.0` (matching R113/R114 canary behavior). On `anneal` and `aps`, pair layers stay silent so `v31` matches `v8` bit-for-bit.

### 4. `ood_fortress` — **Distribution-Shift & Tail-Guard Mode**
- **Origin:** R97 (`H1H2` 5% mass floor, `hmin_frac=0.05`) + R113 Dykstra centering and box-gated safe tier (`include_safe_tier=True`, `edge_freeze=True`).
- **Role:** Prioritizes minimum sample mass per shelf (`max(25, 0.05 * sum(H))`) and edge-cell protection under spatial/covariate shift.
- **Mechanical Signature:** The 5% mass floor aggressively merges low-count tail bins, yielding ultra-small cards (**878–1276 B** across all 16 folds) at the expected IID trade-off documented in R97 (`cal_rand` mean R² **0.75249**, `anneal` Toof mean **0.25479**).

### 5. `sensor_rare` — **Rare-Event & Missingness-Coordination Detector**
- **Origin:** R107/R111 (`potts_objective="qfix_dev"`), R112 (`include_qmiss=True`), R82 (`oof_merge=True`).
- **Role:** Designed for sensor/industrial datasets with informative missingness and rare positive events. Uses fixed-curvature deviance splitting, OOF bin merging, and post-fit `QMissAtom` coordination probing.
- **Mechanical Signature:** `aps` wallets shrink to **1983–2154 B** (`~1.6x` smaller than `v8` or `~0.73x` depending on panel fold, with `oof_merge` performing `8–475` merges/fold). `QMissAtom` remains structurally silent on canonical panels where main effects absorb linear missingness (0/16 admissions), while firing decisively on threshold-NaN synthetic controls (`gain 49.93 > barrier 4.70`).

### 6. `scorecard_1d` — **Bank-Style 1D Scorecard (Pairless)**
- **Origin:** R71/R82 1D additive scorecard (`enable_pairs=False`, `oof_merge=True`, `star_types=("g4a","g4b","g4c","g4d")`).
- **Role:** Strictly 1D main effects + 1D projected rays (no 2D `p1`/`m2` pair grids), suitable for regulated credit scoring where full additive transparency and classic PDO points tables (`to_scorecard_df(pdo=20)`) are required.
- **Mechanical Signature:** Ledger evaluates 80 candidates (vs 120 when pairs are enabled), admits zero pair atoms (`p1`/`m2`), produces lighter `aps` cards (**2125–2320 B**), and matches `sensor_rare` on `anneal` where 2D pairs are inactive in multiclass.

### 7. `unregularized_32` — **Unfused 32-Bin Baseline (`kappa = 0`)**
- **Origin:** R71/R73 `card_kappa0` (`extra={"kappa_zero": True}`, `wallet_bytes=16384`, `max_wallet_bytes=16384`).
- **Role:** Reference unfused 32-quantile additive model (`kappa_mult = 0.0`), preserving all 32 quantile shelves per feature without L0 merging.
- **Mechanical Signature:** On `aps`, cards expand to **15790–15978 B** (confirming the R73/R75 12–16 KB floor where `< 1.4 KB` is arithmetically impossible without L0 merging); on `california`, wallets are **2568–3096 B**; deploy `anneal` wallet is **2263 B** (vs `1390 B` for `v8`).
