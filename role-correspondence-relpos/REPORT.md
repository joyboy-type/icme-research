# Paper 1 Experimental Report: 2D-RelPos Cross-Patch Context + Orthogonal Semantic-Role Subspace Decomposition

## 1. Protocol & Zero-Deviation Baseline Verification

All experiments strictly follow the **v1.1-coordinate-correction** evaluation protocol from `soco_grid_corrected`:
- **Coordinate Grid**: Physical patch centers `((i + 0.5) / 20)` for the `20×20 = 400` dense target grid; exact bilinear interpolation with `align_corners=False` (`padding_mode='zeros'`) for source and target keypoint sampling (verified `max_abs_diff = 0.0` against `F.grid_sample`).
- **Fixed Category-Disjoint Splits (192 unique images, 740 pairs)**:
  - `train`: 108 images / 485 pairs (6 categories: `chair`, `armchair`, `washer`, `car`, `bicycle`, `tiger`)
  - `dev`: 36 images / 109 pairs (3 unseen categories: `office_chair`, `bus`, `leopard`) — **used exclusively for hyperparameter & `alpha` selection**
  - `confirm`: 48 images / 146 pairs (4 unseen categories: `bench`, `dishwasher`, `folding_chair`, `train`) — **strictly held out until all hyperparameters were frozen**
- **Features & Seeds**: Cached frozen DINOv2 ViT-S/14 (`384 × 20 × 20`) features from `soco_role_pilot_20261121/features.pt`, evaluated across 3 random seeds `[17, 29, 43]`.
- **Zero-Deviation Check**:
  - `frozen` on `confirm`: `SOC PCK@0.1 = 44.4117%`, `CC same-query PCK@0.1 = 65.7940%`, `CC all-query PCK@0.1 = 65.3055%`, `dense_wrong_role = 21.3823%` — **bit-exact match** with `soco_grid_corrected/analysis.log`.
  - `task_conditioned_single` on `confirm`: 3-seed mean `SOC = 42.8241%` (seeds `42.53 / 44.94 / 41.01`), `CC same-query = 59.3607%` — **bit-exact match** with `soco_grid_corrected/analysis.log`.

---

## 2. Main Confirmation Results (4 Unseen Categories, 48 Images, 146 Pairs)

| Model | Params | Train Time (s/seed) | Latency (ms/pair) | SOC PCK@0.1 (↑) | CC Same-Query (↑) | CC All-Query (↑) | Wrong-Role Error (↓) | Oracle Cand. SOC (↑) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **(1) Frozen DINOv2-S** | 0 | 0.0s | 0.47 | 44.41 | 65.79 | 65.31 | 21.38 | 53.01 |
| **(2) Task-Conditioned Single MLP** | 73,984 | 1.2s | 0.86 | 42.82 ± 1.62 | 59.36 ± 1.48 | 57.74 ± 1.48 | 14.96 ± 0.46 | 57.23 ± 0.91 |
| **(3) 2-Layer 2D-RelPos Adapter (Single / Mixed)** | 364,552 | 49.4s | 12.80 | 42.96 ± 0.67 | 60.94 ± 1.30 | 60.11 ± 1.23 | 17.98 ± 0.67 | 51.25 ± 0.57 |
| **(4a) Ablation: 2D-RelPos + Orth Role (Frozen `z_sem`)** | 364,936 | 54.8s | 16.38 | 48.53 ± 1.08 | 65.59 ± 0.00* | 65.13 ± 0.00* | 18.49 ± 0.70 | 57.15 ± 0.30 |
| **(4b) 2D-RelPos + Orthogonal Sem-Role Subspace (Full)** | **414,088** | **75.9s** | **15.39** | **50.75 ± 1.00** | **73.23 ± 0.51** | **72.81 ± 0.44** | **18.08 ± 0.48** | **59.01 ± 0.91** |

*\*Note on (4a): `relpos_orth_frozen_sem` normalizes patch vectors before bilinear interpolation (`dense_norm`), whereas `frozen` interpolates raw dense features then normalizes, causing a negligible -0.20pp (95% CI `[-0.83, +0.31]`) difference on `z_sem`.*

**Per-Seed Stability of Full Model (4b)**:
- Seed 17: `SOC = 49.70%`, `CC_sq = 73.17%`, `WR = 17.73%`
- Seed 29: `SOC = 52.09%`, `CC_sq = 72.64%`, `WR = 17.76%`
- Seed 43: `SOC = 50.45%`, `CC_sq = 73.88%`, `WR = 18.76%`
Every single seed outperforms Frozen DINOv2-S (`44.41%`), Task-Conditioned MLP (`42.82%`), and RelPos-Single (`42.96%`) by > +5.2pp on SOC and > +6.8pp on CC.

---

## 3. Paired Shared-Image Vertex Bootstrap (4,000 Replicates, 95% CI)

Using the exact within-category shared-image vertex bootstrap (`source * target` multiplicity weighting, seed `20261121`):

| Comparison | Δ SOC PCK@0.1 (95% CI) | Δ CC Same-Query (95% CI) | Δ CC All-Query (95% CI) | Δ Wrong-Role (95% CI) | Categories Positive (SOC) |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **(4b) Full vs (1) Frozen** | **+6.34pp `[+1.20, +11.51]`** | **+7.43pp `[+3.81, +11.19]`** | **+7.51pp `[+3.95, +11.16]`** | **-3.30pp `[-6.46, -0.27]`** | **3 / 4** |
| **(4b) Full vs (2) MLP** | **+7.92pp `[+4.20, +11.89]`** | **+13.87pp `[+9.53, +18.52]`** | **+15.07pp `[+11.22, +19.35]`** | +3.12pp `[+0.65, +5.86]` | **4 / 4** |
| **(4b) Full vs (3) RelPos-Single** | **+7.79pp `[+4.62, +10.96]`** | **+12.29pp `[+8.34, +17.06]`** | **+12.70pp `[+8.88, +17.22]`** | +0.10pp `[-2.14, +2.32]` | **4 / 4** |
| **(4b) Full vs (4a) Frozen-`z_sem`** | +2.21pp `[-0.66, +5.01]` | **+7.64pp `[+4.19, +11.29]`** | **+7.69pp `[+4.32, +11.15]`** | -0.40pp `[-2.17, +1.39]` | **3 / 4** |
| **(4a) Frozen-`z_sem` vs (1) Frozen** | +4.12pp `[-0.49, +8.70]` | -0.20pp `[-0.83, +0.31]` | -0.18pp `[-0.76, +0.29]` | **-2.90pp `[-5.68, -0.11]`** | **3 / 4** |
| **(3) RelPos-Single vs (2) MLP** | +0.14pp `[-3.68, +4.03]` | +1.58pp `[-3.09, +6.12]` | +2.37pp `[-2.02, +6.82]` | +3.02pp `[+0.33, +5.89]` | 2 / 4 |

**Gate Evaluation** (Original pilot gate: `ΔSOC ≥ +3pp`, `ΔCC_samequery ≥ -1pp`, `≥ 3/4` unseen categories positive, bootstrap 95% CI lower bound `> 0`):
- **vs Frozen DINOv2-S**: **PASS** (`+6.34pp ≥ 3`, `+7.43pp ≥ -1`, `3/4` cats, `CI_lo = +1.20 > 0`)
- **vs Task-Conditioned MLP**: **PASS** (`+7.92pp ≥ 3`, `+13.87pp ≥ -1`, `4/4` cats, `CI_lo = +4.20 > 0`)
- **vs RelPos-Single**: **PASS** (`+7.79pp ≥ 3`, `+12.29pp ≥ -1`, `4/4` cats, `CI_lo = +4.62 > 0`)

---

## 4. Per-Category Breakdown on Confirm Split

### SOC PCK@0.1 (%)
| Category | (1) Frozen | (2) MLP | (3) RelPos-Single | (4a) Frozen-`z_sem` | (4b) Full | Δ (4b vs Frozen) | Δ (4b vs MLP) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bench` | 35.54 | 34.24 | 31.38 | 37.94 | **41.00** | **+5.46** | **+6.75** |
| `dishwasher` | 48.06 | 47.41 | 50.86 | 57.03 | **60.08** | **+12.02** | **+12.67** |
| `folding_chair` | 48.90 | 55.42 | 54.12 | 56.38 | **59.58** | **+10.68** | **+4.16** |
| `train` | **45.15** | 34.22 | 35.48 | 42.78 | 42.34 | -2.81 | **+8.11** |
| **Macro Mean** | 44.41 | 42.82 | 42.96 | 48.53 | **50.75** | **+6.34** | **+7.92** |

### CC Same-Query PCK@0.1 (%)
| Category | (1) Frozen | (2) MLP | (3) RelPos-Single | (4a) Frozen-`z_sem` | (4b) Full | Δ (4b vs Frozen) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bench` | 62.56 | 54.20 | 54.39 | 62.64 | **72.40** | **+9.84** |
| `dishwasher` | 57.88 | 54.11 | 60.40 | 57.88 | **71.67** | **+13.79** |
| `folding_chair` | 81.50 | 77.81 | 81.68 | 81.50 | **87.21** | **+5.71** |
| `train` | 61.24 | 51.31 | 47.29 | 60.35 | **61.64** | **+0.40** |
| **Macro Mean** | 65.79 | 59.36 | 60.94 | 65.59 | **73.23** | **+7.43** |

---

## 5. Mechanism Attribution: Cross-Patch Context vs. Orthogonal Subspace Decoupling

The 5-way controlled comparison cleanly disentangles where the gain comes from:

1. **Cross-patch 2D-RelPos context ALONE is insufficient (`(3) RelPos-Single`: 42.96% SOC, 60.94% CC)**:
   - Without subspace decoupling, mixing SOC role hard negatives (`cc & ~soc`) and CC positives in a single 384-D space causes gradient interference that degrades concept localization by **-4.85pp** relative to Frozen DINOv2-S, canceling out any spatial-context benefit (`+0.14pp` vs MLP).
2. **Orthogonal Role Subspace + 2D-RelPos Context unlocks +4.12pp SOC purely from role disambiguation (`(4a) Frozen z_sem`: 48.53% SOC)**:
   - Even with `z_sem` completely frozen to DINOv2-S, adding the 2-layer 2D-RelPos orthogonal role subspace `z_role` reduces `dense_wrong_role` by **-2.90pp** (`95% CI [-5.68, -0.11]`) and lifts SOC from `44.41%` to `48.53%` (`+4.12pp`).
3. **Jointly adapting `z_sem` with dense cosine anchoring (`w_anchor=10.0`) + CC supervision adds another +2.21pp SOC and +7.43pp CC (`(4b) Full`: 50.75% SOC, 73.23% CC)**:
   - Because `z_role` absorbs the role-hard-negative gradients in the orthogonal complement of `z_sem`, `z_sem` is free to use the 2D-RelPos cross-patch context purely to sharpen concept localization and suppress background distractors (via dense negative patches + cosine anchoring), lifting CC from `65.79% -> 73.23%` (`+7.43pp`) and pushing SOC to `50.75%` (`+6.34pp` vs Frozen, `+7.79pp` vs RelPos-Single).
4. **Conclusion**: The breakthrough is **strictly synergistic** — neither MLP dual-head separation (`45.09%` SOC in prior pilot, `-4.07pp` CC drop) nor single-head 2D-RelPos (`42.96%` SOC, `-4.85pp` CC drop) passes the gate. Only **2D-RelPos Cross-Patch Context + Orthogonal Semantic-Role Subspace Decomposition** simultaneously delivers a statistically significant SOC gain (`+6.34pp`) and a large CC gain (`+7.43pp`) on unseen categories.

---

## 6. Diagnostics & Observed Limitations

1. **`train` category diagnostic (-2.81pp SOC vs Frozen, +8.11pp vs MLP)**:
   - On `train`, `(4b) Full` improves `CC_samequery` (`61.24% -> 61.64%`) and `oracle_candidate_soc` (`61.35% -> 64.01%`, **+2.66pp**), and reduces `dense_wrong_role` (`16.09% -> 12.70%`, **-3.39pp**). However, `cc_rolebranch_pck` drops (`61.24% -> 55.04%`), indicating that on elongated, repetitive-structure objects under strong perspective foreshortening, a fixed `alpha = 0.5` occasionally pulls dense argmax off the target concept onto background/repetitive patches. A query-adaptive `alpha` (e.g., gated by semantic entropy / ambiguity) is a natural next improvement.
2. **Inference latency**: `15.39 ms/pair` on MPS (for 2 images through the 2-layer 400-token RelPos adapter + dense matching), lightweight enough for real-time correspondence (~65 pairs/sec) with only `414K` trainable parameters (~1.9% of DINOv2-S).
3. **Go / No-Go Verdict**: **CLEAR GO** to advance to full-scale SOCO / SPair-71k / multi-backbone (DINOv2-B/L, SD+DINO) evaluation.

---

## 7. Artifacts

- Code: `data_utils.py`, `models.py`, `train_eval.py`, `dev_search.py`, `run_confirm.py`
- Structured summary: `summary.json`
- Dev sweep log: `dev_search_results.json`
- Per-query predictions & weights: `outputs/per_query_confirm_*.csv`, `outputs/weights/*.pt`
