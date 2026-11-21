# Paper 2 · Round 2 Decisive Experiment Report (AVE)

- **Script**: `train_ave_r2.py` (untouched Round 1 baseline preserved in `train_ave.py` & `results/ave_pilot_results.json`)
- **Raw Output**: `results/ave_r2_results.json`
- **Protocol**: Official AVE splits (`3,339` train / `402` val / `402` test), 3 seeds (`42, 43, 44`), 25 epochs, `batch_size=128`, Apple Silicon MPS, single-process sequential execution.
- **Fixes Applied**: `R1` (hard-negative-only BG loss, `w_b=0.15`), `R2` (label-grounded CE + teacher-gated per-segment KL), `R3` (`clip_grad_norm=5.0`, 5-epoch clean warmup + 5-epoch linear ramp), `R4` (**PCAR** paired counterfactual attention-logit ranking + per-head **CAL**), every-epoch validation from epoch 5 with unified constrained-first checkpoint selection (`SwV A-Ret ≥ 0.70`, `SwA V-Ret ≥ 0.65`, `AV-Rej ≥ 0.90`).

---

## 1. Main Comparison Table (3-Seed Mean ± Std, Official AVE Test Set)

| Variant | Clean Acc (%) | Clean F1 (%) | AV-Ev / BG (%) | Ev→BG Err ↓ | Clean A / V (%) | SwV: A-Ret ↑ | SwV: V-Drag ↓ | SwV: AV-Rej ↑ | SwA: V-Ret ↑ | SwA: A-Drag ↓ | SwA: AV-Rej ↑ | $\alpha_{\text{null}}$ (Ev / BG / SwV / SwA) | $\alpha$-AUROC ↑ | FR (Ev / Sw) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0` (`baseline_xattn`)** | **73.08 ± 1.40** | **75.61 ± 1.19** | 78.9 / 46.0 | **10.0%** | 78.6 / 78.1 | 37.6 ± 1.0 | 18.9 ± 0.6 | 33.2 ± 4.8 | 31.2 ± 1.1 | 27.3 ± 0.9 | 30.2 ± 3.0 | — | 0.500 | 0.68 / 0.67 (1.00×) |
| **`A1` (`baseline_cfswap_v2`)** | 67.41 ± 1.01 | 71.60 ± 0.47 | 69.3 / **58.5** | 22.9% | 75.5 / 70.1 | 73.1 ± 1.3 | **1.1 ± 0.1** | 93.5 ± 1.7 | 66.8 ± 1.3 | 1.9 ± 0.5 | 90.8 ± 2.3 | — | 0.500 | 0.39 / 0.38 (1.02×) |
| **`A2` (`nullslot_cfswap_abs`)** | 67.12 ± 1.22 | 71.46 ± 0.57 | 70.6 / 51.2 | 21.3% | **76.3 / 71.2** | **73.7 ± 1.0** | 1.3 ± 0.3 | 91.7 ± 2.2 | **66.9 ± 1.1** | 1.8 ± 0.3 | 90.3 ± 3.9 | `0.625 / 0.706 / 0.821 / 0.840` | 0.823 | 0.28 / 0.14 (1.97×) |
| **`A3` (`nullslot_cfswap_pcar`)** | 67.06 ± 0.88 | 71.21 ± 1.39 | 70.2 / 52.6 | 21.8% | 75.5 / 69.8 | 73.6 ± 0.9 | 1.4 ± 0.5 | 91.2 ± 1.7 | 66.3 ± 0.8 | 2.5 ± 0.3 | 89.0 ± 3.1 | **`0.129 / 0.232 / 0.757 / 0.723`** | **0.931** | **0.49 / 0.16 (3.05×)** |
| **`B3` (`sigmoid_gate_cfswap_v2`)** | 67.89 ± 0.78 | 71.86 ± 1.01 | 70.6 / 55.2 | 21.9% | 76.1 / 71.2 | 72.9 ± 0.5 | 1.5 ± 0.3 | **93.7 ± 0.7** | 66.9 ± 1.0 | **1.4 ± 0.3** | **93.0 ± 0.2** | `0.427 / 0.499 / 0.692 / 0.693` | 0.842 | 0.41 / 0.24 (1.71×) |

*(Notes: $\alpha_{\text{null}}$ Ev/BG are averaged over directional `a2v` and `v2a`; $\alpha$-AUROC is the average of `alpha_auroc_swap_v` and `alpha_auroc_swap_a`; FR = effective fusion ratio $\|W_o\,\text{ctx}\|/\|h\|$.)*

---

## 2. Diagnostic Analysis: Two Clear Findings

### Finding 1 — PCAR Mechanism Worked Exactly as Predicted (Strong Internal Routing Win)
1. **Global null drift (R4) is completely eliminated**: Moving from absolute target (`A2`) to **PCAR + CAL (`A3`)** pulled clean-event abstention down from **`0.625 → 0.129`** while maintaining high swapped-distractor abstention (**`0.757 / 0.723`**), tripling the clean-vs-swap separation margin (`+0.205 → +0.611`) and boosting **$\alpha$-AUROC from `0.823 → 0.931` (+10.8 pp)**.
2. **W1 Rebuttal Confirmed (`A3` vs `B3` query-only sigmoid gate)**: Because a query-only gate $g(q)$ cannot see opposite-modality keys in Layer 1, `B3` stalls at $\alpha_{\text{null}}^{\text{ev}} = 0.427$, $\alpha$-AUROC = **`0.842`** (`-8.9 pp` behind `A3`), and an effective fusion ratio contrast of only **`1.71×`** (`0.41 / 0.24`). By contrast, `A3` achieves a **`3.05×`** functional fusion contrast (**`0.49 / 0.16`**), proving Null-Slot + PCAR genuinely performs key-aware selective abstention rather than indiscriminate pathway shrinkage (`A1`: `0.39 / 0.38 = 1.02×`).

### Finding 2 — Honest Go/No-Go Failure on AVE Clean Accuracy & Downstream Parity (`A3 ≈ A1 ≈ B3`)
1. **Clean Acc did NOT recover to ≥ 73%**: Across ALL four CF variants (`A1`, `A2`, `A3`, `B3`), Clean Acc stayed at **`67.06% – 67.89%`** (only `+0.6 pp` over Round 1's `66.45–66.81%`, still **`-5.2` to `-6.0 pp`** below `A0`'s `73.08%`).
   - **Exact cause (visible in `Ev→BG Err`)**: Even with `w_b = 0.15` and hard-negative masking, enforcing `AV-Rej ≥ 90%` on swapped pairs pushed `Ev→BG Err` from **`10.0% → 21.3–22.9%`** (more than doubling false rejections of true clean AV events, while `bg_unaligned_acc` rose from `46.0% → 51.2–58.5%`). On AVE, training the joint AV head to reject cross-video swaps creates an inherent precision-recall trade-off on the BG class under fixed argmax decoding (without inference-time BG logit calibration or decoupled alignment scoring).
2. **Downstream Accuracy Parity (`A3 ≈ A1 ≈ B3`)**: On AVE supervised metrics, `A3` (`67.06%` clean, `73.6% / 66.3%` swap retention, `91.2% / 89.0%` AV rejection) did **not** beat `A1` (`67.41%` clean, `73.1% / 66.8%` retention, `93.5% / 90.8%` rejection) or `B3` (`67.89%` clean, `72.9% / 66.9%` retention, `93.7% / 93.0%` rejection). Because AVE provides dense supervised segment labels, any architecture trained with `CE(logits_swap, y)` can achieve ~73%/66% unimodal retention simply by shrinking cross-modal fusion globally (`A1` `FR=0.39`).

---

## 3. Go / No-Go Verdict Against `ROUND2_DESIGN.md` Criteria

| Criterion (`ROUND2_DESIGN.md` §4) | Target | Observed | Pass? |
| :--- | :---: | :---: | :---: |
| `A1` Clean Acc recovery vs Round 1 (`66.81%`) | ≥ +4.0 pp (≥ 71.0%) | +0.60 pp (`67.41%`) | **FAIL** |
| `A3` Clean Acc | ≥ 73.0% (ideal ≥ 73.5%) | `67.06%` (`-6.02 pp` vs `A0`) | **FAIL** |
| `A3` − `A1` Clean Acc & Retention advantage | Clean ≥ +0.5 pp & higher Ret | Clean `-0.35 pp`, Ret `+0.5 / -0.5 pp` | **FAIL** |
| `A3` PCAR internal routing (`α_ev ≤ 0.3`, AUROC, W1 vs `B3`) | `α_ev ≤ 0.30`, `A3 > B3` | `α_ev = 0.129`, **AUROC `0.931`** (`+8.9 pp` vs `B3`), **FR `3.05×`** | **PASS** |

### Strategic Recommendation
Per the pre-registered decision rule in `ROUND2_DESIGN.md` (§4, L251):
- **Do NOT proceed with AVE as the primary accuracy showcase for Paper 2** under the current joint-head BG-rejection objective (`loss_bg` directly trades `-5.5 pp` clean accuracy for `91%` swap rejection, and `A3 ≈ A1 ≈ B3` on AVE downstream accuracy).
- **Two viable paths if the team still wants to invest in Paper 2**:
  1. **Pivot primary benchmark to LLP / AVVP** (`download_llp.sh` + `train_llp.py`), where Audio-only and Visual-only events are *real native test classes* (no artificial `loss_bg` trade-off, and indiscriminate fusion shrinkage like `A1` hurts joint AV F1 while `A3`'s `3.05×` selective gating + `0.931` AUROC can directly improve native Audio/Visual/AV F1 and zero-shot unaligned-segment AUROC).
  2. **If staying on AVE**: drop `loss_bg` from the joint classification head (preserving `73%+` Clean Acc) and instead use `A3`'s **$\alpha_{\text{null}}$ (`AUROC = 0.931`) as a decoupled alignment/abstention detector** (e.g., inference-time BG gating or calibrated threshold) — turning Paper 2 into a clean *misalignment-aware routing & detection* story rather than joint-head BG-bias training.
- Otherwise, **pause / deprioritize Paper 2** in favor of higher-margin ICME 2027 candidates.
