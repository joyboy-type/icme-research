"""AVE Round 2 Decisive Experiments: Fixing R1+R2+R3 + PCAR (R4) + W1 Rebuttal (B3).

Runs 5 decisive groups sequentially on MPS (3 seeds: 42, 43, 44; 25 epochs; batch_size=128):
  - A0 (baseline_xattn)         : Standard Cross-Attention baseline
  - A1 (baseline_cfswap_v2)     : Standard Cross-Attention + Fixed CF-Swap (R1+R2+R3 fixed)
  - A2 (nullslot_cfswap_abs)    : Null-Slot + Fixed CF-Swap + Old Absolute Null Target
  - A3 (nullslot_cfswap_pcar)   : Null-Slot + Fixed CF-Swap + PCAR + CAL (Main Method)
  - B3 (sigmoid_gate_cfswap_v2) : Query-Only Sigmoid Gate + Fixed CF-Swap + PCAR + CAL (W1 Rebuttal)

Saves incrementally after each group to `results/ave_r2_results.json`.
"""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.nullslot_av import AVNullSlotModel

BG_CLASS = 28
NUM_CLASSES = 29

R2_CFG = {
    "seeds": [42, 43, 44],
    "epochs": 25,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.2,
    "cf_warmup": 5,
    "cf_ramp": 5,
    "rank_margin": 2.0,
    "tau_ev": 0.3,
    "tau_sw": 0.7,
    "w_rank": 0.3,
    "w_cal": 0.1,
    "w_null_abs": 0.3,
    "inv_kl_beta": 0.5,
    "cf_invariance_weight": 0.5,
    "cf_bg_weight": 0.15,
    "clip": 5.0,
}

VARIANTS = [
    ("A0_baseline_xattn", {"attn_mode": "standard", "use_cf": False, "null_loss": "none"}),
    ("A1_baseline_cfswap_v2", {"attn_mode": "standard", "use_cf": True, "null_loss": "none"}),
    ("A2_nullslot_cfswap_abs", {"attn_mode": "nullslot", "use_cf": True, "null_loss": "abs"}),
    ("A3_nullslot_cfswap_pcar", {"attn_mode": "nullslot", "use_cf": True, "null_loss": "pcar"}),
    ("B3_sigmoid_gate_cfswap_v2", {"attn_mode": "sigmoid_gate", "use_cf": True, "null_loss": "pcar"}),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_first_h5_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]
        return np.array(f[key])


def compute_auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """Exact rank-based AUROC where pos_scores (swapped) should exceed neg_scores (clean)."""
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return 0.5
    scores = np.concatenate([neg_scores, pos_scores])
    labels = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    # Average ranks for ties
    n = len(scores)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1
    n_pos = float(len(pos_scores))
    n_neg = float(len(neg_scores))
    pos_rank_sum = ranks[sorted_labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


class AVEDataset(Dataset):
    def __init__(
        self,
        audio: np.ndarray,
        visual: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.audio = audio[indices].astype(np.float32)
        self.visual = visual[indices].astype(np.float32)
        raw_y = labels[indices]
        if raw_y.ndim == 3:
            self.labels = np.argmax(raw_y, axis=-1).astype(np.int64)
        else:
            self.labels = raw_y.astype(np.int64)

        self.video_cls = np.full(len(self.labels), BG_CLASS, dtype=np.int64)
        for i in range(len(self.labels)):
            non_bg = self.labels[i][self.labels[i] != BG_CLASS]
            if len(non_bg) > 0:
                vals, counts = np.unique(non_bg, return_counts=True)
                self.video_cls[i] = vals[np.argmax(counts)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.audio[idx]),
            torch.from_numpy(self.visual[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.tensor(self.video_cls[idx], dtype=torch.long),
        )


def make_cross_class_perm(video_cls: torch.Tensor) -> torch.Tensor:
    B = video_cls.shape[0]
    cls_np = video_cls.detach().cpu().numpy()
    perm = np.arange(B)
    for i in range(B):
        diff = np.where(cls_np != cls_np[i])[0]
        if len(diff) > 0:
            perm[i] = diff[np.random.randint(len(diff))]
        else:
            perm[i] = (i + 1) % B
    return torch.from_numpy(perm).to(video_cls.device)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fp + fn == 0:
            continue
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate_ave(
    model: AVNullSlotModel,
    dataset: AVEDataset,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_y, all_pred_av, all_pred_a, all_pred_v = [], [], [], []
    clean_a2v_null, clean_v2a_null = [], []
    clean_fr_a2v, clean_fr_v2a = [], []

    for audio, visual, y, _ in loader:
        audio, visual = audio.to(device), visual.to(device)
        out = model(audio, visual)
        all_y.append(y.numpy())
        all_pred_av.append(out["logits_av"].argmax(dim=-1).cpu().numpy())
        all_pred_a.append(out["logits_a"].argmax(dim=-1).cpu().numpy())
        all_pred_v.append(out["logits_v"].argmax(dim=-1).cpu().numpy())

        fr_a = torch.stack([x["fusion_ratio"] for x in out["aux_a2v"]], dim=0).mean(dim=0).cpu().numpy()
        fr_v = torch.stack([x["fusion_ratio"] for x in out["aux_v2a"]], dim=0).mean(dim=0).cpu().numpy()
        clean_fr_a2v.append(fr_a)
        clean_fr_v2a.append(fr_v)

        if out["alpha_null_a2v"] is not None:
            clean_a2v_null.append(out["alpha_null_a2v"].cpu().numpy())
            clean_v2a_null.append(out["alpha_null_v2a"].cpu().numpy())

    y_all = np.concatenate(all_y, axis=0)  # [N, 10]
    pred_av_all = np.concatenate(all_pred_av, axis=0)
    pred_a_all = np.concatenate(all_pred_a, axis=0)
    pred_v_all = np.concatenate(all_pred_v, axis=0)
    fr_a2v_all = np.concatenate(clean_fr_a2v, axis=0)
    fr_v2a_all = np.concatenate(clean_fr_v2a, axis=0)

    y_flat = y_all.reshape(-1)
    pred_av_flat = pred_av_all.reshape(-1)
    pred_a_flat = pred_a_all.reshape(-1)
    pred_v_flat = pred_v_all.reshape(-1)

    event_mask = y_flat != BG_CLASS
    bg_mask = y_flat == BG_CLASS
    ev_to_bg_err = float(np.mean(pred_av_flat[event_mask] == BG_CLASS))

    # Deterministic cross-class counterfactual stress test
    N = len(dataset)
    cls_np = dataset.video_cls
    swap_idx = np.zeros(N, dtype=np.int64)
    for i in range(N):
        candidates = np.where((cls_np != cls_np[i]) & (cls_np != BG_CLASS))[0]
        if len(candidates) > 0:
            swap_idx[i] = candidates[(i * 7 + 13) % len(candidates)]
        else:
            swap_idx[i] = (i + 1) % N

    audio_all = torch.from_numpy(dataset.audio).to(device)
    visual_all = torch.from_numpy(dataset.visual).to(device)
    y_swap = dataset.labels[swap_idx]

    swap_v_pred_a, swap_v_pred_av, null_swap_v, fr_swap_v = [], [], [], []
    swap_a_pred_v, swap_a_pred_av, null_swap_a, fr_swap_a = [], [], [], []

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        a_i = audio_all[start:end]
        v_i = visual_all[start:end]
        a_j = audio_all[swap_idx[start:end]]
        v_j = visual_all[swap_idx[start:end]]

        out_sv = model(a_i, v_j)
        swap_v_pred_a.append(out_sv["logits_a"].argmax(dim=-1).cpu().numpy())
        swap_v_pred_av.append(out_sv["logits_av"].argmax(dim=-1).cpu().numpy())
        fr_swap_v.append(torch.stack([x["fusion_ratio"] for x in out_sv["aux_a2v"]], dim=0).mean(dim=0).cpu().numpy())
        if out_sv["alpha_null_a2v"] is not None:
            null_swap_v.append(out_sv["alpha_null_a2v"].cpu().numpy())

        out_sa = model(a_j, v_i)
        swap_a_pred_v.append(out_sa["logits_v"].argmax(dim=-1).cpu().numpy())
        swap_a_pred_av.append(out_sa["logits_av"].argmax(dim=-1).cpu().numpy())
        fr_swap_a.append(torch.stack([x["fusion_ratio"] for x in out_sa["aux_v2a"]], dim=0).mean(dim=0).cpu().numpy())
        if out_sa["alpha_null_v2a"] is not None:
            null_swap_a.append(out_sa["alpha_null_v2a"].cpu().numpy())

    sv_pred_a = np.concatenate(swap_v_pred_a, axis=0)
    sv_pred_av = np.concatenate(swap_v_pred_av, axis=0)
    sa_pred_v = np.concatenate(swap_a_pred_v, axis=0)
    sa_pred_av = np.concatenate(swap_a_pred_av, axis=0)
    fr_sv_all = np.concatenate(fr_swap_v, axis=0)
    fr_sa_all = np.concatenate(fr_swap_a, axis=0)

    active_diff_mask = (y_all != BG_CLASS) & (y_swap != BG_CLASS) & (y_all != y_swap)
    ev_2d = y_all != BG_CLASS
    bg_2d = y_all == BG_CLASS

    if clean_a2v_null:
        ca2v = np.concatenate(clean_a2v_null, axis=0)
        cv2a = np.concatenate(clean_v2a_null, axis=0)
        nsv = np.concatenate(null_swap_v, axis=0)
        nsa = np.concatenate(null_swap_a, axis=0)
        null_a2v_ev = float(np.mean(ca2v[ev_2d]))
        null_v2a_ev = float(np.mean(cv2a[ev_2d]))
        null_a2v_bg = float(np.mean(ca2v[bg_2d]))
        null_v2a_bg = float(np.mean(cv2a[bg_2d]))
        null_sv = float(np.mean(nsv[active_diff_mask]))
        null_sa = float(np.mean(nsa[active_diff_mask]))
        auroc_sv = compute_auroc(ca2v[active_diff_mask], nsv[active_diff_mask])
        auroc_sa = compute_auroc(cv2a[active_diff_mask], nsa[active_diff_mask])
    else:
        null_a2v_ev = null_v2a_ev = null_a2v_bg = null_v2a_bg = 0.0
        null_sv = null_sa = 0.0
        auroc_sv = auroc_sa = 0.5

    return {
        "clean_acc": float(np.mean(pred_av_flat == y_flat)),
        "clean_macro_f1": macro_f1(y_flat, pred_av_flat),
        "av_event_acc": float(np.mean(pred_av_flat[event_mask] == y_flat[event_mask])),
        "bg_unaligned_acc": float(np.mean(pred_av_flat[bg_mask] == y_flat[bg_mask])),
        "ev_to_bg_err": ev_to_bg_err,
        "clean_audio_event_acc": float(np.mean(pred_a_flat[event_mask] == y_flat[event_mask])),
        "clean_visual_event_acc": float(np.mean(pred_v_flat[event_mask] == y_flat[event_mask])),
        "swap_v_audio_retain_acc": float(np.mean(sv_pred_a[active_diff_mask] == y_all[active_diff_mask])),
        "swap_v_visual_drag_rate": float(np.mean(sv_pred_a[active_diff_mask] == y_swap[active_diff_mask])),
        "swap_v_av_reject_rate": float(np.mean(sv_pred_av[active_diff_mask] == BG_CLASS)),
        "swap_a_visual_retain_acc": float(np.mean(sa_pred_v[active_diff_mask] == y_all[active_diff_mask])),
        "swap_a_audio_drag_rate": float(np.mean(sa_pred_v[active_diff_mask] == y_swap[active_diff_mask])),
        "swap_a_av_reject_rate": float(np.mean(sa_pred_av[active_diff_mask] == BG_CLASS)),
        "null_a2v_ev": null_a2v_ev,
        "null_v2a_ev": null_v2a_ev,
        "null_a2v_bg": null_a2v_bg,
        "null_v2a_bg": null_v2a_bg,
        "null_a2v_swap_v": null_sv,
        "null_v2a_swap_a": null_sa,
        "alpha_auroc_swap_v": auroc_sv,
        "alpha_auroc_swap_a": auroc_sa,
        "fusion_ratio_ev": float(0.5 * (np.mean(fr_a2v_all[ev_2d]) + np.mean(fr_v2a_all[ev_2d]))),
        "fusion_ratio_swap": float(0.5 * (np.mean(fr_sv_all[active_diff_mask]) + np.mean(fr_sa_all[active_diff_mask]))),
    }


def train_one_run(
    spec: dict,
    seed: int,
    train_ds: AVEDataset,
    val_ds: AVEDataset,
    test_ds: AVEDataset,
    cfg: dict,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(seed)
    attn_mode = spec["attn_mode"]
    use_cf = spec["use_cf"]
    null_loss_mode = spec["null_loss"]

    model = AVNullSlotModel(
        d_audio=train_ds.audio.shape[-1],
        d_visual=train_ds.visual.shape[-1],
        num_classes=NUM_CLASSES,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        attn_mode=attn_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    best_score = -1.0
    best_val_clean = -1.0
    best_state = None
    grad_norms = []

    for epoch in range(cfg["epochs"]):
        model.train()
        lam = (
            0.0
            if epoch < cfg["cf_warmup"]
            else min(1.0, (epoch - cfg["cf_warmup"] + 1) / float(cfg["cf_ramp"]))
        )

        for audio, visual, y, video_cls in train_loader:
            audio = audio.to(device)
            visual = visual.to(device)
            y = y.to(device)
            video_cls = video_cls.to(device)

            out = model(audio, visual)
            loss_av = F.cross_entropy(out["logits_av"].reshape(-1, NUM_CLASSES), y.reshape(-1))
            loss_a = F.cross_entropy(out["logits_a"].reshape(-1, NUM_CLASSES), y.reshape(-1))
            loss_v = F.cross_entropy(out["logits_v"].reshape(-1, NUM_CLASSES), y.reshape(-1))
            loss = loss_av + 0.3 * loss_a + 0.3 * loss_v

            if use_cf and lam > 0.0:
                perm = make_cross_class_perm(video_cls)
                y_perm = y[perm]
                vid_diff = (video_cls != video_cls[perm]).unsqueeze(1).expand_as(y)
                ev_i = y != BG_CLASS
                M = vid_diff & ev_i
                hard = vid_diff & ev_i & (y_perm != BG_CLASS) & (y != y_perm)

                out_sv = model(audio, visual[perm])
                out_sa = model(audio[perm], visual)

                # 1. Null-slot / gate abstention loss
                loss_null = audio.new_zeros(())
                if null_loss_mode == "pcar" and M.any():
                    rank_terms, cal_terms = [], []
                    for l in range(len(out["aux_a2v"])):
                        for clean_aux, sw_aux in (
                            (out["aux_a2v"][l], out_sv["aux_a2v"][l]),
                            (out["aux_v2a"][l], out_sa["aux_v2a"][l]),
                        ):
                            d = (sw_aux["logodds"] - clean_aux["logodds"]).permute(0, 2, 1)  # [B, T, H]
                            rank_terms.append(F.softplus(cfg["rank_margin"] - d[M]).mean())
                            a_cl = clean_aux["alpha_h"].permute(0, 2, 1)
                            a_sw = sw_aux["alpha_h"].permute(0, 2, 1)
                            cal_terms.append(
                                (F.relu(a_cl[ev_i] - cfg["tau_ev"]) ** 2).mean()
                                + (F.relu(cfg["tau_sw"] - a_sw[M]) ** 2).mean()
                            )
                    loss_null = (
                        cfg["w_rank"] * torch.stack(rank_terms).mean()
                        + cfg["w_cal"] * torch.stack(cal_terms).mean()
                    )
                elif null_loss_mode == "abs" and vid_diff.any():
                    null_sv = out_sv["alpha_null_a2v"]
                    null_sa = out_sa["alpha_null_v2a"]
                    loss_null_swap = 0.5 * (
                        ((1.0 - null_sv[vid_diff]) ** 2).mean()
                        + ((1.0 - null_sa[vid_diff]) ** 2).mean()
                    )
                    if ev_i.any():
                        clean_null = 0.5 * (out["alpha_null_a2v"] + out["alpha_null_v2a"])
                        loss_null_align = (F.relu(clean_null[ev_i] - 0.25) ** 2).mean()
                    else:
                        loss_null_align = audio.new_zeros(())
                    loss_null = cfg["w_null_abs"] * (loss_null_swap + 0.5 * loss_null_align)

                # 2. Label-grounded + teacher-gated per-segment invariance (fixes R2)
                loss_inv = audio.new_zeros(())
                if M.any():
                    ce = 0.5 * (
                        F.cross_entropy(out_sv["logits_a"][M], y[M])
                        + F.cross_entropy(out_sa["logits_v"][M], y[M])
                    )
                    ok_a = M & (out["logits_a"].argmax(dim=-1) == y)
                    ok_v = M & (out["logits_v"].argmax(dim=-1) == y)
                    kl = audio.new_zeros(())
                    if ok_a.any():
                        kl = kl + F.kl_div(
                            F.log_softmax(out_sv["logits_a"][ok_a], dim=-1),
                            F.softmax(out["logits_a"][ok_a].detach(), dim=-1),
                            reduction="batchmean",
                        )
                    if ok_v.any():
                        kl = kl + F.kl_div(
                            F.log_softmax(out_sa["logits_v"][ok_v], dim=-1),
                            F.softmax(out["logits_v"][ok_v].detach(), dim=-1),
                            reduction="batchmean",
                        )
                    loss_inv = ce + cfg["inv_kl_beta"] * 0.5 * kl

                # 3. Hard-negative-only BG rejection (fixes R1)
                loss_bg = audio.new_zeros(())
                if hard.any():
                    bg_t = torch.full((int(hard.sum().item()),), BG_CLASS, dtype=torch.long, device=device)
                    loss_bg = 0.5 * (
                        F.cross_entropy(out_sv["logits_av"][hard], bg_t)
                        + F.cross_entropy(out_sa["logits_av"][hard], bg_t)
                    )

                loss = loss + lam * (
                    loss_null
                    + cfg["cf_invariance_weight"] * loss_inv
                    + cfg["cf_bg_weight"] * loss_bg
                )

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            grad_norms.append(float(gn.item()))
            optimizer.step()

        scheduler.step()

        # Every-epoch validation starting after warmup
        if epoch >= cfg["cf_warmup"] - 1:
            val_m = evaluate_ave(model, val_ds, device)
            ok = (
                val_m["swap_v_audio_retain_acc"] >= 0.70
                and val_m["swap_a_visual_retain_acc"] >= 0.65
                and min(val_m["swap_v_av_reject_rate"], val_m["swap_a_av_reject_rate"]) >= 0.90
            )
            score = val_m["clean_acc"] + (1.0 if ok else 0.0)
            if score > best_score:
                best_score = score
                best_val_clean = val_m["clean_acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_m = evaluate_ave(model, test_ds, device)
    test_m["best_val_clean_acc"] = best_val_clean
    test_m["constraint_met"] = float(best_score >= 1.0)
    test_m["mean_grad_norm"] = float(np.mean(grad_norms))
    return test_m


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "ave"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "ave_r2_results.json"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[AVE Round 2] Running on device: {device}", flush=True)

    audio = _read_first_h5_array(data_dir / "audio_feature.h5")
    visual = _read_first_h5_array(data_dir / "visual_feature_vec.h5")
    labels = _read_first_h5_array(data_dir / "labels.h5")
    train_idx = _read_first_h5_array(data_dir / "train_order.h5").astype(np.int64)
    val_idx = _read_first_h5_array(data_dir / "val_order.h5").astype(np.int64)
    test_idx = _read_first_h5_array(data_dir / "test_order.h5").astype(np.int64)

    train_ds = AVEDataset(audio, visual, labels, train_idx)
    val_ds = AVEDataset(audio, visual, labels, val_idx)
    test_ds = AVEDataset(audio, visual, labels, test_idx)

    all_results: Dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, "r") as f:
            all_results = json.load(f)

    for name, spec in VARIANTS:
        t0 = time.time()
        seed_runs: List[Dict[str, float]] = []
        print(f"\n=== Running {name} ===", flush=True)
        for seed in R2_CFG["seeds"]:
            m = train_one_run(spec, seed, train_ds, val_ds, test_ds, R2_CFG, device)
            seed_runs.append(m)
            print(
                f"  [{name} | s={seed}] clean={m['clean_acc']*100:.2f}% (ev={m['av_event_acc']*100:.1f}%, bg={m['bg_unaligned_acc']*100:.1f}%, ev->bg={m['ev_to_bg_err']*100:.1f}%) | "
                f"A/V_cl={m['clean_audio_event_acc']*100:.1f}/{m['clean_visual_event_acc']*100:.1f}% | "
                f"SwV(ret={m['swap_v_audio_retain_acc']*100:.1f}%, drag={m['swap_v_visual_drag_rate']*100:.1f}%, rej={m['swap_v_av_reject_rate']*100:.1f}%) | "
                f"SwA(ret={m['swap_a_visual_retain_acc']*100:.1f}%, drag={m['swap_a_audio_drag_rate']*100:.1f}%, rej={m['swap_a_av_reject_rate']*100:.1f}%) | "
                f"a_null(ev={0.5*(m['null_a2v_ev']+m['null_v2a_ev']):.3f}, bg={0.5*(m['null_a2v_bg']+m['null_v2a_bg']):.3f}, sv={m['null_a2v_swap_v']:.3f}, sa={m['null_v2a_swap_a']:.3f}) | "
                f"AUROC={0.5*(m['alpha_auroc_swap_v']+m['alpha_auroc_swap_a']):.3f} | FR(ev/sw)={m['fusion_ratio_ev']:.2f}/{m['fusion_ratio_swap']:.2f}",
                flush=True,
            )
            if device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        keys = seed_runs[0].keys()
        summary = {
            k: {
                "mean": float(np.mean([r[k] for r in seed_runs])),
                "std": float(np.std([r[k] for r in seed_runs])),
            }
            for k in keys
        }
        all_results[name] = {
            "spec": spec,
            "runs": seed_runs,
            "summary": summary,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  -> Saved {name} ({all_results[name]['elapsed_sec']}s) to {out_path}", flush=True)

    print("\n[AVE Round 2] ALL 5 GROUPS COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
