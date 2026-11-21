"""AVE Pilot Experiment: Standard Cross-Attention vs Null-Slot vs Null-Slot + CF-Swap.

Runs multi-seed training and evaluation on the official AVE pre-extracted features
(VGGish audio + VGG-19 GAP visual) and outputs full clean + counterfactual stress-test
metrics to `results/ave_pilot_results.json`.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.nullslot_av import AVNullSlotModel

BG_CLASS = 28
NUM_CLASSES = 29


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
        # labels can be one-hot [N, 10, 29] or integer [N, 10]
        raw_y = labels[indices]
        if raw_y.ndim == 3:
            self.labels = np.argmax(raw_y, axis=-1).astype(np.int64)
        else:
            self.labels = raw_y.astype(np.int64)

        # Dominant event class per video (for cross-class counterfactual pairing)
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
    """Create a batch permutation where video_cls[perm[i]] != video_cls[i] whenever possible."""
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

    all_y = []
    all_pred_av = []
    all_pred_a = []
    all_pred_v = []
    null_event_vals = []
    null_bg_vals = []

    for audio, visual, y, _ in loader:
        audio, visual = audio.to(device), visual.to(device)
        out = model(audio, visual)
        pred_av = out["logits_av"].argmax(dim=-1).cpu().numpy()
        pred_a = out["logits_a"].argmax(dim=-1).cpu().numpy()
        pred_v = out["logits_v"].argmax(dim=-1).cpu().numpy()
        y_np = y.numpy()

        all_y.append(y_np)
        all_pred_av.append(pred_av)
        all_pred_a.append(pred_a)
        all_pred_v.append(pred_v)

        if out["alpha_null_a2v"] is not None:
            na = 0.5 * (out["alpha_null_a2v"] + out["alpha_null_v2a"]).cpu().numpy()
            if np.any(y_np != BG_CLASS):
                null_event_vals.append(na[y_np != BG_CLASS])
            if np.any(y_np == BG_CLASS):
                null_bg_vals.append(na[y_np == BG_CLASS])

    y_flat = np.concatenate(all_y, axis=0).reshape(-1)
    pred_av_flat = np.concatenate(all_pred_av, axis=0).reshape(-1)
    pred_a_flat = np.concatenate(all_pred_a, axis=0).reshape(-1)
    pred_v_flat = np.concatenate(all_pred_v, axis=0).reshape(-1)

    event_mask = y_flat != BG_CLASS
    bg_mask = y_flat == BG_CLASS

    # Deterministic cross-class counterfactual stress test on the entire split
    N = len(dataset)
    cls_np = dataset.video_cls
    swap_idx = np.zeros(N, dtype=np.int64)
    for i in range(N):
        # Deterministic partner with different class
        candidates = np.where((cls_np != cls_np[i]) & (cls_np != BG_CLASS))[0]
        if len(candidates) > 0:
            swap_idx[i] = candidates[(i * 7 + 13) % len(candidates)]
        else:
            swap_idx[i] = (i + 1) % N

    audio_all = torch.from_numpy(dataset.audio).to(device)
    visual_all = torch.from_numpy(dataset.visual).to(device)
    y_all = dataset.labels  # [N, 10]
    y_swap = dataset.labels[swap_idx]  # [N, 10]

    # Swap-V: true Audio_i + mismatched Visual_j (off-screen sound / visual distractor)
    swap_v_pred_a, swap_v_pred_av, null_swap_v = [], [], []
    # Swap-A: mismatched Audio_j + true Visual_i (silent visual / audio distractor)
    swap_a_pred_v, swap_a_pred_av, null_swap_a = [], [], []

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        a_i = audio_all[start:end]
        v_i = visual_all[start:end]
        a_j = audio_all[swap_idx[start:end]]
        v_j = visual_all[swap_idx[start:end]]

        out_sv = model(a_i, v_j)
        swap_v_pred_a.append(out_sv["logits_a"].argmax(dim=-1).cpu().numpy())
        swap_v_pred_av.append(out_sv["logits_av"].argmax(dim=-1).cpu().numpy())
        if out_sv["alpha_null_a2v"] is not None:
            null_swap_v.append(out_sv["alpha_null_a2v"].cpu().numpy())

        out_sa = model(a_j, v_i)
        swap_a_pred_v.append(out_sa["logits_v"].argmax(dim=-1).cpu().numpy())
        swap_a_pred_av.append(out_sa["logits_av"].argmax(dim=-1).cpu().numpy())
        if out_sa["alpha_null_v2a"] is not None:
            null_swap_a.append(out_sa["alpha_null_v2a"].cpu().numpy())

    sv_pred_a = np.concatenate(swap_v_pred_a, axis=0)
    sv_pred_av = np.concatenate(swap_v_pred_av, axis=0)
    sa_pred_v = np.concatenate(swap_a_pred_v, axis=0)
    sa_pred_av = np.concatenate(swap_a_pred_av, axis=0)

    # Evaluate on segments where BOTH target and distractor are active event segments of DIFFERENT classes
    active_diff_mask = (y_all != BG_CLASS) & (y_swap != BG_CLASS) & (y_all != y_swap)

    # Audio-only stress test (Swap-V): does audio stream retain true y_i, or get dragged to visual distractor y_j?
    sv_audio_retain = float(np.mean(sv_pred_a[active_diff_mask] == y_all[active_diff_mask]))
    sv_visual_drag = float(np.mean(sv_pred_a[active_diff_mask] == y_swap[active_diff_mask]))
    sv_av_reject = float(np.mean(sv_pred_av[active_diff_mask] == BG_CLASS))

    # Visual-only stress test (Swap-A): does visual stream retain true y_i, or get dragged to audio distractor y_j?
    sa_visual_retain = float(np.mean(sa_pred_v[active_diff_mask] == y_all[active_diff_mask]))
    sa_audio_drag = float(np.mean(sa_pred_v[active_diff_mask] == y_swap[active_diff_mask]))
    sa_av_reject = float(np.mean(sa_pred_av[active_diff_mask] == BG_CLASS))

    # Clean unimodal stream accuracy on active event segments (reference before swap)
    clean_a_ev = float(np.mean(pred_a_flat[event_mask] == y_flat[event_mask]))
    clean_v_ev = float(np.mean(pred_v_flat[event_mask] == y_flat[event_mask]))

    metrics = {
        "clean_acc": float(np.mean(pred_av_flat == y_flat)),
        "clean_macro_f1": macro_f1(y_flat, pred_av_flat),
        "av_event_acc": float(np.mean(pred_av_flat[event_mask] == y_flat[event_mask])),
        "bg_unaligned_acc": float(np.mean(pred_av_flat[bg_mask] == y_flat[bg_mask])),
        "clean_audio_event_acc": clean_a_ev,
        "clean_visual_event_acc": clean_v_ev,
        "swap_v_audio_retain_acc": sv_audio_retain,
        "swap_v_visual_drag_rate": sv_visual_drag,
        "swap_v_av_reject_rate": sv_av_reject,
        "swap_a_visual_retain_acc": sa_visual_retain,
        "swap_a_audio_drag_rate": sa_audio_drag,
        "swap_a_av_reject_rate": sa_av_reject,
        "null_mass_clean_event": float(np.mean(np.concatenate(null_event_vals))) if null_event_vals else 0.0,
        "null_mass_clean_bg": float(np.mean(np.concatenate(null_bg_vals))) if null_bg_vals else 0.0,
        "null_mass_swap_v": float(np.mean(np.concatenate(null_swap_v)[active_diff_mask])) if null_swap_v else 0.0,
        "null_mass_swap_a": float(np.mean(np.concatenate(null_swap_a)[active_diff_mask])) if null_swap_a else 0.0,
    }
    return metrics


def train_one_run(
    variant: str,
    seed: int,
    train_ds: AVEDataset,
    val_ds: AVEDataset,
    test_ds: AVEDataset,
    cfg: dict,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(seed)
    use_null = variant in ("nullslot_xattn", "nullslot_cfswap")
    use_cf = variant in ("baseline_cfswap", "nullslot_cfswap")

    d_audio = train_ds.audio.shape[-1]
    d_visual = train_ds.visual.shape[-1]

    model = AVNullSlotModel(
        d_audio=d_audio,
        d_visual=d_visual,
        num_classes=NUM_CLASSES,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        use_null_slot=use_null,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    best_val_score = -1.0
    best_state = None

    for epoch in range(cfg["epochs"]):
        model.train()
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

            if use_cf:
                perm = make_cross_class_perm(video_cls)
                y_perm = y[perm]
                diff_mask = (video_cls != video_cls[perm]).unsqueeze(1).expand_as(y)  # [B, T]
                # Both swapped pairs
                out_sv = model(audio, visual[perm])
                out_sa = model(audio[perm], visual)

                # 1. Null-slot abstention supervision on mismatched pairs (only when use_null=True)
                if use_null:
                    null_sv = out_sv["alpha_null_a2v"]
                    null_sa = out_sa["alpha_null_v2a"]
                    loss_null_swap = (
                        ((1.0 - null_sv[diff_mask]) ** 2).mean()
                        + ((1.0 - null_sa[diff_mask]) ** 2).mean()
                    ) * 0.5
                    ev_mask = y != BG_CLASS
                    if ev_mask.any():
                        clean_null = 0.5 * (out["alpha_null_a2v"] + out["alpha_null_v2a"])
                        # Encourage clean aligned events to actually attend (keep null mass <= 0.25)
                        loss_null_align = (F.relu(clean_null[ev_mask] - 0.25) ** 2).mean()
                    else:
                        loss_null_align = torch.tensor(0.0, device=device)
                    loss_null = loss_null_swap + 0.5 * loss_null_align
                else:
                    loss_null = torch.tensor(0.0, device=device)

                # 2. Unimodal preservation / anti-drag KL invariance on swapped pairs
                p_a_clean = F.softmax(out["logits_a"].detach(), dim=-1)
                p_v_clean = F.softmax(out["logits_v"].detach(), dim=-1)
                kl_a = F.kl_div(F.log_softmax(out_sv["logits_a"], dim=-1), p_a_clean, reduction="batchmean")
                kl_v = F.kl_div(F.log_softmax(out_sa["logits_v"], dim=-1), p_v_clean, reduction="batchmean")
                loss_inv = 0.5 * (kl_a + kl_v)

                # 3. Joint AV background rejection on mismatched pairs
                bg_target = torch.full((diff_mask.sum().item(),), BG_CLASS, dtype=torch.long, device=device)
                if bg_target.numel() > 0:
                    loss_bg = 0.5 * (
                        F.cross_entropy(out_sv["logits_av"][diff_mask], bg_target)
                        + F.cross_entropy(out_sa["logits_av"][diff_mask], bg_target)
                    )
                else:
                    loss_bg = torch.tensor(0.0, device=device)

                loss = (
                    loss
                    + cfg["cf_null_weight"] * loss_null
                    + cfg["cf_invariance_weight"] * loss_inv
                    + cfg["cf_bg_weight"] * loss_bg
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == cfg["epochs"] - 1:
            val_m = evaluate_ave(model, val_ds, device)
            # Select purely on clean validation accuracy (fair model selection!)
            if val_m["clean_acc"] > best_val_score:
                best_val_score = val_m["clean_acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_m = evaluate_ave(model, test_ds, device)
    test_m["best_val_clean_acc"] = best_val_score
    return test_m


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "ave"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(root / "protocol.json", "r") as f:
        proto = json.load(f)
    cfg = proto["training"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[AVE Pilot] Running on device: {device}")

    audio = _read_first_h5_array(data_dir / "audio_feature.h5")
    visual = _read_first_h5_array(data_dir / "visual_feature_vec.h5")
    labels = _read_first_h5_array(data_dir / "labels.h5")
    train_idx = _read_first_h5_array(data_dir / "train_order.h5").astype(np.int64)
    val_idx = _read_first_h5_array(data_dir / "val_order.h5").astype(np.int64)
    test_idx = _read_first_h5_array(data_dir / "test_order.h5").astype(np.int64)

    print(
        f"[AVE Data] audio={audio.shape}, visual={visual.shape}, labels={labels.shape}, "
        f"splits=(train:{len(train_idx)}, val:{len(val_idx)}, test:{len(test_idx)})"
    )

    train_ds = AVEDataset(audio, visual, labels, train_idx)
    val_ds = AVEDataset(audio, visual, labels, val_idx)
    test_ds = AVEDataset(audio, visual, labels, test_idx)

    variants = ["baseline_xattn", "nullslot_xattn", "baseline_cfswap", "nullslot_cfswap"]
    all_results: Dict[str, dict] = {}

    for variant in variants:
        t0 = time.time()
        seed_runs = []
        for seed in cfg["seeds"]:
            m = train_one_run(variant, seed, train_ds, val_ds, test_ds, cfg, device)
            seed_runs.append(m)
            print(
                f"  [{variant} | seed={seed}] "
                f"clean_acc={m['clean_acc']*100:.2f}% | "
                f"ev={m['av_event_acc']*100:.2f}% bg={m['bg_unaligned_acc']*100:.2f}% | "
                f"SwapV(A_ret={m['swap_v_audio_retain_acc']*100:.1f}%, V_drag={m['swap_v_visual_drag_rate']*100:.1f}%, AV_rej={m['swap_v_av_reject_rate']*100:.1f}%) | "
                f"SwapA(V_ret={m['swap_a_visual_retain_acc']*100:.1f}%, A_drag={m['swap_a_audio_drag_rate']*100:.1f}%, AV_rej={m['swap_a_av_reject_rate']*100:.1f}%) | "
                f"null(ev={m['null_mass_clean_event']:.3f}, bg={m['null_mass_clean_bg']:.3f}, sv={m['null_mass_swap_v']:.3f}, sa={m['null_mass_swap_a']:.3f})"
            )

        keys = seed_runs[0].keys()
        summary = {
            k: {
                "mean": float(np.mean([r[k] for r in seed_runs])),
                "std": float(np.std([r[k] for r in seed_runs])),
            }
            for k in keys
        }
        all_results[variant] = {
            "runs": seed_runs,
            "summary": summary,
            "elapsed_sec": round(time.time() - t0, 2),
        }

    out_path = results_dir / "ave_pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Saved results to {out_path}")


if __name__ == "__main__":
    main()
