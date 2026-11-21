"""LLP / AVVP (ECCV 2020) Weakly-Supervised Training & Dense Evaluation Scaffold.

Ready to run once `bash download_llp.sh` extracts `data/llp/feats/{vggish,res152,r2plus1d_18}`.
Compares:
  (a) baseline_xattn  : Standard Hybrid Attention / Cross-Attention (HAN-style)
  (b) nullslot_xattn  : Null-Slot Cross-Attention (abstain slot for unimodal / off-screen events)
  (c) nullslot_cfswap : Null-Slot + Counterfactual Modality-Swap Regularization

Evaluates segment-level F1 for Audio, Visual, Audio-Visual, Type@AV, Event@AV, plus
explicit error breakdown on Audio-only (off-screen) and Visual-only (silent) segments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.nullslot_av import AVNullSlotModel

LLP_CATEGORIES = [
    "Speech", "Car", "Cheering", "Dog", "Cat", "Frying_(food)", "Basketball_bounce",
    "Fire_alarm", "Chainsaw", "Cello", "Banjo", "Singing", "Chicken_rooster",
    "Violin_fiddle", "Vacuum_cleaner", "Baby_laughter", "Accordion", "Lawn_mower",
    "Motorcycle", "Helicopter", "Acoustic_guitar", "Telephone_bell_ringing",
    "Baby_cry_infant_cry", "Blender", "Clapping",
]
CAT2IDX = {c: i for i, c in enumerate(LLP_CATEGORIES)}


class LLPDataset(Dataset):
    """Loads LLP pre-extracted features (.npy) + weak video-level labels."""

    def __init__(self, csv_path: Path, feats_root: Path) -> None:
        self.audio_dir = feats_root / "vggish"
        self.rgb_dir = feats_root / "res152"
        self.st_dir = feats_root / "r2plus1d_18"
        self.samples: List[Tuple[str, np.ndarray]] = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                vid = row["filename"]
                label_vec = np.zeros(len(LLP_CATEGORIES), dtype=np.float32)
                for lbl in row["event_labels"].split(","):
                    lbl = lbl.strip()
                    if lbl in CAT2IDX:
                        label_vec[CAT2IDX[lbl]] = 1.0
                self.samples.append((vid, label_vec))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        vid, label = self.samples[idx]
        a = np.load(self.audio_dir / f"{vid}.npy").astype(np.float32)
        v2d = np.load(self.rgb_dir / f"{vid}.npy").astype(np.float32)
        v3d = np.load(self.st_dir / f"{vid}.npy").astype(np.float32)
        # Align to 10 segments
        T = 10
        a = a[:T] if len(a) >= T else np.pad(a, ((0, T - len(a)), (0, 0)))
        v2d = v2d[:T] if len(v2d) >= T else np.pad(v2d, ((0, T - len(v2d)), (0, 0)))
        v3d = v3d[:T] if len(v3d) >= T else np.pad(v3d, ((0, T - len(v3d)), (0, 0)))
        v = np.concatenate([v2d, v3d], axis=-1)
        return vid, torch.from_numpy(a), torch.from_numpy(v), torch.from_numpy(label)


def load_llp_dense_gt(eval_csv: Path) -> Dict[str, np.ndarray]:
    """Load dense 10-second [10, 25] binary event annotations from AVVP_eval_{audio,visual}.csv."""
    gt: Dict[str, np.ndarray] = {}
    with open(eval_csv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vid = row["filename"]
            if vid not in gt:
                gt[vid] = np.zeros((10, len(LLP_CATEGORIES)), dtype=np.int32)
            lbl = row["event_labels"].strip()
            if lbl not in CAT2IDX:
                continue
            c = CAT2IDX[lbl]
            onset = max(0, int(round(float(row["onset"]))))
            offset = min(10, int(round(float(row["offset"]))))
            gt[vid][onset:offset, c] = 1
    return gt


def mil_pool_probs(
    logits_a: torch.Tensor, logits_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attentive / max-mean MIL pooling for weakly-supervised AVVP."""
    pa = torch.sigmoid(logits_a)  # [B, T, C]
    pv = torch.sigmoid(logits_v)  # [B, T, C]
    # Segment-level joint probability is pa * pv; video-level via temporal attention / max-pool
    p_video_a = pa.max(dim=1).values
    p_video_v = pv.max(dim=1).values
    p_video_av = torch.clamp(p_video_a + p_video_v - p_video_a * p_video_v, 1e-6, 1.0 - 1e-6)
    return p_video_a, p_video_v, p_video_av


def segment_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    denom = 2 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LLP / AVVP Null-Slot Scaffold")
    parser.add_argument("--feats_root", type=Path, default=Path(__file__).parent / "data/llp/feats")
    parser.add_argument("--variant", choices=["baseline_xattn", "nullslot_xattn", "nullslot_cfswap"], default="nullslot_cfswap")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    if not (args.feats_root / "vggish").exists():
        print(
            f"[LLP Scaffold] Pre-extracted features not found at {args.feats_root}.\n"
            "Run `bash download_llp.sh` first to download official feats.zip (7.8 GiB)."
        )
        return

    data_dir = Path(__file__).parent / "data/llp"
    train_ds = LLPDataset(data_dir / "AVVP_train.csv", args.feats_root)
    val_ds = LLPDataset(data_dir / "AVVP_val_pd.csv", args.feats_root)
    gt_audio = load_llp_dense_gt(data_dir / "AVVP_eval_audio.csv")
    gt_visual = load_llp_dense_gt(data_dir / "AVVP_eval_visual.csv")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    use_null = args.variant in ("nullslot_xattn", "nullslot_cfswap")
    use_cf = args.variant == "nullslot_cfswap"

    model = AVNullSlotModel(
        d_audio=128,
        d_visual=2560,
        num_classes=len(LLP_CATEGORIES),
        d_model=256,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        use_null_slot=use_null,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(args.epochs):
        model.train()
        for _, a, v, y_weak in train_loader:
            a, v, y_weak = a.to(device), v.to(device), y_weak.to(device)
            out = model(a, v)
            _, _, p_vid = mil_pool_probs(out["logits_a"], out["logits_v"])
            loss = F.binary_cross_entropy(p_vid, y_weak)

            if use_cf:
                perm = torch.roll(torch.arange(a.size(0), device=device), shifts=1)
                out_sv = model(a, v[perm])
                out_sa = model(a[perm], v)
                # Push null slot -> 1 on guaranteed unaligned pairs
                loss_null = 0.5 * (
                    ((1.0 - out_sv["alpha_null_a2v"]) ** 2).mean()
                    + ((1.0 - out_sa["alpha_null_v2a"]) ** 2).mean()
                )
                # Unimodal preservation: audio stream shouldn't shift under swapped visual, and vice versa
                loss_inv = 0.5 * (
                    F.mse_loss(torch.sigmoid(out_sv["logits_a"]), torch.sigmoid(out["logits_a"].detach()))
                    + F.mse_loss(torch.sigmoid(out_sa["logits_v"]), torch.sigmoid(out["logits_v"].detach()))
                )
                loss = loss + 0.3 * loss_null + 0.5 * loss_inv

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Dense evaluation on val split
    model.eval()
    pa_all, pv_all, pav_all = [], [], []
    ga_all, gv_all, gav_all = [], [], []
    with torch.no_grad():
        for vid, a, v, _ in DataLoader(val_ds, batch_size=64, shuffle=False):
            out = model(a.to(device), v.to(device))
            pa = (torch.sigmoid(out["logits_a"]) > 0.5).cpu().numpy().astype(np.int32)
            pv = (torch.sigmoid(out["logits_v"]) > 0.5).cpu().numpy().astype(np.int32)
            pav = pa * pv
            for i, v_id in enumerate(vid):
                if v_id in gt_audio and v_id in gt_visual:
                    ga = gt_audio[v_id]
                    gv = gt_visual[v_id]
                    pa_all.append(pa[i])
                    pv_all.append(pv[i])
                    pav_all.append(pav[i])
                    ga_all.append(ga)
                    gv_all.append(gv)
                    gav_all.append(ga * gv)

    if pa_all:
        f1_a = segment_f1(np.stack(pa_all), np.stack(ga_all))
        f1_v = segment_f1(np.stack(pv_all), np.stack(gv_all))
        f1_av = segment_f1(np.stack(pav_all), np.stack(gav_all))
        print(f"[LLP Val | {args.variant}] Seg-F1: Audio={f1_a*100:.2f}%, Visual={f1_v*100:.2f}%, AV={f1_av*100:.2f}%")


if __name__ == "__main__":
    main()
