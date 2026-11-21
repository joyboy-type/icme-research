import collections
import copy
import hashlib
import io
import json
import pathlib
import tarfile
import time
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from data_utils import SEEDS, get_device, sample_dense_map, sync_device
from run_g1_decomp import G1DecompAdapter
from run_dinov2_lora import DINOv2LoRAAdapter, load_dinov2_tail

OUT = pathlib.Path('/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos')
WEIGHTS_DIR = OUT / 'outputs' / 'weights'
SPAIR_TAR = pathlib.Path('/Users/joyboy/Documents/ChatGPT/ICME/work/icme_spair_preflight_20261121/SPair-71k.tar.gz')

# SOCO training categories (6): chair, armchair, washer, car, bicycle, tiger
# Overlapping with SPair-71k (3): chair, car, bicycle
SOCO_SEEN_SPAIR_CATS = {'chair', 'car', 'bicycle'}

# Symmetric / repeated physical part groups (0-based keypoint IDs) from codex_paper1_external_eval_prep.md
# Geometric contour categories (bottle, pottedplant, tvmonitor) are excluded from the physical-part symmetric subset
# (they are still included in All-Keypoint PCK@0.10).
SYMMETRIC_PART_GROUPS = {
    'aeroplane': [
        {4, 5}, {6, 7}, {8, 9}, {10, 11},
        {12, 13, 14, 15}, {16, 17}, {18, 19, 20, 21},
    ],
    'bicycle': [
        {0, 1}, {2, 3}, {6, 7}, {9, 10},
    ],
    'bird': [
        {1, 2}, {4, 5}, {7, 8}, {10, 11}, {12, 13}, {14, 15},
    ],
    'boat': [
        {1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12},
    ],
    'bus': [
        {0, 1}, {2, 3}, {4, 7}, {5, 6},
        {11, 14, 21, 24}, {10, 12, 13, 15, 20, 22, 23, 25},
        {16, 17, 26, 27}, {18, 19, 28, 29},
    ],
    'car': [
        {0, 1}, {2, 3}, {4, 8}, {5, 9}, {6, 7},
        {11, 14, 21, 24}, {10, 12, 13, 15, 20, 22, 23, 25},
        {16, 17, 26, 27}, {18, 19, 28, 29},
    ],
    'cat': [
        {0, 1}, {2, 3}, {4, 5}, {6, 7}, {9, 10, 11, 12},
    ],
    'chair': [
        {0, 1}, {2, 3}, {4, 5, 6, 7}, {8, 9}, {10, 11}, {12, 13},
    ],
    'cow': [
        {0, 1}, {2, 3}, {4, 5}, {6, 7}, {9, 10, 11, 12}, {15, 16, 17, 18}, {19, 20},
    ],
    'dog': [
        {0, 1}, {2, 3}, {4, 5}, {9, 10, 11, 12},
    ],
    'horse': [
        {0, 1}, {2, 3}, {4, 5}, {6, 7}, {10, 11, 12, 13}, {16, 17, 18, 19},
    ],
    'motorbike': [
        {0, 1}, {2, 3}, {4, 5}, {6, 7}, {11, 12},
    ],
    'person': [
        {0, 1}, {2, 3}, {8, 9}, {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19},
    ],
    'sheep': [
        {0, 1}, {2, 3}, {4, 5}, {6, 7}, {9, 10, 11, 12}, {15, 16, 17, 18}, {19, 20},
    ],
    'train': [
        {0, 1, 4, 5}, {2, 3, 6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}, {16, 17},
    ],
}

# Build fast lookup: (cat, kp_id_int) -> set of counterpart kp_ids (excluding self)
COUNTERPARTS = {}
for _cat, _groups in SYMMETRIC_PART_GROUPS.items():
    for _g in _groups:
        for _k in _g:
            COUNTERPARTS[(_cat, _k)] = _g - {_k}


def stream_spair_test():
    """Single-pass streaming load of all 12,234 official SPair-71k test pairs & 481 unique images."""
    ts = time.perf_counter()
    test_lines = None
    pair_jsons = {}
    img_annos = {}
    img_bytes = {}

    with tarfile.open(SPAIR_TAR, 'r:gz') as tf:
        for m in tf:
            nm = m.name
            if nm == 'SPair-71k/Layout/large/test.txt':
                test_lines = tf.extractfile(m).read().decode().strip().splitlines()
            elif nm.startswith('SPair-71k/PairAnnotation/test/') and nm.endswith('.json'):
                key = nm.split('/')[-1][:-5]
                pair_jsons[key] = json.loads(tf.extractfile(m).read().decode())
            elif nm.startswith('SPair-71k/ImageAnnotation/') and nm.endswith('.json'):
                parts = nm.split('/')
                cat, stem = parts[2], parts[3][:-5]
                img_annos[(cat, stem)] = json.loads(tf.extractfile(m).read().decode())
            elif nm.startswith('SPair-71k/JPEGImages/') and nm.endswith('.jpg'):
                parts = nm.split('/')
                cat, fname = parts[2], parts[3]
                img_bytes[(cat, fname)] = tf.extractfile(m).read()

    assert test_lines is not None and len(test_lines) == 12234
    manifest_sha = hashlib.sha256('\n'.join(test_lines).encode()).hexdigest()

    # Keep only test images referenced by test_lines
    needed_imgs = set()
    pairs = []
    for ln in test_lines:
        p = pair_jsons[ln]
        cat = p['category']
        s_key = (cat, p['src_imname'])
        t_key = (cat, p['trg_imname'])
        needed_imgs.add(s_key)
        needed_imgs.add(t_key)
        t_stem = p['trg_imname'][:-4]
        t_anno_kps = img_annos[(cat, t_stem)]['kps']
        pairs.append({
            'id': ln,
            'category': cat,
            's_key': s_key,
            't_key': t_key,
            'src_kps': p['src_kps'],
            'trg_kps': p['trg_kps'],
            'kps_ids': [int(k) for k in p['kps_ids']],
            'trg_bndbox': p['trg_bndbox'],
            'trg_all_kps': {int(k): v for k, v in t_anno_kps.items() if v is not None},
        })

    needed_imgs = sorted(needed_imgs)
    img_data = {k: img_bytes[k] for k in needed_imgs}
    print(
        f"Streamed SPair-71k test in {time.perf_counter()-ts:.1f}s | "
        f"pairs={len(pairs)} | unique_imgs={len(img_data)} | manifest_sha256={manifest_sha[:16]}...",
        flush=True,
    )
    return pairs, img_data, manifest_sha


@torch.inference_mode()
def extract_spair_dinov2_features(img_data, device):
    """Extract Block-7 tokens and Frozen Block-11 dense maps (280x280 white-padded square, matching SOCO)."""
    ts = time.perf_counter()
    vit = load_dinov2_tail(device).to(device)
    keys = sorted(img_data.keys())

    b7_dict = {}
    froz_dense = {}
    pads = {}

    for start in range(0, len(keys), 16):
        batch = []
        batch_keys = keys[start:start + 16]
        for k in batch_keys:
            im = Image.open(io.BytesIO(img_data[k])).convert('RGB')
            w, h = im.size
            pad = max(w, h)
            pads[k] = pad
            canvas = Image.new('RGB', (pad, pad), 'white')
            canvas.paste(im, (0, 0))
            im = canvas.resize((280, 280), Image.Resampling.BICUBIC)
            x = TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            batch.append(x)
        t = torch.stack(batch).to(device)
        t = vit.patch_embed(t)
        t = vit._pos_embed(t)
        t = vit.patch_drop(t)
        t = vit.norm_pre(t)
        for i in range(8):
            t = vit.blocks[i](t)
        t_b7 = t.cpu()
        t_tail = t
        for i in range(8, 12):
            t_tail = vit.blocks[i](t_tail)
        t_out = vit.norm(t_tail)[:, 1:, :].reshape(-1, 20, 20, 384).permute(0, 3, 1, 2).cpu()
        for idx, k in enumerate(batch_keys):
            b7_dict[k] = t_b7[idx].contiguous()
            froz_dense[k] = F.normalize(t_out[idx], dim=0).contiguous()

    print(f"Extracted DINOv2-S features for {len(keys)} SPair test images in {time.perf_counter()-ts:.1f}s", flush=True)
    del vit
    return keys, b7_dict, froz_dense, pads


@torch.inference_mode()
def encode_all_images_for_model(keys, model_kind, seed, b7_dict, froz_dense, base_blocks_8_11, base_norm, device):
    """Encode all 481 SPair test images once -> (sem_maps, role_maps) on CPU."""
    if model_kind == 'frozen':
        return froz_dense, None

    if model_kind == 'G1_NoAttn':
        model = G1DecompAdapter(use_attn=False, dim=128, num_heads=4, dropout=0.1).to(device)
        model.load_state_dict(torch.load(WEIGHTS_DIR / f'G1_NoAttn_seed{seed}.pt', map_location=device, weights_only=True))
        model.eval()
        sem_out, role_out = {}, {}
        for start in range(0, len(keys), 32):
            bk = keys[start:start + 32]
            x = torch.stack([froz_dense[k] for k in bk], dim=0).to(device)
            sm, rm, _ = model(x)
            sm, rm = sm.cpu(), rm.cpu()
            for idx, k in enumerate(bk):
                sem_out[k] = sm[idx]
                role_out[k] = rm[idx]
        return sem_out, role_out

    single_head = (model_kind == 'L1_LoRA_Single')
    model = DINOv2LoRAAdapter(base_blocks_8_11, base_norm, single_head=single_head, rank=8, lora_alpha=16.0).to(device)
    sd = torch.load(WEIGHTS_DIR / f'{model_kind}_seed{seed}.pt', map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    # Verify all trainable params were loaded
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names.issubset(set(sd.keys())), f"Missing trainable keys: {trainable_names - set(sd.keys())}"
    model.eval()

    sem_out, role_out = {}, {}
    for start in range(0, len(keys), 32):
        bk = keys[start:start + 32]
        x = torch.stack([b7_dict[k] for k in bk], dim=0).to(device)
        sm, rm, _ = model(x)
        sm, rm = sm.cpu(), rm.cpu()
        for idx, k in enumerate(bk):
            sem_out[k] = sm[idx]
            role_out[k] = None if single_head else rm[idx]
    return sem_out, (None if single_head else role_out)


@torch.inference_mode()
def evaluate_spair_pairs(pairs, pads, sem_maps, role_maps, alpha=0.5, use_sem_only=False):
    """
    Evaluate all 12,234 SPair-71k test pairs.
    Returns per-pair dict mapping pair_key=(cat, pair_id, src_img, trg_img) -> metrics dict
    and category-level summary.
    """
    gy, gx = torch.meshgrid((torch.arange(20) + 0.5) / 20.0, (torch.arange(20) + 0.5) / 20.0, indexing='ij')
    grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (400, 2)

    # Pre-flatten target maps for fast matmul
    sem_flat = {k: v.flatten(1).T for k, v in sem_maps.items()}  # (400, 384)
    role_flat = {k: v.flatten(1).T for k, v in role_maps.items()} if (role_maps is not None and not use_sem_only) else None

    pair_records = {}
    by_cat_all10 = collections.defaultdict(list)
    by_cat_all05 = collections.defaultdict(list)
    by_cat_sym10 = collections.defaultdict(list)
    by_cat_wr = collections.defaultdict(list)
    by_cat_sym_cc10 = collections.defaultdict(list)

    total_kps = 0
    total_sym_kps = 0
    total_wr_kps = 0

    for p in pairs:
        cat = p['category']
        sk = p['s_key']
        tk = p['t_key']
        s_pad = float(pads[sk])
        t_pad = float(pads[tk])
        bbox = p['trg_bndbox']
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / t_pad

        s_xy = torch.tensor(p['src_kps'], dtype=torch.float32) / s_pad  # (K, 2)
        t_xy = torch.tensor(p['trg_kps'], dtype=torch.float32) / t_pad  # (K, 2)

        zs_sem = sample_dense_map(sem_maps[sk], s_xy)
        sim = zs_sem @ sem_flat[tk].T  # (K, 400)
        if role_flat is not None:
            zs_role = sample_dense_map(role_maps[sk], s_xy)
            sim = sim + alpha * (zs_role @ role_flat[tk].T)

        q = sim.argmax(dim=-1)
        pred_xy = grid_xy[q]  # (K, 2)
        dists = torch.linalg.norm(pred_xy - t_xy, dim=-1) / scale  # (K,)

        pck10 = (dists < 0.10).float().numpy()
        pck05 = (dists < 0.05).float().numpy()
        p_all10 = float(pck10.mean())
        p_all05 = float(pck05.mean())
        by_cat_all10[cat].append(p_all10)
        by_cat_all05[cat].append(p_all05)
        total_kps += len(pck10)

        sym_hits = []
        wr_hits = []
        sym_cc_hits = []
        t_all = p['trg_all_kps']
        for i, kid in enumerate(p['kps_ids']):
            cparts = COUNTERPARTS.get((cat, kid))
            if cparts is not None:
                sym_hits.append(float(pck10[i]))
                total_sym_kps += 1
                # Check visible counterparts in target image
                cp_coords = [t_all[ck] for ck in cparts if ck in t_all]
                if cp_coords:
                    cp_xy = torch.tensor(cp_coords, dtype=torch.float32) / t_pad
                    cp_dist = float(torch.linalg.norm(pred_xy[i:i + 1] - cp_xy, dim=-1).min() / scale)
                    is_wrong_role = (float(dists[i]) >= 0.10) and (cp_dist < 0.10)
                    is_same_concept = (float(dists[i]) < 0.10) or (cp_dist < 0.10)
                    wr_hits.append(float(is_wrong_role))
                    sym_cc_hits.append(float(is_same_concept))
                    total_wr_kps += 1

        p_sym10 = float(np.mean(sym_hits)) if sym_hits else None
        p_wr = float(np.mean(wr_hits)) if wr_hits else None
        p_sym_cc = float(np.mean(sym_cc_hits)) if sym_cc_hits else None
        if p_sym10 is not None:
            by_cat_sym10[cat].append(p_sym10)
        if p_wr is not None:
            by_cat_wr[cat].append(p_wr)
        if p_sym_cc is not None:
            by_cat_sym_cc10[cat].append(p_sym_cc)

        pair_records[(cat, p['id'], sk[1], tk[1])] = {
            'all10': p_all10,
            'all05': p_all05,
            'sym10': p_sym10,
            'wr': p_wr,
            'sym_cc10': p_sym_cc,
        }

    cats = sorted(by_cat_all10.keys())
    sym_cats = sorted(by_cat_wr.keys())
    unseen_cats = [c for c in cats if c not in SOCO_SEEN_SPAIR_CATS]
    unseen_sym_cats = [c for c in sym_cats if c not in SOCO_SEEN_SPAIR_CATS]

    by_cat = {}
    for c in cats:
        by_cat[c] = {
            'pairs': len(by_cat_all10[c]),
            'all_pck10': float(np.mean(by_cat_all10[c])) * 100.0,
            'all_pck05': float(np.mean(by_cat_all05[c])) * 100.0,
            'sym_pck10': float(np.mean(by_cat_sym10[c])) * 100.0 if by_cat_sym10[c] else None,
            'sym_cc10': float(np.mean(by_cat_sym_cc10[c])) * 100.0 if by_cat_sym_cc10[c] else None,
            'sym_wr': float(np.mean(by_cat_wr[c])) * 100.0 if by_cat_wr[c] else None,
            'unseen_in_soco': c not in SOCO_SEEN_SPAIR_CATS,
        }

    summary = {
        'all18_macro': {
            'all_pck10': float(np.mean([by_cat[c]['all_pck10'] for c in cats])),
            'all_pck05': float(np.mean([by_cat[c]['all_pck05'] for c in cats])),
            'sym_pck10': float(np.mean([by_cat[c]['sym_pck10'] for c in sym_cats])),
            'sym_cc10': float(np.mean([by_cat[c]['sym_cc10'] for c in sym_cats])),
            'sym_wr': float(np.mean([by_cat[c]['sym_wr'] for c in sym_cats])),
        },
        'unseen15_macro': {
            'all_pck10': float(np.mean([by_cat[c]['all_pck10'] for c in unseen_cats])),
            'all_pck05': float(np.mean([by_cat[c]['all_pck05'] for c in unseen_cats])),
            'sym_pck10': float(np.mean([by_cat[c]['sym_pck10'] for c in unseen_sym_cats])),
            'sym_cc10': float(np.mean([by_cat[c]['sym_cc10'] for c in unseen_sym_cats])),
            'sym_wr': float(np.mean([by_cat[c]['sym_wr'] for c in unseen_sym_cats])),
        },
        'counts': {
            'pairs': len(pairs),
            'total_kps': total_kps,
            'sym_kps': total_sym_kps,
            'wr_eval_kps': total_wr_kps,
        },
        'by_category': by_cat,
    }
    return summary, pair_records


def seed_average_pair_records(list_of_pair_records):
    keys = sorted(list_of_pair_records[0].keys())
    out = {}
    for k in keys:
        rec = {}
        for m in ['all10', 'all05', 'sym10', 'wr', 'sym_cc10']:
            vals = [r[k][m] for r in list_of_pair_records if r[k][m] is not None]
            rec[m] = float(np.mean(vals)) if vals else None
        out[k] = rec
    return out


def bootstrap_spair(cand_pr, ref_pr, cats_subset, metric, n_boot=2000, seed=20261121):
    """Within-category shared-image vertex bootstrap on SPair-71k."""
    rng = np.random.default_rng(seed)
    cat_keys = {}
    for c in sorted(cats_subset):
        kk = [k for k in cand_pr if k[0] == c and cand_pr[k][metric] is not None and ref_pr[k][metric] is not None]
        if kk:
            ims = sorted({n for k in kk for n in k[2:]})
            im2i = {im: idx for idx, im in enumerate(ims)}
            s_idx = np.array([im2i[k[2]] for k in kk], dtype=np.int32)
            t_idx = np.array([im2i[k[3]] for k in kk], dtype=np.int32)
            diffs = np.array([cand_pr[k][metric] - ref_pr[k][metric] for k in kk], dtype=np.float64)
            cat_keys[c] = (len(ims), s_idx, t_idx, diffs)

    point_cats = {c: float(np.mean(v[3]) * 100.0) for c, v in cat_keys.items()}
    point_mean = float(np.mean(list(point_cats.values())))

    boot = np.zeros(n_boot, dtype=np.float64)
    cat_list = list(cat_keys.values())
    for b in range(n_boot):
        c_means = []
        for n_im, s_idx, t_idx, diffs in cat_list:
            mult = np.bincount(rng.integers(0, n_im, size=n_im), minlength=n_im)
            w = mult[s_idx] * mult[t_idx]
            ws = w.sum()
            c_means.append((w * diffs).sum() / ws if ws > 0 else diffs.mean())
        boot[b] = np.mean(c_means) * 100.0
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    return {'delta_pp': point_mean, 'ci95_pp': ci, 'by_category_pp': point_cats}


def main():
    device = get_device()
    print(f"Running FULL SPair-71k Zero-Shot External Evaluation on device={device}", flush=True)
    pairs, img_data, manifest_sha = stream_spair_test()
    keys, b7_dict, froz_dense, pads = extract_spair_dinov2_features(img_data, device)

    full_vit = load_dinov2_tail(torch.device('cpu'))
    base_blocks_8_11 = nn.ModuleList([copy.deepcopy(full_vit.blocks[i]) for i in range(8, 12)])
    base_norm = copy.deepcopy(full_vit.norm)
    del full_vit

    models_to_eval = [
        ('frozen',                [0]),
        ('G1_NoAttn',             SEEDS),
        ('L1_LoRA_Single',        SEEDS),
        ('L2_LoRA_Dual_NoAnchor', SEEDS),
        ('L3_LoRA_Dual_Anchored', SEEDS),
    ]

    results_models = {}
    avg_pair_records = {}

    for mname, seeds in models_to_eval:
        t0 = time.perf_counter()
        seed_summaries = []
        seed_prs = []
        for sd in seeds:
            sem_m, role_m = encode_all_images_for_model(
                keys, mname, sd, b7_dict, froz_dense, base_blocks_8_11, base_norm, device
            )
            summ, pr = evaluate_spair_pairs(pairs, pads, sem_m, role_m, alpha=0.5, use_sem_only=False)
            seed_summaries.append(summ)
            seed_prs.append(pr)

        avg_pair_records[mname] = seed_average_pair_records(seed_prs)
        cats = sorted(seed_summaries[0]['by_category'].keys())
        by_cat_mean = {}
        for c in cats:
            by_cat_mean[c] = {
                'pairs': seed_summaries[0]['by_category'][c]['pairs'],
                'unseen_in_soco': seed_summaries[0]['by_category'][c]['unseen_in_soco'],
            }
            for k_m in ['all_pck10', 'all_pck05', 'sym_pck10', 'sym_cc10', 'sym_wr']:
                vals = [s['by_category'][c][k_m] for s in seed_summaries if s['by_category'][c][k_m] is not None]
                by_cat_mean[c][k_m] = float(np.mean(vals)) if vals else None

        results_models[mname] = {
            'seeds': seeds,
            'eval_seconds': time.perf_counter() - t0,
            'all18_macro': {
                k_m: float(np.mean([s['all18_macro'][k_m] for s in seed_summaries]))
                for k_m in seed_summaries[0]['all18_macro']
            },
            'all18_std': {
                k_m: float(np.std([s['all18_macro'][k_m] for s in seed_summaries]))
                for k_m in seed_summaries[0]['all18_macro']
            },
            'unseen15_macro': {
                k_m: float(np.mean([s['unseen15_macro'][k_m] for s in seed_summaries]))
                for k_m in seed_summaries[0]['unseen15_macro']
            },
            'unseen15_std': {
                k_m: float(np.std([s['unseen15_macro'][k_m] for s in seed_summaries]))
                for k_m in seed_summaries[0]['unseen15_macro']
            },
            'by_category': by_cat_mean,
            'counts': seed_summaries[0]['counts'],
        }
        a18 = results_models[mname]['all18_macro']
        u15 = results_models[mname]['unseen15_macro']
        print(
            f"[{mname:22s}] ({time.perf_counter()-t0:4.1f}s) | "
            f"All18: PCK@0.1={a18['all_pck10']:.2f} SymPCK={a18['sym_pck10']:.2f} SymWR={a18['sym_wr']:.2f} | "
            f"Unseen15: PCK@0.1={u15['all_pck10']:.2f} SymPCK={u15['sym_pck10']:.2f} SymWR={u15['sym_wr']:.2f}",
            flush=True,
        )

    all18_cats = sorted(results_models['frozen']['by_category'].keys())
    unseen15_cats = [c for c in all18_cats if c not in SOCO_SEEN_SPAIR_CATS]

    comparisons = {}
    for cand, ref in [
        ('L3_LoRA_Dual_Anchored', 'frozen'),
        ('L3_LoRA_Dual_Anchored', 'G1_NoAttn'),
        ('L3_LoRA_Dual_Anchored', 'L1_LoRA_Single'),
        ('L3_LoRA_Dual_Anchored', 'L2_LoRA_Dual_NoAnchor'),
        ('G1_NoAttn',             'frozen'),
    ]:
        key = f"{cand}_vs_{ref}"
        comparisons[key] = {
            'all18': {
                m: bootstrap_spair(avg_pair_records[cand], avg_pair_records[ref], all18_cats, m)
                for m in ['all10', 'sym10', 'wr', 'sym_cc10']
            },
            'unseen15': {
                m: bootstrap_spair(avg_pair_records[cand], avg_pair_records[ref], unseen15_cats, m)
                for m in ['all10', 'sym10', 'wr', 'sym_cc10']
            },
        }

    out_obj = {
        'protocol': {
            'dataset': 'SPair-71k Official Layout/large/test.txt (FULL 12,234 pairs, 481 unique images, 18 categories)',
            'manifest_sha256': manifest_sha,
            'zero_shot': True,
            'soco_seen_overlap_cats': sorted(SOCO_SEEN_SPAIR_CATS),
            'strictly_unseen_cats': unseen15_cats,
            'preprocessing': '280x280 white-padded square bicubic, ImageNet norm (100% identical to SOCO training)',
            'metric': 'Official SPair-71k bbox-normalized PCK@0.10 (and @0.05), per-pair mean then category macro; Symmetric Physical-Part PCK@0.10 & Wrong-Role Error (WR) across 15 physical-part categories',
            'bootstrap': '2000 replicates within-category shared-image vertex bootstrap, seed 20261121',
        },
        'models': results_models,
        'comparisons': comparisons,
    }
    (OUT / 'spair_zeroshot_results.json').write_text(json.dumps(out_obj, indent=2))

    print("\n=== KEY BOOTSTRAP DELTAS ON FULL SPAIR-71K (12,234 pairs, pp, 95% CI) ===", flush=True)
    for k in ['L3_LoRA_Dual_Anchored_vs_frozen', 'L3_LoRA_Dual_Anchored_vs_G1_NoAttn', 'L3_LoRA_Dual_Anchored_vs_L1_LoRA_Single', 'G1_NoAttn_vs_frozen']:
        for subset in ['all18', 'unseen15']:
            c = comparisons[k][subset]
            print(
                f"{k:40s} [{subset:8s}] | "
                f"All10 {c['all10']['delta_pp']:+.2f} [{c['all10']['ci95_pp'][0]:+.2f}, {c['all10']['ci95_pp'][1]:+.2f}] | "
                f"Sym10 {c['sym10']['delta_pp']:+.2f} [{c['sym10']['ci95_pp'][0]:+.2f}, {c['sym10']['ci95_pp'][1]:+.2f}] | "
                f"SymWR {c['wr']['delta_pp']:+.2f} [{c['wr']['ci95_pp'][0]:+.2f}, {c['wr']['ci95_pp'][1]:+.2f}]",
                flush=True,
            )
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
