import collections
import csv
import json
import pathlib
import time
import numpy as np
import torch
from torch.nn import functional as F

PILOT_DIR = pathlib.Path('/Users/joyboy/Documents/ChatGPT/ICME/work/soco_role_pilot_20261121')
CORRECTED_DIR = pathlib.Path('/Users/joyboy/Documents/ChatGPT/ICME/work/icme_direction_audit_20261121/soco_grid_corrected')

SEEDS = [17, 29, 43]
H = 384
D_MLP = 772


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def sync_device(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def mlp_inputs(c, kps=None):
    """Exact replica of soco_grid_corrected/run.py inputs()."""
    f = c['dense'].unsqueeze(0)
    if kps is None:
        y, x = torch.meshgrid(
            (torch.arange(20) + 0.5) / 20,
            (torch.arange(20) + 0.5) / 20,
            indexing='ij',
        )
        xy = torch.stack([x.flatten(), y.flatten()], 1)
        loc = f[0].flatten(1).T
    else:
        xy = torch.tensor([k['pos'] for k in kps], dtype=torch.float32) / c['pad']
        loc = F.grid_sample(f, (xy * 2 - 1)[None, None], align_corners=False)[0, :, 0].T
    loc = F.normalize(loc, dim=-1)
    ctx = F.normalize(c['cls'], dim=-1).expand(len(loc), -1)
    return torch.cat([loc, ctx, xy, torch.zeros(len(loc), 2)], -1)


def sample_dense_map(feat_map_chw, xy):
    """
    Sample keypoint features from a (C, H, W) feature map using the exact
    corrected coordinate grid (bilinear, align_corners=False, padding_mode='zeros',
    xy in [0, 1]). Pure PyTorch ops so backward works natively on MPS/CUDA/CPU.
    Returns L2-normalized features of shape (K, C).
    """
    C, H_g, W_g = feat_map_chw.shape
    ix = xy[:, 0] * W_g - 0.5
    iy = xy[:, 1] * H_g - 0.5

    x0 = torch.floor(ix).long()
    y0 = torch.floor(iy).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wx1 = ix - x0.float()
    wy1 = iy - y0.float()
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    def gather_corner(yy, xx):
        valid = (yy >= 0) & (yy < H_g) & (xx >= 0) & (xx < W_g)
        yc = yy.clamp(0, H_g - 1)
        xc = xx.clamp(0, W_g - 1)
        vals = feat_map_chw[:, yc, xc].T  # (K, C)
        return vals * valid.float().unsqueeze(-1)

    sampled = (
        gather_corner(y0, x0) * (wy0 * wx0).unsqueeze(-1)
        + gather_corner(y0, x1) * (wy0 * wx1).unsqueeze(-1)
        + gather_corner(y1, x0) * (wy1 * wx0).unsqueeze(-1)
        + gather_corner(y1, x1) * (wy1 * wx1).unsqueeze(-1)
    )
    return F.normalize(sampled, dim=-1)


def load_data():
    cache = torch.load(PILOT_DIR / 'features.pt', map_location='cpu', weights_only=False)
    pairs = json.loads((PILOT_DIR / 'pairs.json').read_text())

    # Precompute normalized dense maps (384, 20, 20) and MLP inputs
    dense_mlp = {n: mlp_inputs(v) for n, v in cache.items()}
    dense_norm = {}
    for n, v in cache.items():
        # Normalize each patch vector along channel dim (384)
        f = v['dense']  # (384, 20, 20)
        dense_norm[n] = F.normalize(f, dim=0)

    # Precompute 20x20 physical patch center coordinates: (400, 2) in (x, y) order
    gy, gx = torch.meshgrid(
        (torch.arange(20) + 0.5) / 20,
        (torch.arange(20) + 0.5) / 20,
        indexing='ij',
    )
    grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (400, 2)

    dataset = []
    for p in pairs:
        s = f"Images/{p['src_category']}/{p['src_image']}"
        t = f"Images/{p['trg_category']}/{p['trg_image']}"
        sk = p['src_keypoints']
        tk = p['trg_keypoints']
        sm = {k['name']: i for i, k in enumerate(sk)}
        tm = {k['name']: i for i, k in enumerate(tk)}
        assert len(sm) == len(sk) and len(tm) == len(tk)

        soc = torch.zeros(len(sk), len(tk), dtype=torch.bool)
        cc = torch.zeros_like(soc)
        for name in p['semantic_overlap']:
            if name in sm and name in tm:
                a, b = sm[name], tm[name]
                assert sk[a]['id'] == tk[b]['id']
                soc[a, b] = True
        for cm in p['concept_matches']:
            for sn in cm['src_keypoints']:
                for tn in cm['trg_keypoints']:
                    if sn in sm and tn in tm:
                        cc[sm[sn], tm[tn]] = True
        assert not (soc & ~cc).any(), p['official_pair_path']

        scale = max(
            p['trg_bndbox'][2] - p['trg_bndbox'][0],
            p['trg_bndbox'][3] - p['trg_bndbox'][1],
        ) / cache[t]['pad']
        s_xy = torch.tensor([k['pos'] for k in sk], dtype=torch.float32) / cache[s]['pad']
        t_xy = torch.tensor([k['pos'] for k in tk], dtype=torch.float32) / cache[t]['pad']

        dataset.append({
            'p': p,
            's': s,
            't': t,
            'xs': mlp_inputs(cache[s], sk),
            'xt': mlp_inputs(cache[t], tk),
            's_xy': s_xy,
            't_xy': t_xy,
            'soc': soc,
            'cc': cc,
            'scale': scale,
            'xy': t_xy,
        })

    return cache, dense_mlp, dense_norm, grid_xy, dataset


def aggregate(rows):
    """Exact replica of soco_grid_corrected/run.py aggregate() + cc_samequery_pck."""
    bycat = {}
    for cat in sorted({r['category'] for r in rows}):
        rr = [r for r in rows if r['category'] == cat]
        pairgroups = collections.defaultdict(list)
        for r in rr:
            pairgroups[r['pair']].append(r)
        out = {}
        for metric in [
            'cc_pck',
            'cc_samequery_pck',
            'cc_rolebranch_pck',
            'soc_pck',
            'dense_wrong_role',
            'oracle_candidate_cc',
            'oracle_candidate_soc',
        ]:
            vals = []
            for pr in pairgroups.values():
                if metric == 'cc_pck':
                    sub = pr
                else:
                    sub = [r for r in pr if r['has_soc']]
                key = 'cc_pck' if metric == 'cc_samequery_pck' else metric
                pv = [float(r[key]) for r in sub if r[key] is not None]
                if pv:
                    vals.append(np.mean(pv))
            out[metric] = float(np.mean(vals)) * 100.0 if vals else None
        out['queries'] = len(rr)
        out['pairs'] = len(pairgroups)
        bycat[cat] = out

    metrics = [k for k in next(iter(bycat.values())) if k not in ['queries', 'pairs']]
    return {
        'category_macro': {
            k: float(np.mean([v[k] for v in bycat.values() if v[k] is not None]))
            for k in metrics
        },
        'by_category': bycat,
    }


@torch.inference_mode()
def evaluate_pair_predictor(
    predict_pair_fn,
    dataset,
    split,
    name,
    seed,
    out_dir=None,
    params=0,
    device=torch.device('cpu'),
):
    """
    predict_pair_fn(d) -> (sim_dense_cc, sim_dense_soc, sim_cand_cc, sim_cand_soc)
      sim_dense_*: (Ks, 400) similarity matrix from source kps to target 20x20 dense grid
      sim_cand_*:  (Ks, Kt)  similarity matrix from source kps to target annotated kps
    """
    items = [d for d in dataset if d['p']['split'] == split]
    rows = []

    # Warmup once for accurate per-pair timing
    if items:
        _ = predict_pair_fn(items[0])
        sync_device(device)

    ts = time.perf_counter()
    for d in items:
        sim_dense_cc, sim_dense_soc, sim_cand_cc, sim_cand_soc = predict_pair_fn(d)
        q0 = sim_dense_cc.argmax(-1).cpu()
        q1 = sim_dense_soc.argmax(-1).cpu()
        pred0 = torch.stack([((q0 % 20) + 0.5) / 20, ((q0 // 20) + 0.5) / 20], -1)
        pred1 = torch.stack([((q1 % 20) + 0.5) / 20, ((q1 // 20) + 0.5) / 20], -1)
        cand1 = sim_cand_soc.argmax(-1).cpu()

        for i, k in enumerate(d['p']['src_keypoints']):
            cs = d['cc'][i].nonzero().flatten()
            ss = d['soc'][i].nonzero().flatten()
            if not len(cs):
                continue
            ec = float(torch.linalg.norm(pred0[i] - d['xy'][cs], dim=-1).min() / d['scale'])
            ecrole = float(torch.linalg.norm(pred1[i] - d['xy'][cs], dim=-1).min() / d['scale'])
            es = (
                float(torch.linalg.norm(pred1[i] - d['xy'][ss], dim=-1).min() / d['scale'])
                if len(ss)
                else None
            )
            wrong = d['cc'][i] & ~d['soc'][i]
            wp = wrong.nonzero().flatten()
            wronghit = bool(
                len(wp)
                and (torch.linalg.norm(pred1[i] - d['xy'][wp], dim=-1).min() / d['scale'] < 0.1)
            )
            ci = int(cand1[i])
            ctk = d['p']['trg_keypoints'][ci]
            rows.append({
                'model': name,
                'seed': seed,
                'split': split,
                'category': d['p']['src_category'],
                'pair': d['p']['official_pair_path'],
                'source_image': d['s'],
                'target_image': d['t'],
                'source_name': k['name'],
                'source_id': k['id'],
                'concept_id': k['concept_id'],
                'source_x': k['pos'][0],
                'source_y': k['pos'][1],
                'has_soc': bool(len(ss)),
                'cc_pck': ec < 0.1,
                'cc_rolebranch_pck': ecrole < 0.1,
                'soc_pck': es < 0.1 if es is not None else None,
                'cc_error': ec,
                'soc_error': es,
                'dense_wrong_role': wronghit and es is not None and es >= 0.1,
                'oracle_candidate_cc': bool(d['cc'][i, ci]),
                'oracle_candidate_soc': bool(d['soc'][i, ci]) if len(ss) else None,
                'oracle_candidate_target': ctk['name'],
                'pred_role_x': float(pred1[i, 0]),
                'pred_role_y': float(pred1[i, 1]),
                'pred_concept_x': float(pred0[i, 0]),
                'pred_concept_y': float(pred0[i, 1]),
            })
    sync_device(device)
    elapsed = time.perf_counter() - ts

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'per_query_{split}_{name}_seed{seed}.csv'
        with path.open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    summary = aggregate(rows)
    summary.update({
        'model': name,
        'seed': seed,
        'split': split,
        'seconds': elapsed,
        'ms_per_pair': (elapsed / len(items)) * 1000.0 if items else 0.0,
        'queries': len(rows),
        'soc_queries': sum(r['has_soc'] for r in rows),
        'pairs': len(items),
        'params': int(params),
    })
    return summary, rows


def paired_image_bootstrap(rows_cand, rows_ref, n_boot=4000, seed=20261121):
    """
    Exact replica of soco_grid_corrected/analyze.py shared-image vertex bootstrap.
    Returns:
      delta_dict: {soc_pck, cc_samequery_pck, cc_pck, dense_wrong_role}
      ci_95: {metric: [lo, hi]}
      cat_soc_delta: {cat: delta_pp}
    """
    def to_pairvals(rows):
        # SOC + cc_samequery + dense_wrong_role on has_soc queries
        g_soc = collections.defaultdict(list)
        g_all = collections.defaultdict(list)
        for r in rows:
            has_soc = r['has_soc'] is True or r['has_soc'] == 'True'
            cc_v = r['cc_pck'] is True or r['cc_pck'] == 'True'
            g_all[(r['category'], r['pair'], str(r['seed']))].append((r['source_image'], r['target_image'], float(cc_v)))
            if has_soc:
                soc_v = r['soc_pck'] is True or r['soc_pck'] == 'True'
                wr_v = r['dense_wrong_role'] is True or r['dense_wrong_role'] == 'True'
                g_soc[(r['category'], r['pair'], str(r['seed']))].append(
                    (r['source_image'], r['target_image'], float(soc_v), float(cc_v), float(wr_v))
                )
        per_soc = collections.defaultdict(list)
        for (c, p, _sd), rs in g_soc.items():
            key = (c, p, rs[0][0], rs[0][1])
            per_soc[key].append([
                np.mean([x[2] for x in rs]),
                np.mean([x[3] for x in rs]),
                np.mean([x[4] for x in rs]),
            ])
        per_all = collections.defaultdict(list)
        for (c, p, _sd), rs in g_all.items():
            key = (c, p, rs[0][0], rs[0][1])
            per_all[key].append(np.mean([x[2] for x in rs]))
        pv_soc = {k: np.mean(v, axis=0) for k, v in per_soc.items()}
        pv_all = {k: float(np.mean(v)) for k, v in per_all.items()}
        return pv_soc, pv_all

    cand_soc, cand_all = to_pairvals(rows_cand)
    ref_soc, ref_all = to_pairvals(rows_ref)
    keys = sorted(cand_soc)
    assert set(keys) == set(ref_soc)
    keys_all = sorted(cand_all)
    assert set(keys_all) == set(ref_all)
    cats = sorted({k[0] for k in keys})

    rng = np.random.default_rng(seed)
    boot_soc = []
    boot_all = []
    for _ in range(n_boot):
        diffs_soc = []
        diffs_all = []
        for c in cats:
            kk = [k for k in keys if k[0] == c]
            ims = sorted({n for k in kk for n in k[2:]})
            mult = np.bincount(rng.choice(len(ims), len(ims)), minlength=len(ims))
            wmap = dict(zip(ims, mult))
            w = np.array([wmap[k[2]] * wmap[k[3]] for k in kk])
            d = np.stack([cand_soc[k] - ref_soc[k] for k in kk])
            diffs_soc.append((w[:, None] * d).sum(0) / w.sum() if w.sum() else d.mean(0))

            kk_a = [k for k in keys_all if k[0] == c]
            w_a = np.array([wmap.get(k[2], 0) * wmap.get(k[3], 0) for k in kk_a])
            d_a = np.array([cand_all[k] - ref_all[k] for k in kk_a])
            diffs_all.append((w_a * d_a).sum() / w_a.sum() if w_a.sum() else d_a.mean())
        boot_soc.append(np.mean(diffs_soc, axis=0) * 100.0)
        boot_all.append(np.mean(diffs_all) * 100.0)

    ci_soc = np.percentile(boot_soc, [2.5, 97.5], axis=0).T.tolist()
    ci_all = np.percentile(boot_all, [2.5, 97.5]).tolist()

    # Exact point estimates
    cat_soc_delta = {}
    deltas_soc = []
    deltas_all = []
    for c in cats:
        kk = [k for k in keys if k[0] == c]
        d = np.mean([cand_soc[k] - ref_soc[k] for k in kk], axis=0) * 100.0
        deltas_soc.append(d)
        cat_soc_delta[c] = float(d[0])
        kk_a = [k for k in keys_all if k[0] == c]
        deltas_all.append(float(np.mean([cand_all[k] - ref_all[k] for k in kk_a]) * 100.0))

    mean_soc = np.mean(deltas_soc, axis=0)
    return {
        'delta_pp': {
            'soc_pck': float(mean_soc[0]),
            'cc_samequery_pck': float(mean_soc[1]),
            'dense_wrong_role': float(mean_soc[2]),
            'cc_pck': float(np.mean(deltas_all)),
        },
        'ci95_pp': {
            'soc_pck': ci_soc[0],
            'cc_samequery_pck': ci_soc[1],
            'dense_wrong_role': ci_soc[2],
            'cc_pck': ci_all,
        },
        'category_soc_delta_pp': cat_soc_delta,
    }


if __name__ == '__main__':
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()

    def frozen_predict(d):
        a = F.normalize(d['xs'][..., :H], dim=-1)
        b_dense = F.normalize(dense_mlp[d['t']][..., :H], dim=-1)
        b_cand = F.normalize(d['xt'][..., :H], dim=-1)
        sd = a @ b_dense.T
        sc = a @ b_cand.T
        return sd, sd, sc, sc

    for sp in ['dev', 'confirm']:
        summ, _ = evaluate_pair_predictor(frozen_predict, dataset, sp, 'frozen', 0)
        print(sp, json.dumps(summ['category_macro'], indent=2))
