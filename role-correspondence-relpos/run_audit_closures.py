import copy
import json
import pathlib
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data_utils import (
    SEEDS,
    evaluate_pair_predictor,
    get_device,
    load_data,
    sample_dense_map,
    sync_device,
)
from train_eval import (
    contrastive_pair_loss,
    prepare_train_pairs_on_device,
)
from run_ablation_2x2 import (
    aggregate_model_runs,
    bootstrap_linear_combo,
    load_csv_rows,
    to_pairvals_full,
)
from run_dinov2_lora import (
    DINOv2LoRAAdapter,
    build_or_load_block7_cache,
    load_dinov2_tail,
    make_lora_predictor,
)
from run_spair_zeroshot import (
    SOCO_SEEN_SPAIR_CATS,
    bootstrap_spair,
    encode_all_images_for_model,
    evaluate_spair_pairs,
    extract_spair_dinov2_features,
    seed_average_pair_records,
    stream_spair_test,
)

OUT = pathlib.Path('/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos')
OUTPUTS_DIR = OUT / 'outputs'
WEIGHTS_DIR = OUTPUTS_DIR / 'weights'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


class L0FrozenSameHead(nn.Module):
    """Exact same dual residual heads as L3_LoRA_Dual_Anchored, but with frozen DINOv2-S (no LoRA)."""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.sem_norm = nn.LayerNorm(384)
        self.sem_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 384, bias=False),
        )
        nn.init.zeros_(self.sem_head[-1].weight)

        self.role_norm = nn.LayerNorm(384)
        self.role_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 384),
        )
        nn.init.normal_(self.role_head[-1].weight, std=0.02)
        nn.init.zeros_(self.role_head[-1].bias)

    def forward(self, frozen_dense_norm):
        # frozen_dense_norm: (B, 384, 20, 20), L2-normalized per patch
        B = frozen_dense_norm.shape[0]
        x0 = frozen_dense_norm.flatten(2).transpose(1, 2)  # (B, 400, 384)
        z_sem = F.normalize(x0 + self.sem_head(self.sem_norm(x0)), dim=-1)
        z_role = F.normalize(self.role_head(self.role_norm(x0)), dim=-1)
        anchor_loss = (1.0 - (z_sem * x0).sum(dim=-1)).mean()
        sem_map = z_sem.transpose(1, 2).reshape(B, 384, 20, 20)
        role_map = z_role.transpose(1, 2).reshape(B, 384, 20, 20)
        return sem_map, role_map, anchor_loss


def train_L0(dataset, dense_norm, grid_xy, train_ids, seed, device, lr=3e-4, steps=300, batch_size=8, alpha_train=0.6, tau=0.07):
    rng = np.random.default_rng(seed)
    schedule = [rng.choice(train_ids, batch_size, replace=True).tolist() for _ in range(steps)]
    torch.manual_seed(seed)

    model = L0FrozenSameHead(dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    train_imgs = sorted({dataset[i]['s'] for i in train_ids} | {dataset[i]['t'] for i in train_ids})
    dn_dev = {k: dense_norm[k].to(device) for k in train_imgs}
    pre = prepare_train_pairs_on_device(dataset, train_ids, grid_xy, device)

    ts = time.perf_counter()
    model.train()
    for step, ids in enumerate(schedule):
        opt.zero_grad()
        items = [pre[i] for i in ids]
        uniq_imgs = []
        img_to_idx = {}
        for d in items:
            for nm in (d['s'], d['t']):
                if nm not in img_to_idx:
                    img_to_idx[nm] = len(uniq_imgs)
                    uniq_imgs.append(nm)
        x0_batch = torch.stack([dn_dev[nm] for nm in uniq_imgs], dim=0)
        sem_batch, role_batch, anchor_loss = model(x0_batch)

        cc_losses, soc_role_losses, soc_comb_losses = [], [], []
        for d in items:
            si, ti = img_to_idx[d['s']], img_to_idx[d['t']]
            zs_sem = sample_dense_map(sem_batch[si], d['s_xy'])
            zt_sem = sample_dense_map(sem_batch[ti], d['t_xy'])
            sim_sem_all = torch.cat([zs_sem @ zt_sem.T, zs_sem @ sem_batch[ti].flatten(1)], dim=-1)

            zs_role = sample_dense_map(role_batch[si], d['s_xy'])
            zt_role = sample_dense_map(role_batch[ti], d['t_xy'])
            sim_role_all = torch.cat([zs_role @ zt_role.T, zs_role @ role_batch[ti].flatten(1)], dim=-1)
            sim_comb_all = sim_sem_all.detach() + alpha_train * sim_role_all

            lc = contrastive_pair_loss(sim_sem_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
            lsr = contrastive_pair_loss(sim_role_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            lsc = contrastive_pair_loss(sim_comb_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            if lc is not None: cc_losses.append(lc)
            if lsr is not None: soc_role_losses.append(lsr)
            if lsc is not None: soc_comb_losses.append(lsc)

        loss = (
            10.0 * anchor_loss
            + 0.5 * torch.stack(cc_losses).mean()
            + 0.3 * torch.stack(soc_role_losses).mean()
            + 1.0 * torch.stack(soc_comb_losses).mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    sync_device(device)
    model.eval()
    return model, time.perf_counter() - ts


def train_L1a(base_blocks_8_11, base_norm, dataset, b7_cache, dense_norm, grid_xy, train_ids, seed, device, lr=3e-4, steps=300, batch_size=8, tau=0.07):
    """L1_LoRA_Single with w_anchor=10.0 turned ON."""
    rng = np.random.default_rng(seed)
    schedule = [rng.choice(train_ids, batch_size, replace=True).tolist() for _ in range(steps)]
    torch.manual_seed(seed)

    model = DINOv2LoRAAdapter(base_blocks_8_11, base_norm, single_head=True, rank=8, lora_alpha=16.0).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    train_imgs = sorted({dataset[i]['s'] for i in train_ids} | {dataset[i]['t'] for i in train_ids})
    b7_dev = {k: b7_cache[k].to(device) for k in train_imgs}
    dn_dev = {k: dense_norm[k].to(device) for k in train_imgs}
    pre = prepare_train_pairs_on_device(dataset, train_ids, grid_xy, device)

    ts = time.perf_counter()
    model.train()
    for step, ids in enumerate(schedule):
        opt.zero_grad()
        items = [pre[i] for i in ids]
        uniq_imgs = []
        img_to_idx = {}
        for d in items:
            for nm in (d['s'], d['t']):
                if nm not in img_to_idx:
                    img_to_idx[nm] = len(uniq_imgs)
                    uniq_imgs.append(nm)
        b7_batch = torch.stack([b7_dev[nm] for nm in uniq_imgs], dim=0)
        x0_batch = torch.stack([dn_dev[nm] for nm in uniq_imgs], dim=0)
        sem_batch, _, anchor_loss = model(b7_batch, x0_batch)

        cc_losses, soc_losses = [], []
        for d in items:
            si, ti = img_to_idx[d['s']], img_to_idx[d['t']]
            zs = sample_dense_map(sem_batch[si], d['s_xy'])
            zt = sample_dense_map(sem_batch[ti], d['t_xy'])
            sim_all = torch.cat([zs @ zt.T, zs @ sem_batch[ti].flatten(1)], dim=-1)
            lc = contrastive_pair_loss(sim_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
            ls = contrastive_pair_loss(sim_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            if lc is not None: cc_losses.append(lc)
            if ls is not None: soc_losses.append(ls)

        loss = 10.0 * anchor_loss + 0.5 * torch.stack(cc_losses).mean() + 0.5 * torch.stack(soc_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
    sync_device(device)
    model.eval()
    return model, time.perf_counter() - ts


def make_L0_predictor(model, dense_norm, device, alpha=0.5):
    model.eval()
    def predict_pair(d):
        x = torch.stack([dense_norm[d['s']], dense_norm[d['t']]], dim=0).to(device)
        sem_b, role_b, _ = model(x)
        s_xy, t_xy = d['s_xy'].to(device), d['t_xy'].to(device)
        zs_sem, zt_sem = sample_dense_map(sem_b[0], s_xy), sample_dense_map(sem_b[1], t_xy)
        zs_role, zt_role = sample_dense_map(role_b[0], s_xy), sample_dense_map(role_b[1], t_xy)
        sd_cc = zs_sem @ sem_b[1].flatten(1)
        sc_cc = zs_sem @ zt_sem.T
        sd_soc = sd_cc + alpha * (zs_role @ role_b[1].flatten(1))
        sc_soc = sc_cc + alpha * (zs_role @ zt_role.T)
        return sd_cc, sd_soc, sc_cc, sc_soc
    return predict_pair


@torch.inference_mode()
def encode_spair_custom(keys, model, is_L0, b7_dict, froz_dense, device):
    sem_out, role_out = {}, {}
    for start in range(0, len(keys), 32):
        bk = keys[start:start + 32]
        if is_L0:
            x = torch.stack([froz_dense[k] for k in bk], dim=0).to(device)
            sm, rm, _ = model(x)
        else:
            x = torch.stack([b7_dict[k] for k in bk], dim=0).to(device)
            sm, rm, _ = model(x)
        sm, rm = sm.cpu(), rm.cpu()
        for idx, k in enumerate(bk):
            sem_out[k] = sm[idx]
            role_out[k] = rm[idx] if is_L0 else None
    return sem_out, (role_out if is_L0 else None)


def main():
    device = get_device()
    print(f"Running Paper 1 Final Audit Closures (L0_Frozen_SameHead & L1a_LoRA_Single_Anchored) on {device}", flush=True)
    b7_cache = build_or_load_block7_cache(device)
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()
    train_ids = [i for i, d in enumerate(dataset) if d['p']['split'] == 'train']

    full_vit = load_dinov2_tail(torch.device('cpu'))
    base_blocks_8_11 = nn.ModuleList([copy.deepcopy(full_vit.blocks[i]) for i in range(8, 12)])
    base_norm = copy.deepcopy(full_vit.norm)
    del full_vit

    # 1. Train & Evaluate on SOCO Confirm
    soco_models = {}
    soco_rows = {}
    trained_L0, trained_L1a = {}, {}

    runs_L0, rows_L0 = [], []
    for seed in SEEDS:
        m, dt = train_L0(dataset, dense_norm, grid_xy, train_ids, seed, device)
        torch.save(m.state_dict(), WEIGHTS_DIR / f'L0_Frozen_SameHead_seed{seed}.pt')
        trained_L0[seed] = m
        summ, rows = evaluate_pair_predictor(
            make_L0_predictor(m, dense_norm, device, alpha=0.5),
            dataset, 'confirm', 'L0_Frozen_SameHead', seed,
            out_dir=OUTPUTS_DIR, params=sum(p.numel() for p in m.parameters()), device=device,
        )
        summ['train_seconds'] = dt
        runs_L0.append(summ)
        rows_L0.extend(rows)
        cm = summ['category_macro']
        print(f"[SOCO L0_Frozen_SameHead       seed={seed}] params={summ['params']} train={dt:.1f}s | SOC={cm['soc_pck']:.2f} CC_sq={cm['cc_samequery_pck']:.2f} CC_role={cm['cc_rolebranch_pck']:.2f} WR={cm['dense_wrong_role']:.2f}", flush=True)
    soco_models['L0_Frozen_SameHead'] = aggregate_model_runs(runs_L0)
    soco_rows['L0_Frozen_SameHead'] = rows_L0

    runs_L1a, rows_L1a = [], []
    for seed in SEEDS:
        m, dt = train_L1a(base_blocks_8_11, base_norm, dataset, b7_cache, dense_norm, grid_xy, train_ids, seed, device)
        trainable_sd = {k: v.cpu() for k, v in m.state_dict().items() if k in {n for n, p in m.named_parameters() if p.requires_grad}}
        torch.save(trainable_sd, WEIGHTS_DIR / f'L1a_LoRA_Single_Anchored_seed{seed}.pt')
        trained_L1a[seed] = m
        summ, rows = evaluate_pair_predictor(
            make_lora_predictor(m, b7_cache, device, single_head=True, alpha=0.5),
            dataset, 'confirm', 'L1a_LoRA_Single_Anchored', seed,
            out_dir=OUTPUTS_DIR, params=m.trainable_param_count(), device=device,
        )
        summ['train_seconds'] = dt
        runs_L1a.append(summ)
        rows_L1a.extend(rows)
        cm = summ['category_macro']
        print(f"[SOCO L1a_LoRA_Single_Anchored seed={seed}] params={summ['params']} train={dt:.1f}s | SOC={cm['soc_pck']:.2f} CC_sq={cm['cc_samequery_pck']:.2f} CC_role={cm['cc_rolebranch_pck']:.2f} WR={cm['dense_wrong_role']:.2f}", flush=True)
    soco_models['L1a_LoRA_Single_Anchored'] = aggregate_model_runs(runs_L1a)
    soco_rows['L1a_LoRA_Single_Anchored'] = rows_L1a

    # Load existing SOCO models (L1, L2, L3, G1_NoAttn, frozen)
    lora_res = json.loads((OUT / 'lora_results.json').read_text())
    for k in ['frozen', 'G1_NoAttn', 'L1_LoRA_Single', 'L2_LoRA_Dual_NoAnchor', 'L3_LoRA_Dual_Anchored']:
        soco_models[k] = lora_res['models'][k]
        soco_rows[k] = load_csv_rows(k, [0] if k == 'frozen' else SEEDS)

    pv_soco = {k: to_pairvals_full(v) for k, v in soco_rows.items()}
    soco_comps = {
        'L3_minus_L0_pure_lora':       bootstrap_linear_combo(pv_soco, {'L3_LoRA_Dual_Anchored': 1.0, 'L0_Frozen_SameHead': -1.0}),
        'L3_minus_L1a_dual_given_anc': bootstrap_linear_combo(pv_soco, {'L3_LoRA_Dual_Anchored': 1.0, 'L1a_LoRA_Single_Anchored': -1.0}),
        'L1a_minus_L1_anc_given_sgl':  bootstrap_linear_combo(pv_soco, {'L1a_LoRA_Single_Anchored': 1.0, 'L1_LoRA_Single': -1.0}),
        'L3_minus_L2_anc_given_dual':  bootstrap_linear_combo(pv_soco, {'L3_LoRA_Dual_Anchored': 1.0, 'L2_LoRA_Dual_NoAnchor': -1.0}),
        'L2_minus_L1_dual_given_noan': bootstrap_linear_combo(pv_soco, {'L2_LoRA_Dual_NoAnchor': 1.0, 'L1_LoRA_Single': -1.0}),
    }

    # 2. Evaluate L0, L1a, AND re-evaluate L1, L2, L3, frozen on FULL SPair-71k (12,234 pairs) for exact paired bootstrap
    pairs, img_data, manifest_sha = stream_spair_test()
    keys, spair_b7, spair_froz, pads = extract_spair_dinov2_features(img_data, device)

    spair_models = {}
    spair_prs = {}

    for mname, is_L0, mdict in [('L0_Frozen_SameHead', True, trained_L0), ('L1a_LoRA_Single_Anchored', False, trained_L1a)]:
        s_summs, s_prs = [], []
        for sd in SEEDS:
            sem_m, role_m = encode_spair_custom(keys, mdict[sd], is_L0, spair_b7, spair_froz, device)
            summ, pr = evaluate_spair_pairs(pairs, pads, sem_m, role_m, alpha=0.5)
            s_summs.append(summ)
            s_prs.append(pr)
        spair_prs[mname] = seed_average_pair_records(s_prs)
        spair_models[mname] = {
            'all18_macro': {k_m: float(np.mean([s['all18_macro'][k_m] for s in s_summs])) for k_m in s_summs[0]['all18_macro']},
            'unseen15_macro': {k_m: float(np.mean([s['unseen15_macro'][k_m] for s in s_summs])) for k_m in s_summs[0]['unseen15_macro']},
        }
        a18, u15 = spair_models[mname]['all18_macro'], spair_models[mname]['unseen15_macro']
        print(f"[SPair {mname:24s}] All18: PCK@0.1={a18['all_pck10']:.2f} SymPCK={a18['sym_pck10']:.2f} WR={a18['sym_wr']:.2f} | Unseen15: PCK@0.1={u15['all_pck10']:.2f} SymPCK={u15['sym_pck10']:.2f} WR={u15['sym_wr']:.2f}", flush=True)

    # Also compute pair_records for L1, L2, L3 on SPair for exact paired bootstrap
    spair_prev = json.loads((OUT / 'spair_zeroshot_results.json').read_text())
    for mname in ['L1_LoRA_Single', 'L2_LoRA_Dual_NoAnchor', 'L3_LoRA_Dual_Anchored']:
        s_prs = []
        for sd in SEEDS:
            sem_m, role_m = encode_all_images_for_model(keys, mname, sd, spair_b7, spair_froz, base_blocks_8_11, base_norm, device)
            _, pr = evaluate_spair_pairs(pairs, pads, sem_m, role_m, alpha=0.5)
            s_prs.append(pr)
        spair_prs[mname] = seed_average_pair_records(s_prs)
        spair_models[mname] = {
            'all18_macro': spair_prev['models'][mname]['all18_macro'],
            'unseen15_macro': spair_prev['models'][mname]['unseen15_macro'],
        }

    all18_cats = sorted(spair_prev['models']['frozen']['by_category'].keys())
    unseen15_cats = [c for c in all18_cats if c not in SOCO_SEEN_SPAIR_CATS]
    spair_comps = {}
    for kname, cand, ref in [
        ('L3_minus_L0_pure_lora',       'L3_LoRA_Dual_Anchored', 'L0_Frozen_SameHead'),
        ('L3_minus_L1a_dual_given_anc', 'L3_LoRA_Dual_Anchored', 'L1a_LoRA_Single_Anchored'),
        ('L1a_minus_L1_anc_given_sgl',  'L1a_LoRA_Single_Anchored', 'L1_LoRA_Single'),
        ('L3_minus_L2_anc_given_dual',  'L3_LoRA_Dual_Anchored', 'L2_LoRA_Dual_NoAnchor'),
        ('L2_minus_L1_dual_given_noan', 'L2_LoRA_Dual_NoAnchor', 'L1_LoRA_Single'),
    ]:
        spair_comps[kname] = {
            'all18': {m: bootstrap_spair(spair_prs[cand], spair_prs[ref], all18_cats, m) for m in ['all10', 'sym10', 'wr']},
            'unseen15': {m: bootstrap_spair(spair_prs[cand], spair_prs[ref], unseen15_cats, m) for m in ['all10', 'sym10', 'wr']},
        }

    out_obj = {
        'soco_models': {k: soco_models[k] for k in ['frozen', 'G1_NoAttn', 'L0_Frozen_SameHead', 'L1_LoRA_Single', 'L1a_LoRA_Single_Anchored', 'L2_LoRA_Dual_NoAnchor', 'L3_LoRA_Dual_Anchored']},
        'soco_comparisons': soco_comps,
        'spair_models': spair_models,
        'spair_comparisons': spair_comps,
    }
    (OUT / 'paper1_complete_2x2_audit.json').write_text(json.dumps(out_obj, indent=2))

    print("\n=== SOCO 2x2 + PURE LORA SUMMARY ===", flush=True)
    for k in ['L0_Frozen_SameHead', 'L1_LoRA_Single', 'L1a_LoRA_Single_Anchored', 'L2_LoRA_Dual_NoAnchor', 'L3_LoRA_Dual_Anchored']:
        m = soco_models[k]['mean']
        print(f"{k:26s} | params={soco_models[k]['params']:7d} | SOC={m['soc_pck']:.2f} | CC_sq={m['cc_samequery_pck']:.2f} | CC_role={m['cc_rolebranch_pck']:.2f} | WR={m['dense_wrong_role']:.2f}", flush=True)
    print("\n=== SOCO KEY BOOTSTRAP DELTAS (pp, 95% CI) ===", flush=True)
    for k, c in soco_comps.items():
        d, ci = c['delta_pp'], c['ci95_pp']
        print(f"{k:28s} | SOC {d['soc_pck']:+.2f} [{ci['soc_pck'][0]:+.2f}, {ci['soc_pck'][1]:+.2f}] | CC_sq {d['cc_samequery_pck']:+.2f} [{ci['cc_samequery_pck'][0]:+.2f}, {ci['cc_samequery_pck'][1]:+.2f}] | WR {d['dense_wrong_role']:+.2f} [{ci['dense_wrong_role'][0]:+.2f}, {ci['dense_wrong_role'][1]:+.2f}]", flush=True)
    print("\n=== SPAIR-71K (12,234 PAIRS) KEY BOOTSTRAP DELTAS (pp, 95% CI) ===", flush=True)
    for k, c in spair_comps.items():
        for sub in ['all18', 'unseen15']:
            cs = c[sub]
            print(f"{k:28s} [{sub:8s}] | All10 {cs['all10']['delta_pp']:+.2f} [{cs['all10']['ci95_pp'][0]:+.2f}, {cs['all10']['ci95_pp'][1]:+.2f}] | Sym10 {cs['sym10']['delta_pp']:+.2f} [{cs['sym10']['ci95_pp'][0]:+.2f}, {cs['sym10']['ci95_pp'][1]:+.2f}] | WR {cs['wr']['delta_pp']:+.2f} [{cs['wr']['ci95_pp'][0]:+.2f}, {cs['wr']['ci95_pp'][1]:+.2f}]", flush=True)
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
