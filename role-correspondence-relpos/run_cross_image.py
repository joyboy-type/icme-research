import collections
import csv
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
from models import RelPosBackbone, RelPosBlock
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

OUT = pathlib.Path('/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos')
OUTPUTS_DIR = OUT / 'outputs'
WEIGHTS_DIR = OUTPUTS_DIR / 'weights'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Exact same frozen hyperparameters as G1 (zero extra tuning)
CFG = {
    'dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'steps': 300,
    'batch_size': 8,
    'lr': 5e-4,
    'weight_decay': 1e-2,
    'alpha_train': 0.6,
    'alpha_eval': 0.5,
    'w_cc': 0.5,
    'w_soc_role': 0.3,
    'w_soc_comb': 1.0,
    'w_anchor': 10.0,
    'tau': 0.07,
}


class BidirectionalCrossRoleBlock(nn.Module):
    """
    1-layer weight-shared Bidirectional Cross-Image Attention + FFN on role tokens.
    Parameter-matched to 1 RelPosBlock(use_relpos=False) (~132.7K vs 132.5K params).
    """
    def __init__(self, dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def _cross(self, x_q, x_kv):
        B, N, C = x_q.shape
        q = self.q_proj(self.norm_q(x_q)).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(self.norm_kv(x_kv)).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x_q + self.proj_drop(self.proj(out))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, h_s, h_t):
        # Symmetric bidirectional interaction: both computed from pre-update (h_s, h_t)
        r_s = self._cross(h_s, h_t)
        r_t = self._cross(h_t, h_s)
        return r_s, r_t


class RoleCorrespondenceAdapter(nn.Module):
    """
    Unified architecture for:
      - mode='intra_param_match' (G1-ParamMatch): 2-layer shared intra-attn + 1-layer role intra-attn
      - mode='cross_role'        (C1 / C2):       2-layer shared intra-attn + 1-layer bidirectional cross-attn
    Semantic branch z_sem is strictly single-image in all modes.
    """
    def __init__(self, mode='cross_role', in_dim=384, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        assert mode in ('intra_param_match', 'cross_role')
        self.mode = mode
        self.backbone = RelPosBackbone(
            in_dim=in_dim, dim=dim, num_heads=num_heads, num_layers=2, dropout=dropout, use_relpos=False
        )
        self.sem_proj = nn.Linear(dim, in_dim, bias=False)
        nn.init.zeros_(self.sem_proj.weight)

        if mode == 'intra_param_match':
            self.role_block = RelPosBlock(dim=dim, num_heads=num_heads, mlp_ratio=2.0, dropout=dropout, use_relpos=False)
        else:
            self.role_block = BidirectionalCrossRoleBlock(dim=dim, num_heads=num_heads, mlp_ratio=2.0, dropout=dropout)

        self.role_norm = nn.LayerNorm(dim)
        self.role_proj = nn.Linear(dim, in_dim, bias=True)
        nn.init.normal_(self.role_proj.weight, std=0.02)
        nn.init.zeros_(self.role_proj.bias)

    def encode_single(self, feat_map_bchw):
        B, C, H, W = feat_map_bchw.shape
        tokens, h = self.backbone(feat_map_bchw)
        z_sem = F.normalize(tokens + self.sem_proj(h), dim=-1)
        anchor_loss = (1.0 - (z_sem * tokens).sum(dim=-1)).mean()
        sem_map = z_sem.transpose(1, 2).reshape(B, C, H, W)
        return h, sem_map, anchor_loss

    def encode_role_pair(self, h_s, h_t):
        # h_s, h_t: (B, 400, dim)
        B = h_s.shape[0]
        if self.mode == 'intra_param_match':
            r_s = self.role_block(h_s)
            r_t = self.role_block(h_t)
        else:
            r_s, r_t = self.role_block(h_s, h_t)
        z_s = F.normalize(self.role_proj(self.role_norm(r_s)), dim=-1).transpose(1, 2).reshape(B, 384, 20, 20)
        z_t = F.normalize(self.role_proj(self.role_norm(r_t)), dim=-1).transpose(1, 2).reshape(B, 384, 20, 20)
        return z_s, z_t


def apply_dual_softmax_penalty(sim_kp_all, sim_400_all, tau=0.07):
    """
    Exact Dual-Softmax mutual-consistency adjustment in similarity space:
      log(softmax_row(S/tau) * softmax_col(S/tau)) / 2
      = S/tau - 0.5 * logsumexp_row(S_400/tau) - const(query)
    Returns adjusted similarity matrix of same shape as sim_kp_all: (Ks, M),
    where M = Kt + 400 (during training) or 400 / Kt (during eval).
    sim_400_all: (400, M) similarities from all 400 source patches to the same M target points.
    Zero target GT leakage: column normalizer sums only over the 400 source patches!
    """
    col_lse = tau * torch.logsumexp(sim_400_all / tau, dim=0, keepdim=True)  # (1, M)
    return sim_kp_all - 0.5 * col_lse


def train_model(
    mode,
    use_dual_softmax,
    dataset,
    dense_norm,
    grid_xy,
    train_ids,
    seed,
    device,
):
    rng = np.random.default_rng(seed)
    schedule = [rng.choice(train_ids, CFG['batch_size'], replace=True).tolist() for _ in range(CFG['steps'])]
    torch.manual_seed(seed)

    model = RoleCorrespondenceAdapter(
        mode=mode, dim=CFG['dim'], num_heads=CFG['num_heads'], dropout=CFG['dropout']
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['steps'], eta_min=CFG['lr'] * 0.1)

    train_imgs = sorted({dataset[i]['s'] for i in train_ids} | {dataset[i]['t'] for i in train_ids})
    img_dev = {k: dense_norm[k].to(device) for k in train_imgs}
    pre = prepare_train_pairs_on_device(dataset, train_ids, grid_xy, device)

    tau = CFG['tau']
    alpha_tr = CFG['alpha_train']

    ts = time.perf_counter()
    trace = []
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
        x_batch = torch.stack([img_dev[nm] for nm in uniq_imgs], dim=0)
        h_batch, sem_batch, anchor_loss = model.encode_single(x_batch)

        s_indices = [img_to_idx[d['s']] for d in items]
        t_indices = [img_to_idx[d['t']] for d in items]
        h_s = h_batch[s_indices]  # (B, 400, 128)
        h_t = h_batch[t_indices]  # (B, 400, 128)
        role_s_batch, role_t_batch = model.encode_role_pair(h_s, h_t)

        cc_losses = []
        soc_role_losses = []
        soc_comb_losses = []

        for j, d in enumerate(items):
            si = s_indices[j]
            ti = t_indices[j]

            zs_sem = sample_dense_map(sem_batch[si], d['s_xy'])
            zt_sem = sample_dense_map(sem_batch[ti], d['t_xy'])
            ft_sem_flat = sem_batch[ti].flatten(1).T  # (400, 384)
            zt_sem_all = torch.cat([zt_sem, ft_sem_flat], dim=0)  # (Kt+400, 384)
            sim_sem_all = zs_sem @ zt_sem_all.T  # (Ks, Kt+400)

            zs_role = sample_dense_map(role_s_batch[j], d['s_xy'])
            zt_role = sample_dense_map(role_t_batch[j], d['t_xy'])
            ft_role_flat = role_t_batch[j].flatten(1).T  # (400, 384)
            zt_role_all = torch.cat([zt_role, ft_role_flat], dim=0)  # (Kt+400, 384)
            sim_role_all = zs_role @ zt_role_all.T  # (Ks, Kt+400)

            if use_dual_softmax:
                fs_role_flat = role_s_batch[j].flatten(1).T  # (400, 384)
                sim_role_400_all = fs_role_flat @ zt_role_all.T  # (400, Kt+400)
                sim_role_eff = apply_dual_softmax_penalty(sim_role_all, sim_role_400_all, tau=tau)

                fs_sem_flat = sem_batch[si].flatten(1).T  # (400, 384)
                sim_comb_400_all = (fs_sem_flat @ zt_sem_all.T).detach() + alpha_tr * sim_role_400_all
                sim_comb_raw = sim_sem_all.detach() + alpha_tr * sim_role_all
                sim_comb_eff = apply_dual_softmax_penalty(sim_comb_raw, sim_comb_400_all, tau=tau)
            else:
                sim_role_eff = sim_role_all
                sim_comb_eff = sim_sem_all.detach() + alpha_tr * sim_role_all

            lc = contrastive_pair_loss(sim_sem_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
            if lc is not None:
                cc_losses.append(lc)
            lsr = contrastive_pair_loss(sim_role_eff, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            if lsr is not None:
                soc_role_losses.append(lsr)
            lsc = contrastive_pair_loss(sim_comb_eff, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            if lsc is not None:
                soc_comb_losses.append(lsc)

        loss = (
            CFG['w_anchor'] * anchor_loss
            + CFG['w_cc'] * torch.stack(cc_losses).mean()
            + CFG['w_soc_role'] * torch.stack(soc_role_losses).mean()
            + CFG['w_soc_comb'] * torch.stack(soc_comb_losses).mean()
        )
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == CFG['steps'] - 1:
            trace.append({
                'step': step,
                'loss': float(loss.detach().cpu()),
                'anchor': float(anchor_loss.detach().cpu()),
            })

    sync_device(device)
    model.eval()
    return model, time.perf_counter() - ts, trace


def make_predictor(model, dense_norm, device, use_dual_softmax=False, alpha=0.5, tau=0.07):
    model.eval()
    def predict_pair(d):
        x = torch.stack([dense_norm[d['s']], dense_norm[d['t']]], dim=0).to(device)
        h_b, sem_b, _ = model.encode_single(x)
        role_s, role_t = model.encode_role_pair(h_b[0:1], h_b[1:2])

        s_xy = d['s_xy'].to(device)
        t_xy = d['t_xy'].to(device)

        zs_sem = sample_dense_map(sem_b[0], s_xy)
        zt_sem = sample_dense_map(sem_b[1], t_xy)
        ft_sem = sem_b[1].flatten(1).T

        zs_role = sample_dense_map(role_s[0], s_xy)
        zt_role = sample_dense_map(role_t[0], t_xy)
        ft_role = role_t[0].flatten(1).T

        sd_cc = zs_sem @ ft_sem.T
        sc_cc = zs_sem @ zt_sem.T

        sd_soc = sd_cc + alpha * (zs_role @ ft_role.T)
        sc_soc = sc_cc + alpha * (zs_role @ zt_role.T)

        if use_dual_softmax:
            fs_sem = sem_b[0].flatten(1).T
            fs_role = role_s[0].flatten(1).T
            sd_400 = (fs_sem @ ft_sem.T) + alpha * (fs_role @ ft_role.T)
            sc_400 = (fs_sem @ zt_sem.T) + alpha * (fs_role @ zt_role.T)
            sd_soc = apply_dual_softmax_penalty(sd_soc, sd_400, tau=tau)
            sc_soc = apply_dual_softmax_penalty(sc_soc, sc_400, tau=tau)

        return sd_cc, sd_soc, sc_cc, sc_soc
    return predict_pair


def main():
    device = get_device()
    print(f"Running Cross-Image Role Attention & Dual-Softmax experiment on device={device}", flush=True)
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()
    train_ids = [i for i, d in enumerate(dataset) if d['p']['split'] == 'train']

    specs = [
        ('G1_param_match',    'intra_param_match', False),
        ('C1_cross_role',     'cross_role',        False),
        ('C2_cross_dualsm',   'cross_role',        True),
    ]

    all_models = {}
    all_rows = {}

    for name, mode, use_ds in specs:
        runs = []
        rows_all = []
        for seed in SEEDS:
            model, tr_sec, trace = train_model(
                mode, use_ds, dataset, dense_norm, grid_xy, train_ids, seed, device
            )
            torch.save(model.state_dict(), WEIGHTS_DIR / f'{name}_seed{seed}.pt')
            summ, rows = evaluate_pair_predictor(
                make_predictor(model, dense_norm, device, use_dual_softmax=use_ds, alpha=CFG['alpha_eval'], tau=CFG['tau']),
                dataset,
                'confirm',
                name,
                seed,
                out_dir=OUTPUTS_DIR,
                params=sum(p.numel() for p in model.parameters()),
                device=device,
            )
            summ['train_seconds'] = tr_sec
            summ['loss_trace'] = trace
            runs.append(summ)
            rows_all.extend(rows)
            m = summ['category_macro']
            print(
                f"[{name:16s} seed={seed}] params={summ['params']} train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | "
                f"SOC={m['soc_pck']:.2f} CC_sq={m['cc_samequery_pck']:.2f} CC_role={m['cc_rolebranch_pck']:.2f} WR={m['dense_wrong_role']:.2f}",
                flush=True,
            )
        all_models[name] = aggregate_model_runs(runs)
        all_rows[name] = rows_all

    # Load G1 and Frozen directly from saved outputs
    abl = json.loads((OUT / 'ablation_2x2.json').read_text())
    all_models['G1'] = abl['models']['G1_plain_attn_plain_dual']
    all_rows['G1'] = load_csv_rows('G1_plain_attn_plain_dual', SEEDS)
    all_models['frozen'] = abl['models']['frozen']
    all_rows['frozen'] = load_csv_rows('frozen', [0])

    pv_map = {k: to_pairvals_full(v) for k, v in all_rows.items()}

    comparisons = {
        'G1_param_match_vs_G1':     bootstrap_linear_combo(pv_map, {'G1_param_match': 1.0, 'G1': -1.0}),
        'C1_vs_G1':                 bootstrap_linear_combo(pv_map, {'C1_cross_role': 1.0, 'G1': -1.0}),
        'C1_vs_G1_param_match':     bootstrap_linear_combo(pv_map, {'C1_cross_role': 1.0, 'G1_param_match': -1.0}),
        'C2_vs_G1':                 bootstrap_linear_combo(pv_map, {'C2_cross_dualsm': 1.0, 'G1': -1.0}),
        'C2_vs_G1_param_match':     bootstrap_linear_combo(pv_map, {'C2_cross_dualsm': 1.0, 'G1_param_match': -1.0}),
        'C2_vs_C1':                 bootstrap_linear_combo(pv_map, {'C2_cross_dualsm': 1.0, 'C1_cross_role': -1.0}),
        'C1_vs_frozen':             bootstrap_linear_combo(pv_map, {'C1_cross_role': 1.0, 'frozen': -1.0}),
        'C2_vs_frozen':             bootstrap_linear_combo(pv_map, {'C2_cross_dualsm': 1.0, 'frozen': -1.0}),
    }

    out_obj = {
        'protocol': {
            'description': 'Cross-Image Bidirectional Role Attention (C1) + Dual-Softmax Mutual Consistency (C2) vs G1 & G1-ParamMatch',
            'cfg': CFG,
            'seeds': SEEDS,
            'bootstrap_method': '4000 replicates, within-category shared image vertex bootstrap, seed 20261121',
        },
        'models': {k: all_models[k] for k in ['frozen', 'G1', 'G1_param_match', 'C1_cross_role', 'C2_cross_dualsm']},
        'comparisons': comparisons,
    }
    (OUT / 'cross_image_results.json').write_text(json.dumps(out_obj, indent=2))

    print("\n=== CROSS-IMAGE EXPERIMENT SUMMARY ===", flush=True)
    for k in ['frozen', 'G1', 'G1_param_match', 'C1_cross_role', 'C2_cross_dualsm']:
        info = all_models[k]
        m = info['mean']
        cats_str = " ".join(f"{c[:4]}={info['by_category'][c]['soc_pck']:.1f}" for c in ['bench', 'dishwasher', 'folding_chair', 'train'])
        print(
            f"{k:16s} | params={info['params']:7d} | SOC={m['soc_pck']:.2f} | CC_sq={m['cc_samequery_pck']:.2f} | "
            f"CC_role={m['cc_rolebranch_pck']:.2f} | WR={m['dense_wrong_role']:.2f} | {cats_str}",
            flush=True,
        )
    print("\n=== KEY BOOTSTRAP DELTAS (pp, 95% CI) ===", flush=True)
    for k in ['C1_vs_G1', 'C1_vs_G1_param_match', 'C2_vs_G1', 'C2_vs_G1_param_match', 'C2_vs_C1', 'C2_vs_frozen']:
        c = comparisons[k]
        d = c['delta_pp']
        ci = c['ci95_pp']
        print(
            f"{k:22s} | SOC {d['soc_pck']:+.2f} [{ci['soc_pck'][0]:+.2f}, {ci['soc_pck'][1]:+.2f}] | "
            f"CC_sq {d['cc_samequery_pck']:+.2f} [{ci['cc_samequery_pck'][0]:+.2f}, {ci['cc_samequery_pck'][1]:+.2f}] | "
            f"CC_role {d['cc_rolebranch_pck']:+.2f} [{ci['cc_rolebranch_pck'][0]:+.2f}, {ci['cc_rolebranch_pck'][1]:+.2f}] | "
            f"WR {d['dense_wrong_role']:+.2f} [{ci['dense_wrong_role'][0]:+.2f}, {ci['dense_wrong_role'][1]:+.2f}] | "
            f"cats={c['category_soc_delta_pp']}",
            flush=True,
        )
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
