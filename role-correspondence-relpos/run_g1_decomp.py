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
from models import RelPosBackbone
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
    'tau': 0.07,
}


class PerPatchMLPBlock(nn.Module):
    """
    Parameter-matched per-patch MLP block replacing RelPosBlock(use_relpos=False).
    132,352 params vs RelPosBlock's 132,480 params (0.1% diff); zero cross-patch mixing.
    """
    def __init__(self, dim=128, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.mlp1(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        return x


class PerPatchBackbone(nn.Module):
    def __init__(self, in_dim=384, dim=128, num_layers=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            PerPatchMLPBlock(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, feat_map_bchw):
        B, C, H, W = feat_map_bchw.shape
        tokens = feat_map_bchw.flatten(2).transpose(1, 2)
        h = self.in_norm(self.in_proj(tokens))
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return tokens, h


class G1DecompAdapter(nn.Module):
    def __init__(self, use_attn=True, in_dim=384, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        if use_attn:
            self.backbone = RelPosBackbone(
                in_dim=in_dim, dim=dim, num_heads=num_heads, num_layers=2, dropout=dropout, use_relpos=False
            )
        else:
            self.backbone = PerPatchBackbone(
                in_dim=in_dim, dim=dim, num_layers=2, dropout=dropout
            )
        self.sem_proj = nn.Linear(dim, in_dim, bias=False)
        nn.init.zeros_(self.sem_proj.weight)
        self.role_proj = nn.Linear(dim, in_dim, bias=True)
        nn.init.normal_(self.role_proj.weight, std=0.02)
        nn.init.zeros_(self.role_proj.bias)

    def forward(self, feat_map_bchw):
        B, C, H, W = feat_map_bchw.shape
        tokens, h = self.backbone(feat_map_bchw)
        z_sem = F.normalize(tokens + self.sem_proj(h), dim=-1)
        z_role = F.normalize(self.role_proj(h), dim=-1)
        anchor_loss = (1.0 - (z_sem * tokens).sum(dim=-1)).mean()
        sem_map = z_sem.transpose(1, 2).reshape(B, C, H, W)
        role_map = z_role.transpose(1, 2).reshape(B, C, H, W)
        return sem_map, role_map, anchor_loss


def train_g1_variant(
    use_attn,
    w_anchor,
    use_detach,
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

    model = G1DecompAdapter(
        use_attn=use_attn, dim=CFG['dim'], num_heads=CFG['num_heads'], dropout=CFG['dropout']
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
        sem_batch, role_batch, anchor_loss = model(x_batch)

        cc_losses = []
        soc_role_losses = []
        soc_comb_losses = []

        for d in items:
            si = img_to_idx[d['s']]
            ti = img_to_idx[d['t']]

            zs_sem = sample_dense_map(sem_batch[si], d['s_xy'])
            zt_sem = sample_dense_map(sem_batch[ti], d['t_xy'])
            ft_sem_flat = sem_batch[ti].flatten(1).T
            sim_sem_all = torch.cat([zs_sem @ zt_sem.T, zs_sem @ ft_sem_flat.T], dim=-1)

            zs_role = sample_dense_map(role_batch[si], d['s_xy'])
            zt_role = sample_dense_map(role_batch[ti], d['t_xy'])
            ft_role_flat = role_batch[ti].flatten(1).T
            sim_role_all = torch.cat([zs_role @ zt_role.T, zs_role @ ft_role_flat.T], dim=-1)

            sem_for_comb = sim_sem_all.detach() if use_detach else sim_sem_all
            sim_comb_all = sem_for_comb + alpha_tr * sim_role_all

            lc = contrastive_pair_loss(sim_sem_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
            if lc is not None:
                cc_losses.append(lc)
            lsr = contrastive_pair_loss(sim_role_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            if lsr is not None:
                soc_role_losses.append(lsr)
            lsc = contrastive_pair_loss(sim_comb_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            if lsc is not None:
                soc_comb_losses.append(lsc)

        loss = (
            w_anchor * anchor_loss
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


def make_g1_predictor(model, dense_norm, device, alpha=0.5):
    model.eval()
    def predict_pair(d):
        x = torch.stack([dense_norm[d['s']], dense_norm[d['t']]], dim=0).to(device)
        sem_b, role_b, _ = model(x)
        s_xy = d['s_xy'].to(device)
        t_xy = d['t_xy'].to(device)

        zs_sem = sample_dense_map(sem_b[0], s_xy)
        zt_sem = sample_dense_map(sem_b[1], t_xy)
        ft_sem = sem_b[1].flatten(1).T

        zs_role = sample_dense_map(role_b[0], s_xy)
        zt_role = sample_dense_map(role_b[1], t_xy)
        ft_role = role_b[1].flatten(1).T

        sd_cc = zs_sem @ ft_sem.T
        sc_cc = zs_sem @ zt_sem.T
        sd_soc = sd_cc + alpha * (zs_role @ ft_role.T)
        sc_soc = sc_cc + alpha * (zs_role @ zt_role.T)
        return sd_cc, sd_soc, sc_cc, sc_soc
    return predict_pair


def main():
    device = get_device()
    print(f"Running G1 internal component decomposition on device={device}", flush=True)
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()
    train_ids = [i for i, d in enumerate(dataset) if d['p']['split'] == 'train']

    specs = [
        ('G1_NoAttn',   False, 10.0, True),
        ('G1_NoAnchor', True,   0.0, True),
        ('G1_NoDetach', True,  10.0, False),
    ]

    all_models = {}
    all_rows = {}

    for name, use_attn, w_anchor, use_detach in specs:
        runs = []
        rows_all = []
        for seed in SEEDS:
            model, tr_sec, trace = train_g1_variant(
                use_attn, w_anchor, use_detach, dataset, dense_norm, grid_xy, train_ids, seed, device
            )
            torch.save(model.state_dict(), WEIGHTS_DIR / f'{name}_seed{seed}.pt')
            summ, rows = evaluate_pair_predictor(
                make_g1_predictor(model, dense_norm, device, alpha=CFG['alpha_eval']),
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
                f"[{name:12s} seed={seed}] params={summ['params']} train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | "
                f"SOC={m['soc_pck']:.2f} CC_sq={m['cc_samequery_pck']:.2f} CC_role={m['cc_rolebranch_pck']:.2f} WR={m['dense_wrong_role']:.2f}",
                flush=True,
            )
        all_models[name] = aggregate_model_runs(runs)
        all_rows[name] = rows_all

    # Reuse G1 and Frozen from existing saved artifacts
    abl = json.loads((OUT / 'ablation_2x2.json').read_text())
    all_models['G1'] = abl['models']['G1_plain_attn_plain_dual']
    all_rows['G1'] = load_csv_rows('G1_plain_attn_plain_dual', SEEDS)
    all_models['frozen'] = abl['models']['frozen']
    all_rows['frozen'] = load_csv_rows('frozen', [0])

    pv_map = {k: to_pairvals_full(v) for k, v in all_rows.items()}

    comparisons = {
        'G1_minus_G1_NoAttn':   bootstrap_linear_combo(pv_map, {'G1': 1.0, 'G1_NoAttn': -1.0}),
        'G1_minus_G1_NoAnchor': bootstrap_linear_combo(pv_map, {'G1': 1.0, 'G1_NoAnchor': -1.0}),
        'G1_minus_G1_NoDetach': bootstrap_linear_combo(pv_map, {'G1': 1.0, 'G1_NoDetach': -1.0}),
        'G1_NoAttn_minus_frozen':   bootstrap_linear_combo(pv_map, {'G1_NoAttn': 1.0, 'frozen': -1.0}),
        'G1_NoAnchor_minus_frozen': bootstrap_linear_combo(pv_map, {'G1_NoAnchor': 1.0, 'frozen': -1.0}),
        'G1_NoDetach_minus_frozen': bootstrap_linear_combo(pv_map, {'G1_NoDetach': 1.0, 'frozen': -1.0}),
    }

    out_obj = {
        'protocol': {
            'description': 'G1 internal 3-component leave-one-out ablation (Self-Attn, Semantic Anchor w_anchor=10, Stop-Grad Detach)',
            'cfg': CFG,
            'seeds': SEEDS,
            'bootstrap_method': '4000 replicates, within-category shared image vertex bootstrap, seed 20261121',
        },
        'models': {k: all_models[k] for k in ['frozen', 'G1', 'G1_NoAttn', 'G1_NoAnchor', 'G1_NoDetach']},
        'comparisons': comparisons,
    }
    (OUT / 'g1_decomp_results.json').write_text(json.dumps(out_obj, indent=2))

    print("\n=== G1 COMPONENT DECOMPOSITION SUMMARY ===", flush=True)
    for k in ['frozen', 'G1', 'G1_NoAttn', 'G1_NoAnchor', 'G1_NoDetach']:
        info = all_models[k]
        m = info['mean']
        cats_str = " ".join(f"{c[:4]}={info['by_category'][c]['soc_pck']:.1f}" for c in ['bench', 'dishwasher', 'folding_chair', 'train'])
        print(
            f"{k:12s} | params={info['params']:7d} | SOC={m['soc_pck']:.2f} | CC_sq={m['cc_samequery_pck']:.2f} | "
            f"CC_role={m['cc_rolebranch_pck']:.2f} | WR={m['dense_wrong_role']:.2f} | {cats_str}",
            flush=True,
        )
    print("\n=== LEAVE-ONE-OUT NET CONTRIBUTIONS (G1 minus Ablated, pp, 95% CI) ===", flush=True)
    for k in ['G1_minus_G1_NoAttn', 'G1_minus_G1_NoAnchor', 'G1_minus_G1_NoDetach']:
        c = comparisons[k]
        d = c['delta_pp']
        ci = c['ci95_pp']
        print(
            f"{k:22s} | SOC {d['soc_pck']:+.2f} [{ci['soc_pck'][0]:+.2f}, {ci['soc_pck'][1]:+.2f}] | "
            f"CC_sq {d['cc_samequery_pck']:+.2f} [{ci['cc_samequery_pck'][0]:+.2f}, {ci['cc_samequery_pck'][1]:+.2f}] | "
            f"CC_role {d['cc_rolebranch_pck']:+.2f} [{ci['cc_rolebranch_pck'][0]:+.2f}, {ci['cc_rolebranch_pck'][1]:+.2f}] | "
            f"WR {d['dense_wrong_role']:+.2f} [{ci['dense_wrong_role'][0]:+.2f}, {ci['dense_wrong_role'][1]:+.2f}]",
            flush=True,
        )
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
