import copy
import json
import math
import pathlib
import time
import numpy as np
import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from data_utils import (
    PILOT_DIR,
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

OUT = pathlib.Path('/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos')
OUTPUTS_DIR = OUT / 'outputs'
WEIGHTS_DIR = OUTPUTS_DIR / 'weights'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PATH = PILOT_DIR.parent / 'image_preference_round3_20261120/dinov2_vits14.safetensors'
BLOCK7_CACHE = OUT / 'cache_block7_fp32.pt'


def load_dinov2_tail(device):
    """Load DINOv2-S/14 and return deep-copied frozen blocks[8:12] + norm."""
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, dynamic_img_size=True)
    model.load_state_dict(load_file(str(WEIGHT_PATH)))
    model.eval()
    return model


def build_or_load_block7_cache(device):
    if BLOCK7_CACHE.exists():
        b7 = torch.load(BLOCK7_CACHE, map_location='cpu', weights_only=False)
        if len(b7) == 192:
            return b7

    print("Caching Block-7 tokens (fp32) for 192 images...", flush=True)
    model = load_dinov2_tail(device).to(device)
    manifest = json.loads((PILOT_DIR / 'image_manifest.json').read_text())
    feat_orig = torch.load(PILOT_DIR / 'features.pt', map_location='cpu', weights_only=False)

    b7_cache = {}
    max_err = 0.0
    ts = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(manifest), 16):
            batch = []
            names = []
            for n, _, _ in manifest[start:start + 16]:
                im = Image.open(PILOT_DIR / 'data' / n).convert('RGB')
                w, h = im.size
                pad = max(w, h)
                canvas = Image.new('RGB', (pad, pad), 'white')
                canvas.paste(im, (0, 0))
                im = canvas.resize((280, 280), Image.Resampling.BICUBIC)
                x = TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                batch.append(x)
                names.append(n)
            t = torch.stack(batch).to(device)
            t = model.patch_embed(t)
            t = model._pos_embed(t)
            t = model.patch_drop(t)
            t = model.norm_pre(t)
            for i in range(8):
                t = model.blocks[i](t)
            t_b7 = t.cpu()
            # Verify exact match through frozen blocks 8..11 + norm
            t_tail = t
            for i in range(8, 12):
                t_tail = model.blocks[i](t_tail)
            t_out = model.norm(t_tail)[:, 1:, :].reshape(-1, 20, 20, 384).permute(0, 3, 1, 2).cpu()
            for idx, n in enumerate(names):
                b7_cache[n] = t_b7[idx].contiguous()
                err = (t_out[idx] - feat_orig[n]['dense']).abs().max().item()
                if err > max_err:
                    max_err = err

    print(f"Cached {len(b7_cache)} images in {time.perf_counter()-ts:.1f}s | max_err vs features.pt = {max_err:.2e}", flush=True)
    assert max_err < 1e-4, f"Cache mismatch: {max_err}"
    torch.save(b7_cache, BLOCK7_CACHE)
    del model
    return b7_cache


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.scale = alpha / float(r)
        self.drop = nn.Dropout(dropout)
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(self.drop(x)))


class LoRAQKV(nn.Module):
    """Separate rank-r LoRA on Q, K, V inside stacked qkv Linear(D, 3D)."""
    def __init__(self, base_qkv: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_qkv
        for p in self.base.parameters():
            p.requires_grad = False
        d = base_qkv.in_features
        assert base_qkv.out_features == 3 * d
        self.scale = alpha / float(r)
        self.drop = nn.Dropout(dropout)
        self.q_A = nn.Linear(d, r, bias=False)
        self.q_B = nn.Linear(r, d, bias=False)
        self.k_A = nn.Linear(d, r, bias=False)
        self.k_B = nn.Linear(r, d, bias=False)
        self.v_A = nn.Linear(d, r, bias=False)
        self.v_B = nn.Linear(r, d, bias=False)
        for a in (self.q_A, self.k_A, self.v_A):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.q_B, self.k_B, self.v_B):
            nn.init.zeros_(b.weight)

    def forward(self, x):
        xd = self.drop(x)
        dq = self.q_B(self.q_A(xd))
        dk = self.k_B(self.k_A(xd))
        dv = self.v_B(self.v_A(xd))
        return self.base(x) + self.scale * torch.cat([dq, dk, dv], dim=-1)


class DINOv2LoRAAdapter(nn.Module):
    """
    DINOv2-S/14 Blocks 8..11 with rank-8 Q/K/V/Proj LoRA + lightweight head(s).
    - single_head=True  (L1): ~394K trainable params (98K LoRA + 296K single residual head)
    - single_head=False (L2/L3): ~395K trainable params (98K LoRA + 98K sem residual head + 198K role head)
    """
    def __init__(self, base_blocks_8_11, base_norm, single_head=False, rank=8, lora_alpha=16.0, dropout=0.1):
        super().__init__()
        self.single_head = single_head
        self.blocks = copy.deepcopy(base_blocks_8_11)
        self.norm = copy.deepcopy(base_norm)
        for p in self.blocks.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

        for blk in self.blocks:
            blk.attn.qkv = LoRAQKV(blk.attn.qkv, r=rank, alpha=lora_alpha, dropout=0.05)
            blk.attn.proj = LoRALinear(blk.attn.proj, r=rank, alpha=lora_alpha, dropout=0.05)

        if single_head:
            self.head_norm = nn.LayerNorm(384)
            self.head = nn.Sequential(
                nn.Linear(384, 384),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(384, 384, bias=False),
            )
            nn.init.zeros_(self.head[-1].weight)
        else:
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

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, b7_tokens, frozen_dense_norm=None):
        # b7_tokens: (B, 401, 384)
        t = b7_tokens
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)[:, 1:, :]  # (B, 400, 384)
        z_bb = F.normalize(t, dim=-1)
        B = t.shape[0]

        if self.single_head:
            z = F.normalize(z_bb + self.head(self.head_norm(z_bb)), dim=-1)
            fmap = z.transpose(1, 2).reshape(B, 384, 20, 20)
            anchor_loss = torch.zeros((), device=t.device)
            if frozen_dense_norm is not None:
                x0 = frozen_dense_norm.flatten(2).transpose(1, 2)
                anchor_loss = (1.0 - (z * x0).sum(dim=-1)).mean()
            return fmap, fmap, anchor_loss
        else:
            z_sem = F.normalize(z_bb + self.sem_head(self.sem_norm(z_bb)), dim=-1)
            z_role = F.normalize(self.role_head(self.role_norm(z_bb)), dim=-1)
            anchor_loss = torch.zeros((), device=t.device)
            if frozen_dense_norm is not None:
                x0 = frozen_dense_norm.flatten(2).transpose(1, 2)
                anchor_loss = (1.0 - (z_sem * x0).sum(dim=-1)).mean()
            sem_map = z_sem.transpose(1, 2).reshape(B, 384, 20, 20)
            role_map = z_role.transpose(1, 2).reshape(B, 384, 20, 20)
            return sem_map, role_map, anchor_loss


def train_lora_variant(
    base_blocks_8_11,
    base_norm,
    single_head,
    w_anchor,
    dataset,
    b7_cache,
    dense_norm,
    grid_xy,
    train_ids,
    seed,
    device,
    lr=3e-4,
    steps=300,
    batch_size=8,
    alpha_train=0.6,
    tau=0.07,
):
    rng = np.random.default_rng(seed)
    schedule = [rng.choice(train_ids, batch_size, replace=True).tolist() for _ in range(steps)]
    torch.manual_seed(seed)

    model = DINOv2LoRAAdapter(base_blocks_8_11, base_norm, single_head=single_head, rank=8, lora_alpha=16.0).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    train_imgs = sorted({dataset[i]['s'] for i in train_ids} | {dataset[i]['t'] for i in train_ids})
    b7_dev = {k: b7_cache[k].to(device) for k in train_imgs}
    dn_dev = {k: dense_norm[k].to(device) for k in train_imgs}
    pre = prepare_train_pairs_on_device(dataset, train_ids, grid_xy, device)

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
        b7_batch = torch.stack([b7_dev[nm] for nm in uniq_imgs], dim=0)
        x0_batch = torch.stack([dn_dev[nm] for nm in uniq_imgs], dim=0)
        sem_batch, role_batch, anchor_loss = model(b7_batch, x0_batch)

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

            if single_head:
                lc = contrastive_pair_loss(sim_sem_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
                ls = contrastive_pair_loss(sim_sem_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
                if lc is not None:
                    cc_losses.append(lc)
                if ls is not None:
                    soc_comb_losses.append(ls)
            else:
                zs_role = sample_dense_map(role_batch[si], d['s_xy'])
                zt_role = sample_dense_map(role_batch[ti], d['t_xy'])
                ft_role_flat = role_batch[ti].flatten(1).T
                sim_role_all = torch.cat([zs_role @ zt_role.T, zs_role @ ft_role_flat.T], dim=-1)
                sim_comb_all = sim_sem_all.detach() + alpha_train * sim_role_all

                lc = contrastive_pair_loss(sim_sem_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
                if lc is not None:
                    cc_losses.append(lc)
                lsr = contrastive_pair_loss(sim_role_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
                if lsr is not None:
                    soc_role_losses.append(lsr)
                lsc = contrastive_pair_loss(sim_comb_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
                if lsc is not None:
                    soc_comb_losses.append(lsc)

        if single_head:
            loss = 0.5 * torch.stack(cc_losses).mean() + 0.5 * torch.stack(soc_comb_losses).mean()
        else:
            loss = (
                w_anchor * anchor_loss
                + 0.5 * torch.stack(cc_losses).mean()
                + 0.3 * torch.stack(soc_role_losses).mean()
                + 1.0 * torch.stack(soc_comb_losses).mean()
            )
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append({
                'step': step,
                'loss': float(loss.detach().cpu()),
                'anchor': float(anchor_loss.detach().cpu()),
            })

    sync_device(device)
    model.eval()
    return model, time.perf_counter() - ts, trace


def make_lora_predictor(model, b7_cache, device, single_head=False, alpha=0.5):
    model.eval()
    def predict_pair(d):
        x = torch.stack([b7_cache[d['s']], b7_cache[d['t']]], dim=0).to(device)
        sem_b, role_b, _ = model(x)
        s_xy = d['s_xy'].to(device)
        t_xy = d['t_xy'].to(device)

        zs_sem = sample_dense_map(sem_b[0], s_xy)
        zt_sem = sample_dense_map(sem_b[1], t_xy)
        ft_sem = sem_b[1].flatten(1).T

        sd_cc = zs_sem @ ft_sem.T
        sc_cc = zs_sem @ zt_sem.T

        if single_head:
            return sd_cc, sd_cc, sc_cc, sc_cc
        else:
            zs_role = sample_dense_map(role_b[0], s_xy)
            zt_role = sample_dense_map(role_b[1], t_xy)
            ft_role = role_b[1].flatten(1).T
            sd_soc = sd_cc + alpha * (zs_role @ ft_role.T)
            sc_soc = sc_cc + alpha * (zs_role @ zt_role.T)
            return sd_cc, sd_soc, sc_cc, sc_soc
    return predict_pair


def main():
    device = get_device()
    print(f"Running DINOv2 Blocks 8-11 LoRA experiment on device={device}", flush=True)
    b7_cache = build_or_load_block7_cache(device)
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()
    train_ids = [i for i, d in enumerate(dataset) if d['p']['split'] == 'train']

    full_vit = load_dinov2_tail(torch.device('cpu'))
    base_blocks_8_11 = nn.ModuleList([copy.deepcopy(full_vit.blocks[i]) for i in range(8, 12)])
    base_norm = copy.deepcopy(full_vit.norm)
    del full_vit

    # ------------------------------------------------------------------
    # Step 1: Quick Dev-only LR & steps selection (seed 17, 36 dev imgs)
    # ------------------------------------------------------------------
    print("\n--- Dev-only LR/steps check for L3_LoRA_Dual_Anchored (seed=17) ---", flush=True)
    dev_trials = []
    for lr, steps in [(1e-4, 300), (3e-4, 300), (5e-4, 300), (3e-4, 500)]:
        m, dt, tr = train_lora_variant(
            base_blocks_8_11, base_norm, False, 10.0,
            dataset, b7_cache, dense_norm, grid_xy, train_ids, 17, device,
            lr=lr, steps=steps,
        )
        s, _ = evaluate_pair_predictor(
            make_lora_predictor(m, b7_cache, device, single_head=False, alpha=0.5),
            dataset, 'dev', 'L3_dev', 17, params=m.trainable_param_count(), device=device,
        )
        cm = s['category_macro']
        print(
            f"[DEV L3 lr={lr} steps={steps}] params={m.trainable_param_count()} train={dt:.1f}s | "
            f"SOC={cm['soc_pck']:.2f} CC_sq={cm['cc_samequery_pck']:.2f} CC_role={cm['cc_rolebranch_pck']:.2f} WR={cm['dense_wrong_role']:.2f}",
            flush=True,
        )
        dev_trials.append({'lr': lr, 'steps': steps, 'soc': cm['soc_pck'], 'cc_sq': cm['cc_samequery_pck'], 'wr': cm['dense_wrong_role']})

    best_dev = max(dev_trials, key=lambda x: x['soc'])
    best_lr, best_steps = best_dev['lr'], best_dev['steps']
    print(f"Selected from DEV: lr={best_lr}, steps={best_steps} (dev SOC={best_dev['soc']:.2f}, CC_sq={best_dev['cc_sq']:.2f})\n", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Frozen-config 3-seed Confirmation Suite (48 imgs, 146 pairs)
    # ------------------------------------------------------------------
    specs = [
        ('L1_LoRA_Single',        True,  0.0),
        ('L2_LoRA_Dual_NoAnchor', False, 0.0),
        ('L3_LoRA_Dual_Anchored', False, 10.0),
    ]

    all_models = {}
    all_rows = {}

    for name, single_head, w_anchor in specs:
        runs = []
        rows_all = []
        for seed in SEEDS:
            model, tr_sec, trace = train_lora_variant(
                base_blocks_8_11, base_norm, single_head, w_anchor,
                dataset, b7_cache, dense_norm, grid_xy, train_ids, seed, device,
                lr=best_lr, steps=best_steps,
            )
            # Save only trainable state dict to keep disk tiny
            trainable_sd = {k: v.cpu() for k, v in model.state_dict().items() if k in {n for n, p in model.named_parameters() if p.requires_grad}}
            torch.save(trainable_sd, WEIGHTS_DIR / f'{name}_seed{seed}.pt')
            summ, rows = evaluate_pair_predictor(
                make_lora_predictor(model, b7_cache, device, single_head=single_head, alpha=0.5),
                dataset, 'confirm', name, seed,
                out_dir=OUTPUTS_DIR, params=model.trainable_param_count(), device=device,
            )
            summ['train_seconds'] = tr_sec
            summ['loss_trace'] = trace
            runs.append(summ)
            rows_all.extend(rows)
            cm = summ['category_macro']
            print(
                f"[{name:22s} seed={seed}] params={summ['params']} train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | "
                f"SOC={cm['soc_pck']:.2f} CC_sq={cm['cc_samequery_pck']:.2f} CC_role={cm['cc_rolebranch_pck']:.2f} WR={cm['dense_wrong_role']:.2f}",
                flush=True,
            )
        all_models[name] = aggregate_model_runs(runs)
        all_rows[name] = rows_all

    # Load G1_NoAttn and Frozen for direct paired bootstrap
    decomp = json.loads((OUT / 'g1_decomp_results.json').read_text())
    all_models['G1_NoAttn'] = decomp['models']['G1_NoAttn']
    all_rows['G1_NoAttn'] = load_csv_rows('G1_NoAttn', SEEDS)
    all_models['frozen'] = decomp['models']['frozen']
    all_rows['frozen'] = load_csv_rows('frozen', [0])

    pv_map = {k: to_pairvals_full(v) for k, v in all_rows.items()}

    comparisons = {
        'L3_minus_G1_NoAttn': bootstrap_linear_combo(pv_map, {'L3_LoRA_Dual_Anchored': 1.0, 'G1_NoAttn': -1.0}),
        'L3_minus_L1':        bootstrap_linear_combo(pv_map, {'L3_LoRA_Dual_Anchored': 1.0, 'L1_LoRA_Single': -1.0}),
        'L3_minus_L2':        bootstrap_linear_combo(pv_map, {'L3_LoRA_Dual_Anchored': 1.0, 'L2_LoRA_Dual_NoAnchor': -1.0}),
        'L3_minus_frozen':    bootstrap_linear_combo(pv_map, {'L3_LoRA_Dual_Anchored': 1.0, 'frozen': -1.0}),
        'L1_minus_frozen':    bootstrap_linear_combo(pv_map, {'L1_LoRA_Single': 1.0, 'frozen': -1.0}),
        'L2_minus_frozen':    bootstrap_linear_combo(pv_map, {'L2_LoRA_Dual_NoAnchor': 1.0, 'frozen': -1.0}),
    }

    out_obj = {
        'protocol': {
            'description': 'DINOv2-S/14 Blocks 8-11 LoRA (rank=8 Q/K/V/Proj) + Semantic Anchor + Dual-Granularity Residual Heads',
            'dev_trials': dev_trials,
            'selected_lr': best_lr,
            'selected_steps': best_steps,
            'seeds': SEEDS,
            'bootstrap_method': '4000 replicates, within-category shared image vertex bootstrap, seed 20261121',
        },
        'models': {k: all_models[k] for k in ['frozen', 'G1_NoAttn', 'L1_LoRA_Single', 'L2_LoRA_Dual_NoAnchor', 'L3_LoRA_Dual_Anchored']},
        'comparisons': comparisons,
    }
    (OUT / 'lora_results.json').write_text(json.dumps(out_obj, indent=2))

    print("\n=== DINOV2 BLOCKS 8-11 LORA CONFIRMATION SUMMARY ===", flush=True)
    for k in ['frozen', 'G1_NoAttn', 'L1_LoRA_Single', 'L2_LoRA_Dual_NoAnchor', 'L3_LoRA_Dual_Anchored']:
        info = all_models[k]
        m = info['mean']
        cats_str = " ".join(f"{c[:4]}={info['by_category'][c]['soc_pck']:.1f}" for c in ['bench', 'dishwasher', 'folding_chair', 'train'])
        print(
            f"{k:22s} | params={info['params']:7d} | SOC={m['soc_pck']:.2f} | CC_sq={m['cc_samequery_pck']:.2f} | "
            f"CC_role={m['cc_rolebranch_pck']:.2f} | WR={m['dense_wrong_role']:.2f} | {cats_str}",
            flush=True,
        )
    print("\n=== KEY BOOTSTRAP DELTAS (pp, 95% CI) ===", flush=True)
    for k in ['L3_minus_G1_NoAttn', 'L3_minus_L1', 'L3_minus_L2', 'L3_minus_frozen']:
        c = comparisons[k]
        d = c['delta_pp']
        ci = c['ci95_pp']
        print(
            f"{k:20s} | SOC {d['soc_pck']:+.2f} [{ci['soc_pck'][0]:+.2f}, {ci['soc_pck'][1]:+.2f}] | "
            f"CC_sq {d['cc_samequery_pck']:+.2f} [{ci['cc_samequery_pck'][0]:+.2f}, {ci['cc_samequery_pck'][1]:+.2f}] | "
            f"CC_role {d['cc_rolebranch_pck']:+.2f} [{ci['cc_rolebranch_pck'][0]:+.2f}, {ci['cc_rolebranch_pck'][1]:+.2f}] | "
            f"WR {d['dense_wrong_role']:+.2f} [{ci['dense_wrong_role'][0]:+.2f}, {ci['dense_wrong_role'][1]:+.2f}] | "
            f"cats={c['category_soc_delta_pp']}",
            flush=True,
        )
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
