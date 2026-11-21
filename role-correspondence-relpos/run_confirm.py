import json
import pathlib
import numpy as np
import torch
from torch.nn import functional as F

from data_utils import (
    H,
    SEEDS,
    evaluate_pair_predictor,
    get_device,
    load_data,
    paired_image_bootstrap,
)
from train_eval import (
    make_mlp_predictor,
    make_relpos_orth_predictor,
    make_relpos_single_predictor,
    train_mlp_single,
    train_relpos_orth,
    train_relpos_single,
)

OUT = pathlib.Path('/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos')
OUTPUTS_DIR = OUT / 'outputs'
WEIGHTS_DIR = OUTPUTS_DIR / 'weights'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters strictly frozen from dev-split search (office_chair, bus, leopard).
# Confirm split (bench, dishwasher, folding_chair, train) was NEVER used for tuning.
CFG_RELPOS_SINGLE = {
    'dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'steps': 300,
    'batch_size': 8,
    'lr': 5e-4,
    'weight_decay': 1e-2,
    'use_dense_negs': True,
    'tau': 0.07,
}

CFG_RELPOS_ORTH_FROZEN_SEM = {
    'dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'steps': 300,
    'batch_size': 8,
    'lr': 3e-4,
    'weight_decay': 1e-2,
    'use_dense_negs': True,
    'freeze_sem': True,
    'alpha_train': 0.5,
    'w_cc': 0.0,
    'w_soc_role': 0.0,
    'w_soc_comb': 1.0,
    'w_anchor': 10.0,
    'w_orth': 0.2,
    'tau': 0.07,
}
ALPHA_EVAL_FROZEN_SEM = 0.5

CFG_RELPOS_ORTH = {
    'dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'steps': 300,
    'batch_size': 8,
    'lr': 5e-4,
    'weight_decay': 1e-2,
    'use_dense_negs': True,
    'freeze_sem': False,
    'alpha_train': 0.6,
    'w_cc': 0.5,
    'w_soc_role': 0.3,
    'w_soc_comb': 1.0,
    'w_anchor': 10.0,
    'w_orth': 0.2,
    'tau': 0.07,
}
ALPHA_EVAL_ORTH = 0.5


def aggregate_model_runs(runs):
    metrics = list(runs[0]['category_macro'].keys())
    cats = sorted(runs[0]['by_category'].keys())
    return {
        'params': runs[0]['params'],
        'seeds': [r['seed'] for r in runs],
        'mean': {m: float(np.mean([r['category_macro'][m] for r in runs])) for m in metrics},
        'std': {m: float(np.std([r['category_macro'][m] for r in runs])) for m in metrics},
        'soc_by_seed': {str(r['seed']): r['category_macro']['soc_pck'] for r in runs},
        'cc_samequery_by_seed': {str(r['seed']): r['category_macro']['cc_samequery_pck'] for r in runs},
        'by_category': {
            c: {m: float(np.mean([r['by_category'][c][m] for r in runs])) for m in metrics}
            for c in cats
        },
        'train_seconds_per_seed': [float(r.get('train_seconds', 0.0)) for r in runs],
        'train_seconds_mean': float(np.mean([r.get('train_seconds', 0.0) for r in runs])),
        'train_seconds_total': float(sum(r.get('train_seconds', 0.0) for r in runs)),
        'ms_per_pair_mean': float(np.mean([r['ms_per_pair'] for r in runs])),
        'eval_seconds_total': float(sum(r['seconds'] for r in runs)),
        'queries_per_seed': runs[0]['queries'],
        'soc_queries_per_seed': runs[0]['soc_queries'],
        'pairs': runs[0]['pairs'],
    }


def main():
    device = get_device()
    print(f"Running confirmation suite on device={device}", flush=True)
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()
    train_ids = [i for i, d in enumerate(dataset) if d['p']['split'] == 'train']

    all_runs = {}
    all_rows = {}

    # ------------------------------------------------------------------
    # (1) Frozen DINOv2-S
    # ------------------------------------------------------------------
    def frozen_predict(d):
        a = F.normalize(d['xs'][..., :H], dim=-1)
        b_dense = F.normalize(dense_mlp[d['t']][..., :H], dim=-1)
        b_cand = F.normalize(d['xt'][..., :H], dim=-1)
        sd = a @ b_dense.T
        sc = a @ b_cand.T
        return sd, sd, sc, sc

    s_froz, rows_froz = evaluate_pair_predictor(
        frozen_predict, dataset, 'confirm', 'frozen', 0,
        out_dir=OUTPUTS_DIR, params=0, device=torch.device('cpu'),
    )
    s_froz['train_seconds'] = 0.0
    all_runs['frozen'] = [s_froz]
    all_rows['frozen'] = rows_froz
    print("[1/5 frozen] confirm:", json.dumps(s_froz['category_macro']), f"ms/p={s_froz['ms_per_pair']:.2f}", flush=True)

    # Verify exact match with soco_grid_corrected audit
    assert abs(s_froz['category_macro']['soc_pck'] - 44.41169273353657) < 1e-6
    assert abs(s_froz['category_macro']['cc_samequery_pck'] - 65.79396201695955) < 1e-6

    # ------------------------------------------------------------------
    # (2) Task-Conditioned Single Head MLP (exact baseline replica)
    # ------------------------------------------------------------------
    all_runs['task_conditioned_single'] = []
    all_rows['task_conditioned_single'] = []
    for seed in SEEDS:
        model, tr_sec, trace = train_mlp_single(
            dataset, train_ids, seed, steps=600, batch_size=8, lr=1e-3, device=torch.device('cpu')
        )
        torch.save(model.state_dict(), WEIGHTS_DIR / f'task_conditioned_single_seed{seed}.pt')
        summ, rows = evaluate_pair_predictor(
            make_mlp_predictor(model, dense_mlp, torch.device('cpu')),
            dataset, 'confirm', 'task_conditioned_single', seed,
            out_dir=OUTPUTS_DIR, params=sum(p.numel() for p in model.parameters()),
            device=torch.device('cpu'),
        )
        summ['train_seconds'] = tr_sec
        summ['loss_trace'] = trace
        all_runs['task_conditioned_single'].append(summ)
        all_rows['task_conditioned_single'].extend(rows)
        print(f"[2/5 task_conditioned_single seed={seed}] train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | SOC={summ['category_macro']['soc_pck']:.2f} CC_sq={summ['category_macro']['cc_samequery_pck']:.2f} WR={summ['category_macro']['dense_wrong_role']:.2f}", flush=True)

    # ------------------------------------------------------------------
    # (3) 2-Layer 2D-RelPos Cross-Patch Adapter (Single Head / Mixed)
    # ------------------------------------------------------------------
    all_runs['relpos_single'] = []
    all_rows['relpos_single'] = []
    for seed in SEEDS:
        model, tr_sec, trace = train_relpos_single(
            dataset, dense_norm, grid_xy, train_ids, seed, device=device, **CFG_RELPOS_SINGLE
        )
        torch.save(model.state_dict(), WEIGHTS_DIR / f'relpos_single_seed{seed}.pt')
        summ, rows = evaluate_pair_predictor(
            make_relpos_single_predictor(model, dense_norm, device),
            dataset, 'confirm', 'relpos_single', seed,
            out_dir=OUTPUTS_DIR, params=sum(p.numel() for p in model.parameters()),
            device=device,
        )
        summ['train_seconds'] = tr_sec
        summ['loss_trace'] = trace
        all_runs['relpos_single'].append(summ)
        all_rows['relpos_single'].extend(rows)
        print(f"[3/5 relpos_single seed={seed}] train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | SOC={summ['category_macro']['soc_pck']:.2f} CC_sq={summ['category_macro']['cc_samequery_pck']:.2f} WR={summ['category_macro']['dense_wrong_role']:.2f}", flush=True)

    # ------------------------------------------------------------------
    # (4a) Ablation: 2-Layer 2D-RelPos + Orthogonal Role Subspace (Frozen z_sem)
    # ------------------------------------------------------------------
    all_runs['relpos_orth_frozen_sem'] = []
    all_rows['relpos_orth_frozen_sem'] = []
    for seed in SEEDS:
        model, tr_sec, trace = train_relpos_orth(
            dataset, dense_norm, grid_xy, train_ids, seed, device=device, **CFG_RELPOS_ORTH_FROZEN_SEM
        )
        torch.save(model.state_dict(), WEIGHTS_DIR / f'relpos_orth_frozen_sem_seed{seed}.pt')
        summ, rows = evaluate_pair_predictor(
            make_relpos_orth_predictor(model, dense_norm, device, alpha=ALPHA_EVAL_FROZEN_SEM),
            dataset, 'confirm', 'relpos_orth_frozen_sem', seed,
            out_dir=OUTPUTS_DIR, params=sum(p.numel() for p in model.parameters()),
            device=device,
        )
        summ['train_seconds'] = tr_sec
        summ['loss_trace'] = trace
        all_runs['relpos_orth_frozen_sem'].append(summ)
        all_rows['relpos_orth_frozen_sem'].extend(rows)
        print(f"[4/5 relpos_orth_frozen_sem seed={seed}] train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | SOC={summ['category_macro']['soc_pck']:.2f} CC_sq={summ['category_macro']['cc_samequery_pck']:.2f} WR={summ['category_macro']['dense_wrong_role']:.2f}", flush=True)

    # ------------------------------------------------------------------
    # (4b) Full Model: 2-Layer 2D-RelPos + Orthogonal Semantic-Role Subspace
    # ------------------------------------------------------------------
    all_runs['relpos_orth_subspace'] = []
    all_rows['relpos_orth_subspace'] = []
    for seed in SEEDS:
        model, tr_sec, trace = train_relpos_orth(
            dataset, dense_norm, grid_xy, train_ids, seed, device=device, **CFG_RELPOS_ORTH
        )
        torch.save(model.state_dict(), WEIGHTS_DIR / f'relpos_orth_subspace_seed{seed}.pt')
        summ, rows = evaluate_pair_predictor(
            make_relpos_orth_predictor(model, dense_norm, device, alpha=ALPHA_EVAL_ORTH),
            dataset, 'confirm', 'relpos_orth_subspace', seed,
            out_dir=OUTPUTS_DIR, params=sum(p.numel() for p in model.parameters()),
            device=device,
        )
        summ['train_seconds'] = tr_sec
        summ['loss_trace'] = trace
        all_runs['relpos_orth_subspace'].append(summ)
        all_rows['relpos_orth_subspace'].extend(rows)
        print(f"[5/5 relpos_orth_subspace seed={seed}] train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | SOC={summ['category_macro']['soc_pck']:.2f} CC_sq={summ['category_macro']['cc_samequery_pck']:.2f} WR={summ['category_macro']['dense_wrong_role']:.2f}", flush=True)

    # ------------------------------------------------------------------
    # Aggregate & Shared-Image Vertex Bootstrap Comparisons
    # ------------------------------------------------------------------
    models_summary = {name: aggregate_model_runs(runs) for name, runs in all_runs.items()}

    comparisons = {}
    pairs_to_compare = [
        ('relpos_orth_subspace', 'frozen'),
        ('relpos_orth_subspace', 'task_conditioned_single'),
        ('relpos_orth_subspace', 'relpos_single'),
        ('relpos_orth_subspace', 'relpos_orth_frozen_sem'),
        ('relpos_orth_frozen_sem', 'frozen'),
        ('relpos_single', 'task_conditioned_single'),
        ('relpos_single', 'frozen'),
    ]
    for cand, ref in pairs_to_compare:
        key = f"{cand}_vs_{ref}"
        comparisons[key] = paired_image_bootstrap(all_rows[cand], all_rows[ref], n_boot=4000, seed=20261121)

    # Check original pilot gate criteria for relpos_orth_subspace vs frozen AND vs task_conditioned_single
    c_vs_f = comparisons['relpos_orth_subspace_vs_frozen']
    c_vs_m = comparisons['relpos_orth_subspace_vs_task_conditioned_single']
    gate_vs_frozen = (
        c_vs_f['delta_pp']['soc_pck'] >= 3.0
        and c_vs_f['delta_pp']['cc_samequery_pck'] >= -1.0
        and sum(v > 0 for v in c_vs_f['category_soc_delta_pp'].values()) >= 3
        and c_vs_f['ci95_pp']['soc_pck'][0] > 0.0
    )
    gate_vs_mlp = (
        c_vs_m['delta_pp']['soc_pck'] >= 3.0
        and c_vs_m['delta_pp']['cc_samequery_pck'] >= -1.0
        and sum(v > 0 for v in c_vs_m['category_soc_delta_pp'].values()) >= 3
        and c_vs_m['ci95_pp']['soc_pck'][0] > 0.0
    )

    final_summary = {
        'protocol': {
            'coordinate_grid': 'v1.1-coordinate-correction (physical patch center (i+0.5)/20, align_corners=False)',
            'splits': {'train': '108 images / 485 pairs (6 cats)', 'dev': '36 images / 109 pairs (3 cats)', 'confirm': '48 images / 146 pairs (4 cats)'},
            'seeds': SEEDS,
            'device': str(device),
            'bootstrap_method': '4000 replicates, within-category shared image vertex bootstrap; weight each directed edge by source*target image multiplicities; seed-average edges then category macro',
            'dev_selected_configs': {
                'relpos_single': CFG_RELPOS_SINGLE,
                'relpos_orth_frozen_sem': {**CFG_RELPOS_ORTH_FROZEN_SEM, 'alpha_eval': ALPHA_EVAL_FROZEN_SEM},
                'relpos_orth_subspace': {**CFG_RELPOS_ORTH, 'alpha_eval': ALPHA_EVAL_ORTH},
            },
        },
        'models': models_summary,
        'paired_bootstrap_comparisons': comparisons,
        'gates': {
            'relpos_orth_vs_frozen_pass': bool(gate_vs_frozen),
            'relpos_orth_vs_mlp_pass': bool(gate_vs_mlp),
        },
    }

    (OUT / 'summary.json').write_text(json.dumps(final_summary, indent=2))
    print("\n=== CONFIRMATION SUMMARY ===", flush=True)
    for name, info in models_summary.items():
        m = info['mean']
        print(
            f"{name:26s} | params={info['params']:7d} | train={info['train_seconds_mean']:5.1f}s | ms/p={info['ms_per_pair_mean']:5.2f} | "
            f"SOC={m['soc_pck']:.2f} | CC_sq={m['cc_samequery_pck']:.2f} | CC_all={m['cc_pck']:.2f} | WR={m['dense_wrong_role']:.2f}",
            flush=True,
        )
    print("\n=== KEY BOOTSTRAP DELTAS ===", flush=True)
    for k in ['relpos_orth_subspace_vs_frozen', 'relpos_orth_subspace_vs_task_conditioned_single', 'relpos_orth_subspace_vs_relpos_single', 'relpos_orth_frozen_sem_vs_frozen']:
        c = comparisons[k]
        print(
            f"{k}: SOC {c['delta_pp']['soc_pck']:+.2f}pp (95%CI [{c['ci95_pp']['soc_pck'][0]:+.2f}, {c['ci95_pp']['soc_pck'][1]:+.2f}]), "
            f"CC_sq {c['delta_pp']['cc_samequery_pck']:+.2f}pp (95%CI [{c['ci95_pp']['cc_samequery_pck'][0]:+.2f}, {c['ci95_pp']['cc_samequery_pck'][1]:+.2f}]), "
            f"WR {c['delta_pp']['dense_wrong_role']:+.2f}pp, cats={c['category_soc_delta_pp']}",
            flush=True,
        )
    print("GATES:", final_summary['gates'], flush=True)
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
