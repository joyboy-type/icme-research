import collections
import csv
import json
import pathlib
import numpy as np
import torch

from data_utils import (
    SEEDS,
    evaluate_pair_predictor,
    get_device,
    load_data,
)
from train_eval import (
    make_relpos_orth_predictor,
    train_relpos_orth,
)

OUT = pathlib.Path('/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos')
OUTPUTS_DIR = OUT / 'outputs'
WEIGHTS_DIR = OUTPUTS_DIR / 'weights'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_CFG = {
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
    'tau': 0.07,
}
ALPHA_EVAL = 0.5


def load_csv_rows(model_name, seeds):
    rows = []
    for sd in seeds:
        path = OUTPUTS_DIR / f'per_query_confirm_{model_name}_seed{sd}.csv'
        with path.open() as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def to_pairvals_full(rows):
    """Per-pair seed-averaged [soc_pck, cc_samequery_pck, cc_rolebranch_pck, dense_wrong_role] and cc_pck."""
    g_soc = collections.defaultdict(list)
    g_all = collections.defaultdict(list)
    for r in rows:
        has_soc = r['has_soc'] is True or r['has_soc'] == 'True'
        cc_v = r['cc_pck'] is True or r['cc_pck'] == 'True'
        g_all[(r['category'], r['pair'], str(r['seed']))].append((r['source_image'], r['target_image'], float(cc_v)))
        if has_soc:
            soc_v = r['soc_pck'] is True or r['soc_pck'] == 'True'
            cc_role_v = r['cc_rolebranch_pck'] is True or r['cc_rolebranch_pck'] == 'True'
            wr_v = r['dense_wrong_role'] is True or r['dense_wrong_role'] == 'True'
            g_soc[(r['category'], r['pair'], str(r['seed']))].append(
                (r['source_image'], r['target_image'], float(soc_v), float(cc_v), float(cc_role_v), float(wr_v))
            )
    per_soc = collections.defaultdict(list)
    for (c, p, _sd), rs in g_soc.items():
        key = (c, p, rs[0][0], rs[0][1])
        per_soc[key].append([
            np.mean([x[2] for x in rs]),
            np.mean([x[3] for x in rs]),
            np.mean([x[4] for x in rs]),
            np.mean([x[5] for x in rs]),
        ])
    per_all = collections.defaultdict(list)
    for (c, p, _sd), rs in g_all.items():
        key = (c, p, rs[0][0], rs[0][1])
        per_all[key].append(np.mean([x[2] for x in rs]))
    pv_soc = {k: np.mean(v, axis=0) for k, v in per_soc.items()}
    pv_all = {k: float(np.mean(v)) for k, v in per_all.items()}
    return pv_soc, pv_all


def bootstrap_linear_combo(pv_map, weights_by_name, n_boot=4000, seed=20261121):
    """
    Compute exact point estimate and 95% shared-image vertex bootstrap CI for any linear
    combination of models, e.g. {'G4': +1, 'G3': -1} or interaction {'G4': +1, 'G3': -1, 'G2': -1, 'G1': +1}.
    """
    names = list(weights_by_name.keys())
    keys = sorted(pv_map[names[0]][0])
    keys_all = sorted(pv_map[names[0]][1])
    for nm in names[1:]:
        assert set(keys) == set(pv_map[nm][0])
        assert set(keys_all) == set(pv_map[nm][1])
    cats = sorted({k[0] for k in keys})

    def combo_soc(k):
        return sum(weights_by_name[nm] * pv_map[nm][0][k] for nm in names)

    def combo_all(k):
        return sum(weights_by_name[nm] * pv_map[nm][1][k] for nm in names)

    rng = np.random.default_rng(seed)
    boot_soc = []
    boot_all = []
    for _ in range(n_boot):
        ds = []
        da = []
        for c in cats:
            kk = [k for k in keys if k[0] == c]
            ims = sorted({n for k in kk for n in k[2:]})
            mult = np.bincount(rng.choice(len(ims), len(ims)), minlength=len(ims))
            wmap = dict(zip(ims, mult))
            w = np.array([wmap[k[2]] * wmap[k[3]] for k in kk])
            d = np.stack([combo_soc(k) for k in kk])
            ds.append((w[:, None] * d).sum(0) / w.sum() if w.sum() else d.mean(0))

            kk_a = [k for k in keys_all if k[0] == c]
            w_a = np.array([wmap.get(k[2], 0) * wmap.get(k[3], 0) for k in kk_a])
            d_a = np.array([combo_all(k) for k in kk_a])
            da.append((w_a * d_a).sum() / w_a.sum() if w_a.sum() else d_a.mean())
        boot_soc.append(np.mean(ds, axis=0) * 100.0)
        boot_all.append(np.mean(da) * 100.0)

    ci_soc = np.percentile(boot_soc, [2.5, 97.5], axis=0).T.tolist()
    ci_all = np.percentile(boot_all, [2.5, 97.5]).tolist()

    cat_soc_delta = {}
    deltas_soc = []
    deltas_all = []
    for c in cats:
        kk = [k for k in keys if k[0] == c]
        d = np.mean([combo_soc(k) for k in kk], axis=0) * 100.0
        deltas_soc.append(d)
        cat_soc_delta[c] = float(d[0])
        kk_a = [k for k in keys_all if k[0] == c]
        deltas_all.append(float(np.mean([combo_all(k) for k in kk_a]) * 100.0))

    mean_soc = np.mean(deltas_soc, axis=0)
    return {
        'delta_pp': {
            'soc_pck': float(mean_soc[0]),
            'cc_samequery_pck': float(mean_soc[1]),
            'cc_rolebranch_pck': float(mean_soc[2]),
            'dense_wrong_role': float(mean_soc[3]),
            'cc_pck': float(np.mean(deltas_all)),
        },
        'ci95_pp': {
            'soc_pck': ci_soc[0],
            'cc_samequery_pck': ci_soc[1],
            'cc_rolebranch_pck': ci_soc[2],
            'dense_wrong_role': ci_soc[3],
            'cc_pck': ci_all,
        },
        'category_soc_delta_pp': cat_soc_delta,
    }


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
        'cc_rolebranch_by_seed': {str(r['seed']): r['category_macro']['cc_rolebranch_pck'] for r in runs},
        'by_category': {
            c: {m: float(np.mean([r['by_category'][c][m] for r in runs])) for m in metrics}
            for c in cats
        },
        'train_seconds_per_seed': [float(r.get('train_seconds', 0.0)) for r in runs],
        'train_seconds_mean': float(np.mean([r.get('train_seconds', 0.0) for r in runs])),
        'ms_per_pair_mean': float(np.mean([r['ms_per_pair'] for r in runs])),
    }


def main():
    device = get_device()
    print(f"Running 2x2 strictly aligned ablation on device={device}", flush=True)
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()
    train_ids = [i for i, d in enumerate(dataset) if d['p']['split'] == 'train']

    grid_specs = [
        ('G1_plain_attn_plain_dual', False, False, 0.0),
        ('G2_plain_attn_orth',       False, True,  0.2),
        ('G3_relpos_plain_dual',     True,  False, 0.0),
    ]

    all_models = {}
    all_rows = {}

    for name, use_relpos, use_orth, w_orth in grid_specs:
        runs = []
        rows_all = []
        for seed in SEEDS:
            model, tr_sec, trace = train_relpos_orth(
                dataset,
                dense_norm,
                grid_xy,
                train_ids,
                seed,
                device=device,
                use_relpos=use_relpos,
                use_orth=use_orth,
                w_orth=w_orth,
                **BASE_CFG,
            )
            torch.save(model.state_dict(), WEIGHTS_DIR / f'{name}_seed{seed}.pt')
            summ, rows = evaluate_pair_predictor(
                make_relpos_orth_predictor(model, dense_norm, device, alpha=ALPHA_EVAL),
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
                f"[{name} seed={seed}] params={summ['params']} train={tr_sec:.1f}s ms/p={summ['ms_per_pair']:.2f} | "
                f"SOC={m['soc_pck']:.2f} CC_sq={m['cc_samequery_pck']:.2f} CC_role={m['cc_rolebranch_pck']:.2f} WR={m['dense_wrong_role']:.2f}",
                flush=True,
            )
        all_models[name] = aggregate_model_runs(runs)
        all_rows[name] = rows_all

    # Load G4 (existing 4b) and Frozen directly from saved outputs
    prev_summary = json.loads((OUT / 'summary.json').read_text())
    all_models['G4_relpos_orth'] = prev_summary['models']['relpos_orth_subspace']
    all_rows['G4_relpos_orth'] = load_csv_rows('relpos_orth_subspace', SEEDS)
    all_models['frozen'] = prev_summary['models']['frozen']
    all_rows['frozen'] = load_csv_rows('frozen', [0])

    pv_map = {k: to_pairvals_full(v) for k, v in all_rows.items()}

    comparisons = {
        'G4_minus_G3_orth_given_relpos': bootstrap_linear_combo(pv_map, {'G4_relpos_orth': 1.0, 'G3_relpos_plain_dual': -1.0}),
        'G4_minus_G2_relpos_given_orth': bootstrap_linear_combo(pv_map, {'G4_relpos_orth': 1.0, 'G2_plain_attn_orth': -1.0}),
        'G4_minus_G1_total_gain':        bootstrap_linear_combo(pv_map, {'G4_relpos_orth': 1.0, 'G1_plain_attn_plain_dual': -1.0}),
        'G3_minus_G1_relpos_given_plain': bootstrap_linear_combo(pv_map, {'G3_relpos_plain_dual': 1.0, 'G1_plain_attn_plain_dual': -1.0}),
        'G2_minus_G1_orth_given_plain':  bootstrap_linear_combo(pv_map, {'G2_plain_attn_orth': 1.0, 'G1_plain_attn_plain_dual': -1.0}),
        'interaction_G4_G3_minus_G2_G1': bootstrap_linear_combo(pv_map, {'G4_relpos_orth': 1.0, 'G3_relpos_plain_dual': -1.0, 'G2_plain_attn_orth': -1.0, 'G1_plain_attn_plain_dual': 1.0}),
        'G1_minus_frozen':               bootstrap_linear_combo(pv_map, {'G1_plain_attn_plain_dual': 1.0, 'frozen': -1.0}),
        'G2_minus_frozen':               bootstrap_linear_combo(pv_map, {'G2_plain_attn_orth': 1.0, 'frozen': -1.0}),
        'G3_minus_frozen':               bootstrap_linear_combo(pv_map, {'G3_relpos_plain_dual': 1.0, 'frozen': -1.0}),
        'G4_minus_frozen':               bootstrap_linear_combo(pv_map, {'G4_relpos_orth': 1.0, 'frozen': -1.0}),
    }

    out_obj = {
        'protocol': {
            'description': '2x2 strictly aligned factorial ablation (2D-RelPos ON/OFF x Orthogonal Subspace ON/OFF)',
            'base_cfg': BASE_CFG,
            'alpha_eval': ALPHA_EVAL,
            'seeds': SEEDS,
            'bootstrap_method': '4000 replicates, within-category shared image vertex bootstrap, seed 20261121',
        },
        'models': {k: all_models[k] for k in ['G1_plain_attn_plain_dual', 'G2_plain_attn_orth', 'G3_relpos_plain_dual', 'G4_relpos_orth', 'frozen']},
        'comparisons': comparisons,
    }
    (OUT / 'ablation_2x2.json').write_text(json.dumps(out_obj, indent=2))

    print("\n=== 2x2 STRICTLY ALIGNED ABLATION SUMMARY ===", flush=True)
    for k in ['frozen', 'G1_plain_attn_plain_dual', 'G2_plain_attn_orth', 'G3_relpos_plain_dual', 'G4_relpos_orth']:
        info = all_models[k]
        m = info['mean']
        print(
            f"{k:26s} | params={info['params']:7d} | SOC={m['soc_pck']:.2f} | CC_sq={m['cc_samequery_pck']:.2f} | "
            f"CC_role={m['cc_rolebranch_pck']:.2f} | WR={m['dense_wrong_role']:.2f} | oraSOC={m['oracle_candidate_soc']:.2f}",
            flush=True,
        )
    print("\n=== KEY BOOTSTRAP DELTAS (pp, 95% CI) ===", flush=True)
    for k in [
        'G4_minus_G3_orth_given_relpos',
        'G4_minus_G2_relpos_given_orth',
        'G4_minus_G1_total_gain',
        'G3_minus_G1_relpos_given_plain',
        'G2_minus_G1_orth_given_plain',
        'interaction_G4_G3_minus_G2_G1',
        'G1_minus_frozen',
    ]:
        c = comparisons[k]
        d = c['delta_pp']
        ci = c['ci95_pp']
        print(
            f"{k:32s} | SOC {d['soc_pck']:+.2f} [{ci['soc_pck'][0]:+.2f}, {ci['soc_pck'][1]:+.2f}] | "
            f"CC_sq {d['cc_samequery_pck']:+.2f} [{ci['cc_samequery_pck'][0]:+.2f}, {ci['cc_samequery_pck'][1]:+.2f}] | "
            f"CC_role {d['cc_rolebranch_pck']:+.2f} [{ci['cc_rolebranch_pck'][0]:+.2f}, {ci['cc_rolebranch_pck'][1]:+.2f}] | "
            f"WR {d['dense_wrong_role']:+.2f} [{ci['dense_wrong_role'][0]:+.2f}, {ci['dense_wrong_role'][1]:+.2f}]",
            flush=True,
        )
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
