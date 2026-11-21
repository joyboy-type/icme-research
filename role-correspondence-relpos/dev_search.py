import json
import pathlib
import torch
from data_utils import get_device, load_data, evaluate_pair_predictor
from train_eval import (
    make_mlp_predictor,
    make_relpos_orth_predictor,
    make_relpos_single_predictor,
    train_mlp_single,
    train_relpos_orth,
    train_relpos_single,
)

OUT = pathlib.Path('/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos')


def fmt(m):
    return (
        f"SOC={m['soc_pck']:.2f} "
        f"CC_sq={m['cc_samequery_pck']:.2f} "
        f"CC_all={m['cc_pck']:.2f} "
        f"WR={m['dense_wrong_role']:.2f} "
        f"oraSOC={m['oracle_candidate_soc']:.2f}"
    )


def main():
    device = get_device()
    print('Device:', device, flush=True)
    cache, dense_mlp, dense_norm, grid_xy, dataset = load_data()
    train_ids = [i for i, d in enumerate(dataset) if d['p']['split'] == 'train']
    seed = 17

    results = []

    # 1. Task-conditioned MLP baseline on dev
    m_mlp, t_mlp, _ = train_mlp_single(dataset, train_ids, seed, device=torch.device('cpu'))
    s_mlp, _ = evaluate_pair_predictor(
        make_mlp_predictor(m_mlp, dense_mlp, torch.device('cpu')),
        dataset, 'dev', 'mlp_dev', seed, params=sum(p.numel() for p in m_mlp.parameters()),
    )
    print(f"[MLP] train={t_mlp:.1f}s | {fmt(s_mlp['category_macro'])}", flush=True)
    results.append(('mlp', s_mlp['category_macro']))

    # 2. RelPosSingleAdapter sweep on dev
    print("\n--- RelPosSingleAdapter Dev Sweep ---", flush=True)
    for cfg in [
        {'dim': 128, 'steps': 300, 'lr': 3e-4, 'use_dense_negs': False},
        {'dim': 128, 'steps': 300, 'lr': 3e-4, 'use_dense_negs': True},
        {'dim': 128, 'steps': 500, 'lr': 3e-4, 'use_dense_negs': True},
        {'dim': 96,  'steps': 300, 'lr': 5e-4, 'use_dense_negs': True},
        {'dim': 128, 'steps': 300, 'lr': 5e-4, 'use_dense_negs': True},
    ]:
        m, dt, tr = train_relpos_single(
            dataset, dense_norm, grid_xy, train_ids, seed, device=device, **cfg
        )
        s, _ = evaluate_pair_predictor(
            make_relpos_single_predictor(m, dense_norm, device),
            dataset, 'dev', 'relpos_single_dev', seed,
            params=sum(p.numel() for p in m.parameters()), device=device,
        )
        print(f"[Single {cfg}] train={dt:.1f}s ms/p={s['ms_per_pair']:.2f} | {fmt(s['category_macro'])}", flush=True)
        results.append((f"single_{cfg}", s['category_macro']))

    # 3. RelPosOrthSubspaceAdapter sweep on dev
    print("\n--- RelPosOrthSubspaceAdapter Dev Sweep ---", flush=True)
    orth_cfgs = [
        {'dim': 128, 'steps': 300, 'lr': 3e-4, 'use_dense_negs': True,  'w_anchor': 5.0,  'w_orth': 0.2, 'alpha_train': 0.6, 'w_soc_role': 0.5, 'w_soc_comb': 1.0},
        {'dim': 128, 'steps': 300, 'lr': 3e-4, 'use_dense_negs': False, 'w_anchor': 5.0,  'w_orth': 0.2, 'alpha_train': 0.6, 'w_soc_role': 0.5, 'w_soc_comb': 1.0},
        {'dim': 128, 'steps': 400, 'lr': 3e-4, 'use_dense_negs': True,  'w_anchor': 10.0, 'w_orth': 0.5, 'alpha_train': 0.8, 'w_soc_role': 0.5, 'w_soc_comb': 1.0},
        {'dim': 128, 'steps': 300, 'lr': 5e-4, 'use_dense_negs': True,  'w_anchor': 10.0, 'w_orth': 0.2, 'alpha_train': 0.6, 'w_soc_role': 0.3, 'w_soc_comb': 1.0},
        {'dim': 128, 'steps': 300, 'lr': 3e-4, 'use_dense_negs': True,  'w_anchor': 10.0, 'w_orth': 0.2, 'alpha_train': 0.5, 'w_soc_role': 0.0, 'w_soc_comb': 1.0, 'freeze_sem': True},
    ]
    for cfg in orth_cfgs:
        m, dt, tr = train_relpos_orth(
            dataset, dense_norm, grid_xy, train_ids, seed, device=device, **cfg
        )
        print(f"\n[Orth {cfg}] train={dt:.1f}s final_trace={tr[-1]}", flush=True)
        for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 'role_only']:
            ro = (alpha == 'role_only')
            a_val = 1.0 if ro else alpha
            s, _ = evaluate_pair_predictor(
                make_relpos_orth_predictor(m, dense_norm, device, alpha=a_val, role_only=ro),
                dataset, 'dev', 'relpos_orth_dev', seed,
                params=sum(p.numel() for p in m.parameters()), device=device,
            )
            print(f"   alpha={str(alpha):>9s} | {fmt(s['category_macro'])}", flush=True)
            results.append((f"orth_{cfg}_a{alpha}", s['category_macro']))

    (OUT / 'dev_search_results.json').write_text(json.dumps(results, indent=2))
    print("\nDEV SEARCH DONE.", flush=True)


if __name__ == '__main__':
    main()
