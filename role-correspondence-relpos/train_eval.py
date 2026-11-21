import time
import numpy as np
import torch
from torch.nn import functional as F

from data_utils import (
    D_MLP,
    H,
    evaluate_pair_predictor,
    sample_dense_map,
    sync_device,
)
from models import (
    RelPosOrthSubspaceAdapter,
    RelPosSingleAdapter,
    TaskConditionedMLP,
)


def contrastive_pair_loss(sim, pos_mask, valid_mask, tau=0.07):
    """
    sim: (Ks, M) similarity matrix (unscaled by tau)
    pos_mask: (Ks, M) bool
    valid_mask: (Ks, M) bool (denominator mask, must include pos_mask)
    Returns scalar mean loss over active source queries (those with >=1 positive), or None.
    """
    active = pos_mask.any(dim=-1)
    if not active.any():
        return None
    logits = (sim[active] / tau).masked_fill(~valid_mask[active], -1e4)
    num = logits.masked_fill(~pos_mask[active], -1e4).logsumexp(dim=-1)
    den = logits.logsumexp(dim=-1)
    return (den - num).mean()


def compute_pair_dense_neg_mask(t_xy, pos_mat, grid_xy, scale, thresh=0.12):
    """
    For each source query i (Ks,), returns mask (Ks, 400) True where dense target patch p
    is at least `thresh * scale` away from ALL positive target keypoints of query i.
    """
    # dists_kp_grid: (Kt, 400)
    dists = torch.cdist(t_xy, grid_xy)  # (Kt, 400)
    far_from_kp = dists >= (thresh * scale)  # (Kt, 400)
    # For each query i, patch p is valid negative iff it is far from all k where pos_mat[i, k] is True
    # Equivalently: NOT exists k with pos_mat[i, k] and ~far_from_kp[k, p]
    near_any_pos = (pos_mat.float() @ (~far_from_kp).float()) > 0.5  # (Ks, 400)
    return ~near_any_pos


def train_mlp_single(dataset, train_ids, seed, steps=600, batch_size=8, lr=1e-3, device=torch.device('cpu')):
    """Exact replica of soco_grid_corrected/run.py training for task_conditioned_single."""
    rng = np.random.default_rng(seed)
    schedule = [rng.choice(train_ids, batch_size, replace=True).tolist() for _ in range(steps)]
    torch.manual_seed(seed)
    model = TaskConditionedMLP().to(device)
    assert sum(p.numel() for p in model.parameters()) == 73984
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batch_data(ids):
        items = [dataset[i] for i in ids]
        ns = max(len(d['xs']) for d in items)
        nt = max(len(d['xt']) for d in items)
        xs = torch.zeros(len(ids), ns, D_MLP)
        xt = torch.zeros(len(ids), nt, D_MLP)
        cc = torch.zeros(len(ids), ns, nt, dtype=torch.bool)
        soc = torch.zeros_like(cc)
        valid = torch.zeros(len(ids), nt, dtype=torch.bool)
        for j, d in enumerate(items):
            a, b = len(d['xs']), len(d['xt'])
            xs[j, :a] = d['xs']
            xt[j, :b] = d['xt']
            cc[j, :a, :b] = d['cc']
            soc[j, :a, :b] = d['soc']
            valid[j, :b] = True
        return xs.to(device), xt.to(device), cc.to(device), soc.to(device), valid.to(device)

    ts = time.perf_counter()
    trace = []
    for step, ids in enumerate(schedule):
        opt.zero_grad()
        xs, xt, cc, soc, valid = batch_data(ids)
        losses = []
        for task in (0, 1):
            pos = cc if task == 0 else soc
            logits = model(xs, task) @ model(xt, task).transpose(-1, -2) / 0.07
            denom = valid[:, None].expand_as(logits).clone()
            logits = logits.masked_fill(~denom, -1e4)
            active = pos.any(-1)
            numerator = logits.masked_fill(~pos, -1e4).logsumexp(-1)
            losses.append((logits.logsumexp(-1) - numerator)[active].mean())
        loss = sum(losses) / 2.0
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append({'step': step, 'loss': float(loss.detach().cpu())})
    sync_device(device)
    return model, time.perf_counter() - ts, trace


def prepare_train_pairs_on_device(dataset, train_ids, grid_xy, device):
    """Precompute dense negative masks and move train pair tensors to device once."""
    grid_xy_cpu = grid_xy.cpu()
    pre = {}
    for i in train_ids:
        d = dataset[i]
        neg_cc = compute_pair_dense_neg_mask(d['t_xy'], d['cc'], grid_xy_cpu, d['scale'])
        neg_soc = compute_pair_dense_neg_mask(d['t_xy'], d['soc'], grid_xy_cpu, d['scale'])
        zeros_400 = torch.zeros(len(d['s_xy']), 400, dtype=torch.bool)
        ones_kp = torch.ones_like(d['cc'])
        pre[i] = {
            's': d['s'],
            't': d['t'],
            's_xy': d['s_xy'].to(device),
            't_xy': d['t_xy'].to(device),
            'cc': d['cc'].to(device),
            'soc': d['soc'].to(device),
            'ones_kp': ones_kp.to(device),
            'pos_cc_all': torch.cat([d['cc'], zeros_400], dim=-1).to(device),
            'val_cc_all': torch.cat([ones_kp, neg_cc], dim=-1).to(device),
            'pos_soc_all': torch.cat([d['soc'], zeros_400], dim=-1).to(device),
            'val_soc_all': torch.cat([ones_kp, neg_soc], dim=-1).to(device),
        }
    return pre


def train_relpos_single(
    dataset,
    dense_norm,
    grid_xy,
    train_ids,
    seed,
    dim=128,
    num_heads=4,
    dropout=0.1,
    steps=400,
    batch_size=8,
    lr=3e-4,
    weight_decay=1e-2,
    use_dense_negs=True,
    tau=0.07,
    device=torch.device('cpu'),
):
    rng = np.random.default_rng(seed)
    schedule = [rng.choice(train_ids, batch_size, replace=True).tolist() for _ in range(steps)]
    torch.manual_seed(seed)

    model = RelPosSingleAdapter(dim=dim, num_heads=num_heads, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    train_imgs = sorted({dataset[i]['s'] for i in train_ids} | {dataset[i]['t'] for i in train_ids})
    img_dev = {k: dense_norm[k].to(device) for k in train_imgs}
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
        x_batch = torch.stack([img_dev[nm] for nm in uniq_imgs], dim=0)  # (U, 384, 20, 20)
        f_batch = model(x_batch)  # (U, 384, 20, 20)

        cc_losses = []
        soc_losses = []
        for d in items:
            fs = f_batch[img_to_idx[d['s']]]
            ft = f_batch[img_to_idx[d['t']]]
            zs = sample_dense_map(fs, d['s_xy'])
            zt = sample_dense_map(ft, d['t_xy'])

            sim_kp = zs @ zt.T
            if use_dense_negs:
                ft_flat = ft.flatten(1).T
                sim_all = torch.cat([sim_kp, zs @ ft_flat.T], dim=-1)
                lc = contrastive_pair_loss(sim_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
                ls = contrastive_pair_loss(sim_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
            else:
                lc = contrastive_pair_loss(sim_kp, d['cc'], d['ones_kp'], tau=tau)
                ls = contrastive_pair_loss(sim_kp, d['soc'], d['ones_kp'], tau=tau)

            if lc is not None:
                cc_losses.append(lc)
            if ls is not None:
                soc_losses.append(ls)

        loss = 0.5 * torch.stack(cc_losses).mean() + 0.5 * torch.stack(soc_losses).mean()
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append({'step': step, 'loss': float(loss.detach().cpu())})

    sync_device(device)
    model.eval()
    return model, time.perf_counter() - ts, trace


def train_relpos_orth(
    dataset,
    dense_norm,
    grid_xy,
    train_ids,
    seed,
    dim=128,
    num_heads=4,
    dropout=0.1,
    steps=400,
    batch_size=8,
    lr=3e-4,
    weight_decay=1e-2,
    use_dense_negs=True,
    freeze_sem=False,
    use_relpos=True,
    use_orth=True,
    alpha_train=0.6,
    w_cc=0.5,
    w_soc_role=0.5,
    w_soc_comb=1.0,
    w_anchor=5.0,
    w_orth=0.2,
    tau=0.07,
    device=torch.device('cpu'),
):
    rng = np.random.default_rng(seed)
    schedule = [rng.choice(train_ids, batch_size, replace=True).tolist() for _ in range(steps)]
    torch.manual_seed(seed)

    model = RelPosOrthSubspaceAdapter(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        freeze_sem=freeze_sem,
        use_relpos=use_relpos,
        use_orth=use_orth,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    train_imgs = sorted({dataset[i]['s'] for i in train_ids} | {dataset[i]['t'] for i in train_ids})
    img_dev = {k: dense_norm[k].to(device) for k in train_imgs}
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
        x_batch = torch.stack([img_dev[nm] for nm in uniq_imgs], dim=0)
        sem_batch, role_batch, orth_loss, anchor_loss = model(x_batch)

        cc_losses = []
        soc_role_losses = []
        soc_comb_losses = []

        for d in items:
            si = img_to_idx[d['s']]
            ti = img_to_idx[d['t']]

            zs_sem = sample_dense_map(sem_batch[si], d['s_xy'])
            zt_sem = sample_dense_map(sem_batch[ti], d['t_xy'])
            zs_role = sample_dense_map(role_batch[si], d['s_xy'])
            zt_role = sample_dense_map(role_batch[ti], d['t_xy'])

            sim_sem_kp = zs_sem @ zt_sem.T
            sim_role_kp = zs_role @ zt_role.T

            if use_dense_negs:
                ft_sem_flat = sem_batch[ti].flatten(1).T
                ft_role_flat = role_batch[ti].flatten(1).T
                sim_sem_all = torch.cat([sim_sem_kp, zs_sem @ ft_sem_flat.T], dim=-1)
                sim_role_all = torch.cat([sim_role_kp, zs_role @ ft_role_flat.T], dim=-1)
                sim_comb_all = sim_sem_all.detach() + alpha_train * sim_role_all

                if not freeze_sem and w_cc > 0:
                    lc = contrastive_pair_loss(sim_sem_all, d['pos_cc_all'], d['val_cc_all'], tau=tau)
                    if lc is not None:
                        cc_losses.append(lc)

                if w_soc_role > 0:
                    lsr = contrastive_pair_loss(sim_role_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
                    if lsr is not None:
                        soc_role_losses.append(lsr)
                if w_soc_comb > 0:
                    lsc = contrastive_pair_loss(sim_comb_all, d['pos_soc_all'], d['val_soc_all'], tau=tau)
                    if lsc is not None:
                        soc_comb_losses.append(lsc)
            else:
                sim_comb_kp = sim_sem_kp.detach() + alpha_train * sim_role_kp
                if not freeze_sem and w_cc > 0:
                    lc = contrastive_pair_loss(sim_sem_kp, d['cc'], d['ones_kp'], tau=tau)
                    if lc is not None:
                        cc_losses.append(lc)
                if w_soc_role > 0:
                    lsr = contrastive_pair_loss(sim_role_kp, d['soc'], d['ones_kp'], tau=tau)
                    if lsr is not None:
                        soc_role_losses.append(lsr)
                if w_soc_comb > 0:
                    lsc = contrastive_pair_loss(sim_comb_kp, d['soc'], d['ones_kp'], tau=tau)
                    if lsc is not None:
                        soc_comb_losses.append(lsc)

        loss = w_orth * orth_loss
        if not freeze_sem:
            loss = loss + w_anchor * anchor_loss
            if cc_losses:
                loss = loss + w_cc * torch.stack(cc_losses).mean()
        if soc_role_losses:
            loss = loss + w_soc_role * torch.stack(soc_role_losses).mean()
        if soc_comb_losses:
            loss = loss + w_soc_comb * torch.stack(soc_comb_losses).mean()

        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append({
                'step': step,
                'loss': float(loss.detach().cpu()),
                'orth': float(orth_loss.detach().cpu()),
                'anchor': float(anchor_loss.detach().cpu()),
            })

    sync_device(device)
    model.eval()
    return model, time.perf_counter() - ts, trace


def make_mlp_predictor(model, dense_mlp, device):
    model.eval()
    def predict_pair(d):
        xs = d['xs'].to(device)
        xt = d['xt'].to(device)
        td = dense_mlp[d['t']].to(device)
        a0 = model(xs, 0)
        b0 = model(td, 0)
        bt0 = model(xt, 0)
        a1 = model(xs, 1)
        b1 = model(td, 1)
        bt1 = model(xt, 1)
        return a0 @ b0.T, a1 @ b1.T, a0 @ bt0.T, a1 @ bt1.T
    return predict_pair


def make_relpos_single_predictor(model, dense_norm, device):
    model.eval()
    def predict_pair(d):
        x = torch.stack([dense_norm[d['s']], dense_norm[d['t']]], dim=0).to(device)
        f = model(x)
        fs, ft = f[0], f[1]
        zs = sample_dense_map(fs, d['s_xy'].to(device))
        zt = sample_dense_map(ft, d['t_xy'].to(device))
        ft_flat = ft.flatten(1).T
        sd = zs @ ft_flat.T
        sc = zs @ zt.T
        return sd, sd, sc, sc
    return predict_pair


def make_relpos_orth_predictor(model, dense_norm, device, alpha=0.6, role_only=False):
    model.eval()
    def predict_pair(d):
        x = torch.stack([dense_norm[d['s']], dense_norm[d['t']]], dim=0).to(device)
        sem_b, role_b, _, _ = model(x)
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

        if role_only:
            sd_soc = zs_role @ ft_role.T
            sc_soc = zs_role @ zt_role.T
        else:
            sd_soc = sd_cc + alpha * (zs_role @ ft_role.T)
            sc_soc = sc_cc + alpha * (zs_role @ zt_role.T)
        return sd_cc, sd_soc, sc_cc, sc_soc
    return predict_pair
