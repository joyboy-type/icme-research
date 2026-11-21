# Paper 2 · Round 2 技术设计：Null-Slot Cross-Attention + Counterfactual Modality-Swap

> 作者：claude-av-design · 纯代码阅读 + 机制推导，**未运行任何训练**。
> 依据：`REPORT.md`、`train_ave.py`、`models/nullslot_av.py`、`train_llp.py`、`protocol.json`、`results/ave_pilot_results.json`。
> 唯一做过的本地计算：读取 `labels.h5` 统计 BG 比例（train 16.9% / test 17.8%）。

---

## 0. 一句话结论

**Clean Acc 掉 7.3pp 的主因不是 Null-Slot 过度弃权，而是两个 CF 变体共享的三个 CF 损失实现问题**——证据：没有 Null-Slot 的 `baseline_cfswap` 掉得一样多（73.76→66.81，`nullslot_cfswap` 是 66.45）。$\alpha_{\text{null}}^{ev}=0.635$ 是**次要症状**，来自一个与 query 无关、在所有输入间共享的 null 偏置被"绝对目标 = 1"的 swap 损失推高。修复思路：(i) 修好共享的 CF 项（KL 归约方式、BG 先验偏移、梯度裁剪挤占）；(ii) 把 null 监督从**绝对目标**改成**同 query 配对的 logit 空间排序损失**。在第 1 层，这个损失里 null 偏置和 null key **严格相消**，所以只训练"key 匹配"这一能力，不会再出现全局漂移。

---

## 1. 根因诊断

### 1.1 Clean Acc 下降全部来自事件段，而且是共享 CF 项造成的

| 指标 | baseline_xattn | baseline_cfswap | nullslot_cfswap |
|---|---|---|---|
| Clean Acc | 73.76 | 66.81 | 66.45 |
| AV event acc (82% 的段) | **79.47** | 71.68 | **70.87** |
| BG acc (18% 的段) | 47.37 | 44.29 | 46.01 |
| Clean A-Ev / V-Ev | 79.5 / 78.4 | 72.2 / 67.2 | 75.5 / 70.0 |

核对：0.822×70.87 + 0.178×46.0 ≈ 66.4 ✔。BG acc 几乎没变，掉点**全部**来自事件段 → 事件段被判成 BG 或判错类。要回到 ≥73.5，AV event acc 需要回到 ≈79%。

### 1.2 四个确切的代码/梯度原因（按影响从大到小）

**R1 · BG 拒识损失让 AV 头的 BG 先验变成约 2.6 倍**（`train_ave.py` L319–327）
- 干净 CE：每个 batch B·T 段，其中约 17% 是 BG。
- `loss_bg`：Swap-V 和 Swap-A 两份均值的平均，权重 `cf_bg_weight=0.5`，**目标 100% 是 BG**。
- AV 头监督中 BG 的有效占比：$(0.17\cdot 1 + 1.0\cdot 0.5)/(1+0.5)\approx \mathbf{44.6\%}$，原本只有 17%。最终线性层的 BG bias 会跟着往上走，于是边缘的事件段被判成 BG。这一点同时解释了两个 CF 变体的 event acc 都掉了约 8pp，而 BG acc 没有涨。

**R2 · KL 不变性损失实际被放大约 10 倍，而且从第 0 轮起就拿近似均匀的 teacher 去约束**（L313–317）
- `F.kl_div(..., reduction="batchmean")` 作用在 `[B,T,C]` 上时**只除以 B**，T=10 个段的 KL 是求和的，所以每段的实际权重是 0.5×10 = **5.0**，而 null 项只有 0.3、BG 项 0.5。
- teacher `p_clean` 包括 BG 段和预测错的段；训练初期它接近均匀分布，KL 等于一个强熵正则，再加上两次 forward 用了不同 dropout mask，又叠了一层 R-Drop 式噪声。
- 在没有 Null-Slot 的模型里，最小化 $\mathrm{KL}(p_a(A_i,V_i)\,\|\,p_a(A_i,V_j))$ 的捷径是**把跨模态通路整体关掉**（`v_proj`/`out_proj` 缩小、或者均匀注意力），于是干净流也失去跨模态增益：baseline_cfswap 的 Clean V-Ev 从 78.4 掉到 67.2。

**R3 · 全局梯度裁剪 `clip_grad_norm_(…, 1.0)` 被 CF 梯度挤占**（L338）
- 总损失 = 干净 CE + 等效约 5 倍的 KL + BG + null，而且是三次 forward。裁剪之后，干净 CE 梯度在更新方向里的份额被按比例压缩，相当于干净任务的有效学习率下降；在 25 epoch、余弦退火的条件下就会欠拟合（val clean 只有 69.0/69.9，而 baseline 是 74.7）。

**R4 · Null 监督用了绝对目标，梯度经过与 query 无关的共享参数，造成全局漂移**（L293–308）
- 对单个 head：
  $$\operatorname{logit}\alpha_{\text{null}} = s_{\text{null}} - \operatorname{LSE}_j(s_j),\quad s_{\text{null}}=\tfrac{q^\top k_{\text{null}}}{\sqrt d}+b,\ \ s_j=\tfrac{q^\top k_j}{\sqrt d}$$
- `loss_null_swap = (1-α_sw)^2`：目标是 1.0（softmax 下要求 logit→∞），作用在 2·B·T 个元素上（包括 BG 段），权重 0.3。降低它**最便宜**的方法是增大 $b$，并把 $k_{\text{null}}$ 对齐到 query 的均值方向——这两者对干净输入同样生效。
- 对抗项 `loss_null_align = relu(α_cl − 0.25)^2` 的实际权重只有 0.3×0.5 = 0.15，只覆盖事件段，又是平方 hinge，靠近阈值时梯度很小；而且它约束的是**两个方向 × 层 × head 的平均值**，个别 head 可以完全弃权。
- 结果：所有输入的 α 一起上移（ev 0.43→0.635，bg 0.52→0.72，swap 0.41→0.88），排序保住了，但绝对值漂了。
- **可辨识性问题**（审稿人也可能攻击）：输出是 $(1-\alpha)\,\text{ctx}$，下游的 `out_proj`/残差/LayerNorm 可以通过放大 $\|W_oW_v\|$ 抵消 α 的缩放。证据：不带 CF 的 `nullslot_xattn` 中 α_ev 已经是 0.43，但 Clean Acc 只掉了 0.5pp。所以 **α 数值大 ≠ 功能上弃权**，需要另报"有效融合比" $r=\|W_o\text{ctx}\|/\|h\|$。

**次要问题**：评估中 clean α 是 a2v、v2a 两个方向的平均，swap α 只取单一方向，口径不一致，应改为分方向报告；验证每 5 个 epoch 才做一次，只有 5 个候选 checkpoint，选点太粗。

### 1.3 Null-Slot 自身其实是正贡献

在相同 CF 损失下，Null-Slot 让 Clean A-Ev +3.4、V-Ev +2.8、Swap-V A-Ret +2.3、Swap-A V-Ret +3.2、val clean +0.9。所以 Round 2 叙事应改为：**"CF 正则本身会伤害 clean；Null-Slot 是让 CF 正则变得廉价的结构条件"**，而不是"Null-Slot 引入了 trade-off"。

---

## 2. Clean-Acc 恢复方案（v2 损失）

### 2.1 总损失

$$\mathcal L=\mathcal L_{\text{clean}}+\lambda(t)\Big[w_r\mathcal L_{\text{rank}}+w_c\mathcal L_{\text{cal}}+w_i\mathcal L_{\text{inv}}+w_b\mathcal L_{\text{bg}}\Big],\quad
\lambda(t)=\begin{cases}0 & t<5\\ (t-5)/5 & 5\le t<10\\ 1 & t\ge10\end{cases}$$

**(a) 配对反事实排序损失 PCAR（核心创新，替换 `(1-α)^2`）**：同一 query（A_i 的第 ℓ 层、head h、段 t）在匹配的 key（V_i）和不匹配的 key（V_j）下比较，
$$\mathcal L_{\text{rank}}=\frac{1}{|\mathcal M|}\sum_{\ell,h,t\in\mathcal M}\operatorname{softplus}\!\Big(m-\big[\operatorname{logit}\alpha^{\ell h}_{\text{sw}}(t)-\operatorname{logit}\alpha^{\ell h}_{\text{cl}}(t)\big]\Big),\quad m=2.0$$
$\mathcal M$ = {y_i(t)≠BG 且 video_cls 不同}，两个方向都算（a2v 用 Swap-V，v2a 用 Swap-A）。
**关键性质**：第 1 层的 query 只由本模态自注意力得到，所以在 clean 和 swap 两次 forward 中**完全相同**，
$$\operatorname{logit}\alpha_{\text{sw}}-\operatorname{logit}\alpha_{\text{cl}}=\operatorname{LSE}_j(q^\top k^{(i)}_j)-\operatorname{LSE}_j(q^\top k^{(j)}_j)\;/\sqrt d$$
其中 $b$ 和 $k_{\text{null}}$ **严格相消，从排序损失得到的梯度为 0**。这个损失只训练"跨模态 key 匹配"，从机制上消除 R4 的全局漂移。在 logit 空间计算也避免了 softmax 饱和。

**(b) 轻量绝对校准 CAL**（只用来固定 α 的绝对位置）：
$$\mathcal L_{\text{cal}}=\overline{\operatorname{relu}(\alpha^{\ell h}_{\text{cl}}-\tau_{ev})^2}_{y\neq BG}+\overline{\operatorname{relu}(\tau_{sw}-\alpha^{\ell h}_{\text{sw}})^2}_{\mathcal M},\quad \tau_{ev}=0.3,\ \tau_{sw}=0.7$$
按**每个 head、每一层**分别施加，不再约束平均值；目标从 1.0 改成 0.7 这个下界，避免把 logit 推向无穷。

**(c) 带标签、带置信门控的不变性 INV**（修 R2）：
$$\mathcal L_{\text{inv}}=\underbrace{\mathrm{CE}(z_a^{sv},y_i)+\mathrm{CE}(z_v^{sa},y_i)}_{\text{仅 }y_i\neq BG}\cdot\tfrac12+\beta\cdot\underbrace{\mathrm{KL}(p^{cl}\|p^{sw})}_{\text{按段平均，仅 teacher 正确的事件段}},\ \beta=0.5$$
AVE 的事件段按定义 audible 且 visible，所以 y_i 对 A_i 流、V_i 流都是有效标签，直接监督保留率，不需要压低熵。

**(d) 去偏的 BG 拒识**（修 R1）：只在 hard negative 段 $\{y_i\neq BG,\ y_j\neq BG,\ y_i\neq y_j\}$ 上施加（和评估掩码 `active_diff_mask` 一致），$w_b=0.15$。AV 头 BG 有效占比从 44.6% 降到约 $(0.17+0.15\cdot0.7)/(1.15)\approx 24\%$。可选：对 BG 的 logit 做 $-\log(\pi'_{BG}/\pi_{BG})$ 的 logit adjustment，只在推理时使用。

**(e) 可选创新 ARC：弃权→拒识耦合**（作为 Variant A3 做消融）：
$$z_{av}[BG]\leftarrow z_{av}[BG]+\gamma\cdot\tfrac12\big(\operatorname{logit}\bar\alpha_{a2v}+\operatorname{logit}\bar\alpha_{v2a}\big),\ \gamma\ \text{可学习，初始化为 0}$$
让 AV 的 BG 判决**由 α 解释**，而不是靠分类器 bias，同时把 Null-Slot 从"内部门控"变成"可操作的决策变量"，用来回应审稿人"α 只是可视化"的质疑。

**推荐初始超参**：$w_r=0.3,\ w_c=0.1,\ w_i=0.5,\ w_b=0.15$，warmup 5 + ramp 5，`clip_grad_norm` 从 1.0 改为 **5.0**（并记录裁剪前的范数），从第 5 个 epoch 起**每个 epoch** 验证一次。

**选点规则**（所有变体统一）：在 val 上满足 Swap-V A-Ret ≥0.70、Swap-A V-Ret ≥0.65、AV-Rej ≥0.90 的约束下取 clean acc 最大的 checkpoint；没有满足约束的就取 clean 最大。论文中如实声明使用约束选点，baseline 也用同一规则。

**预期**（推断，需 Round 2 验证）：R1+R2+R3 修复后，两个 CF 变体的 clean 预计各回升 4–6pp；PCAR 把 α_ev 拉到 ≤0.3，同时 α_sw−α_ev 的间隔扩大，Null 变体预计比 baseline_cfswap_v2 再高 +0.5–1.5pp，达到 ≥73.5 的目标。

### 2.2 代码 diff 建议（先不执行）

**`models/nullslot_av.py`：返回每个 head 的 null log-odds**
```diff
@@ class NullSlotCrossAttention.forward
-            probs_aug = F.softmax(logits_aug, dim=-1)
-            alpha_null = probs_aug[..., -1].mean(dim=1)  # [B, T_q]
+            probs_aug = F.softmax(logits_aug, dim=-1)
+            alpha_null_h = probs_aug[..., -1]                                  # [B, H, T_q]
+            # logit(alpha) = s_null - LSE_j(s_j)  (bias/null_k cancel in paired diffs)
+            null_logodds = null_logit.squeeze(-1) - torch.logsumexp(logits, dim=-1)  # [B, H, T_q]
+            alpha_null = alpha_null_h.mean(dim=1)
             probs = self.attn_drop(probs_aug[..., :-1])
         else:
             probs = self.attn_drop(F.softmax(logits, dim=-1))
-            alpha_null = None
+            alpha_null, alpha_null_h, null_logodds = None, None, None
         ctx = torch.matmul(probs, v)
         ctx = ctx.transpose(1, 2).contiguous().view(B, T_q, D)
-        return self.out_proj(ctx), alpha_null
+        out = self.out_proj(ctx)
+        self.last_aux = {"alpha_h": alpha_null_h, "logodds": null_logodds,
+                         "fusion_ratio": out.norm(dim=-1) / (q_in.norm(dim=-1) + 1e-6)}
+        return out, alpha_null
```
在 `AVNullSlotModel.forward` 中收集各层的 `xa_a2v.last_aux` / `xa_v2a.last_aux`，作为 `out["aux_a2v"]`、`out["aux_v2a"]` 返回（list，长度 = n_layers）。可选 ARC：
```diff
+        self.arc_gamma = nn.Parameter(torch.zeros(1)) if (use_null_slot and use_arc) else None
 ...
         logits_av = self.head_av(joint_feat)
+        if self.arc_gamma is not None:
+            la = torch.logit(alpha_null_a2v.clamp(1e-4, 1-1e-4)); lv = torch.logit(alpha_null_v2a.clamp(1e-4, 1-1e-4))
+            bg_shift = torch.zeros_like(logits_av); bg_shift[..., -1] = self.arc_gamma * 0.5 * (la + lv)
+            logits_av = logits_av + bg_shift   # BG_CLASS is last index (28)
```

**`train_ave.py`：CF 块整体替换**（L285–334）
```diff
-            if use_cf:
+            lam = 0.0 if epoch < cfg.get("cf_warmup", 5) else min(1.0, (epoch - cfg.get("cf_warmup", 5) + 1) / cfg.get("cf_ramp", 5))
+            if use_cf and lam > 0:
                 perm = make_cross_class_perm(video_cls)
                 y_perm = y[perm]
-                diff_mask = (video_cls != video_cls[perm]).unsqueeze(1).expand_as(y)
+                vid_diff = (video_cls != video_cls[perm]).unsqueeze(1).expand_as(y)
+                ev_i = y != BG_CLASS
+                hard = vid_diff & ev_i & (y_perm != BG_CLASS) & (y != y_perm)   # == eval active_diff_mask
+                M = vid_diff & ev_i
                 out_sv = model(audio, visual[perm])
                 out_sa = model(audio[perm], visual)
 
-                if use_null:  ... (1-alpha)^2 ... relu(alpha-0.25)^2 ...
+                loss_null = audio.new_zeros(())
+                if use_null and M.any():
+                    rank, cal = [], []
+                    for l in range(len(out["aux_a2v"])):
+                        for clean_aux, sw_aux in ((out["aux_a2v"][l], out_sv["aux_a2v"][l]),
+                                                  (out["aux_v2a"][l], out_sa["aux_v2a"][l])):
+                            d = (sw_aux["logodds"] - clean_aux["logodds"]).permute(0, 2, 1)   # [B,T,H]
+                            rank.append(F.softplus(cfg["rank_margin"] - d[M]).mean())
+                            a_cl = clean_aux["alpha_h"].permute(0, 2, 1); a_sw = sw_aux["alpha_h"].permute(0, 2, 1)
+                            cal.append((F.relu(a_cl[ev_i] - cfg["tau_ev"]) ** 2).mean()
+                                       + (F.relu(cfg["tau_sw"] - a_sw[M]) ** 2).mean())
+                    loss_null = cfg["w_rank"] * torch.stack(rank).mean() + cfg["w_cal"] * torch.stack(cal).mean()
 
-                kl_a = F.kl_div(..., reduction="batchmean") ...
+                # label-grounded + teacher-gated invariance (per-segment mean, NOT batchmean over [B,T,C])
+                loss_inv = audio.new_zeros(())
+                if M.any():
+                    ce = 0.5 * (F.cross_entropy(out_sv["logits_a"][M], y[M]) + F.cross_entropy(out_sa["logits_v"][M], y[M]))
+                    ok_a = M & (out["logits_a"].argmax(-1) == y); ok_v = M & (out["logits_v"].argmax(-1) == y)
+                    kl = audio.new_zeros(())
+                    if ok_a.any():
+                        kl = kl + F.kl_div(F.log_softmax(out_sv["logits_a"][ok_a], -1), F.softmax(out["logits_a"][ok_a].detach(), -1), reduction="batchmean")
+                    if ok_v.any():
+                        kl = kl + F.kl_div(F.log_softmax(out_sa["logits_v"][ok_v], -1), F.softmax(out["logits_v"][ok_v].detach(), -1), reduction="batchmean")
+                    loss_inv = ce + cfg["inv_kl_beta"] * 0.5 * kl
 
-                bg_target = torch.full((diff_mask.sum().item(),), BG_CLASS, ...)
+                loss_bg = audio.new_zeros(())
+                if hard.any():
+                    bg_t = torch.full((int(hard.sum()),), BG_CLASS, dtype=torch.long, device=device)
+                    loss_bg = 0.5 * (F.cross_entropy(out_sv["logits_av"][hard], bg_t) + F.cross_entropy(out_sa["logits_av"][hard], bg_t))
 
-                loss = loss + cfg["cf_null_weight"] * loss_null + cfg["cf_invariance_weight"] * loss_inv + cfg["cf_bg_weight"] * loss_bg
+                loss = loss + lam * (loss_null + cfg["cf_invariance_weight"] * loss_inv + cfg["cf_bg_weight"] * loss_bg)
 ...
-            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
+            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 5.0))   # log gn
 ...
-        if (epoch + 1) % 5 == 0 or epoch == cfg["epochs"] - 1:
+        if epoch >= cfg.get("cf_warmup", 5) - 1:
             val_m = evaluate_ave(model, val_ds, device)
-            if val_m["clean_acc"] > best_val_score:
+            ok = (val_m["swap_v_audio_retain_acc"] >= 0.70 and val_m["swap_a_visual_retain_acc"] >= 0.65
+                  and min(val_m["swap_v_av_reject_rate"], val_m["swap_a_av_reject_rate"]) >= 0.90)
+            score = val_m["clean_acc"] + (1.0 if ok else 0.0)       # constrained-first selection
+            if score > best_val_score: ...
```
`protocol.json` 新增：`cf_warmup=5, cf_ramp=5, rank_margin=2.0, tau_ev=0.3, tau_sw=0.7, w_rank=0.3, w_cal=0.1, inv_kl_beta=0.5, cf_invariance_weight=0.5, cf_bg_weight=0.15, clip=5.0`。

**`evaluate_ave` 补充指标**（对审稿价值很高，几乎不增加计算）：
- `event→BG` 误判率（直接验证 R1 是否修好）；
- α 按 `a2v`/`v2a` 分方向、按层报告，并报 `fusion_ratio`（应对可辨识性质疑）；
- α 区分 matched vs swapped 的 **AUROC**；
- 下文 §4 C 组的 held-out 扰动指标。

---

## 3. 创新壁垒：审稿人最可能攻击的 3 个弱点及应对

### W1 ·「Null-Slot = 已知技巧」（Attention Sink / StreamingLLM、softmax-off-by-one、Register tokens（ICLR'24）、Gated Attention（2025）、sigmoid attention）
**应对机制**：贡献不在"加一个 slot"，而在两点：
1. **Key-aware 弃权**：$\operatorname{logit}\alpha=s_{\text{null}}-\operatorname{LSE}_j(s_j)$ 依赖 query 与**对方模态 key** 的匹配度。Gated attention / output gate 的 $g(q)$ 只看 query，**在第 1 层从结构上无法区分 (A_i,V_i) 和 (A_i,V_j)**（同一 query、不同 key），这是一个可以写成命题的不可能性论证。
2. **PCAR 的 bias 相消性质**：反事实监督只训练匹配函数，不训练全局 sink 偏置，这是 Attention-Sink 类工作（emergent、无监督、用于 VQA 幻觉）没有的。

**实验堵口**（同 backbone、同 CF 损失，只换算子）：
(i) softmax+1（null logit 固定为 0、无可学习参数）；(ii) learnable sink/register，value **非零**；(iii) query-only sigmoid gate（Gated-Attn 式）；(iv) Null-Slot + 绝对目标（v1）；(v) Null-Slot + PCAR（v2）。
预测：(iii) 在 AV-Rej 和 α-AUROC 上明显更差；(ii) 出现 distractor 泄漏（value 非零，等于固定偏置向量）。

### W2 ·「反事实 swap 只是负样本 / Both-or-Neither 已经在 AVEL 上做了 mute/swap/shift 审计；CoLeaF 已经过滤跨模态噪声」
**应对机制**：B-o-N 在**后期 logit/门控层**强制"要么都有要么都没有"，对 LLP 中合法的 A-only/V-only 事件**从定义上是有害的**。我们是**算子级、逐 token、逐 head** 的弃权，并且对单模态流显式做**保留监督**（INV 带标签 CE），两者目标相反。CoLeaF 是双分支对比损失，不改注意力算子，也没有用反事实监督弃权。
**实验堵口**：
- 同 backbone 复现 B-o-N 式 late gating（AV logit × 一致性门控 + swap/mute 审计损失）作为 baseline。在 AVE 上它可能和我们持平，**关键在 LLP**：报告 A-only / V-only 段的 F1 和 Type@AV、Event@AV，预计 B-o-N 式会抑制单模态事件。
- LLP 上与 CoLeaF/HAN/JoMoLD 用官方特征对比，并把模块**插到 HAN（以及一个更强的 baseline）上**展示增量收益，说明这是正交、可叠加的贡献。

### W3 ·「自造 swap 训练、自造 swap 测试，循环论证；而且可能学到的是'视频指纹/录音条件'捷径而不是语义对齐；clean 还掉点」
**应对机制**：
1. 训练中加入**同类跨视频正样本 swap**：(A_i, V_k)，其中 cls_k = cls_i，要求 α 低、AV = 该类。这样"是否同一视频"的捷径会受到惩罚，只有语义对齐才能同时满足正负两类样本。
2. 评估时全部用**训练未见的扰动族**：同视频时间错位（±2~5 段）、静音/高斯噪声、局部 swap（只换中间 4 段，检验时间局部性）、同类跨视频 swap（期望**不弃权**）。
3. **LLP 天然非对齐段的零样本 α-AUROC**：用 dense 标注里的 A-only/V-only 段 vs AV 段，检验 α 能否在**没有任何 swap 的真实数据**上把它们区分开——这是最有力的反循环证据。
4. Clean 不低于 baseline（§2 的目标），并给出 λ 扫描的 Pareto 曲线证明我们**支配** baseline_cfswap。

### 额外风险（必须提前处理）
AVE 上 73.8% 的 backbone 明显弱于 PSP/CMRAN/CMBS（76–79%，VGG 特征）。审稿人会说"在弱 baseline 上加 trick"。建议 Round 3 把 Null-Slot 插入一个公开的 CMBS 或 PSP 实现（同样只用预提取特征，算力小），证明在 ≥77% 的 backbone 上 clean 也不掉点，同时 stress test 提升。

---

## 4. Round 2 决定性实验清单

### AVE Round 2（本地，3 seeds，25 epochs；按 Round 1 实测耗时估算，每个 CF 变体约 7–10 分钟，全部约 1.5–2 小时）

| ID | 变体 | 验证目的 / 判定 |
|---|---|---|
| A0 | baseline_xattn（复用 R1 结果，补齐新指标后重跑一次） | 参照点 |
| A1 | baseline_cfswap **v2**（R1+R2+R3 修复，无 Null） | 验证"掉点主因在共享 CF 项"：clean 应回到 ≥71 |
| A2 | nullslot_cfswap v2-abs（共享项修复 + 旧的绝对 null 目标） | 单独隔离 R4：α_ev 是否仍漂到 >0.5 |
| **A3** | **nullslot_cfswap v2（PCAR + CAL）** | **主结果**：Clean ≥73.5、Ret ≥70/65、Rej ≥90、α_ev ≤0.3 |
| A4 | A3 + ARC（弃权→拒识耦合） | 是否进一步提升 clean BG acc / AV-Rej；γ 是否学成正值 |
| A5 | A3 + 同类跨视频正样本 swap | 反捷径：同类 swap 下 α 应低、AV 应正确 |
| B1 | softmax+1 + v2 CF | W1：可学习 null key 是否必要 |
| B2 | 非零 value 的 sink/register + v2 CF | W1：零 value 是否必要（泄漏） |
| B3 | query-only sigmoid gate + v2 CF | W1：key-aware 是否必要（核心命题） |
| D1 | 最佳 Null 变体 vs A1，λ∈{0.25, 0.5, 1, 2} | Pareto 支配曲线（clean vs retention） |
| C* | **仅评估、不训练**：对 A0/A1/A3/A5/B3 跑 held-out 扰动（时间错位、静音/噪声、局部 swap、同类 swap） | W3：泛化到未见扰动；局部 swap 检验逐段弃权 |

**优先级**：A1 → A3 → A2 → B3 → A5 → C* → B1/B2 → A4 → D1。如果 A3 达不到 73.5，按顺序调 $w_b\downarrow0.1$ → `cf_warmup` 8 → $\tau_{ev}$ 0.2，**每次只改一个**。

**判定规则**：A1 回升 ≥4pp 且 A3 − A1 ≥ +0.5pp clean，同时 retention 更高 → 叙事成立。若 A3 ≈ A1 → Null 的价值只剩可解释性，需要把卖点转到 LLP 的单模态 F1 上。

### LLP / AVVP（主战场，特征下载后进行）
`train_llp.py` 需要先修：
(1) `torch.roll` 配对**不能保证标签互斥**——LLP 是多标签，Speech 非常常见，必须要求 $Y_i\cap Y_j=\varnothing$ 才能当负样本，交集非空时当作该类的正样本；
(2) 当前 `p_video_av` 是 a/v 的 noisy-OR，没有模态区分的弱标签；建议接 HAN 的 label smoothing / MA 式模态伪标签；
(3) 同样改为 PCAR + CAL + 按段平均的 INV；
(4) 加入官方 event-level F1 评估和 test split。

| ID | 实验 | 目的 |
|---|---|---|
| L1 | HAN-style baseline / +Null / +CF v2 / +Null+CF v2（2×2） | 官方 segment- / event-level A、V、AV、Type、Event F1 |
| L2 | B-o-N 式 late gating 复现 | W2：在 A-only / V-only F1 上对比，证明我们保留单模态 |
| L3 | query-only gate vs Null-Slot（+相同 CF） | W1 在 LLP 上复现 |
| L4 | α 零样本 AUROC：A-only/V-only vs AV 段（dense 标注） | W3：无需 swap 的真实非对齐检测 |
| L5 | 插入更强的 baseline（JoMoLD / CoLeaF 式去噪） | 正交、可叠加；对标 2024–25 SOTA |

**论文最小主表**：AVE（A0/A1/A3 + B3） + LLP（L1 + L2 + L5），α-AUROC（L4）和 Pareto（D1）作为分析图。
