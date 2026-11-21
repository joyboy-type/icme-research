**裁决：【有混杂因素需补 1 组公平对照后再定】。** 我从逐 query CSV 按图像对、类别、种子重新汇总，复得 (4b) **SOC 50.75%、同查询 CC 73.23%**。整套方法的提升可信，但现有实验无法把提升归因于“正交子空间与 2D-RelPos 的严格协同”。[实验报告](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/REPORT.md:80)

**1．公平性：dense negatives 已对齐，训练信号和推理流程没有对齐。**

| 实际运行 | 训练配置 | 生效损失 | 推理 |
|---|---|---|---|
| (2) MLP，73,984 参数 | 600 步，Adam，lr 1e-3，仅标注关键点参与分母 | 0.5 CC + 0.5 SOC | 按任务分别打分 |
| (3) Single，364,552 参数 | 300 步，AdamW，lr 5e-4，dense negatives | 0.5 CC + 0.5 SOC，同一张相似度图 | CC、SOC 共用一个 argmax |
| (4a) 冻结语义，364,936 参数 | 300 步，AdamW，lr 3e-4，dense negatives | 融合 SOC（α=0.5）+ 0.2 正交；CC、独立 role SOC、anchor 均未生效 | CC 用语义分数，SOC 用语义 + 0.5×角色 |
| (4b) Full，414,088 参数 | 300 步，AdamW，lr 5e-4，dense negatives | 0.5 CC + 0.3 role SOC + 融合 SOC（α=0.6）+ 10 anchor + 0.2 正交 | 与 (4a) 相同，推理 α=0.5 |

(3)/(4a)/(4b) 同种子抽取相同的 300×8 训练对，使用同一套 400 格候选与按任务正点距离筛选的掩码。**没有 (4b) 独享的负样本或人工标签**；它独享生效的逐 patch DINO 语义锚定和独立 role SOC 辅助损失。相对 (3) 还多 49,536 个参数，并允许两个任务输出不同位置。尤其值得注意：**按实际 SOC 预测位置计算，(4b) 的同查询 CC 是 68.83%；73.23% 来自单独的语义分支。** [运行配置](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/run_confirm.py:32) · [负样本](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/train_eval.py:101) · [损失](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/train_eval.py:279) · [推理](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/train_eval.py:358)

**2．泄露与坐标：主指标通过代码审计，“确认集从未被看过”无法认证。**

- 实查清单，train/dev/confirm 的图像标识和类别两两无交集，confirm 确为 48 图、146 对。搜索脚本只用 train 更新、dev 评分，确认脚本使用固定配置。但旧 pilot 已评过同一确认集，确认脚本还硬编码了旧确认分数；现有记录只能证明本轮脚本没有在 confirm 上搜索。另，选定 (4b) 配置在 dev 上 α=0.25 的 SOC 为 57.00%，α=0.5 为 56.93%，未见预设选择规则。这不构成泄露证据。[开发搜索](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/dev_search.py:67) · [开发结果](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/dev_search_results.json) · [旧分数断言](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/run_confirm.py:139)
- Dense PCK 只需给定源查询点，目标端在全部 400 格取最大值；目标真值坐标进入并行的 oracle 候选诊断和事后评分，不影响 dense argmax。与 `soco_grid_corrected` 共用同一特征和配对文件，patch 中心、`pos/pad`、零填充双线性采样、目标框最长边归一化及严格 `<0.1` 均一致。**特征处理有差别**：RelPos 先归一化各 patch 再插值，旧 frozen 先插值再归一化，冻结语义分支的 CC 因此相差约 0.20 个百分点。[当前实现](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/data_utils.py:52) · [评分](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/data_utils.py:225) · [原协议](/Users/joyboy/Documents/ChatGPT/ICME/work/icme_direction_audit_20261121/soco_grid_corrected/run.py:14)

**3．最强的两个审稿攻击点，以及能改变裁决的实验。**

1. **普通双头加辅助损失就可能解释提升。** SOCO 已定义概念与角色对应的区别，MARCO 已展示 DINO 对应适配与稠密训练信号的价值。当前 Gram–Schmidt 只保证同一 patch 的两个向量正交，共享骨干仍接收角色梯度；(4b) 对 (4a) 的 SOC 差也只有 +2.21 pp，95% CI 为 [−0.66, +5.01]。**首要公平对照**是同参数普通双头：仅关闭 Gram–Schmidt 和正交惩罚，保留角色归一化、全部其余损失、detach、负样本、训练预算和推理 α。[模型实现](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/models.py:218) · [SOCO](https://arxiv.org/abs/2605.31597) · [MARCO](https://arxiv.org/abs/2604.18267)
2. **尚无证据表明显式二维几何解决了重复部件歧义。** Shape-of-You 已用结构匹配处理此问题，Geometry Matters 已用三维先验处理对称侧与重复部件；本方法相对 (3) 的 wrong-role 反而是 +0.10 pp，95% CI 为 [−2.14, +2.32]。**若保留“严格协同”主张**，把上一对照扩成一组 2×2：显式 RelPos 开/关 × 正交开/关，关闭 RelPos 时保留普通自注意力，同时去掉 RoPE 和相对位置偏置。在新独立确认集上预设同概念异角色子集，报告 SOC、wrong-role 和配对交互效应。[结果表](/Users/joyboy/.gemini/antigravity/scratch/icme2027-research/paper1_soco_relpos/REPORT.md:42) · [Shape-of-You](https://arxiv.org/abs/2603.11618) · [Geometry Matters](https://arxiv.org/abs/2605.30093)