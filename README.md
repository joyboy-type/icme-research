# ICME 2027 Research

本仓库按**“每方向一个独立目录、仅保存轻量代码快照 + 实验协议 + JSON 汇总 + 审稿审计报告”**的规范维护研究记录。完整数据集、预提取特征、模型权重与逐样本缓存保存在本地归档，不入库。

---

## 一、主推活跃方向（ICME 2027 候选）

| 目录 | 模态 / ICME 主题领域 | 核心机制 | 当前状态与实测信号 |
|---|---|---|---|
| [role-correspondence-relpos/](role-correspondence-relpos/) | 视觉表征 / Visual Content Understanding & Correspondence | 2D 相对位置跨块自注意力 + 语义/角色正交子空间解耦 | **预实验大幅过关**（SOCO 修正口径：SOC `44.41% → 50.75%` [+6.34pp]、CC `65.79% → 73.23%` [+7.43pp]）；正在补 Codex 红队审计要求的 2×2 归因消融。 |
| [av-nullslot-pcar/](av-nullslot-pcar/) | 音视频多模态 / Multimodal & Audio-Visual Understanding | Null-Slot Cross-Attention + PCAR 成对反事实注意力 Logit 排序 | **机制验证过关**（AVE Round 2：`α-AUROC = 0.931`、有效融合选择比 `3.05×`，完胜 Sigmoid Gate `0.842`）；下一步验证去 `loss_bg` 解耦门控与 LLP/AVVP。 |

---

## 二、已归档 / 暂缓历史备选（原样保留，不删除、不重编号）

| 目录 | 方向 | 当前状态 |
|---|---|---|
| [web-change/](web-change/) | 网页截图的细微变更分类 | 暂缓（缺独立站点来源与强对照增益） |
| [foreground-fidelity/](foreground-fidelity/) | 换背景后的主体保真评估 | 暂缓（偏应用评测，暂不作为主线） |
| [superscript-ocr/](superscript-ocr/) | 文档 OCR 的上标引用漏读 | 暂缓（自动补插未过门槛） |

---

## 仓库规范

- 仅提交代码、协议、精简 JSON 汇总和 Markdown 报告，严禁提交权重、原始音视频/图像与大缓存文件。
- 布局参考 [OpenMMLab MMPreTrain](https://github.com/open-mmlab/mmpretrain/tree/main/configs) 与 [DVC](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files) 的版本化实验记录方式。
