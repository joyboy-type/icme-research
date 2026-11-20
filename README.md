# ICME Research

本仓库按三个研究方向留存代码快照和可追溯的轻量实验记录。

- [网页截图的细微变更分类](web-change/)
- [换背景后的主体保真评估](foreground-fidelity/)
- [文档 OCR 的上标引用漏读](superscript-ocr/)

每轮用稳定编号把代码、运行信息、指标和选定的逐样本记录放在一起。运行结果追加到新编号。研究方案有实质变更时才更新任务书。完整输入快照、模型权重、缓存、原始日志和任务书由 Mac 项目归档保存。当前仓库是轻量副本，独立复跑需要对应输入和模型。

布局参考 [OpenMMLab MMPreTrain](https://github.com/open-mmlab/mmpretrain/tree/main/configs) 的多任务配置目录以及 [DVC](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files) 的版本化输入与产物记录。
