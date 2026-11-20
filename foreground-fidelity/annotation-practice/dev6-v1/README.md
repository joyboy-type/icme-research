# 6 张开发原图：AI 辅助主体定位诊断

这是已有开发集的练习材料，不是独立确认实验。图像按长宽比覆盖选取，未按方法得分或偏好选择；P001–P006 只是匿名编号。这里不提供历史模型框、A/B 图像、偏好标签或方法结果。

## 使用

在仓库根目录执行 `python3 foreground-fidelity/annotation-practice/dev6-v1/fetch_images.py`。只需 Python 3 标准库，无需模型、GPU、训练或付费服务。脚本从公开数据集获取 6 张原图，要求固定版本及逐图 SHA-256 相符；不保存元数据或下载编辑图。失败时停止，不替换图片。

图片获取后，可以直接查看 `images/P001.jpg` 至 `P006.jpg`；需要拖框时，用浏览器打开 `index.html`。该页面沿用通用练习填写格式；AI 交付请使用 `AI_RESULT_TEMPLATE.json`，如使用页面导出，需把记录转换到该模板，明确披露 AI 来源。

- `image_index.json`：匿名文件索引、原图尺寸、文件校验值。
- `正式规则与空模板.json`：通用原图主体定位规则；正式模板没有分配任何图片或填写任何答案。
- `AI_RESULT_TEMPLATE.json`：6 条待填写的 AI 诊断记录。
- `PROMPT_zh.txt`：可直接交给智能体的任务提示词。
- `validate_result.py`：只检查结果结构与坐标合法性，不判断标注正确性。

原图按项目现有归档方式从数据源获取，不提交到 Git；忽略 `images/`、`results/`。图像权利及数据集条款仍依原始来源，本仓库不为原图另行授权。

多个智能体先分别完成并保存，再比较分歧；保留原始答案，不为提高一致率强行统一。所有输出均标为 AI 辅助诊断，不能称为独立人工、专家框或真值。预留确认数据不在本包范围内。

原始来源：https://huggingface.co/datasets/GEditBench-v2/VCReward-Bench

固定版本：`4dfccf0bb4a4448aed140e8ff8ccb412e0f9b2d2`。
