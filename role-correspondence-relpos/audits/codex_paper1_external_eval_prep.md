# 候选一外测数据与强基线接入清单

核查日期：2026-11-21。仅查阅本地文件，未修改文件或运行实验。

**结论：SPair-71k 数据已具备；AP-10K 完整对应评测仍缺固定配对清单及完整原图。MARCO 应显式选择 `marco_spair.pth`，避免默认 Hub 权重训练来源不明的问题。**

**一、对称重复部件映射**

以下关键点索引均为 **0-based**；左右沿用标注角色，不按画面横坐标判断。SPair 英文名称来自本地第三方[名称表](/Users/joyboy/Documents/ChatGPT/ICME/work/paper1_external_audit_20261121/sources/spair_keypoint_names.csv)，不是官方 JSON 中的文字标签；几何边界与独立物理部件应分开统计。

SPair 表内 `R/L=a/b` 表示 right 对应 a、left 对应 b。

| SPair 类别 | 左右部件名称与索引 | 前后重复部件 |
|---|---|---|
| aeroplane | wheel 4/5；engine-fan 6/7；wing-tip 8/9；engine 10/11；wing-front 12/13；wing-back 14/15；stabilizer-tip 16/17；stabilizer-front 18/19；stabilizer-back 20/21 | 同侧 wing-front/back：12/14、13/15；stabilizer-front/back：18/20、19/21 |
| bicycle | handlebar 2/3；back seat 6/7；pedal 9/10 | front/back wheel=0/1 |
| bird | wing 1/2；cheek 4/5；eye 7/8；foot 10/11；knee 12/13；hip 14/15 | — |
| boat | front/mid/back deck：1/2、3/4、5/6；front/mid/back hull：7/8、9/10、11/12 | 同侧 deck：R=1/3/5、L=2/4/6；hull：R=7/9/11、L=8/10/12 |
| bottle | cap 0/1；neck 2/3；top/center/bottom body：4/5、6/7、8/9 | 轮廓位置，单列几何诊断 |
| bus | mirror 0/1；headlight 2/3；tail-light 5/6；back wheel 11/21；front wheel 14/24；对应 back-liner：10/20、13/23；front-liner：12/22、15/25 | front/back license-plate=4/7；车头底/顶角 R/L=16/17、18/19；车尾=26/27、28/29 |
| car | mirror 0/1；headlight 2/3；tail-light 6/7；wheel、liner 索引同 bus | front/back license-plate=4/8；emblem=5/9；前挡风玻璃底/顶角=16/17、18/19；后挡风玻璃=26/27、28/29 |
| cat | ear 0/1；ear-tip 2/3；eye 4/5；nostril 6/7；front foot 9/10；back foot 11/12 | 四足：front-R/front-L/back-R/back-L=9/10/11/12 |
| chair | front-seat 0/1；bottom back 2/3；front leg 4/5；back leg 6/7；top back 8/9；armrest 10/11；elbowrest 12/13 | 四腿：front-R/front-L/back-R/back-L=4/5/6/7 |
| cow、sheep | ear 0/1；ear-tip 2/3；eye 4/5；nostril 6/7；front foot 9/10；back foot 11/12；front knee 15/16；back knee 17/18；horn 19/20 | 四足=9/10/11/12；四膝=15/16/17/18，顺序均为前右、前左、后右、后左 |
| dog | ear 0/1；ear-tip 2/3；eye 4/5；front foot 9/10；back foot 11/12 | 四足=9/10/11/12；6 是单点 nose |
| horse | ear 0/1；ear-tip 2/3；eye 4/5；nostril 6/7；front foot **10/11**；back foot **12/13**；front knee 16/17；back knee 18/19 | 四足=10/11/12/13；四膝=16/17/18/19 |
| motorbike | mirror 0/1；handlebar 2/3 | front/back wheel=6/7；wheelhub=11/12；light=4/5 |
| person | eye 0/1；ear 2/3；shoulder 8/9；elbow 10/11；hand 12/13；knee 14/15；ankle 16/17；foot 18/19 | — |
| pottedplant | rim 0/2；body 4/5；base 6/8 | back/front rim=1/3；属于盆体边界 |
| train | **L/R**：front 顶角0/1、底角2/3；back 顶角4/5、底角6/7；windshield 顶8/9、底10/11；windshield-center 顶12/13、底14/15；headlight 16/17 | front/back 同角位置=0/4、1/5、2/6、3/7 |
| tvmonitor | frame 顶角0/2、底角6/4、侧边7/3；screen 顶角8/10、底角14/12、侧边15/11 | 边界型；frame 与 screen 分组 |

注意：bicycle 的 11、12 是 `N/A` 占位，chainring 保留索引 13，不能压紧编号。类别有该点定义，不代表每个 pair 都有可评分标注。

AP-10K **54 类共享相同的 17 点定义**。以下按官方 `category_id` 顺序列出全部类别；裁剪版已经将它们合并为单一 `animal`，不能直接从裁剪版恢复物种。

| category_id | 类别，按 ID 顺序 |
|---|---|
| 1–10 | antelope、argali sheep、bison、buffalo、cow、sheep、arctic fox、dog、fox、wolf |
| 11–20 | beaver、alouatta、monkey、noisy night monkey、spider monkey、uakari、deer、moose、hamster、elephant |
| 21–30 | horse、zebra、bobcat、cat、cheetah、jaguar、king cheetah、leopard、lion、panther |
| 31–40 | snow leopard、tiger、giraffe、hippo、chimpanzee、gorilla、orangutan、rabbit、skunk、mouse |
| 41–50 | rat、otter、weasel、raccoon、rhino、marmot、squirrel、pig、mole、black bear |
| 51–54 | brown bear、panda、polar bear、bat |

| AP-10K 重复部件 | left 名称：索引 | right 名称：索引 |
|---|---|---|
| eye | left_eye：0 | right_eye：1 |
| shoulder | left_shoulder：5 | right_shoulder：8 |
| elbow | left_elbow：6 | right_elbow：9 |
| front paw | left_front_paw：7 | right_front_paw：10 |
| hip | left_hip：11 | right_hip：14 |
| knee | left_knee：12 | right_knee：15 |
| back paw | left_back_paw：13 | right_back_paw：16 |

同侧前后 paw：左 `(7,13)`、右 `(10,16)`；四 paw 顺序 `[left-front,right-front,left-back,right-back]=[7,10,13,16]`。单点为 nose=2、neck=3、root_of_tail=4。肩/髋、肘/膝不宜直接视为同一语义部件。

**二、划分、格式与读取接口**

| 数据/协议 | val | test | 本地核查结果 |
|---|---:|---:|---|
| SPair `Layout/large` | **5,384 对** | **12,234 对** | 已流式清点压缩包；Layout 行数与 PairAnnotation JSON 数一致 |
| AP-10K 官方 split1 | 995 图／1,272 实例 | 1,997 图／2,634 实例 | 官方姿态数据，**没有预定义图像对** |
| AP-10K 官方 split2 | 995 图／1,342 实例 | 1,997 图／2,594 实例 | 同上 |
| AP-10K 官方 split3 | 995 图／1,260 实例 | 1,997 图／2,647 实例 | 同上 |
| GeoAware/MARCO 对应协议 | 论文约 17k 对 | 论文约 36k 对 | **本地无固定 pair manifest，精确数待确认**；不能将约数填写为实测数量 |
| 当前 AP 裁剪预实验 | 未下载 val 标注 | test 标注含2,634个裁剪实例；实际24张图组成12对 | 双向共96个 paw 查询；不是完整跨物种评测 |

AP 三套官方划分是重复划分方案，不是三套互斥数据。MARCO 预处理合并官方 split1 的 train/val/test 后重新划分，生成 intra-species、cross-species、cross-family pairs；目录遍历与随机配对未固定，不能仅凭脚本还原作者原清单。当前 AP loader 只支持 `trn/test`，评估脚本固定读取 `test`。

| 项目 | SPair-71k | AP-10K |
|---|---|---|
| 文件 | tar.gz 内含 JPEG、分割 PNG、逐图/逐对 JSON、Layout TXT | 官方 COCO JSON＋原图；当前预实验为512×512 JPEG＋COCO JSON |
| 对应关系 | pair JSON 的 `kps_ids` 对齐 `src_kps/trg_kps` | 固定17槽 `(x,y,v)`；MARCO pair JSON 保存 `src_json_path/trg_json_path` |
| bbox | `[xmin,ymin,xmax,ymax]` | `[x,y,w,h]` |
| PCK 归一化 | 目标框长边；官方 loader 将图直接缩放为 S×S，`L=max((xmax−xmin)S/W,(ymax−ymin)S/H)` | 保持比例缩放最长边至 S，右/下补零；`L=max(w,h)S/max(W,H)` |
| 读取接口 | `SPairDataset`；或 `tarfile + json + PIL` 只读读取 | 官方标注用 `json`/COCO 接口；配对数据用 `AP10kDataset/load_ap10k_data` |
| 有效点与汇总 | pair 已提供公共对应点；逐 pair 求 PCK 后平均 | 作者 loader 取两图 `v>0` 交集，包含遮挡点；逐 pair 求 PCK 后平均 |

现有24图诊断使用裁剪坐标、目标框 `max(w,h)`，并筛选四 paw `v=2`；它与完整作者协议应分别命名。具体依据见本地[数据读取代码](/Users/joyboy/Documents/ChatGPT/ICME/work/paper1_external_audit_20261121/sources/marco/datasets)及[官方 AP 标注](/Users/joyboy/Documents/ChatGPT/ICME/work/paper1_external_audit_20261121/sources/ap10k_official)。

**三、MARCO 官方权重**

依据本地[官方 README 快照](/Users/joyboy/Documents/ChatGPT/ICME/work/paper1_external_audit_20261121/sources/marco/README.md)、代码和[原文文本](/Users/joyboy/Documents/ChatGPT/ICME/work/soco_role_pilot_20261121/sources/marco.txt)：

| 权重 | 作者声明训练来源 | AP-10K 跨数据集外测 | 本地已记录下载入口 |
|---|---|---|---|
| **marco_spair.pth** | SPair-71k；未进行 AP-10K 任务训练 | **选择此项** | [官方 Drive](https://drive.google.com/file/d/1_of8iQjenTttF5Jld69LNf9M0vnM2Xbx/view) |
| marco_ap_spair.pth | AP-10K 预训练，再 SPair-71k 微调 | 不用于“未在 AP-10K 训练”比较 | [官方 Drive](https://drive.google.com/file/d/1d2If4nBUy7SjFberZVuXsFr_zqLXvoU8/view) |
| marco_release.pth | release 记录未注明对应哪种训练来源 | **待联网确认来源** | [官方 Release](https://github.com/visinf/MARCO/releases/download/v1.0/marco_release.pth) |

- **骨干**：DINOv2 ViT-L/14；代码具体为 `dinov2_vitl14_reg`，含 registers；完整模型约323M参数。
- **输入分辨率**：SPair/AP YAML 默认 **770×770**；README 示例及 Hub helper 默认 **840×840**；论文附录速度比较明确使用840，不能据此断言所有主表统一840。
- **权重状态**：本地有下载链接及 ZIP 元数据探测记录，未见完整权重加载验证。Drive 文件约71.6MB，保存 adapter/upscale 参数，冻结骨干另行获取；旧记录将323M全模型大小当作 checkpoint 大小，不宜沿用。
- **Hub 注意**：`pretrained=True` 默认下载 `marco_release.pth`，不能默认等同 SPair-only。
- **配置差异**：正文写最后12层适配器；本地配置及权重元数据对应零基15–23共9层，应按发布权重接入。

**四、直接接入建议（10行）**

1. SOCO 过关后冻结模型；SPair/AP 外测标签仅用于评分，不用于调参。
2. SPair 按 `Layout/large` 固定 val/test；按 `kps_ids` 保留原始编号。
3. AP 优先取得作者三种子协议固定 manifest；取得前不宣称复现其完整评测。
4. 若自行配对，明确另命协议，冻结图像、实例、方向、随机种子及清单散列，所有基线重评。
5. 补齐 AP 原图与完整 MARCO 仓库；当前代码快照和24张裁剪图不足以直接跑全量。
6. 显式下载 `marco_spair.pth`，另取 `dinov2_vitl14_reg`；核对 adapter/upscale 参数完整加载。
7. 显式指定 S；两数据集各自沿用上述预处理与 bbox 口径，同一数据集内各方法保持一致。
8. 接口统一为两图 `(B,2,3,S,S)`＋源点 `(B,N,2)`，输出目标点；目标真值不进入模型。
9. 同报逐 pair 平均 PCK@0.01/0.05/0.10、角色混淆及有效分母；四足 oracle 诊断单列。
10. 先核对一个固定 pair 的坐标与输出，再跑全量；登记已查看的24图，并另核对 SOCO 与外测原图重合。