# task_plan：task3 局部交互读出 2×2 诊断

## 1. 任务标识与状态

- 任务 ID：`SRPPI-T3-LOCAL-001`
- 版本：`v1.0`
- 制定日期：2026-09-29
- 状态：用户已确认按此方向生成任务单；执行 agent 可在本文件范围内实现、训练和验证。完成回执后停止。
- 研究对象：task3 蛋白对 PPI 二分类，`task3-structure-v2` cluster-disjoint developmental validation。
- 本地项目：`/home/zy/projects/Surface-residue`
- 集群项目：`/datapool/home/zhangye/projects/surface-residue-ppi`
- 本任务单：`plan/task_plan_SRPPI-T3-LOCAL-001_v1.0.md`
- 回执：`review_back/review_back_SRPPI-T3-LOCAL-001_v1.0.md`
- 附件目录：`review_back/SRPPI-T3-LOCAL-001_v1.0/`

本任务检验 mean pooling 是否限制局部信息用于 PPI 判别；不是完整 Pairformer、正式测试或新 Phase 的全面扩展。执行完成后等待决策 agent 审阅，不得自动衔接其他实验。

## 2. 依据的交接包、代码和数据版本

执行前须在集群实际使用文件上重新计算 SHA256。哈希变化时先停止并回报，不能静默混用。

| 依据 | SHA256 | 用途 |
|---|---|---|
| `review_back/review_back_SRPPI-EVID-002_v1.0.md` | `8ccad69ebc78ced60791b1ea34e9e977e6068917dac464365204cfebfb5e9cf2` | task3 样本流转核验 |
| `review_back/review_back_SRPPI-EVID-002_v1.1.md` | `09027666c19c0f11d175edc8912d17f905806ea8f387bd3b4eb864ac7c21ec3b` | task2 结果限制；本任务不重做 task2 |
| `review_back/review_back_SRPPI-T2-SYM-001_v1.1.md` | 执行前记录本地及集群实际文件 SHA256 | task2 对称 head 背景；不是 task3 输入 |
| `plan/task_plan_SRPPI-EVID-002_v1.0.md` | `a471cae4fe7c9fa932bb2a49bdd9a3a348862165a197c5f2c09a467f9d646567` | task3 既有协议依据 |
| `runs/phase3_task3_protocol/20260928/structure_v2/report.json` | `ffaadc6e790accf4b119676edc8bb9a22803caae4dbaad59fe5c4fbd6c136593` | 冻结协议及训练配置 |
| `runs/phase3_task3_protocol/20260928/structure_v2/train.json` | `8b8c1c3aacb28dacb5f5c36607754c252579a33d65f3eddfc1f820b8892639cc` | 固定训练 pair 顺序 |
| `runs/phase3_task3_protocol/20260928/structure_v2/val.json` | `47b7e2805f2e8313b58c440745160130c8a1e9355839cf72fc099772b743007e` | 固定验证 pair 顺序及标签 |
| `runs/phase3_task3_protocol/20260928/structure_v2/proteins.json` | `b623c551a93a089b62c3d6df2d7255736f6f7368690a1a7c23ecc8b7d3892e5e` | 蛋白、序列和结构来源映射 |
| `runs/phase3_task3_summary/20260928/structure_v2/report.json` | `5ed9b3046f696696fb399c056fa1a65f5395e2419d16f47b448590bafcfad730` | 已完成 mean-pooling 结果，仅作历史背景 |
| `scripts/train_phase3_task3_structures.py` | `98d3c1b6106b3ef97398ccf0b95b7a31048c7643639f4ab040396ddd054b9fca` | 既有 Readout、训练和关系头的实现依据；不得覆盖 |
| `reports/phase3_task3_inputs_verified_20260928.json` | `2da919953b8b8e26111a900bb42a2c3738db5d82bd4c9f254f434d399b2b176d` | 已核验 task3 输入记录 |

实际缓存、特征汇总、normalization 及每个蛋白特征的哈希不在此逐个枚举；执行端须从既有报告/缓存元数据读取并写入新配置。任何 cache、mask、normalization 或 backbone 来源变化均停止。

## 3. 当前问题与本次目标

structure-v2 的既有 mean-pooling 三 seed 比较中，ES-B 相对 ES 的 AP 均值差为 `-0.003442`，109 个共享连通组件的 95% paired bootstrap 区间为 `[-0.013714, +0.003377]`；没有稳定表面增益。验证集只有 194 个蛋白对，因此区间宽度和有效独立组件数是解释限制。

本次只检验：**相较逐蛋白 mean pooling，固定的局部残基对读出是否改变 ES 的 PPI 判别，以及是否使 ES-B 相对 ES 出现不同的表面效应。**

可能结果包括局部读出改善 ES、改善 ES-B、两者都无改善、表面差异不变或仍不确定。性能提升不是完成条件。

## 4. 固定 2×2 设计及公平比较

本任务从头重训完整四格，避免把历史模型选择、初始化或产物差异混入主要 2×2 对照。

| 表征 | Mean readout | Local-pair readout |
|---|---|---|
| ES（ESM2 + GearNet，无表面 adapter） | `ES-MEAN` | `ES-LOCAL` |
| ES-B（ES + 原 B adapter） | `ES-B-MEAN` | `ES-B-LOCAL` |

- 三个固定 seed：`20260927`、`20260928`、`20260929`；共 12 个新训练组合。
- 四格使用同一 backbone、对应表征定义、输入行顺序、训练/验证数据、归一化规则、关系 MLP、损失、优化器、训练上限、checkpoint 选择规则和 seed。
- 唯一设计因素为残基表征如何转成蛋白对表示：先逐蛋白 mean 后构造对称 pair feature，或先对完整残基对集合构造 pair feature 再做固定局部汇聚。
- 既有 ES/ES-B mean 模型和预测只作为历史背景，不与本轮 local 结果混作主配对比较；本轮新训的 mean 格用于主要 2×2 推断。
- ES-C、E 和 task2 separable baseline 不纳入本任务。

预先定义的主要估计量（对 AP 与 AUROC 分别报告）：

1. 读出效应：`ES-LOCAL − ES-MEAN`；`ES-B-LOCAL − ES-B-MEAN`。
2. 表面效应：`ES-B-MEAN − ES-MEAN`；`ES-B-LOCAL − ES-LOCAL`。
3. difference-in-differences：`(ES-B-LOCAL − ES-LOCAL) − (ES-B-MEAN − ES-MEAN)`。

## 5. 固定模型定义

### 5.1 输入与 mask

- 使用既有 task3 Readout 的残基表征 `h_i ∈ R^256`。ES 使用 ESM2+GearNet 投影；ES-B 使用同一表征加原 B adapter。
- ESM2、GearNet、dMaSIF backbone/checkpoint 冻结；仅训练原 task3 中相应的 projection/adapter 和本任务关系 MLP。
- 每条链只使用既有 `common_valid` mask。PPI 标签仅进入损失和评估；partner 坐标、残基接触标签、真实界面位置不进入 encoder，也不参与候选筛选。
- 两侧输入在数学上无角色差别；蛋白交换后输出应一致。

### 5.2 Mean readout

对每个蛋白分别在 common-valid 残基上取算术均值，得到 `u_A,u_B ∈ R^256`。构造对称特征：

`phi_mean = concat(u_A + u_B, abs(u_A - u_B), u_A * u_B) ∈ R^768`。

### 5.3 Local-pair readout

候选集合固定为两侧 common-valid 残基的完整笛卡尔积，不得按距离、界面标签、PPI 标签、预测分数或其他信息裁剪：

`C(A,B) = valid(A) × valid(B)`，`N = |valid(A)| × |valid(B)|`。

对每个候选残基对构造与 mean 头同维度、同形式的对称特征：

`phi_ij = concat(h_i + h_j, abs(h_i - h_j), h_i * h_j) ∈ R^768`。

用唯一固定的逐通道、长度归一化 log-mean-exp 汇聚：

`z_k = tau * log((1/N) * sum_(i,j in C) exp(phi_ij,k / tau))`，所有通道 `k=1..768` 固定 `tau=1.0`。

使用数值稳定的 logsumexp。可以按固定顺序分块计算完整集合，但必须数学等价；不得采样、top-k、近邻筛选或移除 `1/N`。候选矩阵只用于生成上述汇总向量，不保存或解释为真实接触图。

### 5.4 相同关系 MLP

Mean 和 Local 两种读出均进入同一个结构、相同维数和相同训练协议的关系头：

`Linear(768,128) → GELU → Dropout(0.1) → Linear(128,1)`。

四格该关系头参数量完全相同；不额外加入长度、候选数、暴露度、额外池化、阈值或校准模型。这样模型容量和分类头保持固定，主要读出差异是构造 768 维蛋白对表示时聚合发生在逐蛋白 mean 之前还是之后。

## 6. 固定数据、训练、指标和统计协议

### 6.1 数据

- 冻结协议：`task3-structure-v2`，1,112 train（690 正/422 负）、194 validation（124 正/70 负）。
- 固定 1,518 个训练蛋白、325 个验证蛋白；训练与验证蛋白及 MMseqs 40% identity/80% coverage 同源簇交集均为 0。
- 固定 row order、标签、protein IDs、结构 cache、AlphaFold 单体输入和 common masks；复用训练蛋白归一化规则。
- 不下载新数据，不改划分，不读测试集。旧 D-SCRIPT test 已有历史访问，必须在报告限制中持续注明。

### 6.2 训练

沿用 structure-v2 协议 report 的设置：30 epoch 上限，patience 5，`min_delta=1e-4`，batch size 24，AdamW learning rate `1e-3`，weight decay `1e-4`，hidden dimension 256，原 BCEWithLogitsLoss 正例权重、gradient clipping 和 validation AP checkpoint 选择规则。

四格每个 seed 均从头初始化，不 warm-start mean 或旧模型 checkpoint。每个 seed 下数据 row shuffle 应按既有 seed 规则复现；variant 的随机状态和调用顺序须一致记录，不能中途根据结果更改训练顺序或挑选 seed。任何训练配置与上述协议不符时先停止并回报。

### 6.3 主要指标和配对区间

- 逐 seed 报告 validation AP、AUROC、固定样本正负计数和逐蛋白对分数。
- 复用既有 task3 summary 的 paired connected-component percentile bootstrap：109 个共享同源簇连通组件、2,000 次重采样、三个固定 seed 上同一抽样的配对比较。报告 AP/AUROC 的区间与逐 seed 差值；不得改变组件、次数或聚合方式。
- 返回第 4 节列出的读出效应、两种读出下的表面效应及 difference-in-differences；区间跨 0 或较宽时结论为不确定。
- 验证每个 local checkpoint 的交换不变性：`logit(A,B)` 与 `logit(B,A)` 以及在该实现中等价的同一 pair feature 应在预设容差 `atol=1e-6, rtol=1e-5` 下相等；容差运行前固定。
- 保存 194 个验证 pair 的 `source_row`、protein IDs、label、两侧 common-valid residue 数、候选数 `N`、mean/local 表示摘要、logit 与预测。

所有结果仍是 developmental validation；不把 AP 与 task1/task2 横向比较。

## 7. 明确执行范围

1. 执行前核对本任务单、`AGENTS.md`、structure-v2 report、train/val/proteins manifest、特征和训练 normalization 的哈希；记录执行端集群项目版本。哈希不符则停止。
2. 检查完整候选集合的上界与分块实现；只做资源估算和不读测试数据的静态核验。所有模型 smoke/preflight 通过 Slurm。
3. 在项目自有代码新增独立训练入口、配置、局部汇聚实现和报告，不覆盖历史 task3 训练脚本或产物。
4. 通过 Slurm GPU preflight 检查：common mask、完整笛卡尔积计数、分块与非分块 toy tensor 数值一致、对称性、有限值、一个固定小 batch 的前向/反向和显存使用。preflight 不用于选超参数。
5. preflight 通过且集群资源合法后，按固定三 seed 训练 4 格，共 12 组合；不得追加 seed、改候选或扫描配置。
6. 对所有 checkpoint strict load；复算验证指标、交换不变性和 109-component 配对区间。
7. 写回执、配置、checkpoint inventory、逐样本结果、统计摘要、Slurm 记录和 evidence index；结束任务并等待审阅。

## 8. 本次不做什么

- 不训练 E、ES-C、task2 separable baseline 或其他模型；不把局部残基对输出解释为真实接触概率或化学互补。
- 不搜索 tau、特征 rank、候选阈值、top-k、近邻规则、多个汇聚方式或超参数；不增加 seed。
- 不按真实界面、partner 坐标、接触标签、PPI 标签或中途预测分数裁剪/采样候选。
- 不加 cross-attention、Pairformer、patch-pair、多轮推理、APBS、电荷输入、额外长度特征；不扩大 ESM2 或解冻任何 backbone。
- 不访问 D-SCRIPT test、DIPS test 或新的外部测试结构、标签、预测和性能；不以旧 D-SCRIPT test 指导本任务。
- 不迁移 task2 adapter，不重做 task2 交换检查或 separable baseline，不正式宣称 Phase 4 全面验收。
- 不覆盖旧结果/日志/checkpoint/cache，不安装环境，不修改共享或只读路径。

## 9. 允许修改范围

允许在项目自有代码和新运行目录中新增本任务的 local/mean 对照入口、固定配置、toy 数值测试、Slurm 脚本、汇总脚本、训练日志及报告。实现等价的显存分块允许，但需报告分块策略、源代码 hash 和与未分块小矩阵的数值核对。所有修改记录 diff；不修改原 protocol、旧模型脚本、旧预测、原始数据和 feature cache。

## 10. 资源预算及执行约束

- 用户已确认研究范围，但**数值化 GPU 小时、GPU 分区、CPU、内存、时限和并发预算尚未提供**；不得把旧作业资源当成本次已批准预算，也不得把估计写作实测。
- 执行端提交前通过当前集群 `sinfo`/`scontrol` 核实 GPU 分区和字段，选择能完整执行 12 个组合的最小合法配置；在运行记录中区分申请和实际资源。若本项目工作流要求用户批准数值预算，则先返回资源需求与 preflight 估计并停止训练，等待确认。
- 所有模型训练、推理、feature/cache 加载型模型检查、smoke test 和重型预处理必须通过 Slurm；CPU 工作用 `hpc`，GPU 工作用 GPU 分区。不能在登录节点运行模型。
- 不取消无关作业，不使用 `scancel -u`；仅凭本任务精确 job ID 管理本任务作业。不删除旧产物。

## 11. 验收标准

科学上无增益或不确定也可正确完成，必须满足：

1. 12 个固定组合的状态、配置/输入/代码 hash、checkpoint SHA256、strict load 和 Slurm provenance 完整；失败组合单列。
2. 四格的唯一设计差异与本任务定义一致；mean/local 使用相同维度的 768 特征和完全相同的关系 MLP，训练协议与 seed 固定。
3. Local 使用完整 common-valid 笛卡尔积、逐通道 `tau=1.0` 长度归一化 log-mean-exp；toy 全量与分块结果数值一致；交换检验通过固定容差，或任务标为实现失败并停止性能解释。
4. 数据划分、row order、labels、common masks、normalization 和 structure-v2 输入不变；`test_records_read=0`、`test_labels_read=0`。
5. 返回四个单元的逐 seed 和逐样本 AP/AUROC、读出效应、两种表面效应、difference-in-differences 及预定义组件区间。
6. 不要求 local 高于 mean、不要求 ES-B 高于 ES、不要求区间排除 0。性能下降、无效应或结果不确定都可验收。

预先解释规则：

- 若 ES-local 改善，而 ES-B-local 相对 ES-local 没有稳定优势，说明局部读出可能帮了 PPI 判别，但表面额外价值尚未建立。
- 若 mean 下 ES-B−ES 无优势，而 local 下 ES-B−ES 稳定转正且 difference-in-differences 支持变化，才支持“mean pooling 限制表面信息使用”这一解释。
- 若两种读出下 ES-B 均无稳定优势，收窄表面增益对 PPI 的结论，优先回看任务差异、结构筛选和样本量，不继续叠加局部架构。
- 若区间宽或包含多种方向，结论保持不确定；不得靠增加 seed、换 bootstrap 或改协议制造确定性。

## 12. 必须返回的证据和附件

主回执至少包含：

- 本任务 ID/version、依据文件及实际路径/hash、固定设计和实际偏差；
- 四格每个 seed 的 Slurm job ID、分区、节点、GPU 型号、CPU、内存、起止时间、exit code、依赖版本；
- 12 个 checkpoint 的路径、大小、SHA256、strict load 状态和 best epoch；
- 实际模型公式、参数量、局部候选计数/分块策略、toy 数值与交换检查；
- 训练/验证 manifest、feature cache、common mask、normalization、backbone checkpoint、代码与配置的 hash；
- 每 seed/单元 AP、AUROC、mean±sample SD；全部配对差值与 109-component bootstrap 区间；
- 逐验证 pair 预测、labels、蛋白对身份、有效残基数、候选数、预测分数，以及可重算说明和 SHA256；
- `test_records_read`、`test_labels_read`、失败/未完成项、未知资源状态和测试暴露限制。

建议附件：`run_config.json`、`checkpoint_inventory.tsv`、`metrics_by_seed.tsv`、逐 pair predictions NPZ/TSV、`candidate_counts.tsv`、`symmetry_summary.json`、bootstrap 明细、每组合 history、source diff、Slurm 记录和 `evidence_index.tsv`。

## 13. 停止条件

以下情形停止受影响步骤并回报，不能自行改变方案：

- protocol、manifest、row order、feature cache、normalization 或 backbone 来源 hash 与依据版本不符；
- 无法保持四格 common mask、样本顺序、标签和训练/验证条件一致；
- 完整 valid×valid 候选集无法在合法 GPU 资源内运行，需改采样/top-k/阈值或池化定义；
- 当前集群规则要求额外资源批准但数值预算未确认；
- 分块实现与完整矩阵 toy 结果不一致、出现 NaN/Inf 或交换容差失败；
- 需要读测试数据、改划分、标签、模型冻结范围、增加 seed 或改指标/bootstrap；
- 需要修改共享环境、GearNet/Surface-GearNet 或其他只读路径；
- 中途结果引出增加 ES-C、更多 pooling、超参数搜索、正式测试或其他新实验。

## 14. 当前决策摘要

- 当前目标：用固定 task3 structure-v2 的 ES/ES-B × mean/local 2×2 检验局部读出能否改变 PPI 判别及表面效应。
- 已确认结论：当前 mean-pooling structure-v2 validation 未显示稳定 ES-B 对 ES 增益；task2 对称 head 下的局部接触增益不能直接推导 PPI 增益。
- 本次已获用户批准的范围：四格从头训练、三 seed、固定完整局部候选与单一 log-mean-exp 读出、developmental validation。
- 未批准：task2 separable baseline、ES-C local、候选扫描、额外 seed、正式测试、迁移训练和更复杂交互模型。
- 有效任务 ID：`SRPPI-T3-LOCAL-001 v1.0`。
