# task_plan：task2 显式交换不变 pair head 重训（动态采样协议修订版）

## 1. 任务标识与授权

- 任务 ID：`SRPPI-T2-SYM-001`
- 版本：`v1.1`
- 制定日期：2026-09-28
- 状态：用户已确认允许原 task2 的确定性动态采样；执行 agent 可在本文件范围内实现、训练和验证。
- 研究对象：task2 残基接触预测；结果仍属于 sampled-negative developmental validation。
- 本地项目：`/home/zy/projects/Surface-residue`
- 集群项目：`/datapool/home/zhangye/projects/surface-residue-ppi`
- 本任务单：`plan/task_plan_SRPPI-T2-SYM-001_v1.1.md`
- 回执位置：`review_back/review_back_SRPPI-T2-SYM-001_v1.1.md`；附件目录为 `review_back/SRPPI-T2-SYM-001_v1.1/`

本版本修订 v1.0 的训练输入条件：原 task2 没有保存可复用的固定训练 pair 流，训练负例由原脚本运行时动态生成。允许按原 sampler 确定性重建训练流，但不得声称逐 pair 复现历史 ordered 训练。现有 ordered head 结果仍只作历史对照，不覆盖旧 checkpoint、旧预测、旧报告或旧运行目录。完成回执后停止，等待下一轮决策；不得自动衔接 separable baseline、任务三局部交互读出或正式测试。

## 2. 依据的交接包、代码和数据版本

本任务以以下已完成证据为依据。执行 agent 必须在训练前再次记录实际使用文件的 SHA256；若哈希变化，先停止并回报，不静默混用版本。

| 依据 | 版本或 SHA256 |
|---|---|
| `review_back/review_back_SRPPI-EVID-002_v1.1.md` | `09027666c19c0f11d175edc8912d17f905806ea8f387bd3b4eb864ac7c21ec3b` |
| `review_back/SRPPI-EVID-002_v1.1/task2_swap_summary.tsv` | `340319221a6831390de9ce2edf070136cd8ba6df0b669bb019d91e92eeee82ef` |
| `review_back/SRPPI-EVID-002_v1.1/results/run_config.json` | `53840aba09394c6c5fc0fc0fcab4fc04a693814fd25a1433db0df3986c004d16` |
| `review_back/SRPPI-EVID-002_v1.1/checkpoint_inventory.json` | `eb0a6ac8f73acd63d931f178ce094485e32048f0a89a512c4997eeb60996a4e6` |
| `review_back/SRPPI-EVID-002_v1.1/canonical_reproduction_check.json` | `6b672005552b59b301822279fb3c87e8352862e85f2ee91ebd8294fba41c885b` |
| `plan/task_plan_SRPPI-T2-SYM-001_v1.0.md` | `b0d117b56c64a69f777925ae7a2749fd4b81f7eac834ffa95f96a21b0ce64028` |
| `review_back/review_back_SRPPI-T2-SYM-001_v1.0.md` | `f033bad98a6e682ccd82481e3fe050719aca30a66eaeb44f75f4311f0d0745ee` |
| `AGENTS.md` | `03f307554c52c0d8cadec0f13b8ae8eccdd1704598acd2c08314224a6ec0341b` |
| task2 protocol report | `d10e45df9750e8b730d0f300c1ea9dd13f1f9d8d2b7a03c7f302fe8eb01874c2` |
| task2 feature report | `f3d2c81aaaec607a5911548a3831bc6ef42192740a9ca23d51e3e4e256f9c081` |
| validation manifest（含 strict manifest） | `476416f9e286f691011bd075a4132aefcdaabcce9f4bfc44aa414bd77ec73b27` |
| normalization | `fb7a3476a1cfeb9e0be0493551c20373629f2e1b95f65e093e512256034c2d68` |
| task2 loader source | `f0214c8bbd1beeb39c9f41d879f38feee7e66f8f9ecd471e748b8f4e5a13bab7` |
| task2 training/model definitions | `685f688b83621f0b0d03ad644c29c17ff2481adb1abc4636a33321d9c3e8bb15` |
| adapter source | `b5d43fa7f4d36cdc75c8b850c20f8d735d44a605aed2d8982258141a90d0775f` |

固定背景事实：`task2-common-mask-v2` 有 467 个 validation complexes；每个历史 seed/variant 使用 669,235 个 validation pairs，其中 60,857 个正例、608,378 个负例；三个 seed 为 `20260927`、`20260928`、`20260929`。现有 9 个 checkpoint 已在 EVID-002 v1.1 中成功定位和加载，但其 ordered pair head 对 AB/BA 交换明显敏感。训练前核查确认：原 task2 没有保存固定 train pair/label/offset/`lmdb_indices` 流，`scripts/train_phase3_task2.py` 通过 `sample_pairs(record, seed, epoch, ratio=3)` 动态采样；本版本将此作为已批准的协议修订。

## 3. 当前问题、本次目标与待检验假设

当前 pair head 使用有序的左右分支。对同一物理残基对，原模型的 `S_AB(i,j)` 与交换输入后转置的 `S_BA(j,i)` 在 9 个历史模型中均明显不同，平均绝对差为 `0.461–0.774`，因此当前 task2 不能解释为交换不变的蛋白对接触预测器。

本次目标是只替换 pair head 的对称性约束，重训 ES、ES-B、ES-C，检验：

1. 新 head 是否在固定容差内满足 `S_AB(i,j) ≈ S_BA(j,i)`；
2. 在满足交换不变性后，表面分支 ES-B/ES-C 相对 ES 的开发验证差异是否仍存在；
3. 现有表面增益是否依赖 ordered head 的方向敏感性。

科学上允许的结果包括性能下降、无表面增益、区间很宽或 ES-B/ES-C 仍有增益；完成标准不是提升 AP/AUROC。

## 4. 对称 head 的固定定义

新模型必须使用一个数学上显式交换不变的分数函数。以原 task2 ordered head 的标量 logit 函数记为 `f_θ(x,y)`，新 head 固定为：

```text
g_θ(x, y) = 0.5 * ( f_θ(x, y) + f_θ(y, x) )
```

其中 `f_θ` 使用同一组参数、同一套已有左右投影；`x`、`y` 是已经得到的两侧残基表征。训练损失直接作用于 `g_θ`，不是分别训练两个方向后再挑选输出。推理时用同一对表征计算两个方向并取平均，或使用严格等价的已证明实现。

该定义的目的，是保留原 ordered head 的参数化和大致容量，只加入交换不变约束，避免同时引入注意力、池化或新的交互机制。若实际源代码的 `f_θ` 结构与上述假设不一致，执行 agent 必须先回报实际结构和拟采用的等价对称化公式，未获确认不得自行改成共享投影、MLP、attention 或其他新 head。

编码器、ESM2/GearNet/dMaSIF 冻结状态、A/B/C adapter、输入 mask、归一化和训练可更新模块均保持原 task2 定义。不得新增蛋白角色专属模块。训练时两侧表征应各计算一次，再在 pair head 内计算两个方向；评估时关闭 dropout 并使用确定性推理。

新模型不得从历史 ordered checkpoint warm-start。应按原 task2 的初始化和随机种子协议，从头初始化可训练 adapter 与新对称 head；冻结 backbone 的权重和来源保持不变。若原训练协议确实包含可训练模块之外的额外状态，必须按原配置保留并在回执中列出。

输出命名固定为 `ES-SYM`、`ES-B-SYM`、`ES-C-SYM`。ES 与 ES-B 是主要比较；ES-C 是预先指定的次要版本，不得根据中途结果临时添加或删除。

## 5. 固定不变的数据、模型、指标与比较条件

### 5.1 数据与划分

- 只使用 `task2-common-mask-v2` 既有 train/validation protocol；validation 为 467 个 complexes。
- 验证必须复用原 task2 已保存的 validation pair、label、offset、`lmdb_indices`、common mask、feature cache、normalization 和 manifest，保持 669,235 pairs 的验证顺序。
- 训练负例允许使用原脚本 `sample_pairs(record, seed, epoch, ratio=3)` 动态生成；不得引入新负例算法、不同 ratio、不同 rejection 条件或新的采样数据源。训练 manifest、record 顺序、seed、epoch/step 采样调用顺序和 sampler 源码 hash 必须固定并写入配置。
- 每个 seed 的每个 epoch 先确定一份 sampler 流，再供该 seed 下的 ES-SYM、ES-B-SYM、ES-C-SYM 使用；至少保存每个 epoch 的 pair 数量、正负计数、pair-stream hash 和生成配置。若无法让三个 variant 使用同一流，停止训练。
- 三个 seed 固定为 `20260927`、`20260928`、`20260929`；每个 seed 训练 ES-SYM、ES-B-SYM、ES-C-SYM，共 9 个组合。
- 所有输入、标签、partner 坐标、接触边和 mask 的 hash 在训练前写入配置。标签和 partner 坐标不能进入 encoder 输入。
- 训练开始前，对每个 seed 至少重复生成两次固定 epoch 的 sampler 流；pair-stream hash、顺序、正负计数必须完全一致。该检查只证明本次重建流可重复，不证明它等于历史 ordered 训练时实际使用的逐 pair 流。

### 5.2 模型与训练协议

- ESM2、GearNet、dMaSIF 及其 checkpoint、冻结范围和表面特征缓存不变。
- A/B/C adapter 定义不变；ES-B 与 ES-C 仍使用原先各自的邻域和局部注意力定义。
- batch size、optimizer、学习率、scheduler、epoch/step 上限、loss、class weighting、gradient clipping、normalization、checkpoint 保存和 validation checkpoint 选择规则全部复用原 task2 配置；训练 pair 由第 5.1 节规定的原 sampler 动态生成。
- 唯一模型设计变量是 ordered pair head 替换为第 4 节的 `g_θ`。不得以交换后的指标、测试集或结果趋势选择 checkpoint。
- 不 warm-start ordered head；记录每个组合的初始化 seed、代码 hash、配置 hash 和实际 checkpoint hash。

### 5.3 指标与比较

每个 seed/variant 必须返回：

1. canonical AB 与 swapped BA-transposed 的 complex-macro AP、AUROC；
2. 逐 pair `score_ab`、`score_ba_transposed`、`abs_delta`、`relative_delta`；
3. 绝对差 mean/median/max、q50/q95/q99，以及固定容差通过数和比例；
4. 以同一 validation complexes 为单位的 ES-B-SYM−ES-SYM、ES-C-SYM−ES-SYM 配对差值；
5. 与历史 ordered ES/ES-B/ES-C 的同协议对照，明确标为跨运行历史对照，不混写成同一 checkpoint 的结果。

交换容差沿用并在运行前固定：

```text
abs_delta <= 1e-6 + 1e-5 * max(abs(score_ab), abs(score_ba_transposed))
```

若已有 task2 报告定义了组件 bootstrap，则复用其组件、次数和聚合规则；若无法确认原区间规则，只报告逐 seed、均值和标准差，不临时发明新的显著性协议。所有指标仍标为 developmental validation，不能称为正式测试性能。

## 6. 明确执行范围

1. 在集群项目内核对实际 task2 训练代码、配置、train/validation manifest、validation pair 文件、原 ordered head 的 `f_θ` 定义和 `sample_pairs` 实现；记录执行前 hash，并确认本任务使用的是协议修订版动态采样。
2. 在项目自有代码范围内新增对称 head 和独立配置/运行目录，不覆盖历史代码、配置或 checkpoint。
3. 先做小规模确定性 preflight：重复运行 sampler，核对固定 epoch 的 pair-stream hash、顺序、正负计数和三 variant 共享性；再用固定的少量 validation pair 验证 `g(x,y)` 与 `g(y,x)` 转置一致，并确认 labels、pair order、mask 和 feature shape。任何模型 smoke test 必须通过 Slurm；不得在登录节点运行模型。
4. 用三个固定 seed 训练 ES-SYM、ES-B-SYM、ES-C-SYM。每个组合沿用原 task2 训练/验证流程和 checkpoint 选择规则。
5. 在相同 validation manifest 和保存 pair 集上执行 canonical 与 AB/BA 交换推理；不得读取任何测试数据。
6. 对新模型与历史 ordered 对照生成逐 complex、逐 pair 和汇总指标；不根据中途结果添加 seed、改变负例、扫描超参数或扩大候选集。
7. 生成回执、配置、训练日志索引、checkpoint inventory、逐 pair 结果哈希、逐 complex 指标和 evidence index，然后停止。

## 7. 本次不做什么

- 不训练或比较 separable baseline，不做 surface×surface 完整候选矩阵或 top-k 新协议。
- 不做任务三局部交互矩阵、2×2 pooling 诊断、冻结 adapter 迁移或正式 PPI 测试。
- 不做 Pairformer、patch-pair、多轮推理、APBS、电荷输入、扩大 ESM2 或解冻 ESM2/GearNet/dMaSIF。
- 不读取 DIPS test、D-SCRIPT test 或任何新的外部测试结构、标签、预测和性能。
- 不重新划分数据、不改变同源隔离、不增加 seed、不根据结果选择模型或超参数；不引入除原 `sample_pairs` 外的负例策略，也不把本次动态重建流称为历史逐 pair 复现。
- 不把 symmetry pass 自动称为真实接触概率校准；对称性只解决交换不变性问题。
- 不覆盖旧 run、旧 checkpoint、旧预测、旧日志或共享 GearNet/Surface-GearNet/gearnet 环境；不自动删除任何失败产物。

## 8. 允许修改的范围

允许在项目自有代码和新运行目录中：

- 新增对称 pair head、训练入口适配、独立配置和必要的确定性验证脚本；
- 新增 sampler preflight、pair-stream hash 和训练 provenance 记录；允许为保证三 variant 共享同一动态流而增加最小包装代码，但不得改变 `sample_pairs` 的采样语义；
- 新增本任务的 Slurm 提交脚本、日志索引、报告和结果文件；
- 修复为实现同一 `g_θ` 所必需的局部 bug，但每项修改必须记录 diff、文件 hash 和理由。

禁止修改：

- 冻结 encoder 权重、特征缓存、原始数据、标签、mask、manifest、历史配置和历史结果；
- 共享环境及 AGENTS.md 指定的只读项目；
- 交换容差、指标定义、数据划分或 checkpoint 选择规则。

## 9. 资源预算与执行约束

用户已批准本实验的研究范围和新训练；但当前未给出数值化的 GPU 小时、分区、CPU、内存、并发或时限预算。执行 agent 不得从旧作业 `276472` 或其他项目推断资源参数。

提交前必须：

- 用当前集群实际的 `sinfo`/`scontrol` 核对可用 GPU 分区和资源字段；
- 选择能完成本任务的最小有效 GPU 配置，记录申请值和实际分配值；
- 记录 job ID、节点、GPU 型号、CPU、内存、时限、起止时间、exit code、Python/PyTorch/NumPy 版本和代码/配置 hash；
- 训练、推理、特征读取和模型 smoke test 均通过 Slurm；不在登录节点运行模型。

如果当前集群无法在项目规则下确定合法 GPU 预算，先完成代码静态核对、sampler preflight 和小型配置核对，随后停止并请示，不用本地运行替代。资源消耗必须报告实际值；预计值不得写成实测值。

## 10. 验收标准

任务可在科学上无增益时正确完成，须满足：

1. `g_θ` 的实现与第 4 节公式一致，且没有引入未批准的额外 head、读出、数据或超参数搜索。
2. 9 个预定 seed/variant 的训练状态、checkpoint、代码/配置/input hash 和 Slurm provenance 完整；失败组合必须明确列出，不能用成功组合代替。
3. 动态 sampler 的源码、manifest、seed、epoch、ratio、调用顺序和每个 epoch 的 pair-stream hash 完整；重复 preflight 产生完全一致的流，且同一 seed 下三个 variant 使用相同流。该证据不被表述为历史 ordered 训练流的逐 pair 复现。
4. 新模型在固定 AB/BA 检查中全部满足预先固定容差，或若未满足则把任务标为对称实现失败并停止性能解释；不得调宽容差掩盖失败。
5. canonical 与 swapped 的 pair 顺序、labels、offsets、mask 和 validation manifest 可核对；无测试访问。
6. ES-SYM、ES-B-SYM、ES-C-SYM 的 AP/AUROC 和交换差异均有逐 seed、逐 complex、逐 pair 证据；无须高于 ordered 对照。
7. 表面增益判断只使用同一新对称协议中的 ES-SYM 对比 ES-B-SYM/ES-C-SYM，并同时报告每个 seed 和预先定义的区间/标准差；不能把单侧增益写成 PPI 或真实接触概率结论。
8. 旧 ordered 结果保留为历史对照，新的 symmetry 结果单独命名为 developmental validation；明确注明训练 pair 流不是历史逐 pair exact replay，不生成正式测试结论。

以下任一科学结果均可作为正确完成：

- 对称 head 通过且 ES-B/ES-C 仍有增益；支持表面信息在对称读出下仍有局部价值；
- 对称 head 通过但表面增益消失；说明先前增益可能依赖 ordered head，表面结论需要收窄；
- 对称 head 通过但所有结果不确定或性能下降；说明结构约束可验证，但当前数据不足以证明表面优势；
- 对称 head 未通过；任务只能作为实现/数值失败回执，停止进一步机制解释。

## 11. 必须返回的指标、逐样本结果、配置和证据

主回执至少包含：

- 任务 ID/version、依据文件 hash、实际代码和配置 hash；
- 协议修订说明：原固定训练 pair 不存在；使用的 `sample_pairs` 源码、seed/epoch/ratio、manifest 顺序和 sampler hash；
- 9 个组合的训练状态、checkpoint 路径和 SHA256；
- Slurm job ID、实际节点/分区/GPU/CPU/内存/时长/exit code、依赖版本；
- `test_records_read`、`test_labels_read`，应为 0；
- 对称 head 公式、实际实现文件和数学/数值验证结果；
- canonical/swapped AP、AUROC、逐 pair 差异摘要和容差通过率；
- ES-B-SYM−ES-SYM、ES-C-SYM−ES-SYM 的逐 seed/逐 complex 结果、均值及已预先定义的区间或标准差；
- 与历史 ordered 对照的明确映射和解释限制；
- 未完成组合、失败原因、协议偏差和所有未知项。

附件至少包括：

- `run_config.json`：数据、manifest、feature、normalization、代码、配置、seed、head 公式和环境 hash；
- `checkpoint_inventory.tsv/json`：每个 checkpoint 的路径、大小、SHA256、strict load 状态；
- `train_history_<seed>_<variant>.json` 或等价逐 epoch 记录；
- `metrics_by_complex_<seed>_<variant>.tsv/json`；
- `predictions_<seed>_<variant>.npz`：至少含 `score_ab`、`score_ba_transposed`、`abs_delta`、`relative_delta`、`pairs`、`labels`、`offsets`、`lmdb_indices`；若体积过大，返回集群路径、SHA256 和可重算说明；
- `symmetry_summary.tsv/json`：逐组合容差、通过数、差异分位数及 canonical/swapped 指标；
- `slurm_job_<id>.txt`、代码 diff/source hash 和 `evidence_index.tsv`。

## 12. 停止条件与需要请示的情况

在以下情况立即停止受影响步骤并回执：

- 原 task2 validation pair、manifest、feature cache、normalization、代码或 checkpoint hash 与依据版本不一致；
- 无法确认训练/验证使用同一 pair 集、labels、mask 或 negative protocol；
- `sample_pairs` 源码、manifest 顺序、seed/epoch/ratio 或采样调用顺序发生变化；
- sampler 重复生成不一致，或无法证明同一 seed 下 ES-SYM、ES-B-SYM、ES-C-SYM 使用同一 pair 流；
- 原 ordered head 不是可按第 4 节公式对称化的结构，或需要引入新架构/新超参数；
- 需要 warm-start、改变 sampler 语义/ratio、修改标签、改变数据划分、增加 seed 或访问测试数据；
- 对称实现未通过固定容差，且无法证明只是记录或数值实现错误；
- Slurm GPU 分区、时限、CPU、内存或并发预算无法按项目规则确定；
- 发现必须修改冻结 backbone、共享环境或只读路径；
- 任一中途结果诱发扩大搜索、扫描额外 head 或追加实验的要求。

## 13. 当前决策摘要

- 当前目标：在 task2 中以最小结构变化建立交换不变的 pair head，并判断表面局部关系增益是否在该约束下保留。
- 已确认：旧 ordered head 对 AB/BA 顺序明显敏感；已有 task2 结果仍是 sampled-negative developmental validation；ES-B/ES-C 的表面增益尚未被此检查否定。
- 已确认：原 task2 没有保存固定训练 pair 流；本版本已获用户批准使用原 `sample_pairs` 的确定性动态重建，并将其作为协议偏差记录。执行范围仍固定为 3 个 seed × 3 个 variant 的显式对称 head 重训及验证。
- 未批准：separable baseline、任务三局部交互读出、独立封存测试、迁移训练、Pairformer、patch-pair 和多轮推理。
- 有效任务 ID：`SRPPI-T2-SYM-001 v1.1`。
