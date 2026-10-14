# task_plan：证据补齐与任务二交换一致性预审

## 1. 任务标识与交付位置

- 任务 ID：`SRPPI-EVID-002`
- 版本：`v1.0`
- 日期：2026-09-28
- 状态：用户已确认生成；执行前仍须遵守本文件的资源和停止条件。
- 本地项目：`/home/zy/projects/Surface-residue`
- 本任务单：`plan/task_plan_SRPPI-EVID-002_v1.0.md`
- 主回执：`review_back/review_back_SRPPI-EVID-002_v1.0.md`
- 附件目录：`review_back/SRPPI-EVID-002_v1.0/`

本任务是 `SRPPI-EVID-001 v1.0` 的后续预审任务。完成后停止，等待决策 agent 和用户审阅；不自动衔接任务二新基线、任务三新读出或正式测试。

## 2. 任务依据与当前科学问题

依据文件及版本：

| 文件 | SHA256 |
|---|---|
| `plan/task_plan_SRPPI-EVID-001_v1.0.md` | `479a1def993a338352ce2668f0693b56d0463e36adfaf9bd7be50cc1459da30a` |
| `review_back/review_back_SRPPI-EVID-001_v1.0.md` | `88bc078e43d88274eccb11e81369e121b4e5755e00a5b806e0cad5f2a1af40e6` |
| `review_back/SRPPI-EVID-001_v1.0/checkpoint_provenance.md` | `66142cfabfab21882efac978a8af24cc44db6ee025c51f9694320e194cb00bb9` |
| `runs/phase3_diagnostics_summary/20260928/main_v1/report.json` | `3dd78cf5b116ee210859acda2b9310617022458e7795f5ca36bfcd061b8c33d1` |
| `runs/phase3_task1_summary/20260927/main_v1/report.json` | `d79fa09c71d5d6d1d0ad001646afd034a11d398c6812daae391a57f406c56d3e` |
| `runs/phase3_task2_summary/20260928/common_mask_v2/report.json` | `255f84332fc589b55efb92a78a93591ef7c2eaef17a307b601265d9fdf84de7c` |
| `runs/phase3_task3_protocol/20260928/structure_v2/report.json` | `ffaadc6e790accf4b119676edc8bb9a22803caae4dbaad59fe5c4fbd6c136593` |
| `runs/phase3_task3_protocol/20260928/structure_v2/exclusions.json` | `e478a247caa421d4bf89cef7c91270b3a5bbb12c785d2cbcd993538b04c7bb7f` |

当前接受的研究判断：

- 任务一有稳定的开发验证增益；任务二信号更值得追踪；任务三当前 structure-v2 子集没有稳定的表面增益。
- B/C 没有稳定差异，下一轮优先关注 B；C 可作为次要描述版本。
- dMaSIF 权重字节身份已追溯到官方资产，但实际训练成员仍未知。`SRPPI-EVID-001` 已完成其可执行范围；本任务不重复同样的 provenance 搜索。
- 下一阶段要检验局部交互信息及其未能传递到 PPI 的候选原因，不假设 mean pooling 已经是唯一原因。

本任务只做前置证据与正确性核查，为后续任务二机制实验提供进入条件。

## 3. 本次目标与待检验假设

### 目标 A：补齐已有结果中影响解释的事实

1. 将任务三 2,291 个结构可用 pair 到最终 1,306 个 pair 的去向写成逐阶段流转表，区分正负样本和排除原因。
2. 对任务一 49 条全正验证链做不改变主指标的诊断，报告链身份、有效长度分布、标签/共同 mask 范围，并重算仅含 885 条同时含正负残基链的辅助指标。
3. 从已保存的任务一诊断报告提取 ES-C 与容量、暴露/深度、训练/验证置乱对照的配对差值及区间；不得把它们与 ES-C−ES 混写，也不得重复训练。

### 目标 B：检查任务二当前 pair head 是否依赖任意的链顺序

对既有 `task2-common-mask-v2` 验证 checkpoint，在相同缓存、相同验证复合物和相同已保存候选 pair 上比较：

\[
S_{AB}(i,j)\quad\text{与}\quad S_{BA}(j,i).
\]

待检验假设是：当前模型对交换蛋白顺序近似不敏感。当前代码使用独立 `left`/`right` 投影，并不提供数学上的对称保证，因此该检查可能发现协议解释限制。

如果交换差异明显，本任务只报告问题并停止；是否重训一个显式对称 head 另立任务。

## 4. 固定数据、模型、指标和比较条件

### 4.1 只允许引用的既有协议

- 任务一：`expanded_v1`，4,861 train / 467 validation complexes，934 validation chains；bound DIPS 单体。
- 任务二：`task2-common-mask-v2`，467 validation complexes；每个 seed 已保存的正例与 sampled-negative pair 集合；不得重新采样。
- 任务三：`task3-structure-v2`，原始 5,000 official TRAIN pairs，结构可用 2,291 对，最终 train/validation 1,306 对；AlphaFold 独立单体。
- 既有 seed：20260927、20260928、20260929。不得增加 seed 或根据结果筛选 seed。
- dMaSIF、ESM2、GearNet checkpoint、A/B/C 定义、common masks、训练归一化和历史特征哈希均保持原值。

### 4.2 指标与输出口径

- 任务一主指标仍为 chain-macro AP，AUROC 只在同时含正负链上定义。
- 任务一辅助子集必须明确写为“885 条 mixed-label validation chains”，不得替换主报告的 934 条链。
- 任务二交换检查返回逐 complex 的绝对差、相对差、最大差和预先定义的数值容差；可同时报告 canonical 与 swapped 方向的 complex-macro AP/AUROC，但不得据此选择 checkpoint。
- 所有历史区间仍是条件于已选 validation checkpoint 的组件 bootstrap；本任务新增的交换差异不自动视为新的显著性检验。
- 不读取 DIPS test 或 D-SCRIPT test 的结构、标签、预测或性能。

## 5. 明确执行范围

### 5.1 允许的只读与复算

- 读取上述现有 JSON、manifest、NPZ、预测元数据和报告；生成新的小型审计表与汇总，不修改原产物。
- 从任务三 `exclusions.json` 统计以下固定流转：

  | 阶段 | 总数 | 正例 | 负例 |
  |---|---:|---:|---:|
  | 原始 official TRAIN | 5,000 | 2,500 | 2,500 |
  | 结构/质量排除 | 2,709 | 1,096 | 1,613 |
  | 结构可用 | 2,291 | 1,404 | 887 |
  | 跨 cluster pool 排除 | 985 | 590 | 395 |
  | 最终使用 | 1,306 | 814 | 492 |
  | 其中 train | 1,112 | 690 | 422 |
  | 其中 validation | 194 | 124 | 70 |

  上表数值须由执行前实际文件核对；若文件变化，报告实际值及哈希，不静默沿用表中数字。

- 对任务一 49 条全正链：从现有验证预测的 offsets、labels、seq_ids 与 validation manifest 建立 chain identity 表；报告有效长度的 min/median/max、标签计数、共同 mask 计数和 chain ID。不得读取新的测试数据。
- 由现有预测重算 885 条 mixed-label 链的每 seed AP/AUROC、均值和样本标准差；若实现组件 bootstrap，组件定义和复采样规则必须与主报告一致，否则只报告逐 seed 及 mean±SD。
- 从现有诊断 JSON 直接提取并核对下列 ES-C 配对比较：
  - ES-C − capacity-matched ES；
  - ES-C − exposure/depth ES；
  - ES-C − train+validation permutation ES-C。
  返回 AP/AUROC 的逐 seed delta、均值、区间和对应 subset。不得因本次任务重算旧模型。

### 5.2 任务二交换检查

- 仅针对当前三个 seed 和 `ES`、`ES-B`、`ES-C`；不得引入 A、E、S 或新 head。
- 使用每个 seed 已保存的 validation `pairs`、`labels`、`offsets` 和 `lmdb_indices`；AB 与 BA 必须使用同一批 pair 的相反方向，不能重新生成负例。
- 使用与原训练相同的 frozen feature cache、normalization、checkpoint 和 adapter 配置；labels、partner 坐标和接触边只用于既有 pair 对齐/评估，不进入 encoder。
- 推理必须通过符合 `AGENTS.md` 的 Slurm GPU 作业；不在登录节点运行模型。作业只做验证集推理和数值比较，不训练。
- 每个 seed/variant 返回：
  - 逐 pair `score_ab`、`score_ba_transposed` 或可重算等价物；
  - `abs_delta`、`relative_delta` 的 mean/median/max、预先固定分位数（50/95/99%）；
  - canonical 与 swapped 方向的逐 complex AP/AUROC（若可定义）；
  - checkpoint、feature cache、manifest、代码和 Slurm job ID 哈希/路径。
- 数值容差必须在运行前写入配置。不得看完差异后再调宽容差。若出现 NaN/Inf、pair 顺序不一致、mask 不一致或无法复现原保存候选，停止该检查并回报。

### 5.3 关于 C 置零和刚体测试的边界

- 本任务不把“C 内容打分置零后必须退化为 B”作为训练结果验收。B 与 C 是分别训练的 adapter，且 C 的 value/output/gate 参数不与 B 共享；直接置零不能证明等于 B。若以后做此测试，必须另建共享参数的单元测试协议。
- 本任务不补做完整 ESM/GearNet/dMaSIF 刚体不变性实验；已有 adapter 点排列测试和其限制只需在回执中引用。完整编码器刚体测试另立任务。

## 6. 本次不做什么

- 不训练、不调参、不提交新的 ES/ES-B/ES-C 任务二模型。
- 不实现或运行“不建模匹配”的 separable baseline。
- 不计算新的 surface×surface 完整候选矩阵，不新增 top-k 规则。
- 不实现任务三局部交互矩阵、smooth-max/top-k 汇聚或 2×2 读出实验。
- 不做 Pairformer、patch-pair、多轮推理、APBS、电荷输入、扩大 ESM2、解冻 backbone 或迁移训练。
- 不重新计算 dMaSIF 训练成员，不重复 EVID-001 的有界 provenance 搜索；只引用其 unresolved 风险。
- 不打开任何测试结构、标签、预测或性能；不根据验证诊断选择后续模型。
- 不修改现有代码、配置、缓存、数据、checkpoint、历史报告或共享环境；不删除失败产物。

## 7. 允许新增的文件和修改范围

只允许在以下位置新增本任务回执和小型附件：

- `review_back/review_back_SRPPI-EVID-002_v1.0.md`
- `review_back/SRPPI-EVID-002_v1.0/`

建议附件：

- `task3_sample_flow.tsv/json`
- `task1_all_positive_chain_table.tsv`
- `task1_mixed_chain_metrics.json`
- `task1_existing_paired_deltas.json`
- `task2_swap_check_config.json`
- `task2_swap_check_summary.json`
- `task2_swap_predictions_<seed>_<variant>.npz`（若文件体积和项目路径允许；否则返回可重算路径及哈希）
- `evidence_index.tsv`

不得覆盖同名旧附件；若回执已存在，使用修订号。

## 8. 具体执行步骤

1. 读取本任务单、适用 `AGENTS.md`、EVID-001 回执和既有报告；记录执行前 SHA256。
2. 先完成纯读取/复算部分：任务三样本流转、任务一全正链与 mixed-chain 诊断、已有 ES-C 配对差值提取。
3. 核对 task2 既有 saved NPZ 的 validation complex 顺序、pair 顺序、labels、offsets 和三 variant 的共同候选集合；不一致时停止交换检查。
4. 提交一个仅验证集推理的 Slurm GPU 作业，按固定配置计算 AB 与 BA 交换结果；保留 job ID、实际节点、分区、GPU、起止时间和依赖版本。
5. 独立重算交换摘要和必要指标，核对保存数组哈希；不根据结果改变容差、mask、模型或样本。
6. 写回执，逐项回应验收标准，给出“可进入 separable baseline / 需要先修正对称 head / 证据不足”的候选判断，然后停止。

## 9. 资源预算

| 资源 | 本任务批准状态 |
|---|---|
| 现有报告/manifest/NPZ 的轻量读取与 CPU 复算 | 在本任务范围内允许；记录实际主机、时长和 Python 版本 |
| 新 Slurm GPU 推理 | **待用户确认具体 GPU、分区、时限、CPU/内存和并发预算**；确认前不得提交 |
| 新训练 | 0；不允许 |
| 测试数据访问 | 0；不允许 |
| 新数据下载、环境安装、共享环境修改 | 0；不允许 |

若 GPU 预算未确认，执行 agent 可以先完成第 8 步中的纯读取/复算部分，但必须把交换检查标记为 `blocked_pending_resource_approval`，不能用本地推理替代 Slurm。

## 10. 验收标准

任务在以下条件下可正确完成；不要求性能提升：

1. 任务三 5,000→2,291→1,306 的流转、正负计数和两个排除原因与实际 protocol/exclusions 文件一致。
2. 49 条全正链的身份、有效长度和 mask/label 摘要返回；885 条 mixed-label 链的辅助指标与主指标明确区分，不能替换主指标。
3. 已有 ES-C 与三个诊断对照的配对差值被准确提取，AP/AUROC、subset、seed 和区间不混淆；不重复训练。
4. 任务二三个 seed、三个 variant 的 AB/BA pair 对齐经过检查；交换检查成功或以明确的失败/不可访问状态结束。
5. 若交换检查成功，返回逐 seed/variant 的差异摘要、容差、指标和完整 provenance；若差异显著，明确标为协议解释限制，不擅自重训。
6. 不修改旧产物，不读取新测试数据，不产生新训练结果；Slurm 作业（若批准）只做验证集推理。
7. 回执区分已核验事实、历史记录、解释假设、失败和未知，并给出下一步候选而不自动扩大范围。

科学上出现以下结果均可验收：交换近似成立、交换差异明显、某些 seed/variant 无法完成、mixed-chain 指标无改善或样本流转暴露出选择偏差。完成标准不是提升性能或证明表面机制成立。

## 11. 必须返回的证据

- 主回执：任务 ID、版本、依据哈希、实际主机/节点、时间、依赖、作业状态、测试访问计数。
- 所有新增附件的 SHA256、配置、manifest/protocol/checkpoint/feature cache 哈希。
- 任务三样本流转表及每一排除原因的正负计数。
- 任务一 49 条全正链逐链表、mixed-chain 指标和重算入口。
- 既有 ES-C 配对差值的逐 seed、均值、区间和 subset。
- 任务二交换检查的逐 seed/variant 摘要与可重算逐 pair 结果或哈希。
- 明确记录 `SRPPI-EVID-001` 已完成但训练成员仍未知；不得把本任务写成已解决 provenance。

## 12. 停止条件与请示事项

立即停止受影响步骤并回执说明：

- 当前 checkpoint、feature cache、protocol、manifest 或 saved pair 的 hash 与既有报告不一致；
- 无法确认 AB/BA 使用同一 pair 集合、mask 或 validation 顺序；
- 需要重新生成负例、访问测试数据、修改标签或修改历史结果；
- 需要训练、安装环境、下载替代权重、扩大到全量候选矩阵或运行新模型头；
- 交换差异显示当前任务二是明显有序模型，需要重训对称 head 才能继续科学解释；
- Slurm 分区、GPU、时限、内存或并发预算未获确认；
- 发现新的 provenance 线索需要扩大 EVID-001 的搜索范围。

## 13. 当前决策摘要

- 当前目标：补齐任务一/三解释证据，并确认任务二现有接触 head 的交换一致性。
- 已确认：任务一/二为 developmental validation；任务三当前 structure-v2 无稳定表面增益；C 未证明优于 B；dMaSIF 训练成员仍未知。
- 本任务完成后才决定是否批准 task2 separable baseline 和 surface×surface 候选评估。
- 任务三局部交互 2×2、冻结 adapter 迁移、独立封存测试和 Pairformer 均未批准。
- 当前有效任务：`SRPPI-EVID-002 v1.0`。
