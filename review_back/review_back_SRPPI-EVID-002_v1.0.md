# Review back：SRPPI-EVID-002 v1.0

- 任务单：`/home/zy/projects/Surface-residue/plan/task_plan_SRPPI-EVID-002_v1.0.md`
- 任务单 SHA256：`a471cae4fe7c9fa932bb2a49bdd9a3a348862165a197c5f2c09a467f9d646567`
- 执行范围：任务三样本流转、任务一 49 条全正链诊断、已有任务一 ES-C 配对差值提取、任务二已保存 ES/ES-B/ES-C 的交换检查前置对齐。
- 本回执状态：前置核查完成；任务二 AB/BA 模型交换推理受资源和 checkpoint 可加载性阻塞。
- 标签含义：`【产物已核验】` 表示当前文件直接支持；`【历史记录，当前未核验】` 表示引用既有记录；`【分析或假设】` 表示解释；`【待确认】` 表示不能由当前证据确定。

## 1. 完成内容

### 1.1 任务三样本流转

【产物已核验】直接读取并统计了 structure-v2 的 `report.json`、`exclusions.json`、`train.json` 和 `val.json`。数量守恒全部通过：

| 阶段 | 总数 | 正例 | 负例 | 证据 |
|---|---:|---:|---:|---|
| 原始 official TRAIN | 5,000 | 2,500 | 2,500 | `structure_v2/report.json` |
| 结构/质量排除 | 2,709 | 1,096 | 1,613 | `exclusions.json`, reason=`missing_or_quality_excluded_structure` |
| 结构可用 | 2,291 | 1,404 | 887 | report 字段 `structure_eligible_*` |
| 跨 cluster pool 排除 | 985 | 590 | 395 | `exclusions.json`, reason=`cross_pool_pair_excluded_for_cluster_disjointness` |
| 最终使用 | 1,306 | 814 | 492 | 2,291−985 |
| train | 1,112 | 690 | 422 | `train.json` |
| validation | 194 | 124 | 70 | `val.json` |

核对结果为：5,000=2,709+2,291；2,291=985+1,306；1,306=1,112+194；报告中的 train/validation 数量与实际文件一致。该协议报告 `test_records_read=0`、`test_labels_read=0`。详细 TSV/JSON 见附件 `task3_sample_flow.tsv/json`。

### 1.2 任务一 49 条全正验证链

【产物已核验】使用任务一验证 manifest 的链顺序、三 seed 的 `ES_C_val_predictions.npz` 的 `offsets`、`labels`、`seq_ids` 和对应 metadata 建立链身份映射。934 条验证链全部成功映射，没有 identity、seq_id 或 label mismatch：

- 全正链：49 条；全负链：0 条；同时含正负残基的 mixed-label 链：885 条。
- 49 条全正链有效长度 min/median/max：3/8/19。
- 885 条 mixed-label 链有效长度 min/median/max：7/233/1366。
- 每条链的 common-mask count 等于 prediction slice 长度；逐链表见 `task1_all_positive_chain_table.tsv`，完整诊断 JSON 见 `task1_all_positive_chain_diagnostic.json`。

【产物已核验】按任务单要求，仅将 885 条 mixed-label 链的 chain-macro 指标作为辅助复算，不替换主报告的 934-chain 指标。三 seed 逐 seed 结果如下；seed 顺序为 20260927/20260928/20260929：

| 版本 | mixed-chain AP 逐 seed | mean±sample SD | mixed-chain AUROC 逐 seed | mean±sample SD |
|---|---|---:|---|---:|
| ES | 0.398648 / 0.405648 / 0.403983 | 0.402760±0.003657 | 0.709640 / 0.713223 / 0.712655 | 0.711839±0.001926 |
| ES-B | 0.423630 / 0.429523 / 0.422945 | 0.425366±0.003617 | 0.727120 / 0.733887 / 0.728588 | 0.729865±0.003559 |
| ES-C | 0.428119 / 0.426884 / 0.427340 | 0.427448±0.000625 | 0.731186 / 0.731130 / 0.733082 | 0.731799±0.001111 |

AUROC 与主报告一致，因为主报告本来只在 885 条含正负链上定义 AUROC。mixed-chain AP 是去除全正链后的辅助口径，不能与主报告的 934-chain AP 混写，也没有用于选择 checkpoint。

### 1.3 已有任务一 ES-C 配对差值

【产物已核验】从既有诊断报告直接提取，没有重训或重算旧模型。以下均为 **ES-C 减去指定诊断对照**，不是 ES-C−ES。区间为既有的、条件于已选 validation checkpoint 的 paired component percentile bootstrap（2,000 次；335 components）。

**all 残基子集**

| 比较 | 指标 | 逐 seed（20260927/28/29） | mean±sample SD | 95% interval |
|---|---|---|---:|---|
| ES-C−capacity-matched ES | AP | +0.026413 / +0.023274 / +0.022497 | +0.024061±0.002073 | [+0.015628, +0.032638] |
| ES-C−capacity-matched ES | AUROC | +0.019724 / +0.019135 / +0.021895 | +0.020251±0.001454 | [+0.013653, +0.026763] |
| ES-C−exposure/depth ES | AP | +0.023988 / +0.022480 / +0.021591 | +0.022687±0.001211 | [+0.014637, +0.030662] |
| ES-C−exposure/depth ES | AUROC | +0.018530 / +0.018659 / +0.018360 | +0.018516±0.000150 | [+0.011767, +0.024669] |
| ES-C−train+validation permutation ES-C | AP | +0.024828 / +0.021041 / +0.021221 | +0.022363±0.002136 | [+0.013904, +0.030941] |
| ES-C−train+validation permutation ES-C | AUROC | +0.016945 / +0.016468 / +0.017999 | +0.017137±0.000783 | [+0.010606, +0.023573] |

**surface 子集**

| 比较 | 指标 | 逐 seed（20260927/28/29） | mean±sample SD | 95% interval |
|---|---|---|---:|---|
| ES-C−capacity-matched ES | AP | +0.025496 / +0.022862 / +0.022249 | +0.023536±0.001725 | [+0.015166, +0.032040] |
| ES-C−capacity-matched ES | AUROC | +0.018959 / +0.019635 / +0.022777 | +0.020457±0.002038 | [+0.013390, +0.027424] |
| ES-C−exposure/depth ES | AP | +0.023709 / +0.022766 / +0.021584 | +0.022686±0.001065 | [+0.014623, +0.030755] |
| ES-C−exposure/depth ES | AUROC | +0.019630 / +0.020436 / +0.020030 | +0.020032±0.000403 | [+0.012848, +0.026639] |
| ES-C−train+validation permutation ES-C | AP | +0.024839 / +0.021428 / +0.021483 | +0.022584±0.001954 | [+0.014056, +0.031172] |
| ES-C−train+validation permutation ES-C | AUROC | +0.018211 / +0.018198 / +0.020266 | +0.018892±0.001190 | [+0.011874, +0.025859] |

原始字段、subset、seed 和 bootstrap 元数据保存在 `task1_existing_paired_deltas.json`。

### 1.4 任务二已保存 NPZ 的交换前置对齐

【产物已核验】针对正式 seed 20260927、20260928、20260929 的 `ES`、`ES_B`、`ES_C` 验证 NPZ，完成了 CPU 只读结构核查：

- 每个 seed、每个 variant 都有 669,235 个 pair、60,857 个正例、608,378 个负例和 467 个 validation complexes。
- 同一 seed 内，ES/ES-B/ES-C 的 `labels`、`pairs`、`offsets`、`lmdb_indices` 完全一致。
- 每个 seed 的 `lmdb_indices` 和 validation manifest 顺序一致；offsets 长度为 468、边界有效且覆盖全部 labels。
- pair 均在两侧 common mask 内；重复 pair 数为 0；logits 无 NaN/Inf。
- 三个 seed 的 labels、offsets、lmdb_indices 相同；pair 集合在不同 seed 之间不同。这符合既有协议“同一 seed 内 variant 共享候选集、不同 seed 可使用不同 sampled-negative 集合”的约束。
- 每个 variant 的 JSON metadata 和 NPZ SHA256 已写入 `task2_saved_pair_alignment.json`。

该结果只证明 saved candidate 对齐，不能证明 AB/BA 的模型交换一致性。

## 2. 证据索引

核心来源及 SHA256：

| 来源 | 绝对路径 | SHA256 |
|---|---|---|
| EVID-002 任务单 | `/home/zy/projects/Surface-residue/plan/task_plan_SRPPI-EVID-002_v1.0.md` | `a471cae4fe7c9fa932bb2a49bdd9a3a348862165a197c5f2c09a467f9d646567` |
| AGENTS | `/home/zy/projects/Surface-residue/AGENTS.md` | `03f307554c52c0d8cadec0f13b8ae8eccdd1704598acd2c08314224a6ec0341b` |
| task1 主报告 | `/home/zy/projects/Surface-residue/runs/phase3_task1_summary/20260927/main_v1/report.json` | `d79fa09c71d5d6d1d0ad001646afd034a11d398c6812daae391a57f406c56d3e` |
| task1 诊断报告 | `/home/zy/projects/Surface-residue/runs/phase3_diagnostics_summary/20260928/main_v1/report.json` | `3dd78cf5b116ee210859acda2b9310617022458e7795f5ca36bfcd061b8c33d1` |
| task2 主报告 | `/home/zy/projects/Surface-residue/runs/phase3_task2_summary/20260928/common_mask_v2/report.json` | `255f84332fc589b55efb92a78a93591ef7c2eaef17a307b601265d9fdf84de7c` |
| task3 protocol | `/home/zy/projects/Surface-residue/runs/phase3_task3_protocol/20260928/structure_v2/report.json` | `ffaadc6e790accf4b119676edc8bb9a22803caae4dbaad59fe5c4fbd6c136593` |
| task3 exclusions | `/home/zy/projects/Surface-residue/runs/phase3_task3_protocol/20260928/structure_v2/exclusions.json` | `e478a247caa421d4bf89cef7c91270b3a5bbb12c785d2cbcd993538b04c7bb7f` |

新增附件及用途：

| 附件 | 用途 |
|---|---|
| `task3_sample_flow.tsv/json` | 任务三逐阶段总数、正负计数、排除原因和守恒检查 |
| `task1_all_positive_chain_table.tsv` | 49 条全正链逐链身份、长度、标签和 mask |
| `task1_all_positive_chain_diagnostic.json` | 全正/mixed/all-negative 计数、映射检查和长度摘要 |
| `task1_mixed_chain_metrics.json` | 885 条 mixed-label 链三 seed 辅助指标 |
| `task1_existing_paired_deltas.json` | 已有 ES-C 三类对照的 all/surface 配对差值 |
| `task2_saved_pair_alignment.json` | task2 三 seed/三 variant 的 saved NPZ 对齐和有限值检查 |
| `task2_swap_check_config.json` | 未来 AB/BA 检查的固定输入、输出和资源要求草案 |
| `task2_swap_check_summary.json` | AB/BA 检查未执行的阻塞状态 |
| `evidence_index.tsv` | 附件和来源的 SHA256 索引 |

所有新增文件均位于：`/home/zy/projects/Surface-residue/review_back/SRPPI-EVID-002_v1.0/`。

## 3. 偏差

1. 任务二 AB/BA 模型推理没有执行。任务单要求通过 Slurm GPU；本次没有获批具体 GPU、分区、时限、CPU/内存和并发预算，因此没有提交作业，也没有在本地 CPU 替代。
2. 当前工作站 `user-ThinkStation-P3-Tower` 没有 `squeue`/`sacct` 客户端，不能刷新集群队列。既有交接包最后一次集群查询（2026-09-28 11:23，Asia/Shanghai）记录为无运行或排队作业；本回执不把该历史状态写成当前实时状态。
3. 任务二三个正式训练目录保存了 prediction NPZ、metadata 和 report 中的 checkpoint SHA256，但当前本地同步目录中没有对应 `.pt`、`.pth` 或 `.ckpt` 文件及可加载路径。因此即使 GPU 预算获批，执行交换推理前仍需先恢复/定位原 checkpoint；不能用已有 logits 冒充 AB/BA 结果。
4. 本轮只做允许的读、校验和复算；没有修改旧代码、配置、环境、数据、缓存、checkpoint、历史报告或共享工程。新增文件仅在 EVID-002 回执目录。
5. 本轮 mixed-chain AP 是辅助口径，不是主指标；没有用它或任何新核查结果选择 checkpoint、调参或改变研究范围。

## 4. 问题

- 【已解决】任务三流转中的结构质量排除和跨 cluster pool 排除均已由 `exclusions.json` 逐条计数并与 protocol report 对齐。
- 【已解决】任务一验证 manifest 与 prediction offsets/labels/seq_ids 的 934 条链映射无 mismatch。
- 【已解决】任务二同一 seed 内三个 variant 的 saved pair/label/order 一致性、common-mask 范围、offsets、重复 pair 和有限值检查通过。
- 【受阻】任务二 AB/BA 交换数值尚无结果；不能据此判断当前 pair head 是否对链顺序敏感。
- 【受阻】任务二模型交换所需 checkpoint 文件在当前同步目录中不可加载/不可定位；report 中只有 checkpoint SHA256。
- 【待确认】本地文件与集群项目同步的最新时点，以及集群当前作业队列；本地没有 Slurm 查询命令。
- 【历史风险，当前未重复扩展】任务三早期 sequence pilot/subset preparation 曾访问 official test，详见既有更正记录；本轮未读取新的测试结构、标签、预测或性能。

## 5. 待决事项

### 用户/决策 agent 需要确认

1. 是否批准任务二交换推理的新增 GPU 预算，并固定 GPU 类型（A100 优先或 A40 回退）、分区、时限、CPU、内存和并发上限。
2. 是否能提供或允许在集群上只读定位三个 seed、ES/ES-B/ES-C 的可加载 checkpoint 路径；必须与现有 report 的 checkpoint SHA256 一致。
3. 是否接受“saved NPZ 对齐已通过、AB/BA 尚未测量”为本任务的停止点。

### 候选下一步（仅建议，不是执行指令）

- **候选 A：先恢复 checkpoint 后做 Slurm validation-only AB/BA。** 回答当前 ordered pair head 的交换差异；依赖 GPU 预算、checkpoint 路径和固定容差；成本是 3 seed×3 variant 的一次验证推理。
- **候选 B：在无法恢复 checkpoint 时停止交换检查。** 保留 saved alignment 证据，不把协议一致性前置检查误写成模型对称性证据；成本最低，但不能回答 AB/BA 假设。
- **候选 C：另立任务重训/改造显式对称 head。** 只有交换差异明确且决策 agent 另行批准时考虑；不属于本任务，不能由本回执自动启动。

## 6. 验收标准逐项回应

| 验收项 | 结果 |
|---|---|
| 任务三 5,000→2,291→1,306 流转、正负计数和排除原因 | 通过；见 `task3_sample_flow.tsv/json` |
| 49 条全正链身份、长度、label/mask 摘要 | 通过；934 条映射无 mismatch，49 条逐链表已保存 |
| 885 条 mixed-label 辅助指标与主指标区分 | 通过；三 seed 逐 seed 和 mean±SD 已保存，未替换主指标 |
| 三类既有 ES-C 对照配对差值 | 通过；all/surface、AP/AUROC、逐 seed、mean、区间均直接提取 |
| 任务二三 seed/三 variant saved pair 对齐 | 通过；同 seed 完全一致，跨 seed pair 差异符合 sampled-negative 协议 |
| AB/BA 交换检查 | 未完成，明确受阻；未提交 Slurm，未伪造结果 |
| 不读取新测试数据、不产生新训练结果、不修改旧产物 | 通过；本轮测试访问计数为 0，未启动作业 |
| 回执包含完成内容、证据、偏差、问题、待决事项 | 通过；本文件及附件已生成 |

## 7. 执行环境与作业记录

- 实际只读核查主机：`user-ThinkStation-P3-Tower`。
- 项目工作区：`/home/zy/projects/Surface-residue`。
- 本轮开始/结束记录：约 2026-09-28 16:27—17:02（Asia/Shanghai）；纯 CPU 文件读取、NPZ 校验和指标复算。
- Python：3.13.13；本轮未安装依赖、未修改环境。
- 测试访问计数：本轮 `test_records_read=0`、`test_labels_read=0`；任务三 protocol 既有报告同样为 0，但其报告另有历史 upstream test-access deviation，未在本轮扩展。
- Slurm：本轮未提交、取消、重启或重复提交任何作业；job ID：无。
- 既有结果作业、模型/缓存路径及集群规则请以 `SurfaceResidue-PPI_事实交接包_20260928.md` 和本回执的证据索引为准。

本回执完成后停止，不自动进入下一轮实验。
