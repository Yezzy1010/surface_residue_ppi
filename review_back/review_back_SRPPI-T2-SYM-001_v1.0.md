# review_back：SRPPI-T2-SYM-001 v1.0

## 完成内容

- 已读取并按范围核对任务单 `plan/task_plan_SRPPI-T2-SYM-001_v1.0.md` 和项目规则 `AGENTS.md`。
- 已在集群项目 `/datapool/home/zhangye/projects/surface-residue-ppi` 做只读核查：固定协议、训练/验证 manifest、feature report、task2 配置、原 task2 训练入口、已有运行目录和当前 Slurm 状态。
- 已核对当前版本的关键 SHA256：

  | 项目 | SHA256 |
  |---|---|
  | `scripts/train_phase3_task2.py` | `685f688b83621f0b0d03ad644c29c17ff2481adb1abc4636a33321d9c3e8bb15` |
  | `src/surface_residue_ppi/adapters.py` | `b5d43fa7f4d36cdc75c8b850c20f8d735d44a605aed2d8982258141a90d0775f` |
  | `configs/experiments/phase3_20260927.json` | `baa6475493430130d1af2cc40290c4cef17b3047920e3f7eaee2571e43869fe9` |
  | `runs/phase3_protocol/20260927/expanded_v1/report.json` | `d10e45df9750e8b730d0f300c1ea9dd13f1f9d8d2b7a03c7f302fe8eb01874c2` |
  | `runs/phase3_protocol/20260927/expanded_v1/val_manifest.json` | `476416f9e286f691011bd075a4132aefcdaabcce9f4bfc44aa414bd77ec73b27` |
  | `runs/phase3_protocol/20260927/expanded_v1/strict_val_manifest.json` | `476416f9e286f691011bd075a4132aefcdaabcce9f4bfc44aa414bd77ec73b27` |
  | `runs/phase3_protocol/20260927/expanded_v1/train_manifest.json` | `4107096e8ef2acffab31c388c1f3156659186addcbcacd30b25dcbdc71733d72` |
  | `runs/phase3_features/expanded_v1/report.json` | `f3d2c81aaaec607a5911548a3831bc6ef42192740a9ca23d51e3e4e256f9c081` |
  | `runs/phase3_task2_summary/20260928/common_mask_v2/report.json` | `255f84332fc589b55efb92a78a93591ef7c2eaef17a307b601265d9fdf84de7c` |

- 已确认旧 task2 的 ordered head 代码仍是 `PairHead` 的独立 `left`/`right` 投影；本任务尚未新增对称 head，也没有修改代码、配置、环境或共享项目。
- 已确认当前集群 GPU 分区和作业状态。查询时间：`2026-09-28 13:21:08 UTC`（集群本地约 `21:21:08 +08:00`）。`squeue -u zhangye` 为空，本任务没有提交作业、没有运行 preflight、没有启动训练或推理。

## 证据

### 1. 停止原因：缺少可复用的固定训练 pair

任务单要求复用原 task2 的固定 train pair、labels、offsets、`lmdb_indices`，并明确规定如果该文件不存在，不得重新采样替代。

集群项目内的只读文件搜索结果显示，在排除原始 LMDB 和第三方目录后，没有保存的 task2 固定训练 pair、offset、negative 或 `lmdb_indices` 文件。命中的相关文件只有：

- `src/surface_residue_ppi/paired_statistics.py`
- `tests/test_paired_statistics.py`
- 对应的 source snapshot 测试/源码

训练脚本本身证明原 task2 训练不是从固定 pair 文件读取负例，而是运行时动态生成：

- `scripts/train_phase3_task2.py:106`：`def sample_pairs(record, seed, epoch, ratio=3)`；
- `scripts/train_phase3_task2.py:206`：训练循环调用 `sample_pairs(record,seed,epoch,3)`；
- `scripts/train_phase3_task2.py:165`：验证调用 `sample_pairs(record, seed, 100000 + ratio, ratio)`；
- `scripts/train_phase3_task2.py:258`：运行报告记录“up to three ... negatives per positive for training; up to ten for validation”。

因此，现有 `train_manifest.json` 只提供样本/结构清单，不能代替任务单要求的已保存 pair/label/offset/index 集。已有 `runs/phase3_features/.../*_labels.npy` 是逐侧 feature cache 的标签数组，不能证明存在 task2 固定 pair 流，也不能恢复原训练负例顺序。

### 2. 固定协议和历史结果仍在

- protocol report：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_protocol/20260927/expanded_v1/report.json`
- train manifest：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_protocol/20260927/expanded_v1/train_manifest.json`
- validation manifest：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_protocol/20260927/expanded_v1/val_manifest.json`
- strict validation manifest：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_protocol/20260927/expanded_v1/strict_val_manifest.json`
- feature report：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_features/expanded_v1/report.json`
- task2 summary：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_summary/20260928/common_mask_v2/report.json`

协议报告显示 train 4861、validation 467 个 complexes；历史 task2 report 显示每个 seed/variant 的既有开发验证结果和 `test_records_read=0`、`test_labels_read=0`。这些历史结果保留为对照，但不能提供本任务要求的固定训练负例文件。

### 3. 当前 Slurm 资源快照

当前 `sinfo` 显示：`a100` 分区 3 节点、A100 共 23 张；`a40` 分区 4 节点、A40 共 32 张；默认 GPU 分区、`h800` 等也处于 `up` 状态。各相关分区 `Timelimit=infinite`，但本任务没有因此提交作业，因为输入协议已先触发停止条件。查询时用户无排队或运行作业。

### 4. 代码/仓库状态

集群项目是 git 工作树，HEAD 为 `4704b706ba3943377e59bab5990b703096a1d6dc`（2024-05-07）。工作树本来就存在大量上级目录删除/未跟踪状态；本任务没有写入集群项目，不能把这些历史状态归因于本任务。

## 偏差

- 没有实现 `g(x,y)=0.5[f(x,y)+f(y,x)]`，因为实现和训练前的固定 pair 验收尚未满足输入条件。
- 没有执行 Slurm preflight、9 个组合训练、验证集 AB/BA 推理、checkpoint 生成或指标计算。
- 没有读取测试数据；本任务新增的 `test_records_read=0`、`test_labels_read=0`。
- 没有重新采样负例、改变划分、增加 seed、warm-start 旧 ordered checkpoint 或修改任何共享环境。

上述不是协议偏离，而是按任务单第 5.1、9、12 节的停止条件停止。当前状态应记为“训练前受阻”，不能把旧 ordered task2 的完成状态写成本任务结果。

## 问题

核心问题是：原 task2 训练入口只在运行时按 `seed` 和 `epoch` 动态 rejection-sampling 负例；项目中没有可核验的原始固定训练 pair、labels、offsets、`lmdb_indices` 产物。因此无法在不改变负例协议的前提下重训 9 个对称组合。

## 待决事项

决策 agent 需要先决定以下一个方向，执行 agent 才能继续：

1. 提供或恢复原 task2 的固定训练 pair 流（包括 labels、offsets、`lmdb_indices` 及其 hash），然后按本任务继续；或
2. 明确批准修改本任务单，允许用原脚本的确定性动态采样重建训练 pair，并把该变化作为协议偏差重新审查。

在获得其中一种明确决定前，不应实现、提交或运行对称 head 训练。当前没有可报告的训练 job ID、checkpoint、AB/BA 指标或表面增益结论。

## 证据索引

| 证据 | 主机/环境 | 用途 |
|---|---|---|
| `plan/task_plan_SRPPI-T2-SYM-001_v1.0.md` | 本地 `/home/zy/projects/Surface-residue` | 本任务范围与停止条件 |
| `AGENTS.md` | 本地 `/home/zy/projects/Surface-residue` | Slurm、共享路径和权限规则 |
| `scripts/train_phase3_task2.py` | 集群项目 | 原 task2 pair/负例生成和训练流程 |
| `runs/phase3_protocol/20260927/expanded_v1/*manifest.json` | 集群项目 | 固定结构/样本划分；不含固定负例流 |
| `runs/phase3_features/expanded_v1/report.json` | 集群项目 | feature cache 清单和 hash |
| `runs/phase3_task2_summary/20260928/common_mask_v2/report.json` | 集群项目 | 历史 task2 协议和指标摘要 |
| `runs/phase3_task2_training/276314_20260929`, `276315_20260927`, `276316_20260928` | 集群项目 | 历史 ordered checkpoint；本任务未修改 |
| `sinfo`, `scontrol show partition`, `squeue -u zhangye` | 集群 mgt01 | 当前资源和作业状态快照 |

