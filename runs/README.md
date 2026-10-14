# 实验产物导航

保留原来的实验名、日期和 job ID 路径。数据、特征和 checkpoint 不进入 Git，本导航进入 Git。整理前集群共约 48G。

| 路径 | 内容与用途 |
|---|---|
| `atom3d_audit/` | 发布包、LMDB 和接触审计 |
| `atom3d_residue_inventory/` | 全量残基库存 |
| `atom3d_residue_chemistry/` | 残基化学身份与修饰检查 |
| `atom3d_residue_labels/` | 标签重建诊断，非正式训练标签发布 |
| `atom3d_sequence_mapping/` | 序列与结构身份映射 |
| `phase2_candidates/`、`phase2_candidate_audit/` | 候选与接触完整性审计 |
| `phase2_protocol/` | 冻结的 Phase 2 train/val 协议 |
| `phase2_features/` | Phase 2 分块、失败尝试和 final/v2 特征缓存，约 5.8G |
| `phase2_task1_training/` | Phase 2 单 seed 训练 |
| `phase3_candidates/` | Phase 3 扩展候选 |
| `phase3_protocol/` | Phase 3 划分、掩码、标签与样本，约 2.2G |
| `phase3_features/` | Phase 3 主特征缓存，约 28G |
| `phase3_mmcif_prefetch/` | 结构下载与重试记录 |
| `phase3_orchestration/` | 自动续跑/提交记录 |
| `phase3_smoke/`、`smoke/` | 工程验证，包含历史失败运行 |
| `phase3_task1_training/`、`phase3_task1_summary/` | 任务一训练和汇总 |
| `phase3_diagnostics/`、`phase3_diagnostics_summary/` | 任务一机制诊断 |
| `phase3_task2_training/`、`phase3_task2_summary/` | 任务二 ordered 主比较，包含被后续修订替代的旧运行 |
| `phase3_task2_swap_check/` | AB/BA 交换审计 |
| `phase3_task2_sym_preflight/`、`phase3_task2_sym_training/` | T2-SYM v1.1 预检、三 seed 训练和 `summary_v1.1_light/` |
| `phase3_task3_training/` | 早期 ESM 序列探索；曾访问 D-SCRIPT 测试，不能当作当前主比较 |
| `phase3_task3_protocol/` | 任务三开发划分和结构协议 |
| `phase3_task3_structure_audit/` | 任务三结构匹配与审计，约 1.1G |
| `phase3_task3_features/` | 任务三结构特征，约 8.9G |
| `phase3_task3_structure_training/` | 任务三结构主比较训练 |
| `phase3_task3_summary/` | 任务三各历史版本汇总，正式开发比较使用 `structure_v2/` |
| `phase3_task3_local/` | T3-LOCAL 训练与预检；12 个组合已完成 |
| `phase3_task3_local_exchange_audit/` | 277006/277015 输出分数交换审计 |
| `phase3_task3_local_evidence/277039/` | 表示级交换失败证据；任务尚未验收 |

主要报告（集群完整保留，本地可能只有回传子集）：

- `phase3_task1_summary/20260927/report.json`
- `phase3_task2_summary/20260928/common_mask_v2/report.json`
- `phase3_task3_summary/20260928/structure_v2/report.json`
- `phase3_task2_sym_training/summary_v1.1_light/`：T2-SYM 完成证据
- `phase3_task3_local_evidence/277039/report.json`：T3-LOCAL 失败状态

`reports/` 和 `review_back/` 提供可随 Git 保存的文本证据与哈希入口。新作业写入新的 `runs/<experiment>/<date-or-job>/`，保留输入、模型、源码哈希和实际资源。不得直接覆盖已有 job 目录或自动删除失败运行。
