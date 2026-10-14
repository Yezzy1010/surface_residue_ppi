# 2026-10-14 目录整理方案与执行记录

## 目标与分类

把项目按代码、协议文档、运行数据、实验证据和历史归档分类，给每个部分提供入口说明，准备阶段 Git 快照。为了保留现有作业与哈希引用，沿用原有 `scripts/`、`slurm/`、`runs/` 和 `data/` 内部路径。

| 类别 | 目录 | 整理方式 |
|---|---|---|
| 当前代码 | `scripts/`、`slurm/`、`src/`、`tests/`、`configs/` | 保持入口，补齐两端缺失源码 |
| 研究协议 | `docs/`、`plan/`、`envs/`、根目录方案与交接包 | 保留原文，新增导航 |
| 文本证据 | `reports/`、`review_back/` | 小型 JSON/TSV/哈希/源码快照进入 Git |
| 运行数据 | `data/`、`runs/`、`logs/` | 保留实际内容和路径，只让目录 README 进入 Git |
| 外部依赖 | `tools/`、`third_party/`、`.venv-audit/`、`.dscript-src/` | 内容排除于 Git；不修改共享环境 |
| 旧副本 | `archive/legacy_root_20261014/` | 带 SHA256 归档，不删除 |

## 实际移动

- 工作站：11 个文件，96,354,285 bytes。D-SCRIPT TSV/FASTA → `data/legacy/dscript/`；Debian 包 → `tools/packages/`；Task3 xref → `data/task3/`；旧文档 → `archive/legacy_root_20261014/docs/`。
- 集群：42 个顶层移动项、54 个文件，74,953,298 bytes。根目录旧代码、两个审计副本目录、两个 staging 目录和日志归档；20 个 CIF → `data/structures/root_inputs_20250925/`；`scripts/` 中误放的 sbatch 和 `slurm/` 中误放的旧 Python 副本归档。
- 实测文件移动/校验：工作站 `user-ThinkStation-P3-Tower`、Python 3.13.13、0.134 秒；集群 `mgt01`、Python 3.6.8、3.579 秒。这是文件管理耗时，不是模型或整个整理任务耗时。
- [合并清单](../archive/manifest.tsv) 的 65 行分别在对应主机逐文件核验 SHA256 通过。详细报告见 `reports/reorganization_20261014/move_result_{local,cluster}.json`。
- 两端外部数据没有全量互拷；另端同名目录可能为空。`origin` 决定原文件实际归属。

## 版本与同步

集群独有的 T2-SYM 代码和 Slurm 入口回传本地；本地独有的 T3-LOCAL 汇总等入口补到集群。已有同路径文件均保持原字节。唯一既存代码差异是 `scripts/build_phase2_candidates.py`：工作站调用 `cluster_selection.read_clusters`，集群保留历史内联映射实现。未擅自合并；集群版本另存 `archive/code_variants_20261014/cluster/scripts/build_phase2_candidates.py` 并纳入文本快照。

`plan/` 与 Markdown 回执补到集群。工作站的完整回执文本附件保留在本地，集群原生实验证据仍在 `runs/`；两端不是完整镜像。`source_sync.json` 列出新增同步与 SHA256。

原 README（两端原文相同）保留为 `docs/history/README_before_20261014.md`。根目录正式方案、规范和交接包保留原路径，避免任务单引用改变。旧计划、回执和既有源码不因整理重写。

## 状态与边界

T2-SYM v1.1 已完成。T3-LOCAL v1.0 仍为受阻：277039 的 ES-LOCAL 表示级交换检查失败，后续统计未完成。本轮不改变研究验收状态。

未运行模型、提交/取消 Slurm 作业、修改共享 GearNet 路径或删除历史产物。账号队列里的 `entity_*` 作业 WorkDir 属于另一个数据集项目；上一轮把它们归属本项目的描述已更正。

## Git 与回退

集群上一级 `projects` 是无关的 ProteinMAE Git 仓库，不能在父目录暂存本项目。集群项目根已有独立 `.git`，`git rev-parse --show-toplevel` 已确认指向本项目。本阶段按 README 的边界只暂存源码、文档和轻量 provenance；不执行 commit 或 push。

Git 保留源码、版本化协议、轻量证据、目录说明、整理清单与代码差异副本；排除数据、模型二进制、环境和大历史归档。提交候选使用独立临时 Git 元数据核验，不触碰项目或父仓库索引。

集群本次暂存时额外发现 `reports/misplaced_sync_copies_20260928/` 中的原始序列及 train/test TSV 数据快照，已在 `.gitignore` 中排除；该目录仅保留 `report.json` 元数据。集群项目实际暂存 283 个文件、2,343,152 bytes；暂存清单以 `git diff --cached` 为准，旧 `git_candidates.tsv` 是整理阶段的预演记录。没有 commit 或 push。

回退单个移动时，根据 `archive/manifest.tsv` 的 origin 在相应主机核验目标 SHA256，确认原路径不存在，再从 new_path 移回 original_path。旧 README 也可从历史版本恢复。不要把旧归档覆盖到新增文件上。

## 验收

[verification.json](../reports/reorganization_20261014/verification.json) 记录最终检查：65 个移动文件全部验证；工作站 115 条、集群 122 条原有源码/配置/文档记录的内容哈希保持不变（已移动的按映射校验）。新增导航链接通过。Git 独立临时索引预演为 396 个文件、7,911,723 bytes，不含数据/权重或超过 1MB 的文件；文本凭据模式扫描无匹配。预演清单保存为 `reports/reorganization_20261014/git_candidates.tsv`，清单和验收报告本身为之后新增的两个小文件。

验证范围为目录整理、源码/文档保留与 Git 候选检查；未运行模型测试，也未重新计算整个 87G 数据缓存的哈希。
