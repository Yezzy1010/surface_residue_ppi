# SurfaceResidue-PPI

SurfaceResidue-PPI 比较冻结的 ESM2、GearNet 和 dMaSIF 表征，以及表面点到残基的 A/B/C 融合方式。模型训练、推理、特征提取和 smoke test 必须通过 Slurm；ATOM3D CPU 数据审计是用户明确授权的工作站例外。

本 README 同时是工作站和集群项目目录的导航。工作站目录为 `/home/zy/projects/Surface-residue`，集群项目路径为：

```text
/datapool/home/zhangye/projects/surface-residue-ppi
```

## 当前阶段

目录整理基线：2026-10-14。整理前集群目录约 87G，其中 `data/` 约 40G、`runs/` 约 48G。大文件是可复用数据和实验产物，不属于源码提交内容。

- Phase 3 任务一和任务二已有开发验证结果，详情见 `docs/phase3_task1_diagnostics_20260928.md`、`docs/phase3_task23_main_20260928.md` 和对应 `runs/` 报告。
- Phase 3 任务三结构主比较已有训练和汇总结果，详情见 [任务二、三结果](docs/phase3_task23_main_20260928.md)。
- T2-SYM v1.1 已完成 3 seeds × 3 variants，9 个 checkpoint 严格加载和交换检查通过，详见 [完成回执](review_back/review_back_SRPPI-T2-SYM-001_v1.1.md)。
- T3-LOCAL v1.0 的 12 个训练组合和输出分数交换检查已完成，但补证作业 `277039` 的 ES-LOCAL 768 维表示交换检查失败；该任务没有通过最终验收，详见 `review_back/review_back_SRPPI-T3-LOCAL-001_v1.0.md`。
- 测试集边界和历史偏差以 `docs/resume.md`、`reports/phase3_resume_corrections_20260928.json` 为准。不能把开发验证结果表述为正式测试性能。

## 目录约定

```text
.
├── README.md                         # 项目入口和目录导航
├── AGENTS.md                         # 执行约束，优先级高于一般说明
├── SurfaceResidue-PPI_事实交接包_20260928.md # 阶段事实快照，任务单按根路径引用
├── SurfaceResidue-PPI_最终执行方案与集群约束.md # 正式方案，保留兼容路径
├── 集群使用规范.md                    # 集群规范，保留兼容路径
├── .gitignore                        # 排除外部数据、缓存、日志和大实验产物
├── docs/                             # 方案、数据边界、阶段记录、续跑说明
├── plan/                             # 版本化任务单，定义范围和验收标准
├── scripts/                          # 可复用 Python 入口和汇总脚本
├── slurm/                            # sbatch 作业入口；按 hpc/GPU 分区执行
├── src/surface_residue_ppi/          # 可复用项目源码和数据审计模块
├── tests/                            # 单元测试和不依赖模型的工程检查
├── configs/                          # 实验配置；本地路径配置不入 Git
├── envs/                             # 环境版本记录和安装说明
├── reports/                          # 小型校验记录、提交记录和 provenance
├── review_back/                      # 任务回执；大 payload 保留在集群但不入 Git
├── data/                             # 外部数据、缓存、mmCIF 和任务输入（不入 Git）
├── runs/                             # 按实验和 job 保存的结果、checkpoint、特征（不入 Git）
├── logs/                             # Slurm 和审计日志（不入 Git）
├── tools/                            # 第三方工具和安装包（不入 Git）
├── third_party/                      # 第三方源码（只读，不入 Git）
└── archive/                          # 历史根目录副本和整理清单
```

## 数据与实验产物

`data/atom3d/` 保存 [ATOM3D PPI-DIPS-split](https://zenodo.org/records/4911102) 发布包和 LMDB；`data/rcsb/` 保存 RCSB seqres 和 mmCIF；`data/task3/` 保存任务三输入；`data/cache/` 保存运行缓存。历史 D-SCRIPT 输入已从工作站根目录移入工作站的 `data/legacy/dscript/`；20 个根目录 mmCIF 已从集群根目录移入集群的 `data/structures/root_inputs_20250925/`。另一端的同名目录可能为空，移动清单的 `origin` 标明文件实际所在主机；本轮没有复制这些大型数据。完整说明见 [data/README.md](data/README.md)。

`runs/` 按实验阶段组织：

- `atom3d_*`：ATOM3D 审计、残基库存、化学检查、标签诊断和序列映射。
- `phase2_*`：候选生成、特征、协议和任务一训练。
- `phase3_features/`、`phase3_protocol/`、`phase3_candidates/`：Phase 3 共享输入和协议。
- `phase3_task1_*`、`phase3_task2_*`、`phase3_task3_*`：三个任务的训练、诊断、交换检查和汇总。
- `phase3_task3_local*`：T3-LOCAL 训练、交换审计、补证和失败证据。
- `smoke/`、`phase3_smoke/`：工程 smoke 运行。

完整阶段目录和主报告入口见 [runs/README.md](runs/README.md)，原始实验路径保留。

每个新实验应使用 `runs/<experiment>/<date-or-job>/`，并在同一目录保存报告、输入哈希、模型哈希和资源信息。不要覆盖旧 job 目录，不要自动删除失败运行。

## 代码和作业

- `scripts/` 是 Python 运行入口；脚本内使用相对于项目根的路径。
- `slurm/` 是唯一的模型训练、推理、特征提取、smoke 和重型预处理提交入口。CPU 作业使用实际核验过的 `hpc` 分区，GPU 作业使用实际核验过的 GPU 分区。
- `src/surface_residue_ppi/` 放数据审计、残基标签和模型适配器等可复用模块。
- `tests/` 中的测试不替代 Slurm 运行；涉及模型的检查仍必须提交作业。
- `configs/experiments/` 保存版本化实验配置；`configs/paths.local.json` 只能作为本地未跟踪配置。

## 文档、报告和回执

- `docs/`：执行方案、集群约束、数据审计、阶段结果和续跑位置。
- `docs/reorganization_20261014.md`：本次整理的分类规则、移动范围、Git 边界和回退方式。
- 根目录的事实交接包、执行方案和集群规范保留原路径，因为版本化任务单或历史校验直接引用这些路径；方案和规范在 `docs/` 中保留已有副本，事实交接包以根目录文件为准。
- `plan/`：用户批准的任务单。执行范围、资源和验收标准以对应版本为准。
- `reports/`：小型 JSON/TSV、SHA256 清单、Slurm 提交记录和源文件 provenance。 `reports/misplaced_sync_copies_20260928/` 只将 `report.json` 纳入 Git；FASTA 和 train/test TSV 是数据快照，原文件保留但排除暂存。
- `review_back/`：任务回执及证据。Markdown、JSON、TSV、源码快照和哈希清单进入 Git，`.pt/.npz/.npy` 等二进制附件排除；原始二进制仍保留在各自主机。Git 克隆不等于完整实验数据备份。
- `archive/`：本次整理前根目录文件的保留副本。`archive/README.md` 和 `archive/manifest.tsv` 说明原路径、现路径、用途和校验信息。

## 环境和第三方依赖

`envs/` 只保存环境说明和版本清单。工作站审计使用项目专属 `.venv-audit`；共享 `gearnet` 环境和旧项目只读。`tools/`、`third_party/` 保存下载的工具或源码，不在本项目 Git 快照中提交；工作站安装包位于 `tools/packages/`。历史集群环境为 `surface-residue-dmasif`，Python 3.8.20、PyTorch 1.8.1+cu111、NumPy 1.24.4、SciPy 1.10.1（来自 275819 版本记录，本轮没有加载模型复验）。详见 [环境记录](envs/README.md)。

## Git 快照边界

集群项目位于 `/datapool/home/zhangye/projects` 这个已有的 ProteinMAE 父仓库下面；父仓库与本项目无关，不能在父目录直接执行 `git add`。本项目已经有独立的 Git 仓库，仓库根目录是 `/datapool/home/zhangye/projects/surface-residue-ppi`，当前 `master` 还没有提交。本阶段只暂存源码、配置、文档、目录说明和小型 provenance 文件，不提交、不推送：

```bash
cd /datapool/home/zhangye/projects/surface-residue-ppi
git rev-parse --show-toplevel
git status --short
git add .
git diff --cached --stat
git status --short
```

`.gitignore` 排除数据、runs、logs、tools 内容（各目录 README 除外）、第三方源码、环境、历史大归档和模型/数据二进制证据；保留轻量报告与源码快照。暂存前确认没有凭据、私钥、模型权重、缓存或外部数据进入暂存区。2026-10-14 已暂存 283 个文件、2,343,152 bytes；没有 commit 或 push，后续提交由用户明确决定。

工作站目录 `/home/zy/projects/Surface-residue` 是整理过程的本地参考副本；集群目录是本次 Git 操作目标。两端不是完整数据镜像，集群上的原始数据和实验产物继续保留在各自目录中。

## 运行入口

先阅读 [AGENTS.md](AGENTS.md) 和对应 `plan/`，再核验 `sinfo` / `scontrol show partition` 的实际资源。历史作业入口包括 `slurm/train_phase3_task1.sbatch`、`slurm/train_phase3_task2_sym_a100.sbatch` 和 `slurm/train_phase3_task3_structures_a100.sbatch`，使用 `sbatch slurm/<对应任务>.sbatch` 提交。目录整理本身不批准新的实验。T3-LOCAL 仍受 v1.0 停止条件约束，不能仅因脚本存在就运行统计或放宽容差。

## 执行边界

不得修改 `/datapool/home/zhangye/projects/GearNet`、`/datapool/home/zhangye/projects/Surface-GearNet` 或共享 `gearnet` 环境。标签和 partner 坐标必须在 encoder 输入之外；比较实验保留共同 mask 和哈希。未实际核验环境、checkpoint、加载结果和输出张量前，不宣称 dMaSIF 或 ESM2 可用。

## 整理记录

- 2026-10-14：归档集群 54 个、工作站 11 个文件，逐文件 SHA256 校验通过；整理方案、路径清单、两端源码差异见 [整理记录](docs/reorganization_20261014.md) 和 [归档清单](archive/manifest.tsv)。旧 README 原文保存在 [历史版本](docs/history/README_before_20261014.md)。
