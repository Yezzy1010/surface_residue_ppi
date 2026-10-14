# SurfaceResidue-PPI

本项目执行 SurfaceResidue-PPI 研究方案：冻结 ESM2、GearNet 和 dMaSIF，比较表面点到残基的 A/B/C 融合方式。Phase 3 任务一已完成三 seed 冻结 backbone 验证；严格 cluster-disjoint 验证上的 ES-B/ES-C 相对 ES 有稳定增益。结果仍是开发验证，不是正式测试性能；dMaSIF checkpoint 的确切监督成员仍未核实。

## 当前状态

- 2026-09-28：任务一主比较与诊断、任务二三 seed 主比较均完成。任务二 AP：ES 0.3736、ES-B 0.4240、ES-C 0.4229；ES-C−ES 的 95% 组件 bootstrap 区间 [0.0382, 0.0611]，ES-C−ES-B 区间跨 0。[任务二完整统计](runs/phase3_task2_summary/20260928/common_mask_v2/report.json)使用修正后的共同有效掩码协议，旧 `276301` 产物保留且排除。任务一详见[诊断结果](docs/phase3_task1_diagnostics_20260928.md)。
- 任务三结构主比较已完成：协议含 1,112 train / 194 val 对、1,843 蛋白，蛋白和同源簇交集为 0；E/ES/ES-B/ES-C 在同一子集比较。A100 特征提取 `276324`、训练检查 `276326`、正式数组 `276327` 全部通过。三 seed 均值 AP：E 0.9181、ES 0.9201、ES-B 0.9166、ES-C 0.9162；配对区间均跨 0，未显示稳定表面融合增益。任务二、三机制诊断待用户决定。
- 旧 ESM 序列试验提前读取 D-SCRIPT 测试标签与指标，只是探索记录，详见[偏差与修正](reports/phase3_resume_corrections_20260928.json)。**DIPS 测试仍封存；不能再称 D-SCRIPT 测试从未读取。** 当前结构主比较只用官方 TRAIN 建立开发划分。真实作业和续跑入口见[续跑记录](docs/phase3_resume_20260928.md)。
- 历史进展（2026-09-27，仅任务一）：Phase 3 任务一的数据审计、特征提取和正式三 seed 训练均已完成。最终 4,861 train / 467 val、失败 0、簇交集 0；172 条质量排除保留明细。训练数组 `276257` 的三个 seed 全部通过，934 条验证链。AP：ES 0.4341、ES-B 0.4555、ES-C 0.4575；AUROC：ES 0.7118、ES-B 0.7299、ES-C 0.7318。ES-C 相对 ES 的配对组件 bootstrap 区间为 AP [0.0150, 0.0320]、AUROC [0.0131, 0.0263]；ES-C 相对 ES-B 区间跨 0。当时 DIPS 测试仍封存，后续诊断已于 09-28 完成。详见 [Phase 3 记录](docs/phase3_20260927.md)、[完成摘要](reports/phase3_task1_completion_20260927.json) 和 [续跑位置](docs/resume.md)。
- 最新进展（2026-09-25）：修复逐残基 pandas 分组的速度瓶颈，保存前 5,149 条结果后用 8 个进程续跑，剩余 113,204 条耗时 5 分 27 秒，0 条读取/统计失败。全量库存中 train/val 分别有 949/2,623 条记录存在上游残基键合并。详见 [残基准备记录](docs/residue_preparation_20260924.md)。
- `275858` 全量库存 Slurm 作业未启动即取消；沿用用户授权的工作站 CPU 数据审计路径。新标签模块在 20 条诊断样本上保留插入码并重建接触；原始报告和官方 LMDB 均未改写。诊断标签尚未用作正式训练标签。
- 项目目录：`/datapool/home/zhangye/projects/surface-residue-ppi`
- 模型特征提取、推理、训练继续通过 Slurm；按用户 2026-09-24 授权，本次 CPU 数据审计在 Linux 工作站运行并上传集群。旧 GearNet 项目和共享 `gearnet` 环境只读引用。
- 独立环境：`surface-residue-dmasif`，由 `gearnet` 克隆后增量安装 dMaSIF/PyG 依赖。
- 作业 `275819` 已在 NVIDIA A40 上通过 10 条真实 EC 训练单链 smoke：GearNet 逐残基特征、ESM2、dMaSIF、重原子到残基映射、A/B/C adapter、共享表面点缓存、空邻域和梯度检查。
- smoke 结果：ESM2 480 维，GearNet 3072 维，dMaSIF `embedding_1` 16 维；所有样本 common-valid 覆盖输入中观察到的残基序列，尚不包括完整生物序列的缺失区段；最高报告 GPU 分配 998,311,424 bytes。
- smoke 仅作工程验证，不能称为界面预测或 PPI 性能结果。dMaSIF checkpoint 是 search 任务变体，使用 `embedding_1` 的任务来源和潜在监督重叠风险需在正式实验中保留。
- Phase 2 完整冻结协议特征已在集群 GPU 作业 `275876`/定向修正 `275880` 完成；最终组合缓存为 1,026 条记录、2,052 个 `.npz` 和独立 labels，测试结构/标签未读。单 seed 训练复核作业 `275892` 已比较 E/ES/S/ES-A/ES-B/ES-C；按 162 条验证链宏平均，最高 ES-C AP=0.4910、AUROC=0.7155（AUROC 对 150 条有两类标签的链计算）。这是 Phase 2 筛选结果，不代表多 seed 稳定增益或正式测试性能。详情见 [续跑记录](docs/resume.md) 和 `reports/phase2_task1_training_275892.json`。
- ATOM3D PPI 的 README、record 元数据和 DIPS-Split 索引已校验。13.6 GB DIPS-Split 已在本地 Linux 工作区用 16 路 Range 下载，并在本地和集群两端通过官方 MD5；正式文件已上传到集群。30.9 GB raw 包未下载。
- 原 `hpc` 下载作业 `275826` 和依赖审计作业 `275828` 已取消。工作站成果已全部上传并通过 SHA256 校验，约 14.9 GB 解压数据已放入集群正式 `lmdb/` 目录；重复审计作业 `275845` 于 2026-09-24 18:13:05 +08:00 取消，实际运行时间为零。最新续跑入口见 [docs/resume.md](docs/resume.md)。
- 工作站完整本次审计耗时 **156.50 秒**，其中 MD5 18.40 秒、解压 136.99 秒、元数据 0.39 秒、20 条接触审计 0.71 秒；峰值 RSS 196.1 MiB。该耗时不代表全量残基清洗、全量标签重建或模型训练耗时。
- 本次解压数据传输墙钟时间约 793.94 秒（13 分 14 秒，含首次目录并发创建冲突及恢复），集群 SHA256 校验约 20 秒。原主压缩包此前已上传，本次没有再次传输该压缩包。

## 目录与证据

- 方案、集群规则和 Phase 0 审计：`docs/`
- ATOM3D 发布审计：`docs/atom3d_zenodo_audit.md`
- smoke 报告：`runs/smoke/20260924/275819/report.json`
- 环境清单：`envs/conda-explicit-275819.txt`、`envs/pip-freeze-275819.txt`
- 工作站环境：`.venv-audit`，版本清单 `envs/pip-freeze-local-audit-20260924.txt`。
- 本次 CPU 审计：`runs/atom3d_audit/20260924/local_20260924_workstation_01/report.json`；定向复核为同目录 `followup_cases.json`，精确运行源码保存在同目录 `source/`。
- 全量库存：`runs/atom3d_residue_inventory/20260924/local_20260924_residue_inventory_02/report.json`；续用 `_01` 的 5,149 条已保存前缀。
- 标签重建诊断：`runs/atom3d_residue_labels/20260924/local_20260924_residue_labels_01/report.json`；`diagnostic_labels/` 仅为诊断，不是训练数据发布。
- 每次运行使用独立 `runs/<experiment>/<seed>/<run_id>/`；日志保留在 `logs/`。

## 数据边界

ATOM3D Zenodo README 将跨链氨基酸正例定义为任意重原子在 6 Å 内。20/20 抽查记录的发布正接触与全部发布重原子严格 `< 6 Å` 重建完全一致。实际记录包含水、配体及离子；这个一致性不意味着发布键已可直接作为蛋白残基训练标签。`3mg7` 还存在插入码被上游键合并的问题。必须先确定蛋白残基集合、完整身份键和清洗后的标签重建规则。索引与 LMDB 的 train/val/test 数目均为 87,303/31,050/15,268；与 README 总数及约 80/10/10 的差异仍未解释，不修改官方划分。

新审计只访问训练/验证样本的坐标和标签，测试划分只读元数据。PDB 级预训练列表重叠检查不等于序列同源排除，也不证明该 checkpoint 的准确训练成员。partner 坐标和标签不能进入 encoder；正式训练还需要单体与标签物理分离、完整序列映射、共同 mask 和逐样本接触完整性检查。

## 本次新增入口与验证状态

- `scripts/audit_atom3d.py`：主包大小/MD5 校验、安全解压、LMDB 字段/索引检查、训练/验证接触审计、预训练列表 PDB 重叠。
- `src/surface_residue_ppi/ppi_audit.py`：独立于模型的只读 LMDB reader 与完整重原子接触重建。
- `tests/test_ppi_audit.py`：严格阈值、氢排除、稀疏/稠密计算一致性、缺失正例、插入码冲突和 LMDB JSON 回读。
- `slurm/audit_atom3d.sbatch`：先运行上述测试，通过后再审计主包，不启动训练。
- `scripts/inspect_atom3d_audit_cases.py`：只读复核 3 条既有 train/val 审计记录，说明插入码冲突和非蛋白条目，不修改原审计或生成训练标签。
- 5 项 CPU 单元测试已全部通过，真实审计已运行。主报告保留 `needs_review`，不是全量数据验证通过；测试结构和标签未读。

## 执行约束

CPU 工作使用已核验的 `hpc` 分区，GPU 工作使用已核验的 GPU 分区；资源申请、Job ID、模型哈希和失败日志写入运行记录。共享旧项目、权重、数据和环境不修改。
