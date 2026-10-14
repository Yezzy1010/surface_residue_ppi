## 2026-09-28 续跑状态（替代下方历史进度）

- **任务一主比较及诊断、任务二主比较、任务三结构主比较均已完成；不要重复提交已完成训练。任务二、任务三机制诊断等待用户决定。**
- 任务一诊断收尾数组 `276310` 全部完成，结果见 `runs/phase3_diagnostics_summary/20260928/main_v1/report.json`。ES-C 相对 ES 的全有效残基 AP 增益为 0.023393，95% 组件 bootstrap 区间 [0.015010, 0.031972]；容量、暴露/深度、训练与验证打乱对照相对 ES 的区间均跨 0。仅推理打乱属于敏感性证据。
- 任务二修正协议 `task2-common-mask-v2` 的数组 `276314` 三 seed 全部完成并经独立统计核验。467 个验证复合物上的宏平均 AP：ES 0.373617、ES-B 0.423972、ES-C 0.422889；ES-C−ES 区间 [0.038195, 0.061139]，ES-C−ES-B 区间 [-0.006202, 0.003504]。结果见 `runs/phase3_task2_summary/20260928/common_mask_v2/report.json`。原数组 `276301` 因负对未限制共同有效掩码而无效，原产物保留且未混入结果。这是自定义采样负例开发验证，不是官方 benchmark。
- 任务三结构审计完成：5,983 个候选 TRAIN 蛋白中 3,754 可用、2,229 排除、0 失败。原 50 个 gzip 解码失败保留证据后已仅重试失败项。冻结协议 `runs/phase3_task3_protocol/20260928/structure_v2/`：1,112 train / 194 val 对，1,843 个蛋白；蛋白与同源簇交集均为 0。使用精确全长匹配和质量过滤的 AlphaFold 独立单体；MMseqs 40% identity、80% coverage 同源簇划分。
- 任务三在相同子集比较 E/ES/ES-B/ES-C，三个固定 seed。A100 特征 smoke `276318` 已通过；旧 `276317` 的 KeOps 架构缓存失败证据保留。正式提取数组 `276324` 四片已全部完成；1,843/1,843 蛋白元数据覆盖与来源核验通过，汇总 `runs/phase3_task3_features/structure_v1/report.json`。GPU 训练 smoke `276326` 已通过四方法检查（Slurm 1分21秒），正式数组 `276327` 三个 seed 均在 A100 完成；最终汇总见 `runs/phase3_task3_summary/20260928/structure_v2/report.json`。**均已完成，不要再次提交。**
- 自动收集器 `scripts/collect_phase3_task23.py` 已完成任务二和任务三的下载、独立核验与汇总；状态 `runs/phase3_orchestration/20260928/collect_v2/state.json` 为 `both_main_comparisons_collected`。它只下载/验证/统计，不提交训练。
- 任务三结构主结果（三 seed，194 对验证集）：E AP 0.918147 / AUROC 0.853648；ES AP 0.920089 / AUROC 0.861482；ES-B AP 0.916647 / AUROC 0.856644；ES-C AP 0.916211 / AUROC 0.850614。ES 相对 E、ES-B/ES-C 相对 ES 的配对组件 bootstrap 区间均跨 0，当前子集未显示稳定的表面融合增益。
- **DIPS 测试仍封存；D-SCRIPT 测试历史上已被提前读取。** 序列基线数组 `276306` 和原协议准备曾读 D-SCRIPT 测试标签，原 reads=0 记录错误。更正见 `reports/phase3_resume_corrections_20260928.json`。旧序列结果仅为探索记录，不算结构主比较；当前 structure_v2 只读官方 TRAIN，不据旧测试性能选模型。
- 用户授权边界：先完成任务二/三主实验并汇报；两项机制诊断等用户审阅后决定。统计区间条件于已选验证 checkpoint，不代表正式测试推断。

# 续跑位置（2026-09-27）

## 历史进度：Phase 3 任务一正式训练已完成（2026-09-27）

- **从当前进度继续，不重跑 Phase 2。** 新配置 `configs/experiments/phase3_20260927.json` 固定 ES/ES-B/ES-C，三个 seed 20260927/28/29；目标候选 5,000 train / 500 val。
- 候选选择已完成：`runs/phase3_candidates/20260927/expanded_v1/report.json`。修复 MMseqs representative/member 列解析后重新核实，新候选与旧 Phase 2 的实际 train/val 簇交集均为 0。旧协议 945 train + 81 val 全部保留。
- CPU 数据审计已完成，最终 **4,861 train + 467 val = 5,328** 条，失败 0、簇交集 0。初次处理 39 分 57 秒，原来的两条网络失败在文件补齐后仅用 2.49 秒定向重试；其余 5,498 条直接续用。初次报告保存在 `report.before_network_retry.json`。质量排除 172 条：映射率低/空 mask 88、上游残基键冲突 76、序列身份检查 5、空结构 mask 3。产物 `runs/phase3_protocol/20260927/expanded_v1/`，状态 `protocol_ready`。**不要重启审计。**
- 压缩 mmCIF 预取与 PDBe 官方镜像补齐均已完成。来源、哈希和错误记录在 `runs/phase3_mmcif_prefetch/20260927/`，不能将网络失败作为质量排除。CPU 环境及依赖/源码记录为 `reports/phase3_local_environment_20260927.json`。
- GPU 检查 `276237` 已在 A40/node201 通过，耗时 1 分 7 秒，ES/B/C 梯度、预测导出和 sklearn 独立指标重算全部通过；峰值 RSS 约 1.1 GiB。不同前向分数最大差为 ES 0、B 1.79e-7、C 2.09e-7。Phase 3 显式禁用 TF32。`276234/276235/276236` 的原始检查失败证据保留，不作训练成绩。
- 冻结协议和 28 个代码/配置/作业文件已在集群 SHA256 核验通过；正式特征数组 **276238** 已完成，四个分片均通过。只提取新增 **4,302** 条；旧 `275876_final_v2` 的 1,026 条精确残基身份、标签、序列键/长度、GearNet 覆盖均已对照新协议验证后直接复用。汇总报告 `runs/phase3_features/expanded_v1/report.json`，5328/5328 条。
- 元数据合并脚本 `scripts/merge_phase3_features.py` 和配对统计 `scripts/summarize_phase3_task1.py` 已写好；对应 8 项针对性测试通过。训练 loader 已检查实际 NPZ/mask/labels。完整汇总报告为 `runs/phase3_features/expanded_v1/report.json`。
- 工作站自动衔接进程已完成：状态 `runs/phase3_orchestration/20260927/run_v1/state.json` 为 `main_comparison_complete`。它提交并等待训练数组 `276257`，实际三个 seed 作业 276258/276259/276257 均 `COMPLETED`，然后独立下载预测并重算统计。不要重提训练。
- 首次产出核对：`276238_0` 实际 JobID 276239、`276238_1` 实际 JobID 276240，报告目录分别为 `runs/phase3_features/276239_0/`、`276240_1/`。已观测新增 144+143 条、失败 0，后续以报告为准；不要假设数组任务编号等于实际 JobID。证据 `reports/phase3_feature_start_20260927.json`。5,500 个候选 mmCIF 已全部上传并通过集群 SHA256 校验，见 `reports/phase3_mmcif_cluster_verify_20260927.txt`；启动总记录 `reports/phase3_launch_20260927.json`。
- 正式训练完成摘要：`reports/phase3_task1_completion_20260927.json`；完整独立汇总为 `runs/phase3_task1_summary/20260927/main_v1/report.json`。训练集 2,317,791 个有效残基/417,360 个正例，验证集 209,314 个有效残基/38,063 个正例。AP 在全部 934 条链定义；AUROC 对 885 条双类别链定义，49 条全正链明确记为 undefined。每 seed 报告、checkpoint 和预测校验和均保留。
- 主结果：ES AP 0.43409±0.00346、AUROC 0.71184±0.00193；ES-B AP 0.45551±0.00343、AUROC 0.72987±0.00356；ES-C AP 0.45749±0.00059、AUROC 0.73180±0.00111。ES-C−ES 的组件 bootstrap 95% 区间为 AP [0.01501, 0.03197]、AUROC [0.01311, 0.02627]；ES-C−ES-B 为 AP [-0.00156, 0.00542]、AUROC [-0.00061, 0.00467]。这是验证集开发证据，未校正验证模型选择，也不是正式测试推断。
- 操作次序、科学边界和后续诊断见 [Phase 3 记录](phase3_20260927.md)。测试结构、标签、性能仍未读取。

## 已完成：Phase 2 任务一单 seed 筛选

- 冻结 backbone 的 E/ES/S/ES-A/ES-B/ES-C 训练已在 A40 作业 `275892` 完成，墙钟约 8 分钟，报告 `passed`、失败 0；使用训练集 945 个复合物/1,890 条链和严格簇隔离验证集 81 个复合物/162 条链。训练有效残基 386,094、正例 72,706；验证有效残基 32,745、正例 5,690；train/val 簇交集为 0。
- 训练使用独立 labels、冻结预提取 backbone 特征，所有归一化只在 train 拟合；`test_records_read=0`、`test_labels_read=0`。不要解读为正式测试性能或多 seed 确认。
- 指标严格按方案对 162 条验证链宏平均；AP 使用 162 条链，AUROC 有 12 条全正链无定义并排除，故使用其余 150 条链，未静默删除。pooled residue AP/AUROC 仅作为诊断。early stopping 和 checkpoint 选择使用宏平均 AP。
- 宏平均结果：E 0.4424/0.6351，ES 0.4543/0.6926，S 0.4623/0.6674，ES-A 0.4847/0.7137，ES-B 0.4849/0.7151，ES-C 0.4910/0.7155（依次为 AP/AUROC）。单 seed 的 ES-C 当前最高，只能作为下一阶段候选；不可据此宣称增益稳定。
- 首次作业 `275891` 的报告与六个 checkpoint 保留为原始证据；复核方案后发现它用 pooled residue AP 做 early stopping，不符合方案的实例宏平均主指标，因此不用于方法排序。修正后的代码测试和作业 `275892` 重新运行均已通过。
- 完整报告：`runs/phase2_task1_training/275892/report.json`；六个 checkpoint 在同目录；日志为 `logs/srppi_p2train_275892.out/.err`；集群/本地 SHA256 对照为 `reports/phase2_task1_training_275892_cluster.sha256`；摘要为 `reports/phase2_task1_training_275892.json`。
- Phase 2 单 seed 筛选已完成；2026-09-27 用户已授权启动 Phase 3，最新运行状态见本文顶部。继续保留 dMaSIF checkpoint 训练数据重叠来源不确定的限制。

集群项目：`/datapool/home/zhangye/projects/surface-residue-ppi`。
本地工作区：`/home/zy/projects/Surface-residue`。用户已授权本次 CPU 数据审计在工作站执行，然后上传产物；模型特征提取和训练仍在集群通过 Slurm 运行。

## 最新完成点：全量残基库存和标签身份诊断

- 全量 train/val 共 **118,353 条**库存扫描完成，失败 0 条，测试结构/标签未读。报告：`runs/atom3d_residue_inventory/20260924/local_20260924_residue_inventory_02/report.json`。
- 原 `_01` 扫描的 5,149 条保留并继承；该旧报告的 `running` 是原 KeyboardInterrupt 收尾逻辑遗留，真实状态见同目录 `interruption.json`。不要重启旧进程或重新扫描前缀。
- 优化后 8 进程扫描剩余 113,204 条，用时 **326.95 秒**；两段实际计算合计 **638.90 秒**，不包含开发、排队和同步时间。
- train/val 有残基键合并的记录为 **949/2,623 条**；标准非 hetero 氨基酸实例中无真正 Cα 为 **2,447/568 个**。水/配体和修饰氨基酸需进一步区分，所有计数是复合物内出现次数，不是去重蛋白。
- 插入码保留与标签重建模块已在原 20 条真实样本通过诊断；`3mg7` L144 的 18 个真实残基只有 9 个为界面正例，现已逐一分开。报告：`runs/atom3d_residue_labels/20260924/local_20260924_residue_labels_01/report.json`。
- 5 项库存/续跑测试、5 项残基标签测试通过。标签范围暂为标准非 hetero 氨基酸，状态 `diagnostic_passed_training_not_enabled`，不代表正式清洗协议完成。
- 当前没有待执行的本项目 CPU 作业。`275858` 因 `hpc` 无资源，在开始前取消；保留提交、资源、取消证据。
- 全量聚合完整；优化前 5,149 条仅保留原程序有限异常示例，其余区间有 `issues.jsonl`。不要宣称所有历史异常逐条明细完整。
- 本轮代码、报告和日志已上传至集群并通过 SHA256 校验；证据入口：`reports/residue_preparation_completion_20260924.json`、`reports/residue_preparation_files_20260924.sha256`。说明见 `docs/residue_preparation_20260924.md`。

## 当前完成点

- 独立集群环境 `surface-residue-dmasif` 已安装，Conda explicit 与 pip freeze 已导出，不重新安装。
- `275819` 的 10 条真实 EC 单链 GPU 工程 smoke 已通过，不重跑，不当作 PPI 成绩。
- 13,568,121,746 B 的 DIPS-Split 主包已下载并在本地和集群通过官方 MD5 `b6095db528e4a2e1139fddc64b6965de`。不重新下载；旧失败日志和 203 个 Range 分块保留。
- 本地 `.venv-audit` 已配置，5 项 CPU 单元测试全部通过。
- 本地审计 `local_20260924_workstation_01` 已完成，实际 **156.50 秒（2 分 36 秒）**，峰值 RSS 200,832 KiB。MD5 18.40 秒，解压 136.99 秒，元数据 0.39 秒，接触审计 0.71 秒。
- LMDB 的 train/val/test 数量为 87,303/31,050/15,268，与各自索引一致；ID 和 index 交集均为零。README 总数和比例差异仍未解释，序列同源约束未独立验证。
- train/val 各 10 条，固定 seed 20260924、按 ID 哈希且每个 PDB 至多一条。20/20 发布正接触与全部发布重原子严格 `< 6 Å` 重建一致。这是抽样检查，不是全量检查或正式训练子集。
- 修正了序列映射的链名大小写回退：存在精确链名时不再合并大小写不同的链；`4INT` 诊断重跑 20/20 通过，ambiguous=0。最新报告：`runs/atom3d_sequence_mapping/20260925/local_20250925_sequence_mapping_02/report.json`。
- 化学盘点摘要已生成：`runs/atom3d_residue_chemistry/20260925/local_20250925_chemistry_inventory_01/summary.md` 和 `summary.json`。摘要是描述性盘点，不冻结最终清洗规则；MSE 等修饰残基仍需结合 mmCIF 化学身份和原子组成判断。
- 已在工作站完成 Phase 2 任务一候选清单：`runs/phase2_candidates/20260925/local_20250925_phase2_candidates_03/report.json`。使用 RCSB `pdb_seqres.txt.gz` 和本地 MMseqs2 13-45111+ds-2，序列聚类参数为 90% identity、80% coverage；只读取 train/val ID 映射，`test_records_read=0`。
  - train 候选 1,000 条、val 候选 100 条；每个 PDB 和双链簇签名最多一次。
  - 另有严格 val 100 条，所选链簇与 train 候选没有交集；这是一条 Phase 2 诊断轨道，不能替代正式测试集。
- 120,528 个请求链中 120,457 个有 RCSB SEQRES，71 个缺失；缺失项保留在报告中，未静默纳入候选。
- 1,100 条 train/val 候选已在工作站完成逐条接触和残基身份审计：`runs/phase2_candidate_audit/20260925/local_20250925_phase2_audit_02/report.json`。完整逐条记录在 `records.jsonl`，可用于回溯。
  - 1,000 train 中 980 条、100 val 中 82 条通过当前结构 mask；其余 37 条因上游残基键合并被阻塞。没有发现发布接触重建不一致或 JSON 审计失败。
  - 当前结构 common-mask 定义为：标准非 hetero 氨基酸、恰好一个 Cα、无 altloc、坐标有限、插入码/名称/hetero 身份无歧义；界面标签只按发布 contact 的 residue key 分配。
  - 可进入特征提取的清单为 `runs/phase2_candidate_audit/20260925/local_20250925_phase2_audit_02/pilot_ready_manifest.json`。这仍是 Phase 2 工程候选，不是正式测试集。
- 最新代码、报告、20 个 RCSB mmCIF、SEQRES 压缩包和候选清单已同步到集群，SHA256 校验记录见 `reports/sequence_chemistry_sync_verify_20260925.txt`。

 - GPU 特征提取 pilot `275874` 已通过：a40/node201，train/val 各 10 条，20/20 成功。ESM2 为 480 维，GearNet 为 3072 维，dMaSIF `embedding_1` 为 16 维，峰值显存约 0.53 GiB。
 - 完整冻结协议首次作业 `275875` 在 SEQRES 预取阶段因链名大小写失败，原报告保留为失败证据。
 - 完整作业 `275876` 在 30 分 45 秒内完成 1023/1026 条；发现 SEQRES 键大小写碰撞后，定向续跑 `275880` 修正 4 条记录（其中 3 条原失败、1 条原结果使用了错误同名链）。修正版优先使用 protocol 链名、回退原始 ID 链名，并保留链大小写。
 - 最终组合缓存位于集群 `runs/phase2_features/275876_final_v2/`，报告显示 `passed`、1026/1026 records、2052 个 `.npz` 和 2052 个独立 labels 文件，`test_records_read=0`。报告已通过精确 SEQRES key/长度、缓存配对和缺失文件检查。

## 审计发现及边界

原始报告状态为 **`needs_review`**，不是数据验证全部通过：

1. `val` 的 `3mg7.pdb1.gz_1_L_3mg7.pdb1.gz_1_V`（LMDB index 12442）存在插入码残基键合并：L 链 70/70A、106/106A/106B、144/144A–144Q，V 链 105/105A。L144 合并键参与了 8 条发布正接触。不能将合并标签直接分配给每个真实残基。
2. `2ad7` 和 `4wba` 中大量“无 Cα”是水、配体等非蛋白条目。定向复核中这些链的标准非 hetero 氨基酸均有 Cα。`2ad7` A 链的钙离子原子名也为 CA，必须结合元素与残基身份判断。原审计的该统计不代表蛋白缺失残基比例。
3. 仅作诊断的标准非 hetero 氨基酸过滤使 `2ad7` 正接触键数量从 910 变为 180、`4wba` 从 178 变为 74。尚未把这个简单过滤设为正式清洗策略，未生成训练标签。
4. 固定 dMaSIF 仓库 `training_ppi.txt` 与 train/val/test 的 PDB 重叠分别为 2810/230/129 个。它是仓库列表重叠，不等于 checkpoint 精确训练成员，也不是序列同源审核。

测试集只读 LMDB metadata/id map；复制文件和字节校验不解码测试结构、标签或性能。原始审计和发布记录未修改，定向复核另存 `followup_cases.json`。

## 上传与作业状态

| Job ID | 功能 | 当前状态 |
| --- | --- | --- |
| 275826 | hpc 下载 | 已取消；本地下载替代已完成 |
| 275828 | 依赖原下载的 CPU 审计 | 已取消 |
| 275845 | 独立 CPU 审计 | `CANCELLED by 1042`，2026-09-24 18:13:05 +08:00；Elapsed 00:00:00，从未执行 |
| 275858 | 全量残基库存 | `CANCELLED by 1042`，2026-09-24 23:04:41 +08:00；Elapsed 00:00:00，从未执行；工作站续跑已完成 |
| 275875 | 完整 GPU 特征提取（旧版本） | `FAILED`；SEQRES 链名大小写预取错误，保留原报告 |
| 275876 | 完整 GPU 特征提取（首次） | `FAILED/partial`；1023/1026 完成，3 条失败，保留原报告和缓存 |
| 275880 | 定向 GPU 修正 | `COMPLETED`；4/4 记录通过，修正大小写碰撞 |
| 275891 | Phase 2 单 seed 初训（pooled AP 口径） | `COMPLETED`；原始产物保留；不用于方法排序 |
| 275892 | Phase 2 单 seed 复训（链实例宏平均） | `COMPLETED`；6/6 variants、验证通过、test 未读 |
| 276237 | Phase 3 真实缓存 GPU 检查 | `COMPLETED`；ES/ES-B/ES-C 梯度、导出和独立指标复算通过 |
| 276238 | Phase 3 新增特征提取数组 | `COMPLETED`；4 个分片、4,302 新记录，失败 0 |
| 276257 | Phase 3 三 seed 正式训练数组 | `COMPLETED`；三个 seed 作业全部通过、test 未读 |

解压数据共 14,900,185,049 B（含提取标记）、14 个文件，已在集群对照 `reports/lmdb_files_local_20260924.sha256` 全部校验通过，并于 2026-09-24 18:12:11 +08:00 将 staging 原子改名为正式 `data/atom3d/zenodo-4911102/lmdb/`。传输墙钟时间 793.94 秒（13 分 14 秒，含首次并发创建目录失败后的恢复），集群 SHA256 校验约 20 秒。审计报告、源码、环境记录及日志也已同步校验。不要重新提交下载、解压或原审计作业。

## 证据入口

- 主报告：`runs/atom3d_audit/20260924/local_20260924_workstation_01/report.json`
- 同目录：`sample_manifest.json`、`archive_members.json`、`followup_cases.json`、`source/`
- 原审计日志：`logs/atom3d_audit_local_20260924_workstation_01.log`
- 测试：`logs/local_audit_tests_clean_20260924.log`
- 复核日志：`logs/atom3d_followup_local_20260924.log`
- 工作站：`reports/local_audit_workstation_20260924.json`、`reports/local_audit_io_probe_20260924.json`
- 环境：`envs/pip-freeze-local-audit-20260924.txt`
- 解压校验清单：`reports/lmdb_files_local_20260924.json` / `.sha256`
- 集群校验：`reports/lmdb_upload_verify_local_20260924.txt`
- 作业取消证据：`reports/audit_275845_cancelled_20260924.txt`
- 本轮完成摘要：`reports/local_audit_completion_20260924.json`
- 原压缩包上传证据：`data/atom3d/zenodo-4911102/upload_verification.txt`
- 旧文档快照：`reports/source_snapshots/before_local_audit_results_20260924/`

## 下一步

1. 按原方案执行容量匹配、暴露度/深度对照和链内表面特征打乱诊断；审查 ES-C 相对 ES-B 的不确定性。
2. 正式测试仍保持封存；Phase 3 协议和模型选择冻结后才可评估。
3. 继续保留修饰/非标准残基和 dMaSIF 监督重叠来源不确定等限制。

当前已有三 seed 验证确认结果，但没有正式测试性能。训练结果不替代正式测试，仍保留 dMaSIF 监督成员未知等限制。

## 2026-09-28 续跑状态（替代前文过时进度）

- 任务一主比较已完成，勿重新提交。三个诊断收尾作业 276310 全部完成；汇总：runs/phase3_diagnostics_summary/20260928/main_v1/report.json。之前失败日志及原报告均保留。收尾复用 ES 已存表面掩码和归一化参数，只读验证缓存补推理。
- 任务一诊断全残基 AP 均值：ES 0.434092，ES-C 0.457485，容量对照 0.433424，暴露/深度 0.434799，训练与验证打乱 0.435122。后面三个对照相对 ES 的配对组件区间均跨 0；ES-C 相对各对照区间均高于 0。仅推理打乱为敏感性证据，不作单独因果证明。
- 任务二原数组 276301 的负例使用完整序列而未限制共同有效 mask，旧结果无效；有效正例映射没有受影响。已保留全部日志/权重，取消仍运行的两个 seed。修正后协议版本 task2-common-mask-v2，GPU smoke 276313 通过，正式数组 276314 三 seed A100 运行。禁止将旧数组与新结果混合。新增逐对预测和每 epoch checkpoint。
- 任务三 ESM 序列基线数组 276306 完成，但不代表表面融合主实验完成。它提前读取 D-SCRIPT 测试标签和性能；原协议准备也已读取测试标签分层，却错误填写 reads=0。这些历史文件保留，审计更正见 reports/phase3_resume_corrections_20260928.json、runs/phase3_task3_summary/20260928/sequence_v1/audit_addendum.json。DIPS 测试集仍封存；后续开发不使用 D-SCRIPT 测试表现选模型。
- 任务三结构主实验继续：只使用既有 5,000 对官方 TRAIN 子集。UniProt 参考蛋白组含 isoform FASTA 已下载校验（169,651 条参考序列）；5,983 个训练蛋白中 5,871 精确匹配。以全长序列精确匹配 AlphaFold 单体，预定 mean pLDDT>=70 且 pLDDT<50 比例<=0.30；逐蛋白可续跑审计正在工作站下载，目录 runs/phase3_task3_structure_audit/20260928/exact_v1。
- 任务三同源划分使用这些 TRAIN 序列的 MMseqs 40% identity、双向 80% coverage 聚类，官方 TEST 不参与。结构 GPU 提取 smoke 276317 已提交 A100，复用冻结 ESM2/GearNet/dMaSIF。正式主比较计划 E/ES/ES-B/ES-C，同一结构子集、同一对称头、三 seed；任务二和三机制诊断待用户审阅主结果后决定。
