# Review back：SRPPI-EVID-001 v1.0

## 1. 任务信息与完成状态

| 项目 | 内容 |
|---|---|
| 任务 ID | `SRPPI-EVID-001` |
| 版本 | `v1.0` |
| 任务单 | `/home/zy/projects/Surface-residue/plan/task_plan_SRPPI-EVID-001_v1.0.md` |
| 主机 | 本地 Linux 工作站 `user-ThinkStation-P3-Tower`；远端只读查询节点 `mgt01` |
| 本地项目 | `/home/zy/projects/Surface-residue` |
| 集群项目 | `/datapool/home/zhangye/projects/surface-residue-ppi` |
| 本地开始时间 | `2026-09-28T12:47:21.234203+08:00` |
| 本地结束时间 | `2026-09-28T13:00:44+08:00` |
| 本地解释器 | `/home/zy/miniconda3/bin/python`，Python 3.13.13 |
| 依赖 | 仅 Python 标准库；未安装依赖，未导入模型 |
| 执行模式 | 现有文档/元数据/字节哈希只读核查；公开固定 commit 的轻量 URL 读取；远端文件/hash/git 元数据只读查询 |
| 状态 | **完成：补证范围内证据已整理；核心未知项保留为未知** |

本任务的依据文件哈希全部匹配任务单中给出的版本锚点。实际哈希和路径见附件 `SRPPI-EVID-001_v1.0/version_anchors.json`。任务单本身在本轮开始时的 SHA256 为 `479a1def993a338352ce2668f0693b56d0463e36adfaf9bd7be50cc1459da30a`。

本轮只新增本回执及 `review_back/SRPPI-EVID-001_v1.0/` 内的小型证据附件。没有修改代码、配置、数据、缓存、checkpoint、历史报告或共享环境。

### 要求字段速览

- **完成内容：** 核对了目标权重当前字节身份、官方固定 commit/tree/blob、模型资产加入提交、官方训练代码与参数示例、历史 dMaSIF PDB 交集审计的统计单位和范围，并检索当前 `expanded_v1` 子集是否存在已有重叠结果。
- **证据：** 见 `evidence_index.tsv`、`checkpoint_provenance.md`、`overlap_scope.tsv` 及其引用的本地报告、集群只读查询和官方固定 commit 附件。
- **偏差：** 没有找到目标权重绑定的原训练配置、训练成员或当前子集交集；因此不能把 `training_ppi.txt` 升级成实际训练成员，也不能把历史 PDB 交集解释成当前监督泄漏。官方原始下载交易记录也未找到。
- **问题：** 权重来源资产身份可靠，但实际监督成员、训练时过滤/split 和当前子集重叠仍未知；D-SCRIPT 测试历史访问限制仍然有效。
- **待决事项：** 是否接受 provenance unresolved risk 并保守使用现有开发结果；若要形成独立评测结论，是否另行批准带完整 provenance 的 checkpoint 或新的独立评测/成员审计任务。

## 2. 两个核心问题的直接答案

### 2.1 目标权重能否与可信来源、发布版本、原训练配置和训练成员建立完整追溯链？

**答案：只能完成“当前字节 → 官方仓库固定资产”的关联；不能完成“权重 → 原训练配置 → 原训练成员”的完整关联。**

【产物已核验】集群当前目标文件：

`/datapool/home/zhangye/projects/surface-residue-ppi/third_party/dMaSIF/models/dMaSIF_search_3layer_12A_16dim`

- 文件大小：469,296 bytes。
- 当前 SHA256：`9451669d97cf2fcaeecf594e8741a529f28b7db3b2d6ed1419271b748ae6c798`。
- 当前 Git blob SHA-1：`9dae9e03760f1e020c390f4cfa42642d3ad2c7b7`。
- 集群源码 HEAD：`0dcc26c3c218a39d5fe26beb2e788b95fb028896`。
- 官方固定 tree 返回同一路径、同一 blob、同一大小。
- 官方 API 显示该模型文件加入提交为 `60a472967c0413443052ae5f17e84130d0b18b31`，提交说明 `Added model`，文件状态为 `added`。

因此，**模型文件身份和官方 dMaSIF search 资产身份已可靠对应**。该结论不等于获得了原作者实际训练日志。

【待确认】没有找到绑定目标 SHA256 的原训练命令、完整参数快照、原始数据 manifest、过滤后成员文件、训练/验证索引、optimizer 运行日志或原训练 commit。`third_party/dMaSIF/surface_data/processed/training_pairs_data_ids_ppi.npy` 和 `training_pairs_data_ppi.pt` 在集群当前项目路径不存在；这只是两个直接相关产物不存在，不是对整个集群文件系统的穷尽证明。

### 2.2 已有重叠审计覆盖什么范围？是否已有当前 4,861/467 复合物结果？

**答案：已有审计只覆盖 dMaSIF 仓库列表与 DIPS-Split 发布全集的 PDB-code 交集；没有找到当前 `expanded_v1` 4,861 train / 467 validation 复合物的独立交集结果。**

【产物已核验】历史审计使用 DIPS train/val/test 发布全集，统计单位为 PDB code 和“含任意共享 PDB 的发布记录数”。`training_ppi.txt` 的结果为：

| DIPS split | 发布记录分母 | 可用唯一 PDB 分母 | 共享 PDB 数 | 含共享 PDB 的记录数 |
|---|---:|---:|---:|---:|
| train | 87,303 | 28,291 | 2,810 | 10,902 |
| val | 31,050 | 3,571 | 230 | 1,667 |
| test | 15,268 | 3,610 | 129 | 980 |

同一审计还记录了其他仓库列表与完整发布 split 的 PDB 交集：

| 列表 | train PDB/记录 | val PDB/记录 | test PDB/记录 |
|---|---:|---:|---:|
| `training.txt` | 1,946 / 7,407 | 96 / 1,169 | 66 / 615 |
| `testing_ppi.txt` | 477 / 1,860 | 28 / 424 | 25 / 106 |
| `testing.txt` | 279 / 957 | 9 / 29 | 10 / 68 |

【待确认】没有当前 4,861/467 子集的 PDB、chain/protein、interface、序列同源或逐项成员交集结果。附件 `overlap_scope.tsv` 将这些行明确标记为 `not_available`，没有把缺失值填成 0。

## 3. 来源链和证据层级

来源链为：

```text
历史 feature 报告中的 dMaSIF SHA256
  → 集群当前 469,296-byte 权重文件
  → 官方固定 commit/tree 中同一 Git blob
  → 2021-08-31 的 “Added model” 资产加入提交
  → search 任务和 benchmark 示例的配置线索
  → 原训练参数、成员 manifest、过滤后 IDs：缺失
```

官方固定 commit 的源码和小型出处附件：

- `official_README.md`：说明 dMaSIF search/site 用途、训练/推理入口和许可证。
- `official_main_training.py.txt`：显示原始代码读取 `surface_data`、`training=True/False` 数据、`random_split`，并按 validation loss 保存 checkpoint。
- `official_data.py.txt`：显示读取 `training_ppi.txt`、缺失预处理文件跳过、`iface_valid_filter` 过滤，以及保存 processed IDs。
- `official_Arguments.py.txt`：显示若未传参时 `emb_dims=8`、`n_layers=1`、`radius=9`、`seed=42`、`validation_fraction=0.1` 等默认值。
- `official_search_benchmark.sh.txt`：显示官方 search 示例使用 3 layers、12 Å、`--search True`，但没有显式给出目标 16 维 checkpoint 的完整训练参数。

这些文件只能支持“官方代码和示例如何工作”的判断，不能证明目标权重正是按默认值或 benchmark 命令训练。尤其是当前项目实际使用 16 维 embedding，而官方 `Arguments.py` 默认是 8 维；这是原训练配置未复原的证据，不能单独解释为错误。

完整证据索引在 `SRPPI-EVID-001_v1.0/evidence_index.tsv`；来源链详述在 `SRPPI-EVID-001_v1.0/checkpoint_provenance.md`。

## 4. 审计范围和重叠层级

历史审计脚本的实际语义是：从 LMDB ID 中提取 PDB code，转换为小写集合，用集合交集计算 unique PDB 数；记录数则按每个 LMDB ID 是否包含任意共享 PDB 计数。它没有计算：

- checkpoint 实际优化过的成员；
- chain/protein 精确交集；
- interface 或 residue label 交集；
- 序列同源或远缘同源重叠；
- 当前筛选后子集的交集。

历史报告明确写明：`Repository list PDB overlap only; checkpoint membership and sequence homology remain unverified.`

当前 task1/task2 协议的身份锚点是：

- candidate 选择报告：5,000 train / 500 val 候选，90% identity、80% coverage 聚类，train/val cluster overlap 0；
- 结构与有效性准备后：4,861 train / 467 validation complexes，失败 0，质量排除保留在协议报告；
- 这些报告没有 dMaSIF training-member overlap 字段，已有审计目录中也没有找到对应当前子集的成员结果。

因此，当前证据只支持：**存在与 dMaSIF 仓库训练列表相关的 PDB 风险线索；无法判断目标权重是否真正监督过这些条目，更无法判断当前选中子集的实际监督重叠。**

## 5. 历史测试访问边界

- 本轮没有读取 DIPS/D-SCRIPT 测试结构、标签、预测或性能；没有新增测试访问。
- DIPS 历史审计读取过 test split 的元数据、索引和计数，但记录说明未读取测试结构/标签/性能。
- D-SCRIPT test 的历史访问已经在 `reports/phase3_resume_corrections_20260928.json` 中记录：早期序列 pilot `276306` 和 subset preparation 访问了官方 test；已知 unique official test pairs 为 1,000，另有 192 个 strict subset pairs。这里不将两者简单相加，因为 strict subset 是嵌套范围。
- 本轮不重新打开这些测试产物，也不把历史计数改回 0。

## 6. 证据状态表

| 事实 | 状态 | 证据 |
|---|---|---|
| 当前集群权重的 SHA256 | 本轮直接核验 | `cluster_identity.json`、集群 `sha256sum` |
| 当前权重与官方固定 tree/blob 相同 | 本轮直接核验 | `official_tree.json`、`cluster_identity.json` |
| 权重为 dMaSIF search 资产 | 本轮直接核验/官方来源 | 固定源码、模型文件名、search benchmark |
| 权重加入官方仓库的提交 | 本轮直接核验 | `official_checkpoint_commit_detail.json` |
| 本项目历史特征摘要使用相同权重哈希 | 本轮直接核验历史报告字段 | task1/task3 feature summary、smoke summary |
| 原训练配置 | 未知 | 未找到绑定 args/config/log；只能引用官方默认值和示例 |
| 原训练 commit | 未知 | 当前源码 commit 不是权重训练 commit |
| 原训练成员 | 未知 | 没有成员 manifest/processed IDs/原日志绑定权重 |
| `training_ppi.txt` 是候选/仓库列表 | 已证实 | 固定源码和官方文件存在 |
| `training_ppi.txt` 是目标 checkpoint 实际优化成员 | 未证实 | 没有权重到列表的训练运行绑定 |
| 发布全集 PDB 交集 2,810/230/129 | 历史报告，本轮直接读取报告字段 | `runs/atom3d_audit/.../report.json` |
| 当前 4,861/467 子集交集 | 缺失 | bounded search 未发现；标记 `not_available` |
| 序列同源重叠 | 未检查/未知 | 任务明确禁止本轮新同源审计 |
| D-SCRIPT 历史 test 访问 | 已证实历史偏差 | `reports/phase3_resume_corrections_20260928.json` |

## 7. 冲突、缺失和限制

### 已发现的证据差异

1. 官方 `Arguments.py` 默认 `emb_dims=8`，本项目使用 16 维 dMaSIF 输出。没有原训练配置，不能把官方默认值当作 checkpoint 配置。
2. 官方训练代码默认从 `surface_data`/MaSIF Zenodo 入口读取数据；当前项目下游使用 ATOM3D DIPS。两者的源码入口不同，不能据此推定训练数据重叠，也不能据此排除重叠。
3. `reports/environment_275816.json` 的整体作业状态是 failed，虽然其中记录了 dMaSIF checkpoint top-level keys；不能把该作业写成成功加载验证。成功 smoke 的摘要是另一个历史证据 `reports/smoke_275819_summary.json`。
4. `reports/source_snapshots/local_audit_20260924/audit_atom3d.py` 与实际 `local_20260924_workstation_01/report.json` 中记录的执行脚本 SHA256 不同；本轮没有合并两份版本。历史执行版本以 report 内 `source_sha256` 为准，快照只作为早期版本记录。

### 缺失项

- 目标权重的原始下载 URL、下载日志和完整来源交易记录；当前只建立了固定官方仓库资产对应。
- 与目标权重 SHA256 绑定的原始训练命令、args/config、seed、数据包哈希、过滤统计、训练/验证 IDs 和训练日志。
- `training_ppi.txt` 到目标权重实际优化成员的绑定证明。
- 当前 4,861/467 子集的 dMaSIF PDB/chain/interface/sequence-homology 交集。
- 完整序列同源审计和 checkpoint 监督重叠审计。

## 8. 本轮实际操作和资源边界

【产物已核验】本轮执行内容：

- 读取并核对适用 `AGENTS.md` 和任务单版本锚点。
- 读取已有 dMaSIF、ATOM3D、Phase 3 数据协议和偏差更正报告的元数据/文本。
- 对目标 checkpoint 和 `training_ppi.txt` 做本地/集群 SHA256 与 Git blob 只读核对。
- 只读查询集群时间、主机、Git HEAD/tree/hash-object、目标文件状态；未读取凭据。
- 通过 `urllib` 轻量读取官方固定 commit 的 GitHub API/raw 文本；保存 URL、访问时间和响应 SHA256。
- 对有限范围的既有报告/文档/源码快照做字符串检索；搜索结果明确记录在 `scope_search.json`。
- 保存小型来源摘录、历史交集摘录、证据索引和范围表。

没有做的事情：

- 没有 `torch.load`、模型导入、推理、训练、特征提取或 smoke test。
- 没有读取结构、标签、特征数组、预测分数或新的测试索引。
- 没有提交、取消或重启 Slurm 作业；新增 Slurm 预算为 0，GPU 使用为 0。
- 没有安装依赖、下载模型、大型数据或替代 checkpoint。
- 没有计算新的 PDB 交集、序列同源交集、指标或新的逐样本结果。

## 9. 验收标准逐项回应

1. **两个问题逐项回答：通过。** 见第 2 节；未知环节明确列出。
2. **权重身份、官方来源和训练成员分开：通过。** 当前字节身份已核验；训练成员仍 unknown。
3. **历史发布全集和当前子集分开：通过。** 2,810/230/129 只写为发布全集 PDB 交集；当前 4,861/467 标记 `not_available`。
4. **已有成员产物引用：通过。** 未发现目标权重绑定的成员产物，没有从汇总数字编造成员名单。
5. **测试访问边界保留：通过。** 本轮新增测试访问为 0；D-SCRIPT 历史访问明确保留。
6. **无新实验或修改：通过。** 没有作业、模型执行、数据/配置/缓存修改。
7. **影响下一步判断并停止：通过。** 结论和候选决策见第 10 节；本任务在此停止。

## 10. 对研究决策的影响和候选下一步

【分析或假设】当前最稳妥的科学表述是：dMaSIF checkpoint 的文件来源和身份可靠，但监督成员资格不可追溯；已有 PDB 交集证明存在风险线索，不足以证明泄漏，也不足以证明没有泄漏。任务一/二的开发增益因此不能被解释为完全独立于 dMaSIF 预训练监督。

候选下一步仅供决策 agent 选择，不是执行指令：

1. **接受当前补证结论，保守限定开发结果。** 将 dMaSIF 监督重叠写为 unresolved risk；不把现有开发结果升级为无泄漏正式结论。成本最低，无新实验依赖。
2. **若必须形成独立评测结论，先要求可追溯 checkpoint 或新独立评测轨道。** 需要新的权重 provenance，或按已知 PDB/序列风险设计独立数据和成员审计；这是新任务，未批准。
3. **若继续诊断，先由用户/决策 agent 明确是否接受该 provenance 风险。** 任务二/三机制诊断不会解决 checkpoint 训练成员未知这一根本缺口，不能代替来源补证。

## 11. 附件清单

| 附件 | 用途 |
|---|---|
| `version_anchors.json` | 任务单列出的 9 个文件版本锚点、实际哈希和主机/解释器记录 |
| `evidence_index.tsv` | 29 条结论/出处/版本/哈希/定位/状态记录 |
| `checkpoint_provenance.md` | 目标权重来源链、官方源码限制和成员缺口 |
| `overlap_scope.tsv` | 14 条发布全集/当前子集审计范围记录；缺失项使用 `not_available` |
| `historical_overlap_excerpt.json` | 从已有 ATOM3D 审计报告复制的历史计数摘要；没有重新计算 |
| `scope_search.json` | 有界搜索的目录范围、文件数、命中和缺失声明 |
| `cluster_identity.json` | 集群目标文件的 SHA256、Git tree/blob、HEAD、时间和只读命令记录 |
| `cluster_member_artifact_lookup.json` | 两个直接相关 processed member 文件的定向存在性查询 |
| `official_access.json` | 官方固定 commit tree、README、训练代码、数据代码和参数文件访问记录 |
| `official_followup_access.json` | 官方 search benchmark 和模型路径历史查询记录 |
| `official_commit_access.json` | 官方模型资产加入提交的 API 访问记录 |
| `official_*` 文本摘录 | 官方固定 commit 的小型出处文件，不包含模型二进制 |

**回执状态：完成，等待用户与决策 agent 审阅。**
