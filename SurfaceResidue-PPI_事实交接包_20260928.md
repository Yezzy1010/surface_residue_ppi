# SurfaceResidue-PPI 事实交接包

核查范围：本地工作区 `/home/zy/projects/Surface-residue`、集群项目 `/datapool/home/zhangye/projects/surface-residue-ppi`。
最近队列查询时间：2026-09-28 11:23（Asia/Shanghai）。当前无运行或排队作业。本轮未启动、取消、重启或修改任何实验。

标签含义：

- `【产物已核验】`：当前文件、报告或作业状态直接支持。
- `【历史记录，当前未核验】`：已有历史记录，本轮未重新复核原始运行过程。
- `【分析或假设】`：基于已核验结果的解释。
- `【待确认】`：当前证据不足，不能下结论。

## A. 决策摘要

项目要验证的核心假设是：冻结 ESM2、GearNet 和 dMaSIF 主干后，把单体序列、结构和表面信息映射到残基表示，是否能稳定改善残基界面倾向预测，并在跨蛋白残基接触和蛋白对 PPI 二分类中保留优势。表面映射比较 A、B、C：A 为最近残基硬归属，B 为受限多对多邻域，C 在与 B 完全相同的邻域上加入几何感知局部注意力。

实际工作已经超过 Phase 3 的主要开发验证部分。Phase 0 审计、Phase 1 真实样本 smoke、Phase 2 单 seed 筛选已经完成。Phase 3 任务一三 seed 主比较、任务一诊断和任务二修正协议主比较均完成；任务三作为 Phase 4 的开发性结构实验也已完成。正式独立测试、冻结 adapter 的跨任务迁移，以及任务二/任务三机制诊断尚未完成。

目前最可信的结果分任务而异。任务一中，ES-C 相对 ES 的三 seed 验证 AP 提升 `+0.023393`，组件 bootstrap 95% 区间 `[0.015010, 0.031972]`；AUROC 提升 `+0.019960`，区间 `[0.013111, 0.026272]`。容量匹配、暴露/深度和训练加验证表面打乱对照相对 ES 的区间均跨 0。任务二中，ES-B 和 ES-C 相对 ES 的 AP 增益约 `+0.0504` 和 `+0.0493`，但 ES-C 与 ES-B 无稳定差异。任务三结构子集上，ES 均值最高，ES-B 和 ES-C 相对 ES 均无稳定增益。

最重要的限制是：这些都是开发验证结果，不是正式测试结论；DIPS 测试仍未读取，但 D-SCRIPT 测试在早期序列 pilot 和 subset preparation 中曾被读取，原始计数记录错误，已单独保留并更正。dMaSIF checkpoint 的确切训练数据成员未知，存在监督重叠风险。任务二使用采样负例，任务三使用操作性随机负配对；两者都不能解释成实验确认的非相互作用。任务一和任务二使用 DIPS bound 结构，不能称作 unbound 泛化。

接手后最先需要决定：是否继续任务二/任务三机制诊断；是否接受当前开发协议作为停止点；以及是否重新设计一个不受历史 D-SCRIPT 测试访问影响的独立评测轨道。正式测试与跨任务迁移不能直接从当前结果推出。

## B. 阶段状态表

| 阶段 | 状态 | 已完成内容 | 尚缺内容 | 验收证据 |
|---|---|---|---|---|
| Phase 0 | 部分完成 | 环境、模型权重、集群规则、DIPS 数据和资源审计；独立环境建立；ESM2/GearNet/dMaSIF 可加载 | 初始 ATOM3D 审计曾为 `needs_review`；dMaSIF 训练监督成员未知；部分数据/几何审计未形成完整独立证明 | `docs/cluster_audit.md`、`docs/atom3d_zenodo_audit.md`、`reports/phase3_local_environment_20260927.json`【产物已核验】 |
| Phase 1 | 已验收 | 10 条真实 EC 单链 smoke；ESM/GearNet/dMaSIF、A/B/C、空邻域、梯度、缓存回读通过 | 全量编码器刚体不变性未完整验证 | Slurm job `275819`，`runs/smoke/20260924/275819/report.json`【产物已核验】 |
| Phase 2 | 已验收 | 945 train / 81 val 复合物的小规模任务一筛选；E、ES、S、ES-A、ES-B、ES-C 完成 | 仅单 seed，不能作为稳定增益或测试结论 | job `275892`，`runs/phase2_task1_training/275892/report.json`【产物已核验】 |
| Phase 3 | 部分完成 | 任务一三 seed 主比较；任务一诊断；任务二 common-mask-v2 三 seed 主比较 | 计划中的正式测试尚未完成；任务二仍是 sampled-negative developmental validation | `runs/phase3_task1_summary/...`、`runs/phase3_diagnostics_summary/...`、`runs/phase3_task2_summary/...`【产物已核验】 |
| Phase 4 | 部分完成 | 任务三结构主实验 E/ES/ES-B/ES-C 三 seed 完成 | 冻结 adapter 跨任务迁移、正式独立测试、机制诊断均未完成 | `runs/phase3_task3_protocol/...`、`runs/phase3_task3_summary/...`【产物已核验】 |

“已经做到第三阶段”对应执行编号 **Phase 3：增益确认**，不是科研架构中的 Stage 3。当前完成的是 Phase 3 的开发性主比较和诊断；正式测试部分未做。Stage 1/2/3 与任务一/二/三、Phase 0–4 是不同层级。当前仅能核验 Stage 1 已实现：单体独立编码、缓存和 partner-independent 输入。Stage 2、Stage 3 在现有计划材料中没有正式定义，状态为【待确认】。

## C. 实际实现与计划差异

### 编码器

- ESM2：`esm2_t12_35M_UR50D`，layer 12，输出 480 维；加载后 `eval()`、冻结、缓存。权重 SHA256：`7f21e80e61d16a71735163ef555d3009afb0c98da74c48e29df08606973cc55e`【产物已核验】。精确绝对 checkpoint 路径未在当前配置中单独记录，属于【待确认】。
- GearNet：TorchDrug `models.GearNet`，hidden dims `[512] * 6`，input_dim 21，num_relation 7，edge_input_dim 59，num_angle_bin 8，`concat_hidden=true`，`short_cut=true`。图构建使用 AlphaCarbonNode、顺序边 max distance 2、空间边 radius 10/min distance 5、KNN k=10/min distance 5；逐残基输出 3072 维。checkpoint：`/datapool/home/zhangye/projects/Surface-GearNet/checkpoints/mc_gearnet_edge.pth`，SHA256 `6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec`【产物已核验】。
- dMaSIF：源码 commit `0dcc26c3c218a39d5fe26beb2e788b95fb028896`；checkpoint：`/datapool/home/zhangye/projects/surface-residue-ppi/third_party/dMaSIF/models/dMaSIF_search_3layer_12A_16dim`；SHA256 `9451669d97cf2fcaeecf594e8741a529f28b7db3b2d6ed1419271b748ae6c798`。实际使用 `embedding_1`，16 维，同时保存 `embedding_2`。这是 dMaSIF search 变体，原训练监督成员未知【产物已核验】【待确认】。

### 表面输入与 A/B/C

实际 dMaSIF 参数包括 curvature scales `[1,2,3,5,10]`、atom_dims 6、in_channels 16、emb_dims 16、orientation_units 16、post_units 8、3 层、radius 12、resolution 1、sup_sampling 20、distance 1.05、`search=true`、`site=false`、`use_mesh=false`、`no_chem=false`、`no_geom=false`。

原子化学输入是 C/H/O/N/S/Se 六类 one-hot。没有显式电荷、partial charge、APBS 或 Poisson–Boltzmann 通道，因此不能称为“已加入电荷属性”【产物已核验】。

点—残基对应：

- 点坐标来自单体原子坐标。
- A：最近残基硬归属；`r_max=4.0 Å`；Gaussian 距离权重，`sigma=2.0`。
- B：`r_max=4.0 Å`、`epsilon=1.0 Å`；一个点可被多个满足条件的残基读取；同样使用 Gaussian 距离权重。
- C：完全复用 B 的邻域；query 为残基表示，key 为表面点表示，使用 LayerNorm、线性投影和 scaled dot product；距离偏置为固定 `-d²/(2 sigma²)`，没有学习 RBF。
- chunk size 256；没有额外的每残基点数截断。
- A/B/C 在 feature cache 中共用同一批 surface points、`point_owner`、原子坐标、`atom_to_seq` 和 edge arrays。

### 冻结、任务头和训练

- 三个 backbone 均冻结并保持 eval；训练只更新投影、adapter、任务头。该事实由 smoke、正式报告和 cache provenance 支持【产物已核验】。
- 任务一：E 使用 ESM2 480→256；ES/ES-B/ES-C 使用 ESM2+GearNet 3552→256；统一 head 为 `256→128→1`，GELU、Dropout 0.1；AdamW，lr 0.001，weight decay 0.0001；30 epoch、patience 5、min_delta 0.0001；只用 train 拟合 normalization；positive class weight 动态计算；按 validation chain-macro AP 选 checkpoint。
- 任务二：正式实现只有 ES、ES-B、ES-C；共用双线性 pair head，左右各 256→128，逐维乘积求和后除以 `sqrt(128)`；`BCEWithLogitsLoss(pos_weight=3.0)`；train 每个正例最多 3 个有效负例，validation 最多 10 个。
- 任务三：E、ES、ES-B、ES-C；逐蛋白 common-mask mean pooling；pair 输入 `[u+v, abs(u-v), u*v]`，768 维；MLP `768→128→1`；positive weight 为 train 负/正比例；batch 24，30 epoch，patience 5；最终 evaluate 检查对称性。

### 已知偏离计划

1. 正式 Phase 3 任务一只比较 ES、ES-B、ES-C，E、S、ES-A 只在 Phase 2 单 seed 筛选中运行。偏离依据是 Phase 2 筛选结果和后续用户授权；没有独立批准文件记录【历史记录，当前未核验】。
2. dMaSIF 使用现有 search checkpoint，不是 site checkpoint；原因是集群可用资产限制，训练监督成员仍未知。
3. 任务二旧数组 `276301` 负例未限制在两侧共同有效 mask，结果作废；修正为 `task2-common-mask-v2`。
4. 任务三使用 D-SCRIPT official TRAIN 的结构可获取 AlphaFold 子集，属于开发性结构验证，不是正式独立测试。
5. 早期 D-SCRIPT pilot 读取了 test labels/predictions，偏离测试封存要求；更正记录已保留。
6. 任务一诊断最终复用保存的预测和身份掩码，避免严格 `assert_allclose` 对 GPU 浮点差的误报；原失败作业未删除。

## D. 数据与评估协议

### 数据和划分

- 任务一：DIPS bound structures；4,861 train complexes、467 validation complexes、934 validation chains。训练有效残基 2,317,791，其中正例 417,360；验证有效残基 209,314，其中正例 38,063【产物已核验】。
- 任务二：同一 DIPS train/validation 来源，467 validation complexes；每个 seed 有 60,857 positive 和 608,378 sampled negative pairs。train 最多 3 个负例/正例，validation 最多 10 个负例/正例。
- 任务三：选定的 5,000 对 D-SCRIPT official TRAIN；5,983 proteins 中 3,754 通过结构质量和匹配过滤，2,229 排除，失败 0；可用 pair 2,291，其中正例 1,404。最终 train 1,112 pairs（690 positive/422 negative，1,518 proteins），validation 194 pairs（124 positive/70 negative，325 proteins）。蛋白重叠 0，同源簇重叠 0；validation 有 109 个 homology-linked components，最大 21 pairs【产物已核验】。

### 标签、mask 和负例

- 任务一正标签来自 DIPS 发布接触标注投影到残基；有效 mask 之外的残基被排除。负例是共同有效残基中的非界面残基。不能把发布标签等同于实验意义上的完整生物界面。
- 任务二正例由两侧共同有效残基映射到接触对；负例从共同有效 mask 的笛卡尔积补集采样。负例是操作性 sampled negatives，不是实验确认非接触。
- 任务三正负来自 D-SCRIPT TRAIN 的官方标签/随机配对协议；随机负配对不能解释为所有样本都经过实验确认。
- 序列/结构缺失、残基身份不一致、空 mask 均进入过滤或共同 valid mask；task2 v2 明确禁止无效位置进入负例。
- task1/task2 使用 bound 结构；task3 使用 AlphaFold predicted monomer。三者输入条件不同，不能直接视作同一泛化任务。

### 指标

- 任务一 AP：对 934 条链做 chain-macro mean；AUROC：仅对 885 条同时含正负的链计算，49 条全正链记为 undefined。
- 任务二：对 467 个 complex 进行 macro AP/AUROC；组件 bootstrap 按 homology-linked complex components 重采样。
- 任务三：对 pair-level validation 计算 AP/AUROC；组件 bootstrap 按 109 个 homology-linked components 重采样。
- checkpoint 均由 validation AP 选择；所有正式报告注明区间条件于已选 checkpoint，未做模型选择校正。

### 测试集访问

- DIPS test：当前正式任务报告均为 `test_records_read=0`、`test_labels_read=0`，仍封存【产物已核验】。
- D-SCRIPT test：早期序列 pilot `276306` 和 subset preparation 曾读取 test labels/predictions；原始计数错误，已在 `reports/phase3_resume_corrections_20260928.json` 中更正【产物已核验】。
- 因此 task3 structure-v2 虽只使用 official TRAIN，但不能把后续测试称为完全独立、未受历史访问影响的最终评测。

## E. 实验结果账本

### Phase 2 任务一单 seed

job `275892`，seed `20260925`，81 validation complexes / 162 chains；验证集 AP/AUROC：

| variant | AP | AUROC | 状态 |
|---|---:|---:|---|
| E | 0.442428 | 0.635109 | passed |
| ES | 0.454336 | 0.692570 | passed |
| S | 0.462323 | 0.667376 | passed |
| ES-A | 0.484659 | 0.713715 | passed |
| ES-B | 0.484850（报告摘要） | 0.7151（报告摘要） | passed |
| ES-C | 0.490999（报告摘要） | 0.7155（报告摘要） | passed |

证据：`/home/zy/projects/Surface-residue/runs/phase2_task1_training/275892/report.json`。旧 job `275891` 使用 pooled AP 做 early stopping，保留但不用于排序【产物已核验】。

### Phase 3 任务一主比较

正式数组 `276257`；实际 seed 作业：`276258`=20260927、`276259`=20260928、`276257`=20260929。每个作业约 25–29 分钟，三 seed 均 passed。

| variant | AP（20260927 / 28 / 29） | AP mean±SD | AUROC（20260927 / 28 / 29） | AUROC mean±SD |
|---|---|---:|---|---:|
| ES | 0.430197 / 0.436829 / 0.435252 | 0.434092±0.003465 | 0.709640 / 0.713223 / 0.712655 | 0.711839±0.001926 |
| ES-B | 0.453868 / 0.459452 / 0.453218 | 0.455513±0.003427 | 0.727120 / 0.733887 / 0.728588 | 0.729865±0.003559 |
| ES-C | 0.458121 / 0.456951 / 0.457383 | 0.457485±0.000592 | 0.731186 / 0.731130 / 0.733082 | 0.731799±0.001111 |

ES-C−ES：AP `+0.023393`，95% CI `[0.015010, 0.031972]`；AUROC `+0.019960`，CI `[0.013111, 0.026272]`。ES-C−ES-B：AP CI `[-0.001557, 0.005420]`；AUROC CI `[-0.000613, 0.004674]`。证据：`/home/zy/projects/Surface-residue/runs/phase3_task1_summary/20260927/main_v1/report.json`。

### Phase 3 任务一诊断

正式收尾数组 `276310`，三 seed 均完成；最终报告 `runs/phase3_diagnostics_summary/20260928/main_v1/report.json`。

全残基 AP mean±SD：

| 版本 | AP | AUROC |
|---|---:|---:|
| ES | 0.434092±0.003465 | 0.711839±0.001926 |
| capacity-matched ES | 0.433424±0.001604 | 0.711548±0.000411 |
| exposure/depth ES | 0.434799±0.000876 | 0.713283±0.001249 |
| train+validation permutation ES-C | 0.435122±0.001588 | 0.714662±0.000421 |
| ES-C | 0.457485±0.000592 | 0.731799±0.001111 |
| inference-only permutation ES-C | 0.378775±0.003338 | 0.662923±0.003265 |

表面有效子集上，ES AP `0.437182±0.003526`、ES-B `0.458295±0.003263`、ES-C `0.460000±0.000492`；对应 AUROC 分别为 `0.694802`、`0.713328`、`0.714730`【产物已核验】。

`276276`、`276277`、`276278` 因重算 logits 与保存值严格 `assert_allclose` 失败，GPU 浮点误差约 `1.9e-6–3.1e-6`；这些失败作业必须保留。最终 `276310` 使用已保存预测完成收尾。

### Phase 3 任务二

修正协议 `task2-common-mask-v2`；正式数组 `276314`，实际 seed 作业 `276315`、`276316`、`276314`。每个正式 seed 约 39–42 分钟，三 seed passed。

| variant | AP（20260927 / 28 / 29） | AP mean±SD | AUROC（20260927 / 28 / 29） | AUROC mean±SD |
|---|---|---:|---|---:|
| ES | 0.377347 / 0.370205 / 0.373299 | 0.373617±0.003582 | 0.830398 / 0.826405 / 0.825702 | 0.827502±0.002533 |
| ES-B | 0.426008 / 0.420428 / 0.425480 | 0.423972±0.003080 | 0.859321 / 0.855804 / 0.862652 | 0.859259±0.003424 |
| ES-C | 0.426807 / 0.419366 / 0.422493 | 0.422889±0.003737 | 0.859998 / 0.858555 / 0.861958 | 0.860170±0.001708 |

ES-B−ES：AP `+0.050355`，CI `[0.037894, 0.064792]`。ES-C−ES：AP `+0.049272`，CI `[0.038195, 0.061139]`。ES-C−ES-B：AP `-0.001083`，CI `[-0.006202, 0.003504]`。这是自定义 sampled-negative developmental validation，不是官方 ATOM3D benchmark。旧数组 `276301` 作废，不得与 v2 结果混用。

### Phase 4 任务三结构主实验

协议 `task3-structure-v2`；正式训练数组 `276327`，实际 seed 作业 `276331`、`276332`、`276327`，每个约 9–12 分钟；四个 variant、三 seed 全 passed。

| variant | AP（20260927 / 28 / 29） | AP mean±SD | AUROC（20260927 / 28 / 29） | AUROC mean±SD |
|---|---|---:|---|---:|
| E | 0.915452 / 0.922506 / 0.916482 | 0.918147±0.003810 | 0.847350 / 0.857028 / 0.856567 | 0.853648±0.005459 |
| ES | 0.920569 / 0.916398 / 0.923299 | 0.920089±0.003475 | 0.859447 / 0.869700 / 0.855300 | 0.861482±0.007413 |
| ES-B | 0.913156 / 0.915990 / 0.920796 | 0.916647±0.003862 | 0.848618 / 0.853226 / 0.868088 | 0.856644±0.010175 |
| ES-C | 0.911963 / 0.916436 / 0.920234 | 0.916211±0.004140 | 0.846889 / 0.853687 / 0.851267 | 0.850614±0.003445 |

ES−E 的 AP CI `[-0.015897, 0.020686]`；ES-B−ES `[-0.013714, 0.003377]`；ES-C−ES `[-0.019778, 0.006126]`。所有主要表面融合比较区间均跨 0。验证集阳性率为 63.92%，AP 不能与任务一、任务二直接比较。

逐样本预测已保存：

- task1：各 seed 目录下 `ES_val_predictions.npz`、`ES_B_val_predictions.npz`、`ES_C_val_predictions.npz`
- task2：`runs/phase3_task2_training/276315_20260927/` 等目录
- task3：`runs/phase3_task3_structure_training/276331_20260927/` 等目录

这些 `.npz` 可以独立重算指标【产物已核验】。

## F. 正确性检查与已知问题

| 检查项 | 状态 | 证据或限制 |
|---|---|---|
| 原子—残基—序列映射 | 已通过 | Phase 2/3 loader 核验 residue keys、SEQRES IDs、长度、mask、labels |
| 空邻域 | 已通过 | adapter 空邻域回退到 residue representation；`275819` smoke 覆盖 |
| NaN/Inf | 已通过 | feature extraction、training loader、task2/task3 report finite checks |
| B/C 邻域一致性 | 部分通过 | 代码复用相同 `BC` edge cache，smoke 通过；没有全量独立邻域 hash 报告 |
| 排列检查 | 已通过 | adapter 单元测试 |
| 刚体变换 | 部分通过 | 有 point permutation + adapter rigid-motion 单元测试；未证明完整 ESM/GearNet/dMaSIF 编码器刚体不变性 |
| backbone 冻结 | 已通过 | 正式报告 `backbone_features_frozen=true`，训练只更新 projection/adapter/head |
| cache 一致性 | 已通过 | task3 三 seed 的 1,843 个 protein feature hashes、normalization hash 相同 |
| Stage 1 partner independence | 已通过 | encoder 只读取单体；task2 partner 只用于构造标签 pair |
| 跨链几何进入 encoder | 已检查，未发现 | partner 坐标未进入 encoder |
| 联合表面或界面裁剪 | 已检查，未发现 | 表面由单体坐标生成 |
| 标签进入 encoder | 已检查，未发现 | 标签在 pair loader 中构造 |
| 任务二旧负例 mask | 失败后已修正 | `276301` 约 13% 候选含无效位置，结果作废；v2 已限制 common mask |
| 环境导入 | 历史失败后已修复 | `275816` 缺 torch_sparse；`275818` 缺 libcusparse；`275819` smoke 通过 |
| task3 KeOps cache | 历史失败后已修复 | `276317` 失败；`276318` smoke 通过 |
| D-SCRIPT test read counter | 历史记录错误 | 早期 test 已读，但原报告写为 0；更正文件已保留 |

仍存在的临时绕过：

- 诊断最终使用保存预测和身份掩码，避免 GPU 浮点差造成的严格比较失败。
- dMaSIF checkpoint 的训练监督成员没有追溯到原始训练数据。
- B/C 没有全量独立邻域 hash 审计。
- 完整刚体不变性没有验证。

## G. 工程与作业状态

### 主机、目录和权限

- 本地工作区：`/home/zy/projects/Surface-residue`
- 集群项目：`/datapool/home/zhangye/projects/surface-residue-ppi`
- 集群登录：`zhangye@10.9.9.66`，登录节点 `mgt01`
- Conda 环境：`/datapool/home/zhangye/protein-models/miniconda3/envs/surface-residue-dmasif`
- 环境记录：`envs/conda-explicit-275819.txt`、`envs/pip-freeze-275819.txt`
- 运行规则：训练、推理、特征提取、GPU smoke 使用 Slurm；CPU 审计曾获授权在 Linux 工作站执行；不修改共享 GearNet、Surface-GearNet 或共享环境。

主要依赖：Python 3.8.20、torch 1.8.1+cu111、fair-esm 2.0.0、TorchDrug 0.2.1、PyKeOps 1.4.1、RDKit 2024.3.5、torch-scatter 2.0.8。

### 代码版本

项目根目录不是 Git repository，`git status` 返回“不是 git 仓库”。因此没有可提供的 branch/commit。当前以报告中的 source/config SHA256、作业报告和文件快照追踪版本。旧代码快照位于 `reports/source_snapshots/`【产物已核验】。

### 关键脚本和配置

- 配置：`configs/smoke.json`、`configs/experiments/phase3_20260927.json`
- 特征：`scripts/extract_phase2_features.py`、`scripts/extract_phase3_task3_structures.py`
- 训练：`scripts/train_phase3_task1.py`、`scripts/train_phase3_task2.py`、`scripts/train_phase3_task3_structures.py`
- 汇总：`scripts/summarize_phase3_task1.py`、`scripts/summarize_phase3_task2.py`、`scripts/summarize_phase3_task3_structures.py`
- 诊断：`scripts/run_phase3_diagnostics.py`、`scripts/finalize_phase3_diagnostics.py`
- Slurm：`slurm/train_phase3_task1.sbatch`、`slurm/train_phase3_task2_a100.sbatch`、`slurm/train_phase3_task3_structures_a100.sbatch`、`slurm/phase3_diagnostics_a100.sbatch`

### 数据、缓存、checkpoint 和预测

- DIPS LMDB：`/datapool/home/zhangye/projects/surface-residue-ppi/data/atom3d/zenodo-4911102/lmdb`
- task1 feature cache：`runs/phase3_features/expanded_v1/`
- task3 feature cache：`runs/phase3_task3_features/structure_v1/`
- task1 结果：`runs/phase3_task1_training/`、`runs/phase3_task1_summary/`
- task2 结果：`runs/phase3_task2_training/`、`runs/phase3_task2_summary/`
- task3 结果：`runs/phase3_task3_structure_training/`、`runs/phase3_task3_summary/`
- 诊断结果：`runs/phase3_diagnostics/`、`runs/phase3_diagnostics_summary/`
- 日志：集群项目下 `/datapool/home/zhangye/projects/surface-residue-ppi/logs/`
- 汇总器状态：`runs/phase3_orchestration/20260928/collect_v2/state.json`，状态为 `both_main_comparisons_collected`

最近成功运行入口，仅列出，不执行：

- `slurm/smoke.sbatch` → job `275819`
- `slurm/check_phase3_training.sbatch` → job `276237`
- `slurm/train_phase3_task1.sbatch` → array `276257`
- `slurm/phase3_diagnostics_a100.sbatch` 和 `slurm/finalize_phase3_diagnostics_a100.sbatch` → `276310`
- `slurm/train_phase3_task2_a100.sbatch` → array `276314`
- `slurm/extract_phase3_task3_structures_a100.sbatch` → array `276324`
- `slurm/train_phase3_task3_structures_a100.sbatch` → array `276327`
- 指标重算入口：`scripts/phase3_diagnostic_metrics.py`、`scripts/summarize_phase3_task1.py`、`scripts/summarize_phase3_task2.py`、`scripts/summarize_phase3_task3_structures.py`

### 当前作业

最近一次 `squeue -u zhangye` 查询为空。现无运行或排队作业。近期已结束作业状态保留在 Slurm `sacct` 和各自 report 中。不得把历史 `COMPLETED` 作业理解为当前仍在运行。

## H. 已确认决定与待决问题

### 用户已确认的决定

- 使用集群执行正式 GPU 任务；A100 优先、A40 回退。
- 从中断位置继续，不重复 Phase 2 或已完成训练。
- 先完成任务二和任务三主实验，再决定是否做机制诊断。
- 训练、特征提取、推理通过 Slurm；CPU 审计可在 Linux 工作站执行。
- 当前交接阶段只读，不启动新实验、不修改代码、环境、配置或结果。

### 曾讨论但尚未批准

- 任务二机制诊断。
- 任务三机制诊断。
- 正式独立测试。
- 冻结 adapter 的跨任务迁移。
- 增加显式电荷相关输入。
- 更换 dMaSIF checkpoint 或追溯其原训练监督成员。

### 已放弃或作废方案

- task2 job `276301`：负例未限制共同有效 mask，结果作废。
- task3 早期序列 pilot `276306`：探索性 ESM 序列基线，split 变化且曾访问 D-SCRIPT test，不用于主比较或模型选择。
- Phase 2 job `275891`：pooled AP early stopping，不用于方法排序。
- task1 诊断早期 `276276–276278`：严格浮点重算检查失败，保留为失败证据，不作为成功诊断。

### 当前待决问题与候选下一步

1. **停止在当前开发验证，先做研究判断。** 回答：表面融合是否值得继续投入。依赖现有报告和风险审查，成本最低。
2. **若批准，补做任务二/任务三机制诊断。** 回答：任务二/三的增益是否可由容量、暴露度、结构选择偏差或表面置乱解释。依赖现有 feature cache，预计仍需 A100 三 seed 训练和汇总。
3. **重新设计独立评测轨道。** 回答：在不受历史 D-SCRIPT test 访问影响的情况下，能否获得可信最终结论。依赖新的数据协议、同源划分和用户对资源/范围的确认，成本和数据准备最高。

以上均为候选建议，不是执行指令。

## I. 交接清单

决策 agent 优先阅读：

1. `/home/zy/projects/Surface-residue/docs/SurfaceResidue-PPI_最终执行方案与集群约束.md`：研究方案、Phase 0–4 定义、A/B/C 规范和验收条件。
2. `/home/zy/projects/Surface-residue/AGENTS.md`：集群、环境、权限和安全边界。
3. `/home/zy/projects/Surface-residue/docs/phase3_resume_20260928.md`：当前续跑位置、任务二/三结果和已知偏差。
4. `/home/zy/projects/Surface-residue/runs/phase3_task1_summary/20260927/main_v1/report.json`：任务一三 seed 主结果。
5. `/home/zy/projects/Surface-residue/runs/phase3_diagnostics_summary/20260928/main_v1/report.json`：任务一诊断和失败收尾证据。
6. `/home/zy/projects/Surface-residue/runs/phase3_task2_summary/20260928/common_mask_v2/report.json`：任务二修正协议和结果。
7. `/home/zy/projects/Surface-residue/runs/phase3_task3_protocol/20260928/structure_v2/report.json`：任务三结构数据筛选、同源划分和限制。
8. `/home/zy/projects/Surface-residue/runs/phase3_task3_summary/20260928/structure_v2/report.json`：任务三结构主结果。
9. `/home/zy/projects/Surface-residue/reports/phase3_task3_inputs_verified_20260928.json`：三 seed cache、feature hash 和 normalization 一致性。
10. `/home/zy/projects/Surface-residue/reports/phase3_resume_corrections_20260928.json`：D-SCRIPT test 访问和历史偏差更正。

交接仍缺少的信息：

- dMaSIF checkpoint 原始训练数据和监督成员；
- ESM2 checkpoint 的精确绝对文件路径；
- 完整 B/C 邻域独立 hash 审计；
- 完整 ESM/GearNet/dMaSIF 编码器刚体不变性证明；
- 无历史测试访问污染的独立 D-SCRIPT 评测轨道；
- Phase 4 冻结 adapter 跨任务迁移结果；
- 项目根目录 Git branch/commit。

因此，当前可以成立的是“开发验证结果和工程实现状态”；不能成立的是“正式测试泛化结论”“已完成跨任务迁移”“dMaSIF 表面特征已包含电荷信息”或“所有表面增益都已被因果诊断证明”。

本文件是在当前只读核查范围内生成的事实交接包。未创建 `review_back.md`，因为本轮边界禁止写入其他文件。
