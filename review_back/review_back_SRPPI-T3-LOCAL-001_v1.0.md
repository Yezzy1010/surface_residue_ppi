# review_back：SRPPI-T3-LOCAL-001 v1.0

## 任务状态

**未完成最终验收；表示级交换一致性失败，按任务单停止性能解释。本文件是受阻回执，不是成功完成证明。**

- 集群当前检查时间：2026-09-29T20:13:59+08:00；回执生成时间：2026-09-29T20:21:54.328894+08:00。
- 本地项目：/home/zy/projects/Surface-residue；本地主机：user-ThinkStation-P3-Tower；集群查询主机：mgt01。
- 集群项目：/datapool/home/zhangye/projects/surface-residue-ppi。
- 执行依据：plan/task_plan_SRPPI-T3-LOCAL-001_v1.0.md 第11节第3项及第13节。交换容差失败时停止受影响步骤、回报且不能自行改方案。
- 本次接续进行了当前状态核查、文件哈希核对、证据回传与回执归档；没有提交、取消或重启作业，没有修改实验代码、模型、容差、缓存、环境。
- 本地仅使用Python 3.13.13标准库处理JSON、文本和文件字节校验，未加载模型或计算新的模型输出。

## 完成内容

1. **12/12训练组合已运行完成。** 三个seed为20260927、20260928、20260929；每个包含ES-MEAN、ES-LOCAL、ES-B-MEAN、ES-B-LOCAL。实际训练job为276590、276592、276589，均COMPLETED、退出码0:0。
2. **12个checkpoint和12个原始验证预测的当前集群SHA256，与对应报告及本地副本一致。** 计划列出的协议、manifest、历史源码版本锚点全部匹配。
3. **12个checkpoint已在GPU job 277015中strict load，12个最终logit交换检查通过。** 本轮核对其报告、逐pair输出与当前集群文件的hash，没有重新运行模型。
4. **768维表示检查没有全部通过。** job 277039在第2个组合seed=20260927、ES-LOCAL停止；2个组合已导出细节，1个通过、1个失败，其余10个未完成表示级检查。
5. 已归档失败report、日志、2个details NPZ、common mask manifest、preflight报告、集群源码快照、12组合inventory及Slurm记录。

## 证据

### 1. 验收失败

集群原文件：/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task3_local_evidence/277039/report.json。

本地字节一致副本：/home/zy/projects/Surface-residue/review_back/SRPPI-T3-LOCAL-001_v1.0/resume_check_20260929/277039/report.json。

| 检查 | seed 20260927 ES-MEAN | seed 20260927 ES-LOCAL |
|---|---:|---:|
| strict load | 通过 | 通过 |
| 表示尺寸 | 194×768 | 194×768 |
| representation_pass | true | **false** |
| 表示AB/BA最大绝对差 | 0 | 0.0000152587890625 |
| score_pass | true | true |
| logit AB/BA最大绝对差 | 0 | 0.00000095367431640625 |
| original_score_pass | true | true |
| canonical与保存logits最大差 | 0 | 0 |

固定判据是numpy.allclose(z_ab,z_ba,atol=1e-6,rtol=1e-5)，即逐元素abs(z_ab-z_ba) <= 1e-6 + 1e-5*abs(z_ba)。representation_pass=false是这个固定判据的实际结果；全矩阵最大绝对差不能单独代替逐元素相对容差判断。失败日志：ValueError: Fixed tolerance failed; stop interpretation。

logit通过不能替代任务单要求的同一pair feature/中间表示通过。277015不检查全部768维表示，因此它与277039不矛盾。277039源码在检查前已关闭matmul及cudnn TF32，不能直接归因于首次审计的TF32漏配。证据在集群源码快照finalize_phase3_task3_local_evidence.py第29–30、102–123行。

### 2. 实现、输入与已通过的预检

- 数据为task3-structure-v2：1,112 train、194 validation（124正/70负）；1,518/325个train/val蛋白。固定common-valid mask、feature cache和训练蛋白normalization。
- Mean先逐蛋白均值再构造[u+v,abs(u-v),u*v]；Local对完整valid×valid残基对构造同形式768维phi，逐通道logsumexp(phi)-log(N)，tau=1.0，目标分块32,768对，无采样、top-k或标签筛选。
- 关系MLP均为Linear(768,128)→GELU→Dropout(0.1)→Linear(128,1)，关系头98,561参数；全部可训练参数ES两格1,008,641，ES-B两格1,209,601。
- 训练采用AdamW，lr=0.001，weight_decay=0.0001，batch=24，最多30epoch，patience=5，min_delta=0.0001，grad clip=1.0；BCE正类权重422/690，按validation AP选checkpoint。
- preflight 276583记录1,306个train+val蛋白对候选计数条目，最大单pair残基对候选数590,250；toy全量/分块最大差2.384185791015625e-7。**1,306是蛋白对条目数，不是残基候选总数。**
- 源码中交换两侧会改变分块布局与归约顺序，可能与表示差异有关；这仅为待检验假设，未执行诊断证明，不作为豁免失败的依据。

### 3. 既有逐seed指标（未验收，仅原样保留）

下表提取已有训练报告，没有本轮重算，不用于方法排序或性能解释。本轮的配对差值、汇总效应、difference-in-differences和bootstrap区间尚未生成。

| seed | cell | 原报告AP | 原报告AUROC | 状态 |
|---|---|---:|---:|---|
| 20260927 | ES-MEAN | 0.9183956774 | 0.8579493088 | 未验收 |
| 20260927 | ES-LOCAL | 0.8890617780 | 0.8099078341 | 未验收 |
| 20260927 | ES-B-MEAN | 0.9166777979 | 0.8532258065 | 未验收 |
| 20260927 | ES-B-LOCAL | 0.8936437499 | 0.8270737327 | 未验收 |
| 20260928 | ES-MEAN | 0.9188832807 | 0.8601382488 | 未验收 |
| 20260928 | ES-LOCAL | 0.8946056243 | 0.8190092166 | 未验收 |
| 20260928 | ES-B-MEAN | 0.9184355387 | 0.8578341014 | 未验收 |
| 20260928 | ES-B-LOCAL | 0.8827120295 | 0.8139400922 | 未验收 |
| 20260929 | ES-MEAN | 0.9232696286 | 0.8688940092 | 未验收 |
| 20260929 | ES-LOCAL | 0.8865915885 | 0.8125576037 | 未验收 |
| 20260929 | ES-B-MEAN | 0.9165255429 | 0.8573732719 | 未验收 |
| 20260929 | ES-B-LOCAL | 0.8954556646 | 0.8250000000 | 未验收 |

完整checkpoint路径、大小、SHA256、best epoch、epochs run与分层验收状态见checkpoint_inventory.tsv。原始验证预测在cluster_runs各目录，logit交换输出在cluster_exchange_audit/277015目录；每文件对应194个固定validation pair。

### 4. Slurm provenance

全部时间为集群Asia/Shanghai；ElapsedRaw为实测调度秒数。表内均为原会话作业，申请1GPU、8CPU、a100分区；已有报告GPU型号为NVIDIA A100 80GB PCIe。请求内存不代表实际峰值，详细TRES、MaxRSS与TotalCPU另见原文及TSV。

| Slurm ID / 实际ID | 状态/退出码 | 节点 | 请求内存 | 实际秒数 | 开始 | 结束 |
|---|---|---|---|---:|---|---|
| 276570 / 276570 | FAILED 1:0 | node101 | 96G | 51 | 2026-09-29T10:43:01 | 2026-09-29T10:43:52 |
| 276571 / 276571 | FAILED 1:0 | node101 | 96G | 395 | 2026-09-29T10:48:25 | 2026-09-29T10:55:00 |
| 276580 / 276580 | FAILED 1:0 | node101 | 96G | 282 | 2026-09-29T10:59:56 | 2026-09-29T11:04:38 |
| 276583 / 276583 | COMPLETED 0:0 | node104 | 96G | 277 | 2026-09-29T11:07:20 | 2026-09-29T11:11:57 |
| 276589_0 / 276590 | COMPLETED 0:0 | node104 | 96G | 835 | 2026-09-29T11:14:45 | 2026-09-29T11:28:40 |
| 276589_1 / 276592 | COMPLETED 0:0 | node104 | 96G | 753 | 2026-09-29T11:28:40 | 2026-09-29T11:41:13 |
| 276589_2 / 276589 | COMPLETED 0:0 | node104 | 96G | 794 | 2026-09-29T11:41:13 | 2026-09-29T11:54:27 |
| 277006 / 277006 | COMPLETED 0:0 | node101 | 96G | 285 | 2026-09-29T16:17:02 | 2026-09-29T16:21:47 |
| 277015 / 277015 | COMPLETED 0:0 | node101 | 96G | 287 | 2026-09-29T16:29:05 | 2026-09-29T16:33:52 |
| 277039 / 277039 | FAILED 1:0 | node101 | 32G | 251 | 2026-09-29T16:50:51 | 2026-09-29T16:55:02 |

当前squeue为空。当日sacct未发现srppi_t3local_summary作业，集群runs/phase3_task3_local_summary不存在。统计脚本已存在，不代表已执行统计。实际sinfo/scontrol原文保存于slurm_status_and_resources.txt；hpc为CPU分区、查询时128/128CPU已分配，本轮没有申请新资源。

### 5. 文件身份

| 文件（相对项目根） | 当前SHA256 |
|---|---|
| runs/phase3_task3_protocol/20260928/structure_v2/report.json | ffaadc6e790accf4b119676edc8bb9a22803caae4dbaad59fe5c4fbd6c136593 |
| runs/phase3_task3_protocol/20260928/structure_v2/train.json | 8b8c1c3aacb28dacb5f5c36607754c252579a33d65f3eddfc1f820b8892639cc |
| runs/phase3_task3_protocol/20260928/structure_v2/val.json | 47b7e2805f2e8313b58c440745160130c8a1e9355839cf72fc099772b743007e |
| runs/phase3_task3_protocol/20260928/structure_v2/proteins.json | b623c551a93a089b62c3d6df2d7255736f6f7368690a1a7c23ecc8b7d3892e5e |
| runs/phase3_task3_summary/20260928/structure_v2/report.json | 5ed9b3046f696696fb399c056fa1a65f5395e2419d16f47b448590bafcfad730 |
| scripts/train_phase3_task3_structures.py | 98d3c1b6106b3ef97398ccf0b95b7a31048c7643639f4ab040396ddd054b9fca |
| reports/phase3_task3_inputs_verified_20260928.json | 2da919953b8b8e26111a900bb42a2c3738db5d82bd4c9f254f434d399b2b176d |
| review_back/review_back_SRPPI-EVID-002_v1.0.md | 8ccad69ebc78ced60791b1ea34e9e977e6068917dac464365204cfebfb5e9cf2 |
| review_back/review_back_SRPPI-EVID-002_v1.1.md | 09027666c19c0f11d175edc8912d17f905806ea8f387bd3b4eb864ac7c21ec3b |
| plan/task_plan_SRPPI-EVID-002_v1.0.md | a471cae4fe7c9fa932bb2a49bdd9a3a348862165a197c5f2c09a467f9d646567 |
| plan/task_plan_SRPPI-T3-LOCAL-001_v1.0.md | 5de2f5eea600650b62514739aa3d7f1ba18f1d3826f4a737686de4f93b9058f2 |
| AGENTS.md | 03f307554c52c0d8cadec0f13b8ae8eccdd1704598acd2c08314224a6ec0341b |
| review_back/review_back_SRPPI-T2-SYM-001_v1.1.md | 4e8d9f29c90e05fa4bc60bbe087db63fb2473e73e1c1a47a3acab03d72481055 |

normalization、feature list、common-mask manifest、编码主干来源和源码hash见run_config.json及input_artifacts.tsv。preflight/audit使用的逻辑feature mapping hash与训练JSON文件字节hash因序列化不同分字段保留。本轮未重新加载主干，不对ESM2/dMaSIF当前环境可用性作新增宣称。

## 偏差

1. **训练前preflight没有实际执行交换检查。** 已读源码只有候选计数、toy全量/分块和四格前反向有限值检查。训练中及训练后先检查的是logit，表示层检查直到277039才暴露失败，不能追认完整训练前验收已通过。
2. 276570环境变量、276571分块聚合公式、276580输出形状失败为此前过程记录，全部保留。最终训练与成功preflight的源码hash相同。
3. 首次审计277006虽然Slurm成功且顶层report为passed，其6个local组合symmetry.pass_为false。历史定位为TF32设置偏差，277015修正设置后12个logit检查通过；不能只看顶层passed。
4. 277039在TF32已关闭时发生表示层失败，与首次审计区分记录。
5. 277039将依赖版本采集安排在全部组合成功之后，提前失败导致未写入完整environment；原运行精确依赖版本仍缺失，不拿本地或其他作业版本填充。
6. 本地根目录没有Git元数据。保留集群实际源码快照与hash，不能提供虚构的commit或历史Git diff。新增文件位置与源码hash可审阅。

## 问题与未完成项

- 表示级交换容差失败，触发v1.0第11节第3项及第13节的停止条件；当前标为实现验收失败，停止性能解释。
- 12组合完整表示摘要/交换证据不足：1通过、1失败、10未到达。
- 固定109组件、2,000次配对bootstrap未运行，读出效应、两种表面效应、difference-in-differences及区间未返回。
- 完整依赖版本、最终逐pair合并表、统计附件和成功验收尚缺。
- 原训练与交换报告记录test_records_read=0/test_labels_read=0；失败补证report没有完整计数字段，不能称该report独立证明了计数。其源码仅加载train/val，本轮未读取测试数据。旧D-SCRIPT test历史访问限制继续有效。
- 结果仍属developmental validation，dMaSIF原训练监督成员未知等限制不变；没有开展正式测试或扩大研究范围。

## 待决事项

需要用户/决策agent处理预设停止条件后才能继续受影响步骤。建议新任务首先限定为数值诊断：固定原checkpoint、特征、mask、顺序和容差，通过Slurm定位表示差异。如授权数学等价的数值实现修复，应使用独立版本/目录保留失败记录，明确旧checkpoint可否复用及重放一致性条件；不得默认放宽容差、事后平均AB/BA或重训12组合。

当表示/分数两层验收都满足批准要求后，再完成缺失的全量细节、hpc固定bootstrap和成功回执。本次已接续核查并补齐受阻回执；没有将未通过验收的实验标成完成。

## 附件

附件根目录：/home/zy/projects/Surface-residue/review_back/SRPPI-T3-LOCAL-001_v1.0/resume_check_20260929

- resume_status.json：机器可读状态与停止依据。
- 277039/report.json、日志及同目录details NPZ：失败原始证据。
- checkpoint_inventory.tsv：12个checkpoint和分层验收状态。
- metrics_by_seed_unaccepted.tsv：原始未验收指标。
- symmetry_status.json：logit与representation分别记录。
- slurm_jobs.tsv、slurm_status_and_resources.txt：作业与资源实查。
- version_anchors.json、input_artifacts.tsv、run_config.json、source_inventory.tsv、source_snapshot：输入及代码身份。
- /home/zy/projects/Surface-residue/review_back/SRPPI-T3-LOCAL-001_v1.0/evidence_index.tsv：全部本地附件SHA256索引。
- /home/zy/projects/Surface-residue/review_back/SRPPI-T3-LOCAL-001_v1.0/receipt.sha256：回执校验值。
