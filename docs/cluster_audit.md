# 集群执行审计

日期：2026-09-24。用户已明确授权到集群执行方案，并安装缺失环境。
本记录更新 `phase0_local_audit.md` 中的待授权、待指定目录等历史状态。

最新状态：工作站 CPU 审计已按用户授权完成，耗时 156.50 秒，结果 `needs_review`；全部成果上传校验完成，解压数据已放入集群正式 `lmdb/` 目录，`275845` 已取消。以下各阶段记录按发生时状态保留；当前续跑入口和作业状态以 `docs/resume.md` 为准。

后续更新（2026-09-25）：工作站全量 train/val 残基库存已完成，原 20 条样本的残基身份标签重建诊断通过。全量库存作业 `275858` 因 hpc 无资源在执行前取消，Elapsed 00:00:00。没有模型训练；本轮详细结果见 `docs/residue_preparation_20260924.md`。

## 目录与资源

- 登录方式：`ssh zhangye@10.9.9.66`；实际登录节点 `mgt01`。
- 新项目：`/datapool/home/zhangye/projects/surface-residue-ppi`，本次创建前不存在。
- 本地源码与记录镜像：`/home/zy/projects/Surface-residue`；计算执行在集群。
- 已检查目标目录各级父目录，没有发现额外适用的 `AGENTS.md`。
- 规则来源为用户提供的 `集群使用规范.md` 和 SSH 登录公告；原文与哈希保存在集群 `docs/`。
- 实际 CPU 分区是 `hpc`；`cpu` 只是原规范的示例。GPU 分区包含 `a40`、`a100`、`h800`。
- 用户 GPU 上限为 8 张；本项目当前采用一个单 GPU 验证作业，提交前检查已有使用量。
- `scontrol show partition` 显示相关分区允许所有账号/QOS，未设置默认时限；实际提交不额外猜测账号/QOS，明确申请时限与内存。
- 首个作业 `275816` 在 `node201` 的 NVIDIA A40 上确认 CUDA 可用，驱动 570.86.15，torch 运行时 CUDA 11.1，设备总显存 47,709,159,424 bytes。
- 共享 CUDA toolkit：`/datapool/software/cuda/cuda_11.1.0`。登录节点 `/usr/local/cuda` 不能作为计算节点共同路径。
- 审计时 `hpc` 的 128 个 CPU 已被四个无限时作业占用；CPU 工作不得转移到 GPU 节点执行。后续预处理应独立排队。

## 已有资产与环境

- 共享 `gearnet`：Python 3.8.20、torch 1.8.1+cu111、TorchDrug 0.2.1、torch-scatter 2.0.8、torch-cluster 1.5.9、fair-esm 2.0.0、RDKit 2024.3.5。
- 本次克隆为独立 `surface-residue-dmasif`，路径仍在规定的 Conda `envs/` 根目录；共享环境不修改。
- 克隆命令：`conda create -n surface-residue-dmasif --clone gearnet -y`。共享盘复制耗时约半小时；克隆成功退出。
- 增量安装版本见 `envs/requirements-dmasif.txt`。原始安装分两次完成，先装 PyG/PyKeOps，再补二进制扩展、BioPython 和旧接口兼容依赖。
- pip 对 TorchDrug 的 `rdkit-pypi` 依赖名有提示；实际安装是提供同一 `rdkit` 模块的 `rdkit` 发行包。以 GPU 作业的解析与加载结果判断，不能仅用该提示认定环境损坏。
- 登录节点本次两个 PyTorch 导入探测长期未返回，已精确终止本次探测进程；其他历史进程未处理。后续模型导入与运行均在 Slurm 节点验证。

## 模型来源

| 模型 | 位置或版本 | 核验状态 |
| --- | --- | --- |
| GearNet-Edge | 旧 `Surface-GearNet/checkpoints/mc_gearnet_edge.pth`，80,700,937 bytes | SHA256 与旧来源记录一致；实际 tensor 已读取 |
| dMaSIF | 官方 commit `0dcc26c3c218a39d5fe26beb2e788b95fb028896` | 官方仓库已克隆；源码未修改 |
| dMaSIF 权重 | `models/dMaSIF_search_3layer_12A_16dim`，469,296 bytes | 含 `model_state_dict` 及两套编码分支；SHA256 `9451669d97cf2fcaeecf594e8741a529f28b7db3b2d6ed1419271b748ae6c798` |
| ESM2 | smoke 配置为 `esm2_t12_35M_UR50D` | 35M 为工程起点；权重加载和 GPU 输出以 smoke 报告为准 |

GearNet SHA256：`6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec`。
dMaSIF 固定读取 `embedding_1`，同时保存 `embedding_2` 供追溯；不依 partner 选择分支。
checkpoint 没有完整训练配置，架构依 strict load 确认，几何参数来源为官方默认值、benchmark 和 checkpoint 名称，仍需保留来源限制。

## 数据与科学边界

- smoke 从旧清单中明确标记为 EC `train` 的真实单链选取 10 条，按长度分位选样；不读取旧任务标签或模型成绩。
- 该 smoke 的 ESM 序列是观察到的残基序列。完整序列、缺失区段的 token 映射是正式数据流程的后续要求。
- 没有用 EC 样本生成界面标签；反向传播使用明确标注的数值检查目标，不是任务训练或实验效果。
- 旧 DB5 pilot 数据虽存在，但不能继承为 ATOM3D 官方训练/测试协议；此前被用于开发的 DB5 也不能再次称作未见最终测试。
- ATOM3D 官方源码固定到 `4c2f3b7e9efe128791b83f03b2e8cae91e78b018`。
- 官方 `examples/ppi/dataset/prepare_lmdb.py` 保存 `atoms_pairs` 和 `atoms_neighbors`；后者由完整正接触构造。`neighbors.py` 的负例由候选笛卡尔积补集构造，示例训练 loader 随后采样。
- 该构建脚本的默认值是 CA 距离 8 Å，并同时支持 heavy 模式；不能把默认值当作发布数据的实际 cutoff，也不能直接沿用旧项目的重原子 5 Å。
- 官方下载入口是 Zenodo DOI `10.5281/zenodo.4911102`；尚未下载完整 DIPS/DB5 发布包。发布元数据、实际样本、划分、完整正接触及预训练重叠审核完成前不启动正式筛选。

## 作业证据

- `275816`：CUDA 可用；记录了两个 checkpoint 的 tensor 形状；dMaSIF 因缺少扩展依赖失败，已按错误补装。
- `275818`：三项 GPU adapter 单元测试通过；模型导入因 `libcusparse.so.11` 未在库路径而失败。后续脚本显式设置共享 CUDA 11.1 的 `lib64`。
- 每次失败保留 `logs/` 与 `runs/smoke/<seed>/<job_id>/report.json`。后续执行状态以对应作业报告为准。

## smoke 作业 275819

- 分区/节点：GPU 分区 a40，node201，NVIDIA A40。
- 状态：passed；报告：runs/smoke/20260924/275819/report.json。
- 样本：旧 EC 清单中明确标记为 train 的 10 条单链，按长度分位选取；没有读取旧任务标签或模型成绩。
- 输出：ESM2 esm2_t12_35M_UR50D 为 480 维；GearNet-Edge 为 3072 维；dMaSIF 固定使用 embedding_1，为 16 维。
- 结构检查：10/10 真实样本通过 A/B/C adapter、空邻域分支、梯度检查和缓存读回；每条样本 common_valid 等于观察到的残基数。点排列/刚体变换是同一作业中的合成 adapter 单元测试，未验证整个编码器的刚体不变性。缺失残基/异常原子的正式数据覆盖仍未完成。
- 资源：报告最高 GPU 分配约 998,311,424 bytes（约 0.93 GiB），单条最长样本 753 个残基、18,861 个表面点。
- 模型哈希：ESM2 7f21e80e61d16a71735163ef555d3009afb0c98da74c48e29df08606973cc55e；GearNet 6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec；dMaSIF 9451669d97cf2fcaeecf594e8741a529f28b7db3b2d6ed1419271b748ae6c798。
- 限制：该作业是工程 smoke，不是界面预测、PPI 二分类、训练或性能评估。

## 环境证据

- envs/conda-explicit-275819.txt SHA256：35957aac88f00ba63c4878acad039dedde24851e96edd1d5e31529ef9090e15a。
- envs/pip-freeze-275819.txt SHA256：822d3dcb9274a3e108cbacd11eb3569d530f11b7027c67bb240a621f50ac25b9。
- 运行时仍以 GPU 作业中的导入、checkpoint strict-load 和输出张量为准；登录节点 conda 查询不作为模型可用性证据。

## 只读 MaSIF 资产审计

- /datapool/home/zhangye/masif 为 2021 年官方 MaSIF 源码仓库（HEAD 2a370518e0d0d0b0d6f153f2f10f6630ae91f149），包含 TensorFlow 版 masif_ppi_search 模型文件、旧 PDB/表面数据和 Apptainer 镜像。
- 该目录存在进入本项目之前的 git 删除/未跟踪状态；本项目没有修改、清理或复用这些资产。
- 这些模型不是当前 dMaSIF PyTorch checkpoint，未接入 encoder 或训练流程；旧 MaSIF 标签/输出不能写成当前 dMaSIF 结果。

## ATOM3D 发布审计

- Zenodo DOI 10.5281/zenodo.4911102，许可证 CC-BY-4.0。
- 已下载并校验 README、record JSON 和 313,097 B 的 DIPS-Split index archive；详细 MD5/SHA256、索引行数和标签协议见 docs/atom3d_zenodo_audit.md。
- PPI-DIPS-split.tar.gz（约 13.6 GB）和 PPI-raw.tar.gz（约 30.9 GB）尚未下载；在完整数据审计完成前不启动正式训练。

## Phase 2 数据准备启动

- 已检查 /datapool 可用空间约 1.8 PB；未发现 DIPS 主包的现有副本。
- 按集群大文件下载规范创建 slurm/download_atom3d_dips.sbatch，使用 hpc CPU 分区、4 CPU、8 GB 内存、2 天时限和可续传 .part 文件。
- 下载作业：275826，当前排队等待 hpc 节点；未在登录节点长时间下载。
- 目标：PPI-DIPS-split.tar.gz，13,568,121,746 B，下载完成后必须匹配 Zenodo MD5 b6095db528e4a2e1139fddc64b6965de，再将 .part 原子改名为正式文件。
- 下载、校验和失败日志均保留；主包校验完成前不执行 LMDB 解包或训练。

## 接续审计与调度状态

- 新增 CPU 审计作业 `275828`，依赖 `afterok:275826`，hpc/4 CPU/16 GB/6 小时。执行新增的接触/LMDB 测试，再校验、解压和抽查真实发布数据。
- 2026-09-24 续跑时 `hpc` 的 128 CPU 全部被四个其他用户作业占用。`internet` 分区实际包含 node101/node201 GPU 节点，`arm` 节点也登记 GPU 资源；没有据此绕过 GPU 节点禁跑纯 CPU 任务的规则。
- 下载作业 `275826` 的已提交脚本有日志命名问题：变量被转义，应用级 report/wget 文件名会含字面量 `${SLURM_JOB_ID}`，且 report 是 key=value 文本，扩展名误写为 `.json`。Slurm 自身日志 `%j` 正常，主包大小和 MD5 校验逻辑不受影响。其原始快照保存在 `reports/download_275826_submitted.sbatch`，本次保留该排队作业，不重复下载或提交第二个下载作业。
- 磁盘上的下载模板已修正扩展名、变量展开和失败记录，增加互斥锁，已有不匹配正式包改为保留并报错。修改模板不会改变 Slurm 已保存的 `275826` 快照，后续审计直接重新校验正式包，不依赖旧文本 report。
- 所有新增源码已经同步至集群。Slurm 脚本通过 `bash -n`；新增 Python 测试与主包审计尚在排队，不能写成已通过。
- 详细状态、日志路径和从哪里继续见 `docs/resume.md`。

## 本地并行下载与集群校验

- 因 `hpc` 长时间无可用 CPU，按用户授权取消了 `275826`/`275828`，保留取消记录和旧脚本快照。
- 登录节点单连接实测约 50–60 KB/s，不能用于可靠快速下载；本地工作区 16 路 HTTP Range 下载成功，使用 203 个 64 MiB 分块，最终本地文件大小 13,568,121,746 B，Zenodo MD5 `b6095db528e4a2e1139fddc64b6965de`。
- rsync 上传到集群 `.upload.partial` 后，集群端再次计算相同 MD5 和大小，并原子改名为 `PPI-DIPS-split.tar.gz`。集群 SHA256 为 `30dfe46f749087c028d3a8dd13ec266b3a98445b71974adb4f33d8ba87ca5198`；记录见 `data/atom3d/zenodo-4911102/upload_verification.txt`。
- 新审计作业 `275845` 已提交到 `hpc`，不再依赖下载作业；下载阶段已经结束，当前只等待 CPU 审计资源。

## 工作站 CPU 审计（后续更新）

- 用户明确允许本次 CPU 审计在 Linux 工作站完成再上传。`.venv-audit` 仅服务本项目；实际 Python 3.13.13、NumPy 2.4.6、SciPy 1.18.0、LMDB 2.2.0、pandas 2.3.3，完整 freeze 已保存。
- 工作站为 i9-14900K、32 逻辑线程、125 GiB 内存；本次审计 156.50 秒，峰值 RSS 196.1 MiB。大部分耗时来自解压（136.99 秒），不是标签计算。
- 5 项 CPU 单元测试通过。主包 MD5、LMDB/index 数量和跨 split 的 ID/index 无交集检查完成；20/20 抽查正接触完全复现，但一条真实 insertion_code 键冲突使报告保持 `needs_review`。
- 进一步确认发布结构含水、配体及离子；“无 Cα”不能直接作为蛋白缺失残基统计。`3mg7` 的上游残基键合并必须在正式标签前处理，详见 `docs/atom3d_zenodo_audit.md`。
- `275845` 先设为 `JobHeldUser`，本地结果上传校验后于 18:13:05 +08:00 取消，Elapsed 00:00:00，避免重复审计。未占用 GPU 节点运行纯 CPU 任务，也未修改其他用户作业。
- 精确终止了本项目之前遗留的孤儿 wget/tee 下载进程 PID 1342871/1342872，并再次确认已退出；部分下载文件和日志保留。
- 约 14.9 GB 本地解压数据传输用时 793.94 秒（含首次并发创建目录冲突及恢复）；集群端 14 个文件全部 SHA256 通过，用时约 20 秒，18:12:11 +08:00 原子改名为正式 `lmdb/`。报告、源码、日志与环境记录同步校验完成，训练、特征提取继续使用 Slurm。
