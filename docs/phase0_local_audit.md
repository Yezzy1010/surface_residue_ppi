# SurfaceResidue-PPI Phase 0 本地审计

日期：2026-09-23  
范围：仅检查当前工作区、本机可见的旧项目和公开源码；没有 SSH 登录、没有访问远程集群、没有提交 Slurm 作业、没有下载模型权重、没有修改旧项目。

## 已确认事实

### 当前工作区

- 当前目录为 `/home/zy/projects/Surface-residue`。
- 工作区只有 `SurfaceResidue-PPI_最终执行方案与集群约束.md` 和 `集群使用规范.md`，没有 `AGENTS.md`、源码、数据、配置或 Git 元数据。
- 本机没有可用的 `sbatch`、`squeue`、`sinfo` 或 `nvidia-smi`。
- 当前 Conda 只有 `base`（Python 3.13.13，CPU PyTorch 2.13.0）；没有 `torchdrug`、`rdkit` 或 `esm`。
- 方案中列出的 `/datapool/home/zhangye/...` 路径在本机不可见，不能据此判断远程路径存在或当前用户有权访问。

### 旧项目候选资产（只读核对）

本机存在 `/home/zy/projects/Surface-GearNet`，本次没有写入。它包含：

- 官方 GearNet-Edge checkpoint：
  `checkpoints/mc_gearnet_edge.pth`，80,700,937 bytes，SHA256
  `6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec`。
- 旧项目实现的 GearNet 配置为 6 层、每层 512、`num_relation=7`、`edge_input_dim=59`、`num_angle_bin=8`、`concat_hidden=True`，逐残基输出 3072 维；这是旧项目代码的事实，不能直接替代新项目的适配审计。
- 旧 README 记录过历史集群环境：Python 3.8.20、torch 1.8.1+cu111、TorchDrug 0.2.1；还记录过 `hpc` CPU 分区和 `a40` GPU 分区。这些是历史记录，不能视为本次作业的当前授权或有效资源配置。
- 旧项目已经有 SAS 近似表面、Round 3 表征和 Round 4 DB5 pilot 产物，但没有找到 ESM2 或 dMaSIF 源码/权重。旧 SAS 产物不能写成 dMaSIF 特征，旧 Round 4 结果也不能替代新方案的 E/ES/S/ES-A/ES-B/ES-C 比较。

### 公开实现核对

- dMaSIF 官方 `model.py` 在 `dMaSIF.embed()` 中保存逐点 `P["embedding_1"]`；它先由原子坐标和原子类型在线生成伪表面，再计算曲率和原子邻域化学特征，最后执行表面卷积。因此“点生成器”与“完整表面编码器”确实是两个步骤。
- 官方仓库可见模型文件 `models/dMaSIF_search_3layer_12A_16dim`（GitHub API 标示 469,296 bytes），名称表明它是 search 任务变体；本次没有下载或验证其 state dict、训练数据、参数维度和输出角色。
- dMaSIF README 列出的旧依赖包括 PyTorch 1.4/1.6、PyKeOps 1.4.x 和 PyTorch Geometric 1.5/1.6；许可证为 CC BY-NC-ND 4.0。不能假设它可直接装入旧 GearNet 环境，也不能未经许可修改或再分发。
- ESM 官方接口支持提取 ESM-2 的 per-residue representations，但模型规模、层号、维度和权重来源必须在项目配置中固定；当前本机没有 ESM 安装或权重。
- dMaSIF 的 `atoms_to_points_normals()` 使用 `torch.randn()` 生成候选点，再做梯度迭代和网格降采样；只在调用前设置全局 seed 不能保证不同批次/进程得到同一点集。因此 A/B/C 应缓存同一次生成的点、法向、点特征和编码输出，不能让三个方案各自重新调用生成器。
- dMaSIF 生成器返回坐标、法向和 batch，没有稳定的“生成原子 ID”；原子到点的关系若用于本项目必须由独立、可复现的距离映射重建，不能把 dMaSIF 内部的近邻化学聚合误写成硬残基归属。
- `dMaSIF_search_*` 同时定义 `embedding_1`/`embedding_2` 分支并以配对 search 任务训练；把 `embedding_1` 当作单体冻结点编码需要明确记录分支、checkpoint 任务和潜在监督重叠，不能默认等价于 site checkpoint 或通用表面编码器。

## 需要修正或暂不采信的约束

1. `集群使用规范.md` 给出的 `a40`、`a100`、`cpu` 分区是示例，和旧项目历史记录（`a40`/`hpc`）不一致；在实际 `sinfo`/管理员规则核实前不能写入新作业脚本。
2. 规范一方面禁止登录节点长时间下载，另一方面又要求登录节点预下载全部外部依赖；应改为遵循集群认可的传输/下载入口，不能自动选择 `nohup` 或登录节点长下载。
3. 规范中的 `scancel -u zhangye` 会影响该用户的其他作业，与“只取消本项目明确 job ID”冲突；新项目只允许使用精确 job ID。
4. `find logs -mtime +30 -delete` 与方案中“保留失败、重试和旧日志”冲突；新项目不能自动删除运行证据。
5. “一个项目一个环境”不能覆盖 GearNet/dMaSIF 版本冲突。更安全的默认是只读复用 GearNet，并在得到授权后为 dMaSIF/ESM2 建独立环境，使用缓存文件交接。
6. 集群规范的 Slurm 模板没有确认账号、QOS、时限、内存、GPU 型号或作业并发限制，也没有保证 `logs/` 在 Slurm 打开输出文件前存在；这些都必须由实际集群信息填充。
7. 旧项目已存在且有大量历史产物，不能把它当作新项目空目录，也不能把旧 SAS、旧任务标签或旧资源配置直接迁移成新方案事实。

## 当前阻塞项

在进行远程只读审计、创建集群项目、安装环境、下载 checkpoint 或提交作业之前，仍缺少：

- 新项目的绝对目标目录（是否使用当前 `/home/zy/projects/Surface-residue` 仅凭当前工作目录不能推断）；
- 允许使用的远程访问方式及明确授权范围；
- 当前有效的集群规则来源、Slurm 账号/分区/QOS/资源上限和试运行预算；
- 可合法取得且与任务匹配的 GearNet、ESM2、dMaSIF checkpoint 及其来源/哈希；
- 数据根目录、版本、许可证和允许使用的缓存位置；
- 是否允许只读引用旧 `/home/zy/projects/Surface-GearNet`，以及是否必须建立完全独立的项目目录。

## 解阻后的顺序

收到上述信息后，先做远程只读审计并把实际路径、版本和资源写入新项目文档；随后只做小规模真实样本 smoke test。未完成 checkpoint 加载核验、样本映射核验和资源测量前，不提交正式训练，也不把旧项目结果称为新方案结果。
