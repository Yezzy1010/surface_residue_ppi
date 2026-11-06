# ESM-GearNet 源码与权重接入（2026-11-06）

## 本次范围和研究方向

用户授权拉取官方源码、下载先前选定的 Multiview Contrast 权重，并将项目接入配置上传到现有 GitHub 仓库。源码与轻量记录进入 Git；模型权重、下载断点和运行记录留在数据目录。

新方案采用官方 ESM-GearNet 预训练初始化，随后各下游任务独立反向传播和微调。ESM、GearNet、新增投影、表面注入、patch 与 residue–patch 模块的训练设置将在实验配置中定义；dMaSIF 第一版保持冻结。统一冻结表征不是当前主实验要求。本次没有实现这些新模块、安装环境、读取测试集或运行模型。

## 官方源码

- 仓库：https://github.com/DeepGraphLearning/ESM-GearNet
- 固定 commit：`bc90557bbf9ee2676e87b9fc84570aa126daa808`（`Update README.md`）
- 项目路径：`third_party/ESM-GearNet/`，由 `.gitmodules` 管理。
- `configs/dependencies/esm_gearnet.json` 记录关键源码和配置 SHA256。
- 初次完整克隆因传输提前结束失败；随后浅克隆成功，固定到上述 commit。官方工作树保持未修改。

恢复当前版本：

```bash
git submodule update --init third_party/ESM-GearNet
git -C third_party/ESM-GearNet rev-parse HEAD
git -C third_party/ESM-GearNet status --short
```

## 权重

- 官方发布页：https://zenodo.org/records/10034578
- 本轮选定：`mc_esm_gearnet.pth`，Multiview Contrast，官方页面显示约 2.7 GB。
- 官方 MD5：`ed4cd0f6da31e24fa7ddc5d8107a194e`。
- 目标路径：`data/models/esm_gearnet/mc_esm_gearnet.pth`。
- 只下载该初始化；没有批量下载六种预训练权重，也没有安装或自动下载其他模型资产。

下载及恢复命令：

```bash
python3 scripts/download_esm_gearnet_weights.py
python3 scripts/download_esm_gearnet_weights.py --verify-only
```

脚本仅使用标准库和 curl，允许在本地进行文件下载与字节校验，不导入 torch 或模型。下载写入 `.partial`；重复调用继续下载，MD5 通过后才更名为最终文件，同时记录 SHA256、大小、主机、耗时和工具版本。失败与校验不符的文件保留，不自动覆盖既有 checkpoint。`--direct` 仅对当次请求绕过代理，不修改系统配置。

**当前状态：源码已下载；权重尚未下载完成，模型未验证加载。** 本机系统 curl/LibreSSL 与 Python/OpenSSL 直连 Zenodo 均出现 TLS EOF；系统代理返回 `503 Service Unavailable`，浏览器也报告 `ERR_TUNNEL_CONNECTION_FAILED`。用户要求重试后再次访问官方端点仍失败。实际下载脚本运行结果保存在上述数据目录的 `download_records/`，不将“地址存在”表述为“权重可用”。网络恢复后可使用同一命令继续；最终成功状态必须为 `checksum_verified`。

## 后续接入边界

官方配置采用 ESM-2-650M（1280 维）和六层 GearNet。串行模式先把 ESM 输出送入 GearNet，当前官方 `FusionNetwork` 再拼接两路输出。现有独立 GearNet 缓存不是该串行模型的输出。

ESM-GearNet 新建项目专属环境；官方 README 的参考环境使用 Python 3.7/3.8 和 PyTorch 1.12.1。现有 dMaSIF 独立环境继续保留，不修改共享 `gearnet` 环境。具体安装版本与 GPU 参数需在集群实际核验后另记。

checkpoint 严格加载、输出张量、梯度、特征提取、smoke 和训练均须通过 Slurm。当前没有宣称 ESM-GearNet、ESM2 或 dMaSIF 已在新环境可用。后续恢复模型时还需按实际 ESM 加载器准备其依赖资产。

GitHub 保存源码 submodule 引用、来源清单、下载工具和说明；数据目录继续被 `.gitignore` 排除，仓库克隆不包含 2.7 GB 权重。

## 本轮检查

- submodule commit 与清单一致，官方源码工作树无修改。
- 清单中六个源码/配置 SHA256 均与本地官方源码匹配。
- 下载脚本 AST 解析通过；使用已知 `abc` 字节串核对 MD5、SHA256 和文件大小通过，未执行模型。
- checkpoint 目标路径的 Git ignore 检查通过。
- 官方权重直链实际下载重试三次，curl 返回 35，下载记录的 partial size 为 0；该检查不构成成功下载或加载验证。
