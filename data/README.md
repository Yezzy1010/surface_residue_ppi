# 数据目录

数据不进入 Git，本文件记录用途。Git clone 后需要恢复对应外部数据；不是运行下载脚本就代表完成验收。

| 路径 | 内容 | 整理前集群占用 |
|---|---|---:|
| `atom3d/zenodo-4911102/` | 官方 PPI-DIPS 压缩包、LMDB、发布索引；来源 https://zenodo.org/records/4911102 | 约 32G |
| `rcsb/derived_data/`、`rcsb/mmcif/` | RCSB 序列和结构，支持序列映射与结构特征 | 合计约 7.5G |
| `cache/` | 模型或下载缓存 | 约 290M |
| `task3/` | D-SCRIPT/UniProt 任务三输入；工作站另有已归位的 `task3_xref_sample.json` | 约 98M |
| `manifests/` | 预留的数据清单位置 | 空目录 |
| `models/esm_gearnet/` | 官方 Multiview Contrast 权重、断点文件及逐次下载记录；不进入 Git | 新接入，下载状态见接入记录 |
| `legacy/dscript/` | 工作站历史 D-SCRIPT TSV/FASTA 下载副本；没有宣称重新验证用途 | 工作站独有 |
| `structures/root_inputs_20250925/` | 集群原根目录 20 个 mmCIF，归档保留 | 集群独有 |

两端大文件分布不同，详见 [移动清单](../archive/manifest.tsv)。新实验继续使用原有 `rcsb/`、ATOM3D 与 runs 协议路径。DIPS 测试结构/标签保持封存；D-SCRIPT 测试早期曾读取，不能称全部测试从未访问。标签和 partner 坐标不进入 encoder 输入。
