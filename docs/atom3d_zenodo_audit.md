# ATOM3D PPI Zenodo 审计

日期：2026-09-24。已完成 DIPS-Split 主包下载校验、完整解压、LMDB 元数据检查以及 20 条 train/val 接触抽查；结果 `needs_review`。未做全量接触或序列同源验证，未读测试结构/标签。

2026-09-25 更新：全量 train/val 残基库存（118,353 条）已完成，0 条读取/统计失败；原 20 条诊断样本的完整残基身份接触重建通过。库存不是全量接触验证，原 `needs_review` 报告未覆盖改写。统计、性能和边界见 `docs/residue_preparation_20260924.md`。

## 发布记录

- 标题：ATOM3D: Protein-Protein Interactions (PPI) Dataset
- Zenodo DOI：`10.5281/zenodo.4911102`
- Zenodo record：`4911102`
- 许可证：CC-BY-4.0
- API 元数据文件：`data/atom3d/zenodo-4911102/record.json`
- record.json SHA256：`e9a4ad1efa05ac2ab10abe6b0de3004ad828097bb2239251938be673dc5e2e36`

## 已下载并校验的小文件

| 文件 | 大小 | Zenodo MD5 | 本地 MD5 | 本地 SHA256 |
|---|---:|---|---|---|
| `README.md` | 2,483 B | `441276eebafaae60f3158b1d4a5b6e76` | `441276eebafaae60f3158b1d4a5b6e76` | `4ffdc574e01c47c4792053ff301ca8d60f9af0f273a3ae414ea9981006bf253d` |
| `PPI-DIPS-split-indices.tar.gz` | 313,097 B | `40b9f6117acffaa9ef14f30ccf40bfc3` | `40b9f6117acffaa9ef14f30ccf40bfc3` | `6714bac64273a7517b8d17c5f3f2bb901f73b271a1e8134d1361be4fc401730a` |

索引压缩包已解压到 `data/atom3d/zenodo-4911102/extracted/indices/`，并保留原始压缩包。文件行数和 SHA256 为：

- `train_indices.txt`: 87,303 条目（文件行数 87,302，末尾无换行）；`2586c6bb14fc7e2c3470922a798c45cbc117cc695a4c572d6644e4ad06edf465`
- `val_indices.txt`: 31,050 条目（文件行数 31,049，末尾无换行）；`92537e2246c9d5ec9a4689078b1db40de0aaccdc74f05ad5178d1e196bf05d8f`
- `test_indices.txt`: 15,268 条目（文件行数 15,267，末尾无换行）；`bdc3ae536d30ca2e13f01889148c83523e6a2370221eecea3766437dc37af204`

三份索引合计 133,621 条，主包 LMDB 数量与各索引逐一相符。发布 README 中 DIPS 总量为 134,215 对；差异仍未解释，不能以删除或重划分样本消除差异。

## 从发布 README 确认的标签协议

- 接口正例：来自两条不同蛋白链的氨基酸对，任意重原子在 6 Å 内（原文 within 6 Angstroms）。固定源码 `_get_heavy_neighbors` 使用 `dist < cutoff`；审计将以严格 `< 6 Å` 重建并与发布标签比对，边界不擅自改成 `<=`。
- README 声称 DIPS-Split 按序列身份划分，跨集合链重叠不超过 30%，约 80/10/10。这是发布说明，尚未独立验证序列身份约束；索引条目比例实测约 65.34/23.24/11.43，与文字说明不同，保留官方划分并核查差异。
- DIPS 筛选说明包含蛋白分子类型、链长至少 50、分辨率优于 3.5 Å、Cryo-EM 或 X-ray、结合后埋藏表面积大于 500 Å² 等条件。
- Zenodo README 描述的是发布任务标签；ATOM3D 源码中的默认 cutoff 或 loader 采样规则不能覆盖这一发布定义。

## 主包状态

- `PPI-DIPS-split.tar.gz`：13,568,121,746 B，Zenodo MD5 `b6095db528e4a2e1139fddc64b6965de`；已在本地与集群校验完成。本地完整解压和本次 CPU 审计已完成，14 个解压文件也已在集群通过 SHA256 校验并放入正式 `lmdb/` 目录，最终状态见 `docs/resume.md`。
- `PPI-raw.tar.gz`：30,875,296,746 B，Zenodo MD5 `621977d132b39957e3480a24a30a7358`；未下载。

在完整包下载、校验、LMDB 内容抽样、正负样本生成和预训练重叠审核完成前，不启动正式筛选或训练。

## 索引一致性检查

三份索引按非空记录读取，均升序、均无重复，集合间交集为零：

- train：87,303 条目
- val：31,050 条目
- test：15,268 条目
- union：133,621 条目

文件末尾缺少换行，因此 wc -l 比实际非空记录数少一条；上述条目数以逐行解析为准。索引中的最小/最大整数范围为 56–271,644。主 LMDB 的各 split 数量与索引一致，ID map 无交集；README 的 134,215 对总数差异仍未定位来源。

## 已完成的工作站主包审计

原 `275826` / `275828` 已取消；替代作业 `275845` 在本地成果上传校验后于 18:13:05 +08:00 取消，从未开始计算。用户授权后在 i9-14900K、125 GiB 内存的 Linux 工作站执行了同一审计逻辑，显式选择 `local-workstation`，没有伪造 Slurm 作业 ID。5 项 CPU 单元测试通过；审计总耗时 156.50 秒、峰值 RSS 196.1 MiB。报告位于 `runs/atom3d_audit/20260924/local_20260924_workstation_01/report.json`。

接触审计以固定种子 20260924 按 ID 哈希抽取 train/val 各 10 条，每个 PDB 至多一条；这是字段和标签审计样本，不是正式训练子集，也不代表同源簇去重。20/20 发布正接触与全部发布重原子严格 `< 6 Å` 的完整重建一致；19 条未触发当前检查项，1 条残基键冲突触发 `needs_review`。未触发检查项不代表已完成正式训练前清洗。

测试 LMDB 只读取 `num_examples`、`serialization_format`、`id_to_idx` 元数据。训练/验证样本检查完整正接触、重复记录、插入码导致的残基键冲突和缺失 Cα；PDB 重叠结果对照 dMaSIF 仓库列表，不能当作 checkpoint 精确训练集或同源无泄漏证据。审计脚本不会启动训练，即使抽查通过也仍需对正式选中样本逐项验证。

## 定向复核：残基身份和非蛋白条目

另存的 `followup_cases.json` 只重读 3 条已有 train/val 样本，保留原报告：

- `3mg7.pdb1.gz_1_L_3mg7.pdb1.gz_1_V`，val index 12442：上游六字段键不含 insertion_code。L 链的 70、106、144 分别对应 2、3、18 个完整身份；V 链 105 对应 2 个。L144 合并键涉及 8 条正接触。正式标签必须保留完整残基身份并从坐标重建，不能复制合并键标签给所有变体。
- `2ad7` A/B 链分别有 872/180 个水残基；A 链另有 PQQ 和钙离子。A/B 的 571/69 个标准非 hetero 氨基酸均有真正的 Cα。钙离子的原子名 CA 不能计作蛋白 Cα。
- `4wba` D/E 链有 66/73 个水残基，D 链另有 GOL；41/42 个标准非 hetero 氨基酸均有 Cα。
- 仅诊断性地限于标准非 hetero 氨基酸后，`2ad7` 接触键从 910 降至 180，`4wba` 从 178 降至 74。该计数仍使用上游键，不是正式训练标签，也没有决定非标准氨基酸处理策略。

因此，原报告 `residues_without_ca` 是全发布条目的统计，不能解读为蛋白缺失 Cα 比例；`operational_negative_pairs` 也不直接作为蛋白残基训练负例。下一步需冻结蛋白候选集合、插入码和非标准残基处理规则，并记录清洗对标签的影响。

固定 dMaSIF 仓库 `training_ppi.txt` 与发布 train/val/test 的 PDB 交集分别为 2810/230/129 个，对应 10902/1667/980 条记录。尚未确认这些仓库列表就是当前 checkpoint 的精确训练成员，也未完成序列同源审计。
