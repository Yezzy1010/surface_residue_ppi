# Review back：SRPPI-EVID-002 v1.1

本回执是 `SRPPI-EVID-002 v1.0` 的执行续版，范围限定为：定位 task2 现有 checkpoint，并在 `task2-common-mask-v2` 验证集完成 ES/ES-B/ES-C 的 AB/BA 交换推理。

## 1. 结论

【产物已核验】9 个 checkpoint 全部找到，文件 SHA256 与对应 `report.json` 的 `checkpoint_sha256` 完全一致，并在 Slurm job `276472` 中用 `strict=True` 成功加载。

【产物已核验】AB/BA 交换推理成功完成。三个 seed、三个 variant 均使用各自已经保存的 669,235 个 validation pairs、labels、offsets 和 lmdb_indices；没有重新采样负例，没有训练，没有特征提取，没有读取测试数据。

【产物已核验】当前 ordered pair head **不满足预先固定的交换一致性容差**。对所有 9 个 seed/variant，固定判据

```text
abs(score_ab - score_ba_transposed)
    <= 1e-6 + 1e-5 * max(abs(score_ab), abs(score_ba_transposed))
```

均未对全部 pair 通过。每个组合只有 15–30 / 669,235 个 pair 通过，比例约 `2.24e-5`–`4.48e-5`；平均绝对差为 `0.461`–`0.774`，远大于数值容差。因此当前 task2 pair head 对交换蛋白顺序敏感。这个结果是协议解释限制，不自动触发对称 head 重训，也不启动 separable baseline 或其他新实验。

## 2. Checkpoint 定位与加载

集群项目根目录：`/datapool/home/zhangye/projects/surface-residue-ppi`。

| seed | variant | checkpoint 绝对路径 | 大小（bytes） | SHA256 | 加载 |
|---:|---|---|---:|---|---|
| 20260927 | ES | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276315_20260927/ES.pt` | 4,038,290 | `5950fab31945f60e35f61ab17236a07cebdb36352e89c17d035b5103fceea0bb` | `strict=True` 成功 |
| 20260927 | ES-B | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276315_20260927/ES_B.pt` | 4,843,318 | `953f9509d321edc61961a6a2987327c98a74a0be597d00775482500786612ef` | `strict=True` 成功 |
| 20260927 | ES-C | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276315_20260927/ES_C.pt` | 5,125,804 | `e53a2e55ef4e2e179c45272adc4d925d34a53d88e2aa67fa92c68d197133d4cf` | `strict=True` 成功 |
| 20260928 | ES | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276316_20260928/ES.pt` | 4,038,290 | `9281fe1d85db8475e7aa405cc47ff8b85a5de6c7ad523298ee511ae3f82ae334` | `strict=True` 成功 |
| 20260928 | ES-B | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276316_20260928/ES_B.pt` | 4,843,318 | `df1e1579e3f1a6b07c169cd3d0c4cad5c1ab7fc0e7a89bdbe65f622e02faa2d0` | `strict=True` 成功 |
| 20260928 | ES-C | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276316_20260928/ES_C.pt` | 5,125,804 | `eea56fcbae76dfeca6fb5414caf015d85b51e3f3440cd0054d80e9a14e8ea8d9` | `strict=True` 成功 |
| 20260929 | ES | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276314_20260929/ES.pt` | 4,038,290 | `50a3bba5b69357a971d730d86d74d8a8f355b845676d014bc9a784ad24c0a15f` | `strict=True` 成功 |
| 20260929 | ES-B | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276314_20260929/ES_B.pt` | 4,843,318 | `c90e60b28f4ee098542a4352777017f3d14afbae4633a31dc259d62f85e74eb0` | `strict=True` 成功 |
| 20260929 | ES-C | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_training/276314_20260929/ES_C.pt` | 5,125,804 | `53e3d4435694990c024595fbf75b75033e0d6ec73d8f74059ae4bea73f5b6084` | `strict=True` 成功 |

完整 JSON 清单：`review_back/SRPPI-EVID-002_v1.1/checkpoint_inventory.json`；表格版：`checkpoint_inventory.md`。

## 3. Slurm 执行记录

- job ID：`276472`
- 状态：`COMPLETED`, exit code `0:0`
- 分区：`a100`
- 节点：`node101`
- GPU：`NVIDIA A100 80GB PCIe`, 1 张
- 资源：1 node、4 CPU、96G 内存、8 小时上限；实际耗时 `00:17:58`
- 起止时间：`2026-09-28 19:50:54` 至 `20:08:52`（Asia/Shanghai）
- 环境：Python `3.8.20`、PyTorch `1.8.1+cu111`、NumPy `1.24.4`
- 并发：单作业顺序运行 9 个 seed/variant
- 测试访问：`test_records_read=0`、`test_labels_read=0`
- 本次代码 hash：`task2_swap_inference.py` SHA256 `9221300e7d67e49962cf9118f7ede1b563b73d44df81285ba131aece5c6875f7`

Slurm 详情保存在 `slurm_job_276472.txt`；运行配置、路径和依赖 hash 保存在 `results/run_config.json`。

## 4. 固定输入与一致性检查

【产物已核验】输入固定为：

- protocol：`task2-common-mask-v2`
- validation complexes：467
- seed：20260927、20260928、20260929
- variant：ES、ES-B、ES-C
- 每个 seed/variant：669,235 pairs，60,857 positives，608,378 negatives
- pairs、labels、offsets、lmdb_indices：复用原 NPZ；同一 seed 内三个 variant 完全一致
- AB：`score_head(h_A[i], h_B[j])`
- BA：同一 pair 的反向输入 `score_head(h_B[j], h_A[i])`
- common mask、feature cache、normalization、validation manifest：均复用原文件
- 负例：未重新生成

运行所用固定输入 hash：

| 输入 | SHA256 |
|---|---|
| task2 protocol report | `d10e45df9750e8b730d0f300c1ea9dd13f1f9d8d2b7a03c7f302fe8eb01874c2` |
| feature report | `f3d2c81aaaec607a5911548a3831bc6ef42192740a9ca23d51e3e4e256f9c081` |
| validation manifest | `476416f9e286f691011bd075a4132aefcdaabcce9f4bfc44aa414bd77ec73b27` |
| strict validation manifest | `476416f9e286f691011bd075a4132aefcdaabcce9f4bfc44aa414bd77ec73b27` |
| experiment config | `baa6475493430130d1af2cc40290c4cef17b3047920e3f7eaee2571e43869fe9` |
| normalization（3 seeds identical） | `fb7a3476a1cfeb9e0be0493551c20373629f2e1b95f65e093e512256034c2d68` |
| task2 loader source | `f0214c8bbd1beeb39c9f41d879f38feee7e66f8f9ecd471e748b8f4e5a13bab7` |
| task2 training/model definitions | `685f688b83621f0b0d03ad644c29c17ff2481adb1abc4636a33321d9c3e8bb15` |
| adapters source | `b5d43fa7f4d36cdc75c8b850c20f8d735d44a605aed2d8982258141a90d0775f` |

运行前固定：absolute tolerance `1e-6`，relative tolerance `1e-5`。relative_delta 定义为：

```text
abs_delta / max(max(abs(score_ab), abs(score_ba_transposed)), 1e-12)
```

每个组合的逐 pair 数组保存在 `results/task2_swap_predictions_<seed>_<variant>.npz`，字段包括 `score_ab`、`score_ba_transposed`、`abs_delta`、`relative_delta`、`pairs`、`labels`、`offsets` 和 `lmdb_indices`。

## 5. 交换结果

下表给出 complex-macro AP/AUROC、逐 pair 差值摘要和固定容差通过数。完整逐 complex JSON 位于 `results/task2_swap_summary_<seed>_<variant>.json`，汇总 TSV 位于 `task2_swap_summary.tsv`。

| seed | variant | canonical AP / AUROC | swapped AP / AUROC | abs mean / median / max | abs q95 / q99 | rel mean / median / max | 容差通过数 / 总数 |
|---:|---|---|---|---|---|---|---:|
| 20260927 | ES | 0.377347 / 0.830398 | 0.370519 / 0.824085 | 0.554811 / 0.457864 / 4.823059 | 1.402392 / 1.933297 | 0.349920 / 0.208348 / 1.999933 | 18 / 669235 |
| 20260927 | ES-B | 0.426008 / 0.859321 | 0.416302 / 0.851780 | 0.636384 / 0.514295 / 5.715887 | 1.645229 / 2.339153 | 0.304075 / 0.175661 / 1.999974 | 15 / 669235 |
| 20260927 | ES-C | 0.426807 / 0.859998 | 0.412531 / 0.850344 | 0.560088 / 0.447304 / 5.119585 | 1.470140 / 2.111085 | 0.300462 / 0.171830 / 1.999877 | 30 / 669235 |
| 20260928 | ES | 0.370205 / 0.826405 | 0.357763 / 0.815508 | 0.473512 / 0.387594 / 4.814400 | 1.206157 / 1.670698 | 0.350310 / 0.204503 / 1.999983 | 24 / 669235 |
| 20260928 | ES-B | 0.420428 / 0.855804 | 0.407285 / 0.849029 | 0.580073 / 0.464406 / 5.836118 | 1.518080 / 2.163380 | 0.315298 / 0.185003 / 1.999982 | 19 / 669235 |
| 20260928 | ES-C | 0.419366 / 0.858555 | 0.412420 / 0.852721 | 0.461185 / 0.368824 / 4.067468 | 1.207025 / 1.740216 | 0.292781 / 0.167212 / 1.999983 | 30 / 669235 |
| 20260929 | ES | 0.373299 / 0.825702 | 0.359893 / 0.812658 | 0.773920 / 0.641634 / 5.922539 | 1.946621 / 2.643281 | 0.363828 / 0.222329 / 1.999916 | 19 / 669235 |
| 20260929 | ES-B | 0.425480 / 0.862651 | 0.411495 / 0.852038 | 0.559404 / 0.452472 / 5.075796 | 1.444833 / 2.013895 | 0.324366 / 0.193430 / 1.999995 | 21 / 669235 |
| 20260929 | ES-C | 0.422493 / 0.861958 | 0.414184 / 0.851919 | 0.564297 / 0.452085 / 4.955073 | 1.476952 / 2.092942 | 0.322146 / 0.192336 / 1.999915 | 16 / 669235 |

q50 等于 median，已在逐组合 JSON 中同时保存。所有 9 个组合的 `all_pairs_within_fixed_tolerance=false`。

## 6. 结果解释和限制

【分析或假设】当前结果支持“ordered pair head 对链顺序敏感”，不支持“交换近似成立”。AB→BA 后 AP 和 AUROC 在 9 个组合中均下降；该趋势跨 seed、ES、ES-B、ES-C 一致。

【产物已核验】将复算的 canonical `score_ab` 与原保存 NPZ 的 `logits` 对照：ES 三个 seed 完全相等；ES-B/ES-C 最大绝对差均为 `5.7220458984375e-06`，均值约 `1.75e-7–2.46e-7`。这是逐 complex batch 重算造成的 float32 累加/顺序差异；labels、pairs、offsets、lmdb_indices 均逐数组完全相等。该复算差异远小于观察到的交换平均绝对差（0.461–0.774），不会改变“明显有序”的判断。详细见 `canonical_reproduction_check.json`。

【产物已核验】AB/BA 两次独立 batch 的单体编码在同一复合物上的最大绝对差为 0–`4.76837158203125e-06`，仍远小于交换分数的平均差；这表明主要差异来自有序 pair head，而非交换 batch 造成的单体编码变化。该值作为诊断记录，不改变预先固定的交换容差。

本结果不回答是否应重训显式对称 head，也不构成新的模型选择或正式测试结论。任务二仍是 sampled-negative developmental validation。

## 7. 证据文件

本地同步目录：`/home/zy/projects/Surface-residue/review_back/SRPPI-EVID-002_v1.1/`

- `checkpoint_inventory.json` / `checkpoint_inventory.md`：9 个 checkpoint 的路径、大小、报告 hash、文件 hash 和 strict load 结果。
- `results/task2_swap_check_summary.json`：总运行状态、Slurm、依赖和 9 个结果。
- `results/task2_swap_summary_<seed>_<variant>.json`：逐 complex 结果和差值摘要。
- `results/task2_swap_predictions_<seed>_<variant>.npz`：逐 pair AB/BA 分数和固定输入数组。
- `results/run_config.json`：protocol、manifest、feature、代码和环境 hash。
- `task2_swap_summary.tsv`：紧凑汇总表。
- `canonical_reproduction_check.json`：canonical score 与既有保存 logits 的重算核对。
- `slurm_job_276472.txt`：作业节点、分区、资源、起止时间和状态。

集群原始结果目录：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_swap_check/20260928/results/`。

## 8. 偏差、问题与待决事项

### 偏差

- 为避免 9 个独立作业产生额外并发，使用一个 A100 job 顺序执行所有 seed/variant；这不改变输入、模型或指标。
- canonical 重算按单 complex batch 进行，ES-B/ES-C 相比旧保存 logits 有微小 float32 顺序差异；已单独记录，未修改旧预测。

### 问题

- 【已解决】9 个 checkpoint 定位、SHA256 核对和 strict load。
- 【已解决】三 seed/三 variant 的 saved pair、label、offset、validation 顺序和 common mask 对齐。
- 【已解决】AB/BA 逐 pair 分数、差值、complex-macro 指标和 provenance 生成。
- 【确认】当前 ordered pair head 对交换顺序敏感；是否重训显式对称 head 属于新任务。

### 待决事项

1. 是否接受当前交换检查作为“需要修正/重新设计对称 head”的证据。
2. 若继续，需另行批准显式对称 head 设计和训练预算；本回执不自动启动。
3. 是否在保留当前有序 head 作为对照的前提下，再决定 separable baseline；本回执不自动启动。

