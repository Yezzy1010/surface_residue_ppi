# review_back：SRPPI-T2-SYM-001 v1.1

## 完成内容

本任务已按 `plan/task_plan_SRPPI-T2-SYM-001_v1.1.md` 完成。v1.1 明确批准使用原 task2 的确定性动态 `sample_pairs(record, seed, epoch, ratio=3)` 训练采样；本次没有把该流表述为历史 ordered 训练的逐 pair exact replay。

- 完成了 GPU Slurm preflight：sampler 重复性、三 variant 共享训练流、历史 validation pair 对齐和对称 head smoke。
- 完成 3 个固定 seed × 3 个 variant 的从头训练：`ES-SYM`、`ES-B-SYM`、`ES-C-SYM`，共 9/9 个组合通过。
- 完成相同 669,235 validation pairs 上的 canonical AB 与 swapped BA-transposed 推理。
- 完成 9 个 checkpoint 的独立 GPU fresh load；9/9 `strict=True` 加载成功。
- 所有组合在固定容差下交换一致：669,235/669,235 pairs 通过，`abs_delta` 全部为 0。
- 未读取测试数据；所有训练报告的 `test_records_read=0`、`test_labels_read=0`。

最重要的结果是：显式对称 head 完全消除了旧 ordered head 的 AB/BA 顺序敏感性；在同一动态采样修订协议下，表面分支仍保留开发验证增益。ES-B-SYM 相对 ES-SYM 的跨 seed AP 均值增益为 `0.0535372`，AUROC 为 `0.0360761`；ES-C-SYM 相对 ES-SYM 的 AP 均值增益为 `0.0510777`，AUROC 为 `0.0363016`。这些是 sampled-negative developmental validation 结果，不能作为正式测试或真实接触概率校准结论。

## 证据

### 对称实现与 preflight

实际实现位于集群项目：

- `/datapool/home/zhangye/projects/surface-residue-ppi/scripts/train_phase3_task2_sym.py`
- `/datapool/home/zhangye/projects/surface-residue-ppi/scripts/preflight_phase3_task2_sym.py`

对称 head 使用同一组 `left`/`right` 参数：

```text
g_theta(x,y) = 0.5 * [f_theta(x,y) + f_theta(y,x)]
f_theta(x,y) = sum(left(x) * right(y)) / sqrt(pair_hidden) + bias
```

preflight job `276497`：A100、`node101`、`a100`、4 CPU、96G、`00:39:23`、exit `0:0`。报告路径：

`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_preflight/276497/report.json`

preflight 结果：

- train rows `4861`，validation complexes `467`；validation pairs `669235`，其中正例 `60857`、负例 `608378`。
- 每个 seed 的 epoch 1/7/30 sampler 重复生成完全一致；每 epoch `2547752` pairs（`636938` positives、`1910814` negatives）。
- 同一 seed 下 ES-SYM、ES-B-SYM、ES-C-SYM 的 stream hash 完全相同。
- 历史 ES、ES-B、ES-C 的 validation payload 中 `labels`、`pairs`、`offsets`、`lmdb_indices` 对每个 seed 完全一致。
- 3 个 GPU smoke 变体各检查 64 pairs，最大绝对差 `0.0`，固定容差通过。

sampler stream hash（epoch 1 / 7 / 30）：

| seed | epoch 1 | epoch 7 | epoch 30 |
|---|---|---|---|
| 20260927 | `cfab58c4b3e30d789effc0fe1e877e4d32ad650c068dc9066fefb1566c53b8fd` | `c28530ac4e13274f0aea3aad214e869ddc289dea3c2deb878577b8b2827e5bca` | `ed7ce53515b3942cada1a4ae4836c3449d29d64127c50e730229100f714d45cb` |
| 20260928 | `db6b9a7ea8cfc2600c8d96175295e9b3851daaef167d46247466bd0f4954dc55` | `525a9bbbfd89c5950faf801776357d5b3e32ddbd6e59d5e5d16ec5183cc876ee` | `9e26736f57d2d062eb4dad0396f7363a267a31638d502da910b9dfdc5b88ee68` |
| 20260929 | `84796f664b825f1d7fa570f77621922b21812ab7247262589b5b9a438aae5e6f` | `15969eaef9c8bfed86a89597651d062a0768e07eaaa19e2525fc287ef2d6af2a` | `ec8cf5c09a5c1edc97dbd3d289436621c5ecec86cc6adf6dbbdb8feed596dd46` |

固定交换容差为：

```text
abs_delta <= 1e-6 + 1e-5 * max(abs(score_ab), abs(score_ba_transposed))
```

### 9 个组合结果

AP/AUROC 是 validation complex-macro 指标。`abs_delta` 摘要来自保存的逐 pair predictions；所有组合的 mean、median、max、q50、q95、q99 都是 `0.0`。

| seed | variant | job | node | best epoch | AP canonical=swapped | AUROC canonical=swapped | pairs pass | checkpoint path (bytes) | checkpoint SHA256 |
|---|---|---:|---|---:|---:|---:|---:|---|
| 20260927 | ES-SYM | 276499 | node101 | 4 | 0.3748808213 | 0.8275203480 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276499_20260927/ES_SYM.pt` (4,038,482) | `663266f41d30e986007dfbfc3044005dc5491fcc5821264ebad5d403c9545fb1` |
| 20260927 | ES-B-SYM | 276499 | node101 | 4 | 0.4293890921 | 0.8601113708 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276499_20260927/ES_B_SYM.pt` (4,843,510) | `3cb17a5677c9bb119309b7afc68257faacf152a0efe01026d825145b0175c472` |
| 20260927 | ES-C-SYM | 276499 | node101 | 3 | 0.4254184967 | 0.8600616175 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276499_20260927/ES_C_SYM.pt` (5,125,932) | `e98436d06f37d91df218b5f41690c7ebab1aed2c5b2711dd5c3b39a9dc3a5ef6` |
| 20260928 | ES-SYM | 276500 | node104 | 2 | 0.3708612362 | 0.8269004206 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276500_20260928/ES_SYM.pt` (4,038,482) | `2123fa7bd97a8d4f9308f8d4cde9c56b3b8a203fa1a5c796c03c6a41591028f1` |
| 20260928 | ES-B-SYM | 276500 | node104 | 3 | 0.4243525917 | 0.8590059170 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276500_20260928/ES_B_SYM.pt` (4,843,510) | `78987efa6fa1093d25bc8111f4ef6a0c9d7a2f38e2baf981ea2eff60b25f2380` |
| 20260928 | ES-C-SYM | 276500 | node104 | 2 | 0.4213568974 | 0.8585858997 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276500_20260928/ES_C_SYM.pt` (5,125,932) | `9dc421cd4b1fb54c5f20b7998c9dd3ab8804ba3e7c574aa41e823d3893226d64` |
| 20260929 | ES-SYM | 276498 | node101 | 7 | 0.3752333232 | 0.8185543487 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276498_20260929/ES_SYM.pt` (4,038,482) | `2381a18635cb3cbd2a7c8b4443a27461089f928308eb2f2a2dbd710939044d4b` |
| 20260929 | ES-B-SYM | 276498 | node101 | 3 | 0.4278452562 | 0.8620862686 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276498_20260929/ES_B_SYM.pt` (4,843,510) | `ef67c83c0819d0746c60d16ab2f0f52dfa3346e7963f792f3e082e60b6532505` |
| 20260929 | ES-C-SYM | 276498 | node101 | 3 | 0.4274329642 | 0.8632325171 | 669235/669235 | `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276498_20260929/ES_C_SYM.pt` (5,125,932) | `17035b77f946072f04efe8dfb46712994b9ae69a9ce782a8fea0ca27c4a098d3` |

逐组合 checkpoint、prediction、完整差异分位数和指标路径见集群：

`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/summary_v1.1_light/symmetry_summary.tsv`

### 表面增益

跨三个 seed 的 validation AP/AUROC 均值和样本标准差：

| variant | AP mean ± SD | AUROC mean ± SD |
|---|---:|---:|
| ES-SYM | `0.3736584602 ± 0.0024288703` | `0.8243250391 ± 0.0050071677` |
| ES-B-SYM | `0.4271956466 ± 0.0025803253` | `0.8604011855 ± 0.0015604922` |
| ES-C-SYM | `0.4247361194 ± 0.0030949759` | `0.8606266781 ± 0.0023742859` |

逐 seed 的配对 complex 差值（同一 seed、同一 validation complex）：

| seed | ES-B-SYM − ES-SYM AP | ES-B-SYM − ES-SYM AUROC | ES-C-SYM − ES-SYM AP | ES-C-SYM − ES-SYM AUROC |
|---|---:|---:|---:|---:|
| 20260927 | `0.0545082707` | `0.0325910228` | `0.0505376753` | `0.0325412694` |
| 20260928 | `0.0534913554` | `0.0321054964` | `0.0504956612` | `0.0316854791` |
| 20260929 | `0.0526119330` | `0.0435319199` | `0.0521996410` | `0.0446781684` |
| 均值 ± seed SD | `0.0535371864 ± 0.0009490` | `0.0360761463 ± 0.0064615` | `0.0510776592 ± 0.0009719` | `0.0363016390 ± 0.0072669` |

逐 complex 差值附件：

`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/summary_v1.1_light/surface_paired_deltas.tsv`

### 历史 ordered 对照

历史 ordered 结果来自 `review_back/SRPPI-EVID-002_v1.1.md` 及集群旧 task2 reports，只作跨运行历史对照，不能视为同一 checkpoint 的 paired replay：

| variant | ordered AP mean | symmetric AP mean | ordered AUROC mean | symmetric AUROC mean |
|---|---:|---:|---:|---:|
| ES | 0.3736170094 | 0.3736584602 | 0.8275018888 | 0.8243250391 |
| ES-B | 0.4239719918 | 0.4271956466 | 0.8592590086 | 0.8604011855 |
| ES-C | 0.4228887088 | 0.4247361194 | 0.8601702968 | 0.8606266781 |

旧 ordered head 的 AB/BA 检查在 EVID-002 v1.1 中 9/9 未通过；本任务的 9/9 对称模型全部通过。该对照不能证明性能变化只来自 head，因为 v1.1 允许的动态训练流不是历史 ordered 训练的逐 pair exact replay。

### 训练与资源 provenance

| job | 用途 | 分区/GPU | 节点 | CPU/内存 | 时长 | 状态 |
|---:|---|---|---|---|---:|---|
| 276497 | sampler/validation/head preflight | a100 / 1×A100 | node101 | 4 / 96G | 00:39:23 | COMPLETED |
| 276498_0 | seed 20260927，3 variants | a100 / 1×A100 | node101 | 4 / 96G | 00:50:26 | COMPLETED |
| 276498_1 | seed 20260928，3 variants | a100 / 1×A100 | node104 | 4 / 96G | 00:50:24 | COMPLETED |
| 276498_2 | seed 20260929，3 variants | a100 / 1×A100 | node101 | 4 / 96G | 00:54:14 | COMPLETED |
| 276510 | 9 checkpoint fresh strict load | a100 / 1×A100 | node101 | 2 / 16G | 00:01:57 | COMPLETED |
| 276508 | CPU汇总脚本 | hpc | 未分配 | 8 / 32G | pending | `PENDING (Resources)` |

276509 曾因环境 PyTorch `1.8.1+cu111` 不支持 `torch.load(weights_only=...)` 而失败；随后修正为兼容调用，276510 成功完成 9/9 strict model/head load。该问题不影响训练或预测产物。

checkpoint fresh-load 报告：

`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/checkpoint_load_verification.json`

设备依赖版本：Python `3.8.18`，PyTorch `1.8.1+cu111`，NumPy `1.24.4`；fresh load 使用 `NVIDIA A100 80GB PCIe`。

### 输入与代码哈希

固定输入：

- protocol report：`d10e45df9750e8b730d0f300c1ea9dd13f1f9d8d2b7a03c7f302fe8eb01874c2`
- train manifest：`4107096e8ef2acffab31c388c1f3156659186addcbcacd30b25dcbdc71733d72`
- validation/strict manifest：`476416f9e286f691011bd075a4132aefcdaabcce9f4bfc44aa414bd77ec73b27`
- feature report：`f3d2c81aaaec607a5911548a3831bc6ef42192740a9ca23d51e3e4e256f9c081`
- task2 config：`baa6475493430130d1af2cc40290c4cef17b3047920e3f7eaee2571e43869fe9`
- ordered task2 source：`685f688b83621f0b0d03ad644c29c17ff2481adb1abc4636a33321d9c3e8bb15`
- symmetric training source：`9683ce7305032692c0bcba4c4fd68a2a6d770865bb1ab3a4e201f1f33e249027`
- preflight source：`2f3c68db1ddc35679b70d40f55b6491cd1d17048435c70cb4549547eece0e29d`
- training Slurm script：`a108a460111b164b917381da3a7d2dda15eee0031db45c0dcf3dfd7079925764`
- preflight Slurm script：`000cac474b97aadd3018c37b40c05b025967a082fcd269f7877163545bd9f5db`
- checkpoint verification source：`123770835c673859fb8a786119036806a9365ece2c55cdd5815b280381f0f7a1`

训练使用 cached ESM/GearNet/surface 输入；backbone 仍冻结，更新的是原 task2 `Readout` projection/adapter 与新的 symmetric pair head。没有 warm-start ordered checkpoint，没有修改共享 encoder、feature cache、manifest、标签或环境。每个 variant 的 trainable parameter count 为：ES-SYM `1,008,642`，ES-B-SYM `1,209,602`，ES-C-SYM `1,279,778`。

### 关键附件索引

- 训练报告：
  - `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276499_20260927/report.json`
  - `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276500_20260928/report.json`
  - `/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/276498_20260929/report.json`
- sampler manifest：各 seed 目录内 `sampler_stream_manifest.json`
- history：各 seed 目录内 `ES_SYM_history.json`、`ES_B_SYM_history.json`、`ES_C_SYM_history.json`
- predictions：各 seed 目录内对应 `*_SYM_predictions.npz`；包含 `score_ab`、`score_ba_transposed`、`abs_delta`、`relative_delta`、`labels`、`pairs`、`offsets`、`lmdb_indices`
- complex metrics：各 seed 目录内对应 `*_SYM_metrics_by_complex.json`
- 汇总目录：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/summary_v1.1_light/`
- fresh-load：`/datapool/home/zhangye/projects/surface-residue-ppi/runs/phase3_task2_sym_training/checkpoint_load_verification.json`

## 偏差

1. 训练负例不是历史 ordered 运行的逐 pair exact replay。原因是原 task2 没保存固定 train pair；v1.1 已明确批准原 `sample_pairs` 的确定性动态重建。sampler 语义、ratio、seed、epoch、manifest 顺序和 rejection 条件保持原定义。
2. 由于 hpc 节点长期满载，汇总脚本 `276508` 在最终查询时仍 pending；结果汇总已依据已生成的训练报告和 prediction 文件完成轻量整理。模型 smoke 和 checkpoint fresh load 均通过 Slurm（276497、276510）。
3. 276509 的 `weights_only` 参数兼容性失败已记录并修正为无该参数的兼容加载；276510 成功。

## 问题

- 结果仍是 sampled-negative developmental validation，不能外推为正式 benchmark 或 unbound PPI 性能。
- 动态训练流的重复性已验证，但不能证明其等于历史 ordered 训练实际使用的逐 pair 流。
- ES-B/ES-C 与 ES-SYM 的增益使用同一新对称协议内的 validation complex 配对；历史 ordered 对照仍是跨运行比较。
- hpc job `276508` 仍排队（当前原因 `Resources`），仅用于重复生成汇总附件；截至本回执生成时没有新训练或推理在运行。

## 待决事项

本任务范围内没有未完成的训练或验证组合。下一步由决策 agent 决定是否接受“对称约束下表面增益仍保留”的 developmental 证据，或是否需要另行制定新任务。不得自动启动 separable baseline、任务三局部交互、正式测试或其他扩展实验。
