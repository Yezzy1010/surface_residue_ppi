## 2026-09-28 续跑状态（替代下方历史进度）

- **任务一主比较及诊断、任务二主比较、任务三结构主比较均已完成；不要重复提交已完成训练。任务二、任务三机制诊断等待用户决定。**
- 任务一诊断收尾数组 `276310` 全部完成，结果见 `runs/phase3_diagnostics_summary/20260928/main_v1/report.json`。ES-C 相对 ES 的全有效残基 AP 增益为 0.023393，95% 组件 bootstrap 区间 [0.015010, 0.031972]；容量、暴露/深度、训练与验证打乱对照相对 ES 的区间均跨 0。仅推理打乱属于敏感性证据。
- 任务二修正协议 `task2-common-mask-v2` 的数组 `276314` 三 seed 全部完成并经独立统计核验。467 个验证复合物上的宏平均 AP：ES 0.373617、ES-B 0.423972、ES-C 0.422889；ES-C−ES 区间 [0.038195, 0.061139]，ES-C−ES-B 区间 [-0.006202, 0.003504]。结果见 `runs/phase3_task2_summary/20260928/common_mask_v2/report.json`。原数组 `276301` 因负对未限制共同有效掩码而无效，原产物保留且未混入结果。这是自定义采样负例开发验证，不是官方 benchmark。
- 任务三结构审计完成：5,983 个候选 TRAIN 蛋白中 3,754 可用、2,229 排除、0 失败。原 50 个 gzip 解码失败保留证据后已仅重试失败项。冻结协议 `runs/phase3_task3_protocol/20260928/structure_v2/`：1,112 train / 194 val 对，1,843 个蛋白；蛋白与同源簇交集均为 0。使用精确全长匹配和质量过滤的 AlphaFold 独立单体；MMseqs 40% identity、80% coverage 同源簇划分。
- 任务三在相同子集比较 E/ES/ES-B/ES-C，三个固定 seed。A100 特征 smoke `276318` 已通过；旧 `276317` 的 KeOps 架构缓存失败证据保留。正式提取数组 `276324` 四片已全部完成；1,843/1,843 蛋白元数据覆盖与来源核验通过，汇总 `runs/phase3_task3_features/structure_v1/report.json`。GPU 训练 smoke `276326` 已通过四方法检查（Slurm 1分21秒），正式数组 `276327` 三个 seed 均在 A100 完成；最终汇总见 `runs/phase3_task3_summary/20260928/structure_v2/report.json`。**均已完成，不要再次提交。**
- 自动收集器 `scripts/collect_phase3_task23.py` 已完成任务二和任务三的下载、独立核验与汇总；状态 `runs/phase3_orchestration/20260928/collect_v2/state.json` 为 `both_main_comparisons_collected`。它只下载/验证/统计，不提交训练。
- 任务三结构主结果（三 seed，194 对验证集）：E AP 0.918147 / AUROC 0.853648；ES AP 0.920089 / AUROC 0.861482；ES-B AP 0.916647 / AUROC 0.856644；ES-C AP 0.916211 / AUROC 0.850614。ES 相对 E、ES-B/ES-C 相对 ES 的配对组件 bootstrap 区间均跨 0，当前子集未显示稳定的表面融合增益。
- **DIPS 测试仍封存；D-SCRIPT 测试历史上已被提前读取。** 序列基线数组 `276306` 和原协议准备曾读 D-SCRIPT 测试标签，原 reads=0 记录错误。更正见 `reports/phase3_resume_corrections_20260928.json`。旧序列结果仅为探索记录，不算结构主比较；当前 structure_v2 只读官方 TRAIN，不据旧测试性能选模型。
- 用户授权边界：先完成任务二/三主实验并汇报；两项机制诊断等用户审阅后决定。统计区间条件于已选验证 checkpoint，不代表正式测试推断。
