# ASTER 汇总数据

本目录由 experiments/aster_closed_form/recompute_synthesis.py 从仓库发布的保存分数复算，无 GPU 前向。

| 文件 | 内容 |
| --- | --- |
| [main_comparison.csv](main_comparison.csv) | 7 方法 × fit/dev/test/all，A/P/C/F、宏平均、输入正确数及纠正／退化，保留原分母 |
| [main_input_changes.csv](main_input_changes.csv) | 全部、参考改变、参考未变输入，分开计数 |
| [main_effects.json](main_effects.json) | 配对主效果、完整来源重采样区间 |
| [reference_comparison_intervals.json](reference_comparison_intervals.json) | 同事实参考相对全局均值／直接参考的差值区间 |
| [prediction_agreement.csv](prediction_agreement.csv) | 预测与分数是否一致 |
| [mechanism_chain.csv](mechanism_chain.csv) | 双向联合修补，归一化恢复和全配对回归，原分母与区间 |
| [m1_head_controls.csv](m1_head_controls.csv) | 选中头与等规模随机头对照 |
| [basis_and_selection_evidence.json](basis_and_selection_evidence.json) | 能量、秩、夹角筛选、局部诊断与开发选择 |
| [signal_probe.json](signal_probe.json) | H3 探针的实际读出结果 |
| [historical_comparison.csv](historical_comparison.csv) | 569 输入按模板、输入来源与参考改变范围统计 |
| [autonomous_paths.csv](autonomous_paths.csv) | 432 例与嵌套 100 例路径、配对变化及来源区间；标注缺失单列 |
| [historical_learned_path_results.csv](historical_learned_path_results.csv) | 旧学习式版本 7458 例官方路径 |
| [historical_learned_pairwise.json](historical_learned_pairwise.json) | 旧学习式与各对照的配对计数、来源区间 |
| [experiment_ledger.json](experiment_ledger.json) | 30 个阶段、实际状态、主要结果和证据入口 |
| [history_sources.json](history_sources.json) | 60 份报告的原始位置与发布位置 |
| [server_inventory.json](server_inventory.json) | 24 个服务器实验目录及报告清单 |
| [definition_pilot/recovery.json](definition_pilot/recovery.json) | 6 节点定义先导构造失败、正式流程未启动 |
| [validation/recompute.json](validation/recompute.json) | 全部主表复算、分母与预测一致性验证 |
| [validation/tests.txt](validation/tests.txt) | 27 项测试通过 |

CSV 同名 JSON 便于程序使用。分数及逐例翻转来自[硬筛选](../aster_existing_data_20261106/README.md)、[连续衰减](../aster_soft_angle_20261106/README.md)、[同事实参考](../aster_reference_anchor_20261106/README.md)三轮完整快照。原始运行的源路径见各目录 publication.json。

[分析报告](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md) · [复算说明](../../experiments/aster_closed_form/SYNTHESIS.zh.md)
