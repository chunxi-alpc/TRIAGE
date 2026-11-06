# ASTER 实验阶段与原始报告索引

整理日期：2026-11-06。登记 30 个阶段条目、60 份历史报告快照；服务器检索范围为 24 个 aster_* 实验目录，另纳入仓库已有专项实验与旧工作区报告。阶段条目不是相互独立的 30 次验证。

| 阶段 | 状态 | 结果及作用 | 来源 |
| --- | --- | --- | --- |
| 2026-10-22 至 24：早期验证与 MedGUIDE | 历史开发阶段 | 保留早期算法及外部基线记录；与后续节点／路径任务分开。 | [证据](history/aster_validation_20261022/private/continuation_20261024/repository_README.md) |
| 2026-10-24：E0–E8 | 已完成 | 多阶段与三种子结果；Qwen ASTER_full 48.408%，直接映射 61.300%；复杂编辑未超过简单对照。 | [证据](history/e0_e8.md) |
| 2026-10-24：E0–E8 开发改进 | 已完成，开发探索 | 单种子 800 条；记录裁剪、分支判断和注意力改进，不能作为独立测试。 | [证据](history/e0_e8_improvement.md) |
| 2026-10-25：P0–P3 | 自动测试完成；人工检查未完成 | 候选准确率 Qwen 55.258%、Llama 56.750%；主要表现为减少损伤，没有准确率优势。 | [证据](history/p0_p3.md) |
| 2026-10-25：动作质量与注意力 | 已完成，开发探索 | 注意力分配改动没有提供稳定优于简单融合的证据；旧零强度数值问题单列。 | [证据](history/action_mass.md) |
| 2026-10-25：动态融合与残差 | 已完成，开发探索 | 同参数残差仅净增 Qwen 1 条、Llama 3 条；区间均跨零。 | [证据](history/dynamic_fusion.md) |
| 2026-10-25：稀疏动态编辑 | 已完成，开发探索 | 501/800；相对同参数无残差纠正 2、退化 2，净零。 | [证据](history/sparse_dynamic.md) |
| 2026-10-27：编码修复后 P1–P6 | P1–P5 完成；P6 仅进度快照 | 4 个开发树，CS 仅 17 例；不能将通过阶段条件写成独立正式效果。 | [证据](history/p1_p6_status.md) |
| 2026-10-30：A/B/C 诊断 | 已保存诊断 | 旧表示与执行链的分段定位，包含有正确参考时的对照，未替代最终任务验证。 | [证据](history/aster_abc_diagnostics_20261030/deliverables/TRIAGE_ABC_report.zh.md) |
| 2026-10-30：全规模流程试运行 | 试运行记录 | 保留两种评分与原生流程试运行，不能当作之后的 7458 例官方成绩。 | [证据](history/aster_fullscale_20261030/run02/smoke02_native/REPORT.zh.md) |
| 2026-10-30：旧论文版本与临床输入执行 | 已保存阶段报告 | run04/run05/clinical01 的方法与执行各自保留，不混用后来模块编号。 | [证据](history/aster_manuscript_20261030/run05/deliverables/ASTER_server_report_20261030.md) |
| 2026-10-31：头诊断 | 保存协议 | 没有单独结果报告的协议不计为已完成性能实验。 | [证据](history/aster_head_diagnostics_20261031/protocol.zh.md) |
| 2026-10-31：层／头迁移对照 | 已保存结果 | 旧模块三的层与头对照；不等同当前闭式 M1 头定位。 | [证据](history/aster_head_transfer_20261031/deliverables/模块三层头对照实验结果_20261031.md) |
| 2026-10-31：长输入修复 | 工程修复 | 保留长输入与缓存行为修复记录，修复状态不等于机制成立。 | [证据](history/aster_long_input_20261031/deliverables/ASTER_长输入修复报告_20261031.md) |
| 2026-10-31：官方长输入续跑 | 执行记录 | 正式路径续跑记录；最终数字归入冻结表示版本。 | [证据](history/aster_official_long_input_20261031/deliverables/ASTER_官方续跑记录_20261031.md) |
| 2026-10-31：规则读取与决策诊断 | 已保存结果 | 旧版错误定位区分读取和决策，属于小规模局部诊断。 | [证据](history/aster_rule_reading_20261031/deliverables/模块三规则读取与决策诊断_20261031.md) |
| 2026-10-31：旧模块三最小对照 | 已保存结果 | 均值、反向等对照用于检查方向解释，不能把单例改善当作特异性证据。 | [证据](history/aster_m3_diagnostics_20261031/deliverables/模块三最小对照实验报告_20261031.md) |
| 2026-10-31：旧瓶颈与模块一修复 | 已保存结果 | 端到端瓶颈、规则抽取与剪裁分别分析；不同于当前闭式模块定义。 | [证据](history/aster_bottleneck_20261031/REPORT.zh.md) |
| 2026-11-01：结构后续诊断 | 已保存结果 | 结构修改有纠正也有退化，保留逐例分析。 | [证据](history/aster_bottleneck_20261031/structure_followup_20261101/regression_analysis_20261101.zh.md) |
| 2026-11-01 至 02：可学习表示方法及官方测试 | 冻结版本完成 | Qwen no_js 6653/7458；直接盲读 6403/7458，来源区间支持其完整路径优势。 | [证据](../../docs/ASTER/results/single-turn.json) |
| 2026-11-02：旧输出语义与 BERTScore | 已有输出复算完成 | DTW_PCC no_js 77.15，盲读 77.57；完整路径优势未扩展到该指标。 | [证据](history/aster_representation_20261101/official_test_20261102/bertscore_local/REPORT.zh.md) |
| 2026-11-04：Gemma 层／头及四输入确认 | 独立确认完成 | 100 组：零干预 85，直接盲读 94，整层 91，选头 85。 | [证据](../../experiments/gemma3_head_pilot/results/expanded_confirmation_20261104.zh.md) |
| 2026-11-04：旧 Qwen 参考交换 | 18 组及 100 组完成 | 一致、相反与行政参考表现接近，未支持正确参考事实独有作用。 | [证据](history/aster_reference_swap_20261104/expanded100/REPORT.zh.md) |
| 2026-11-05：修订遮挡与阈值预实验 | 完成并替代旧口径 | 定义依赖 71/275；实际删除 192/205→193/205；补回 11 例均不变。 | [证据](../semantic_masking_followup_20261105/ANALYSIS.zh.md) |
| 2026-11-05：三数据集补充 | 已完成 | MedPAIR 删除损失存在，补回不确定；MedGUIDE、MedEinst 与融合各按原任务报告。 | [证据](../three_dataset_analysis_20261105/ANALYSIS.zh.md) |
| 2026-11-05：最早闭式实验与失败分析 | 已被修订结果替代 | C 中 7 条不合格、2 条不明确，后续事先移除；旧结果保留追溯。 | [证据](history/aster_closed_form_20261105/CHECK_20261106.zh.md) |
| 2026-11-06：新增定义先导 | 构造失败；正式流程未启动 | 6 节点中 0 个通过构造检查，不是一次可评判的方法效果实验。 | [证据](../../results/aster_synthesis_20261106/definition_pilot/recovery.json) |
| 2026-11-06：现成数据闭式硬筛选 | 已完成 | 主测试宏平均 97.08%，但校准同判；自主路径 401/432 低于完整阅读 404。 | [证据](../aster_existing_data_20261106/ANALYSIS.zh.md) |
| 2026-11-06：连续衰减 | 已完成 | P 修补有所增加，主测试新增 1 个错误，路径降至 399/432。 | [证据](../../results/aster_soft_angle_20261106/REPORT.zh.md) |
| 2026-11-06：未筛选方向与同事实参考 | 机制与历史诊断完成 | 改参考测试子集新增错误 51→6，历史子集 90→6；与直接参考在 610 测试输入全部同判。 | [证据](../../results/aster_reference_anchor_20261106/REPORT.zh.md) |

## 历史报告逐份索引

快照首行保留原路径。原报告的局部文件链接转成路径文字，避免发布后指向错误位置；原文中的旧状态、历史计算资源和已被修订的数字保持原样，当前判断以本目录总分析为准。

### aster_abc_diagnostics_20261030

- [c/REPORT.md](history/aster_abc_diagnostics_20261030/c/REPORT.md)
- [a/audit.md](history/aster_abc_diagnostics_20261030/a/audit.md)
- [deliverables/RUNBOOK.zh.md](history/aster_abc_diagnostics_20261030/deliverables/RUNBOOK.zh.md)
- [deliverables/TRIAGE_ABC_report.zh.md](history/aster_abc_diagnostics_20261030/deliverables/TRIAGE_ABC_report.zh.md)
- [b/REPORT.md](history/aster_abc_diagnostics_20261030/b/REPORT.md)
- [c/oracle01/REPORT.md](history/aster_abc_diagnostics_20261030/c/oracle01/REPORT.md)

### aster_bottleneck_20261031

- [clip-comparison-20261101.zh.md](history/aster_bottleneck_20261031/clip-comparison-20261101.zh.md)
- [REPORT.zh.md](history/aster_bottleneck_20261031/REPORT.zh.md)
- [module_one/IMPLEMENTATION.zh.md](history/aster_bottleneck_20261031/module_one/IMPLEMENTATION.zh.md)
- [module_one/REPORT.zh.md](history/aster_bottleneck_20261031/module_one/REPORT.zh.md)
- [structure_followup_20261101/regression_analysis_20261101.zh.md](history/aster_bottleneck_20261031/structure_followup_20261101/regression_analysis_20261101.zh.md)
- [structure_followup_20261101/PILOT.zh.md](history/aster_bottleneck_20261031/structure_followup_20261101/PILOT.zh.md)
- [structure_followup_20261101/REPORT.zh.md](history/aster_bottleneck_20261031/structure_followup_20261101/REPORT.zh.md)

### aster_closed_form_20261105

- [REPORT.zh.md](history/aster_closed_form_20261105/REPORT.zh.md)
- [CHECK_20261106.zh.md](history/aster_closed_form_20261105/CHECK_20261106.zh.md)
- [analysis_failure/subspaces.zh.md](history/aster_closed_form_20261105/analysis_failure/subspaces.zh.md)
- [analysis_failure/behavior.zh.md](history/aster_closed_form_20261105/analysis_failure/behavior.zh.md)
- [analysis_failure/ANALYSIS.zh.md](history/aster_closed_form_20261105/analysis_failure/ANALYSIS.zh.md)
- [analysis_failure/selection_local.zh.md](history/aster_closed_form_20261105/analysis_failure/selection_local.zh.md)

### aster_fullscale_20261030

- [run02/smoke01/REPORT.zh.md](history/aster_fullscale_20261030/run02/smoke01/REPORT.zh.md)
- [run02/smoke02_native/REPORT.zh.md](history/aster_fullscale_20261030/run02/smoke02_native/REPORT.zh.md)

### aster_head_diagnostics_20261031

- [protocol.zh.md](history/aster_head_diagnostics_20261031/protocol.zh.md)

### aster_head_transfer_20261031

- [protocol.zh.md](history/aster_head_transfer_20261031/protocol.zh.md)
- [deliverables/模块三层头对照实验结果_20261031.md](history/aster_head_transfer_20261031/deliverables/模块三层头对照实验结果_20261031.md)

### aster_long_input_20261031

- [deliverables/ASTER_长输入修复报告_20261031.md](history/aster_long_input_20261031/deliverables/ASTER_长输入修复报告_20261031.md)

### aster_m3_diagnostics_20261031

- [interpretation_plan.zh.md](history/aster_m3_diagnostics_20261031/interpretation_plan.zh.md)
- [protocol.zh.md](history/aster_m3_diagnostics_20261031/protocol.zh.md)
- [deliverables/模块三最小对照实验报告_20261031.md](history/aster_m3_diagnostics_20261031/deliverables/模块三最小对照实验报告_20261031.md)

### aster_manuscript_20261030

- [run04/specification_sections_4_2_to_4_4.md](history/aster_manuscript_20261030/run04/specification_sections_4_2_to_4_4.md)
- [clinical01/deliverables/ASTER_clinical01_four_gpu_report_20261030.md](history/aster_manuscript_20261030/clinical01/deliverables/ASTER_clinical01_four_gpu_report_20261030.md)
- [clinical01/status_snapshot_1793365250/interim_report_20261030.md](history/aster_manuscript_20261030/clinical01/status_snapshot_1793365250/interim_report_20261030.md)
- [run05/deliverables/ASTER_server_report_20261030.md](history/aster_manuscript_20261030/run05/deliverables/ASTER_server_report_20261030.md)

### aster_official_long_input_20261031

- [deliverables/ASTER_官方续跑记录_20261031.md](history/aster_official_long_input_20261031/deliverables/ASTER_官方续跑记录_20261031.md)

### aster_reference_swap_20261104

- [REPORT.zh.md](history/aster_reference_swap_20261104/REPORT.zh.md)
- [matched18/REPORT.zh.md](history/aster_reference_swap_20261104/matched18/REPORT.zh.md)
- [expanded100/REPORT.zh.md](history/aster_reference_swap_20261104/expanded100/REPORT.zh.md)

### aster_representation_20261101

- [confirmation-and-official-test-20261102.zh.md](history/aster_representation_20261101/confirmation-and-official-test-20261102.zh.md)
- [development_interim.zh.md](history/aster_representation_20261101/development_interim.zh.md)
- [existing-prediction-metrics-20261102.zh.md](history/aster_representation_20261101/existing-prediction-metrics-20261102.zh.md)
- [REPORT.zh.md](history/aster_representation_20261101/REPORT.zh.md)
- [comparison-progress-20261101.zh.md](history/aster_representation_20261101/comparison-progress-20261101.zh.md)
- [official_test_20261102/PRIMARY_LAUNCH.zh.md](history/aster_representation_20261101/official_test_20261102/PRIMARY_LAUNCH.zh.md)
- [official_test_20261102/PRIMARY_REPORT.zh.md](history/aster_representation_20261101/official_test_20261102/PRIMARY_REPORT.zh.md)
- [official_test_20261102/LAUNCH.zh.md](history/aster_representation_20261101/official_test_20261102/LAUNCH.zh.md)
- [official_test_20261102/REPORT.zh.md](history/aster_representation_20261101/official_test_20261102/REPORT.zh.md)
- [case_analysis_17/REPORT.zh.md](history/aster_representation_20261101/case_analysis_17/REPORT.zh.md)
- [official_test_20261102/with_js_evaluation/REPORT.zh.md](history/aster_representation_20261101/official_test_20261102/with_js_evaluation/REPORT.zh.md)
- [official_test_20261102/semantic/REPORT.zh.md](history/aster_representation_20261101/official_test_20261102/semantic/REPORT.zh.md)
- [official_test_20261102/bertscore_local/REPORT.zh.md](history/aster_representation_20261101/official_test_20261102/bertscore_local/REPORT.zh.md)

### aster_rule_reading_20261031

- [protocol.zh.md](history/aster_rule_reading_20261031/protocol.zh.md)
- [deliverables/模块三规则读取与决策诊断_20261031.md](history/aster_rule_reading_20261031/deliverables/模块三规则读取与决策诊断_20261031.md)

### aster_validation_20261022

- [private/continuation_20261024/repository_README.md](history/aster_validation_20261022/private/continuation_20261024/repository_README.md)

### e0_e8

- [报告.zh.md](history/e0_e8.md)

### e0_e8_improvement

- [报告.zh.md](history/e0_e8_improvement.md)

### p0_p3

- [报告.zh.md](history/p0_p3.md)

### action_mass

- [REPORT.zh.md](history/action_mass.md)

### dynamic_fusion

- [COMPLETED.zh.md](history/dynamic_fusion.md)

### sparse_dynamic

- [REPORT.zh.md](history/sparse_dynamic.md)

### p1_p6

- [REPORT.zh.md](history/p1_p6.md)

### p1_p6_status

- [RUN_STATUS.zh.md](history/p1_p6_status.md)

## 来源与复算

[服务器目录与报告清单](../../results/aster_synthesis_20261106/server_inventory.json) · [历史报告映射](../../results/aster_synthesis_20261106/history_sources.json) · [结构化阶段表](../../results/aster_synthesis_20261106/experiment_ledger.json)

主表与机制、历史诊断、路径复算见 [总分析](ANALYSIS.zh.md) 和 [复算说明](../../experiments/aster_closed_form/SYNTHESIS.zh.md)。历史报告只保留其证据，不假定所有旧阶段都已完成，也不把没有报告的目录当成成功实验。
