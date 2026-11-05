# 三数据集汇总数据

对应[实验与报告入口](../../reports/three_dataset_analysis_20261105/README.md)。这里保存可直接读取的汇总 CSV/JSON、参数、运行概况和验证结果；逐样本输入及分数的位置见 [artifact_locations.json](artifact_locations.json)。

| 内容 | 文件 |
|---|---|
| 历史 MedPAIR 删除对照 | [CSV](table1_medpair_deletion.csv)、[JSON](medpair_deletion.json) |
| 新评分准确率 | [CSV](table1_medpair_new_accuracy.csv)、[JSON](medpair_new_accuracy.json) |
| 补回与 Full/High 配对差值 | [CSV](table1_medpair_addback.csv)、[JSON](medpair_new_comparisons.json)、[长度统计](medpair_addback_lengths.json) |
| MedGUIDE 分支准确率及生成答案补充 | [CSV](table2_medguide_accuracy.csv)、[JSON](medguide_accuracy.json) |
| MedGUIDE 方案交换 | [CSV](table2_medguide_swaps.csv)、[JSON](medguide_swaps.json) |
| MedEinst M3A 事实响应 | [CSV](table3_medeinst_fact_response.csv)、[JSON](medeinst_m3a.json) |
| MedGUIDE 规则阈值变化 | [CSV](table3_medguide_threshold.csv)、[JSON](medguide_threshold.json)、[树内补充区间](medguide_threshold_within_tree_supplement.json) |
| 两路融合 | [CSV](table4_fusion.csv)、[MedPAIR JSON](medpair_fusion.json)、[MedGUIDE JSON](medguide_fusion.json) |
| 固定融合参数与开发集网格结果 | [MedPAIR](medpair_fusion_parameters.json)、[MedGUIDE](medguide_fusion_parameters.json) |
| 原配置与运行状态 | [配置](config.json)、[完成记录](completion.json)、[补评分完成记录](medpair_completion.json) |
| 验证结果 | [验证记录](verification.json)、[MedGUIDE 配对检查](medguide_pairing_checks.json) |
| 历史补充 | [PCD](medpair_pcd_supplement.json)、[MedGUIDE 历史方法](medguide_historical_supplement.json)、[M1](medeinst_m1_supplement.json)、[Work3](medeinst_work3_supplement.json)、[来源说明](historical_source_notes.json) |

JSON 中比例使用 0–1，准确率差亦为 0–1；文档表格转换为百分比/百分点。分子、分母与 ci95 一并保存。分母为零时 value 和区间为空；CS 只有一棵来源树，跨树区间为空。CSV 展开统计对象为字段后缀 _value、_numerator、_denominator、_ci95 等。

表 1 的历史 High 是 56/89，本轮完整选项 High 是 55/89。融合使用本轮分数。表 2 的 generated_accuracy 是历史自由生成答案，主分析 accuracy 是候选分数最高值预测。

历史 CEBU 50 例报告保存在服务器原运行目录，文件索引给出位置；其与较早 65 例来源重叠，不合并。原始运行产物保持原样；export_repository.py 仅复制指定汇总文件和整理文档链接。
