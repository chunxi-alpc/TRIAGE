# ASTER-CR diagnostics and E3 five-fold validation

- [中文报告](REPORT.zh.md)
- [实验配置](configuration.json) / [旧结果引用](references.json)
- [独立截距及拟合来源](independent_intercepts.json)
- [全部指标](metrics.csv) / [来源重采样差值区间](source_bootstrap.csv)
- [纠正与误伤](transitions.csv) / [MedPIC选项分布](medpic_option_counts.csv)
- [M2子空间重叠](m2_subspace_overlap.json) / [M3约束余量](m3_constraint_margins.csv)
- [验证汇总](verification.json) / [独立分数复算](independent_score_verification.json)

E3五折参数在 models/E3/。逐样本和完整路径预测在 evaluation/，发布格式为 .jsonl.gz。
每个评价目录同时包含区分消融版本的分数NPZ及其元数据。五折合并目录引用各折未参与训练的验证预测。
历史报告和旧模型保持原样。本轮E1保持冻结，未增加样本或启动新方法。
