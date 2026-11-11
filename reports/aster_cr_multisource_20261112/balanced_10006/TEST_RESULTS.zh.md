# 均衡配置测试确认

本页只报告 10,006 对均衡配置与冻结 E1 的比较。

实际训练文件：data/fit_pairs.jsonl。逐条统计为 A 4,368、P 635、C 635、F 4,368；F 全部来自 MedEinst。A/P 均为标签保持，共 5,003 对；C/F 均为标签翻转，共 5,003 对。

模型目录：training/E-Multi-Balanced/full/。model.json 的 training_pair_count 为 10006，M1 的四类计数与训练文件一致；m3.json 的 AP_pairs 和 CF_pairs 均为 5003。模型于北京时间 2026-11-11 23:49 完成拟合，于 23:57 完成评价。

评价加载的是该目录下独立拟合的 M1/M2/M3 参数。另从 MEGA test、new60、MedEinst、NLI4CT、MedPIC 各抽取 16 条输入，用已保存的原始特征和独立加载的模型参数重新计算分数；80 条均与均衡配置预测文件完全一致。复核仅使用 CPU，不改训练数据或参数。

旧 E-Multi 是另一套 10,768 对模型（F 8,863），保留在上一级目录；本页均衡列均来自 E-Multi-Balanced。

|指标|冻结 E1|10,006 对均衡配置|
|---|---:|---:|
|test: natural_node_accuracy|98.925%|98.925%|
|test: natural_path_accuracy|95.285%|95.285%|
|test: interference_path_accuracy|96.485%|96.485%|
|test: ap_score_change_rms|2.612|2.575|
|test: ap_vr|1.460%|4.217%|
|test: cf_correction_rms|0.095|0.115|
|test: cf_er|100.102%|99.821%|
|test: cf_over_budget_rate|2.500%|5.000%|
|new60: natural_node_accuracy|98.965%|98.985%|
|new60: natural_path_accuracy|94.389%|94.514%|
|new60: interference_path_accuracy|94.638%|94.638%|
|medeinst: two_candidate_control_accuracy|63.979%|63.886%|
|medeinst: two_candidate_trap_accuracy|51.848%|51.886%|
|medeinst: two_candidate_both_correct|17.945%|17.871%|
|medeinst: counterfactual_direction_correct|71.261%|71.521%|
|medeinst: cf_correction_rms|0.124|0.072|
|medeinst: cf_er|99.744%|100.026%|
|medeinst: cf_over_budget_rate|1.449%|0.037%|
|medeinst: cf_direction_degradation_rate|0.613%|0.353%|
|nli4ct: natural_native_accuracy|75.000%|74.600%|
|medpic: natural_native_accuracy|35.106%|34.574%|
|medcounterfact: native_original_accuracy|70.202%|70.202%|
|medcounterfact: native_replaced_accuracy|67.047%|67.047%|
|medcounterfact: native_pair_both_correct|64.512%|64.512%|

结论：MEGA test 自然路径和干扰路径准确率与 E1 持平；干扰稳定性和 MedEinst 证据保护有所改善，准确率尚无明确提升。95% 区间与完整说明见 RESULTS_NOTE.zh.md。

VR 指方差降低比例，越高越好。

复核记录：status/configuration_confirmation.json。DDXPlus 扩展已按用户澄清暂停。
