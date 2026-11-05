# 三数据集补充分析

自动预实验，复用已有标注和历史结果；零新增标注。本轮为对既有数据的补充分析，不声称测试集此前未使用。真实指南 Q% 暂不填。

MedPAIR 测试删除对照：A_High=+4.49 [+0.00, +10.11] 个百分点（n=89），A_Low=+16.09 [+8.05, +25.29] 个百分点（n=87）。高相关句补回恢复量为 +3.45 [+0.00, +10.34] 个百分点；相对不相关句补回差为 +3.45 [-6.90, +13.79] 个百分点（n=29）。补回收益较小、区间包含零，尚不足以支持稳定优势。相关性标签不等同于语义依赖标签。

MedGUIDE 方案交换的影响随变体而异：B、D 降低可见条件的准确率，C 提高准确率；遮挡后对应输入与分数不变。因此可支持方案文字影响条件判断，不能概括为所有方案文字都造成损害。MedEinst 三个模型的测试配对失误率为 96.88%–99.61%，预测不变率为 62.89%–66.80%；部分病例分数已经向正确方向变化，但仍未选对。这些结果不用于归因到特定训练阶段。

融合可以纠正部分共同错误。MedGUIDE 原始 A 的学习式融合准确率为 73.27% (477/651)，纠正双错 2.14% (3/140)，同时对至少一路正确的样本造成错误 7.24% (37/511)；相对完整输入准确率变化为 +2.61 [+0.36, +4.63] 个百分点。C 变体的概率平均也纠正了 1/127 个共同错误。MedPAIR 本轮四种选定融合均未纠正 28 个共同错误；融合效果依数据和变体而变。

历史 MedPAIR High 为 56/89，本轮完整选项重评分为 55/89；Full 均为 60/89。两批结果分别呈现：历史删除对照不变，融合统一使用本轮分数；不推测差异原因。84 组规则阈值变化均来自同一棵指南树，只报告该来源结果，不估计跨树区间。

所有实验均在 115 的 gcx261014 环境运行，使用 GPU 0、1。MedPAIR 共 540 条条件记录、538 个唯一输入；20 道开发题的 55 次试跑评分已计入。其余分析完全离线完成，GPU 进程已正常退出。

| 阶段 | GPU | 完成评分 | 含加载秒数 | 峰值预留显存/GiB |
|---|---|---|---|---|
| full | 0 | 241 | 18.31 | 16.05 |
| full | 1 | 242 | 66.46 | 16.46 |
| pilot | 0 | 55 | 9.66 | 15.70 |

四张主表：

- [表1：MedPAIR 删除与补回](table1.md)
- [表2：MedGUIDE 方案交换](table2.md)
- [表3：事实响应与规则阈值变化](table3.md)
- [表4：输出融合](table4.md)

服务器结果目录：/home/gcx/three_dataset_analysis_20261105/results。代码入口：/home/gcx/work3-migration-20261104/target/experiments/three_dataset_analysis_20261105。数据入口：/home/gcx/experiment_data_reuse_20261105/data_inventory.json。

逐条输入、来源、预测和完整分数保存在 medpair_complete_views.jsonl、../data/medguide_scores_and_inputs.jsonl、medeinst_independent_predictions.jsonl；固定划分见 ../data/medguide_split.json 和各样本 split 字段；融合参数见 [medpair_fusion_parameters.json](../../results/three_dataset_analysis_20261105/medpair_fusion_parameters.json)、[medguide_fusion_parameters.json](../../results/three_dataset_analysis_20261105/medguide_fusion_parameters.json)。

统计：种子 42、2,000 次配对 bootstrap。MedPAIR 按题、MedEinst 按病例对、MedGUIDE 按指南树重采样。随机删除对照在题内先平均。概率只在有效选项内归一化。四项融合测试及样本配对、划分和概率检查已通过，见 [verification.json](../../results/three_dataset_analysis_20261105/verification.json)。

旧内部干预、PCD、CEBU、自回归 CAD、M1 与 Work3 结果分别保存为补充文件，未训练内部模块，未合并不同协议。

[实验与结果入口](README.md) · [实验说明](EXPERIMENTS.zh.md) · [结果分析](ANALYSIS.zh.md)
