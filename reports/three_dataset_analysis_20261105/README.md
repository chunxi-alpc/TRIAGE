# 三数据集实验与结果（2026-11-05）

本轮自动预实验已完成：复用 MedPAIR、MedGUIDE、MedEinst 的已有标注和历史结果，新增标注为 0，只为 MedPAIR 补齐 538 个唯一输入的评分。所有计算均在 115 的 gcx261014 环境完成。

| 阅读目的 | 入口 |
|---|---|
| 查看结论与运行概况 | [完整报告](REPORT.zh.md) |
| 理解实验设计、指标和复现顺序 | [实验说明](EXPERIMENTS.zh.md) |
| 理解结果支持哪些论点 | [结果分析](ANALYSIS.zh.md) |
| MedPAIR 删除与补回 | [表 1](table1.md) |
| MedGUIDE 方案交换 | [表 2](table2.md) |
| 关键事实响应与规则阈值变化 | [表 3](table3.md) |
| 两路输出融合 | [表 4](table4.md) |
| 汇总 CSV、JSON、融合参数及验证记录 | [结果目录](../../results/three_dataset_analysis_20261105/README.md) |
| 实验和整理脚本 | [代码入口](../../experiments/three_dataset_analysis_20261105/README.md) |

主要发现：删除高相关句降低 MedPAIR 的判断准确率，补回优势仍不确定；MedGUIDE 的方案文字可以损害或改善分支判断；MedEinst 中分数方向改善经常不足以改变最终错误预测。融合能够纠正少量共同错误，但效果随数据和变体变化。

本轮是对既有数据的补充分析，真实指南 Q% 暂不填。逐样本输入、完整分数与原始来源保存在服务器运行目录，位置见[文件索引](../../results/three_dataset_analysis_20261105/artifact_locations.json)。仓库保留实验代码、汇总结果、方法说明和分析。
