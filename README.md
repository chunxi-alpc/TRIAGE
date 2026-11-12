# ASTER-CR

研究问题：在保留完整指南和患者事实的条件下，识别并修正非决定性信息对大语言模型当前条件判断的干扰，同时限制对决定性证据响应的损伤。

## 当前研究的两个主文档

仓库当前只以以下两个文档作为论文主线的权威入口：

1. **[论文正文](docs/ASTER/manuscript.zh.md)**：维护研究问题、相关工作、方法、真实实验结果、分析和结论。
2. **[论文实验框架](docs/ASTER/experiments.zh.md)**：维护论文相关实验的设计、数据和模型设置、已完成/部分完成/待执行状态及后续安排。

其他文档只承担代码说明、数据交付、历史方法与实验追溯，不另行维护一份当前论文或当前实验计划。

## 当前实验进度（2026-11-12）

- **10,006 对最终方法实验已完成**：无截距 M3 相对旧 M3 的五折干扰路径提高 0.643 个百分点（95% CI [0.019, 1.310]）；NLI4CT 完整 5,500 条中，E3 相对 E1 提高 0.582 个百分点（[0.127, 1.025]）。M4 改善概率校准，MedPIC 冻结测试点估计提高；完整 E4 尚未同时满足全部预设成功标准。73 项检查通过，详见[最终结果与结论](reports/aster_cr_final_method_10006_20261112/REPORT.zh.md)。



- 后续默认配对训练数据已固定为 **10,006 对均衡配置**：A 4,368、P 635、C 635、F 4,368，F 全来自 MedEinst；标签保持和翻转各 5,003 对。使用固定配对清单及现有权重，入口为 [默认配置](configs/aster_cr_training_default.json)和[固定数据说明](reports/aster_cr_multisource_20261112/balanced_10006/FROZEN_DATASET.zh.md)。
- 均衡配置的 Qwen3-8B 独立拟合与冻结评价已完成：MEGA 路径准确率与 E1 持平，干扰稳定性改善，尚无整体准确率提升的证据。E1 继续作为冻结基线；[本轮结果](reports/aster_cr_multisource_20261112/balanced_10006/TEST_RESULTS.zh.md)与其他诊断实验分别报告。


- MEGA-CDP四类配对训练数据构建已完成。全部合格池6,345对；均衡集每类635对、共2,540对；四类覆盖80个拟合来源。
- Qwen3-8B的E0—E3全量对照和20个来源验证折均已完成；E1保持开发集选定的冻结主方案。另已完成6个独立截距、收益拆分与80来源的2,000次配对重采样。
- 当前结果：配对修正改善分数稳定性，但相对独立截距基线尚未显示稳定正确率收益；均衡权重未显示稳定正确率优势。MedPIC原生保持85对的下降主要发生在加入截距后。详见[结果与结论](reports/aster_cr_diagnostics_e3cv_20261111/REPORT.zh.md)。
- M3截距校准与MedPIC解码诊断已完成：五折仍未显示配对修正超过简单截距的稳定正确率收益；加入原截距使MedPIC保持配对全对率从78.82%降至63.53%（−15.29个百分点，95%区间[−23.95, −7.50]）。开发集选中的集合级校准恢复至74.12%，但降低翻转表现，整题收益不稳定；数量归一化使46道金标准多选题正确率降至0。详见[M3截距与多选解码结果](reports/aster_cr_m3_intercept_diagnostics_20261112/REPORT.zh.md)。
- 主方法保留五个检查点：Qwen3-8B、Llama-3.1-8B-Instruct、Gemma-2-9B-IT、Gemma-3-27B-IT、MedGemma-27B-Text-IT。
- 外部基线采用分层比较：Qwen3-8B完整比较，其他模型只比较关键方法；RePS、PSR是新增候选，须先核验实现和任务适配。
- 原五模型×三数据集×全部基线的285单元全因子方案已被新方案取代，旧文件仅保留作历史审计。

## 数据、代码与历史资料

- [M3截距与多选解码诊断](reports/aster_cr_m3_intercept_diagnostics_20261112/REPORT.zh.md) · [逐例结果与复算](reports/aster_cr_m3_intercept_diagnostics_20261112/README.md)
- [新旧配对训练结果](reports/aster_cr_retrain_paired_v2_20261111/REPORT.zh.md) · [收益拆分与E3五折验证](reports/aster_cr_diagnostics_e3cv_20261111/REPORT.zh.md)
- [配对数据构建报告](reports/aster_cr_pair_dataset_20261111/REPORT.zh.md) · [复现程序](experiments/aster_cr_pair_dataset_20261111/README.md)
- [实验代码](experiments/README.md) · [结果目录](results/README.md)
- [历史方法与实验索引](docs/ASTER/archive/README.md) · [历史分支记录](docs/BRANCHES.md)

GitHub收录程序、模型参数、指标、压缩预测和报告；配对原文、逐条审核记录及大缓存保留在服务器。
