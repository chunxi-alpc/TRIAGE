# ASTER-CR 最终方法

服务器目录：/home/gcx/aster-cr-balanced-publish-20261112/reports/aster_cr_final_method_10006_20261112/

环境：/home/gcx/gcx261014。主流程 code/run.py；正式 NLI 补齐 code/nli_official.py；逐例复算 code/analysis.py；收尾检查和报告 code/finalize.py。

REPORT.zh.md 给出真实结果及七个问题的回答。metrics.csv 保存指标及分母；source_bootstrap.csv 保存 2,000 次来源重采样区间；transitions.csv 保存错误转移。模型与缓存的历史引用只读，新结果均在本目录。

evaluation/ 保存逐例节点、配对、路径和整题结果；predictions/ 保存可重放的路径节点分数及结构；medpic_candidate_scores/ 保存合法集合能量和概率。status/tests.json 为检查结果，completion.json 只在完整评价及检查结束后标记完成。实际时间和显存见 status/。

论文只更新现有 docs/ASTER/manuscript.zh.md 和 docs/ASTER/experiments.zh.md。原稿及首次错误输出分别在 configs/before_* 和 logs/before_output_fix_* 中保留。逐例中的 raw 为原始分数参考，与 E0 相同。

仓库发布范围：报告、指标、置信区间、校准参数、图表、检查结果与复算代码。逐例错误转移使用 transitions.csv.gz 发布，汇总见 transitions_summary.csv。训练文本、完整逐例预测、模型数组和特征缓存保留在上述服务器路径；复算代码依赖这些服务器原始文件。运行和出版版本见 publication_manifest.json。
