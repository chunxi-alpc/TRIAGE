# 历史配对数据审核结果

本窗口按用户最新分工，仅完成旧数据审核与确定性程序验证。3个独立审核智能体使用 GPT-6.1 Sol / xhigh，各负责互不重叠的样本，共审核一轮；未启动新增生成。

## 最终结果

| 类别 | 历史候选 | 智能体通过 | 最终接受 | 指南来源数 | 病例数 | 到500对的缺口 |
|---|---:|---:|---:|---:|---:|---:|
| A | 4,869 | 4,368 | 4,368 | 80 | 911 | 0 |
| P | 18 | 9 | 9 | 6 | 8 | 491 |
| C | 66 | 51 | 40 | 17 | 29 | 460 |
| F | 69 | 50 | 18 | 8 | 13 | 482 |
| 合计 | 5,022 | 4,478 | 4,435 | — | — | — |

最终接受4,435对，排除587对。指南来源按source_id群集计数，各类病例数不能直接相加当作全数据病例数。

## 审核与验证

智能体审核结果：PASS 4,478；FAIL 323；UNCERTAIN 221。程序验证结果：PASS 4,965；UNCERTAIN 57；FAIL 0。57对存在待确定的近重复，其中43对原本通过智能体审核，最终未纳入接受数据。因此，最终接受4,435对，未接受587对。

每对仅有一条正式审核记录，5022个原始pair_id完整保留；基线1597对和三片后续3425对互不重叠。已核实接受与排除文件互斥并完整覆盖全部历史候选。模型配置为Sol/xhigh，正式审核轮次为1。早期Qwen诊断记录未纳入正式审核。

仅使用拟合集。开发、测试来源不进入该数据池。F0共82对单独保存在f0_diagnostic_pairs.jsonl，不合并到四类主数据。

本轮是智能体审核和程序验证，未经过临床专家确认。human_audit_historical.csv提供92对分层人工抽查清单；实际人工审核0对，通过率尚无数据。

## 交付文件

- historical_pairs.jsonl：恢复的5022对历史样本、原始ID和编辑记录。
- review_results_historical.jsonl：全部正式独立审核记录。
- accepted_historical_pairs.jsonl：4435对最终接受样本。
- rejected_historical_pairs.jsonl：587对未接受样本及原因。
- validation_historical.jsonl：逐对程序验证记录。
- pair_statistics_historical.csv：类别、通过数量、覆盖与缺口。
- source_coverage_historical.csv：来源覆盖分布。
- human_audit_historical.csv：待人工抽查清单。
- reports/pair_counts_and_acceptance_historical.png：数量与通过率图。
- reports/source_distribution_historical.png：来源分布图。
- reports/completion_historical.json：历史阶段完成标志。
- reports/generation_gaps.json：仅含最终数量、覆盖与生成缺口，不含生成指导性审核理由。
- dataset_manifest.json、README.md及scripts/：版本、配置与复现程序。

所有文件位于服务器 /home/gcx/aster_cr_pair_dataset_20261111。原始数据和历史实验保持不变，未修改模型结构、训练目标或论文结论。类别均衡主训练集和完整扩充训练池仍须后续生成与独立审核阶段完成后再构建，不在本窗口自动启动。
