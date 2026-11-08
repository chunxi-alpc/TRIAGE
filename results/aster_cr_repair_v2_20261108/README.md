# repair-v2 结果快照

快照：2026-11-08 16:53:02 UTC。当前阶段：qwen_test_features_batch0。12组训练和4个开发分片完成，Qwen已冻结；后续评价未全部完成。

[分析报告](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md) · [数据修复报告](../../reports/aster_cr_repair_v2_20261108/DATA_REPAIR.zh.md) · [独立普通读出](../ordinary_readout_20261108/REPORT.zh.md)

- snapshot.json：快照时间、运行阶段、各部分完成状态。
- development_analysis.json：全部 Qwen 开发读出/消融、类别区间、路径差值与宏平均差值。
- training_development_analysis.json：12组训练指标、选择、对CR差值及区间。
- comparator_development_analysis.json：全部开发对照及模板结果。
- qwen/：已经产生的机制、诊断、开发预测、路径明细、成本与选参记录。
- data/：冻结分母、类别数量、配对及输入/节点元数据。
- external/：两套外部数据的构造与任务数量；尚无模型正式结果。
- status/、jobs/：当前运行记录；running 状态不是已完成结果。
- verification/：两模型前向验证；gpu_expansion/：四卡扩容记录。

JSONL 明细按 gzip 压缩，解压后可按 group_id、source_id、view_id 和 case_id 复算。未改变数据划分或开发选择。

按仓库现有发布方式，模型权重、激活缓存、完整输入、第三方代码及运行日志保留在服务器；这里同步代码、结果和分析。服务器来源：/home/gcx/aster_cr_repair_v2_run_20261108。构造数据来源：/home/gcx/aster_cr_repair_20261108。

更新快照仅做 CPU 汇总：python -m experiments.aster_cr.export_results --run-root /home/gcx/aster_cr_repair_v2_run_20261108 --destination /path/to/repository 。该命令不运行模型、不选参、不自动推送。
