# ASTER 连续衰减对照

[原运行报告](REPORT.zh.md) · [固定配置](configuration.json) · [主表](results/main_comparison.csv) · [总分析](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md)

保留既有头、基、均值和数据划分，将硬夹角筛选替换为连续衰减。主测试四类宏平均与硬筛选同为 97.08%，但全部输入新增 1 个错误；432 例自主路径正确 399，比硬筛选少 2。详见 [机制对照](results/mechanism_comparison.csv)、[路径对照](results/path_comparison.csv)、[所有预测翻转](results/all_prediction_flips.csv)、[检查汇总](results/verification_summary.json)。

scores、operators、mechanisms、diagnostics、paths 保存本轮分数与产物；输入和 M1 基础产物复用 [原实验快照](../aster_existing_data_20261106/README.md)。大 JSONL 压缩为 .jsonl.gz，读取函数自动兼容 gzip；CSV 未压缩。[publication.json](publication.json) 记录服务器来源与文件对应关系。完整运行保留在 /home/gcx/aster_soft_angle_20261106。
