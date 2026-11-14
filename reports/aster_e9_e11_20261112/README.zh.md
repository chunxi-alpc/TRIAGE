# Codex06：E9 与 E11 交付

- [E9机制分析报告](E9_REPORT.zh.md)：冻结E1的选中头与随机头、OV路径、QK冻结、注意力权重和输出投影对照。
- [E11推理效率报告](E11_REPORT.zh.md)：八种方法共320次测量，含ITI与DoRA补测；[合并统计](e11_shared_missing/merged/REPORT.zh.md)。
- configs/：固定参数和方法元数据。为保护医学输入文本，原始提示与逐对输入记录保留在受限服务器目录，不随公开仓库发布。
- checkpoints/：本次使用的E1、LEACE、LoFiT、LoRA参数快照；ITI的冻结算子快照见e11_shared_missing/checkpoints/。
- e9_predictions.jsonl：1,704条去标识化机制干预分数；e11_measurements.jsonl：原六方法240次测量；e11_shared_missing/merged/e11_measurements.jsonl：完整八方法320次测量。
- e9_statistics.json与CSV：机制统计和来源区间；e11_shared_missing/merged/下的JSON与CSV：完整效率统计和测量波动。
- reused/：直接复用的历史机制、提取速度和CPU后处理结果。
- logs/：日志及实现核对记录；初版无效输出不进入正式统计。

服务器工作区：/home/gcx/aster-e9-e11-20261112。

复现入口：bash reports/aster_e9_e11_20261112/reproduce.sh。默认从已发布的数值记录进行CPU统计复现；涉及原始输入文本和模型前向的完整重跑仅在受限服务器工作区执行。GPU执行命令与完成记录保存在e11_shared_missing/merged/executed_codex06_jobs.json。

本轮实验结果、论文正文和实验状态已同步；患者原始叙述不随公开报告发布。
