# Codex06：E9 与 E11 交付

- [E9机制分析报告](E9_REPORT.zh.md)：冻结E1的选中头与随机头、OV路径、QK冻结、注意力权重和输出投影对照。
- [E11推理效率报告](E11_REPORT.zh.md)：八种方法共320次测量，含ITI与DoRA补测；[合并统计](e11_shared_missing/merged/REPORT.zh.md)。
- configs/：固定参数、头索引、开发样本、计时输入及训练参数元数据。
- checkpoints/：本次使用的E1、LEACE、LoFiT、LoRA参数快照。
- e9_predictions.jsonl：1,704条机制干预分数；e11_measurements.jsonl：原六方法240次测量；e11_shared_missing/merged/e11_measurements.jsonl：完整八方法320次测量。
- e9_statistics.json与CSV：机制统计和来源区间；e11_shared_missing/merged/下的JSON与CSV：完整效率统计和测量波动。
- reused/：直接复用的历史机制、提取速度和CPU后处理结果。
- logs/：日志及实现核对记录，初版文件不进入正式统计。

服务器工作区：/home/gcx/aster-e9-e11-20261112。

复现入口：bash reports/aster_e9_e11_20261112/reproduce.sh。默认从保存的记录进行CPU统计复现；GPU执行命令与完成记录保存在e11_shared_missing/merged/executed_codex06_jobs.json。

仅修改实验代码、报告与docs/ASTER/experiments.zh.md，未修改论文正文。
