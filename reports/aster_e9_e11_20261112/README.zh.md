# Codex06：E9 与 E11 交付

- [E9机制分析报告](E9_REPORT.zh.md)：冻结E1的选中头与随机头、OV路径、QK冻结、注意力权重和输出投影对照。
- [E11推理效率报告](E11_REPORT.zh.md)：六种可运行方法；ITI和DoRA补测已进入统一队列。
- configs/：固定参数、头索引、开发样本、计时输入及训练参数元数据。
- checkpoints/：本次使用的E1、LEACE、LoFiT、LoRA参数快照。
- e9_predictions.jsonl、e11_measurements.jsonl：逐例分数和每次测量。
- e9_statistics.json、e11_statistics.json与CSV：统计汇总、来源区间和测量波动。
- reused/：直接复用的历史机制、提取速度和CPU后处理结果。
- logs/：日志及实现核对记录，初版文件不进入正式统计。

服务器工作区：/home/gcx/aster-e9-e11-20261112。

复现入口：bash reports/aster_e9_e11_20261112/reproduce.sh。已有记录会跳过；新的独立重复须另设输出目录。

仅修改实验代码、报告与docs/ASTER/experiments.zh.md，未修改论文正文。
