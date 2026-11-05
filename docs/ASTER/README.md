# ASTER

## 三数据集补充分析（2026-11-05）

[实验说明、结果与分析](../../reports/three_dataset_analysis_20261105/README.md)比较信息删除与补回、方案干扰、关键事实响应和输出融合。该批结果复用已有标注和历史预测，与上方表示干预主实验保持各自协议。

[论文正文](manuscript.zh.md)介绍处置盲表示干预及其单轮评测结果。[方法核对](method-check.zh.md)说明文字描述与当前最好结果的对应关系。[方法补充](supplementary.zh.md)给出监督信号、配置选择和复现步骤。[主方法代码](../../experiments/aster_representation/)提供冻结权重与运行入口。

[结果附件](results/)提供官方测试的配对计数、全量 BERTScore 汇总和18节点反事实实验的逐节点结果。[固定实验代码与检查点](https://github.com/chunxi-alpc/TRIAGE/tree/d8cf67d53d6b068d9b382114de4a5fa4b22f633b/experiments/aster_representation_20261101/release)可用于追溯对照实现。

此前的规则键编辑方案可从[归档](archive/README.md)查阅。
