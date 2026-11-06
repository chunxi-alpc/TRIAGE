# ASTER

## 闭式方法现成数据实验（2026-11-06，已完成）

[结果与问题分析](../../reports/aster_existing_data_20261106/ANALYSIS.zh.md) · [固定配置、原始分数与表格](../../results/aster_existing_data_20261106/README.md) · [代码及复算](../../experiments/aster_closed_form/EXISTING_DATA.zh.md)。本轮主配对宏平均小幅提高，历史事实诊断和自主路径退化；机制诊断显示低秩截断及夹角筛选损失了大部分可修补干扰效应。论文中的方法有效性表述应依据这些正负结果，不能仅由算子代数性质推出实际优势。

[当前论文正文](manuscript.zh.md)整合临床背景、单步条件判断、四类配对测试、两项挑战、三项机制假设与三模块方法。新版使用冻结模型中的头定位、反事实子空间估计和信号保持擦除；机制比例与方法效果待对应实验补齐。

- [首页概述](../../README.md)：完整研究叙事与入口。
- [新版正文](manuscript.zh.md)：引言、节点规约、三模块完整计算、斜投影与条件化输出融合命题、验证实验。
- [方法补充说明](method-clarifications.zh.md)：本次补齐的接口、数据用途、理论条件及旧版入口。
- [旧版正文与完整结果](manuscript-representation.zh.md)：可训练处置盲表示干预，MEGA-CDP 完整路径正确率89.21%。
- [旧版方法核对](method-check.zh.md)、[旧版方法补充与复现](supplementary.zh.md)、[旧版代码与冻结权重](../../experiments/aster_representation/)。
- [旧版结果附件](results/)：官方测试配对计数、BERTScore 汇总和18节点反事实结果。

## 三数据集补充分析（2026-11-05）

[实验说明、结果与分析](../../reports/three_dataset_analysis_20261105/README.md)比较信息删除与补回、方案干扰、关键事实响应和输出融合。该批结果复用已有标注和历史预测，可为新版配对实验提供背景，不作为新闭式方法的结果。

旧版的[固定实验代码与检查点](https://github.com/chunxi-alpc/TRIAGE/tree/d8cf67d53d6b068d9b382114de4a5fa4b22f633b/experiments/aster_representation_20261101/release)用于追溯已完成对照。此前规则键编辑方案见[归档](archive/README.md)。
