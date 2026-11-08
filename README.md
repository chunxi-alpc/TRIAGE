# ASTER

ASTER 研究模型如何依据完整指南和患者事实判断当前条件，再由程序沿指南树执行诊疗路径。

## 方法

| 路线 | 做法 | 版本与实现 |
|---|---|---|
| 表示编辑 | M1 定位注意力头，M2 估计干扰与信号子空间，M3 修改选中头的表示 | 闭式投影 → 可学习映射与门控 → N/S/NS 配对编辑；论文中的 `paired_ns` 属于此路线 |
| 分数读出 | 冻结基础模型，从选中头的表示提取低维特征，学习最终判断分数的修正量 | ASTER-CR；与普通 PCA 读出、偏置校准比较 |

`repair-v2` 是 ASTER-CR 使用的数据修复版本。P0–P3 是表示编辑路线的实验安排：节点分析、迁移、选头对照和配对训练。

[方法变体与公式](docs/ASTER/versions.zh.md) · [模块与训练目标](docs/ASTER/method-clarifications.zh.md) · [中文论文](docs/ASTER/manuscript.zh.md)

## 结果摘要

结果与运行状态核对时间：**2026-11-08 16:53:16 UTC**。开发集成绩用于选参，正式测试状态另列。

| 实验 | 评价范围 | 结果 | 状态 |
|---|---|---|---|
| 可学习 M3 | MEGA-CDP 测试集，7,458 条完整路径 | 辅助参考 **7,072/7,458（94.82%）**；完整阅读 **7,068/7,458（94.77%）** | 五方法全量测试完成 |
| N/S/NS 配对编辑 | P3；预设主方法 `paired_ns` | 三轮开发 90%、90%、89%，选第1轮；MedGUIDE **103/200（51.5%）**，Full **104/200（52.0%）** | 训练与200题迁移试测完成；7,458路径主测试、522配对诊断未完成 |
| ASTER-CR / repair-v2 | 开发集配对宏平均；420 条完整路径 | CR **79.09%、383/420**；Full **78.26%、384/420**；普通读出 **79.86%、382/420**；LoFiT **79.17%、386/420** | 12组训练与4个开发对照分片完成，配置冻结，test运行；MedGemma及外部评价待完成 |
| 独立普通读出 | 原始测试集，806 条完整路径 | 线性读出与偏置校准均 **767/806**，Full **768/806**；两种读出预测一致 | 测试完成 |

辅助参考 M3 相对完整阅读净增 4 条正确路径，95% 来源重采样差值区间为 [−0.0138, +0.1300] 个百分点。CR 开发宏平均低于普通读出和 LoFiT，正确路径比 Full 少1条；独立普通读出与偏置校准的测试预测相同。各实验使用的划分、指标和分母见对应报告。

[全量 M3 结果](results/aster_learnable_m3_20261106/README.zh.md) · [P0–P3 阶段结果](results/aster_m3_followup_20261107/REPORT.zh.md) · [CR 开发结果](reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md) · [独立普通读出结果](results/ordinary_readout_20261108/REPORT.zh.md)

## 文档入口

| 内容 | 入口 |
|---|---|
| 方法 | [变体对照](docs/ASTER/versions.zh.md)、[模块说明](docs/ASTER/method-clarifications.zh.md)、[中文论文](docs/ASTER/manuscript.zh.md) |
| 运行与进度 | [实验索引](docs/ASTER/experiments.zh.md)、[配对编辑运行说明](experiments/aster_m3_followup/README.zh.md)、[CR 运行说明](experiments/aster_cr/README.zh.md) |
| 数据与结果 | [CR 数据修复](reports/aster_cr_repair_v2_20261108/DATA_REPAIR.zh.md)、[CR 结果与逐例明细](results/aster_cr_repair_v2_20261108/README.md)、[历史实验总索引](reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) |
| 历史资料 | [文档目录](docs/ASTER/README.md)、[TRIAGE / ACRE / CARE 与旧稿](docs/ASTER/archive/README.md)、[实验目录](experiments/README.md) |
