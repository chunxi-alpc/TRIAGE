# ASTER 文档

[仓库主页](../../README.md)

方法分为**表示编辑**和**分数读出**两条路线。表示编辑从闭式投影发展到可学习 M3 和 N/S/NS 配对编辑；ASTER-CR 使用低维表示修正判断分数，`repair-v2` 标识其数据修复版本。

| 入口 | 内容 |
|---|---|
| [方法变体](versions.zh.md) | 两条路线的动机、公式、实现差别与结果摘要 |
| [模块与训练目标](method-clarifications.zh.md) | 表示编辑的 M1/M2/M3、参考方式；CR 的支持空间、读出与损失 |
| [中文主稿](manuscript.zh.md) | 以 `paired_ns` 为方法的论文稿、实验与分析 |
| [实验索引](experiments.zh.md) | 数据规模、快照状态、完整结果和运行入口 |
| [全量 M3 结果](../../results/aster_learnable_m3_20261106/REPORT.zh.md) | 辅助参考、自身参考等五方法的 7,458 例测试 |
| [P0–P3 阶段报告](../../results/aster_m3_followup_20261107/REPORT.zh.md) | 节点分析、迁移、选头与配对训练 |
| [CR 开发结果](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md) | repair-v2 开发比较、类别计数、诊断和消融 |
| [独立普通读出](../../results/ordinary_readout_20261108/REPORT.zh.md) | 原始节点监督的线性读出与偏置校准测试 |
| [M1 诊断](../../results/m1_diagnostics_20261106/README.md) | 双模型机制分析及原文基础试测 |
| [引用说明](references.zh.md) | 引用与支持范围 |
| [历史与快照](archive/README.md) | 闭式理论、旧表示方法、早期协议及历史首页 |

结果快照：P0–P3 为 **2026-11-07 17:41（北京时间）**；CR 与独立普通读出为 **2026-11-08 07:18:26 UTC**。完成状态统一见[实验索引](experiments.zh.md)。
