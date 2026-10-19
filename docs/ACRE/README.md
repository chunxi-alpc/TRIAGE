# ACRE 研究归档

[项目首页](../../README.md) · [文档导航](../README.md) · [结果索引](../../results/README.md)

本目录保留 ACRE 的设计、先导实验和后续评审。它与当前 CARE V6 模块一及 CARE 模块二独立研究分别维护。

**当前归档结论：暂停完整规则 ACRE 的扩展、自标注优化和正式验证。** 在同源参考标注的 80 例主比较中，完整规则固定增强正确 68 例，切换增强正确 66 例。仅目标规则下的三例净纠正不能替代完整规则主比较。依据见 [同源参考对照报告](08-oracle-switch-results.md) 和 [目标切换评审](09-target-switch-review.md)。

## 阅读顺序

| 阶段 | 文档 | 性质 |
| --- | --- | --- |
| 设计 | [01 · 设计评审](01-design-review.md) | 原研究定位与近邻工作 |
| 设计 | [02 · 方法定义](02-method.md) | 方法提案，不代表完整实现已经验证 |
| 设计 | [03 · 实验方案](03-experiments.md) | 原评价计划 |
| 设计 | [04 · 实施与风险](04-plan-and-risks.md) | 原计划与风险记录 |
| 设计 | [05 · 参考核验](05-reference-checklist.md) | 文献与数据声明的核验清单 |
| 先导实验 | [06 · 先导结果](06-pilot-results.md) | 各轮结果与修复情况；[首轮报告](archive/06-pilot-results-round1.md) 单独保留 |
| 后续安排 | [07 · 后续实验](07-next-experiments.md) | 后续条件、实施顺序与停止要求 |
| 已执行诊断 | [08 · 同源参考对照](08-oracle-switch-results.md) | E0/E1 结果与完整规则主比较的负结果 |
| 已存结果复核 | [09 · 目标切换评审](09-target-switch-review.md) | 既有结果解释；文中的六条件机制验证尚未执行 |

## 代码与产物

- [独立实现](../../src/acre_pilot/)：标注、位置映射、注意力分数偏置及评价。
- [同源参考诊断结果](../../results/acre_pilot/oracle_switch_diagnostic/README.md)：结果表、逐例记录和复现说明。
- [目标切换交叉核对](../../results/acre_pilot/oracle_switch_review/cross_scope_audit.json)：后续评审依据。
- [第二轮归档](../../results/acre_pilot/round2/README.md)：解码、标注修复与历史失败记录。

复核结果与未来提案按原状态保留。第 09 篇提出的六条件实验没有因为本次仓库整理而被执行或转为当前默认方法。
