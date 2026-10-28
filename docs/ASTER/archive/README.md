# 历史资料与版本追溯

[当前论文](../manuscript.zh.md) · [当前文档入口](../README.md)

## 整理基准与保存方式

本轮以固定提交 [`bc34f439`](https://github.com/chunxi-alpc/TRIAGE/tree/bc34f439649a9a74344195a101a84893ea06abcd) 为基准。以下旧文件从当前主线撤下，但完整内容、相互链接与原始结果仍保存在该提交中；不通过复制出另一套可变文档维护历史，也不改写旧实验产物。

当前职责划分为：模块一离线学习方向，模块二处置盲评估，模块三门控编辑。旧文本中的模块一指当前模块二，旧模块二指当前模块一；历史代码编号按原版本理解。`run_aster_module1`、公共类名和 `aster.action_blind.v1` 保留兼容，不变更运行语义。

并行完成的[机制预实验固定版本 `f3a17f7`](https://github.com/chunxi-alpc/TRIAGE/tree/f3a17f721118407d6a9855d8faf48cbbb90e9770)保存了重排前的实验接入文档及全部原始结果。其[实验报告](../../../experiments/module2_mechanism_20261028/RESULTS.zh.md)对应当前模块一与模块三；`module2_mechanism_20261028`、运行脚本和产物中的旧编号保持冻结。该预实验的实际证据与正文重排分别记录，不将 0.6B 模型的金标读取诊断填入正式大模型主表。

## 旧文档去向

| 旧文件 | 固定版本与当前去向 |
|---|---|
| 主稿 | [整理前全文](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/manuscript.zh.md)；现以[主稿](../manuscript.zh.md)为唯一正文 |
| 模块一确定版 | [原规范](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/module1-spec.zh.md)；方法并入当前模块二，接口见代码说明 |
| 模块二补充 | [原补充](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/module2-supplement.zh.md)；有效推导并入[方法补充](../supplementary.zh.md) |
| 数据协议 | [原协议](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/dataset-protocol.zh.md)；当前口径并入[实验协议](../experiments.zh.md) |
| 基线协议 | [原协议](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/baseline-protocol.zh.md)；当前口径并入实验协议 |
| 指标公式 | [原指标](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/experiment-metrics.zh.md)；当前口径并入实验协议并统一记号 |
| 实验计划 | [原计划](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/experiment-plan.zh.md)；研究结构见主稿，执行口径见实验协议 |
| 空白结果表与示例段落 | [原模板](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/experiment-tables.zh.md)；主稿第 6 节统一结果入口，不再维护重复空表 |
| 历史实测结果 | [历史 H1–H3 原表及限制](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/historical-results.zh.md)；保留原模型、计数、分母及负结果，不填入新版主表 |
| 主张与证伪 | [原清单](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/claims-and-falsification.zh.md)；相关边界随主稿诊断和讨论维护 |
| 设计决策 | [2026-10-27 决策](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/design-update-20261027.zh.md)；不作为额外现行规范 |
| 实现与证据状态 | [原实现状态](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/implementation-status.zh.md) · [原证据状态](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/evidence-status.zh.md)；当前实现状态只维护于[代码说明](../../../src/aster/README.md) |
| 原稿审计与编辑记录 | [原审计](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/audit.zh.md) · [原编辑记录](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/editorial-notes.zh.md) |
| 引用核验 | [原完整核验任务](https://github.com/chunxi-alpc/TRIAGE/blob/bc34f439649a9a74344195a101a84893ea06abcd/docs/ASTER/references-to-verify.zh.md)；主线相关来源见[引用清单](../references.zh.md) |

## 本轮记号修订

$\mathcal S_g$ 始终保留主稿的五元组，可达性 $e_g$ 单独传递；最大概率从概率向量导出。$a_g=1$ 始终表示弃权，因此接受覆盖率与条件执行子集使用 $1-a_g$ 和 $a_g=0$。原指标文档局部将 $a_g=1$ 写作接受，并复用 $e_g$ 表示实际编辑；当前已消除冲突，实际非零键编辑另记 $\xi_g^m$。这些修订统一已有概念，不改变分类器的既有行为。

原始文档的重排候选数与头数均曾记为 K，部署病例数与问题均曾记为 Q。当前实验协议分别使用 $K_{\mathrm{cand}}$ 和 $N_{\mathrm{deploy}}$，不改变预算或算法。

## 更早版本与历史资产

[2026-10-26 ASTER 固定版本](https://github.com/chunxi-alpc/TRIAGE/tree/781ee863784c60f1f6710ba5eede22bc8ee6b93c)包含阶段掩码、证据员、软否决和对数门控，已非当前方法。[更早三模块快照](https://github.com/chunxi-alpc/TRIAGE/tree/0a831d1127c338119bc11712677170486d40aabb)与[支线索引](../../BRANCHES.md)保留 CARE／ACRE 等历史研究。

[旧图稿](../figures/README.md)、原始输入及算术 JSON 保留原路径，避免破坏历史绘图和审计脚本。旧图 2、图 4 不进入当前方法主线。历史源码、配置、实验预测、指标和二进制产物不修改；`2e16df8` 的文档整理本身没有运行新模型实验，独立机制实验按上方固定版本追溯。
