# ASTER 历史方法与实验资料索引

> 本目录及下列历史报告用于方法演变、负结果、实现差异和版本追溯。它们不作为当前论文正文或当前实验状态的维护入口。

当前只维护两个论文主文档：

- [当前论文正文](../manuscript.zh.md)：ASTER-CR完整论文。
- [当前实验框架](../experiments.zh.md)：论文相关实验的设计、完成状态、待执行事项和执行顺序。

## 1. 历史方法稿与方法演变

| 历史资料 | 内容 |
|---|---|
| [ASTER表示编辑旧稿（2026-11-07）](manuscript_representation_20261107.zh.md) | 旧的M1/M2/M3表示编辑路线；不与ASTER-CR分数修正结果混合 |
| [方法版本与公式](../versions.zh.md) | CARE、ACRE、闭式编辑、参考切换、N/S/NS及其他历史版本 |
| [方法澄清](../method-clarifications.zh.md) | 旧方法的实现定义与概念边界 |
| [闭式方法澄清](../method-clarifications-closed-form.zh.md) | 闭式投影和旧版M3的理论、实现差异 |
| [层与头选择历史](../layer-head-selection.zh.md) | 早期层和注意力头筛选研究 |
| [引用资料](../references.zh.md) | 历史方法和相关工作来源 |

## 2. 历史文档与结果快照

以下文件仍保留在原路径，以免破坏已有引用；它们属于历史材料，不是当前论文正文或当前实验状态入口。

| 历史文件 | 内容 |
|---|---|
| [闭式方法论文稿](../manuscript-closed-form.zh.md) | 早期闭式表示编辑路线 |
| [表示方法论文稿](../manuscript-representation.zh.md) | 旧版表示编辑方法 |
| [实现稿](../manuscript-implementation.zh.md) | 旧方法实现与运行说明 |
| [临床结果（2026-10-30）](../clinical-results-20261030.zh.md) | 早期临床任务评价 |
| [中期457结果](../interim-457-results-20261030.zh.md) | 早期阶段性结果快照 |
| [性能结果（2026-10-30）](../performance-results-20261030.zh.md) | 旧协议性能记录 |
| [服务器结果（2026-10-30）](../server-results-20261030.zh.md) | 旧服务器运行汇总 |
| [服务器执行任务](../server-execution-task.zh.md) | 历史运行安排，不作为当前任务队列 |
| [旧方法检查](../method-check.zh.md) | 早期方法核查 |
| [补充材料](../supplementary.zh.md) | 旧路线的推导、协议与结果补充 |
| [来源与数据说明](../sources/README.md) | 历史数据来源及引用记录 |

## 3. 历史实验报告

| 报告 | 研究内容 |
|---|---|
| [ASTER综合分析（2026-11-06）](../../../reports/aster_synthesis_20261106/ANALYSIS.zh.md) | 早期CARE/ACRE/ASTER多阶段分析与研究问题演变 |
| [30阶段与60份报告索引](../../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) | 早期完整实验路线与原始报告入口 |
| [旧数据分析](../../../reports/aster_existing_data_20261106/ANALYSIS.zh.md) | 旧数据和既有模型结果 |
| [ASTER-CR repair-v2](../../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md) | 新版配对训练前的修复实验 |
| [痕迹预测验证](../../../reports/aster_cancel_story_20261109/meeting/trace_validation/REPORT.zh.md) | 旧配对数据下的预测能力、五折来源验证及预算对照 |
| [新来源修复后续](../../../reports/aster_cancel_story_20261109/meeting/trace_repair_followup/REPORT.zh.md) | 新来源冻结评价及失败条件 |
| [旧285单元矩阵说明](../../../reports/aster_cancel_story_20261109/meeting/full_factorial_20261110/README.zh.md) | 已被分层比较方案取代的历史全因子设计 |

## 4. 历史论文与状态快照

- [重构前实验总表快照](experiments_legacy_20261111.zh.md)：重构前的混合式方法/实验索引，仅供追溯。
- [2026-11-07文档快照](20261107/README.md)：整理前的论文、方法及引用材料。
- [历史分支记录](../../BRANCHES.md)：旧研究分支、提交和归档关系。

## 5. 结果、代码和压缩归档

大体积逐例预测、旧模型缓存和历史运行产物保留在原结果目录、实验目录或服务器归档中。可从以下入口定位：

- [实验代码目录](../../../experiments/README.md)
- [结果目录](../../../results/README.md)
- [ASTER图表与生成说明](../figures/README.md)
- [旧研究归档索引](../../../reports/aster_synthesis_20261106/README.md)

## 6. 维护规则

1. 当前论文的论证、方法和实测结果只更新[论文正文](../manuscript.zh.md)。
2. 当前论文实验的设计、状态、未完成事项和后续安排只更新[实验框架](../experiments.zh.md)。
3. 历史方法和实验只追加到本索引所指向的历史文档或原始报告，不回填为当前结果。
4. 旧稿和旧报告中的“当前”“本轮”“待执行”等措辞均按其原始日期解释。
5. 归档只整理入口，不覆盖历史原始数据、配置、代码或负结果。
