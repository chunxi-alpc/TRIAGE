# ASTER-CR 历史研究材料入口

当前论文正文已统一迁移至[docs/ASTER/manuscript.zh.md](../../docs/ASTER/manuscript.zh.md)，当前实验状态与执行计划统一维护于[docs/ASTER/experiments.zh.md](../../docs/ASTER/experiments.zh.md)。

本目录保留2026-11-09以来的历史报告、逐例结果、配置和复算材料。它们用于追溯对应版本的实验，不作为当前论文正文或当前实验计划的第二份维护入口。

## 主要历史结果

- [旧版Qwen受控比较与机制结果](meeting/real_completion/)
- [痕迹预测和来源五折验证](meeting/trace_validation/REPORT.zh.md)
- [新来源修复后续评价](meeting/trace_repair_followup/REPORT.zh.md)
- [旧285单元全因子矩阵（已被取代）](meeting/full_factorial_20261110/README.zh.md)
- [历史模型选择说明](model_selection.zh.md)

## 新数据构建

[2026-11-11配对数据构建报告](../aster_cr_pair_dataset_20261111/REPORT.zh.md)记录Dataset A和Dataset B的规模、独立智能体审核与数据边界。该轮只构建数据，没有完成新数据模型训练。

本目录中的历史数值均按原始数据版本、提示、模型和评价分母解释。若需要更新当前论文结果，请先核对逐例预测和配置，再更新上述两个主文档。
