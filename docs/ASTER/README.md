# ASTER

## 当前方法方案（2026-11-06）

**保留 M1；沿用此前选定的 M2 及全部 244 个未筛选干扰方向；仅将 M3 改为可训练映射与门控。** 新 M3 尚未实现、训练和验证。

M2 已比较过能量目标与秩上限，现沿用 0.95、4 的已选版本。M1 的 51 个头及 M2 保存基冻结。本轮 M3 训练和评测只用 MEGA-CDP 原版发布记录及标签，200 例训练、100 例开发、7,458 例官方测试，不新增构造样本。

- [首页](../../README.md)：最新方案、实验安排及历史入口。
- [方法正文](manuscript.zh.md)：三模块接口、M3 映射与门控、损失、固定设置和五组对照。
- [方法补充](method-clarifications.zh.md)：M2 已有调整、此次修改及版本区别。
- [原闭式正文](manuscript-closed-form.zh.md)与[原方法补充](method-clarifications-closed-form.zh.md)：历史公式和理论原样保留。
- [旧版可学习正文](manuscript-representation.zh.md)、[复现说明](supplementary.zh.md)、[代码与权重](../../experiments/aster_representation/)：89.21% 的历史来源，不作为新 M3 成绩。
- [旧版方法核对](method-check.zh.md)、[旧版结果附件](results/)与[早期方案归档](archive/README.md)。

## 已完成实验与模块分析

[总分析](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md) · [30 个阶段和 60 份历史报告](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) · [复算数据](../../results/aster_synthesis_20261106/README.md) · [闭式代码与复算](../../experiments/aster_closed_form/SYNTHESIS.zh.md)。

M1 定位已有支持；M2 低秩截断损失部分作用，最大进一步损失发生于 M3 的硬夹角筛选。同事实参考缓解全局均值造成的偏移，但尚未超过直接参考。后续按已确定的方案改造 M3，保留原始结果及数据来源。

[原硬筛选实验](../../reports/aster_existing_data_20261106/ANALYSIS.zh.md)、[连续衰减](../../results/aster_soft_angle_20261106/README.md)、[同事实参考](../../results/aster_reference_anchor_20261106/README.md)均为已完成的历史版本。

## 历史补充

[三数据集分析](../../reports/three_dataset_analysis_20261105/README.md)与[修订遮挡实验](../../reports/semantic_masking_followup_20261105/ANALYSIS.zh.md)提供背景证据，其构造输入及任务指标按原协议保留。本轮不新增配对或修复旧构造病例。

旧版[完整研究快照](https://github.com/chunxi-alpc/TRIAGE/tree/d8cf67d53d6b068d9b382114de4a5fa4b22f633b)用于追溯已完成实验。
