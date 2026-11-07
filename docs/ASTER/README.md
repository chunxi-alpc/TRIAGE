# ASTER

## 当前方法与已完成结果（2026-11-07）

**保留 M1；沿用此前选定的 M2 及全部 244 个未筛选干扰方向；将 M3 改为可训练映射与门控。** 两种参考版本已各完成 3 轮训练、均选第 1 轮，五方法完成 7,458 例官方测试。辅助参考 7,072 例正确，完整阅读 7,068 例，旧闭式 ASTER 7,045 例。辅助参考相对完整阅读净增 4 例，差值为 +0.054 个百分点；来源重采样 95% 区间为 [−0.014, +0.130] 个百分点，来源配对置换和病例级精确 McNemar 检验均为 p=0.289。结果支持其减少旧闭式编辑的损伤，尚未证明相对完整阅读有稳定纠错收益。

M2 已比较过能量目标与秩上限，现沿用 0.95、4 的已选版本。M1 的 51 个头及 M2 保存基冻结。本轮 M3 训练和评测只用 MEGA-CDP 原版发布记录及标签，200 例训练、100 例开发、7,458 例官方测试，不新增构造样本。

- [首页](../../README.md)：已完成结果、后续状态及历史入口。
- [已完成 M3 报告](../../results/aster_learnable_m3_20261106/REPORT.zh.md)与[P0 直接统计](../../results/aster_m3_followup_20261107/p0/REPORT.zh.md)。
- [P0–P3 后续报告](../../results/aster_m3_followup_20261107/REPORT.zh.md)：新版已实现，训练与全量实验尚未完成，按完成状态单列。
- [方法正文](manuscript.zh.md)：三模块接口、M3 映射与门控、损失、固定设置和五组对照。
- [方法补充](method-clarifications.zh.md)：M2 已有调整、此次修改及版本区别。
- [原闭式正文](manuscript-closed-form.zh.md)与[原方法补充](method-clarifications-closed-form.zh.md)：历史公式和理论原样保留。
- [旧版可学习正文](manuscript-representation.zh.md)、[复现说明](supplementary.zh.md)、[代码与权重](../../experiments/aster_representation/)：89.21% 的历史来源，不作为新 M3 成绩。
- [旧版方法核对](method-check.zh.md)、[旧版结果附件](results/)与[早期方案归档](archive/README.md)。

## 已完成实验与模块分析

[总分析](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md) · [30 个阶段和 60 份历史报告](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) · [复算数据](../../results/aster_synthesis_20261106/README.md) · [闭式代码与复算](../../experiments/aster_closed_form/SYNTHESIS.zh.md)。

M1 定位已有支持；M2 低秩截断损失部分作用，最大进一步损失发生于 M3 的硬夹角筛选。同事实参考缓解全局均值造成的偏移，但尚未超过直接参考。学习式 M3 已完成训练与全量比较，保留原始结果及数据来源；进一步的配对训练和选头对照作为 P0–P3 后续新版。

[原硬筛选实验](../../reports/aster_existing_data_20261106/ANALYSIS.zh.md)、[连续衰减](../../results/aster_soft_angle_20261106/README.md)、[同事实参考](../../results/aster_reference_anchor_20261106/README.md)均为已完成的历史版本。

## 历史补充

[三数据集分析](../../reports/three_dataset_analysis_20261105/README.md)与[修订遮挡实验](../../reports/semantic_masking_followup_20261105/ANALYSIS.zh.md)提供背景证据，其构造输入及任务指标按原协议保留。已完成的原版 M3 不新增配对；P3 后续新版另外使用既有构造配对及不跨阈值的数值对照，结果分别归属。

旧版[完整研究快照](https://github.com/chunxi-alpc/TRIAGE/tree/d8cf67d53d6b068d9b382114de4a5fa4b22f633b)用于追溯已完成实验。

### 历史实验继承与复用

本轮继承既有训练／开发病例和已有 JS、条件翻转、自身条件编辑思路，新增检验的是这些目标在冻结 M1/M2 头子空间中的作用。随机头、子空间消融及校准已有历史结果；旧表示版本 JS 相对无 JS 在 7,458 例上净降 80 例，该负结果保留。已复用相同输入的完整阅读和闭式配对分数，校准由完整阅读缓存计算。详见[历史核对与复用记录](../../results/aster_m3_followup_20261107/HISTORY_REUSE.zh.md)。
