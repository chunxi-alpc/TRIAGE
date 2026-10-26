# 历史结果索引

> **2026-10-26 主线说明：** 当前研究主线为 [ASTER](../docs/ASTER/README.md)。下列既有内容保留原实验阶段含义，其中“当前”“默认”不定义新版 ASTER。实验数字和实现不得未经核验迁入新论文。完整旧主线见[归档分支](https://github.com/chunxi-alpc/TRIAGE/tree/archive/pre-aster-main-20261026)。

[项目首页](../README.md) · [实验索引](../experiments/README.md) · [版本记录](../docs/REPOSITORY_STATUS.md)

此目录主要保存 TRIAGE 原流水线和 ACRE 的历史结果。CARE 模块一 V4–V6 与模块二独立研究的结果位于各自 `experiments/` 目录。

## 最新独立结果

| 内容 | 入口 |
| --- | --- |
| 模块一 V6 | [报告与复现范围](../experiments/care_module1_v6/README.md) · [指标](../experiments/care_module1_v6/data/v6_metrics.json) |
| 模块二有符号诊断 | [实验说明](../experiments/care_module2_signed_20261019/README.md) · [完整结果](../experiments/care_module2_signed_20261019/results/analysis/README.md) |
| 模块二 B1–B4 后续 | [结果报告](../experiments/care_qwen3_4b_base_20261016/results/module2_followup/README.md) |

## TRIAGE 历史流水线诊断

以下是 2026-10-14、旧模块一 `module1.v1` 与 Qwen3-8B 的 24 例诊断。它不是当前 V6 流水线的重新运行结果。

| 内容 | 文件 |
| --- | --- |
| 最终摘要与指标定义 | [algorithm_diagnostic/summary.json](algorithm_diagnostic/summary.json) |
| 24 例判定、概率、召回与错误 | [cases.csv](algorithm_diagnostic/cases.csv) |
| 两例自由文本补充 | [free_generation.json](algorithm_diagnostic/free_generation.json) |
| 固定输入与样本选择 | [manifest.json](algorithm_diagnostic/manifest.json) · [data_audit.json](algorithm_diagnostic/data_audit.json) |
| 独立离线参考 | [references.json](algorithm_diagnostic/references.json) |
| 逐例完整模块输出 | [algorithm_diagnostic](algorithm_diagnostic/) 中的 `case_*.json` |
| 先前阶段 | [round1](algorithm_diagnostic/round1/) · [round2](algorithm_diagnostic/round2/) |
| 最初三模块集成 | [历史说明](../TRIAGE_MVP.md) · [triage_mvp_report.json](triage_mvp_report.json) |

最终正确动作选择为：无规则条件原始模型 15/24；完整指南原始模型 13/24；关闭干预的 TRIAGE 提示 16/24；完整 TRIAGE 15/24。完整 TRIAGE 相对无规则基线纠正 4 例，同时使 4 例退化。该阶段没有整体准确率净提升。详细比较及单独列出的参考规则诊断上界见 [最终摘要](algorithm_diagnostic/summary.json)。

## 当前仍复用的历史组件

| 目录 | 作用与来源 |
| --- | --- |
| [module2](module2/) | 候选查询头、校准对和报告；来自旧模块一缓存及非测试校准指南 |
| [module3](module3/) | rank=4 投影、模型与校准指纹、CPU/GPU 检查 |
| [pipeline](pipeline/) | 历史集成冒烟结果；新运行应使用新的输出文件或目录 |
| [diagnostic](diagnostic/) | 历史组件和门控补充检查 |

关键文件：[组件](module2/components.json)、[模块二报告](module2/report.json)、[投影参数](module3/projections.pt)、[投影元数据](module3/projections.json)、[干预检查](diagnostic/module23_checks.json)。复用这些产物不等于已经用 V6 重新校准或完成端到端评价。

## ACRE 归档

| 内容 | 入口 |
| --- | --- |
| 所有阶段的解释 | [ACRE 文档索引](../docs/ACRE/README.md) |
| 第二轮及修复记录 | [acre_pilot/round2](acre_pilot/round2/README.md) |
| 同源参考主比较 | [oracle_switch_diagnostic](acre_pilot/oracle_switch_diagnostic/README.md) |
| 目标切换后续核对 | [cross_scope_audit.json](acre_pilot/oracle_switch_review/cross_scope_audit.json) |

完整规则路线的负结果和停止要求继续保留；未执行的后续提案不会因为目录整理而变成实验结果。

## 环境、数据与准备记录

根目录中的模型、数据下载、环境和验证 JSON 记录早期环境准备与组件检查。它们保留原文件名和原内容，便于追溯；不能把下载完成、解析成功或冒烟通过作为模型性能结论。

模型权重、完整环境、受限原文和大规模原始日志依照各轮约定保留在服务器。后续结果使用新目录，保留来源、模型指纹和参数；不得覆盖旧预测、旧统计或校验清单。
