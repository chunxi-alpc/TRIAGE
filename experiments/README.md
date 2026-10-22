# 实验索引

[项目首页](../README.md) · [模块一](../src/module1/README.md) · [模块二](../src/module2/README.md) · [模块三](../src/module3/README.md)

本目录保存独立实验的代码、协议与结果。当前模块实现及其接口见 `src/`；实验结论按原模型、数据、筛选条件和运行阶段解读。

**模块一研究信号已冻结为 [语义 A1](care_module1_semantic_v8/MODULE1_A1_FREEZE.zh.md)**。这里保存提示、固定示例、批量运行和 A1→模块二信号导出入口；完整历史 GPU 输出仍在 PR #12 的来源分支，不在当前主线重复导入。旧 V6 CLI 和流水线继续保留历史契约。

## ASTER MedGUIDE 反事实实验

[20例真实模型先导实验](aster_medguide_20261022/PILOT.md) · [数据构建](aster_medguide_20261022/prepare_dataset.py) · [模型实验](aster_medguide_20261022/run_pilot.py) · [统计分析](aster_medguide_20261022/analyze_pilot.py)。实际结果见该目录 results/REPORT.zh.txt，不能用历史 Qwen 结果替代。

已有[强度扫描预实验入口](aster_medguide_20261022/README.md)保留；两套输入和输出格式分别使用各自的构建器、运行器和分析器，不能混用。

## 当前优先入口

| 方向 | 最新入口 | 样本与模型 | 已存结论 |
| --- | --- | --- | --- |
| 模块一（研究默认） | [A1 冻结与接口](care_module1_semantic_v8/MODULE1_A1_FREEZE.zh.md) · [原研究 PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12) | Phi-4＋TrialMatchAI LoRA、三个固定 few-shot、直接 T/F/U | 作为模块二固定信号源，U 不编辑，不用金标挑选样本；不声称已具临床准确性 |
| 模块一（旧 V6） | [CARE V6](care_module1_v6/README.md) | 旧 192 条离线重算；新增 100 条；Phi-4 + TrialMatchAI LoRA、BioLinkBERT-MedNLI | 原有适用性与证据门控接口保留用于复现，78.65% / 73.00% 是旧样本结果 |
| 模块二研究 | [有符号诊断](care_module2_signed_20261019/README.md) | 384 个模拟病例、192 对、12 个规则家族；Qwen3-4B-Base | 尚未建立学习方向优于直接分数偏置的证据 |

模块一指标来自 [V6 指标文件](care_module1_v6/data/v6_metrics.json)。模块二结论来自 [保存结果](care_module2_signed_20261019/results/analysis/README.md)。两者的样本和评价目标不同，不能相互替代或合并为端到端结果。

## 模块一版本链

| 阶段 | 入口 | 保留目的 |
| --- | --- | --- |
| V4：结构化证据修复 | [目录说明](care_module1_v4/README.md) · [原报告](care_module1_v4/REPORT.zh.txt) | Qwen3-8B 三组对照，保留改善与新增错误 |
| V5：模型组合 | [目录说明](care_module1_v5/README.md) · [初始报告](care_module1_v5/REPORT.zh.md) | 保留初始解析失败、组合退化与错误分析 |
| V5：解析器 V2 | [解析器复评](care_module1_v5/PARSER_V2_REPORT.zh.md) | 对缓存的 492 条输出做离线解析修复 |
| V5：解析器 V3 | [失败条件重生成报告](care_module1_v5/runs/v5_20261018/parser_v3_retry/RETRY_REPORT.zh.md) | 仅重生成仍失败的 69 条条件，保留其余原输出 |
| V6：证据定位与真假判定分离 | [目录说明](care_module1_v6/README.md) · [原报告](care_module1_v6/REPORT.zh.md) | V6 历史入口；旧样本离线重算与新增样本运行分别保存 |

V4/V5 为历史比较。V5 解析成功率提升并未证明组合判定有效；不要只引用修复后覆盖率而省略组合方案的退化。V6 保留主判定，将 BioLinkBERT 用于证据定位与干预门控；实际门控限制见 [当前模块说明](../src/module1/README.md)。

## CARE 三项观察与模块二后续

| 阶段 | 入口 | 研究范围 |
| --- | --- | --- |
| 2026-10-16：三项观察 | [实验目录](care_qwen3_4b_base_20261016/README.md) · [更新报告](care_qwen3_4b_base_20261016/results/observations_summary.md) | 临床来源规则与模拟患者的 O1；虚构规则的 O2/O3 |
| 2026-10-17：O2 二次统计复核 | [头选择复核](care_qwen3_4b_base_20261016/results/module2_o2_reanalysis/README.md) | 读取既有逐头统计，检查选择规则与数值混杂；没有新模型前向 |
| 2026-10-17：B1–B4 后续 | [完整报告](care_qwen3_4b_base_20261016/results/module2_followup/README.md) | 家族留出对照、随机投影、直接分数偏置与因果干预诊断 |
| 2026-10-19：有符号适用性诊断 | [目录说明](care_module2_signed_20261019/README.md) · [原协议](care_module2_signed_20261019/PROTOCOL.md) | 用参考适用性标签比较原始 B4、有符号 B4、均值位移和直接偏置 |

后两轮结果没有建立可靠的学习子空间优势。可解码性、注意力分数变化和最终决策改善是不同问题。使用参考适用性标签的诊断不包含模块一预测错误，也不是完整流水线的可部署性能。

## 其他历史研究

| 内容 | 入口 |
| --- | --- |
| TRIAGE 原三模块集成与 24 例诊断 | [结果索引](../results/README.md) · [历史最小实现](../TRIAGE_MVP.md) |
| ACRE 先导、同源参考对照与停止结论 | [文档索引](../docs/ACRE/README.md) · [诊断产物](../results/acre_pilot/oracle_switch_diagnostic/README.md) |

## 保存与复现约定

每轮保留协议、数据来源及划分、模型版本、参数、预测、指标、失败样本和校验清单。后续运行使用新目录；原始结果与哈希清单原样保留。目录名表示实验阶段，后续补充不应被误读为同一次运行。

能够从已保存预测重新计算统计量，不等于能够在任意机器重新运行模型。受限原文、模型权重和部分大张量保留在既有服务器或对应归档中；具体依赖见各实验说明和 [运行说明](../docs/RUNNING.md)。
