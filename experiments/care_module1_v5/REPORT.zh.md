# CARE 模块一 V5 实验报告

后续解析器 V2 已对同一批缓存输出进行离线修复和复评；TrialMatchAI 条件级及规则级结果以 [PARSER_V2_REPORT.zh.md](PARSER_V2_REPORT.zh.md) 为更新结果。本报告及原始结果保留作为解析器 V1 基线。

报告生成日期（UTC）：2026-10-18

新模块一使用 Phi-4 基础模型及 TrialMatchAI LoRA 条件判断、BioLinkBERT-MedNLI 证据推断和确定性三值逻辑。Qwen3 只读取 V4 已保存对照，新模块一不调用它。

## 实际运行状态

服务器运行状态：`completed`。已完成 192/192 条。GPU `NVIDIA A100-SXM4-80GB`，设备 `cuda:6`，设备容量 79.279 GiB。开始 `2026-10-18T08:31:21.364948+00:00`，结束 `2026-10-18T08:31:31.908492+00:00`。

192 条回归样本包括 32 条原始合成病例和 160 条控制变体，涉及 16 条研究规则，每类场景各 32 条。主规则共 41 个原子条件、0 条例外，共 492 个逐样本条件请求。该集合已用于 V5 方法设计，属于探索性回归。

实际资源口径：本状态只描述 BioLinkBERT 证据重放阶段；TrialMatchAI 原始生成身份与资源另见下文单卡档案和四卡清单。

TrialMatchAI 已记录适配器：`['default']`；LoRA 参数数 42598400。BioLinkBERT 标签编号由实际配置读取：`{'0': 'entailment', '1': 'neutral', '2': 'contradiction'}`。

## V5 与历史直接判定的统一评价

| 方法 | 判定正确/准确率 | 宏 F1 | 已知证据定位 | 判定与证据同时正确 | 反事实成对正确 | 生成/后处理累计分摊工作秒 |
|---|---:|---:|---:|---:|---:|---:|
| A：TrialMatchAI + 三值逻辑 | 60/192 (31.25%) | 0.2251 | 不适用：无句级证据 | 不适用 | 0/32 (0.00%) | 6178.096 |
| A+数值：TrialMatchAI + 程序比较 | 63/192 (32.81%) | 0.2461 | 5/137 (3.65%) | 25/192 (13.02%) | 0/32 (0.00%) | 6178.128 |
| B：TrialMatchAI + BioLinkBERT + 程序比较 | 104/192 (54.17%) | 0.5254 | 76/137 (55.47%) | 53/192 (27.60%) | 7/32 (21.88%) | 6182.837 |
| C：V4 已保存 Qwen3 直接判定 | 136/192 (70.83%) | 0.7041 | 87/137 (63.50%) | 81/192 (42.19%) | 22/32 (68.75%) | 79.198 |

A 只使用 TrialMatchAI 条件标签和三值逻辑。A+数值另加入可识别的数值、区间及单位比较。B 在同一 TrialMatchAI 输出上增加 BioLinkBERT 证据判断。纯 A 没有句级证据；A+数值的证据指标仅来自程序数字句；B 才纳入 NLI 证据。V5 复用 V4 手工结构规则，历史 Qwen3 直接组接受自由文本研究规则；这是系统方案对比，不能把结构规则的额外信息与全部性能变化都归因于专业模型本身。

## 六类场景（每类 32 条）

| 场景 | A：TrialMatchAI | A+数值 | B：加 BioLinkBERT | C：历史 Qwen3 |
|---|---:|---:|---:|---:|
| 完整原病例（32 条） | 4/32 (12.50%)；预测 U=29 | 9/32 (28.12%)；预测 U=26 | 14/32 (43.75%)；预测 U=12 | 28/32 (87.50%)；预测 U=4 |
| 条件翻转（32 条） | 4/32 (12.50%)；预测 U=28 | 2/32 (6.25%)；预测 U=30 | 16/32 (50.00%)；预测 U=15 | 25/32 (78.12%)；预测 U=0 |
| 信息缺失（32 条） | 21/32 (65.62%)；预测 U=31 | 21/32 (65.62%)；预测 U=31 | 23/32 (71.88%)；预测 U=21 | 11/32 (34.38%)；预测 U=4 |
| 多值与不同主体（32 条） | 2/32 (6.25%)；预测 U=30 | 2/32 (6.25%)；预测 U=30 | 12/32 (37.50%)；预测 U=14 | 21/32 (65.62%)；预测 U=0 |
| 否定表达（32 条） | 0/32 (0.00%)；预测 U=32 | 0/32 (0.00%)；预测 U=32 | 10/32 (31.25%)；预测 U=17 | 25/32 (78.12%)；预测 U=0 |
| 同次评估矛盾（32 条） | 29/32 (90.62%)；预测 U=29 | 29/32 (90.62%)；预测 U=29 | 29/32 (90.62%)；预测 U=29 | 26/32 (81.25%)；预测 U=26 |

完整原病例指数据集原有合成病例文字全部保留，不表示真实患者。每格同时列正确数和预测未知数量，避免只用总体准确率掩盖未知误判。

## 错误类别

| 错误类别 | A：TrialMatchAI | A+数值 | B：加 BioLinkBERT | C：历史 Qwen3 |
|---|---:|---:|---:|---:|
| 已知 T/F 判为 U | 128 | 126 | 62 | 4 |
| 未知 U 判为 T/F | 4 | 3 | 9 | 25 |
| 错误拒绝 T→F | 0 | 0 | 16 | 9 |
| 错误接受 F→T | 0 | 0 | 1 | 18 |
| 已知样本证据定位错误 | 不适用 | 132 | 61 | 50 |

B 相对 A+数值：纠正 51 条，损害原正确判断 10 条；判定改变 78 条，其中 U→T/F 74 条、T/F→U 4 条。该比较覆盖 192 条，运行未完整时只能视为中间状态。

TrialMatchAI 条件请求 492 条，完成 492 条；解析错误 417 条，达到生成长度上限 29 条；生成 token 282370。解析错误细分：`{'expected_exactly_one_target_criterion': 312, 'invalid_json_response': 75, 'generation_truncated': 29, 'invalid_final_decision': 1}`。截断不自动视为已得到有效标签。

固定接口接受 75/492 条，拒绝 417/492 条。当前准确率是固定协议下的端到端结果，不能单独估计模型的医学判断能力；不用诊断性提取的标签重新评分。

本次已完成运行的结构诊断与真实失败样本见 [解析与失败诊断](ANALYSIS_PARSE.zh.md)。主要格式问题包括非目标空占位、JSON 围栏及长生成截断。

汇总错误比较见 [错误分类分析](ANALYSIS_ERRORS.zh.md)；该补充报告只使用既有汇总指标。

## 失败样本与已有诊断

下表只摘录已保存的错误类别与条件诊断，不据此虚构病历语义原因。完整记录包含预测证据编号、足够参考证据组合、冲突编号和全部条件状态。

| 样本 ID | 方法 | 场景 | 金标→预测 | 已保存错误类别 | 已保存条件诊断 |
|---|---|---|---|---|---|
| `C26_contradiction` | B：TrialMatchAI + BioLinkBERT + 程序比较 | contradiction | U→F | 未知 U 判为 T/F | c1: TrialMatchAI=U, 数值=U, NLI=T, 最终=T, 解析错误=invalid_json_response, resolution=mednli_resolved_unknown；c2: TrialMatchAI=U, 数值=U, NLI=F, 最终=F, 解析错误=generation_truncated, 达生成长度上限, resolution=mednli_resolved_unknown |
| `C05_counterfactual` | B：TrialMatchAI + BioLinkBERT + 程序比较 | counterfactual | F→U | 已知 T/F 判为 U | c1: TrialMatchAI=U, 数值=U, NLI=F, 最终=U, 解析错误=expected_exactly_one_target_criterion, 记录有模型冲突, resolution=numeric_observation_missing；c2: TrialMatchAI=U, 数值=U, NLI=F, 最终=U, 解析错误=expected_exactly_one_target_criterion, 记录有模型冲突, resolution=numeric_observation_missing |
| `C01_multiple_values` | B：TrialMatchAI + BioLinkBERT + 程序比较 | multiple_values | T→U | 已知 T/F 判为 U | c1: TrialMatchAI=U, 数值=U, NLI=F, 最终=F, 解析错误=expected_exactly_one_target_criterion, resolution=mednli_resolved_unknown；c2: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, resolution=unresolved_numeric_observation |
| `C01_negation` | B：TrialMatchAI + BioLinkBERT + 程序比较 | negation | T→U | 已知 T/F 判为 U | c1: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, 记录有模型冲突, resolution=unresolved_model_evidence_conflict；c2: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, resolution=unresolved_numeric_observation |
| `C02_original` | B：TrialMatchAI + BioLinkBERT + 程序比较 | original | T→U | 已知 T/F 判为 U | c1: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, 记录有模型冲突, resolution=unresolved_model_evidence_conflict；c2: TrialMatchAI=U, 数值=U, NLI=T, 最终=U, 解析错误=expected_exactly_one_target_criterion, 记录有模型冲突, resolution=numeric_observation_missing |
| `C01_partial_missing` | B：TrialMatchAI + BioLinkBERT + 程序比较 | partial_missing | U→F | 未知 U 判为 T/F | c1: TrialMatchAI=U, 数值=U, NLI=F, 最终=F, 解析错误=expected_exactly_one_target_criterion, resolution=mednli_resolved_unknown；c2: TrialMatchAI=U, 数值=F, NLI=F, 最终=F, 解析错误=expected_exactly_one_target_criterion, resolution=deterministic_numeric |
| `C29_contradiction` | B：TrialMatchAI + BioLinkBERT + 程序比较 | contradiction | U→F | 未知 U 判为 T/F | c1: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, 记录有模型冲突, resolution=unresolved_model_evidence_conflict；c2: TrialMatchAI=U, 数值=U, NLI=F, 最终=F, 解析错误=expected_exactly_one_target_criterion, resolution=mednli_resolved_unknown |
| `C30_contradiction` | B：TrialMatchAI + BioLinkBERT + 程序比较 | contradiction | U→F | 未知 U 判为 T/F | c1: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, 记录有模型冲突, resolution=unresolved_model_evidence_conflict；c2: TrialMatchAI=U, 数值=U, NLI=F, 最终=F, 解析错误=expected_exactly_one_target_criterion, resolution=mednli_resolved_unknown |
| `C06_counterfactual` | B：TrialMatchAI + BioLinkBERT + 程序比较 | counterfactual | F→U | 已知 T/F 判为 U | c1: TrialMatchAI=U, 数值=U, NLI=F, 最终=U, 解析错误=invalid_json_response, 记录有模型冲突, resolution=numeric_observation_missing；c2: TrialMatchAI=U, 数值=U, NLI=F, 最终=U, 解析错误=invalid_json_response, 记录有模型冲突, resolution=numeric_observation_missing |
| `C07_counterfactual` | B：TrialMatchAI + BioLinkBERT + 程序比较 | counterfactual | F→U | 已知 T/F 判为 U | c1: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, resolution=numeric_observation_missing；c2: TrialMatchAI=U, 数值=U, NLI=U, 最终=U, 解析错误=expected_exactly_one_target_criterion, resolution=numeric_observation_missing |

## 实际耗时与显存

| 阶段或资源 | 实际记录 |
|---|---|
| 当前证据重放墙钟（仅 BioLinkBERT 加载和证据处理） | 10.544 秒 |
| 证据重放 GPU / device | NVIDIA A100-SXM4-80GB / cuda:6 |
| 证据重放 BioLinkBERT 加载墙钟 | 2.847 秒 |
| 仅证据重放峰值 allocated / reserved | 0.648 / 0.672 GiB |
| TrialMatchAI 累计生成分摊工作秒（复用单卡 + 四卡） | 6178.096 秒 |
| 四卡阶段墙钟（最早 worker 开始至最晚结束，含加载） | 1313.020 秒 |
| 最初单卡设备 / 状态 | cuda:6 / interrupted_for_requested_parallelization |
| 最初单卡开始 / 中断记录（UTC） | 2026-10-18T07:15:15.750077+00:00 / 2026-10-18T08:06:32.570424+00:00 |
| 最初单卡墙钟（已记录开始至中断） | 3076.820 秒 |
| 最初单卡已刷盘条件 / 合并复用条件 | 256 / 256 |
| 最初单卡 Phi-4 + LoRA 加载墙钟 | 29.331 秒 |
| 最初单卡所有模型峰值 allocated / reserved | 未采集 / 未采集 GiB；未记录时不估计 |
| 四卡设备 | cuda:4, cuda:5, cuda:6, cuda:7 |
| 四卡各自生成阶段最大单卡 allocated 峰值 | 13.859 GiB（cuda:5） |
| 四卡各自生成阶段最大单卡 reserved 峰值 | 15.848 GiB |
| 最初单卡开始到证据重放结束墙钟（含中断、调度、加载和间隔） | 4576.158 秒；不可视为稳定吞吐基准 |
| CPU 用户 / 系统时间 | 未采集 / 未采集 秒 |
| CUDA 内核纯设备计时 | 未采集 秒 |

| worker | GPU / 设备 | 条件数 | 加载墙钟秒 | 进程墙钟秒 | allocated 峰值 GiB | reserved 峰值 GiB |
|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA A100-SXM4-80GB / cuda:4 | 64 | 35.602 | 1313.020 | 13.795 | 15.826 |
| 1 | NVIDIA A100-SXM4-80GB / cuda:5 | 60 | 35.522 | 842.382 | 13.859 | 15.848 |
| 2 | NVIDIA A100-SXM4-80GB / cuda:6 | 56 | 35.592 | 520.055 | 13.775 | 14.561 |
| 3 | NVIDIA A100-SXM4-80GB / cuda:7 | 56 | 30.462 | 680.054 | 13.814 | 14.992 |

生成来源 `parallel_remaining_run`：236 个条件，累计分摊工作 3206.847 秒。

生成来源 `reused_original_run`：256 个条件，累计分摊工作 2971.249 秒。

证据重放不加载 Phi-4 或 TrialMatchAI；适配器身份从已记录的原生成进程继承并验证。原生成与四卡各自的加载和显存不能从 NLI-only 状态推断。累计分摊工作秒是同步生成调用的墙钟工作量，虽然状态字段名含 gpu_seconds，也不是 CUDA 内核设备时间。

最初单卡原始状态档案：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_serial_initial/run_status.json`。

| 证据处理项目 | 已保存实际值 |
|---|---:|
| 数值比较累计墙钟 | 0.032 秒 |
| 候选句筛选累计墙钟 | 0.069 秒 |
| BioLinkBERT 调用累计墙钟 | 4.477 秒 |
| 全部证据处理累计墙钟 | 4.741 秒 |
| B：原生成工作秒 + 后处理墙钟 | 6182.837 秒 |

CPU 用户/系统时间和 CUDA 内核计时未保存时列“未采集”；模型调用墙钟包含主机开销。原先单卡先完成的输出保持不变，剩余条件按原 batch=8 分配四卡；每个原始批内样本顺序、提示和 token 上限不变。中断且未刷盘的批次不纳入结果，也不估计其无效计算时长。

四卡累计生成工作秒不能当作四卡墙钟，证据重放墙钟不能当作全实验运行时间，NLI 重放阶段峰值也不能当作 Phi-4 + LoRA 生成峰值。Qwen3 历史单卡直接判定与 V5 的模型规模、量化、生成长度、逐条件请求和批量协议不同，不报告等价硬件速度或加速比。

## 未使用规则上的新合成探索集

运行状态：`completed`。冻结数据为 4 条完整树规则、24 条自编病例、11 个原子条件和 66 个条件请求。来源指南 ID 为 42、44、80、53，AND/OR/NOT、原始条件局部位置及动作位置已在运行前验证；未读取病历正确路径或答案作为输入。

| 方法 | 判定正确/准确率 | 宏 F1 | 已知证据定位 | 判定与证据同时正确 | 反事实成对正确 | 生成/后处理累计分摊工作秒 |
|---|---:|---:|---:|---:|---:|---:|
| A：TrialMatchAI + 三值逻辑 | 10/24 (41.67%) | 0.3624 | 不适用：无句级证据 | 不适用 | 0/4 (0.00%) | 338.679 |
| A+数值：TrialMatchAI + 程序比较 | 10/24 (41.67%) | 0.3624 | 0/16 (0.00%) | 4/24 (16.67%) | 0/4 (0.00%) | 338.679 |
| B：TrialMatchAI + BioLinkBERT + 程序比较 | 14/24 (58.33%) | 0.6159 | 16/16 (100.00%) | 12/24 (50.00%) | 2/4 (50.00%) | 339.604 |

新集条件请求 66 条；解析失败 43 条，达到生成长度上限 0 条。细分：`{'expected_exactly_one_target_criterion': 30, 'invalid_json_response': 13}`。

该集资源使用单独记录：

| 阶段或资源 | 实际记录 |
|---|---|
| 当前证据重放墙钟（仅 BioLinkBERT 加载和证据处理） | 5.760 秒 |
| 证据重放 GPU / device | NVIDIA A100-SXM4-80GB / cuda:6 |
| 证据重放 BioLinkBERT 加载墙钟 | 2.240 秒 |
| 仅证据重放峰值 allocated / reserved | 0.637 / 0.650 GiB |
| TrialMatchAI 累计生成分摊工作秒（四卡生成） | 338.679 秒 |
| 四卡阶段墙钟（最早 worker 开始至最晚结束，含加载） | 133.635 秒 |
| 四卡设备 | cuda:4, cuda:5, cuda:6, cuda:7 |
| 四卡各自生成阶段最大单卡 allocated 峰值 | 13.174 GiB（cuda:4） |
| 四卡各自生成阶段最大单卡 reserved 峰值 | 13.301 GiB |
| CPU 用户 / 系统时间 | 未采集 / 未采集 秒 |
| CUDA 内核纯设备计时 | 未采集 秒 |

| worker | GPU / 设备 | 条件数 | 加载墙钟秒 | 进程墙钟秒 | allocated 峰值 GiB | reserved 峰值 GiB |
|---|---|---:|---:|---:|---:|---:|
| 0 | NVIDIA A100-SXM4-80GB / cuda:4 | 18 | 32.212 | 133.635 | 13.174 | 13.301 |
| 1 | NVIDIA A100-SXM4-80GB / cuda:5 | 16 | 33.554 | 115.539 | 13.174 | 13.299 |
| 2 | NVIDIA A100-SXM4-80GB / cuda:6 | 16 | 33.043 | 123.010 | 13.174 | 13.299 |
| 3 | NVIDIA A100-SXM4-80GB / cuda:7 | 16 | 33.499 | 110.090 | 13.174 | 13.299 |

生成来源 `parallel_remaining_run`：66 个条件，累计分摊工作 338.679 秒。

证据重放不加载 Phi-4 或 TrialMatchAI；适配器身份从已记录的原生成进程继承并验证。原生成与四卡各自的加载和显存不能从 NLI-only 状态推断。累计分摊工作秒是同步生成调用的墙钟工作量，虽然状态字段名含 gpu_seconds，也不是 CUDA 内核设备时间。

新集没有数值阈值，A+数值不会产生程序数值证据；因此其已知样本证据得分为 0 或联合得分只来自未知样本协议时，反映该组缺少定位输出，不能作为独立证据模型性能，也不能与 B 的联合正确率作等价能力比较。纯 A 的定位与联合指标不适用。新集是模型运行前固定的明确合成事实，标签由规则逻辑确定，未经临床专家复核。它检验新规则上的基础条件与逻辑判断，不能作为真实病历独立临床验证。源树许可证待明确，第三方树和动作仅留服务器；公开自编病例、标签、行号、SHA256 和重建脚本。

## 四个问题的回答

**第一，TrialMatchAI 单独判断的效果如何？**
A 的判定正确为 60/192 (31.25%)，宏 F1 0.2251，反事实成对正确 0/32 (0.00%)。未知误判已知 4 条、已知误判未知 128 条、错误拒绝 0 条。
在保留完整原病例的 32 条样本上，正确 4/32 (12.50%)；这 32 条仍是数据集的合成病例记录。
增加程序数值比较后正确 63/192 (32.81%)，相对纯模型准确率变化 +1.56 个百分点；因此需分别评价专业模型和数值程序的贡献。

**第二，加入 BioLinkBERT 后，证据与最终判断是否改善？**
B 的准确率为 54.17%，相对 A+数值变化 +21.35 个百分点；宏 F1 从 0.2461 变为 0.5254。
逐例比较纠正 51 条、损害原正确判断 10 条，共改变 78 条判断。
B 的已知样本证据定位为 76/137 (55.47%)，判定与证据同时正确为 53/192 (27.60%)。A+数值只提供数值句证据，其证据定位为 5/137 (3.65%)；纯 TrialMatchAI 不生成句级证据，不能把该组的缺失定位指标写成 0%。
已知 T/F 判为 U：A+数值 126 条 → B 62 条。
未知 U 判为 T/F：A+数值 3 条 → B 9 条。
错误拒绝 T→F：A+数值 0 条 → B 16 条。
错误接受 F→T：A+数值 0 条 → B 1 条。
完整病例组正确数由 9/32 变为 14/32。
以上为正确数边际比较；现有汇总指标不含样本交集，不能据此计算原正确病例的保留或损害数量。

**第三，相对历史 Qwen3，准确率、证据和效率有什么变化？**
历史 Qwen3 直接判定准确率 70.83%、宏 F1 0.7041。A 相对它的准确率变化为 -39.58 个百分点；B 相对它变化为 -16.67 个百分点。
已知样本证据定位：Qwen3 87/137 (63.50%)，B 76/137 (55.47%)，差 -8.03 个百分点。联合正确率差 -14.58 个百分点；反事实成对正确率差 -46.88 个百分点。
已记录累计工作计时：Qwen3 历史单卡生成 79.198 秒，A 6178.096 秒，B 6182.837 秒。模型大小、量化、输出长度、条件请求数量和批量协议不同；V5 此处是原先单卡与随后四卡条件生成按批量分摊的累计工作秒，B 再加证据后处理墙钟；不是用户等待的端到端墙钟，也不是 CUDA 内核设备秒。Qwen3 是历史单卡规则级生成墙钟，两者口径与批量不同，不能据此推断纯模型、硬件速度或四卡加速比。

**第四，是否可以作为 CARE 正式模块一？**
当前适合作为可复现的研究实现，不能作为正式模块一替换旧方案。主要问题是固定接口接受率低、已知条件被大量判为未知，BioLinkBERT 补回部分未知时也引入错误拒绝；组合方案的准确率、证据定位和完整病例表现仍低于历史对照。V4 的 16 条研究规则缺少完整原始指南动作，192 条样本已用于设计；新增探索集也只是明确合成事实，没有独立临床专家金标。离线手工结构准备也是部署所需的来源与维护负担。接口兼容已验证，正式应用仍需先修复格式接口并用重新冻结的新数据评价，同时补齐原始指南动作和独立临床标注。
新增合成探索集 A 正确 10/24 (41.67%)，B 正确 14/24 (58.33%)；该结果支持的范围是这些新规则上的明确事实判断，不能升级为临床可靠性结论。

## 方法与适用范围

TrialMatchAI 是 Phi-4 的生成式 LoRA 适配器，不是现成医学三分类接口。入组标签 Met→T、Not Met→F、Unclear/Irrelevant→U；排除侧 Not Violated 不证明条件为假，保守映射 U。主 16 条规则没有 exclusion criterion，新合成集的 NOT 也不是 TrialMatchAI 官方排除标签；排除侧映射目前只有单元测试，尚无真实排除标准的系统实验。适配器加载失败不回退普通 Phi-4 或 Qwen3。

患者按原文句子保留 Python 字符区间 [start,end)，先筛候选句，再由 BioLinkBERT 推断支持、反驳或中立。标签编号读取模型配置；中立不表示阴性。模型冲突保留原始意见和证据，无法解决时设 U。

数值逻辑使用固定字段的数值、区间、单位及明确撤回标记。它没有通用时间线解析或完整的主体消歧能力；复杂旧记录、新记录与其他患者信息仍可能处理错误。无法提取的数值保留 U，不把缺失默认为阴性。程序已经确定的数值比较不因 NLI 意见不同而改成相反结论，分歧仍保存；具体范围以冻结的 care_v5.py 为准。

V4 研究规则是条件改写，16 条均缺少可独立核验的完整原始指南动作；其规则字符位置属于预备改写文本，不冒充原始指南位置。原数据 path 列是病例正确路径，未作为在线指南动作输入。新增探索集才含来源完整树的可重建动作片段。推理冻结文件完整性记录见 `experiments/care_module1_v5/provenance/freeze_integrity_check.json`。模块二/三兼容结构通过接口测试并不等于完成真实临床三模块效果验证。

模块三原有先验冲突分支依赖模块一的闭卷/指南动作比较。V5 不再调用 Qwen3 生成这两种意见，因此该信号明确不可用，prior_competition_weight 为 0。适用性和证据的上下文接口保留，但不能声称原先的全部先验抑制功能已由新模型替代；模块二、三代码没有修改。

## V4 其他已保存参考结果

| 历史方法 | 正确数/准确率 | 宏 F1 | 已知证据定位 | 联合正确 | 反事实成对正确 | 历史生成秒数 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 直接判定 | 136/192 (70.83%) | 0.7041 | 87/137 (63.50%) | 81/192 (42.19%) | 22/32 (68.75%) | 79.198 |
| Qwen3 结构化不核验 | 140/192 (72.92%) | 0.7472 | 56/137 (40.88%) | 68/192 (35.42%) | 25/32 (78.12%) | 672.935 |
| Qwen3 结构化加核验 | 140/192 (72.92%) | 0.7673 | 68/137 (49.64%) | 93/192 (48.44%) | 23/32 (71.88%) | 1331.011 |

历史来源：`/home/gcx/care_module1_20261018_v4/data/comparison.csv and REPORT.zh.txt`，仅读取保存结果，没有重新调用 Qwen3。

## 对应文件位置

- 主运行状态：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_20261018/run_status.json`；最初单卡状态：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_serial_initial/run_status.json`。
- 主指标：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_20261018/metrics.json`；逐例错误：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_20261018/errors_public.jsonl`。
- 逐例公开预测：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_20261018/predictions_public.jsonl`，报告仅汇总其中已记录的耗时字段。
- 原始模型输出与原文证据：服务器 `/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_20261018/private/predictions_private.jsonl`；合并条件原始生成 `/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_parallel_generation/private/merged_trialmatch.jsonl`；合并计时清单 `/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/runs/v5_parallel_generation/private/timing_summary_private.json`。
- 新探索指标：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/exploratory/results/metrics.json`；新探索状态：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/exploratory/runs/frozen_v1/run_status.json`。
- 新探索逐例错误：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/exploratory/results/errors.jsonl`；冻结记录：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/exploratory/data/FREEZE.json`。
- V4 标签和配对：`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/data/labels_public.jsonl`、`/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/data/pairs_public.jsonl`；离线规则 `/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5/data/rules_v5.jsonl`。
- 解析诊断：`ANALYSIS_PARSE.zh.md`、`provenance/parse_failure_diagnostics.json`；公开完整性检查：`provenance/completed_artifact_integrity.json`。
- V4 原文输入和全部历史结果保留 `/home/gcx/care_module1_20261018_v4/`，没有覆盖。
