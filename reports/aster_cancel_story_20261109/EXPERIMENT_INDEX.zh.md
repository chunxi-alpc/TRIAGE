# ASTER-CR 实验总索引

更新日期：2026-11-11  
适用分支：aster-cr-20261108  
**本页是 ASTER-CR 实验状态的统一入口。** 具体方法和完整数字以论文及各实验报告为准；同一结果只在一个主报告中定义，其他文件只提供原始数据或复算材料。

## 1. 研究对象是什么？

ASTER-CR 是**输出端决策分数修正方法**。它读取模型内部痕迹，估计干扰对当前判断分数的贡献，再用两类反事实配对训练修正：

- A/P：有效证据保持不变、干扰发生变化，学习需要抵消的分数变化。
- C/F：有效证据发生变化，限制修正项对证据响应的改动。
- 最终分数：原模型分数 + 痕迹修正 + 独立截距。

当前核心线性实例为 16 维正交痕迹、A/P 岭回归、C/F 逐对预算和独立截距。神经网络输出修正、99 维集成及标签监督后续变体均单独命名，不与 16 维纯 A/P 线性实例混为一个方法。

## 2. 已完成实验：一览

| 实验 | 状态 | 主要结论 | 唯一主报告 |
|---|---|---|---|
| 痕迹机制、随机特征、维度与 MLP 比较 | 已完成 | 16 维痕迹开发集 A 类 R²=0.4063；50 个随机 16 维子空间均值 0.3115 | [痕迹验证报告](meeting/trace_validation/REPORT.zh.md) |
| 完整五折来源验证 | 已完成 | 每折重新选头和子空间；AP R²=0.3454，80 个来源中 62 个来源内 R² 为正 | [痕迹验证报告](meeting/trace_validation/REPORT.zh.md) |
| 证据预算与线性/非线性修复 | 已完成 | 预算约束下可降低部分干扰；非线性提高 VR，但通常增加实际证据改动 | [痕迹验证报告](meeting/trace_validation/REPORT.zh.md) |
| Qwen3-8B 受控留出及表示/输出方法对照 | 已完成 | 16 维线性：VR=4.10%、ER=99.60%、证据修正 RMS=0.2539 logit；动态路径 767/806，对照原模型 768/806 | [受控实验与核验](meeting/real_completion/summary.csv)、[论文第4.5节](paper_story_latest.zh.md#45-mega受控比较干扰下降不等于有效修复) |
| Qwen3-8B 参数与配置优化 | 已完成 | 874 个训练配置；在 fit/dev 上选择后评价既有留出和官方测试 | [优化汇总](meeting/optimization/experiment_summary.json)、[优化结果](meeting/optimization/summary.csv) |
| MEGA-CDP 官方 7,458 条路径 | 已完成 | 原模型 7,037/7,458；trace-only 扩展 7,044；优化平衡版 7,041。相对原模型的来源簇差值区间均包含 0 | [官方测试结果](meeting/primary_full/mega_official_test/summary.csv)、[统计区间](meeting/test_expansion/metrics.json) |
| NLI4CT-2024 官方测试冻结迁移 | 已完成 | 原模型 4,073/5,500；优化平衡版 4,048；trace-only 4,040；没有整体准确率收益 | [NLI 官方结果](meeting/primary_full/nli4ct_all/summary.csv) |
| MedPIC 冻结迁移 | 已完成 | 原模型整题全对 98/467；16 维线性 70/467；平衡及强保护 MLP 各 78/467 | [论文第4.8节](paper_story_latest.zh.md#48-medpic证据保持与集合正确性) |
| 60 个新来源的标签监督后续修正 | 已完成，阴性 | 最终来源自然路径差 −0.58 pp，95% CI [−1.25, +0.12]；A 路径差 −0.34 pp，CI [−0.87, +0.13]；未达到预设成功标准 | [后续实验报告](meeting/trace_repair_followup/REPORT.zh.md) |
| NLI/MedPIC 标签监督与截距后续实验 | 已完成，结果混合/阴性 | NLI 的部分配置准确率下降；MedPIC 收益不稳定，截距解释部分配对变化 | [跨任务后续报告](../aster_cross_task_followup_20261110/REPORT.zh.md) |
| 合成机制实验 | 已完成，合成数据 | 用于检验理论机制，不作为真实模型或临床任务效果 | [合成实验报告](simulated_study/REPORT.zh.md) |
| 同等实际证据代价的特征比较 | 已完成，回顾性/探索性 | dev C/F RMS=0.20 时 N/U/S/50随机均值 VR=4.52%/2.00%/1.90%/1.39%；冻结 test 代价失配，尚未建立严格同代价复现优势 | [同代价主报告](meeting/matched_cost_feature_analysis/REPORT.zh.md) |
| 四模型 MEGA 核心复现 | 受控完成，官方路径完成2/4 | 四模型独立状态均有正的 AP VR、平均 ER 接近100%；自然节点正确数无一致提升；Llama与Gemma-2官方已完成，两个27B模型继续 | [跨模型受控数据与配置](meeting/cross_model_mega/)、[执行状态](meeting/EXPERIMENT_STATUS.md) |
| 临床配对人工审核 | 待完成 | 已准备 30 对 A 和全部 32 对 P 的审核材料，当前 README 标记尚未经过医生审核 | [临床审核材料](clinical/README.zh.md) |
| 五模型 × 三任务 × 285 单元统一基准 | 未完成 | 状态表 285 项均为 not_completed_in_common_protocol；不是已完成实验，也不是本轮必须运行的任务 | [矩阵说明](meeting/full_factorial_20261110/README.zh.md)、[状态表](meeting/full_factorial_20261110/matrix_status_285.csv) |

## 3. 最重要的实测结果

### 3.1 内部痕迹是否包含有用信息？

- 所选 51 个注意力头构造的 16 维正交痕迹：开发集 A 类 R²=0.4063。
- 50 个随机 16 维子空间：R² 均值 0.3115。
- 完整五折重新选头与子空间：AP R²=0.3454；62/80 个来源的来源内 R² 为正。
- 32 维线性痕迹 R²=0.5197；16 维 MLP R²=0.4486±0.0040。

解释：所选痕迹含有高于平均随机特征的预测信息，且预测关系能迁移到多数验证来源；但不同来源存在异质性。证据子空间和未正交化干扰子空间的预测 R² 更高，因此正交痕迹是预测—证据保护之间的折中特征，不是“纯干扰表示”。

### 3.2 预测信息能否转化为受约束修复？

预算 0.1 时：

| 数据划分/特征 | 干扰方差降低 VR | 证据响应保留 ER | 证据修正 RMS（logit） |
|---|---:|---:|---:|
| 开发集：所选痕迹 | 4.58% | 99.82% | 0.2029 |
| 开发集：随机子空间 | 0.53% | 99.82% | 0.2077 |
| 既有留出：所选痕迹 | 4.10% | 99.60% | 0.2539 |
| 既有留出：随机子空间 | 1.53% | 99.80% | 0.1506 |

开发集证据改动接近，所选痕迹的干扰降低更大。留出集实际证据改动不同，随机修复仅一个种子，不能将留出差异称为严格同代价确认性优势。

### 3.3 是否提高最终任务准确率？

MEGA 官方测试：

| 方法 | 正确路径 | 相对原模型差值的 95% 来源簇区间 |
|---|---:|---:|
| 原模型 | 7,037/7,458 | — |
| trace-only 平衡扩展 | 7,044/7,458 | [−0.091, +0.269] pp |
| 优化平衡版 | 7,041/7,458 | [−0.114, +0.212] pp |

两个区间都包含零。当前证据支持干扰分数敏感度下降和证据响应受控，尚未证明完整路径准确率稳定提高。

## 4. 方法版本：必须分开报告

| 名称 | 定义 | 对应材料 |
|---|---|---|
| ASTER-CR Linear / 16-d | 16 维正交痕迹 + A/P 岭回归 + C/F 逐对预算 + 独立截距 | trace_validation、real_completion |
| ASTER-CR MLP Output | 同一类痕迹上的非线性输出修正；按种子和证据预算分别报告 | trace_validation、real_completion |
| Optimized Balanced / Strict | 扩展特征、训练配置、集成或校准后的优化版本；不是 16 维线性主方法的同义词 | optimization、primary_full、test_expansion |
| Trace-only Balanced | 仅痕迹相关输入的扩展输出版本；需按保存配置报告 | optimization、primary_full |
| Label-supervised follow-up | 加入判断标签监督的后续变体；不是纯 A/P 差分回归 ASTER-CR | trace_repair_followup、aster_cross_task_followup |
| ASTER M1/M2/M3 | 旧的内部表示编辑路线；与 ASTER-CR 输出端分数修正是不同方法 | docs/ASTER、results/aster_learnable_m3_20261106 |

## 5. 实验文件地图

| 路径 | 内容 | 使用方式 |
|---|---|---|
| paper_story_latest.zh.md | 当前论文正文 | 论文主张和最终表述以此为准 |
| meeting/matched_cost_feature_analysis/ | N/U/S/50随机，同实际证据代价与完整Pareto曲线 | 本轮实验A主结果，回顾性/探索性 |
| meeting/cross_model_mega/ | 四模型各自独立拟合、受控配对与官方路径 | 本轮实验B，官方推理仍在运行 |
| meeting/trace_validation/ | Q1–Q4 痕迹机制、随机对照、五折、预算修复 | 痕迹机制主证据 |
| meeting/real_completion/ | 受控留出、逐对指标、参数和重放核验 | 16 维主方法及表示/输出对照 |
| meeting/optimization/ | 874 个配置、优化搜索、模型与评分结果 | 优化扩展；不是新增独立测试集 |
| meeting/primary_full/ | MEGA 官方路径、NLI 完整数据集预测 | 完整任务实测 |
| meeting/test_expansion/ | 官方测试的来源级统计、置信区间 | 与 primary_full 的同一轮官方结果配套，不是独立重复实验 |
| meeting/trace_repair_followup/ | 新来源标签监督变体 | 单独报告，不并入纯 A/P 主方法 |
| reports/aster_cross_task_followup_20261110/ | NLI/MedPIC 的标签监督、截距诊断后续 | 跨任务后续探索 |
| trace/ | 同批修补传导与预测统计 | 机制补充 |
| clinical/ | 人工审核样本与表格 | 尚待医生审核 |
| simulated_study/ | 合成机制试验及预设模拟材料 | 合成实验与预设模拟严格区分 |
| meeting/full_factorial_20261110/ | 五模型三任务 285 单元的规范与状态表 | 当前未完成；不是已完成结果 |

## 6. 本轮两项实验状态

1. **同等证据代价下的特征比较已完成**：只使用已有真实 Qwen 缓存。开发集证据改动匹配后，正交痕迹比 U、S 与50随机均值的 VR 更高；冻结留出的实际代价不再匹配，不据此宣称严格同代价优势已经复现。
2. **跨模型 MEGA 复现正在完成官方路径推理**：四模型独立的受控结果和随机控制已完成并归档；GPU0/1/5/6分别运行 Llama、Gemma2、Gemma3、MedGemma 的官方7,458条路径。复用已完成的各模型独立特征和参数，不复用 Qwen 内部状态。

当前结果与最终完成状态以 [RESULTS_SUMMARY.md](meeting/RESULTS_SUMMARY.md) 和 [EXPERIMENT_STATUS.md](meeting/EXPERIMENT_STATUS.md) 为准。原始执行清单保留于 [EXPERIMENT_PLAN_CODEX_20261111.zh.md](meeting/EXPERIMENT_PLAN_CODEX_20261111.zh.md)。

## 7. 统一结论

目前证据支持：Qwen3-8B 内部痕迹具有部分来源级预测能力，正交痕迹在开发集匹配证据改动后取得更高的干扰抑制；严格同代价优势尚未在冻结留出得到复现。四个其他模型独立拟合后均观察到受约束的干扰分数抑制，支持核心分数机制的跨模型复现，但自然节点正确数没有一致提高，Llama官方ASTER比原模型多190条正确路径，但仅截距多209条；Gemma-2路径差值区间包含零，两个27B模型仍在运行。跨任务冻结迁移和新来源标签监督后续实验尚未显示稳定的任务准确率收益。

不要将正交投影、R²、VR、ER、节点准确率和路径准确率混成一个“效果”。每个数字只支持其对应层面的结论。
