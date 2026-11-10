# ASTER-CR：当前研究入口

**本目录只讨论 ASTER-CR：利用模型内部痕迹，在证据保护约束下修正最终判断分数。**

- [当前论文正文](paper_story_latest.zh.md)
- [实验总索引：已做什么、结果是什么、哪些还没做](EXPERIMENT_INDEX.zh.md)
- [10 分钟汇报稿](meeting/meeting_brief.zh.md)
- [Codex 剩余实验清单](meeting/EXPERIMENT_PLAN_CODEX_20261111.zh.md)

## 当前结论

1. **机制已验证（Qwen3-8B）**：16 维正交痕迹在开发集 A 类配对上的预测 R²=0.4063，高于 50 个随机 16 维子空间均值 0.3115。完整五折重新选头的来源验证 AP R²=0.3454，62/80 个来源内 R² 为正。
2. **受约束修复已验证**：既有留出上，16 维线性方法将干扰方差降低 4.10%，平均证据响应保留 99.60%，证据修正 RMS 为 0.2539 logit。
3. **整体任务准确率优势尚未建立**：MEGA 官方测试中，原模型为 7,037/7,458；trace-only 扩展为 7,044/7,458，优化平衡版为 7,041/7,458。相对原模型的来源簇置信区间均包含零。
4. **跨任务迁移没有稳定收益**：NLI4CT 和 MedPIC 的冻结迁移结果均未显示整体准确率改善。
5. **跨模型泛化尚未测试完成**：当前完整机制与主实验主要来自 Qwen3-8B。

## 已完成实验在哪里？

| 路径 | 内容 |
|---|---|
| meeting/trace_validation/ | 随机特征、维度、MLP、完整五折来源验证、证据预算修复 |
| meeting/real_completion/ | 16 维主方法的受控留出、逐对结果、表示/输出编辑对照 |
| meeting/optimization/ | 874 个配置的优化与集成搜索 |
| meeting/primary_full/ | MEGA 官方路径与 NLI 完整数据集结果 |
| meeting/test_expansion/ | 官方测试来源簇统计；与 primary_full/ 是同一轮结果的配套分析，不是独立复现实验 |
| meeting/trace_repair_followup/ | 60 个新来源上的标签监督后续变体；与纯 A/P 差分主方法分开报告 |
| ../aster_cross_task_followup_20261110/ | NLI/MedPIC 标签监督与截距诊断后续 |
| trace/ | 直接传导与总干扰预测的同批机制统计 |
| simulated_study/ | 合成机制实验及预设模拟；不得作为真实模型成绩 |
| clinical/ | A/P 配对人工审核材料；尚未经过医生审核 |

## 还剩什么？

当前只安排两项：

1. **同等证据代价下的特征比较**：复用现有 Qwen 缓存，比较正交痕迹、原始干扰方向、证据方向和随机方向。
2. **跨模型 MEGA 复现**：在 Llama-3.1-8B、Gemma-2-9B、Gemma-3-27B、MedGemma-27B 上分别构造痕迹并训练修正。

执行要求见 [Codex 清单](meeting/EXPERIMENT_PLAN_CODEX_20261111.zh.md)。

## 不要混淆的四种结果

| 名称 | 含义 |
|---|---|
| ASTER-CR Linear / 16-d | 16 维正交痕迹、A/P 岭回归、C/F 逐对预算、独立截距 |
| ASTER-CR MLP / optimized | 非线性或扩展特征版本；按具体配置报告，不等同于 16 维线性版 |
| label-supervised follow-up | 加入判断标签监督的后续变体，不属于纯 A/P 差分主方法 |
| ASTER M1/M2/M3 | 仓库中的另一条表示编辑研究路线，见根目录 README 和 docs/ASTER/；其结果不能与 ASTER-CR 合并 |

## 五模型 × 三任务矩阵

meeting/full_factorial_20261110/ 保存的是早期完整 285 单元矩阵方案。当前状态表 285 项均未完成共同协议评价；本轮不执行该完整矩阵，也不把它当作已完成实验。当前优先级以两项剩余实验清单为准。
