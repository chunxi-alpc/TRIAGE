# 引用核验清单

[主线索引](README.md) · [编辑记录](editorial-notes.zh.md)

## 本次已核对的原始来源

下表在 2026-10-27 查阅了原始论文页面或正文。只记录与本版直接相关的支持范围，不把其他任务的结果移植为 ASTER 的性能。

| 来源 | 支持范围 | 不支持的推断 |
|---|---|---|
| [Guo 等，2017：On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | 单参数温度缩放的后处理校准 | 校准自动纠正排序错误，或取得风险认证 |
| [Romanov 与 Shivade，2018：Lessons from Natural Language Inference in the Clinical Domain](https://aclanthology.org/D18-1187/) | MedNLI 的临床自然语言推断任务 | 任意医学 NLI 都能直接解决互斥指南分支分类 |
| [Jullien 等，2023：SemEval-2023 Task 7](https://aclanthology.org/2023.semeval-1.307/) | 临床试验文本的推断、证据选择与数值推理任务 | 所有 NLI 模型都比部署骨干更差 |
| [Wang 等，2024：“My Answer is C”](https://aclanthology.org/2024.findings-acl.441/) | 首词元评价与自由文本回答的行为区别 | 任何正确实现的受限单词元分类器均无效 |
| [Li 等，2026：Spectral Attention Steering for Prompt Highlighting](https://arxiv.org/abs/2603.01281) | SEKA 键编辑与 AdaSEKA 查询自适应专家路线 | 本任务的适用性方向、缓存语义或速度已验证 |
| [Ge 等，2026：Prism-Δ](https://arxiv.org/abs/2603.10705) | 已有差分互协方差谱方法及键／值编辑研究 | 本文配对二阶差分与均值平移与其完全相同，或差分谱本身是本文新贡献 |

模块一提示、模型参数、下游 M3 有效性仍需本任务实测。谱方法的具体代码版本和公平复现配置在实验前另行固定。

## 原稿中仍待核验的条目

以下是原稿的引文线索，不是已经核验的书目。不生成猜测的作者、年份、论文链接或 BibTeX。核验时优先查原始论文、数据卡和官方实现。

| 主题／方法线索 | 待核验内容 |
|---|---|
| MedGUIDE | 论文题名、作者年份、NCCN 来源与许可、决策树／癌种规模、官方任务定义 |
| MEGA-CDP | 论文版本、指南／病例规模、参考路径构造、PathCost 定义与评审实现 |
| MedPIC-Bench | 题量、配对方式、撤销警示子集、题目许可与官方设置 |
| CAD；原稿写 Shi et al., 2024 | 准确论文与算法定义、实现和对照设置 |
| JuICE | 论文／代码版本、是否患者无关、具体干预头和目标 |
| PASTA；原稿写 Zhang et al., 2024a | 论文与实现、偏置或缩放算子、目标片段选择 |
| AutoPASTA；原稿写 Zhang et al., 2024b | 论文与关键句选择流程、对处置可见性的处理 |
| SEKA／AdaSEKA | 论文已见上表；具体检查点、开源代码版本、公平参数范围及任务迁移仍待核验 |
| Spotlight | 论文、控制接口和适用边界 |
| Context-DPO、CANOE；原稿另列 Bi et al., 2025、Si et al., 2025 | 名称对应关系、训练目标及不纳入比较的合理范围 |
| Longpre et al., 2021；Xie et al., 2024；Wu et al., 2024 | 原稿涉及的具体知识冲突论文及与当前任务的差异 |
| Lampinen et al., 2024 | 内容效应的具体论文、任务和支持范围 |
| 合理性与有效性表征对齐 | 原始论文与实验范围；原稿只有会话引用占位符 |
| Turpin et al., 2023 | 思维链不忠实论文与可支持的具体论断 |
| 看选项前推理；Zheng et al., 2024 | 两条不同引用的具体题名、选择偏差定义和评测方式 |
| 提示重复 | 具体论文与代码，不能只保留会话引用占位符 |
| Chen et al., 2024 | 原稿所指前提顺序效应论文 |
| Geva et al., 2023 | 注意力敲除方法与适用限制 |
| Meng et al., 2022；Zhang & Nanda, 2024 | 激活修补与方法可靠性论述的具体来源 |
| Biran et al., 2024 | 多跳推理解析时序论文，核对能否支持原稿推断 |
| FlexAttention；原稿写 Dong et al., 2024 | 官方 API、发布来源、支持限制与正确引用 |
| DeBERTa-v3-large、MNLI、MedNLI | MedNLI 任务定义已见上表；可选 NLI 基线的模型检查点、许可与实际微调／校准仍待核验 |

原稿形如 `[cite:5ea145c5-3]`、`[cite:7330a3b0-1]` 的标记是不可独立解析的会话内部引用。主线正文已移除这些标记，相关来源全部保留为核验任务。

