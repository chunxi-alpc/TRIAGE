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

## 2026-10-28 已核实的实验来源

本轮核对以下一手论文、作者数据卡及仓库原始结果。已确定来源身份与相关定义；“核实来源”不等于“已完成本任务复现”。正文及配套数据／基线／指标协议保留可点击来源，原论文分数不进入 ASTER 的成绩列。

| 来源 | 已核实内容 | 本研究采用的范围 |
|---|---|---|
| [CAD：Trusting Your Evidence，NAACL 2024](https://aclanthology.org/2024.naacl-short.69/) | 有／无上下文的对比分布解码 | 输出分布干预基线；NC 与冲突同报 |
| [JuICE：Taming Knowledge Conflicts，ICML 2025](https://proceedings.mlr.press/v267/li25c.html) | Just Run Twice；头选择与双次前向 | 头输出干预基线；架构适配仍须验收 |
| [PASTA：Tell Your Model Where to Attend，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b99d6cc40b05809c3d84b57a165448cd-Paper-Conference.pdf) | 对指定片段做注意力重加权 | 全指南与共享 M1 目标两轨 |
| [AutoPASTA](https://arxiv.org/abs/2409.10790) | 自动选择关键上下文后调整注意力 | 与自动选择流程直接相关的扩展 |
| [SEKA／AdaSEKA，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/32ddf8ee55540539eb46083cd72361e8-Paper-Conference.pdf) | 固定谱投影与查询自适应键编辑 | 推理时基线；计入离线估计与选头 |
| [DPO，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html) | 相对固定参考策略的偏好目标 | 相同 SFT 初始化后的额外偏好训练 |
| [继续预训练，ACL 2020](https://aclanthology.org/2020.acl-main.740/) | 领域／任务语料适应 | CPT-only 与 CPT＋相同 SFT 分报 |
| [MedGUIDE 原文](https://arxiv.org/html/2505.11613v1)与[作者数据卡](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K) | 来源规模与指南决策选择题 | 完整树匹配后派生五卷；最终分母待清单 |
| [MEGA-CDP 原文](https://arxiv.org/html/2608.26592v1)与[作者数据卡](https://huggingface.co/datasets/nchennnn/MEGA-CDP) | 官方测试规模、路径任务与 PCC | 单轮 NC／SW，DTW 先求累计最优再归一化 |
| [MedPIC 原文](https://arxiv.org/html/2608.03028v1)与[作者数据卡](https://huggingface.co/datasets/TIM0927/MedPIC-Bench) | 467 题、多选集合匹配、89 组关联对 | 原生与独立规则增强适配分表 |
| [历史 H1–H3](historical-results.zh.md) | 原始计数、分母、版本及统计限制 | 仅作历史诊断，保留负结果 |

进一步的数据版本、来源、许可、字段和样本资格见[数据协议](dataset-protocol.zh.md)。训练和推理基线的算法适配见[基线协议](baseline-protocol.zh.md)。目前仍待冻结的是具体模型修订、运行产物、数据构造清单和评审器，不再将上述已核对论文身份写成未知。

## 原稿中仍待核验的条目

以下是原稿的引文线索，不是已经核验的书目。不生成猜测的作者、年份、论文链接或 BibTeX。核验时优先查原始论文、数据卡和官方实现。

| 主题／方法线索 | 待核验内容 |
|---|---|
| SEKA／AdaSEKA 的运行产物 | 新骨干投影、头筛选及本任务公平复现配置；论文身份与公式已经核实 |
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


