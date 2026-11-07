# 引用与支持范围

[论文主稿](manuscript.zh.md) · [实验协议](experiments.zh.md)

本页沿用仓库 2026-10-27／28 的来源核验记录，本轮只整理引用与用途，不声称重新核实外部论文。来源身份不等于已完成当前任务复现，原论文成绩不进入 ASTER 的结果列。

## 方法与基线

| 来源 | 支持范围 | 不应推出 |
|---|---|---|
| [Guo 等，2017：On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | 单温度后处理校准 | 自动纠正排序或获得风险认证 |
| [Wang 等，2024：“My Answer is C”](https://aclanthology.org/2024.findings-acl.441/) | 受限首词元评分与自由回答的接口差异 | 所有受限分类器均无效 |
| [SEKA／AdaSEKA：Spectral Attention Steering for Prompt Highlighting](https://arxiv.org/abs/2603.01281) | 谱键编辑与查询自适应专家；[官方实现](https://github.com/waylonli/SEKA) | 当前适用性方向、缓存协议或速度已验证 |
| [Prism-Δ](https://arxiv.org/abs/2603.10705) | 差分谱方法及键／值编辑的相关研究 | 差分谱本身是本文独有贡献 |
| [CAD：Trusting Your Evidence，NAACL 2024](https://aclanthology.org/2024.naacl-short.69/) | 有／无上下文分布对比；[实现](https://github.com/xhan77/context-aware-decoding) | TS-CAD 的选择性移除是原文默认方法 |
| [JuICE：Taming Knowledge Conflicts，ICML 2025](https://proceedings.mlr.press/v267/li25c.html) | 选头与双次前向的头输出干预；[实现](https://github.com/GaotangLi/JUICE) | 两次普通生成等价于 JuICE |
| [PASTA：Tell Your Model Where to Attend，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b99d6cc40b05809c3d84b57a165448cd-Paper-Conference.pdf) | 对指定片段的注意力重加权；[实现](https://github.com/QingruZhang/PASTA) | 其作用范围与本方法兄弟规则平移完全相同 |
| [AutoPASTA](https://arxiv.org/abs/2409.10790) | 自动片段选择后调整注意力 | 自动选择必然处置盲或无额外成本 |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 显式推理提示的背景 | 输出解释直接证明内部机制 |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 相对固定参考策略的偏好目标 | 本任务偏好构造和额外监督已被控制 |
| [Don't Stop Pretraining，ACL 2020](https://aclanthology.org/2020.acl-main.740/) | 领域／任务语料适应 | 额外临床语料与金标最小对属于相同监督 |

## 数据与评分

| 来源 | 使用范围与待冻结项 |
|---|---|
| [MedGUIDE 原文](https://arxiv.org/html/2505.11613v1) · [作者数据卡](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K) | 选择题与树来源；完整树匹配、派生五卷和实际分母另行核验 |
| [MEGA-CDP 原文](https://arxiv.org/html/2608.26592v1) · [作者数据卡](https://huggingface.co/datasets/nchennnn/MEGA-CDP) | 路径任务与 PCC；固定官方评分实现或明确标注适配 |
| [MedPIC 原文](https://arxiv.org/html/2608.03028v1) · [作者数据卡](https://huggingface.co/datasets/TIM0927/MedPIC-Bench) | 多选集合匹配与关联对；原生与独立规则增强分表 |
| [仓库数据配置](../../configs/datasets.json) | FaithEval、MedCalc 及下载记录；运行前核验具体修订与许可，不把配置当作实验结果 |

## 待核验与历史来源

数据版本、模型检查点、模板、许可、完整构造清单及评分器须在测试前冻结。机制工具的具体来源、额外训练忠实化方法和未纳入基线的理由，仍按原引用清单逐项核验；未核实作者、题名或支持范围时不补造引文。

医学自然语言推断、旧阶段掩码、证据员及其他已退出默认方法的书目，连同会话引用占位符的核验任务，保留在[固定版本归档](archive/README.md)，不在当前方法中重复展开。历史模型结果保留原版本、分母和限制，不作为新版 ASTER 的证据。
