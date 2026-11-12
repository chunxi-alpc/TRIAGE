# ASTER引用与支持范围

[中文主稿](manuscript.zh.md) · [实验索引](experiments.zh.md)

2026-11-07整理。下列主稿引用已查看论文官方页面或作者数据卡，核对题名和本文所述方法／任务含义。外部方法成绩不作为本项目实验结果。

| 来源 | 本文用途 |
|---|---|
| [CAD，Shi等，2024](https://aclanthology.org/2024.naacl-short.69/) | 有／无上下文的对比输出分布 |
| [PASTA，Zhang等，2024](https://arxiv.org/abs/2311.02262) | 选定注意力头上对指定片段重加权 |
| [ITI，Li等，2023](https://arxiv.org/abs/2306.03341) | 沿部分注意力头中的方向移动激活 |
| [INLP，Ravfogel等，2020](https://aclanthology.org/2020.acl-main.647/) | 迭代分类与零空间投影 |
| [LEACE，Belrose等，2023](https://arxiv.org/abs/2306.03819) | 闭式线性概念擦除 |
| [ReFT，Wu等，2024](https://arxiv.org/abs/2404.03592) | 冻结骨干上的表示微调，LoReFT低秩编辑 |
| [MedGUIDE，Li等，2025](https://arxiv.org/abs/2505.11613) | 指南衍生原生选择题 |
| [MEGA-CDP作者数据卡](https://huggingface.co/datasets/nchennnn/MEGA-CDP) | 指南、病例描述、参考路径；当前评分以代码定义为准 |

待补：投稿所需完整书目和BibTeX；相关方法同模型、同预算的实验比较。项目7,458例、200/100病例身份、配对数和模型配置以仓库保存文件为依据，不从外部总体规模推算。

更早SEKA、JuICE、五卷协议及其他历史书目原样保存于[整理前引用说明](archive/20261107/docs__ASTER__references.zh.md.txt)，其来源和使用范围继续按原阶段理解。


## 2026-11-12 外部基线候选文献

以下条目用于第4.3节外部方法分类和实验基线筛选。**列入文献清单不代表已经在ASTER-CR上复现，也不代表全部进入第4.5节主表。** 官方代码版本和兼容性状态见 [基线兼容性审计](baseline-compatibility-20261112.json)。

### 输出校准与分数修正

- Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML. https://proceedings.mlr.press/v70/guo17a.html
- Zhao et al. (2021). *Calibrate Before Use: Improving Few-Shot Performance of Language Models*. ICML. https://proceedings.mlr.press/v139/zhao21c.html
- Zhou et al. (2024). *Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering*. ICLR. https://proceedings.iclr.cc/paper_files/paper/2024/hash/003e438cf04e3caf0a5c908495e484fe-Abstract-Conference.html
- Shi et al. (2024). *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding*. NAACL. https://aclanthology.org/2024.naacl-short.69/

### 提示与注意力干预

- Zhang et al. (2024). *Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs* (PASTA). https://arxiv.org/abs/2311.02262

### 线性与非线性概念擦除

- Ravfogel et al. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection*. ACL. https://aclanthology.org/2020.acl-main.647/
- Ravfogel et al. (2022). *Linear Adversarial Concept Erasure*. ICML. https://proceedings.mlr.press/v162/ravfogel22a.html
- Belrose et al. (2023). *LEACE: Perfect Linear Concept Erasure in Closed Form*. NeurIPS. https://arxiv.org/abs/2306.03819
- Chowdhury et al. (2023). *Robust Concept Erasure via Kernelized Rate-Distortion Maximization* (KRaM). NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2023/hash/86bd650f85480c595ecab29081a3774e-Abstract-Conference.html
- Saillenfest and Lemberger (2025). *Nonlinear Concept Erasure: A Density Matching Approach* (LEOPARD). ECAI. https://arxiv.org/abs/2507.12341
- Akbari et al. (2025). *Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure*. NeurIPS. https://openreview.net/pdf?id=GcjpjIHDZn

### 表示干预与激活引导

- Li et al. (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model* (ITI). https://arxiv.org/abs/2306.03341
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition* (CAA). ACL. https://arxiv.org/abs/2312.06681
- Qiu et al. (2024). *Spectral Editing of Activations for Large Language Model Alignment* (SEA). https://arxiv.org/abs/2405.09719
- Lee et al. (2025). *Programming Refusal with Conditional Activation Steering* (CAST). ICLR. https://proceedings.iclr.cc/paper_files/paper/2025/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html
- Wu et al. (2025). *Improved Representation Steering for Language Models* (RePS). NeurIPS. https://arxiv.org/abs/2505.20809
- Sun et al. (2025). *HyperSteer: Activation Steering at Scale with Hypernetworks*. https://arxiv.org/abs/2506.03292
- Heyman et al. (2026). *Steer Like the LLM: Activation Steering that Mimics Prompting* (PSR). ICML. https://arxiv.org/abs/2605.03907

### 表示微调与参数高效权重微调

- Yin et al. (2024). *LoFiT: Localized Fine-tuning on LLM Representations*. NeurIPS. https://arxiv.org/abs/2406.01563
- Wu et al. (2024). *ReFT: Representation Finetuning for Language Models*. https://arxiv.org/abs/2404.03592
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. https://arxiv.org/abs/2106.09685
- Liu et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML. https://proceedings.mlr.press/v235/liu24bn.html
- Gao et al. (2024). *Parameter-Efficient Fine-Tuning with Discrete Fourier Transform* (FourierFT). ICML. https://proceedings.mlr.press/v235/gao24o.html

