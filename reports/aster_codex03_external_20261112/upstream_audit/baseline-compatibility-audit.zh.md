# ASTER-CR 基线兼容性审计与 Codex 00/01 交接

审计日期：2026-11-12  
审计分支：`codex/aster-codex00-01-audit-20261112`  
审计基线：`main@742dcb2598631ac20c1ae2f72c85a2dff20cb16b`  
机器可读清单：[baseline-compatibility-audit.json](baseline-compatibility-audit.json)

## 结论摘要

当前外部基线体系**有可复用代码和历史结果，但尚未形成满足 E1 主协议的完整、可直接运行的基线套件**。本次完成的是仓库静态审计、官方实现核对和执行依赖梳理，不是模型实验。

最重要的结论如下。

1. E1 / Dataset A（2,540 对、MEGA-CDP only）仍是唯一正式主训练配置。E2/E3 只能作为消融。
2. 旧 806 路径比较结果不能自动视为 E1 新协议结果，也不能替代 MEGA 官方 7,458 条路径评价。
3. 当前代码已有温度缩放、截距、仿射校准、监督线性读出、CAD、LEACE、ITI、LoFiT 和 LoRA 的实现或适配代码；这些方法仍需逐项核对 E1 数据、模型修订和评价分母。
4. 当前没有 TRIAGE 内部的 CAA、CAST、RePS、PSR、INLP、R-LACE、KRaM、Obliviator、DoRA 或 FourierFT 正式实现。
5. LoReFT 虽有代码，但其依赖 pyreft 不在已提交的环境锁中；历史缓存与原模型相同，不能视为有效结果。
6. 当前比较代码默认指向旧 repair-v2 运行目录。正式补跑前必须建立独立 E1 运行目录和特征缓存。
7. 不能把 ASTER-CR 的 closed-form、legacy M3、last-layer 等内部变体放入外部基线表；这些应进入内部消融或控制实验。
8. 当前连接只能审查 GitHub 内容，不能检查服务器进程、GPU 占用、服务器本地数据或实际安装环境。本次没有启动 GPU 任务，也没有声称任何新实验已经完成。

**论文组织原则：** 第4.3节（或正文重排后的相关小节）可以按方法类别介绍多项代表工作；第4.5节主表只能报告在同一 E1 评价协议下实际运行并通过审计的方法。文献代表不等于实验结果。不得用其他论文的成绩填充本项目的主表。

## 1. 冻结实验协议

| 项目 | 当前冻结设定 |
|---|---|
| 主模型 | Qwen3-8B |
| 主训练配置 | E1 / Dataset A |
| 训练数据 | 2,540 对，A/P/C/F 各 635 对，全部来自 MEGA-CDP |
| 数据划分 | 80 个拟合来源、20 个开发来源；另有 40 个历史受控留出来源 |
| 来源交叉验证 | 5 折；每折 64 个来源拟合、16 个来源验证 |
| 数据消融 | E2、E3；不替代 E1 |
| 跨任务 | NLI4CT、MedPIC，冻结迁移评价 |
| 主要评价 | A/P 干扰抑制、C/F 证据保护、自然节点/路径效用及纠正/新增错误 |

E1 配置、数据版本和模型参数见 [aster_cr_training_default.json](../../configs/aster_cr_training_default.json)。配对原文和完整训练清单位于服务器，不在当前 GitHub 文件中。后续基线必须先确认服务器端 E1 清单与本地缓存一致，不能仅凭配置文件存在就认定数据已准备好。

历史 806 路径结果使用旧数据和旧流程。它们可以作为历史比较，但不能改标为 E1 结果。MEGA 官方 7,458 条路径、NLI4CT 5,500 条声明和 MedPIC 467 道题也必须保留各自评价总体和分母。

## 2. 当前代码实现审计

### 2.1 已有代码或历史结果

| 方法/控制 | 代码位置 | 审计判断 | 进入主表的条件 |
|---|---|---|---|
| 温度缩放 | `experiments/aster_cr/readout.py` | 已实现；正比例缩放不会改变零阈值二分类符号 | 报告校准指标；不能把它解释为路径准确率提升 |
| 截距校准 | `experiments/aster_cr/readout.py` | 已实现，是必要的简单控制 | 使用 E1 拟合/开发协议重新核验 |
| 仿射逻辑校准 | `experiments/aster_cr/readout.py` | 已实现，是自定义分数校准控制 | 明确报告其训练标签和参数量 |
| 监督线性读出 | `experiments/aster_cr/readout.py`、`pipeline.py` | 已有实现 | 确认特征、来源划分和评价分母属于 E1 |
| CAD | `experiments/aster_cr/comparators.py` | 已有历史适配；使用 `3 × full − 2 × reference`，即 α=2 | 重新核验 E1；明确 reference 保留患者、当前节点和当前条件，但移除完整指南树与定义 |
| LEACE | `experiments/aster_closed_form/operators.py`、`comparators.py` | 已有本地适配和历史结果；不是未经修改的官方库调用 | 与官方实现做数值核验；公开概念编码和读出协议 |
| ITI | `experiments/aster_cr/comparators.py` | 已有任务适配和历史结果；方向由正确/错误答案表示估计 | 核验头选择、方向、干预层、强度和 E1 训练来源 |
| LoFiT | `experiments/aster_cr/baselines.py` | 已有 Qwen/Gemma 架构适配代码；不是官方原样复现 | 对照官方两阶段算法，说明 BCE 任务适配与原论文 DPO/SFT 的差异 |
| LoRA | `experiments/aster_cr/baselines.py` | 已有 PEFT 代码；当前只在一个中间层 q_proj 上注入 | 在真实 Qwen3-8B 上确认适配器生效、参数量和 E1 兼容性 |
| LoReFT | `experiments/aster_cr/baselines.py`、`test_models.py` | 有调用代码和微型模型测试，但 pyreft 依赖未进入环境锁；旧缓存与原模型相同 | 安装并固定 pyreft；真实权重上验证参数加载和 logit 改变 |

### 2.2 当前实现中不能混淆的名称

- `reference_fusion` 是使用 full/reference 两个分数和任务标签训练的逻辑回归融合器。它不是 Contextual Calibration（CC），也不是 Batch Calibration（BC）。
- `best_template` 是答案顺序、标签映射和提示措辞的开发集选择。它不等于专门设计的证据聚焦提示基线。
- `closed_form` 是 ASTER 自身算子版本，不是外部方法。
- `legacy_m3` 是旧版 ASTER 内部方法，不是外部方法。
- `last_layer` 是简单的末层残差偏移控制，不是独立文献基线。
- `SEA` 是单独的子空间擦除适配器。代码引用服务器目录 `ROOT/vendor/sea`，而当前 Git 树没有跟踪该 vendor 目录。SEA 不能改称 CAA 或 CAST。
- `configs/baselines.json` 中的 ProbeRAG、SEKA、JUICE 和 TrialGPT 服务于 TRIAGE 的其他工作流，不应自动加入 ASTER-CR 主表。

### 2.3 需要先解决的工程阻断

**E1 数据接口。** `experiments/aster_cr/common.py` 的默认根目录是旧的 `/home/gcx/aster_cr_20261108`。当前 E1 数据和模型产物属于 20261111 配置。正式补跑必须建立独立的 E1 运行根目录，重新导出 E1 的 fit/dev/test 视图、配对、路径案例和特征缓存。不得复用旧 repair-v2 缓存来冒充新协议。

**模型修订。** `configs/models.json` 中 Qwen3-8B 使用 `master`，文件修订为空；`configs/qwen3_8b_modelscope_manifest.json` 的 Revision/Sha256 字段也为空。服务器上必须记录实际模型和分词器快照版本，不能猜测或补造 revision。

**依赖。** `requirements-aster-manuscript.txt` 仅列出 torch 和 transformers。环境锁包含 PEFT、scikit-learn、numpy、torch 2.8.0 和 transformers 4.57.6，但没有 pyreft。LoReFT 必须使用独立、明确固定的可选依赖。KRaM 和 Obliviator 的官方环境要求与当前环境不同，不能直接升级共享环境。

**随机种子。** 当前 `common.py` 使用 20261108；实验框架要求随机训练基线报告 11、22、33。E1 已冻结的训练种子不能改动；外部随机基线应增加显式 seed 参数并按 11、22、33 运行。

**GPU。** 当前代码默认 GPU 0、1；服务器执行说明只允许物理 GPU 0、1、5、6。启动时应显式设置 `ASTER_CR_GPUS=0,1,5,6`，但必须先在服务器检查占用和显存。控制器会查询 nvidia-smi；本次审计没有查询服务器，也没有确认任何 GPU 空闲。

## 3. 建议采用的外部基线分类

### 3.1 提示、输出校准与上下文对比解码

- 提示方法：新增一个预先定义的证据聚焦提示。必须保留完整指南和患者输入，不得裁剪决定性事实。
- 概率校准：Temperature Scaling、Contextual Calibration。
- 批量校准：Batch Calibration。该方法使用批次统计，属于转导设置；应与逐病例归纳式主表分开。
- 上下文对比解码：CAD。应单独成类，不应与普通温度缩放或截距校准混为一谈。

温度缩放在零阈值二分类中不改变分数符号。应报告 Brier、log loss、ECE 等校准指标。CAD 的 reference 定义必须明确：当前实现保留患者和当前条件，但删除完整指南树与定义，因此它检验的是指南上下文的作用，不是无患者信息的预测。

### 3.2 线性与非线性概念擦除

- 迭代线性擦除：INLP。
- 对抗线性擦除：R-LACE。
- 闭式线性擦除：LEACE。
- 核化非线性擦除：KRaM。
- 非线性概念擦除：Obliviator。

概念擦除的目标是降低表示中指定概念的可恢复性，不等于直接修复指定决策。A/P 配对类型不能不加论证地等同于完整干扰概念。所有擦除方法必须同时报告干扰探针、证据探针、任务效用和读出协议。固定原读出与擦除后重新训练读出应分开报告。

### 3.3 激活干预与条件引导

- ITI：固定注意力头方向干预。
- CAA：成对激活差分方向。
- CAST：根据输入条件选择是否施加激活干预。

CAA 和 CAST 的原始任务与医学指南条件判断不同。若使用 E1 标签进行任务适配，必须明确标注为 task-adapted 版本，并保留原方法的方向估计或条件门控机制。

### 3.4 新近表示引导

- RePS：AxBench 中包含对应的实验配置，可作为偏好监督表示引导候选。
- PSR：ICML 2026 的提示蒸馏式激活引导方法，学习位置相关的干预系数。

RePS 和 PSR 均不在当前 TRIAGE 代码中。应先核验官方训练目标、模型支持和数据接口，再决定是否适配 Qwen3-8B。不得把使用 E1 配对监督的任务适配版本称为原始方法。

### 3.5 表示微调

- LoFiT：局部注意力头选择和偏移向量学习。
- LoReFT：低秩表示干预。
- ReFT-r1：AxBench 中的近期表示微调候选。

LoFiT 当前本地代码是架构适配，不是官方原样复现。LoReFT 必须先解决依赖和真实权重有效性问题。

### 3.6 参数高效权重微调

- LoRA：低秩权重更新。
- DoRA：权重幅值与方向分解。
- FourierFT：频域参数化权重更新。

DoRA 和 FourierFT 均可从 PEFT 路线适配。应统一目标模块、参数预算和训练预算。FourierFT 仅支持 nn.Linear，不能用于量化层；Qwen3 的 q_proj 需先做 smoke test。

## 4. 主表纳入规则与优先级

| 优先级 | 方法 | 处理决定 |
|---|---|---|
| P0 | 原模型、截距、温度缩放、仿射校准、监督线性读出 | 先核验 E1 兼容性；作为基础控制 |
| P0 | 证据聚焦提示 | 新增独立提示，不与 verbalizer 搜索混为一谈 |
| P0 | CAD | 复用现有实现，明确 reference 定义后按 E1 重评 |
| P0 | LEACE、ITI | 复核本地适配与历史结果，按 E1 重评 |
| P0 | LoFiT、LoRA | 验证现有适配代码、参数量和真实模型输出变化 |
| P0 | LoReFT | 补依赖并通过真实 Qwen3 有效性测试后再纳入 |
| P1 | CC、INLP/R-LACE、CAA/CAST | 完成官方方法核验和最小任务适配后运行 |
| P1 | RePS、PSR | 优先验证近期表示引导方法；保留原生目标与任务适配版本的区分 |
| P1 | DoRA、FourierFT | 使用 PEFT 实现，先做参数匹配和真实模型 smoke test |
| P2 | KRaM、Obliviator | 作为非线性擦除补充候选；使用隔离环境，不阻塞主表 |
| P2 | BC、全参数微调 | 分别作为转导校准和高成本任务适配上界，单独报告 |

P0/P1 是建议的实验顺序，不表示已经运行。最终主表仅纳入完成同模型、同输入、同评价分母的实测结果。未运行方法只能出现在相关工作和实验设置说明中，不能填入数值。

## 5. 公平比较协议

1. 所有主比较固定同一 Qwen3-8B 模型修订、分词器、系统提示、答案标签和分数计算方式。
2. E1 拟合来源用于训练；开发来源用于有限超参数和提示选择；测试来源只用于冻结后的最终评价。
3. 每种方法保留原生目标。额外加入 A/P 或 C/F 监督时，必须单独标记为任务适配版。
4. 每种需要调参的方法最多搜索 12 个完整候选配置；测试集不得参与搜索。
5. 随机训练方法使用种子 11、22、33，报告逐种子结果；确定性闭式方法不制造种子波动。
6. 按指南来源簇进行配对重采样，报告 95% 置信区间、纠正数和新增错误数。
7. 概念擦除方法同时报告干扰概念可恢复性、决定性证据可恢复性和任务效用。
8. 激活干预报告层、位置、方向、强度、额外前向次数和推理成本。
9. 权重微调报告可训练参数量、训练步数、优化器、学习率、训练时间、峰值显存和推理延迟。
10. 不得引用其他模型或其他任务的原论文成绩作为 Qwen3-8B/E1 的实验结果。

## 6. Codex 00/01 的执行依赖与后续交接

推荐依赖顺序：

**E1 数据接口与模型修订冻结**  
→ **方法依赖和适配器 smoke test**  
→ **P0 方法开发集运行**  
→ **开发集选型与冻结**  
→ **P0 冻结测试**  
→ **P1 方法开发集运行与冻结**  
→ **P1 冻结测试**  
→ **来源级统计与主表审计**。

代码审计、官方仓库核验、配置准备和 CPU 合成测试可以并行。需要加载 Qwen3-8B 的实验必须由统一调度器按空闲 GPU 排队；不能让不同 Codex 会话各自占用全部设备。

### Codex 00 已完成的静态工作

- 核对主分支快照、E1 冻结配置和实验状态。
- 检查 ASTER-CR 代码、比较器、训练入口、环境锁、模型清单和服务器执行说明。
- 识别 E1 数据根目录、模型修订、随机种子、GPU 默认值和依赖阻断。
- 建立方法分类、优先级和实验依赖顺序。

### Codex 01 已完成的静态工作

- 核对各方法的论文/官方仓库入口。
- 区分官方实现、本地适配、历史结果、未实现候选和内部控制。
- 建立机器可读方法清单。
- 明确各方法进入主表前的适配条件。

### 交给 Codex 02—06 的前置条件

1. 先创建独立 E1 baseline run root，并生成数据清单和特征缓存；不要覆盖旧 repair-v2 目录。
2. 在服务器确认实际模型修订、分词器和 GPU 进程。
3. 完成可选依赖安装及版本记录；pyreft、KRaM、Obliviator、PSR 不得直接污染主环境。
4. 先通过小样本真实权重测试，再启动大规模评价。
5. 所有开发集选择完成后写入冻结配置，再启动测试集。
6. 每个方法保存逐例预测、配置、日志、运行 ID、模型修订、数据版本、参数量、前向次数、耗时和峰值显存。

## 7. 本次审计的执行边界

本次只修改审计文档和实验索引，不修改论文正文、ASTER-CR 方法实现、数据、模型权重或历史结果。没有执行服务器命令、没有检查实时 GPU、没有启动推理或训练，也没有产生新的方法分数。

当前主分支正文的基线小节编号与正在编辑的稿件可能不同。最终合并时应按用户正在维护的正文编号调整交叉引用；本次不改写 manuscript.zh.md。

## 8. 官方来源

- [Temperature Scaling, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)
- [Contextual Calibration, ICML 2021](https://proceedings.mlr.press/v139/zhao21c.html)
- [Batch Calibration, 2024](https://research.google/pubs/batch-calibration-rethinking-calibration-for-in-context-learning-and-prompt-engineering/)
- [CAD, NAACL 2024](https://aclanthology.org/2024.naacl-short.69/)
- [INLP, ACL 2020](https://aclanthology.org/2020.acl-main.647/) · [官方代码](https://github.com/shauli-ravfogel/nullspace_projection)
- [R-LACE, ICML 2022](https://proceedings.mlr.press/v162/ravfogel22a.html) · [官方代码](https://github.com/shauli-ravfogel/rlace-icml)
- [LEACE, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) · [官方代码](https://github.com/EleutherAI/concept-erasure)
- [KRaM, NeurIPS 2023](https://github.com/brcsomnath/KRaM)
- [Obliviator, NeurIPS 2025](https://openreview.net/pdf?id=GcjpjIHDZn) · [官方代码](https://github.com/ramin-akbari/Obliviator)
- [ITI, NeurIPS 2023](https://github.com/likenneth/honest_llama)
- [CAA](https://arxiv.org/abs/2312.06681) · [官方代码](https://github.com/nrimsky/CAA)
- [CAST, ICLR 2025](https://arxiv.org/abs/2409.05907) · [官方代码](https://github.com/IBM/activation-steering)
- [RePS / AxBench](https://arxiv.org/abs/2505.20809) · [官方代码](https://github.com/stanfordnlp/axbench)
- [PSR, ICML 2026](https://proceedings.mlr.press/v306/heyman26a.html) · [官方代码](https://github.com/Nokia-Bell-Labs/steer-like-the-llm)
- [LoFiT, NeurIPS 2024](https://arxiv.org/abs/2406.01563) · [官方代码](https://github.com/fc2869/lo-fit)
- [ReFT / LoReFT](https://arxiv.org/abs/2404.03592) · [官方代码](https://github.com/stanfordnlp/pyreft)
- [LoRA](https://arxiv.org/abs/2106.09685) · [PEFT](https://github.com/huggingface/peft)
- [DoRA, ICML 2024](https://arxiv.org/abs/2402.09353) · [官方代码](https://github.com/NVlabs/DoRA)
- [FourierFT, ICML 2024](https://proceedings.mlr.press/v235/gao24o.html) · [PEFT 实现](https://github.com/huggingface/peft)
