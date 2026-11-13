# ASTER-CR 外部基线与 Codex 00/01 执行审计

审计日期：2026-11-12  
审计仓库：`chunxi-alpc/TRIAGE`  
审计基准：`main`，提交 `742dcb2598631ac20c1ae2f72c85a2dff20cb16b`  
工作分支：`codex/aster-codex00-01-audit-20261112`

> 本报告是仓库静态审计和执行编排记录，不代表已经连接服务器、检查 GPU、运行模型或完成任何新增实验。审计基于 GitHub 可见文件、历史结果文件和公开官方方法仓库。

## 1. 执行结论

Codex 00 和 Codex 01 的仓库级工作可以完成，但必须区分“代码存在”“历史开发集结果存在”和“当前 E1 协议下已完成正式比较”。

当前主训练配置已经固定为 E1 / Dataset A（2,540 对，MEGA-CDP only）。因此，旧实验目录中的结果不能仅凭方法名称相同就迁移到当前主表。

本轮审计确认：

- 已有本地实现或适配：温度缩放、偏置/逻辑校准、CAD 风格参考分数对比、LEACE 形式的配对擦除、ITI 适配、SEA 适配、LoFiT、LoReFT 和 LoRA。
- 尚未在当前 ASTER-CR 实验代码中实现：标准 CC、BC、强证据提示、INLP、R-LACE、KRaM、LEOPARD、Obliviator、CAA、CAST、RePS、PSR、HyperSteer、PASTA、DoRA、FourierFT 和全参数微调。
- 已有外部基线结果主要来自旧的 `aster_cr_repair_v2_20261108` 目录。它们是历史协议下的开发集结果，不能直接作为当前 E1 的主表成绩。
- 当前公开仓库没有跟踪 `vendor/` 目录，但部分比较器依赖服务器运行目录下的 `vendor/sea` 和 `vendor/python`。
- 现有 LoRA、LoReFT 实现属于单层、低秩任务适配，不是标准全模型多层 PEFT 设定，也不构成与 ASTER-CR 的严格参数量匹配。

## 2. 审计范围与证据

检查了以下项目文件：

| 文件 | 审计内容 |
|---|---|
| `experiments/aster_cr/comparators.py` | CAD、参考分数融合、LEACE、ITI、SEA 和内部比较器 |
| `experiments/aster_cr/baselines.py` | LoReFT、LoFiT、LoRA 和末层残差控制 |
| `experiments/aster_closed_form/operators.py` | LEACE 形式的配对概念擦除构造 |
| `experiments/aster_cr/readout.py` | 温度、偏置、逻辑校准和输出读出 |
| `experiments/aster_cr/common.py` | 数据根目录、GPU 默认值和模型注册 |
| `requirements.txt`、`requirements-aster-manuscript.txt` | 依赖声明和环境边界 |
| `configs/baselines.json` | 现有基线注册表的实际用途 |
| `docs/ASTER/experiments.zh.md` | 当前 E1 协议和 E5 外部基线状态 |
| `docs/ASTER/manuscript.zh.md` | 当前正文基线表、结果章节和训练协议 |
| `results/aster_cr_repair_v2_20261108/` | 历史开发集比较器、训练记录和运行状态 |

另外核对了各候选方法的论文页面、作者或机构官方代码仓库、README、依赖文件和可见提交版本。机器可读记录见 [baseline-compatibility-20261112.json](baseline-compatibility-20261112.json)。

## 3. 本地实现审计

### 3.1 输出校准与参考分数修正

| 方法或组件 | 当前实现 | 审计结论 |
|---|---|---|
| Temperature Scaling | `readout.py` 中的正温度缩放 | 已实现。正比例缩放不改变二分类分数的符号，因此应报告概率校准指标，不能把它解释为分类准确率改进 |
| Bias / Logistic | 共享偏置及仿射逻辑校准 | 已实现，作为简单输出校准控制 |
| CAD | `3 × full − 2 × reference` | 已有 CAD 风格实现，等价于系数为2的上下文对比形式 |
| Reference fusion | 对 full 和 reference 分数训练逻辑回归 | 已实现，但这是有监督双分数融合，不是 Contextual Calibration |
| CC | 标准内容无关输入校准 | 尚未实现 |
| BC | 批次上下文偏置校准 | 尚未实现；若使用测试批次的无标签统计，必须作为转导评价单列 |
| Strong evidence prompt | 明确要求模型优先依据决定性证据判断的强提示 | 尚未实现 |
| Template selection | 从多个提示格式或标签顺序中选择模板 | 已有模板控制，但不能替代强证据提示基线 |

当前 CAD 参考提示移除了完整指南树和定义，但保留患者文本及当前节点/条件。因此，应准确称为本文任务中的参考上下文适配，而不是不加说明地称为标准内容无关校准。

### 3.2 概念擦除

| 方法 | 当前状态 | 关键问题 |
|---|---|---|
| LEACE | 已有本地配对适配 | 当前实现使用 A/P 配对左右侧及类别/子类型构造概念矩阵，不等同于标准逐样本稳定概念标签 |
| INLP | 未实现 | 需要稳定的逐样本概念标签和独立训练/验证划分 |
| R-LACE | 未实现 | 同样需要明确待擦除概念，不能直接把 A/P 配对类型等同于概念值 |
| KRaM | 未实现 | 官方代码较旧，需离线特征适配和独立环境 |
| LEOPARD | 未实现 | 基于类别条件分布匹配的正交投影，需要逐样本离散概念标签 |
| Obliviator | 未实现 | 非线性擦除，依赖较新的 Python/PyTorch 环境，计算成本较高 |

**LEACE 的特殊问题：** 当前 `fit_operators` 以配对左右侧的正负编码和 A/P 类型/子类型构造概念矩阵。由于一个输入样本可能在不同配对中以不同侧出现，这种构造不能未经论证就解释为标准 LEACE 的实例级概念变量。

后续应提供两个独立结果：

1. 标准 LEACE：使用明确、稳定的实例级干扰概念标签。
2. LEACE-pair adaptation：保留当前配对侧监督，但明确标注为本文任务适配版本。

两者不得混称。若无法建立可靠的实例级概念标签，应将其作为配对擦除控制，而不是声称完成标准概念擦除复现。

### 3.3 表示干预与激活引导

| 方法 | 当前实现 | 审计结论 |
|---|---|---|
| ITI | 正确/错误答案激活差、头选择及方向干预 | 有任务适配实现，但不是原始真实性任务的直接复现 |
| SEA | 正负示例激活的谱投影和末层残差干预 | 有任务适配实现，但官方 AdapterLayer 依赖服务器本地 vendor 路径 |
| CAA | 无 | 官方代码面向 Llama 2，发布流程还使用 GPT-4 生成行为数据；需要 Qwen3 架构适配和任务监督映射 |
| CAST | 无 | 官方实现可用，但需为 ASTER 明确条件向量与行为向量 |
| RePS | 无 | AxBench 有对应 preference/BiPO 实现，需要构造合法偏好对 |
| PSR | 无 | 官方实现需要提示引导教师输入和逐位置激活模仿数据；Python≥3.12、PEFT 0.15.1、vLLM 和 OpenAI API 依赖要求独立环境 |
| HyperSteer | 无 | AxBench 有实现，需大量引导提示和超网络训练，优先级较低 |
| PASTA | 无 | 官方 README 说明其原理可适用于因果注意力，但当前支持/测试模型为 Llama、Llama 2 和 GPT-J；Qwen3 需要注意力适配和片段映射 |

ITI 不能仅凭方向来自正确/错误答案就称为原始 ITI。应报告为 ITI-style task adaptation，并与原始论文的训练数据、层/头选择和干预位置区分。

SEA 当前实现需要特别检查零范数回退。历史开发结果中，`SEA_zero_norm_fallback_positions` 为 4,744。该诊断必须先解释清楚，才能把 SEA 作为有效外部基线。

### 3.4 表示微调与参数高效权重微调

| 方法 | 当前实现 | 参数和协议问题 |
|---|---|---|
| LoFiT | 两阶段本地架构适配 | 最终部署偏置参数约128；头选择阶段另优化约147,456个参数 |
| LoReFT | 使用 PyReFT 的 LoReFT intervention 组件 | 单中间层、单位置、rank=1；约8,193个参数 |
| LoRA | 使用 PEFT | 单中间层 q_proj、rank=1；约8,192个参数 |
| DoRA | 未实现 | 可用 PEFT 或官方实现适配 |
| FourierFT | 未实现 | 官方方法已进入 Hugging Face PEFT 生态，仍需固定版本和运行 smoke test |
| Full SFT | 未实现 | 可作为高成本任务适配参照，不是当前已完成基线 |

当前 ASTER-CR 线性主方法约17个可训练系数。LoFiT 的最终部署参数约128；LoReFT 和 LoRA 分别约8,193和8,192。不能将这几项称为严格参数量匹配。

本地 LoFiT 端口还改变了训练目标：当前 baselines.py 使用二元 BCE，并通过 attention o_proj hook 实现逐坐标缩放和选中头偏移；官方 LoFiT 发布代码按原任务使用 DPO/SFT 训练器和定制模型类。因此，除非完成算法等价性核对，结果应标为 LoFiT-style/task-adapted。LoReFT 调用 pyreft，但 pyreft 未列入当前环境锁；现有微型随机 Qwen 测试不替代真实 Qwen3-8B 权重上的适配器加载和 logit 改变测试。

实现忠实性还需单独审计：本地 LoFiT 通过 attention o_proj hook 实现逐坐标缩放和选中头偏移，并使用二元 BCE 训练；官方 LoFiT 仓库针对 Llama 2/Gemma 7B 定制模型类，原实验按任务使用 DPO 或 SFT。除非逐项证明两阶段头选择、偏移参数化和训练目标等价，正文应称为“LoFiT-style 任务适配”，不能写成官方 LoFiT 原样复现。LoReFT 调用 pyreft，但 pyreft 不在当前 environment.lock.txt 中；test_models.py 的随机微型 Qwen 测试不能替代真实 Qwen3-8B 权重的适配器加载和输出变化验收。

建议未来同时报告：

- 方法原生配置；
- 参数量与训练成本；
- 多预算曲线或预算约束配置。

不能只报告部署参数而忽略 LoFiT 的选择阶段训练参数。

### 3.5 不属于外部文献基线的本地控制

以下方法不应出现在第4.3节外部基线分类中：

- `closed_form`：ASTER-CR 内部方法实现。
- `legacy_m3`：ASTER 历史内部修正版本。
- `last_layer`：本地简单残差控制。
- `reference_fusion`：本文实现的监督双分数融合，不是独立文献方法。

这些控制可以在方法分析、简单基线或消融中报告，但必须明确其性质。

## 4. 历史结果可复用性

### 4.1 旧实验目录

`results/aster_cr_repair_v2_20261108/` 中存在：

- 比较器开发集分片预测和指标；
- LoReFT、LoFiT、LoRA 的训练记录及开发集评价；
- 历史模型和运行配置。

但是，运行快照明确记录 Qwen 测试完成状态为 false，MedGemma 正式评价未完成，外部模型评价未完成。因此，这些记录不能视为当前协议下完成的正式测试结果。

### 4.2 与 E1 的兼容性

当前正式配置为：

- E1 / Dataset A；
- 2,540 对；
- 仅 MEGA-CDP；
- 类别与来源均衡；
- E2、E3 仅作为数据和权重消融。

旧比较器的训练记录来自此前的 repair-v2 流程。即使模型名称和方法名称相同，也必须重新核对训练数据、提示、模型修订、评分器、来源划分和评价分母。只有通过这些检查，历史结果才可以复用。

**所有旧开发集比较结果均不得直接填入当前 E1 主表。**

### 4.3 配置文件边界

`configs/baselines.json` 当前记录的是 ProbeRAG、SEKA、JUICE、TrialGPT 等既有任务基线，不是 ASTER-CR 当前比较器注册表。

不要覆盖该文件。新的 ASTER-CR 方法兼容性信息应使用独立的机器可读审计文件，后续若要建立正式运行注册表，再单独设计。

## 5. 外部方法代表性与可执行性

### 5.1 输出校准与分数修正

代表方法：

- Temperature Scaling；
- Contextual Calibration；
- Batch Calibration；
- Context-Aware Decoding（CAD）。

优先顺序：

1. 保留当前温度和偏置控制；
2. 实现标准 CC；
3. 复核 CAD 的参考输入和系数；
4. BC 仅作为固定批次、无标签转导设置的补充实验。

CC 的内容无关参考与当前患者/条件参考提示不同，必须独立实现。BC 使用批次上下文统计，不得与逐样本源任务冻结结果混排。

### 5.2 提示与注意力控制

代表方法：

- 强证据提示；
- PASTA。

强证据提示应使用完整指南和患者事实，不引入测试标签或额外金标准证据。PASTA 需要预先定义强调片段；若依赖金标准证据跨度，必须说明额外监督，并避免测试标签泄漏。

### 5.3 概念擦除

代表方法：

- INLP；
- R-LACE；
- LEACE；
- KRaM；
- LEOPARD；
- Obliviator。

优先建立合法的实例级干扰概念标签。无法建立时，采用配对适配并明确标注。LEOPARD 和 Obliviator 是重要的近期非线性候选；KRaM 可作为较早的核化率失真路线参照。

### 5.4 表示干预与激活引导

代表方法：

- ITI；
- CAA；
- SEA；
- CAST；
- RePS；
- PSR；
- HyperSteer。

建议优先验证 CAA、CAST、RePS 和 PSR。CAA 的官方数据流程依赖 GPT-4 生成行为数据；RePS 在 AxBench 中提供基于偏好对的 BiPO 配置；PSR 需要提示引导教师和逐位置激活模仿数据。它们的监督信号和干预机制并不相同：

- CAA：对比激活差形成共享方向；
- CAST：使用条件向量选择性触发行为向量；
- RePS：基于偏好对的表示引导；
- PSR：将提示引导产生的激活变化蒸馏为逐位置系数。

不能将四者统一改写成一个自定义向量加法程序后称为原方法复现。

AxBench 仓库同时提供 RePS、HyperSteer、Prompt 和 ReFT-r1 等 rank-1 steering 参照。ReFT-r1 是 AxBench 中的具体基线配置，不能与本地通用 LoReFT 适配器混称。

### 5.5 表示微调与权重微调

代表方法：

- LoFiT；
- LoReFT / ReFT-r1；
- LoRA；
- DoRA；
- FourierFT。

LoReFT 与 ReFT-r1 不是同一个实验配置。当前本地 LoReFT 实现不能直接代表 AxBench 中的 ReFT-r1。

LoRA 需要增加标准多层配置，当前单层 q_proj rank-1 仅能作为局部低秩适配控制。DoRA 和 FourierFT 应使用同一数据、训练预算和评价协议，并分别报告参数量和计算成本。

## 6. 第4.3节与主表的纳入规则

第4.3节负责介绍外部方法类别和代表性文献；主比较表只报告在共同实验协议下实际完成评价的方法。

建议分类：

1. 输出校准与参考分数修正；
2. 提示与注意力干预；
3. 线性及非线性概念擦除；
4. 表示干预与激活引导；
5. 表示微调；
6. 参数高效权重微调；
7. 常规监督微调控制。

每类可以介绍数篇代表性工作，但主表不能把未运行方法当成有数值的方法。

每种方法至少报告：

- 原始训练目标；
- 使用的监督信号；
- 是否使用 A/P 或 C/F 任务适配；
- 模型修订；
- 干预层/位置；
- 可训练参数量；
- 前向次数；
- 自然节点和路径指标；
- A/P 干扰稳定性；
- C/F 证据响应；
- 纠正和新增错误；
- 来源簇置信区间。

当前仓库正文的外部基线比较表位于第4.2节/表2，而第4.5节是 MedPIC 结果。正文的外部比较表仍包含旧训练配置与历史评价总体。后续修改时需要统一章节编号，并在 E1 外部基线实验完成后更新表格，不能直接把旧表改名为新表。

## 7. Codex 00 执行依赖与调度

### 阶段 A：本轮已完成

- 仓库静态审计；
- 本地比较器实现审计；
- 官方论文与代码来源核对；
- 兼容性 JSON 注册表；
- 当前实验框架中的 Codex 执行状态和依赖记录。

### 阶段 B：运行前冻结

1. 确定每种概念擦除方法的实例级标签映射。
2. 冻结原生方法与任务适配版本的命名。
3. 固定官方代码提交、依赖环境和模型架构适配。
4. 固定共同的 E1 训练、开发和评价数据清单。
5. 确定参数量和计算预算的报告方式。

### 阶段 C：Qwen3-8B 主比较

先复核已有方法，再补当前 E1 结果：

- 温度、偏置和逻辑校准；
- CAD；
- 标准 CC；
- 强证据提示；
- 监督读出；
- LEACE 的标准版本与配对适配版本；
- ITI、SEA；
- LoFiT、LoReFT、LoRA。

SEA 回退和 LoReFT 加载问题必须先解决，未通过 smoke test 不进入正式比较。

### 阶段 D：新增外部方法

优先：

- CAA；
- CAST；
- RePS；
- PSR；
- INLP 或 R-LACE；
- DoRA；
- FourierFT。

随后根据标签、依赖和资源情况决定是否运行：

- KRaM；
- LEOPARD；
- Obliviator；
- HyperSteer；
- PASTA。

### 阶段 E：跨模型和消融

在 Qwen3-8B 的主比较协议稳定后，执行其他模型关键基线、跨任务冻结评价、ASTER-CR 内部消融和效率分析。

### 阶段 F：最终审计

重新从逐例预测计算指标、来源簇置信区间、纠正/新增错误和成本。只有真实完成共同评价的方法进入主表。

### 并行规则

- 文献核查、代码阅读和 CPU 单元测试可以并行。
- 多个 Codex 不得同时修改同一个 `comparators.py`、`baselines.py` 或共享训练入口。
- 新方法使用独立工作树和独立输出目录。
- GPU 任务必须由一个调度者统一分配，避免多个会话重复占用同一设备。
- 实验失败后保留日志和输出，不因结果不佳而选择性重跑。

## 8. 运行环境与资源风险

静态代码审计发现：

- `experiments/aster_cr/common.py` 默认根目录仍为旧的 `/home/gcx/aster_cr_20261108`；
- 默认 GPU 字符串为 `0,1`；
- 模型注册与实验框架计划的五模型清单并不完全一致；
- `requirements-aster-manuscript.txt` 仅声明基础 PyTorch 和 Transformers；
- PyReFT 和 SEA 的部分代码依赖没有完整提交到公开仓库。
- `configs/environment.lock.txt` 固定 PEFT 0.21.2、PyTorch 2.8.0 和 Transformers 4.57.6，但没有 pyreft；不能把 pyreft 视为当前环境已安装。
- `experiments/aster_cr/common.py` 的随机种子为 20261108，而当前实验框架要求随机外部基线使用 11、22、33。E1 数据选择种子保持冻结；外部基线代码需增加显式 seed 参数。
- `configs/models.json` 中 Qwen3-8B revision 为 `master` 且 file_revisions 为空；ModelScope manifest 的 Revision/Sha256 字段也为空。服务器运行前必须记录实际本地模型和分词器快照版本。

因此，正式运行必须显式设置新的 `ASTER_CR_ROOT`、GPU 列表、模型路径和独立输出目录。不能直接依赖旧默认值。

另外，当前代码默认随机种子为 20261108，而实验框架要求随机外部基线使用 11、22、33；E1 已冻结的数据选择种子不得因此改变。正式基线代码应增加显式 seed 参数并记录逐种子结果。`configs/models.json` 将 Qwen3-8B revision 记为 `master`，且 `file_revisions` 为空；`configs/qwen3_8b_modelscope_manifest.json` 的 Revision/Sha256 字段也为空。新运行必须在服务器记录实际本地模型和分词器快照版本，不能用移动的 `master` 代替不可变模型修订。`configs/environment.lock.txt` 固定 PEFT 0.21.2、PyTorch 2.8.0 和 Transformers 4.57.6，但没有 pyreft；`requirements-aster-manuscript.txt` 仅列基础 torch/transformers。pyreft、KRaM、Obliviator 和 PSR 应使用单独的可选依赖或隔离环境，不能为了单个基线升级共享主环境。

当前对话没有服务器 shell 或 GPU 监控接口，因此没有检查实时 GPU 空闲情况。启动任务前必须检查实际设备占用、显存和已有任务。服务器执行说明中列出的 GPU 0、1、5、6 及 PR #41 保护要求应在运行时再次核验；不得推断当前设备已经空闲。

`docs/ASTER/server-execution-task.zh.md` 中的部分方法定义属于早期实现路线。当前论文和实验协议以仓库 README、`manuscript.zh.md` 和 `experiments.zh.md` 为准，不能用旧方法规格替代当前 M1—M4 定义。

## 9. 本轮交付状态

| 项目 | 状态 |
|---|---|
| Codex 00：仓库与实验依赖静态审计 | 已完成 |
| Codex 00：执行阶段与并行边界 | 已完成 |
| Codex 01：本地基线代码审计 | 已完成 |
| Codex 01：官方方法及代码来源核对 | 已完成 |
| 机器可读兼容性注册表 | 已创建 |
| 实验框架状态更新 | 已更新 |
| 服务器实时任务和 GPU 检查 | 未执行，需在服务器端完成 |
| 真实模型 smoke test | 未执行 |
| 新增 GPU 实验 | 未启动 |
| 论文正文修改 | 本轮未修改 |