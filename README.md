# TRIAGE · ASTER

ASTER 研究大语言模型如何根据患者事实，准确判断临床指南的单步条件。新方法保留完整指南和病历，在输出“是／否”的位置，对少量注意力头施加闭式计算的干预，减少后续处置方案和无关患者背景对判断的影响，同时保留条件定义与决定性事实对应的表示方向。

**当前文档主线是闭式 ASTER 方法设计。三个内部机制及新方法效果尚待实验验证。** 仓库保留旧版可学习表示干预的实现与结果；其 **89.21%（6,653／7,458）** 完整路径正确率属于旧版 Qwen3-8B 配置，不能作为新方法的实验成绩。

- [新版论文正文：完整叙事、方法公式与验证安排](docs/ASTER/manuscript.zh.md)
- [本次方法补充：关键定义、公式修正与复现说明](docs/ASTER/method-clarifications.zh.md)
- [旧版表示干预论文与实验结果](docs/ASTER/manuscript-representation.zh.md)
- [旧版方法补充与复现步骤](docs/ASTER/supplementary.zh.md)
- [旧版表示干预代码与冻结权重](experiments/aster_representation/)
- [数据集与使用状态](#数据集与使用状态)
- [更早的规则键编辑文档](docs/ASTER/archive/README.md)

## 从临床指南执行到单步条件判断

临床工作既包括根据症状和检查推断诊断，也包括在诊断明确后，依据指南条件选择检查与治疗路径。ASTER 关注后一个阶段：给定指南、患者事实和当前节点，模型需要判断该节点的条件是否成立，再由程序选择对应分支。

完整方案生成中的跳步、顺序错误和处置改写，可以通过程序化流程控制减少：将指南表示为决策树，由程序维护节点顺序，并原样输出树中预先定义的处置。这样，模型的主要职责就成为**单步条件核对**。即使流程由程序控制，模型仍可能误判阈值、条件组合或脚注定义，从而选择错误分支。

我们关注其中一种可能的原因：模型根据分支后面的处置方案或患者整体印象，反过来判断条件是否成立。例如，背景描述使模型倾向某种处置，即使当前节点要求核对的指标支持另一分支。该解释需要通过配对测试和内部干预验证，不能仅凭最终错误或生成的分析文字确认。

## 用 2×2 配对测试区分应当忽略与应当响应的变化

对指南和病历分别构造两类变化：一类保持当前条件真值不变，另一类明确使真值翻转。每对样本仅修改目标内容，其余信息保持一致。

| 信息来源 | 干扰变化：判断应保持不变 | 核心信息变化：判断应随真值翻转 |
|---|---|---|
| 指南文本 | **处置改写 $\tau_A$**：条件及其必要定义不变，对调或替换分支后的处置描述。 | **界定改写 $\tau_C$**：改变阈值或必要脚注定义，使当前患者是否满足条件发生变化。 |
| 患者病历 | **画像扰动 $\tau_P$**：决定性事实不变，修改与当前条件无关的背景描述。 | **决定性事实微调 $\tau_F$**：背景不变，使关键事实跨过判定边界。 |

该测试同时要求模型对无关变化保持稳定、对有效证据变化作出响应。年龄、体能或合并症是否无关，取决于当前条件；若它们参与条件定义，就应作为有效证据保留。

## 两个挑战：完整输入中同时存在有效证据和干扰

**挑战一：指南下文可能同时包含处置方案与必要定义。** 节点可能只写“无相关禁忌证”，具体定义却出现在方案说明或脚注中。统一删除下文可能减少处置诱导，也可能删除条件核对所需的证据。需要统计这种依赖在真实指南中的比例，并比较完整输入、外部遮挡和新方法在 $\tau_A$ 与 $\tau_C$ 上的表现。原叙事中的 $Q\%$ 须区分原文定义依赖与删除导致的实际损害，见下文修订预实验。

**挑战二：删除处置文字仍可能留下患者背景造成的干扰。** 即使指南不显示后续方案，模型仍可能依据患者整体印象，而没有正确核对当前条件。我们用 $\tau_P$ 与 $\tau_F$ 检查这种情况，并将外部遮挡、输出融合和对比解码纳入比较。两路模型同时答错并不构成对比解码必然失败的证明；各方法能否改善判断，由相同配对样本上的结果决定。

## 三个待检验的内部机制

在模型参数冻结的条件下，我们考察产生条件判断的预测位置 $t^\ast$，重点分析该位置各层注意力头的输出。对该位置的干预不会改写更早输入位置的表示，但最终判断仍受到残差、前馈层及后续计算的共同影响。

1. **稀疏传导：干扰效应是否集中于少量注意力头？** 对 $\tau_A$ 与 $\tau_P$ 配对进行激活修补，检查替换少量头的输出能否恢复判断。原叙事中的“80%～90% 的错误反转由不到 $K\%$ 的头解释”属于待验证的数量结论。
2. **方向可分：干扰变化与有效证据变化是否对应不同方向？** 比较四类配对的激活差，检查干扰子空间与信号子空间的夹角，并验证这种区分能否推广到其他指南。
3. **信号充分：仅擦除干扰是否足以改善判断？** 先检查错误样本中能否读取正确的条件信息，再直接验证“不放大信号、仅擦除干扰”的效果。可解码信息的存在本身，不能证明擦除后模型必然答对。

## ASTER 的三个模块

**M1：干扰通路定位——确定在哪里干预。** 离线使用校准指南中的 $\tau_A$ 与 $\tau_P$ 配对，先用一阶梯度近似筛选注意力头，再对候选头做精确激活替换，选出能够共同解释较多干扰效应的紧凑集合 $\mathcal H$。分别得到处置与画像头集合，并向 M2 传递每个头对应的干扰类型 $T_h$；头集合的作用通过双向联合修补验证。

**M2：反事实子空间估计——确定擦除和保留的方向。** 对每个选中头，收集四类配对的激活差，并通过未中心化二阶矩的特征分解估计子空间。各类型先独立按能量选取方向，再合并：每个头的干扰基 $U_N$ 只使用 $T_h$ 中的类型，信号基 $U_S$ 同时使用 $\tau_C,\tau_F$。处置配对两侧的平均激活提供参考均值 $\mu$，它不自动意味着真假中立；固定算子对配对差分的作用为 $E\Delta z$，不依赖该均值。

**M3：信号保持擦除——构造闭式斜投影。** 根据两个子空间的主夹角，保留与信号空间分离充分的干扰方向 $\bar U_N$，构造满足

$$
E U_S=U_S,\qquad E\bar U_N=0
$$

的算子。它精确保留所估计信号子空间中的方向，并擦除筛选后的干扰方向；这一代数性质不等于对所有有效信息或最终答案的正确性保证。

在线推理时，对每个 $h\in\mathcal H$ 执行：

$$
\widetilde z_h=\mu^h+E^h(z_h-\mu^h).
$$


模型参数无需训练或微调；离线阶段包含梯度筛选、激活修补和子空间估计，擦除算子采用闭式计算。在线对每个待判断节点仅使用一次完整上下文计算，无需额外运行处置盲参考模型；工程实现可以复用上下文缓存。

两个挑战分别对应表1的两行，但都经过同一条处理流程：**干扰配对定位头 → 四类配对估计方向 → 保护优先的擦除**。三个模块的接口、未达到恢复目标时的处理、能量截断和在线缓存方式，见[方法正文第4节](docs/ASTER/manuscript.zh.md#4-方法aster)。

输出融合的理论分析采用明确条件：当完整与遮挡阅读共享同一画像贡献，且融合权重在画像配对内固定时，系数和为1的 logit 融合保留该画像响应。这不排除融合通过改善信号间隔提高配对准确率，也不覆盖任意自适应选择器。标准 CAD 强化完整阅读，遮挡主导的反向适配单独比较，见[理论性质](docs/ASTER/manuscript.zh.md#45-理论性质及其可验证范围)。

## 既有实现与实验结果

以下内容保留仓库已有数据、历史结果和复现入口。新闭式 ASTER 的实验结果将在实际运行后单独补充。

旧版可学习表示干预同时使用完整指南与处置盲参考，通过可学习的四维子空间和门控融合两路表示。其冻结版本为 Qwen3-8B 第 18 层、第 3 轮的 `no_js`，在 MEGA-CDP 官方 7,458 例单轮测试上取得 **89.21%** 完整路径正确率。

从仓库根目录查看该旧版的数据准备、训练、预测和评分命令：

```bash
python -m experiments.aster_representation --help
```

## 修订遮挡与关键事实预实验（2026-11-05）

已完成原文定位、完整逻辑标签和融合选参修正，固定 40 份指南、379 个统计节点，最终保存 2,994 条有效评分。原文定义依赖为 71/275（25.82%）；测试集实际删除 205 对，准确率 93.66%→94.15%，11 个定义补回病例均无准确率变化。

关键事实跨阈值测试中，完整输入与直接遮挡均有 16/28 对至少一例判错。对固定预测的补充汇总显示，完整＋条件及祖先融合将失败对从完整输入单路校准的 14/28 降至 11/28，提供有限互补证据；完整＋直接遮挡融合与单路校准预测相同。本轮没有验证内部干预机制，继续作为自动预实验报告。

- [结果分析与旧结果替代说明](reports/semantic_masking_followup_20261105/ANALYSIS.zh.md)
- [冻结报告](reports/semantic_masking_followup_20261105/REPORT_full.zh.md)
- [逐例输入、分数及结果快照](results/semantic_masking_followup_20261105/README.md)
- [代码与 CPU 复算入口](experiments/semantic_masking_followup_20261105/README.md)

## 三数据集补充分析（2026-11-05）

已完成 MedPAIR 信息删除与证据补回、MedGUIDE 方案交换与规则阈值变化、MedEinst 关键事实响应，以及两路输出融合。复用已有标注和历史结果，零新增标注；MedPAIR 新增 538 个唯一输入的评分。

- [实验说明、报告与四张主表](reports/three_dataset_analysis_20261105/README.md)
- [结果分析](reports/three_dataset_analysis_20261105/ANALYSIS.zh.md)
- [汇总数据与固定融合参数](results/three_dataset_analysis_20261105/README.md)
- [实验与复现代码](experiments/three_dataset_analysis_20261105/README.md)

MedPAIR 删除高相关句后，配对准确率下降 16.09 个百分点；证据补回的优势仍不确定。MedGUIDE 的方案交换既可能损害也可能改善判断；MedEinst 中分数支持改善经常不足以得到正确诊断。原始 A 的学习式融合纠正双错 3/140，准确率净提高 2.61 个百分点，同时对至少一路正确的样本产生损害 37/511。以上为既有数据的补充分析，与旧版 ASTER 表示干预结果分别报告。

## 数据集与使用状态

更新日期：2026-11-05。按同系列子集合并、同一数据集的版本不重复计数，仓库已登记准备或实际使用 **16 个外部数据集系列：13 个任务数据集、3 个通用基线数据集**。另有未获取的 MedNLI，以及自建或派生实验集。

计数时将 FaithEval 的三个子集合为一个系列，旧版 NLI4CT 与 NLI4CT-2024 不重复计数；TrialGPT 和通用基线数据的使用记录来自历史实验分支。下载状态主要依据 [数据配置](configs/datasets.json)，**已下载不等于已完成评测，各历史方法的结果分别归属其冻结版本，均不能作为新闭式 ASTER 的实测结果**。以下任务适用性用于说明研究用途，不能代替实际实验结论。

### 任务数据集

| 数据集 | 数据内容及适用理由 | 仓库中的使用状态 |
|---|---|---|
| **[MEGA-CDP](https://huggingface.co/datasets/nchennnn/MEGA-CDP)** | 提供临床指南、结构化决策树、指南衍生病例和参考路径，可评价“条件判断 → 分支选择 → 最终处置”的完整过程，最直接对应逐节点干预。 | 旧版表示干预已完成官方 7,458 例单轮原指南测试，见 [论文结果](docs/ASTER/manuscript-representation.zh.md) 和 [保存指标](docs/ASTER/results/single-turn.json)。 |
| **[MedGUIDE](https://arxiv.org/abs/2505.11613)** | 基于 55 棵 NCCN 指南树、17 类肿瘤构造 7,747 道选择题。结合完整候选树，可检验是否依据患者条件选择指南处置，并构造处置交换、条件变化等实验。 | 历史分支已有 200／800 例及多轮派生实验，见 [200 例研究](https://github.com/chunxi-alpc/TRIAGE/pull/23) 与 [800 例研究](https://github.com/chunxi-alpc/TRIAGE/pull/25)；这些不是旧版表示干预版本的完整外部测试。 新增[800 例分支评分补充分析](reports/three_dataset_analysis_20261105/table2.md)与[输出融合结果](reports/three_dataset_analysis_20261105/table4.md)。 |
| **MedPAIR** | 原生医学多选题及作者已有的句子相关性标注，可比较上下文删除与证据补回的判断代价。 | 本轮复用开发 91 题、测试 89 题；补回子集为 31/29 题，完成 Full/High 与补回的完整候选评分，见[删除与补回结果](reports/three_dataset_analysis_20261105/table1.md)。 |
| **MedEinst** | 关键事实改变后正确诊断随之改变的配对病例，可检验没有后续处置方案时的事实响应。 | 复用三个模型各 1,024 对病例的 M3A 候选分数，主结果为各 256 对测试；M1 和 Work3 单列补充，见[事实响应结果](reports/three_dataset_analysis_20261105/table3.md)。 |
| **[MedPIC-Bench](https://arxiv.org/abs/2608.03028)** | 467 道用药安全题，包含患者信息改变后规则适用性随之改变的反事实配对，适合检验模型是否真正使用患者条件。原生任务为多选，且不提供完整指南树，接入新闭式方法需要适配。 | 已登记下载，历史分支已有适配准备；尚无本次核对到的完整正式评测，见 [执行记录](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-seka-execution-20261025/experiments/aster_seka_execution/REPORT.zh.md)。 |
| **[TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)** | 专家标注患者是否满足临床试验条款，适合单独检查条件适用性，以及信息不足是否被误判为确定答案；不直接评价完整指南路径。 | 历史冻结 A1 已评估全部 1,015 条母集记录。原生六类到 T/F/U 的映射有信息损失，见 [实验报告](https://github.com/chunxi-alpc/TRIAGE/pull/20)。 |
| **[NLI4CT-2024](https://aclanthology.org/2024.semeval-1.271/)** | 根据临床试验文本判断陈述是否成立，并包含扰动，适合检查数值、否定和条件变化下的证据推断与一致性。 | 已登记下载；旧 NLI4CT 条目由 2024 版替代，不重复计算。 |
| **[MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench)** | 根据患者记录完成医学计算，可诊断数值读取、单位与计算错误；与指南分支执行的联系较间接，适合作为专项补充。 | 已登记下载；未列为已完成的完整评测。 |
| **[ConflictMedQA／DriftMed](https://huggingface.co/datasets/RDBH/DriftMed)** | 包含正确医疗建议及人为修改的错误版本，适合研究错误建议对模型的干扰。识别错误建议与忠实执行指定指南是不同评价目标，应分别解释。 | 历史分支做过建议级实验，并非原生全部场景的完整复现，见 [PR #10](https://github.com/chunxi-alpc/TRIAGE/pull/10)。两个名称在此指同一数据来源，不重复计算。 |
| **[MedChangeQA](https://arxiv.org/abs/2509.04304)** | 512 道医学共识随时间改变的问题，可用于诊断是否固守过时知识，为外部证据与模型记忆冲突提供医学场景；原生任务不是指南路径执行。 | 已登记下载；未列为已完成的完整评测。 |
| **[FaithEval](https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0)** | 包含反事实、上下文矛盾、不可回答三个子集，适合检验上下文忠实性与信息不足处理，可作为 CAD 相关研究的通用补充。 | 三个子集均已登记下载；按一个系列计数，未列为已完成的完整评测。 |
| **[ConFiQA](https://github.com/byronBBL/Context-DPO)** | QA、MR、MC 分别涉及单跳、多跳和多处知识冲突，适合检验多步推理是否持续使用给定上下文；属于通用任务。 | 三个子集均已登记下载；按一个系列计数，未列为已完成的完整评测。 |
| **[ClashEval](https://github.com/kevinwu23/StanfordClashEval)** | 通过修改外部信息研究模型如何在内部知识与外部证据之间选择，可诊断过度相信错误上下文；不直接评价患者规则适用性。 | 已登记下载；未列为已完成的完整评测。 |

### 通用基线数据集

以下三项用于基线实现验证或专家构建，不能替代医学任务评价。实际使用见 [SEKA／AdaSEKA 历史执行报告](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-seka-execution-20261025/experiments/aster_seka_execution/REPORT.zh.md)。

| 数据集 | 在仓库中的用途 |
|---|---|
| CounterFact | 以反事实知识及改写提示检验上下文引导；用于 SEKA、CAD 等方法的标准任务比较和相关专家构建。 |
| BiasBios | 以传记中的职业信息检验指定证据的利用；用于标准任务比较及 AdaSEKA 专家构建。 |
| HotpotQA | 提供多跳问答证据；用于 AdaSEKA 专家构建。构建专家不等于已完成该数据集的全面性能评测。 |

### 未获取的数据与自建实验集

- **MedNLI：**数据配置记为未获取原始数据，不计入上述 16 类。使用经过 MedNLI 训练的 BioLinkBERT 模型，不等于在 MedNLI 数据上完成了评测。
- **CARE 模拟病例与受控变体：**用于适用性、证据定位和干预诊断，见 [历史实验索引](experiments/README.md)；不能当作独立临床病例验证。
- **MedGUIDE-CF 等派生版本：**基于原数据构造处置交换、条件变化等实验；200／800 例和不同反事实版本不分别算作新的公开数据集。
- **MEGA-CDP 年龄最小对与四输入反事实组：**用于检查决定性事实变化和处置交换下的行为，见 [方法补充](docs/ASTER/supplementary.zh.md) 与 [Gemma 四输入确认](experiments/gemma3_head_pilot/results/expanded_confirmation_20261104.zh.md)；固定节点的构造反事实成绩与自主完整路径成绩分列。

### 新方法的数据安排建议

以 **MEGA-CDP** 验证完整路径执行，以 **MedGUIDE** 补充肿瘤指南任务，以 **MedPIC-Bench** 检验患者条件变化下的适用性。**TrialGPT** 用于辅助分析条件判断和信息不足错误，**FaithEval** 可补充上下文忠实性评价。上述安排是研究建议；既有完成情况以上表和对应实验报告为准；新闭式方法尚待独立评测。

## 旧版 Gemma 3 27B 选层选头先导与四输入补评

**最新独立确认（100组，四项全部正确）：零干预85%、直接盲读94%、整层91%、选头85%。** [完整同表比较与冻结配置](experiments/gemma3_head_pilot/results/expanded_confirmation_20261104.zh.md)。该指标评测固定节点的构造反事实，和原病例完整路径正确率分列。

[实验说明与复现入口](experiments/gemma3_head_pilot/)：扫描992个层—KV头组，完成三个候选层的整层／选头训练和未选头对照。仅用GPU 0、1；本轮未运行官方7,458例。

原先导按开发PathAcc选参，100例独立确认集上，零干预77%、直接盲读89%、整层85%、选头77%、未选头对照77%。[确认结果](experiments/gemma3_head_pilot/results/pilot_confirmation.zh.md)；[开发集含两条基线的比较](experiments/gemma3_head_pilot/results/pilot_development.zh.md)。

按论文规则忠实执行主指标，新增20组受控开发反事实（10套开发指南，每组四种输入），补评全部18个已有检查点和两条基线。以四项全对率优先、原100例开发PathAcc次之、实际编辑量更小再次之，重新选择层、头组和轮次。[四输入统一对比表](experiments/gemma3_head_pilot/results/fourway_development_20261104.zh.md)。四项全对：零干预16/20、直接盲读19/20、新选整层16/20、新选头17/20。新选头的原开发PathAcc为74%，低于零干预79%，存在指标取舍。此项是20组初筛的开发选择结果；扩大后的独立确认另见下文。


### 100组开发＋100组独立确认（已完成）

在20组初筛基础上，新增来源分离的100组开发和100组确认，各20套指南。先在开发组比较全部18个检查点与两条基线，按四项全对率选择并冻结，再测独立确认；仍仅使用GPU 0、1，不新增训练。[运行协议和复现入口](experiments/gemma3_head_pilot/#100组开发选择与100组独立确认2026-11-04)。本轮已完成：[独立确认统一结果表](experiments/gemma3_head_pilot/results/expanded_confirmation_20261104.zh.md)。100组确认四项全对：零干预85、直接盲读94、整层91、选头85；整层相对零干预净提升6组（来源配对95%区间+2至+11个百分点），选头净提升0。冻结配置为整层索引61第3轮、选头索引59组0/3/12/13第3轮。


---

## 版本是怎样变化的

| 阶段 | 相对上一版改变了什么 | 保留下来的结论／下一步 |
|---|---|---|
| TRIAGE 初始框架 | 将指南规则、患者条件、证据和注意力干预分成模块 | 建立可执行基线；旧主线见 [pre-ASTER 快照](https://github.com/chunxi-alpc/TRIAGE/tree/archive/pre-aster-main-20261026) |
| ACRE（10-15） | 从固定增强规则，改为按适用性切换增强对象 | 完整规则 Oracle 切换未优于固定增强，停止这条具体方案 |
| CARE 模块一（10-18～20） | V4 修证据 → V5 修解析 → V6 仅融合证据 → V7 可靠性门控 → V8／V8.1 语义直判 → 冻结 A1 | A1 负责提供适用性研究信号；V6 保留旧接口，二者不是同一实验配置 |
| CARE 模块二（10-16～21，与模块一并行） | 4B 全层头分析 → B1–B4 子空间 → 有符号干预／偏置 → 因果与结构对照 → N1–N5、decide-or-ask | 发现可解码信息、改动注意力与改善最终答案是三个不同问题；未建立稳定的子空间独立优势 |
| ASTER 先导（10-21～25） | 从 A1 输出转向 MedGUIDE 盲判、裁剪、融合、动态注意力、CAD、SEKA 等可比较方案 | 200／800 例、阈值／顺序和匹配对照属于各自实验；多条复杂路线未超过简单基线 |
| ASTER 离线键方向（10-27～31） | 先在 0.6B 验证差分方向和读取，再在 Qwen3-8B 扫描 KV 头、学习方向、编辑缓存键 | 找到了 8B 的四头配置，但改变局部规则注意力尚未稳定改善最终决策；见下方原始产物 |
| ASTER 表示干预（10-31～11-02，历史实测版本） | 从改指南 KV 键转为在每个分支的注意力输出处，融合完整指南表示与处置盲参考；学习四维子空间与门控 | 选定零起始层 18、训练第 3 轮的 no_js；完整路径 6,653／7,458。这里选的是输出干预层，不再沿用旧四个 KV 头 |
| ASTER 闭式干预（11-05，新方法设计） | 用四类反事实配对定位注意力头、估计干扰与信号子空间，并构造保持信号方向的斜投影；在线无需处置盲参考分支 | 三个机制假设和新方法效果待验证；下文历史成绩不属于本版本 |

旧版表示干预的完整研究快照见 [research/aster-single-turn-20261102](https://github.com/chunxi-alpc/TRIAGE/tree/d8cf67d53d6b068d9b382114de4a5fa4b22f633b)，精简后的复现入口是 [experiments/aster_representation](experiments/aster_representation/)。旧键编辑实验的 457 例中期结果与旧版表示干预的 7,458 例完整测试是两套方法，分开引用。

## 旧版 Qwen3-8B 扫描选头：代码和结果索引

**扫描工作确实做过，入口是 PR #42。** [采集代码](experiments/aster_manuscript_20261030/collect.py)保存各层 KV 表示，[临床分析代码](experiments/aster_manuscript_clinical_20261030/analyze.py)比较秩 1／2／4／8，再由 [select_expert](src/aster/manuscript_core.py)按验证集的方向区分度和正负读取响应选头。有效候选为 **280 个 KV 头**；第 0 层的 8 个头因差分数值秩不足排除，原因记在 rank_selection.json。**rank=8 指子空间维数，最终选中的是 4 个 KV 头。**

所有层号和头号均从 0 开始；“层:头”中的头号是 KV 头号，每个 KV 头对应一组共享它的查询头。

| 冻结批次 | 选中的层:KV头 | 原始记录 |
|---|---|---|
| run03，rank 8 | 33:3、34:0、34:1、31:1 | [秩比较](experiments/aster_manuscript_20261030/public/run03/rank_selection.json) · [选中头](experiments/aster_manuscript_20261030/public/run03/selected_structure.json) · [冻结配置](experiments/aster_manuscript_20261030/public/run03/frozen.json) |
| run05，rank 8 | 21:4、21:6、24:3、29:7 | [秩比较](experiments/aster_manuscript_20261030/public/run05/rank_selection.json) · [选中头](experiments/aster_manuscript_20261030/public/run05/selected_structure.json) · [冻结配置](experiments/aster_manuscript_20261030/public/run05/frozen.json) |
| clinical01，rank 8 | **33:3、31:1、33:4、23:4** | [秩比较](experiments/aster_manuscript_clinical_20261030/public/clinical01/rank_selection.json) · [选中头与留出诊断](experiments/aster_manuscript_clinical_20261030/public/clinical01/structure.json) · [冻结配置](experiments/aster_manuscript_clinical_20261030/public/clinical01/frozen.json) |

三个批次的数据和冻结配置不同，不能混用头组。clinical01 的 43 例确认中，谱平移终点正确 20 例、原模型 19 例、文本裁剪 25 例；[457 例官方快照](experiments/aster_manuscript_clinical_20261030/public/interim_457_20261030/)仍是中期结果。[10 月 31 日选中头／对照头实验](docs/ASTER/archive/head-controls-20261031.zh.md)进一步发现：关闭选中头并未降低这 43 例的终点成绩；因此这组头有表示与读取证据，尚不能称为最有助于正确决策的头。

**选层选头的具体方法见 [说明](docs/ASTER/layer-head-selection.zh.md)。** 复现旧版表示干预方法直接用层 18；继续研究 KV 头时，从 clinical01 四头出发，与同层未选头做相同干预比较，依据开发集完整路径的纠正与退化作选择。

## 每个研究版本做了什么

下表逐一列出截至本次整理前的全部 46 个 PR；缺号 #26 不补造版本。日期为北京时间的 PR 创建日期，执行结果以相应冻结产物为准。开放或草稿只表示 GitHub 状态，不等于没有做过实验。

| 日期／PR | 工作与变化 | 合并状态 |
|---|---|---|
| 10-15 · [#1](https://github.com/chunxi-alpc/TRIAGE/pull/1) | ACRE：Qwen3-8B 的固定增强、适用性切换与同源 Oracle 对照；完整规则切换未优于固定增强。 | 开放，草稿 |
| 10-17 · [#2](https://github.com/chunxi-alpc/TRIAGE/pull/2) | CARE M2：重分析 4B 的 288 个层／KV 头，区分相关性选头与适用性选头及数值混杂。 | 本次已合并 |
| 10-17 · [#3](https://github.com/chunxi-alpc/TRIAGE/pull/3) | CARE M2：B1–B4 子空间、随机方向、偏置和因果干预；未见可靠的学习子空间优势。 | 本次已合并 |
| 10-18 · [#4](https://github.com/chunxi-alpc/TRIAGE/pull/4) | CARE M1 V4：修复证据定位，增加新规则完整病例验证。 | 开放 |
| 10-18 · [#5](https://github.com/chunxi-alpc/TRIAGE/pull/5) | CARE M1 V5：TrialMatchAI 解析修复，复评缓存输出。 | 开放 |
| 10-18 · [#6](https://github.com/chunxi-alpc/TRIAGE/pull/6) | CARE M1 V5 后续：失败条件重生成，解析器升级到 V3。 | 开放 |
| 10-18 · [#7](https://github.com/chunxi-alpc/TRIAGE/pull/7) | CARE M1 V6：仅融合证据，接入 pipeline；与后来的 A1 默认接口分别保留。 | 开放 |
| 10-19 · [#8](https://github.com/chunxi-alpc/TRIAGE/pull/8) | CARE M2：按金标适用性施加有符号 B4，比较均值平移与直接分数偏置。 | 本次已合并 |
| 10-19 · [#9](https://github.com/chunxi-alpc/TRIAGE/pull/9) | CARE V7：可靠性门控、固定对照与真实模型实验。 | 开放，草稿 |
| 10-19 · [#10](https://github.com/chunxi-alpc/TRIAGE/pull/10) | CARE 8B：ConflictMedQA／DriftMed 上下文冲突方向；独立于后来的临床规则选头。 | 开放 |
| 10-19 · [#11](https://github.com/chunxi-alpc/TRIAGE/pull/11) | CARE 因果链：4B 层段×强度与真实键编辑；注意力独立收益有限，结构修复更明显。 | 开放，草稿 |
| 10-19 · [#12](https://github.com/chunxi-alpc/TRIAGE/pull/12) | CARE M1 V8：直接做语义适用性判断，形成 A1 研究信号。 | 开放，草稿 |
| 10-20 · [#13](https://github.com/chunxi-alpc/TRIAGE/pull/13) | CARE M1 V8.1：完整条件与两轮复核；比较后保留 A1。 | 开放，草稿 |
| 10-20 · [#14](https://github.com/chunxi-alpc/TRIAGE/pull/14) | CARE A1：增加未知态 U 对照示例，准确率和 U 识别之间存在取舍。 | 开放，草稿 |
| 10-20 · [#15](https://github.com/chunxi-alpc/TRIAGE/pull/15) | CARE A1：冻结语义信号接口，供模块二研究使用；集成由 PR #16 完成。 | 开放，草稿 |
| 10-20 · [#16](https://github.com/chunxi-alpc/TRIAGE/pull/16) | 将冻结 A1 接入主线；同时保留 V6 的既有调用接口。 | 已合并 |
| 10-20 · [#17](https://github.com/chunxi-alpc/TRIAGE/pull/17) | CARE M2 增量 N1–N5：相关性／适用性、上下文／参数知识、可决断／须询问；原计划后续有实现，按分支最新产物追溯。 | 开放，草稿 |
| 10-21 · [#18](https://github.com/chunxi-alpc/TRIAGE/pull/18) | CARE decide-or-ask：决策／询问、不确定性传播与校准框架。 | 开放，草稿 |
| 10-21 · [#19](https://github.com/chunxi-alpc/TRIAGE/pull/19) | MedGUIDE：冻结 A1 与同输入 Qwen 全量对照的旧执行版本；PR 已关闭。 | 已关闭，未合并 |
| 10-21 · [#20](https://github.com/chunxi-alpc/TRIAGE/pull/20) | TrialGPT：冻结 A1 全量评估，分析 124 条 U 分歧。 | 开放，草稿 |
| 10-22 · [#21](https://github.com/chunxi-alpc/TRIAGE/pull/21) | ASTER MedGUIDE：修复先导实现，记录 Llama／Qwen 的真实试跑。 | 开放 |
| 10-22 · [#22](https://github.com/chunxi-alpc/TRIAGE/pull/22) | ASTER MedGUIDE：A/B/C 反事实预实验，进入主线。 | 已合并 |
| 10-23 · [#23](https://github.com/chunxi-alpc/TRIAGE/pull/23) | ASTER Qwen 200 例：A/B/C、CAD 与分支机制实验。 | 开放，草稿 |
| 10-23 · [#24](https://github.com/chunxi-alpc/TRIAGE/pull/24) | MedGUIDE 扩到 800 例；重算患者条件相关的阈值参考标签。 | 开放，草稿 |
| 10-23 · [#25](https://github.com/chunxi-alpc/TRIAGE/pull/25) | Qwen 800 例：比较 A/B/C/D/CS 方法，分析保留规则和 CS 错误。 | 开放，草稿 |
| 10-24 · [#27](https://github.com/chunxi-alpc/TRIAGE/pull/27) | 对照 Qwen 200／800 例，保存固定 CAD18 的完整词元轨迹。 | 开放，草稿 |
| 10-24 · [#28](https://github.com/chunxi-alpc/TRIAGE/pull/28) | 44 例阈值×规则顺序对照，补充 109 次请求。 | 开放，草稿 |
| 10-24 · [#29](https://github.com/chunxi-alpc/TRIAGE/pull/29) | 44 例先显式核对患者条件、再裁剪规则的对照。 | 开放，草稿 |
| 10-24 · [#30](https://github.com/chunxi-alpc/TRIAGE/pull/30) | ASTER E0–E8：修正金标前缀等混杂，独立控制盲判与门控，保存续跑结果。 | 开放，草稿 |
| 10-25 · [#31](https://github.com/chunxi-alpc/TRIAGE/pull/31) | ASTER P0–P3：独立测试的准确率／伤害取舍，未超过简单基线准确率。 | 开放，草稿 |
| 10-25 · [#32](https://github.com/chunxi-alpc/TRIAGE/pull/32) | ASTER：动作注意力质量守恒；Qwen 实测未超过简单融合。 | 开放，草稿 |
| 10-25 · [#33](https://github.com/chunxi-alpc/TRIAGE/pull/33) | ASTER：两模型动态融合开发实验及匹配负对照。 | 开放，草稿 |
| 10-25 · [#34](https://github.com/chunxi-alpc/TRIAGE/pull/34) | ASTER Qwen seed=0：注意力残差门控的离线对照与 GPU 协议。 | 开放，草稿 |
| 10-25 · [#35](https://github.com/chunxi-alpc/TRIAGE/pull/35) | ASTER Qwen seed=0：完成稀疏动态注意力 GPU 实验和全量对照。 | 开放 |
| 10-25 · [#36](https://github.com/chunxi-alpc/TRIAGE/pull/36) | ASTER／SEKA：官方基线复现、外部评测和后续执行计划。 | 开放，草稿 |
| 10-25 · [#37](https://github.com/chunxi-alpc/TRIAGE/pull/37) | ASTER／SEKA：执行上述研究，保存来源记录、配对结果与提取失败。 | 开放，草稿 |
| 10-27 · [#38](https://github.com/chunxi-alpc/TRIAGE/pull/38) | 离线适用性子空间：0.6B 机制预实验，局部读取改变未带来答案改善。 | 开放，草稿 |
| 10-28 · [#39](https://github.com/chunxi-alpc/TRIAGE/pull/39) | 统一 ASTER 实验章节、三类基线、评价公式和结果表。 | 已合并 |
| 10-28 · [#40](https://github.com/chunxi-alpc/TRIAGE/pull/40) | 0.6B 离线共享方向、双向读取与强编辑读出诊断；部分翻转不足以证明稳定答案控制。 | 开放，草稿 |
| 10-30 · [#41](https://github.com/chunxi-alpc/TRIAGE/pull/41) | 8B 全量旧执行：按用户要求提前停止；确认记录与失败保留，官方测试和 MedGUIDE 未开始。 | 开放，草稿 |
| 10-30 · [#42](https://github.com/chunxi-alpc/TRIAGE/pull/42) | 8B 键编辑：280 候选 KV 头、秩选择、四头冻结、43 例确认与 457 例官方中期快照。本次把原代码和公开产物收入主线。 | 开放，草稿 |
| 10-31 · [#43](https://github.com/chunxi-alpc/TRIAGE/pull/43) | 旧键编辑方案的方法修订：适用性评估、缓存注意力门控与 B0–B5；作为独立设计稿保留。 | 开放，草稿 |
| 11-03 · [#44](https://github.com/chunxi-alpc/TRIAGE/pull/44) | 为首页加入路线历史和 8B 选头索引。 | 已合并 |
| 11-03 · [#45](https://github.com/chunxi-alpc/TRIAGE/pull/45) | 更新历史分支索引。 | 已合并 |
| 11-03 · [#46](https://github.com/chunxi-alpc/TRIAGE/pull/46) | 补齐当前分支清单。 | 已合并 |
| 11-03 · [#47](https://github.com/chunxi-alpc/TRIAGE/pull/47) | 澄清 8B 扫描与 457 例官方中期结果的状态。 | 已合并 |

没有单独 PR 的支线，包括 P1–P6、适用性决策研究、模块说明和单轮表示研究，见 [完整分支索引](docs/BRANCHES.md)。

## 历史整合记录与复现入口

- 合并 PR #2、#3、#8：它们的实验改动已与主线一致，补齐实际合并关系。
- 将 PR #42 中主线缺少的 123 个代码、配置、公开汇总和报告文件原样纳入，保留原目录；该 PR 的整项研究状态仍独立保留。旧报告中的“当前／运行中”指报告当日，不代表现在的执行状态。
- 旧版可学习表示干预由[历史论文](docs/ASTER/manuscript-representation.zh.md)、[旧版补充与复现](docs/ASTER/supplementary.zh.md)和[表示干预代码](experiments/aster_representation/)说明；新闭式 ASTER 的方法与待验证结论见[新版论文](docs/ASTER/manuscript.zh.md)。
- 其余分支按版本索引保留各自结果和代码。PR #41 的提前停止、PR #42 的中期分母以及各版负结果均保留。

历史整理记录：当次整理没有启动模型实验；服务器上原键编辑实现的 37 项 CPU 单元测试通过。本次叙事更新同样没有启动实验。
