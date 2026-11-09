# 10 分钟汇报：对消，而非擦除

**一句话**：A/P学习应减掉的分数变化，C/F限制证据修正，独立校准阈值。忠实性是主要目标，准确率为下游效果。

[六章正文](../paper_story_latest.zh.md)中方法见第3章，实测见4.2，医学模拟见4.3；完整实测见附录C，完整模拟见附录D。


## 1 Introduction：问题

后续处置不应反推当前条件。直接删文本可能丢掉定义：历史路径正确数404/432→394/432。因此保留全文，学习并抵消干扰分数变化。


## 2 Related Work：对照

比较校准、普通读出、表示擦除/编辑与参数高效训练。任务目标是干扰不变性与证据响应，不只比较概念可恢复性。


## 3 Method：方法

16维痕迹给出ℓ′=ℓ+wᵀt+b。A/P拟合干扰、C/F限制证据改动，b单独校准。原fit为4,887对A/P、135对C/F，预算0.2923349 logit，761自然节点校准。理论和证明见附录A；一次原模型前向。


## 4 Experiments

### 4.1 Experimental Setup

主要评价为干扰RMS/VR、证据ER/修正RMS/Dir. Deg.；下游单列Acc.、Pair Acc.、Set EM和动态Path Acc.。RMS注明logit或模拟score，方向退化注明#或%，ER以100%为响应保留参考；↑/↓为评价方向，n/N为正确数/已评价数，pp为百分点。所有指标公式、分母、任务适用范围及零分母规则见正文4.1和附录B.1。无有效配对填“—”，不跨任务求总分。

正式方法名见表中Method列。ASTER-CR (Linear)为16维逐对硬约束实例；同痕迹MLP与99维MLP集成分别命名，平衡/强保护是不同开发集保护档。校准数据与监督适配条件在表注中注明，不混用冻结迁移与目标任务适配。

| Dataset / Task | Official Size / Native Splits | Local Availability | Evaluated Size | Evaluation |
|---|---:|---:|---:|---:|
| MEGA-CDP：指南节点与动态路径 | 2,274 棵树、42,353 路径；train_val 34,895，test 7,458 | 原生划分可用；原 fit/dev/留出为 1,712/420/806 路径 | 受控留出 3,100 输入、806 路径；官方 test 7,458 路径 | 干扰 VR/RMS、证据 ER/修正 RMS/方向、节点及路径正确率 |
| NLI4CT-2024：临床试验文本推断 | train/dev/practice_test/test 为 1,700/200/2,142/5,500 | 9,542 条跨 split 记录、9,342 原 ID；200 dev 也在 practice | 按原 split 分报 9,542 条；官方 test 原生保持/翻转 4,136/864 对 | 原生干预响应、配对全对、分类正确率 |
| MedPIC：患者特异的用药安全多标签题 | test 467 题，每题 5 选项；无原生训练划分 | 467 题、2,335 选项；90 组中 92 病例对映射为 244 翻转、216 保持选项对 | 已有冻结迁移 467 题和 460 选项对；本次不重复 | 答案集合全对、配对全对、证据响应与方向 |
| MedPAIR：医生句级相关性标注的单选题 | 官方标注覆盖 2,000 QA；发布题目/标注表，无本研究的原生 train/dev/test | 清理后 898 题；本地派生 train/dev/test 为 718/91/89 | 当前受约束方法新增实测 0；早期 180 题属历史版本 | 原题准确率，低相关与关键证据删除/补回的分数变化 |
| MedEinst：合成病例 control/trap 诊断配对 | 官方 reference train 10,689 对，test 5,383 对、49 疾病 | 历史行为缓存 train/test 10,244/5,372 对；缺项原因本次未确认 | 当前方法新增实测 0；历史缓存不冒充五模型实测 | 诊断、配对全对、Bias Trap Rate、证据响应 |
| MedGUIDE：指南单选题 | 7,747 题，55 棵 NCCN 树、17 类癌症；HF 单个 train 文件，无原生独立 test | 7,747 题，17 来源家族；历史子集另列 | 当前受约束方法新增实测 0 | 选择准确率、疾病宏平均；无干预配对，ER/VR“—” |
| TrialGPT：患者—试验条款匹配 | 专家条款标注 1,015 条，含证据句；HF 单个 train 文件，另有 training 标记 | 本地有官方代码和检索资源，尚无完整专家条款评测清单 | 当前方法新增实测 0；确定标签子集分母待实际清单确认 | 确定标签的平衡准确率、Macro-F1；无干预配对，ER/VR“—” |

**Table 1:** 医学评测数据集。官方发布、本地可用和实际评价规模分列；原生划分与本地派生划分分别注明。


数据来源与原生划分见正文4.1及附录B。五模型为Qwen3-8B、Llama-3.1-8B-Instruct、Gemma-2-9B-IT、Gemma-3-27B-IT、MedGemma-27B。


### 4.2 Main Results

**已有受控留出实测。**

| Method | $\mathrm{VR}_{I}$ (%) ↑ | $\mathrm{ER}_{E}$ (%) ≈100 | $\mathrm{RMS}_{\Delta c,E}$ (logit) ↓ | Path Correct (n/N) ↑ |
|---|---:|---:|---:|---:|
| Base Model | 0.00 | 100.00 | 0.0000 | 768/806 |
| Bias Calibration | 0.00 | 100.00 | 0.0000 | 767/806 |
| Supervised Linear Readout | 75.20 | 31.06 | 25.3734 | 737/806 |
| LEACE | 6.13 | 98.73 | 1.1118 | 767/806 |
| Closed-form ASTER | 6.76 | 99.88 | 0.5896 | 771/806 |
| ASTER-CR (Linear) | 4.10 | 99.60 | 0.2539 | 767/806 |
| ASTER-CR (MLP, 16-d) | 14.59 | 98.49 | 0.6491 | 770–771/806 |
| MLP Representation Editing | 13.12 | 98.43 | 0.6704 | 768–769/806 |
| ASTER-CR (MLP Ensemble, Strong Protection) | 8.02 | 99.88 | 0.1201 | 767/806 |
| ASTER-CR (MLP Ensemble, Balanced) | 14.59 | 99.77 | 0.2403 | 768/806 |

**Table 2:** 既有受控留出实测方法比较。RMS单位为logit；Path Correct显示正确数/806。


**官方MEGA测试。**

| Method | Path Correct (n/N) ↑ | Path Acc. (%) ↑ | 95% Cluster CI (%) | Leaf Acc. (%) ↑ | Fixed / New Errors (#) | ΔAcc. (pp) | 95% CI of Δ (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Model | 7,037/7,458 | 94.36 | [93.44, 95.20] | 94.36 | — | — | — |
| Bias Calibration | 7,038/7,458 | 94.37 | [93.45, 95.21] | 94.37 | 45/44 | 0.01 | [-0.31, 0.31] |
| ASTER-CR (Linear) | 7,034/7,458 | 94.31 | [93.39, 95.18] | 94.31 | 43/46 | -0.04 | [-0.34, 0.24] |
| ASTER-CR (MLP Ensemble, Strong Protection) | 7,037/7,458 | 94.36 | [93.46, 95.21] | 94.36 | 26/26 | 0.00 | [-0.21, 0.20] |
| ASTER-CR (MLP Ensemble, Balanced) | 7,041/7,458 | 94.41 | [93.51, 95.24] | 94.41 | 20/16 | 0.05 | [-0.11, 0.21] |
| Trace-only MLP | 7,044/7,458 | 94.45 | [93.55, 95.28] | 94.45 | 23/16 | 0.09 | [-0.09, 0.27] |

**Table 3:** MEGA-CDP官方测试的7,458条动态路径。区间为来源簇配对重采样所得95%区间；差值以百分点计，相对于Base Model。Fixed/New Errors分别为同一已评价全集上的纠正与新增错误。


**官方NLI测试：冻结MEGA方法零样本迁移。**

| Subset | Expected N | Method | Correct (n/N) ↑ | Acc. (%) ↑ |
|---|---:|---|---:|---:|
| 官方test声明（零样本迁移） | 5,500 | Base Model | 4,073/5,500 | 74.05 |
| 官方test声明（零样本迁移） | 5,500 | ASTER-CR (Linear) | 4,031/5,500 | 73.29 |
| 官方test声明（零样本迁移） | 5,500 | ASTER-CR (MLP Ensemble, Balanced) | 4,048/5,500 | 73.60 |
| 官方test声明（零样本迁移） | 5,500 | Bias Calibration | 4,033/5,500 | 73.33 |
| 官方test声明（零样本迁移） | 5,500 | ASTER-CR (MLP Ensemble, Strong Protection) | 4,042/5,500 | 73.49 |
| 官方test声明（零样本迁移） | 5,500 | Trace-only MLP | 4,040/5,500 | 73.45 |

**Table 4:** NLI4CT-2024官方测试的5,500条声明。所有修正函数及截距在MEGA上拟合并冻结，属于零样本迁移。


MEGA平衡配置路径差0.05个百分点，区间[−0.11,0.21]。固定原错误池MEGA424路径、NLI1,469记录单列，新模型没有重测原正确全集，不能推算净收益。

**MedPIC：保留另一线程已有实测，本次不重复。**\n
<!-- medpic-frozen-transfer:start -->
官方数据共467题、2,335个选项；现有90组含92个病例对，按选项文本映射得到244个标签翻转对和216个标签保持对。每个选项独立作是／否判断，再还原多选答案集合。全部复用既有权重、幅度和截距，无新增训练或校准。

| Method | $\mathrm{ER}_{E}$ (%) ≈100 | $\mathrm{RMS}_{\Delta c}$ (logit) ↓ | Dir. Deg. (#) ↓ | Option-pair Correct (E) (n/N) ↑ | Option Acc. (%) ↑ | Set EM, n/N (%) ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Base Model | 100.00% | 0.0000 | 0 | 25/244 | 68.65% | 98/467（20.99%） |
| ASTER-CR (Linear) | 100.45% | 0.1179 | 0 | 31/244 | 65.82% | 70/467（14.99%） |
| ASTER-CR (MLP Ensemble, Balanced) | 93.82% | 0.9937 | 0 | 32/244 | 67.02% | 78/467（16.70%） |
| ASTER-CR (MLP Ensemble, Strong Protection) | 96.91% | 0.4968 | 0 | 33/244 | 66.85% | 78/467（16.70%） |

**Table 5:** MedPIC冻结迁移实测：467题、2,335选项、244个翻转选项对。Set EM要求整题集合完全正确；配对正确数以选项对为单位。RMS单位为logit，Dir. Deg.为计数。


本轮结论：原16维方法保留了原模型的平均证据响应，翻转对全对由25/244升至31/244；但整题全对由98/467降至70/467。平衡版和严格保护版的翻转对全对分别为32/244和33/244，整题均为78/467，仍低于原模型。四种方法均无响应方向退化。当前冻结迁移未带来整体准确率提升，响应保留与任务准确性需分别评价。

标签保持的216对单列如下；患者信息仍有变化，不能直接视为无关信息扰动。

| Method | $\mathrm{RMS}_{\mathrm{base}}$ (logit) | $\mathrm{RMS}_{m}$ (logit) | $\mathrm{RMS}_{\Delta c}$ (logit) ↓ | Pair Correct (n/N) ↑ |
|---|---:|---:|---:|---:|
| Base Model | 4.5870 | 4.5870 | 0.0000 | 163/216 |
| ASTER-CR (Linear) | 4.5870 | 4.6249 | 0.0996 | 145/216 |
| ASTER-CR (MLP Ensemble, Balanced) | 4.5870 | 4.4393 | 0.5752 | 156/216 |
| ASTER-CR (MLP Ensemble, Strong Protection) | 4.5870 | 4.5046 | 0.2876 | 152/216 |

**Table 6:** MedPIC的216个标签保持选项对。患者事实仍改变，因此本表不将其命名为纯干扰；三种RMS分别为原响应、修正后响应和方法引起的配对差改动，单位均为logit。


响应保留率沿用既有计分接口，为标签方向加权后的平均响应相对原模型的比值；修正RMS计算配对分数差的改变量，方向退化计原本沿正确标签方向的响应变为非正的对数。响应保留率接近100%只说明接近原响应，需结合配对全对率判断。

全部选项和460个选项对均有结果。两个分片均通过原计分接口的一致性验证，零修正还原原始分数及预测。原16维方法固定保存798个错选项、397个错题和284个错误选项对，后续优化仅复测这些错例。

结果目录：`/home/gcx/aster_medpic_20261109`；逐例结果为 `option_results.jsonl`、`question_results.jsonl`、`pair_results.jsonl`，指标为 `metrics.json`，固定错例位于 `fixed_old_linear_errors/`。

<!-- medpic-frozen-transfer:end -->

<!-- medpic-intercept-adaptation:start -->
**MedPIC-Bench: Legal-Set Decoding and Bias Adaptation.**

按既有病例组抽取18组及57个单例，得到93个开发病例（465个选项）。随机种子20261109，同组不跨集合。四种方法各自仅用开发选项交叉熵拟合一个截距，替换旧截距，权重和修正幅度不变。

复测仅包含开发集以外、事先固定的原16维错题，共324题。另复测656个固定错选项及237个固定错误选项对，其中翻转177对、标签保持60对。以下均为错误子集结果，不代表完整467题准确率。

合法集合解码只允许“以上皆非”单选，或不含该项的其他选项非空子集，按已有logit选择最大独立二分类似然；无此选项的题沿用原独立判断。原模型也使用相同解码和校准。

| Method | Independent Set Correct (n/N) ↑ | Legal-Set Correct (n/N) ↑ | Calibrated Set Correct (n/N) ↑ | Calibration Fixed / New Errors (#) | Fitted Bias (logit) |
|---|---:|---:|---:|---:|---:|
| Base Model | 26/324 | 96/324 | 96/324 | 2／2 | -4.9716 |
| ASTER-CR (Linear) | 0/324 | 90/324 | 96/324 | 10／4 | -6.2211 |
| ASTER-CR (MLP Ensemble, Balanced) | 9/324 | 93/324 | 99/324 | 9／3 | -8.5298 |
| ASTER-CR (MLP Ensemble, Strong Protection) | 8/324 | 90/324 | 96/324 | 9／3 | -6.7534 |

**Table 7:** MedPIC固定324个原错误题的合法集合解码与截距适配。四方法共享解码规则，截距分别在独立93题开发集合拟合；正确数不代表完整467题准确率。

| Method | Legal-Set Pair Correct (n/N) ↑ | Calibrated Pair Correct (n/N) ↑ | ER (%) ≈100 | $\mathrm{RMS}_{\Delta c}$ (logit) ↓ | Dir. Deg. (#) ↓ |
|---|---:|---:|---:|---:|---:|
| Base Model | 28/177 | 21/177 | 100.00% | 0.0000 | 0 |
| ASTER-CR (Linear) | 37/177 | 21/177 | 100.26% | 0.1092 | 0 |
| ASTER-CR (MLP Ensemble, Balanced) | 37/177 | 21/177 | 91.95% | 0.9281 | 0 |
| ASTER-CR (MLP Ensemble, Strong Protection) | 37/177 | 21/177 | 95.97% | 0.4640 | 0 |

**Table 8:** MedPIC固定177个翻转错误选项对。配对全对数分别在合法集合解码与截距适配后计算；ER、修正RMS与方向退化按连续分数评价，截距不改变配对差。

相同解码和校准后，原模型在固定复测错题中全对96题，三个修正版本中最好为99题。修正版本在该固定错题子集上存在数值收益，不能据此推断全数据收益。

本轮校准对整题的收益较小，同时损害了固定翻转错对的配对判断：原模型由28/177降至21/177，三个修正版本均由37/177降至21/177。因此下一步保留合法集合解码作为统一读出方式，这次以独立选项交叉熵拟合的截距暂不作为最终配置。

截距在配对连续分数差中抵消，校准前后的ER、修正RMS及方向退化不变；合法集合的二值配对全对率另行统计。标签保持对仍是患者信息变化，不视作无关扰动。原16维在固定错例上的原准确率为零来自选集定义。

这是已看过整体结果后的探索性适配，开发拟合与本轮复测病例分开，没有根据复测结果选择截距或修改权重。原冻结迁移结果保留。全部复用缓存，没有GPU推理。

结果目录：/home/gcx/aster_medpic_20261109/intercept_adaptation/。split.json保存划分，frozen_intercepts.json保存参数，metrics.json与逐例JSONL保存结果。
<!-- medpic-intercept-adaptation:end -->




### 4.3 Medical Task Simulations

六条件比较原模型、仅校准、无证据约束、线性ASTER-CR、平衡MLP、强保护MLP。五模型×五任务×五种子11/22/33/44/55，共750条结果。下列为五模型等权模拟均值，完整每模型及生成标准差见附录D.3。

本节为待实测替换的模拟占位值。修正特征随证据变化，避免正系数缩放使方向退化必为零；保留原系数和种子，不按退化数量调参。方向退化%=原正向响应变为非正的数量／原模型正向响应数量，分母和整数计数见附录D.3及CSV；MedPIC按变化选项对，MedPAIR按固定margin的关键删除响应计。


**MedPIC。**

**预设模拟，未运行对应模型与数据集，待真实评测替换。**

| Method | ER (%) ≈100 | $\mathrm{RMS}_{\Delta c}$ (score) ↓ | Dir. Deg. (%) ↓ | Pair EM (%) ↑ | Set EM (%) ↑ |
|---|---:|---:|---:|---:|---:|
| Base Model | 100.00 | 0.00 | 0.000 | 49.96 | 57.87 |
| Bias Calibration | 100.00 | 0.00 | 0.000 | 54.17 | 63.73 |
| ASTER-CR w/o Evidence Constraints | 74.94 | 1.39 | 0.464 | 57.52 | 66.49 |
| ASTER-CR (Linear) | 99.47 | 0.37 | 0.000 | 55.70 | 64.84 |
| ASTER-CR (MLP, Balanced) | 96.87 | 1.16 | 0.087 | 55.91 | 64.34 |
| ASTER-CR (MLP, Strong Protection) | 98.92 | 0.62 | 0.000 | 57.96 | 66.70 |

**Table 9:** MedPIC五模型模拟概览。忠实性为ER、修正RMS和条件方向退化率；Pair EM以92病例对计，Set EM以467题计。每病例对含变化选项，方向退化以原正向变化选项对为分母。预设模拟，未运行对应模型与数据集，待真实评测替换。


**MedPAIR。**

**预设模拟，未运行对应模型与数据集，待真实评测替换。**

| Method | $\mathrm{VR}_{\mathrm{low}}$ (%) ↑ | Key-Del. Ret. (%) ≈100 | $\mathrm{RMS}_{\Delta c}$ (score) ↓ | Dir. Deg. (%) ↓ | Original Acc. (%) ↑ |
|---|---:|---:|---:|---:|---:|
| Base Model | 0.00 | 100.00 | 0.00 | 0.000 | 86.71 |
| Bias Calibration | 0.00 | 100.00 | 0.00 | 0.000 | 86.71 |
| ASTER-CR w/o Evidence Constraints | 88.81 | 75.12 | 0.79 | 1.945 | 85.05 |
| ASTER-CR (Linear) | 9.33 | 99.53 | 0.21 | 0.076 | 87.17 |
| ASTER-CR (MLP, Balanced) | 46.33 | 97.17 | 0.65 | 1.008 | 84.93 |
| ASTER-CR (MLP, Strong Protection) | 24.97 | 99.10 | 0.35 | 0.205 | 87.69 |

**Table 10:** MedPAIR五模型模拟概览。低相关删除的VR与关键删除响应保留分别报告；Key-Del. Ret.是固定margin敏感性，不是真值翻转ER；准确率仅针对898道完整原题。预设模拟，未运行对应模型与数据集，待真实评测替换。


**MedEinst。**

**预设模拟，未运行对应模型与数据集，待真实评测替换。**

| Method | ER (%) ≈100 | $\mathrm{RMS}_{\Delta c}$ (score) ↓ | Dir. Deg. (%) ↓ | Pair Acc. (%) ↑ | BTR (%) ↓ |
|---|---:|---:|---:|---:|---:|
| Base Model | 100.00 | 0.00 | 0.000 | 38.85 | 5.97 |
| Bias Calibration | 100.00 | 0.00 | 0.000 | 38.85 | 5.97 |
| ASTER-CR w/o Evidence Constraints | 75.05 | 1.86 | 0.038 | 36.63 | 2.94 |
| ASTER-CR (Linear) | 99.48 | 0.42 | 0.000 | 39.63 | 5.61 |
| ASTER-CR (MLP, Balanced) | 96.94 | 1.33 | 0.009 | 37.29 | 4.42 |
| ASTER-CR (MLP, Strong Protection) | 98.97 | 0.71 | 0.001 | 40.97 | 5.06 |

**Table 11:** MedEinst五模型模拟概览。每种子5,383个control/trap对；条件方向退化与Pair Acc.分别评价响应和双端正确性；BTR以control正确对为分母。预设模拟，未运行对应模型与数据集，待真实评测替换。


**MedGUIDE。**

**预设模拟，未运行对应模型与数据集，待真实评测替换。**

| Method | ER (%) ≈100 | Dir. Deg. (%) ↓ | Acc. (%) ↑ | Disease Macro Acc. (%) ↑ |
|---|---:|---:|---:|---:|
| Base Model | — | — | 80.18 | 80.28 |
| Bias Calibration | — | — | 80.18 | 80.28 |
| ASTER-CR w/o Evidence Constraints | — | — | 81.23 | 81.30 |
| ASTER-CR (Linear) | — | — | 80.65 | 80.82 |
| ASTER-CR (MLP, Balanced) | — | — | 80.00 | 80.14 |
| ASTER-CR (MLP, Strong Protection) | — | — | 81.67 | 81.84 |

**Table 12:** MedGUIDE五模型模拟概览。每种子7,747题、17个疾病组；疾病组等权宏平均。无有效干预配对的忠实性指标填“—”。预设模拟，未运行对应模型与数据集，待真实评测替换。


**TrialGPT。**

**预设模拟，未运行对应模型与数据集，待真实评测替换。**

| Method | ER (%) ≈100 | Dir. Deg. (%) ↓ | Balanced Acc. (%) ↑ | Macro-F1 (%) ↑ |
|---|---:|---:|---:|---:|
| Base Model | — | — | 88.54 | 88.97 |
| Bias Calibration | — | — | 89.32 | 88.48 |
| ASTER-CR w/o Evidence Constraints | — | — | 90.85 | 90.10 |
| ASTER-CR (Linear) | — | — | 89.53 | 88.72 |
| ASTER-CR (MLP, Balanced) | — | — | 89.93 | 89.07 |
| ASTER-CR (MLP, Strong Protection) | — | — | 90.19 | 89.41 |

**Table 13:** TrialGPT五模型模拟概览。1,000条生成标签中排除250未知，750确定标签用于Balanced Acc.与Macro-F1；无干预配对指标填“—”。预设模拟，未运行对应模型与数据集，待真实评测替换。


多标签保留答案集合计分，MedPAIR删句不赋未经核验的标签；MedGUIDE不输入金标路径，TrialGPT未知排除。无干预配对项留空。参数不据结果调排名，波动仅为模拟生成波动。


## 5 Analysis

### 5.1 Ablation and Nonlinear Comparisons

开发集加约束后证据修正RMS 3.0869→0.2029，ER94.81%→99.82%，VR40.71%→4.58%。16维MLP提高VR但带来更大留出证据改动；函数能力与修正位置分别比较。真实合成机制实验在附录D.1，预设模拟分别解释。


### 5.2 Faithfulness and Error Analysis

方向退化补充平均ER：原响应[1,1]变为[3,−1]时ER仍为100%，但方向退化为1/2。零退化也需结合修正RMS和正确率。截距改变分类阈值，不改变配对差分。开发集36条错误路径中34条首次错误分数绝对值超过5，19条集中于PMA时间条件。NLI监督残差在原错池修复207/1,469，错对ER降为40.10%；仅截距对照修复213/214例，保持各自响应。


## 6 Conclusion

已有证据支持部分干扰抑制与受控证据改动，尚未确立总体正确率优势。新增医学模拟用于明确评价表和对照，不作为跨任务实测优势。本次仅服务器CPU生成模拟、改文档，没有占用GPU或重跑MedPIC。
