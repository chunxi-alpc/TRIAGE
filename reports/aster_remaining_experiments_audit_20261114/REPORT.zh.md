# ASTER-CR 剩余实验服务器盘点与证据核验

审计日期：2026-11-14。本文保存本次只读审计结论。

- 主线参考提交：`a783bc6330512028fd167868aff0f85666e747f8`。
- 本轮归档分支：`codex/aster-e5-shared-results-20261114`。
- 审计时归档提交：`b67f7367f85c1ade64c69742a80d81eb73d71ec3`。
- 关联 [PR #69](https://github.com/chunxi-alpc/TRIAGE/pull/69)；新增五种外部方法所属的 [PR #65](https://github.com/chunxi-alpc/TRIAGE/pull/65) 已合并。
- 当前方法协议：E1 / Dataset A，2,540对，A/P/C/F各635对。
- 服务器：`gcx@10.29.128.115`；实验环境：`gcx261014`。

**E5–E8 无需重新训练已有模块。** 确认缺少六个 E6 评价单元；E5 按主线统一评价要求，还缺13行干扰路径前向、CC的CPU补算，以及11行新增方法的配对差值区间。E7全部目标总体已有前向产物，其中Qwen的NLI5,500和MedPIC467逐例结果尚未完整归档。

本次由四个智能体分工完成，只读取文件、核对哈希、统计已有预测和检查进程。没有启动模型推理、训练、下载或调度，没有修改仓库、结果和现有进程。本文的GitHub归档由用户在审计完成后另行授权。

## 1. 剩余实验总表

以下路径简称均指服务器目录；简称加相对路径即为准确路径。

| 简称 | 绝对路径 |
|---|---|
| `R` | `/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112` |
| `Q` | `/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111` |
| `F` | `/home/gcx/aster-cr-balanced-publish-20261112/reports/aster_cr_final_method_10006_20261112` |
| `E5` | `/home/gcx/aster-e5-gaps-20261112/reports/aster_e5_gaps_20261112` |
| `EXT` | `/home/gcx/aster-codex03-external-20261112/reports/aster_codex03_external_20261112` |
| `D` | `/home/gcx/aster-cr-diagnostics-e3cv-20261111/reports/aster_cr_m3_intercept_diagnostics_20261112` |
| `PUB` | `/home/gcx/aster-results-publish-20261114` |

优先搜索了用户指定的七个位置，以及相关历史工作树、结果归档、缓存和模型目录。`/home/gcx/aster-cr-final-method-10006-20261112/` 根目录未找到，对应报告和产物实际保存在上述 `F` 等工作树报告目录中。

### 1.1 状态口径

| 状态 | 含义 |
|---|---|
| COMPLETE_CURRENT | 当前E1协议下完整完成 |
| PARTIAL | 部分完成，有可复用产物 |
| HISTORICAL_ONLY | 只有历史协议结果 |
| NOT_FOUND | 服务器和仓库均未找到结果 |
| NEEDS_VALIDATION | 文件存在，但协议或结果有效性尚未完全核实 |
| OPTIONAL | 不是当前论文主线必需实验 |

状态按完整证据判定。计算和预测已核实、但上游模型修订或运行时代码记录仍有缺项的单元，列为 `NEEDS_VALIDATION`，后续只补追溯记录。存在明确评价缺项的单元列为 `PARTIAL`。

### 1.2 实验单元

| 实验编号 | 模型 | 任务 | 当前状态 | 服务器证据路径 | 缺失内容 | 是否必须补跑 |
|---|---|---|---|---|---|---|
| E6 | 五个IT | E1拟合、开发评价、冻结 | NEEDS_VALIDATION | `Q/models/E1/full/`；`R/models/{llama31,gemma2,gemma3,medgemma}/` | 版本追溯记录；计算产物齐全 | 否 |
| E6 | Qwen IT | 受控806路径 | NEEDS_VALIDATION | `Q/evaluation/test/` | 版本追溯记录 | 否 |
| E6 | Qwen IT | 官方7,458路径 | HISTORICAL_ONLY | 当前参数在`Q`；历史结果在`/home/gcx/aster_next_20261109/primary_results/mega_official_test/` | 当前E1官方预测、汇总、418来源区间 | 补前向和统计 |
| E6 | Llama | 受控806及官方7,458 | NEEDS_VALIDATION | `R/models/llama31/evaluation/{test,mega_official}/` | 版本追溯记录 | 否 |
| E6 | Gemma2 IT | 受控806路径 | NEEDS_VALIDATION | `R/models/gemma2/evaluation/test/` | 版本追溯记录 | 否 |
| E6 | Gemma2 IT | 官方7,458路径 | PARTIAL | `R/models/gemma2/evaluation/mega_official/shard0/` | 已有2,379；剩5,079路径及完整统计 | 接续前向和统计 |
| E6 | Gemma3 | 受控806路径 | PARTIAL | `R/models/gemma3/` | 测试前向、逐路径结果、区间 | 补前向和统计 |
| E6 | Gemma3 | 官方7,458路径 | HISTORICAL_ONLY | 当前参数在`R/models/gemma3/`；历史结果在`/home/gcx/aster_cross_model_mega_20261111/gemma3/` | 当前E1官方评价 | 补前向和统计 |
| E6 | MedGemma | 受控806路径 | PARTIAL | `R/models/medgemma/` | 测试前向、逐路径结果、区间 | 补前向和统计 |
| E6 | MedGemma | 官方7,458路径 | HISTORICAL_ONLY | 当前参数在`R/models/medgemma/`；历史结果在`/home/gcx/aster_cross_model_mega_20261111/medgemma/` | 当前E1官方评价 | 补前向和统计 |
| E7 | Qwen | NLI500 | PARTIAL | `Q/evaluation/nli4ct/` | 独立截距对照及区间 | CPU即可 |
| E7 | Qwen | NLI5,500 | PARTIAL | `F/evaluation/nli4ct_official5500/` | 独立截距对照、全量统计整理、逐例归档 | CPU和归档 |
| E7 | Qwen | MedPIC188 | NEEDS_VALIDATION | `Q/evaluation/medpic/`；`D/predictions/medpic_test/` | 版本追溯记录；三种方法已齐 | 否 |
| E7 | Qwen | MedPIC467 | PARTIAL | `F/evaluation/medpic_official467/` | 独立截距对照及区间、逐例归档 | CPU和归档 |
| E7 | Llama | 四个目标总体 | NEEDS_VALIDATION | `R/models/llama31/evaluation/` | 版本追溯及NLI聚类口径说明 | 否 |
| E7 | Gemma2 IT | 四个目标总体 | NEEDS_VALIDATION | `R/models/gemma2/evaluation/` | 同上 | 否 |
| E7 | Gemma3 | 四个目标总体 | NEEDS_VALIDATION | `R/models/gemma3/evaluation/` | 同上 | 否 |
| E7 | MedGemma | 四个目标总体 | NEEDS_VALIDATION | `R/models/medgemma/evaluation/` | 同上 | 否 |
| E8 | Qwen Base/IT | 头集合、子空间、端点比较 | NEEDS_VALIDATION | `R/base_it/qwen/comparison.json` | IT上游修订记录 | 否 |
| E8 | Gemma2 Base/IT | 同上 | NEEDS_VALIDATION | `R/base_it/gemma2/comparison.json` | 两侧上游修订记录 | 否 |
| E5 | Qwen | 温度、截距、仿射、监督线性读出 | NEEDS_VALIDATION | `E5/output_controls_e1/` | 版本追溯；现有评价完整 | 否 |
| E5 | Qwen | LoRA三个种子 | PARTIAL | `E5/native_supervised_e1/lora/lr0.001_seed{11,22,33}/` | 干扰路径及区间 | 补前向和统计 |
| E5 | Qwen | LoFiT三个种子 | PARTIAL | `E5/native_supervised_e1/lofit/lr0.0001_seed{11,22,33}/` | 同上 | 补前向和统计 |
| E5 | Qwen | LoReFT三个种子 | PARTIAL | `E5/loreft_e1/lr0.0001_seed{11,22,33}/` | 同上 | 补前向和统计 |
| E5 | Qwen | LEACE、ITI、SEA | PARTIAL | `E5/representation_e1_plan/{leace_pair,iti_style,sea_adaptation}/` | 干扰路径；SEA还缺对应残差缓存 | 补前向和统计 |
| E5 | Qwen | 强证据提示 | PARTIAL | `E5/prompt_controls_e1/strong_evidence_prompt/` | 干扰路径及区间 | 补前向和统计 |
| E5 | Qwen | CC | PARTIAL | `E5/prompt_controls_e1/CC/` | 干扰路径汇总及区间 | CPU即可 |
| E5 | Qwen | RePS、PSR、CAA、INLP、DoRA，共11行 | PARTIAL | `EXT/evaluation/test/` | 自然／干扰路径相对raw的配对差值区间 | CPU即可 |
| E5 | Qwen | 原模型、固定缩放、冻结E1参照 | NEEDS_VALIDATION | `E5/e1_reference/`、`E5/output_controls_e1/`、`Q/evaluation/test/` | 版本追溯；已有预测和指标可复用 | 否 |
| E10 | Qwen | 线性收益拆分、截距与合法集合解码 | NEEDS_VALIDATION | `D/` | 版本追溯；评价已完成 | 否 |
| E10 | 相关历史模型 | 当前E1非线性／其他动态系数 | HISTORICAL_ONLY | `PUB/reports/aster_cancel_story_20261109/meeting/trace_validation/`等 | 当前E1正式比较 | 单列扩展任务 |
| 数据审核 | — | 150条人工抽查 | PARTIAL | `/home/gcx/aster_cr_pair_dataset_20261111/human_audit.csv` | 实际人工审核0/150 | 人工工作 |
| 数据审核 | — | MedEinst剩余3,768对语义审核 | NOT_FOUND | `/home/gcx/aster-cr-medeinst-alignment-audit-private-20261112/`及对应公开报告 | 未发现新增审核 | 当前MEGA-only主线不必补 |
| 独立验证 | — | 全新、未使用来源确认集 | NOT_FOUND | 当前实验与来源使用记录 | 独立确认结果 | 先确定数据与协议 |
| 阶段分析 | Qwen、Gemma2 | SFT/RL中间检查点 | NOT_FOUND | 模型目录、E678检查点及历史工作树 | 中间检查点和阶段比较 | 单独设计 |
| 可选基线 | — | R-LACE、LEOPARD、CAST、FourierFT等 | OPTIONAL | 基线候选清单 | 当前正式结果 | 不列入必需补跑 |

### 1.3 E6核验依据

五个IT实际均有完整、可读的 `m1.json / m2.npz / m3.json / m3.npz / model.json`，数值数组有限。参数为：

- Dataset A：2,540对，A/P/C/F各635。
- 51头、16维、λ=0.01、δ=0.2923349。
- M3使用1,270个A/P对、1,270个C/F对，校准节点761，求解成功。
- 五模型开发集均完成420条路径、20来源评价。

| 模型 | 参数目录 | 冻结时间UTC | 冻结记录 |
|---|---|---|---|
| Qwen3-8B | `Q/models/E1/full/` | 2026-11-11 09:34:54 | `Q/frozen.json` |
| Llama-3.1-8B-Instruct | `R/models/llama31/models/E1/full/` | 2026-11-13 04:27:43 | `R/models/llama31/frozen.json` |
| Gemma-2-9B-IT | `R/models/gemma2/models/E1/full/` | 2026-11-13 06:13:26 | `R/models/gemma2/frozen.json` |
| Gemma-3-27B-IT | `R/models/gemma3/models/E1/full/` | 2026-11-13 19:08:49 | `R/models/gemma3/frozen.json` |
| MedGemma-27B-Text-IT | `R/models/medgemma/models/E1/full/` | 2026-11-13 19:44:50 | `R/models/medgemma/frozen.json` |

Gemma3、MedGemma旧 `run_status.json` 中的 stopped 未反映后续开发评价、冻结和迁移完成状态。

逐例复算结果：

| 当前E1评价 | 原模型正确数 | ASTER正确数 | 分母／来源 |
|---|---:|---:|---|
| Qwen受控 | 768 | 768 | 806／40 |
| Llama受控 | 629 | 672 | 806／40 |
| Gemma2受控 | 761 | 763 | 806／40 |
| Llama官方 | 5,689 | 5,899 | 7,458／418 |
| Gemma2官方已完成部分 | 2,160 | 2,167 | 2,379／115 |

三份完整受控评价无重复case或路径错误，逐例统计与汇总一致，40来源、2,000次bootstrap区间存在。

Llama官方独立截距为5,898/7,458；三种方法各7,458个唯一case、418来源，合计22,374条路径预测。按预测路径与金标准路径重新比较，正确数和已有汇总一致；45,627个自然节点的正确数为43,061／43,499／43,485。来源聚类区间按既有代码和随机种子复算一致。

Gemma2官方实际保存三种方法各2,379条路径，0失败；缺5,079条，完整418来源统计尚未完成。

### 1.4 E7核验依据

下表均为已有逐例复算的“原模型／独立截距／ASTER”正确数：

| 模型 | NLI500 | NLI5,500 | MedPIC188 | MedPIC467 |
|---|---:|---:|---:|---:|
| Qwen | 378／待补／375 | 4,073／待补／4,043 | 70／66／66 | 188／待补／183 |
| Llama | 347／342／343 | 3,883／3,941／3,923 | 59／63／63 | 144／149／147 |
| Gemma2 | 359／368／366 | 4,101／4,122／4,109 | 24／23／25 | 78／70／75 |
| Gemma3 | 374／382／381 | 4,280／4,246／4,250 | 61／46／48 | 157／120／124 |
| MedGemma | 364／366／366 | 4,171／4,172／4,176 | 74／77／74 | 199／201／197 |

四个非Qwen模型的独立截距均来自761个MEGA拟合节点，记录明确为 `target_labels_used=false`，逐例截距分数等于raw+b0。16个单元的80份预测及指标文件与PR69归档逐字节一致。

全部20个单元的视图无重复。NLI500包含原声明及扰动共2,864视图；NLI5,500保存5,500条声明；MedPIC188保存2,820选项视图。四个非Qwen模型的MedPIC467保存2,335选项，Qwen包含扰动共7,005视图，其原始选项同为2,335。准确率、配对指标、RMS、VR和ER可以从已有逐例复算，最大浮点差约1.5e-14。

需明确评价口径：

- NLI5,500目录中部分 `natural_*accuracy` 字段统计的是500条原声明；全量5,500正确数以上表为准。
- NLI500现有区间按113个临床试验或试验组合聚类；NLI5,500按500条原始声明及其变体聚类。论文采用哪一级来源，应先确认。
- MedPIC188和467分别按150、376来源聚类。
- **MedPIC467包含开发题，作为全题描述性结果。**
- 现有区间采用2,000次来源聚类抽样及2.5%／97.5%分位。

### 1.5 E8核验依据

| 比较 | Base／IT头数 | 交集 | 并集 | Jaccard | 状态说明 |
|---|---:|---:|---:|---:|---|
| Qwen | 51／51 | 20 | 82 | 0.2439024390 | 拟合、冻结和比较完成；IT上游修订待补 |
| Gemma2 | 51／51 | 36 | 66 | 0.5454545455 | 拟合、冻结和比较完成；两侧上游修订待补 |

谱值与各自 `m2.npz` 一致。保留16个分量的有效秩按S／U／N／投影特征排列：

| 端点 | 有效秩 |
|---|---|
| Qwen Base | 11.653474／11.951567／11.807422／11.816189 |
| Qwen IT | 2.625890／8.905511／14.757180／11.337412 |
| Gemma2 Base | 3.074953／7.901753／14.722417／8.076157 |
| Gemma2 IT | 2.676147／8.771149／14.876226／9.876739 |

头集合、Jaccard和有效秩已独立复核。比较文件及模型目录下的 `source_bootstrap.csv / source_metrics.csv` 已存在，两份比较文件均与PR69相同。

| 文件 | 字节 | 修改时间UTC | SHA256 |
|---|---:|---|---|
| `R/base_it/qwen/comparison.json` | 187,972 | 2026-11-14 01:57:13.362896 | `06753eaf4fe14c296c6ca2642cdec1c34105d5418d556688908904f863b22f61` |
| `R/base_it/gemma2/comparison.json` | 67,330 | 2026-11-13 16:52:13.676131 | `dc28b6703e3c8b92ac764be2a70320f68e4b4d6a86f2b1d62524eb7465b88de9` |

Qwen Base修订确认是 `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`。Qwen Base五分片与下载清单一致，IT五分片与历史权重哈希清单一致；Gemma2两侧全部分片亦已实际计算SHA256。现有结果对应Base/IT端点比较。未发现相应SFT/RL中间检查点。

### 1.6 E5核验依据

当前12种方法形成18行新增结果，另有原模型、固定0.95缩放和冻结ASTER-CR三个参照；RePS、PSR-MSE、CAA、INLP、DoRA五种新增方法另有11行正式结果。

LoRA、LoFiT、LoReFT均保存种子11、22、33。训练清单为4,637个去重有标签端点；LoRA冻结学习率0.001，LoFiT和LoReFT冻结0.0001，选择记录明确未使用测试集选择。

| 方法 | 自然路径正确数/806 | 干扰路径正确数/2,418 |
|---|---:|---:|
| 原模型 | 768 | 2,339 |
| 温度缩放 | 768 | 2,339 |
| E1端点截距 | 763 | 2,317 |
| 仿射校准 | 769 | 2,329 |
| 监督线性读出 | 766 | 2,312 |
| LoRA seed11/22/33 | 756／774／737 | 未保存 |
| LoFiT seed11/22/33 | 768／769／769 | 未保存 |
| LoReFT seed11/22/33 | 764／772／650 | 未保存 |
| LEACE-pair | 767 | 未保存 |
| ITI-style | 321 | 未保存 |
| SEA adaptation | 768 | 未保存 |
| CC | 769 | 可由现存raw缓存计算 |
| 强证据提示 | 772 | 未保存 |
| CAA | 766 | 2,332 |
| INLP | 769 | 2,339 |
| DoRA seed11/22/33 | 786／781／783 | 2,351／2,334／2,345 |
| RePS seed11/22/33 | 474／609／60 | 1,372／1,793／191 |
| PSR-MSE seed11/22/33 | 776／768／777 | 2,343／2,339／2,343 |
| 冻结ASTER-CR E1 | 768 | 2,333 |

14个E5自然路径单元各保存3,100输入、2,339配对、806唯一自然路径及40来源，错误数为0。配对类别为A=2,283、P=10、C=33、F=7、F0=6；F0没有混入40个C/F对。AP RMS、CF修正RMS和路径正确数均与汇总一致。

四个输出控制具有完整自然和干扰路径。EXT全部11单元亦有完整自然和干扰路径、21项指标自身的来源聚类95%区间；相对raw的配对路径差值区间待CPU补齐。

## 2. 已完成但尚未上传的结果

### 2.1 Qwen当前E1全量迁移逐例

**确认存在Qwen当前E1的NLI5,500和MedPIC467逐例漏归档。** 其E1参数与原始2,540对E1参数逐文件哈希相同。指定分支及两个旧ZIP均未包含这些逐例文件；已有总表和部分统计可以继续复用。

下表路径相对于 **`F/evaluation/`**，时间均为UTC，大小为字节。

| 文件 | 大小 | 修改时间 | SHA256 |
|---|---:|---|---|
| `nli4ct_official5500/predictions.jsonl` | 2,783,921 | 2026-11-12 01:27:31.879664 | `7dad20fec338410484aa8fba90845788eadb22ed72ab6112a72fc8f822fc3ed8` |
| `nli4ct_official5500/pair_predictions.jsonl` | 24,653,771 | 2026-11-12 01:27:31.681665 | `06c4829da19ab146230a48eff32f16f438ae779db52b6058a446990c4f397bf5` |
| `nli4ct_official5500/metrics.json` | 105,160 | 2026-11-12 01:27:32.058664 | `580ad399a528379e2af30c701f133ada46f2c49710ff59ebd5b829d658cb8880` |
| `medpic_official467/predictions.jsonl` | 3,052,163 | 2026-11-11 18:04:32.890543 | `d9c6b38db2a8d5fb5217bd47877d992d2f24b983947d65130fc8b4aa25cf909d` |
| `medpic_official467/pair_predictions.jsonl` | 19,204,775 | 2026-11-11 18:04:32.760542 | `779cee5d663c0c4d2a82e0ef4f62f1425a8b21485cc31b22ccf032dfa582baee` |
| `medpic_official467/question_predictions.jsonl` | 1,478,527 | 2026-11-11 18:04:32.794542 | `d62782692e246eb4b6b9678ff6db423ea91ca5ada266e36a11c953dd7d710a9c` |
| `medpic_official467/metrics.json` | 145,291 | 2026-11-11 18:04:33.041544 | `c0f966787aa6e6d3b643516f6c14783c9dd2b69e2d5ff7b591b4599d1700cc69` |

这些文件可直接归档；缺少的独立截距对照可以通过既有raw分数和冻结源任务截距完成CPU补算。

已核查：

- `results/aster_cr_transfer_20261110.zip`：542成员，最新成员时间2026-11-10 03:12:04。
- `results/aster_cr_research_20261109.zip`：779成员，最新成员时间2026-11-10 03:11:10。
- 两包均未包含上述当前E1完整逐例文件；其中NLI属于历史拟合或历史读出。

### 2.2 E5已完成参数与服务器缓存

**E5完整预测已上传。** 核对52份E5预测及33份新增方法预测，均与归档解压后的内容一致。以下已完成参数仍留在服务器，适合补充参数归档：

| 文件 | 大小 | 修改时间UTC | SHA256 |
|---|---:|---|---|
| `E5/representation_e1_plan/sea_adaptation/operator.pt` | 134,219,629 | 2026-11-12 19:18:39.599549 | `dc423b9d09b9587ad802742063e0071f46e905122312df4d801d95a6d456cb91` |
| `EXT/dora/full/seed11/adapter/adapter_model.safetensors` | 16,101,768 | 2026-11-12 16:13:09.259030 | `bbb12c364a25edc9649011b27ee2246826da5476ba03eab3eff7a3eb30923198` |
| `EXT/dora/full/seed22/adapter/adapter_model.safetensors` | 16,101,768 | 2026-11-12 19:20:10.299568 | `ea89e8a8e163e9440a59037796ba837f9c6ad73be6702d1371b8ffbe8503c392` |
| `EXT/dora/full/seed33/adapter/adapter_model.safetensors` | 16,101,768 | 2026-11-12 22:27:07.228827 | `f3be46a4b306236d4b5b463502412d879d6547ffebd644c284ea75067b7ed513` |
| `EXT/inlp/iterations_1/projection.npz` | 62,591,408 | 2026-11-12 15:48:33.261108 | `3e507afdbf930abf2516422916823f431049ced219bc1d9a400d3e12be92125e` |

另有未选中的INLP候选及CAA中间激活留存服务器，无需为归档重新运行：

| 文件 | 大小 | 修改时间UTC | SHA256 |
|---|---:|---|---|
| `EXT/inlp/iterations_4/projection.npz` | 62,542,029 | 2026-11-12 15:48:53.997380 | `34f333fe5d29acade829141e456515f1dfde071c55f26fda1793810e4deb186e` |
| `EXT/inlp/iterations_8/projection.npz` | 62,643,298 | 2026-11-12 15:49:04.648012 | `7fca17b70d01d4b20f05a756b8e70d53db84616161d0094f0951366317dd4224` |
| `EXT/caa/answer_activations/layer18_pos.pt` | 75,974,213 | 2026-11-12 15:45:52.451666 | `8405f2be76c8d7c74bbb4a61040b0526919303905d7cc7818e63440286f98540` |
| `EXT/caa/answer_activations/layer18_neg.pt` | 75,974,213 | 2026-11-12 15:45:52.512666 | `63f2b198b15def168e3a671692dffefd23122a75ad3ef68e6190e2672e05037f` |

### 2.3 E6部分结果与已上传核验

Gemma2官方评价的部分结果应保留接续：

- 文件：`R/models/gemma2/evaluation/mega_official/shard0/path_predictions.jsonl`。
- 大小：5,368,962字节。
- 修改时间：2026-11-13 16:46:51.667414 UTC。
- SHA256：`8c11526a7d3a7dece08bcb3bbf611d467ec1512e8bb80758b3da0420676cac2a`。
- 内容：三方法各2,379路径，共7,137行。

E6未发现漏传的完整当前协议评价。Llama、Gemma2受控预测与PR69对应gzip解压内容一致；Gemma3和MedGemma的M1/M2/M3及model文件与PR69 Git blobs逐字节一致。

Qwen IT历史官方全量已上传至 `results/aster_cr_research_20261109.zip`，包内 `reports/aster_cancel_story_20261109/meeting/primary_full/mega_official_test/path_results_shard{0..3}.jsonl.gz` 与服务器四片的全部 `(case_id, method, correct)` 一致，共44,748行，即7,458路径乘六种旧方法。

## 3. 历史结果与当前E1的协议差异

| 历史产物 | 实际协议／问题 | 当前可复用部分 |
|---|---|---|
| 五模型旧MEGA | A=4,869、P=18；C=66、F=69；合计5,022对 | 原始模型分数、兼容缓存、历史对照 |
| 旧M1/M2/M3 | 头集合、预算不同；MedGemma旧109头，当前51头 | 按头集合和缓存字段逐项复用 |
| 跨模型旧官方结果 | 每模型文件混合420开发、806受控、7,458官方路径，共8,684路径／478来源 | 按`evaluation_group=official_test`筛出历史官方结果 |
| Qwen旧官方7,458 | 六种旧读出，包括`old_linear`、`optimized_*`；已在旧ZIP上传 | 历史官方比较 |
| `qwen_mega_parallel_20261114` | 实际模型为Qwen3-8B-Base | E8的Base官方结果 |
| 旧五模型NLI迁移 | 部分使用目标任务P480/C270及160自然拟合节点 | 目标任务拟合历史结果 |
| 10,006对混合来源实验 | MedEinst参与训练，训练来源与当前E1不同 | 混合来源补充实验；其中独立保存的E1对照可复用 |
| 旧非线性探针、动态融合 | 旧数据、差分预测或旧融合协议 | 历史机制／方法探索 |
| 旧LoReFT | 旧旋转矩阵范数为0，适配器失效 | 故障与修复记录 |
| 806路径受控评价 | 40来源已有使用记录 | 当前E1受控复评 |
| MedPIC467 | 包含开发题 | 全题描述性评价 |

历史官方ASTER正确数为Llama 5,879、Gemma2 6,893、Gemma3 7,078、MedGemma 7,125。当前Llama E1为5,899，应保持协议分列。

两个关键例外：

- `F/models/E1`实际指向原始E1模型，五个参数文件与`Q/models/E1`哈希一致。**目录中的E1对照可复用。**
- 当前E1的A/P损失权重归一合计4,887；实际训练样本仍为2,540对，应以数据清单、类别计数和哈希判定。

E678五模型使用的规范化配对文件SHA256一致：

`1ae3fc5dd11b58c9e8a4913c79f7768516013486c99291a87b9f28be0f113291`

E5完整原始配对文件为：

`/home/gcx/aster_cr_pair_dataset_20261111/balanced_train.jsonl`

大小77,681,550字节，SHA256：

`f0995a8978a590a5ebe8c8331e597fded5d16d25b5ccddb576a85d5f42c65c37`

EXT的 `prepared/E1_pairs_full.jsonl` 与该原始文件SHA256一致。规范化文件和完整原始文件表示不同，核查的样本数、类别和来源符合当前E1。

### 3.1 需要先确认的两项协议差异

1. PR69的18行交付覆盖自然路径；[主线实验协议第9节](https://github.com/chunxi-alpc/TRIAGE/blob/a783bc6330512028fd167868aff0f85666e747f8/docs/ASTER/experiments.zh.md)同时要求干扰路径及相对基线的配对差值区间。13行补前向的必要性按这一主线要求登记。
2. NLI500与NLI5,500现有来源聚类单位不同。后续统一统计前，应明确采用临床试验来源还是原始声明来源。

本次保留差异并报告，未自行变更协议。

### 3.2 代码与版本证据

当前服务器E678的核心代码及E5、EXT发布脚本已与PR69文件逐字节核对。相关工作树HEAD：

| 工作树 | HEAD |
|---|---|
| E678及E5 | `5b8f21e4cd535c0608856dd208a308d799dc510c` |
| Qwen retrain | `d37487a2fe2cbacec33b4737c97ef1d62e81976f` |
| EXT | `742dcb2598631ac20c1ae2f72c85a2dff20cb16b` |

这些是审计时工作树HEAD；运行时提交未逐一封存在冻结记录。E678的模型元数据记录架构、路径、精度、模板和评分方式，但没有完整上游revision；EXT存在明确的 `model_revision=null`。应补齐追溯记录，保留已验证的计算产物。

## 4. 真正需要新运行的实验

| 类型 | 已确认缺项 | 最小补做范围 |
|---|---|---|
| 新训练 | E5–E8无确认缺项 | **0个训练任务** |
| E6补充前向 | 六个评价单元 | Qwen IT官方7,458；Gemma2官方剩5,079；Gemma3和MedGemma各受控806、官方7,458 |
| E5补充前向 | 13行干扰路径 | LoRA/LoFiT/LoReFT各三种子，加LEACE、ITI、SEA、强提示；复用冻结参数 |
| CPU统计 | CC | 用现有raw分数和冻结offset补2,418条干扰路径及区间 |
| CPU统计 | 新增五方法11行 | 补自然／干扰路径相对raw的配对来源区间 |
| CPU统计 | Qwen三项迁移 | NLI500、NLI5,500、MedPIC467的独立源任务截距及区间 |
| 纯文档整理 | Qwen完整迁移、参数、版本 | 归档第2节文件，补权重来源、运行版本和分母说明 |

缓存核验进一步限定了前向规模：

- E6六单元合计涉及29,065条尚缺完整结果的路径。Gemma2已有2,379条保留。
- Gemma3、MedGemma受控测试各有3,100条全头缓存可复用；当前评价输入清单各有63,652条尚无对应缓存。
- Qwen、Gemma3、MedGemma的旧官方缓存分别缺当前选中头中的9、11、2个头，不能直接完成当前E1复算。
- E5十三行现有分数各只能完成42–45条干扰路径，合计30,855条“方法×路径”仍会遇到未知节点。应沿实际访问路径补分数。
- 每行干扰范围有50,022个候选UID；现存修正分数覆盖2,283个，47,739个候选UID未评分。完整候选数不直接作为必跑规模。
- SEA缺失节点的E1缓存有heads但缺最后残差，需复用算子并补对应前向或残差。
- CC冻结offset为-1.0324354956866095，现有7,330个分数均与raw+offset一致；raw缓存覆盖全部候选节点，CPU遍历可覆盖2,418路径，**无需新前向**。

E10当前E1非线性及其他动态系数属于新增方法比较，应单独排期，不混入已有模块接续任务。

## 5. 不需要再运行的实验

- 五个IT及两个Base已经完成的E1拟合、M1/M2/M3、开发评价和冻结。
- Qwen、Llama、Gemma2的806路径受控评价。
- Llama、Qwen Base、Gemma2 Base的完整官方MEGA评价。
- E7五模型的四个目标总体前向；Qwen缺项可以CPU补齐。
- 两组Base/IT头集合、Jaccard、子空间谱和有效秩比较。
- E5全部已有训练、18行自然路径结果、五种新增方法11行的自然及干扰路径前向。
- E0–E3已有拟合和来源五折；E3新增五折位于`/home/gcx/aster-cr-diagnostics-e3cv-20261111/`。
- E9已有1,704条干预记录；E11已有八方法、320次测量。
- E1截距拆分和MedPIC合法集合解码诊断。

### 5.1 LoReFT与失败登记

**LoReFT修复已生效。** 三种子旋转参数更新L2分别为0.457101、0.463157、0.468976；实际保存的旋转矩阵非零、有限、范数约1，正式适配器和预测均存在真实变化。三个选中adapter与已发布文件一致。

旧适配器旋转矩阵范数为0。旧 `gpu_effectiveness.json` 和 `gpu_effectiveness_v2.json` 为失败测试；最终 `gpu_effectiveness_final.json` 通过。未发现旧失败测试混入当前18行。

PSR seed22参数确有更新，保存输出与raw一致；该结果应保留，不列为未运行。

### 5.2 截距与MedPIC合法集合解码

MedPIC188合法集合诊断已有八种方法、150来源的2,000次聚类区间。逐例复算：

| 方法 | 整题集合正确数/188 |
|---|---:|
| raw | 70 |
| pair_only | 70 |
| original_intercept_only | 66 |
| independent_intercept | 66 |
| full E1 | 66 |
| C_mean_full | 74 |
| D_none_calibrated | 65 |
| E_size_calibrated | 66 |

开发集选定的解码为 `D_none_calibrated`，冻结记录明确在测试评价之前确定。该联合评价已完成，不能按旧索引再次列为待运行。

证据目录为 `D/predictions/medpic_test/`。已发布的 `PUB/reports/aster_cr_m3_intercept_diagnostics_20261112/predictions/medpic_test/questions.jsonl.gz` 含4,512行，即八方法乘188题乘三个版本，原始版本每方法188个唯一题目，预测集合与正确标记一致。

## 6. 当前科学证据缺口

| 证据问题 | 核验结论 | 后续职责 |
|---|---|---|
| 临床专家审核 | 150条均为`PENDING`，reviewer与review_date全空，实际0/150 | 完成人工审核并保存记录 |
| MedEinst新增审核 | 公开600条语义审核与私有三组各200条相符；剩3,768对无新增记录 | MEGA-only主线无需等待该扩展 |
| 独立确认 | 现有40来源、60来源和官方集合均有既往使用记录 | 建立明确未用于选择的新来源评价 |
| 跨模型完整性 | E6尚缺六个当前协议评价单元 | 补齐同协议跨架构证据 |
| 强基线比较 | E5十三行缺干扰路径，十一行缺配对差值区间 | 完成主线成功标准要求的联合比较 |
| 输入相关修正贡献 | Llama官方ASTER为5,899，独立截距5,898，相对截距区间含零 | 校准贡献与额外修正贡献分别报告 |
| 后训练阶段归因 | 两组Base/IT端点比较已完成；对应SFT/RL中间检查点未发现 | 阶段因果分析另行设计 |
| 版本追溯 | 本地参数与发布代码可以核对；多模型上游revision及运行时提交未完整记录 | 补来源记录，保留已有计算 |

### 6.1 人工审核证据

`/home/gcx/aster_cr_pair_dataset_20261111/human_audit.csv`：

- 150行；reviewer及review_date均为空，human_status均为PENDING。
- 大小15,615字节，修改时间2026-11-11 04:44:28.284220 UTC。
- SHA256：`5d3a49aab9165f128d4b9b15126e70ed6fbe1a20835ee77b47c48a0c0e11bf88`。

多份工作树的 `human_audit_status.json` 内容一致：selected=150、actually_reviewed=0、clinical_expert_involved=false。

MedEinst公开 `semantic_review.jsonl` 实际600行，三位独立智能体各200条。私有目录包含每组20条pilot和180条main的对应结果。分类为S1=0、S2=7、S3=512、S4=81，临床专家审核数为0；未发现剩余3,768对的新增语义审核。

### 6.2 模型权重身份

Qwen当前权重目录：

- Base：`/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/models/Qwen3-8B-Base`。
- IT：`/home/gcx/TRIAGE/models/modelscope/Qwen3-8B`。

完整分片哈希记录：

- Base：`/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/qwen_base_download_manifest_v2.json`。
- IT：`/home/gcx/TRIAGE-worktrees/aster-dynamic-fusion-review/experiments/aster_action_mass_seed0_20261025/results/model_weight_hashes.json`。

Base清单尚未进入审计时的PR69树，大小4,556字节，修改时间2026-10-25 17:06:21.881469 UTC，SHA256：

`06f4f65124120a2a2130e697807616130c9e68005033f434cd104b010fdedb63`

IT权重哈希清单大小1,491字节，修改时间2026-10-25 12:08:47.705142 UTC，SHA256：

`64c8d23e03731c1252bf4362322911531fcf2e0f4fd4ca0e11e81ddccf3d4312`

Qwen Base五分片实际SHA256与其官方清单一致；Qwen IT五分片实际SHA256与历史清单一致。Qwen IT上游不可变revision未找到。

Gemma2权重根目录分别为 `R/checkpoints/gemma-2-9b` 和 `/home/gcx/models/gemma-2-9b-it`。以下为实际读取的分片SHA256；编号对应 `model-0000N-of-00008.safetensors` 或 `model-0000N-of-00004.safetensors`：

| 分片 | Gemma2 Base | Gemma2 IT |
|---|---|---|
| 1 | `72e486c145f30afdc43a507764a146fd545d26f166e60232c25d01a1510738c0` | `a0d4eb4fcbddfe01b0dc7a58386a3980826ededaf0ecd67d1ef38995b89615ba` |
| 2 | `73b2e0b94e18b25a2f7a3abda70ab350053be6c615599cb3772f27efb2ae21eb` | `712a589c333f49f7b6f3870838a9e0ab4b3c0260abd5130e8c54316e6c99640b` |
| 3 | `95310f8181c07b98a18facfbbbdaa7bcf28aafaddb9c33cd9dfa008706206970` | `5a9f84eb9846840802200c6502ccaaebf0a146ee89dcb8822fab795c392dd667` |
| 4 | `f4aa5f7d5510456fba5eb22421432a65487b1d0864fbd59ca6f8dfac65184d32` | `3349b4960bbc0bf7e47a624e5c520881adb06e2da4bda6ed14b51c7f647fe26c` |
| 5 | `96336ba3f200b986fa952df49998420a9ca1185853449b10a26eccc9bc7fc581` | — |
| 6 | `25d2f65b06a3ffc288fc1653554fa33b9d913f47f9b797be7785db19d3ed1f08` | — |
| 7 | `d0804ef42bdca870b9f5da75fe222448906bfccfddbebbbf61a5361b2f7a46b3` | — |
| 8 | `18e1cb9107e250c97ac91ad62bd950449f75de4a427092319ff368164d4617b2` | — |

Gemma2两侧上游不可变revision未找到。模型元数据补档与已完成的E1拟合、冻结和端点比较分开处理。

## 7. 最小必要实验计划

| 优先级 | 工作 | 科学价值／论文影响 | 计算成本 |
|---|---|---|---|
| 0 | 确认E5评价范围和NLI聚类单位；补版本说明 | 固定后续比较口径 | 文档工作 |
| 1 | 归档Qwen全量迁移；补CC、Qwen截距和11行基线差值区间 | 直接补齐已有证据 | CPU，成本低 |
| 2 | 完成150条临床专家审核；确定独立来源验证方案 | 加强数据与独立验证依据 | 人工为主 |
| 3 | 接续Gemma2官方5,079路径；补两27B受控测试 | 补齐跨模型主表 | GPU，中等 |
| 4 | 按确认后的范围补E5十三行干扰路径 | 支撑强基线联合比较 | GPU，较高 |
| 5 | 补Qwen IT、Gemma3、MedGemma官方MEGA | 完成五IT官方全量比较 | GPU，较高 |
| 6 | E10非线性、其他动态系数及SFT/RL阶段分析 | 扩展方法与机制证据 | 单独立项 |

本次审计止于查找、核验和汇总，未启动任何补实验。
