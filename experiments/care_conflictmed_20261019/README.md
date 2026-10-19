# CARE / ConflictMedQA：上下文—模型参数知识冲突实验（阶段一）

本目录是**新的、独立的实验协议**，不修改 `experiments/care_steering_20261019` 已冻结的结果。所用公开数据集为论文 *Assessing and Mitigating Medical Knowledge Drift and Conflicts in Large Language Models* 所发布的 `RDBH/DriftMed`：194 组正确/错误临床建议，另有 4,290 条含患者背景的对照场景。原数据集基于 **2025 年 2 月** 的临床知识，不能当作 2026 年实时医疗指南。

论文：https://aclanthology.org/2025.findings-emnlp.38/ ；数据：https://huggingface.co/datasets/RDBH/DriftMed 。数据卡片明确标记 **Research Use Only**，因此本仓库只保存构造代码和不含真实医学建议的测试样例，**不重新发布原始数据**。下载后保留原始数据文件及其 SHA256、运行环境和使用许可。

## 实验问题与明确边界

- **E0：真实文本冲突基准（建议对照轨）**：每组取数据集中的 `correct` / `wrong` 建议，构造 A/B、B/A 两种选项顺序，并分别置于 `none`、`excerpt_A` 和 `excerpt_B` 三种环境。后两者把其中一条建议重复呈现为**来源、日期均未核实的外部摘录**。固定问题是「哪一条与数据集所用 2025-02 指南快照一致」，gold 始终来自作者原标签，不从模型输出或附加摘录推导。**每组共六个样本**。这检验有来源干扰时的知识选择，而不是盲目服从上下文。
- **E1：患者场景基准**：4,290 条作者场景只用于无额外摘录的正确性判断（先测是否有足够的内部/推理能力）；按作者给出的正误配对 ID，严格共同分组。**此轨当前不是上下文干预测试**。原文场景可能含心理偏差注入，不能用场景对自动伪造互为等价的患者事实。
- **E2：参数冲突确认**：先要求无上下文同一建议在 A/B、B/A 两个顺序都正确，报告为 `closed_book_consistent`（**行为代理指标，不是参数存储的证据**）。再比较两个未核实的外部摘录方向是否改变同组选择。只有确证有预实验表现、上下文方向和配对标签，才对案例作冲突子集分析。
- **E3：CARE 条件干预**：当前 **尚未运行**；必须经独立标注的患者适用性最小对比（T/F/U）、确认证据片段、规则条件和临床动作 gold 后，才能调用模块一构造结构、离线模块二学角色子空间、模块三在 RoPE 前编辑 KV。**不要**把 E0、E1 的结果当成完整 CARE 的效果。

特别注意：ConflictMedQA 的真值是作者对 **2025-02** 建议正确性的标记；研究评分是符合历史标准，不等于当下临床最佳决策。数据集中正确建议和错误建议来自同一语境，但并非必然包含可逐条验证的患者反事实适用性，不得擅自给反事实病例填治疗金标准。

## 数据下载与准备

在具备 Hugging Face 网络访问的实验服务器执行：

```bash
mkdir -p local_data/conflictmed
curl -fL --retry 3 -o local_data/conflictmed/advice_pairs.jsonl \
  https://huggingface.co/datasets/RDBH/DriftMed/resolve/main/advice_pairs.jsonl
curl -fL --retry 3 -o local_data/conflictmed/ConflictMed.jsonl \
  https://huggingface.co/datasets/RDBH/DriftMed/resolve/main/ConflictMed.jsonl
sha256sum local_data/conflictmed/*.jsonl
python experiments/care_conflictmed_20261019/prepare.py \
  --advice local_data/conflictmed/advice_pairs.jsonl \
  --scenarios local_data/conflictmed/ConflictMed.jsonl \
  --out local_data/conflictmed_prepared_01
```

准备脚本默认拒绝非官方样本数：194 建议对、4,290 病例；核查场景正误配对、重复 ID、每一对样本的划分隔离；生成 `inputs.jsonl`（不含答案）、`labels.jsonl`（离线评估专用）、`manifest.json`（来源与文件 SHA256）。划分使用固定种子 `20261019`，原始配对所属同一划分。**不同配对可能对应相同指南，故不能声称跨指南零泄漏。** `--allow-partial` 仅用于虚构 fixture 的架构测试。

## 模型基线（只能在有授权 GPU 的机器上执行）

先确认 GPU 显存、模型路径与完整权重版本，并检查准备产物的 SHA256。正式测试前只能用训练集与开发集调参，测试集不得参与方向训练、层数/系数选择、门控阈值调参。以下先**只执行无干预基线**，不冒充方向引导或 CARE：

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/care_conflictmed_20261019/run_transformers.py \
  --inputs local_data/conflictmed_prepared_01/inputs.jsonl \
  --model /ABSOLUTE/PATH/TO/VERIFIED/Qwen3-8B \
  --split dev --track advice_choice \
  --out local_data/conflictmed_prepared_01/dev_advice_baseline.jsonl
```

模型脚本对 A/B 两个完整答案的续写对数似然评分；禁用 Qwen3 的思考模式，记录脚本和输出哈希。不要与自由生成准确率混为一谈。适用同一脚本的 `--track scenario_judgment`，以及最终冻结的 `--split test`。`--limit` 仅用于显式记录为不完整的 smoke test，不得送入正式报告。

```bash
python experiments/care_conflictmed_20261019/evaluate.py \
  --inputs local_data/conflictmed_prepared_01/inputs.jsonl \
  --labels local_data/conflictmed_prepared_01/labels.jsonl \
  --predictions local_data/conflictmed_prepared_01/PREDICTIONS_ALL_SPLITS.jsonl \
  --out local_data/conflictmed_prepared_01/analysis.json
```

评估器默认要求**每种方法对每条输入都有输出**；覆盖不全时拒绝输出可比的“总准确率”。当需合并分轨运行的预测文件，可按原始行连接并保持唯一 `(method,id)`。报告按 train/dev/test、任务轨道、上下文类型分层；不合法输出按错误计；按建议/场景原始配对聚类 bootstrap。报告来源摘录遵从率，特别记录「未核实摘录指向错误选项」时的错误采纳比例；不据此自动推断 causal mechanism。

## 现阶段状态

- **已实现、可在 CPU 上测试**：原始数据结构验收、分组、上下文成对扰动、答案隔离、评估逻辑和 GPU 推断入口。
- **尚未拿到本地原始数据**：当前执行容器无外网，因此不能确认下载后的文件结构是否完整匹配；`prepare.py` 的正式模式将拒绝不符合约定的数据。
- **未执行 GPU 推断或注意力干预**：当前容器无 GPU、无 Transformers。任何此前 160 例合成路由成绩都不能称本实验结果。

## 提交前快速检查

```bash
python -m pytest -q experiments/care_conflictmed_20261019/tests
python -m py_compile experiments/care_conflictmed_20261019/*.py
```

## 人群条件反事实：独立、待审查的结构轨

`conditions.py` 支持独立专家核查后的条件标注文件。每条标注必须包含指南出处、审查者、数值条件、阈值、单位、T/F 两种患者取值以及只含一个 `{value}` 占位符的患者观察句。程序验证两种取值确实翻转规则适用性，并生成 T/F/U 三元组和可定位证据/规则字符跨度。

```bash
python experiments/care_conflictmed_20261019/conditions.py \
  --annotations /ABSOLUTE/PATH/TO/INDEPENDENTLY_REVIEWED_RULES.jsonl \
  --out local_data/conditional_prepared_01
```

这是**规则适用性**轨，不是治疗建议正确率，也不是已经验证的模型参数冲突；无患者动作标注时 `clinical_action_gold=null`，不可据此对 CARE 的临床动作决策评分。脚本拒绝没有出处或审查标记的临床条件标注；它不能取代真正的临床内容审核。

## 已运行的本地预检（2026-10-19）

- `pytest`: **6 passed**；`py_compile` 通过。
- 基于 Hugging Face 数据卡片所显示的建议第 3、14 组，**人工转录的两组文本**，以 `--allow-partial` 构造了 12 条冲突输入；生成器记录了本地输入哈希，且状态为 `DATA_PREPARED_NOT_MODEL_EVALUATED`。这不是官方下载文件的完整性验证，也不能作为论文数据结果。
- 当前执行环境不能访问 Hugging Face 外网，且没有 GPU；官方 194 组建议和 4,290 个场景的完整构造检查、真实模型推理、方向干预均标记为**未执行**。