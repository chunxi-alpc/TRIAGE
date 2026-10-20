# CARE 模块一研究方法冻结：A1 → 模块二适用性信号

**决策（2026-10-20）：模块二后续实验的模块一预测信号固定为 PR #12 的语义版 A1。** 这是研究信号来源的冻结，不是声称 A1 的临床准确率、模块二有效性或现有 V6 流水线已经被整体替换。冻结配置见 [`configs/care_module1_for_module2.json`](../../configs/care_module1_for_module2.json)。

## 1. 冻结什么

- 模型：原 Phi-4＋TrialMatchAI LoRA；`semantic_runner.py --arm A1`，不重新训练模型。
- 输入：问题、患者文本、所有候选规则的 `rule_id` 与完整自然语言规则；不输入标准答案、结构化患者金标或正确动作。
- 提示：保留 PR #12 A1 的**三个固定训练演示／四段患者叙述**及同一系统指令。固定提示、示例、模型和生成设置的版本及摘要在配置文件中。
- 预测：对每条规则给出 `T`（适用）、`F`（不适用）或 `U`（信息不足）；原有短证据摘录与简短理由可保留作诊断，但不参与模块二的准入。
- 模块二信号：`T→+1`，`F→−1`，`U→0`（不编辑）；技术失败／无法取得状态→`null`（不编辑，单独记载）。目标动作来自输入规则 ID 和最终提示中的动作位置，**不让模型重新生成动作**。
- 本轮只研究**有可用预测 T/F 信号且对应动作可定位**的病例。不能使用金标判断正确与否、后续注意力干预是否有效来挑选病例；完整母集及无信号数始终保留，模块二所有方法使用同一批病例。

**不做**原子分解、完整逻辑树证明、置信度校准、NLI 一致性或证据充分性硬门控、二次 LLM 复核，也不做 RAG。A1 是一个尽量简单的、可以实际产生适用性信号的冻结上游。任何未来升级都作为独立消融，不得悄悄修改该版本。

## 2. 为什么是 A1，而非最新 PR #14

- [PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12)：A1 在原 96 例开发集得到 220/288（76.39%），历史 192 例 450/576（78.13%），新写 72 例 192/216（88.89%）；已在固定 24 例上验证结构与 Qwen 的接入。简单、完整、可直接复用。
- [PR #13](https://github.com/chunxi-alpc/TRIAGE/pull/13)：一次完整条件检查 B1 和双轮复核 B2 没有稳定优于 A1，B2 成本约翻倍，因此不采用。
- [PR #14](https://github.com/chunxi-alpc/TRIAGE/pull/14)：**不是动态检索，也不是训练/微调。** 只把 A1 原三组固定演示（四段病例）替换成两组新编 OR／例外 T/F/U 对照（六段病例），系统指令、模型、JSON 输出及 greedy／384-token 设置不变。仅在旧 96 例 dev 上得到 230/288 对 A1 的 220/288；目标配对 19/36 对 12/36；F→T 为 18/176 对 24/176。但 U 召回 **10/32 低于 14/32**，宏 F1 从 0.6976 降至 0.6877，总 token 增约 14.52%。没有运行历史 192 或新增 72 例验证，故保持探索性候选，不替换冻结 A1。
- [PR #11 的 A2](https://github.com/chunxi-alpc/TRIAGE/pull/11) 在已有可执行原子结构上的规则状态正确率为 560/576（97.22%），而不代表未经预解析指南的纯自然语言端到端性能。A2 可作为**预解析上限对照**，不输入模块二的主预测信号。

A1 预测也会错，特别是 F→T 和漏识别 U。冻结它不是宣布准确率足够，只是将其噪声固定下来，以便后续研究方向学习本身能否带来效果。

## 3. 可执行输入与模块二桥接

继续用原始 A1 推理程序，强制指定 `--arm A1`：

```bash
/home/gcx/n/.venv/bin/python experiments/care_module1_semantic_v8/semantic_runner.py \
  --cases experiments/care_module1_semantic_v8/data/semantic_confirmation_inputs.jsonl \
  --arm A1 --output /home/gcx/care_a1_future_signal_run
```

已有结果可直接离线导出信号（不调用 GPU）：

```bash
python experiments/care_module1_semantic_v8/export_a1_signals.py \
  --cases experiments/care_module1_semantic_v8/data/semantic_confirmation_inputs.jsonl \
  --predictions experiments/care_module1_semantic_v8/results/confirmation_A1/predictions.jsonl \
  --output /home/gcx/care_a1_signals_20261020.jsonl
```

导出结果对每个原病例都有一行：`case_id`、`split`、`family_id`、`rules:[{rule_id,state,sign,action_text,...}]`、`eligible_for_module2_signal_study`；同时产生同目录 `.summary.json` 计数文件。**动作文本不是 Qwen token 位置**：模块二必须在最终完整提示分词后，按规则 ID 解析动作 token span。现有 [接口代码](interface.py) 可复用 `build_interface` 和 `signed_action_targets` 进行转换。技术失败不混同医学 U；有一条规则失败不会强制取消其他已得到的 T/F 信号。

主分析可限于信号子集，但报告里仍列出母集总数、信号子集数、无信号数与技术失败数，并让直接偏置、学习方向、零干预对照使用**同一个子集**。参考标签只在事后分析信号误差；另立的 oracle 诊断必须明确标注，不准反向筛主实验。

## 4. 与仓库现有入口的关系

`python -m src.module1`、`src/pipeline/runner.py` **目前仍使用 V6 的 `module1.v6` 图和门控契约**。本冻结只作用于 CARE **模块二研究实验的信号上游**，不把 A1 的简化结果冒充 V6 图，也不修改旧三模块入口。需要迁移产品级流水线时，另行实现适配并测试，不能直接把 A1 JSON 塞进要求 V6 图的接口。

本次冻结不新增模型推断，不修改 PR #12—#14 原始实验记录，也不合并或重置其他分支。
