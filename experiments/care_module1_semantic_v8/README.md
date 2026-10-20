# 模块一 A1：主线轻量语义适用性信号

本目录仅选择性纳入 [PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12) 的**原 A1 方法代码与固定示例**，以及 [PR #15](https://github.com/chunxi-alpc/TRIAGE/pull/15) 的冻结信号接口，不复制原实验分支承载的其他几百个历史变更与大体积原始结果。

当前研究默认方法与选择规则：[MODULE1_A1_FREEZE.zh.md](MODULE1_A1_FREEZE.zh.md)。原始实验完整报告、逐例结果和模型运行记录保留在 [PR #12 的原分支](https://github.com/chunxi-alpc/TRIAGE/tree/codex/care-module1-semantic-v8-20261019/experiments/care_module1_semantic_v8)；PR #13（完整检查和复核）与 PR #14（新的 U 对照示例）是未采用的独立消融实验。

## 当前代码

- `prompts.py`：原 A1 system prompt、规则文本构造、输出解析。
- `data/fewshot_examples.json`：PR #12 冻结的三个固定演示／四段患者叙述；不是 PR #14 的六段替代示例。
- `semantic_runner.py --arm A1`：原 PR #12 批量 GPU 推理入口，适用于同格式病例 JSONL；代码保留原有 batch=8、greedy、384 token 设置。
- `interface.py`：把原 A1 预测转换为规则状态和动作 ID；目标字符／词元位置由下游 Qwen 完整提示决定。
- `export_a1_signals.py`：仅转换已有 A1 预测，全病例保留，报告有／无信号数，不需要 GPU。
- `src/module1/semantic_a1.py`：主线单例推理工厂 `create_module1()`；批量 GPU 复现仍以上述原运行器为准。
- `scripts/run_module1_a1.py`：单例 A1 CLI。旧 `python -m src.module1` 仍为 V6，用于复现。

## 真实实验已知结论

A1 在历史 192 个合成病例（576 条规则）为 450/576（78.13%）；原子/逻辑树由作者预先提供的 A2 为 560/576（97.22%），但两者输入结构不同。在 PR #12 新写 72 例（216 条规则）分别为 192/216（88.89%）和 210/216（97.22%）。**A1 并非高精度临床判定器**；现将它冻结为模块二的预测信号来源，以便研究下游方向学习是否具有增益。完整口径见原 PR #12 报告。

## 离线导出示例

```bash
python experiments/care_module1_semantic_v8/export_a1_signals.py \
  --cases /path/to/semantic_confirmation_inputs.jsonl \
  --predictions /path/to/confirmation_A1/predictions.jsonl \
  --output /path/to/module2_A1_signals.jsonl
```

所有模型实验输入、旧完整运行产物均由原 PR #12 维护。本次主线整合没有重新运行 GPU 推理；不能把复制原始已验证代码理解成重跑完成。

