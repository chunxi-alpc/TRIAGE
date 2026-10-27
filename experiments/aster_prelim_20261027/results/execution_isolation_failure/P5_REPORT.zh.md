# P5 金标执行通道实测

状态：**INCOMPLETE**。只在两模型 P4 全部通过后运行。

| 模型 | 卷 | 方法 | n | 准确率 % |
|---|---|---|---:|---:|
| qwen | AC | gold_prompt | 200 | 94.00 |
| qwen | AC | gold_rule | 200 | 99.50 |
| qwen | AC | gold_prune | 200 | 83.00 |
| qwen | AC | gold_prune_isolated | 200 | 19.00 |
| qwen | AC | gate_0 | 200 | 18.00 |
| qwen | AC | gate_0.5 | 200 | 19.00 |
| qwen | AC | gate_1 | 200 | 19.00 |
| qwen | AC | gate_2 | 200 | 19.00 |
| qwen | AC | gate_4 | 200 | 19.00 |
| qwen | AC | gate_8 | 200 | 19.00 |
| qwen | AC | gate_16 | 200 | 19.00 |
| qwen | AC | gate_hard | 200 | 19.00 |
| qwen | AC | gate_8_green | 200 | 18.00 |
| qwen | AC | projection_0.5 | 200 | 18.00 |
| qwen | AC | projection_1 | 200 | 18.00 |
| qwen | SW | gold_prompt | 200 | 93.00 |
| qwen | SW | gold_rule | 200 | 95.00 |
| qwen | SW | gold_prune | 200 | 78.00 |
| qwen | SW | gold_prune_isolated | 200 | 17.50 |
| qwen | SW | gate_0 | 200 | 17.50 |
| qwen | SW | gate_0.5 | 200 | 17.50 |
| qwen | SW | gate_1 | 200 | 17.50 |
| qwen | SW | gate_2 | 200 | 17.50 |
| qwen | SW | gate_4 | 200 | 17.50 |
| qwen | SW | gate_8 | 200 | 17.50 |
| qwen | SW | gate_16 | 200 | 17.50 |
| qwen | SW | gate_hard | 200 | 17.50 |
| qwen | SW | gate_8_green | 200 | 18.50 |
| qwen | SW | projection_0.5 | 200 | 17.50 |
| qwen | SW | projection_1 | 200 | 17.50 |
| qwen | NC | gold_prompt | 50 | 98.00 |
| qwen | NC | gold_rule | 50 | 96.00 |
| qwen | NC | gold_prune | 50 | 94.00 |
| qwen | NC | gold_prune_isolated | 50 | 20.00 |
| qwen | NC | gate_0 | 50 | 18.00 |
| qwen | NC | gate_0.5 | 50 | 18.00 |
| qwen | NC | gate_1 | 50 | 18.00 |
| qwen | NC | gate_2 | 50 | 18.00 |
| qwen | NC | gate_4 | 50 | 18.00 |
| qwen | NC | gate_8 | 50 | 18.00 |
| qwen | NC | gate_16 | 50 | 18.00 |
| qwen | NC | gate_hard | 50 | 18.00 |
| qwen | NC | gate_8_green | 50 | 18.00 |
| qwen | NC | projection_0.5 | 50 | 18.00 |
| qwen | NC | projection_1 | 50 | 18.00 |
| qwen | IC | gold_prompt | 50 | 98.00 |
| qwen | IC | gold_rule | 50 | 98.00 |
| qwen | IC | gold_prune | 50 | 94.00 |
| qwen | IC | gold_prune_isolated | 50 | 20.00 |
| qwen | IC | gate_0 | 50 | 20.00 |
| qwen | IC | gate_0.5 | 50 | 20.00 |
| qwen | IC | gate_1 | 50 | 20.00 |
| qwen | IC | gate_2 | 50 | 20.00 |
| qwen | IC | gate_4 | 50 | 20.00 |
| qwen | IC | gate_8 | 50 | 20.00 |
| qwen | IC | gate_16 | 50 | 20.00 |
| qwen | IC | gate_hard | 50 | 20.00 |
| qwen | IC | gate_8_green | 50 | 20.00 |
| qwen | IC | projection_0.5 | 50 | 20.00 |
| qwen | IC | projection_1 | 50 | 20.00 |
| qwen | CS | gold_prompt | 17 | 100.00 |
| qwen | CS | gold_rule | 17 | 94.12 |
| qwen | CS | gold_prune | 17 | 100.00 |
| qwen | CS | gold_prune_isolated | 17 | 11.76 |
| qwen | CS | gate_0 | 17 | 11.76 |
| qwen | CS | gate_0.5 | 17 | 11.76 |
| qwen | CS | gate_1 | 17 | 11.76 |
| qwen | CS | gate_2 | 17 | 11.76 |
| qwen | CS | gate_4 | 17 | 11.76 |
| qwen | CS | gate_8 | 17 | 11.76 |
| qwen | CS | gate_16 | 17 | 11.76 |
| qwen | CS | gate_hard | 17 | 11.76 |
| qwen | CS | gate_8_green | 17 | 11.76 |
| qwen | CS | projection_0.5 | 17 | 11.76 |
| qwen | CS | projection_1 | 17 | 11.76 |

金标裁剪门槛：`{"qwen": {"best_development_gate": "gate_0.5", "prune_minus_best_gate_pp": {"SW": 60.5, "AC": 64.0}, "prune_minus_hard_gate_pp": {"SW": 60.5, "AC": 64.0}, "gate_matches_prune_within_2pp": false, "selection_is_development_only": true}}`。

- 主裁剪为普通文本裁剪、自然位置重排与原生因果注意力；隔离裁剪另外列出，不替代主对照。
- 门控采用概率下限 1e-4，报告全部强度与硬门控；继续门槛要求硬门控在 SW、AC 上分别距主裁剪不超过 2 个百分点。
- 投影使用匹配模型的既有官方 SEKA 投影，实施 k ← k−γPk，γ∈{0.5,1}。这是算子诊断，不是完整 SEKA 标准任务复现，也不能推广为所有投影方法更弱。
- 有界强调对金标规则的编号、条件和处置加 +1，层段取 P3 开发选择；没有更改病例证据。
- 注意力份额仅在事先固定的 20 个 SW 开发病例记录，从真实 Q/K 与掩码用 FP32 重算，逐层原值另存。不同强度会改变上游表示，端到端非单调不能冒充固定 Q/K 解析性质的反例或证明。
- 全部数据是开发数据，含本轮参数选择，不作为独立测试。P6 仍需其独立数据有效性与人工核验门槛。
