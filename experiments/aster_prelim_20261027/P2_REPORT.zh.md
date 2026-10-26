# P2 诱饵对照实测

P1 SW 门槛通过后才启动本轮。全部为相同的 200 个开发病例；每模型 3 类诱饵 × 3 个控制条件 × 2 种可见性，另做 200 次闭卷前向。

状态：**PASS_NEUTRAL_GATE**；中性代号的两模型平均“盲判−明判”差距为 -5.75 个百分点。

| 模型 | 诱饵 | 控制 | 盲判 % | 明判 % | 盲判−明判（百分点） |
|---|---|---|---:|---:|---:|
| qwen | familiar | standard | 80.50 | 69.50 | +11.00 |
| qwen | familiar | no_options | 80.50 | 74.50 | +6.00 |
| qwen | familiar | reversed | 73.00 | 58.50 | +14.50 |
| qwen | unfamiliar | standard | 80.50 | 90.00 | -9.50 |
| qwen | unfamiliar | no_options | 80.50 | 80.50 | +0.00 |
| qwen | unfamiliar | reversed | 73.00 | 68.50 | +4.50 |
| qwen | neutral | standard | 80.50 | 93.50 | -13.00 |
| qwen | neutral | no_options | 80.50 | 85.00 | -4.50 |
| qwen | neutral | reversed | 73.00 | 71.00 | +2.00 |
| llama | familiar | standard | 66.00 | 50.50 | +15.50 |
| llama | familiar | no_options | 66.00 | 61.50 | +4.50 |
| llama | familiar | reversed | 58.00 | 49.50 | +8.50 |
| llama | unfamiliar | standard | 66.00 | 61.00 | +5.00 |
| llama | unfamiliar | no_options | 66.00 | 73.00 | -7.00 |
| llama | unfamiliar | reversed | 58.00 | 62.00 | -4.00 |
| llama | neutral | standard | 66.00 | 64.50 | +1.50 |
| llama | neutral | no_options | 66.00 | 78.00 | -12.00 |
| llama | neutral | reversed | 58.00 | 57.50 | +0.50 |

| 模型 | 诱饵 | 闭卷置信度档 | 例数 | 盲判−明判（百分点） |
|---|---|---|---:|---:|
| qwen | familiar | low | 67 | +10.45 |
| qwen | familiar | middle | 67 | +8.96 |
| qwen | familiar | high | 66 | +13.64 |
| qwen | unfamiliar | low | 67 | -14.93 |
| qwen | unfamiliar | middle | 67 | -7.46 |
| qwen | unfamiliar | high | 66 | -6.06 |
| qwen | neutral | low | 67 | -14.93 |
| qwen | neutral | middle | 67 | -16.42 |
| qwen | neutral | high | 66 | -7.58 |
| llama | familiar | low | 67 | +23.88 |
| llama | familiar | middle | 67 | +4.48 |
| llama | familiar | high | 66 | +18.18 |
| llama | unfamiliar | low | 67 | +13.43 |
| llama | unfamiliar | middle | 67 | -1.49 |
| llama | unfamiliar | high | 66 | +3.03 |
| llama | neutral | low | 67 | +7.46 |
| llama | neutral | middle | 67 | -4.48 |
| llama | neutral | high | 66 | +1.52 |

实施范围：

- 熟悉处置使用来源原正确动作；陌生处置采用另一开发树且不在本题选项内的动作。这只是预先固定的操作性代理，未独立证明该动作对模型陌生。
- 中性代号按模型匹配完整可见提示的 token 数；病例、条件、问题、选项和另一分支的处置不变。代号只有字母序列，不代表真实治疗。
- 去选项条件仍保留原问题；顺序控制只反转显示次序，分支编号与金标不变。
- 闭卷置信度是选项字母集合内归一化的原动作概率；按模型分成三等份，不按置信度剔除病例。它不是模型校准后的绝对正确概率。
- 全部解析失败计错，原始逐例输出和概率均保留；相同提示仅计算一次。区间按 4 个树聚类，不能作为正式测试结论。
- “中性差距较大”以 3 个百分点作本轮事前操作定义。模型间差异和熟悉−中性直接对照单列，不用平均数隐去。
