# P2 诱饵对照实测

P1 SW 门槛通过后才启动本轮。全部为相同的 169 个修订开发病例；每模型 3 类诱饵 × 3 个控制条件 × 2 种可见性，另做 169 次闭卷前向。

状态：**STOP_VISIBILITY_NARRATIVE**；中性代号的两模型平均“盲判−明判”差距为 4.437869822485207 个百分点。

| 模型 | 诱饵 | 控制 | 盲判 % | 明判 % | 盲判−明判（百分点） |
|---|---|---|---:|---:|---:|
| qwen | familiar | standard | 86.98 | 85.21 | +1.78 |
| qwen | familiar | no_options | 86.98 | 85.80 | +1.18 |
| qwen | familiar | reversed | 79.29 | 66.27 | +13.02 |
| qwen | unfamiliar | standard | 86.98 | 94.08 | -7.10 |
| qwen | unfamiliar | no_options | 86.98 | 91.12 | -4.14 |
| qwen | unfamiliar | reversed | 79.29 | 77.51 | +1.78 |
| qwen | neutral | standard | 86.98 | 92.90 | -5.92 |
| qwen | neutral | no_options | 86.98 | 92.31 | -5.33 |
| qwen | neutral | reversed | 79.29 | 78.11 | +1.18 |
| llama | familiar | standard | 70.41 | 50.30 | +20.12 |
| llama | familiar | no_options | 70.41 | 66.27 | +4.14 |
| llama | familiar | reversed | 60.36 | 56.21 | +4.14 |
| llama | unfamiliar | standard | 70.41 | 55.62 | +14.79 |
| llama | unfamiliar | no_options | 70.41 | 72.19 | -1.78 |
| llama | unfamiliar | reversed | 60.36 | 65.09 | -4.73 |
| llama | neutral | standard | 70.41 | 55.62 | +14.79 |
| llama | neutral | no_options | 70.41 | 75.74 | -5.33 |
| llama | neutral | reversed | 60.36 | 60.95 | -0.59 |

| 模型 | 诱饵 | 闭卷置信度档 | 例数 | 盲判−明判（百分点） |
|---|---|---|---:|---:|
| qwen | familiar | low | 57 | -5.26 |
| qwen | familiar | middle | 56 | +5.36 |
| qwen | familiar | high | 56 | +5.36 |
| qwen | unfamiliar | low | 57 | -12.28 |
| qwen | unfamiliar | middle | 56 | -5.36 |
| qwen | unfamiliar | high | 56 | -3.57 |
| qwen | neutral | low | 57 | -8.77 |
| qwen | neutral | middle | 56 | -5.36 |
| qwen | neutral | high | 56 | -3.57 |
| llama | familiar | low | 57 | +28.07 |
| llama | familiar | middle | 56 | +16.07 |
| llama | familiar | high | 56 | +16.07 |
| llama | unfamiliar | low | 57 | +19.30 |
| llama | unfamiliar | middle | 56 | +16.07 |
| llama | unfamiliar | high | 56 | +8.93 |
| llama | neutral | low | 57 | +17.54 |
| llama | neutral | middle | 56 | +14.29 |
| llama | neutral | high | 56 | +12.50 |

实施范围：

- 熟悉处置使用来源原正确动作；陌生处置采用另一开发树且不在本题选项内的动作。这只是预先固定的操作性代理，未独立证明该动作对模型陌生。
- 中性代号按模型匹配完整可见提示的 token 数；病例、条件、问题、选项和另一分支的处置不变。代号只有字母序列，不代表真实治疗。
- 去选项条件仍保留原问题；顺序控制只反转显示次序，分支编号与金标不变。
- 闭卷置信度是选项字母集合内归一化的原动作概率；按模型分成三等份，不按置信度剔除病例。它不是模型校准后的绝对正确概率。
- 全部解析失败计错，原始逐例输出和概率均保留；相同提示仅计算一次。区间按 4 个树聚类，不能作为正式测试结论。
- “中性差距较大”以 3 个百分点作本轮事前操作定义。模型间差异和熟悉−中性直接对照单列，不用平均数隐去。

**触发 P2 止损：中性代号仍产生较大差距。不能将 P1 的全部效应归因为熟悉处置。应先转为可见性／提示信息影响的叙事；严格按用户要求，不进入 P3–P6。**
