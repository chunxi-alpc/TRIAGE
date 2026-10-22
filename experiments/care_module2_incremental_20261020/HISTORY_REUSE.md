# 历史实验去重与可复用性审计

**原则：旧真实 GPU 实验的结果是证据，不是需要在新任务上“重现一遍”的开销。** 只在新设计有不同输入/目标而需要同条件公平比较时，允许新的基线前向。旧 PR 均在原分支保留，不能误认为全在 main。

| 位置及已有 GPU 工作 | 关键已观测结果 | 复用方式 | 本轮明确不做 |
| --- | --- | --- | --- |
| [PR #1](https://github.com/chunxi-alpc/TRIAGE/pull/1)，ACRE oracle OF/OS 同源目标选择 | 40 对，完整规则 oracle OS 未优于 OF：β=2 时 OF 31/40 对、OS 30/40 对 | 有限的目标切换负证据，区分单纯删除增强与否定目标 | 重做 ACRE 旧 40 对 |
| [PR #2](https://github.com/chunxi-alpc/TRIAGE/pull/2)，O2 288 层头重分析 | 训练适用性头与相关性头位置不同；变化幅度混杂严重 | N1 明确要求 R/A 解耦和距离控制 | 再做旧 O2 的同一 AUC 列表 |
| [PR #3](https://github.com/chunxi-alpc/TRIAGE/pull/3)，`care/module2-subspace-followup-20261017` | 原 384 例、192 对、12 家族：baseline 72/192 对，B1 72、B2 71、B3 72、B4 74；三个随机种子 76/76/71；B4 反事实键交换无稳定收益 | 子空间算子、旧缓存/权重校验、数据分组、随机/直接 bias 设计、统计方法 | 整套 B1–B4/SEKA 再扫描或同病例上重复配对交换 |
| [PR #8](https://github.com/chunxi-alpc/TRIAGE/pull/8)，`codex/care-module2-signed-diagnostics-20261019` | 同 192 对：baseline 72，B4 正向 74，B4 signed 75，均值 signed 75，score bias signed 77；学习子空间未可靠优于直接 bias | N1/N5 的主要简单基线和 BF16/RoPE 分数扰动校准 | 同一 oracle-signed 方法和旧 384 例重新跑 |
| [PR #9](https://github.com/chunxi-alpc/TRIAGE/pull/9)，`codex/care-reliability-20261019-work` | 旧 V6/V7 上游门控和联合：新机制主要 24 对均 0；过严证据门控和错误 U 映射影响使用率 | 门控失败诊断、技术失败保留和信号覆盖率指标 | 在旧 V6/V7 上重新做门控实验；把它当 A1 |
| [PR #10](https://github.com/chunxi-alpc/TRIAGE/pull/10)，`codex/care-steering-20261019` | Qwen3-8B、ConflictMedQA/DriftMed 方向、闭卷筛查、上下文误导：严格无选项已知 test 仅 1 事实组，错误上下文下方向与 baseline 均 0/2 | N2 的真实 P 筛查、同内容对照及失败模式；此结果**不是** Qwen3-4B 的低维双向 C/P 实验 | 再跑 8B 原数据原方向 |
| [PR #11](https://github.com/chunxi-alpc/TRIAGE/pull/11)，`codex/causal-attention-20261019` | 192 例 baseline 71、仅参考结构 160、单独动作注意力 73、组合 161；参考结构直接执行 177。效果主要来自结构而非注意力 | 新 N1/N5 必须包括简单提示/确定性执行、错配、反号和零操作检查 | 原层段/强度/动作+证据扫描、旧 192 例四臂 |
| [PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12)，`codex/care-module1-semantic-v8-20261019` | A1 历史 450/576、新 72 例 192/216 条规则；24 例 Qwen baseline 10、仅结构 15、仅干预 10、组合 15 | 冻结真实 A1 逐例输出，精确 ID/规则 hash 连接；24 例旧 Qwen 输出只读 | 重训 A1、重跑 24 例原四臂或重新筛选 A1 |
| [PR #15](https://github.com/chunxi-alpc/TRIAGE/pull/15) 与 [PR #16](https://github.com/chunxi-alpc/TRIAGE/pull/16) | A1 正式固定到 main 信号政策；V6 CLI 仍属历史接口 | 固定模块一语义与 eligibility 约束 | 用 A1_U/A2/V6 或 oracle 冒充冻结 A1 |
| [离线审计分支](https://github.com/chunxi-alpc/TRIAGE/tree/codex/module2-preexperiment-audit-20261020) | 26 个 CPU 数学/数据检查通过，历史 9,792 记录复算，**新神经实验 0 前向**；V6 old192/new100 缓存非 A1 | 数学 contract、数据连接风险和历史统计复核 | 重复相同复算、把 V6 缓存当 A1 |

## 三种关系不要混淆

1. **已做且足够回答旧问题：** 旧适用性投影/直接 bias 的方法比较、旧层头/强度扫描、旧 A1 接入四臂和旧 8B 冲突方向测试，不再重复。
2. **做过相近实验，但不能回答新问题：** 旧适用性空间不能证明 R/A 因果独立；8B 残差方向不能证明 4B 双向低维来源选择；旧结构+注意力联动不能证明 C/P × D/Q；V6/V7 不是 A1。
3. **新数据必须有自己公平的基线：** N1–N5 的无干预、同信息量文本政策、直接分数偏置或确定性流程需要在相同新评测输入上执行，不能直接拿旧 384/192 或 8B 分数做同分母比较。

## 已完成结果如何在新报告引用

- 逐 PR 保留模型、revision、训练来源、划分、指标分母与 SHA；不以单一排行榜混排不同数据。
- 新任务的数据、oracle、目标答案都可能改变，旧结果不能代替新的零强度基线。
- 各 PR 是否为独立确认、数据是否重复使用及多重选择情况需注明；历史有探索性调参和已见过的机制测试，不能宣称外部临床验证。
- 不修改、覆盖、静默重算历史 journal 和报告。若只读取和复算统计，注明 `offline_replay_no_forward`。

