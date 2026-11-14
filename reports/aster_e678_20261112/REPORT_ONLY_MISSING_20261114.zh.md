# E6/E7/E8 仅补未运行实验：当前状态

<!-- HISTORICAL-SCOPE-START -->
> **历史批次/审计快照。** 以下完成或缺失状态对应当时范围；后续CPU补算已完成，GPU接续按统一队列执行。当前逐项状态见[实验框架](../../docs/ASTER/experiments.zh.md)及其执行状态附件。原始数值和审计证据保留。
<!-- HISTORICAL-SCOPE-END -->

更新时间：2026-11-13 16:52:13。执行依据：用户要求“没跑过的实验才跑”。

Gemma2 IT、Gemma3、MedGemma 的旧训练与 MEGA 评价已有历史完整结果，三个整套流程已停止。已有本轮参数、缓存和逐例结果全部保留。当前继续两个 Base 的 E1 比较，自动队列不会恢复被停止的流程。

| 模型 | 当前安排 | E1 参数冻结 | E1 受控测试 |
|---|---|---|---|
| Gemma-2-9B-IT | 已停止重复流程 | 完成 | 完成 |
| Gemma-2-9B-Base | 继续 Base 比较 | 完成 | 完成 |
| Llama-3.1-8B-Instruct | 已完成，复用 | 完成 | 完成 |
| Gemma-3-27B-IT | 已停止重复流程 | 未完成 | 未完成 |
| MedGemma-27B-Text-IT | 已停止重复流程 | 未完成 | 未完成 |
| Qwen3-8B-Base | 继续 Base 比较 | 未完成 | 未完成 |

## 跨任务评价

Qwen 的 E1 四个评价总体继续直接复用。Llama 和 Gemma2 本轮已完成的跨任务结果保留。
Gemma3 和 MedGemma 已拟合 E1，但源任务开发评价与参数冻结尚未完成；未运行的 NLI 全量和 MedPIC 评价暂列待处理，不自动启动。旧 NLI 500 条记录按历史协议保留。

| 模型 | NLI 500 | NLI 5500 | MedPIC 188 | MedPIC 467 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 已完成 | 已完成 | 已完成 | 已完成 |
| Gemma-2-9B-IT | 已完成 | 已完成 | 已完成 | 已完成 |
| Gemma-3-27B-IT | 复用历史记录 | 待源任务冻结 | 待源任务冻结 | 待源任务冻结 |
| MedGemma-27B-Text-IT | 复用历史记录 | 待源任务冻结 | 待源任务冻结 | 待源任务冻结 |

## Base/IT

- gemma2：比较完成；/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112/base_it/gemma2
- qwen：等待 Base 完成；/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112/base_it/qwen

## 结果使用

旧协议使用 A/P 4887 对、C/F 135 对；当前 E1 使用 A/P/C/F 各635对。两套协议分别标记，旧分数不填入 E1 结果列。
历史806路径与官方7458路径保留既往使用记录。未运行单元保留状态，不填数值。
当前范围完成只表示两个 Base 及其比较收尾完成，不把停止的三个 IT 流程或待处理迁移评价记为完成。

- 调整记录：/home/gcx/aster-gpu-queue-20261113/only_missing_20261114/
- 历史核对：/home/gcx/aster-e5-gaps-20261112/reports/aster_e5_gaps_20261112/running_history_check_20261114/
- 策略：only_missing_policy.json；逐项状态：status_only_missing.json。
