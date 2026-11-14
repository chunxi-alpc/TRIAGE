# E6/E7/E8 仅补未运行实验：当前状态

<!-- HISTORICAL-SCOPE-START -->
> **历史批次/审计快照。** 以下完成或缺失状态对应当时范围；后续CPU补算已完成，GPU接续按统一队列执行。当前逐项状态见[实验框架](../../docs/ASTER/experiments.zh.md)及其执行状态附件。原始数值和审计证据保留。
<!-- HISTORICAL-SCOPE-END -->

更新时间：2026-11-14 02:30:43。执行依据：用户要求“没跑过的实验才跑”。

Gemma2 IT、Gemma3、MedGemma 的三个整套流程继续停止。两个 Base 任务继续；Gemma3、MedGemma 复用已完成 E1 拟合参数，单独补完剩余开发集计算、冻结及缺失迁移评价。调度器只接续已列明的未完成阶段。

| 模型 | 当前安排 | E1 参数冻结 | E1 受控测试 |
|---|---|---|---|
| Gemma-2-9B-IT | 已停止重复流程 | 完成 | 完成 |
| Gemma-2-9B-Base | 已完成，复用 | 完成 | 完成 |
| Llama-3.1-8B-Instruct | 已完成，复用 | 完成 | 完成 |
| Gemma-3-27B-IT | 仅补缺失迁移；复用已拟合参数 | 完成 | 未完成 |
| MedGemma-27B-Text-IT | 仅补缺失迁移；复用已拟合参数 | 完成 | 未完成 |
| Qwen3-8B-Base | 已完成，复用 | 完成 | 完成 |

## 跨任务评价

Qwen 的 E1 四个评价总体继续直接复用。Llama 和 Gemma2 本轮已完成的跨任务结果保留。
Gemma3 和 MedGemma 按既定 E1 配置补完开发集冻结，再评价 NLI 5500 与 MedPIC 188/467。NLI 500 的当前 E1 冻结迁移统计直接使用已有2864条缓存，在CPU计算；历史NLI内拟合结果单独保留。MedPIC 467含既有开发问题，作为全题描述性评价。

| 模型 | NLI 500 | NLI 5500 | MedPIC 188 | MedPIC 467 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 已完成 | 已完成 | 已完成 | 已完成 |
| Gemma-2-9B-IT | 已完成 | 已完成 | 已完成 | 已完成 |
| Gemma-3-27B-IT | 已完成 | 已完成 | 已完成 | 已完成 |
| MedGemma-27B-Text-IT | 已完成 | 已完成 | 已完成 | 已完成 |

## Base/IT

- gemma2：比较完成；/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112/base_it/gemma2
- qwen：比较完成；/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112/base_it/qwen

## 结果使用

旧协议使用 A/P 4887 对、C/F 135 对；当前 E1 使用 A/P/C/F 各635对。两套协议分别标记，旧分数不填入 E1 结果列。
历史806路径与官方7458路径保留既往使用记录。未运行单元保留状态，不填数值。
当前范围完成包含两个 Base、两组 Base/IT 比较及两模型缺失迁移评价。Gemma3、MedGemma的E1源任务受控测试与官方MEGA，以及Gemma2尚未完成的本轮官方MEGA继续列缺项；历史结果按原协议保留，不填入对应E1列。

- 调整记录：/home/gcx/aster-gpu-queue-20261113/only_missing_20261114/
- 历史核对：/home/gcx/aster-e5-gaps-20261112/reports/aster_e5_gaps_20261112/running_history_check_20261114/
- 策略：only_missing_policy.json；逐项状态：status_only_missing.json。
