# 编码修复后 P1–P6 实际进度

旧数据、模型输出与失败记录保存在 [修订开发集旧运行](../aster_prelim_repaired_20261027)。本轮从 P1 重验：Llama 全部重新推理；Qwen 仅在全部词元及权重一致性核对后复用。

| 阶段 | 当前状态 |
|---|---|
| P1 | PASS_SW_GATE |
| P2 | PASS_NEUTRAL_GATE |
| P3 | COMPLETED |
| P4 | PASS |
| P5 | PASS_PRUNE_GATE |
| P6 | AUTO100_FULL_BASELINES_RUNNING |

P1–P5：NC/IC/AC/SW 各 169 例，CS 17 例；4 个开发树。两模型均使用 FP32。测试集推理为 0。P6 的数据准备和资源另见 p6/README.zh.md。

具体修复和标签范围见 [编码修复说明](TOKENIZATION_FIX.zh.md)，冻结指纹见 [freeze.json](results/freeze.json)。

P6 不自动跳过用户要求的人工核验和数据规模门槛；前置通过后由定时任务继续准备。

更新时间（UTC）：2026-10-28T02:32:50.816943+00:00
