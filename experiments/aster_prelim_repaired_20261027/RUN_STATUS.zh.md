# 修订开发集 P1–P6 实际进度

旧数据、模型输出与失败记录保存在 [原实验目录](../aster_prelim_20261027)。本轮重新从 P1 验收，不复用旧预测。

| 阶段 | 当前状态 |
|---|---|
| P1 | PASS_SW_GATE |
| P2 | STOP_VISIBILITY_NARRATIVE |
| P3 | NOT_RUN |
| P4 | NOT_RUN |
| P5 | NOT_RUN |
| P6 | NOT_RUN |

NC/IC/AC/SW 各 169 例，CS 17 例；4 个开发树。两模型均使用 FP32。测试集推理为 0。

具体修复和标签范围见 [修复说明](REPAIR.zh.md)，冻结指纹见 [freeze.json](results/freeze.json)。

P6 不自动跳过用户要求的人工核验和数据规模门槛；前置通过后由定时任务继续准备。

更新时间（UTC）：2026-10-27T04:23:14.129639+00:00
