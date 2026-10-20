# 当前状态

> 2026-10-20 后续离线诊断：已确认候选 token 与模型最高概率输出形式不匹配，并发现 N5 拟合样本不一致。以下完成状态只表示执行结束，原选参与科学解释待纠正；详见 reports/dev_strength_audit_20261020/REPORT.zh.md。本次诊断新增模型前向为0。

本轮已结束，退出码 0；按协议保留门槛停止。

- S1: gpu_smoke_passed
- N1: dev 已评估；test not_run_gate_fail
- N2/N3/N5: gpu_evaluated
- N4: not_run_gate_fail

最终报告：runs/v2_20261020T134304Z/REPORT.zh.md
实际新前向1157次，所有 route 开发选择为零强度；无新L1/L2/L3正证据。
A1冻结，历史实验未重跑，未合并main。
