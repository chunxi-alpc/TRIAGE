# 当前状态

最新附加开发检查：[N3保留均值方向的基底](runs/v4_n3_mean_basis_20261020T151946Z/REPORT.zh.md)。31项CPU测试通过，新增48次开发前向，beta=.5为11/16，mean/旧route为10/16，text为12/16。N2、N5和原测试结果未动；以下仍是原主研究结果。

2026-10-20：候选 token 和 N5 拟合修复完成；22项CPU测试、原oracle/无泄漏检查、672条实际tokenizer边界检查通过。开发与测试阶段均已结束，退出码0。

- S1：gpu_smoke_passed。
- N1：纠正开发四格2/4，route beta=0；测试 not_run_gate_fail。
- N2：corrected_test_complete；双向route 0/8，text 4/8。真实知识轨双方均0/8。
- N3：corrected_test_complete；route beta=1，测试20/32；mean beta=.5，25/32；text 27/32。
- N4：not_run_gate_fail。
- N5：corrected_test_complete；新配对轴route beta=0，9/24；text 11/24，filter 10/24。

本轮新增187次前向（开发/验证43，测试144），GPU0峰值7.663GiB。旧结果按可验证的最高概率候选精确恢复；不能恢复的分数不补造。原始输出、负结果、失败和统计均保留。A1完全冻结，历史实验未重跑，不合并main。

[最终报告](runs/v3_repair_20261020T144100Z/REPORT.zh.md)。测试病例此前已见，本轮属于修复后的探索性重评；无低秩route优于强基线的正证据。
