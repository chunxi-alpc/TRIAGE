# ASTER-CR 多 Backbone 自适应注意力头筛选

自适应 M1 使用训练来源的梯度置换筛选，以及独立开发来源的双向激活修补和随机供体对照确认。最终集合由 BY 显著性、来源级置信区间及有效来源数确定；固定 51 头模式保留用于匹配对照。

训练数据保持 E1 的 2,540 对及 80/20/40 来源划分。经用户确认，原 20 个开发来源新增 43 对缺失类别配对，使 A/P/C/F 各覆盖 20 个来源。

25 项 CPU 测试和 Qwen GPU 冒烟测试通过。七个检查点已进入统一 GPU 队列，使用 0、1、5、6 卡。当前尚无正式七模型对照结论。

- [实验代码与执行说明](../../experiments/aster_e678_adaptive_20261116/README.zh.md)
- [完整实验报告](../../reports/aster_e678_adaptive_20261116/REPORT.zh.md)
- [测试结果](../../reports/aster_e678_adaptive_20261116/test_results.json)
- [GPU 资源与时长计划](../../reports/aster_e678_adaptive_20261116/resource_plan.json)
- [开发配对来源及修改记录](../../reports/aster_e678_adaptive_20261116/development/pair_edits.jsonl)

服务器工作目录：/home/gcx/aster-e678-adaptive-20261116。统一队列目录：/home/gcx/aster-gpu-queue-20261113/adaptive_20261116。队列在全部任务结束后运行 CPU 汇总，更新每模型诊断与 REPORT.zh.md。
