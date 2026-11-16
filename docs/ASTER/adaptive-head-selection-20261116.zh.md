# ASTER-CR 多 Backbone 自适应注意力头筛选

当前任务包含七模型自适应M1/M2/M3、开发与测试评价、阈值敏感性及结构诊断。按用户最新要求，取消置信区间计算和固定51头对照；历史固定预算结果保留。

选头使用训练来源的梯度置换筛选，以及独立开发来源的双向激活修补和随机供体对照。最终要求BY q≤0.05、平均超额干预效应>0、至少16个有效开发来源。来源重采样用于选头稳定性。

训练数据保持2,540对及80/20/40划分。原20个开发来源新增43对，四类各覆盖20个来源。27项CPU测试和GPU冒烟通过，统一队列在GPU 0、1、5、6执行，并复用已提取缓存。

- [代码与执行说明](../../experiments/aster_e678_adaptive_20261116/README.zh.md)
- [实验报告](../../reports/aster_e678_adaptive_20261116/REPORT.zh.md)
- [范围调整与缓存续用](../../reports/aster_e678_adaptive_20261116/scope_revision.json)
- [测试结果](../../reports/aster_e678_adaptive_20261116/test_results.json)

工作目录：/home/gcx/aster-e678-adaptive-20261116。队列：/home/gcx/aster-gpu-queue-20261113/adaptive_20261116。全部任务结束后自动汇总并上传GitHub。
