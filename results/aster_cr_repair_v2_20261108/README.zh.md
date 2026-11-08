# repair-v2 下游实验

数据：/home/gcx/aster_cr_repair_20261108。使用现有 2,782 节点、11,512 输入、8,730 对；不重复抽取。

执行顺序：Qwen fit/dev → P0、定位、H2/H3 → 读出与消融 → 四类训练对照各三个学习率 → dev 选参冻结 → test → MedGemma 关键确认 → NLI4CT/MedPIC 独立评价。

GPU 使用 0、1、5、6；MedGemma 主流程使用 6，独立外部评价可同时使用 0 和 1。四卡扩容后剩余流程暂估 20–40 小时。稳定后不持续监控。

旧结果仅导入原始输入及评分口径一致的 Full 分数、激活和路径缓存。定位、训练和选参在本目录重新产生；不复制旧完成标记。数据接口包含 F/F0，原始节点与新增候选分开报告。

原始节点分母：fit 761、dev 199、test 372。原始路径分母：fit 1,712、dev 420、test 806。P/C/F 配对数、独立节点数、来源数与区间分别记录。

启动前验证见 preflight.json、verification/ 和 logs/regression.log。运行状态见 status/pipeline.json；主日志 logs/controller.log；阶段日志 logs/。结果报告 REPORT.zh.md 在流程推进与完成时自动更新。

外部数据保持原题标签；NLI4CT 使用 Entailment/Contradiction，MedPIC 保留多选答案并按选项文字比较。使用确定性生成；不参与 MEGA 训练、选参、宏平均和路径分母。此处为基础模型原生任务验证，不将 MEGA 二元读出强行用于多选任务。

代码：/home/gcx/TRIAGE-worktrees/aster-cr-20261108/experiments/aster_cr/。
重新启动使用相同环境变量 ASTER_CR_ROOT=/home/gcx/aster_cr_repair_v2_run_20261108、ASTER_CR_GPUS=0,1,5,6；本次扩容从训练阶段接续，入口 python -m experiments.aster_cr.resume_repair --resume-training。无需重新 prepare，也不会重新抽取。
