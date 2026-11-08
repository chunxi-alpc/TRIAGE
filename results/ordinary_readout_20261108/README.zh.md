# 普通线性读出实验

正式状态见 status.json，结果见 REPORT.zh.md 与 results.json。

- 模型和划分：Qwen3-8B；原始节点 761/199/372，开发/测试路径 420/806。
- 训练：仅原始标签。最后层残差固定 64 维 PCA，线性修正原始分数。
- 对照：完整阅读、开发集选择的简单校准。
- CPU 选择已结束。frozen.json 和 model.pkl 固定后启动测试；自动执行三卡分片及最终汇总。
- 测试结果不得反馈修改当前参数。PCA 仅拟合 fit 节点；验证和测试只应用该变换。
- 原始 MEGA 配对、旧选头和子空间均未用于训练。外部数据没有加入选参。

运行代码位于 /home/gcx/TRIAGE-worktrees/aster-cr-20261108/experiments/ordinary_readout.py。

model.pkl 保存实际运行对象。weights.npz 提供可移植数组：先用 mean/scale 标准化，再减 pca_mean，乘 pca_components.T，除以 sqrt(pca_explained_variance)，最后使用 linear_coefficients：原始分数 + 截距 + 投影特征乘权重。calibration_coefficients 在本轮选择 bias，因此只加截距。协议和软件版本见 protocol.json、environment.json。

已运行的任务由 launch.json/processes.json 记录，避免启动重复进程。若任务失败，排除错误后以 --stage run 从已完成病例和节点缓存续跑；完整结果可由 --stage finalize 从缓存复算。失败状态会保留日志，不能按完成状态引用。

本轮旧 M3 只提供历史背景，不同分母不合并比较。逐例新增错误和纠正及按来源配对区间由统一测试病例计算。
