# ASTER-CR 自适应 M1

当前范围：七个检查点各自执行自适应M1、M2/M3、开发与测试评价、阈值敏感性和选头结构诊断。按用户2026-11-16最新要求，取消全部置信区间及其选头条件，取消固定51头基线重跑和对照。

保留E1的2,540对训练数据和80/20/40来源划分。原20个开发来源补充43对后，四类各覆盖20个来源。新增配对与修改记录见reports/aster_e678_adaptive_20261116/development。

梯度候选阶段每类每训练来源取一个固定代表配对，共296个唯一左输入；覆盖全部头并执行2,000次来源级置换，统一BH q≤0.20。开发精确修补使用五个不同来源供体，最终要求BY q≤0.05、平均超额干预效应>0、至少16个有效开发来源。BH结果单列。

M2秩上限16、M3正则化0.01、证据预算0.2923349及自然节点截距校准保持原配置。来源重采样保留用于选头稳定性；性能只报告点估计。

敏感性：梯度q=0.10/0.20/0.30，确认q=0.01/0.05/0.10，供体3/5/10。各场景共用原始修补缓存。未根据测试表现调整阈值。

统一实现为adaptive_m1.py和staged.py；training.fit_group的adaptive模式和m1_plan的adaptive模式均调用该实现。旧fixed模式保留供历史代码使用，本次运行器不调用它。每模型独立的真实/sham SQLite缓存包含目标和供体来源、配对、输入、头编号。

实验服务器工作目录：/home/gcx/aster-e678-adaptive-20261116；环境：/home/gcx/gcx261014。统一队列：/home/gcx/aster-gpu-queue-20261113/adaptive_20261116。GPU限制为0、1、5、6，模型入口检查统一调度器父进程。

执行顺序：fit/dev原始特征 → 自适应M1 → M2/M3 → 开发评价 → 参数冻结 → 测试点估计。干扰或证据集合为空时保存selection_incomplete并停止后续拟合。已提取特征可断点续用。

主要输出：screen.json/csv、per_head_statistics.json/csv、source_effects.jsonl/csv、donors.json、m1.json、sensitivity.json/csv、point_estimates.csv、REPORT.zh.md。模型任务结束后finalize.py自动汇总，并向GitHub分支aster-adaptive-m1-20261116推送报告。

GPU 空闲时，可由统一队列启动 cooperative_extract 辅助进程，共享同一检查点的未完成原始特征提取。每条输入独立加锁，分数写入加锁，完成记录用于续跑；主进程在开发特征齐备后执行 M1。测试输入仍须在参数冻结后提取。
