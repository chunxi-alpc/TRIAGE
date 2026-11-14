# E6、E7、E8 补充实验

所有运行均在服务器 /home/gcx/aster-e678-gaps-20261112，使用 /home/gcx/gcx261014 环境。当前协议为 E1：2,540 对、80 个拟合来源、20 个开发来源、51 个头、16 维、λ=0.01、δ=0.2923349。六个新检查点分别拟合 M1/M2/M3；Qwen IT 的已完成结果直接复用。

## 统一调度（2026-11-13）

全部剩余任务已移交 /home/gcx/aster-gpu-queue-20261113，任务清单为 codex05_jobs.json。旧 CPU 等待控制器已停止，launch.py 和 wait_then_launch.py 已禁用独立派发；旧释放文件仅保留为历史记录。

每个模型的入口是 python -m experiments.aster_e678.unified_worker --model MODEL。它只执行一个检查点，读取统一调度设置的 CUDA_VISIBLE_DEVICES，只接受 0、1、5、6 中的一张卡，不自行选择GPU，不启动其他模型或子队列。

六个 GPU 任务完成后，CPU任务 python -m experiments.aster_e678.finalize 自动生成两个 Base/IT 比较、完整状态报告，并更新 docs/ASTER/experiments.zh.md。最终完成标记为 unified_summary_complete.json；失败模型不会标记完成。

## 数据与产物

prepare.py 已建立数据引用和同检查点原始缓存索引。official_targets.py 准备 NLI 5,500 条和 MedPIC 467 题。run.py 依次执行缺失特征、独立 M1/M2/M3、源任务开发评价、参数冻结、受控测试、冻结迁移和官方 MEGA 7,458 路径。Base 不额外运行跨任务评价。

各检查点保存独立 configuration.json、模型参数、frozen.json、逐例预测、统计及 logs/unified_run.log。每个任务结束时更新当前状态；可从已有本轮产物恢复。旧五模型结果不覆盖，旧修正参数不跨模型复用。目标标签只用于评价。

Gemma2 Base/IT优先；Qwen Base/IT作为第二优先，Qwen IT复用E1。检查点差异不解释为某个特定后训练阶段的因果效应。论文正文不修改，不执行哈希检查。
