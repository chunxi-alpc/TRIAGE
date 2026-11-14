# 只运行未跑过的实验（2026-11-14）

用户最新要求：“没跑过的实验才跑”。

已停止 Gemma2 IT、Gemma3、MedGemma 三个整套 E1 流程；各自历史结果及本轮已产出参数、缓存和预测全部保留。共享队列将三项记为用户取消，不记为实验完成，也不自动重启。

当前继续的 GPU 作业：
- GPU0：Gemma-2-9B-Base。
- GPU4：Qwen3-8B-Base。

GPU1、5、6 已释放。空卡不自动填入重复实验。

Gemma2 本轮 NLI500/5500、MedPIC188/467 已完成，直接保留。Gemma3、MedGemma 的 E1 拟合已产生参数，但源任务开发评价和冻结未完成；其未运行的全量 NLI、MedPIC 评价暂列待处理，不切换旧训练协议，不自动接回整套流程。

两个 Base 完成后，CPU 汇总程序自动生成 Base/IT 比较及本次范围的完成记录。E6/E7 停止或待处理单元继续保留实际状态，不列为完成。Codex02 E5 已完成状态保持原记录。

服务器调整记录：/home/gcx/aster-gpu-queue-20261113/only_missing_20261114/。
Codex05 当前报告：/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112/REPORT_ONLY_MISSING_20261114.zh.md。
