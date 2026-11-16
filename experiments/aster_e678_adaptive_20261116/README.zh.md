# ASTER-CR 自适应 M1

所有实验在服务器环境 /home/gcx/gcx261014 运行。主报告位于 reports/aster_e678_adaptive_20261116/REPORT.zh.md。

## CPU 准备

- complete_dev.py：按用户确认，在原20个开发来源补齐缺失配对。人工编写的文本替换保存于 dev_edits.json。
- prepare.py：保留2,540对训练集及80/20/40来源划分，生成七模型独立输入配置；不导入历史学习结果或原始缓存。
- tests/：单元、集成、来源隔离测试。在 CUDA_VISIBLE_DEVICES 为空时运行 unittest discover。
- report.py：生成 JSON/CSV/Markdown 汇总；有实测结果后生成层分布 PNG/PDF。

## GPU 执行

queue_jobs.json 是待提交的统一队列清单，包含一个Qwen冒烟任务与七个依赖该冒烟的检查点任务。已于2026-11-16提交至统一队列 /home/gcx/aster-gpu-queue-20261113/adaptive_20261116；GPU冒烟通过，七个检查点由队列接续执行。

经用户授权，由 /home/gcx/aster-gpu-queue-20261113/scheduler.py 派发。仅允许 GPU 0、1、5、6。run.py 检查物理卡和调度器父进程；不能用独立 --gpu 参数或布尔标志绕过。

顺序：fit/dev原始特征提取 → 自适应M1 → M2/M3 → 基线兼容性审计/匹配基线 → 开发评价 → 冻结 → 测试提取与配对评价。selection_incomplete 是有效诊断终点，保存 queue_done.json，但模型 completion.json 中 complete 为 false。

## 统一接口

- training.fit_group(..., m1_mode="adaptive", adaptive={dev_pairs, dev_features, total_heads, heads_per_layer}, patcher=ShamDonorPatcher(...))
- m1_plan.make_plan(..., mode="adaptive", adaptive={dev_pairs, total_heads, heads_per_layer})
- m1_plan.finish_plan(plan, features, ShamDonorPatcher(...), outdir)
- E6 run.py 的 --m1-mode adaptive 转交新运行器；默认 fixed 保持历史行为。

自适应计算统一落在 adaptive_m1.py / staged.py。旧 patch_queue 拒绝自适应计划，避免使用缺少供体身份的旧缓存。

## 数据和缓存

主结果每模型独立保存。真实和sham缓存使用两个SQLite库；主键包括类别、目标来源/配对/左右输入、供体来源/配对/左右输入、全局头编号。checkpoint和tokenizer/评分身份独立记录，输入提示按原文逐一比较，不计算哈希。

开发精确修补只读取开发记录；M2/M3与截距校准使用受限训练Store；测试提取与评价要求 frozen.json。开发性能标记为模型选择结果。

## 输出

screen.json/csv包含全部类别—头的归因、p和BH q。main_heads.csv包含候选的真实减sham均值、精确符号检验p、BY/BH q、95%区间、有效来源数、确认标志及重采样诊断。source_effects.jsonl/csv保留逐来源真实效应与供体效应序列，donors.json保留供体来源。

m1.json包含四类集合、干扰/证据/共享头、数量比例、层分布和N_eff。BH_sensitivity保存BH确认集合。sensitivity.json/csv包含全部预设阈值变化；每个情景也有独立JSON/CSV。

稳定性是固定第一阶段候选后，来源重采样的BY支持且均值为正的频率；不嵌套第二层置信区间。最终有效头严格使用原数据的BY、95%区间下界及至少16个来源。

未运行模型不填入选头数或性能数值，不改写论文结论。
