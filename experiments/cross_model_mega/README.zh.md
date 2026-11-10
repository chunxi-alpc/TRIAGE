# 实验 B：跨模型 MEGA 复现

在服务器激活 /home/gcx/gcx261014 后，从仓库根目录运行：

    python -m experiments.cross_model_mega.pipeline

pipeline.py 使用 GPU0: Llama、GPU1: Gemma2、GPU5: Gemma3、GPU6: MedGemma 四模型并行；fit_controlled.py 复用每个模型各自已经完成的 fit80/dev20/test40 激活、51个独立选择的头、S/N、标准化和线性参数，仅拟合一个同维随机16维控制（种子20261111）。相同 ridge=0.01、tau=0.1，每个模型使用该模型 fit A/P RMS 定义 C/F 逐对预算。截距仅用自然 fit 标签拟合。

固定输入来自 /home/gcx/aster_five_models_20261109/data/mega/{views,pairs,path_cases}.jsonl 和 /home/gcx/aster_next_20261109/inventory/official_all_path_cases.jsonl。模型路径记录于 models.json，各模型缓存来自 /home/gcx/aster_five_models_20261109/results/<model>/mega。代码自身依赖的 engine 与 CPU 拟合帮助函数已纳入此模块；通用模型缓存实现与 MEGA prompt 构造来自仓库 experiments/m1_diagnostics/engine.py 与 experiments/aster_closed_form/data.py。

run_paths.py 逐病例实际遍历四方法动态分支，四方法共享节点评分，每个模型独立前向；官方7458条路径与其45627个金标准节点全部评价。缺失或超出模型原生上下文的节点记录 unavailable，保留完整分母，不截断输入。旧受控 raw/bias/ASTER 路径直接复用，只对随机控制需要的新分支补前向。

前缀加速仅复用原生4096预填分块的完整边界，以保持 BF16 计算与原实现一致。最初任意公共前缀切分的试运行因产生0.31 logit数值差异而废弃，原验证记录保留为 prefix_validation_superseded_lcp.json；正式路径全部使用整分块协议，新首病例验证分数差为0。此改动修复计算实现，没有改变模型、训练、预算或测试输入。

GPU缓存与可恢复日志保存在 /home/gcx/aster_cross_model_mega_20261111/<model>/shard0；自动导出位置是 reports/aster_cancel_story_20261109/meeting/cross_model_mega。每个模型保存参数、配置、真实逐例输出、来源级数据、运行日志和汇总。summarize.py 自动计算来源簇配对bootstrap（2000次）95%区间并生成统计图。

AP prediction_r2 指受 C/F 约束的部署修正对 AP 差分的解释量，与其他实验的无约束trace探针R²不同。自然节点采用单节点标签正确数；动态路径要求完整路径与叶子均正确。不同模型不合并logit。checkpoint的不可变上游revision在本地未记录，配置明确标注 unrecorded；记录代码的base git SHA与checkpoint路径。

每模型入口使用服务器fcntl文件锁，避免同模型多个恢复进程同时写入。此次实际启动时原GPU0/1控制器保留，GPU5/6额外接管27B；原控制器排队到27B时会等待该模型锁，随后直接复用已完成结果而不重复前向。
