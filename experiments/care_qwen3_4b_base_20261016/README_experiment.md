# CARE 三项观察：真实 Qwen3-4B-Base 先导实验

已完成模型执行、数据保存、统计分析和10组论文图（每组PNG、PDF、SVG，含3组临床补充图）。不是仅有方案；所有模型观测来自真实预训练权重。

**科学范围：临床O1为真实规则与模拟患者；O2/O3仍为虚构规则机制实验。临床O1已补齐，见下方更新。**

- 结果报告：results/observations_summary.md
- 原始输出与指标：results/obs1、results/obs2、results/obs3
- 论文图：figures/，300dpi PNG及矢量PDF/SVG；图注见figures/CAPTIONS.md
- 完整环境与来源：results/manifest.json、results/logs/environment.txt、results/source_audit.json
- 自定义执行与分析代码：scripts/run_experiments.py、scripts/analyze_results.py
- 官方SEKA及PASTA源码：external/SEKA（未修改，准确提交记录在manifest）
- 权重不放入交付压缩包，保存在服务器models/Qwen3-4B-Base；manifest记录三份权重SHA256并与官方HF逐一匹配。
- 复现入口：REPRODUCE.sh；仅在服务器指定环境运行。
- 烟雾测试单独存于results/smoke_run，不进入正式统计。
- 原始附件说明保留于README_input.md和协议文档。

实际模型：Qwen/Qwen3-4B-Base，BF16，4022468096参数；36层、32个查询头、8个KV头、128维每头。无替换模型或随机权重。
代码来源：https://github.com/waylonli/SEKA
模型来源：https://huggingface.co/Qwen/Qwen3-4B-Base





## 临床观察一更新

合格153例（适用78，不适用75），完整双侧冲突配对25对。详见results/observations_summary.md和results/obs1_clinical/clinical_report.md。新增3组PNG/PDF/SVG图。原始先导结果保留，不能将O2/O3改称临床验证。复现：REPRODUCE_CLINICAL.sh。
