# ASTER-CR：对消，而非擦除

**[当前中文论文正文](paper_story_latest.zh.md)**，研究目标为在保留完整输入时修正指南条件判断的干扰响应。它与docs/ASTER中的历史**表示编辑**研究属于不同方法路线，数值不能直接混用。

## 四类基线与五模型三数据集

第4.3节是**必测完整矩阵**：五个模型（Qwen3-8B、Llama-3.1-8B-Instruct、Gemma-2-9B-IT、Gemma-3-27B-IT、MedGemma-27B-Text-IT）×三个主数据集（MEGA-CDP、NLI4CT-2024、MedPIC）×13种外部基线及3种通用控制，合计240个基线比较单元。再加原模型、ASTER-CR线性及非线性，合计285个实验单元。两个27B模型不只作为附录或模拟。

[五模型全量方案与运行核查](meeting/full_factorial_20261110/README.zh.md) · [完整285单元状态表](meeting/full_factorial_20261110/matrix_status_285.csv) · [结构化方法与指标](meeting/full_factorial_20261110/matrix_spec.json) · [模型清单与比较目的](model_selection.zh.md)

当前只有Qwen3-8B部分真实结果及其既有复算；**尚未完成240个同协议基线实测**。未运行方法不能填写猜测值、文献异协议数字或预设模拟结果。MedPIC逐例记录尚未完整归档。

## 结果入口

论文第4.5节为MEGA受控结果，第4.6节是**仅Qwen3-8B**在三个完整任务上的已有结果；第4.7、4.8节分别报告NLI原生指标及MedPIC历史汇总。附录B含完整指标和全量实验协议，附录C含实测、消融及机制验证，附录D单独收纳补充任务与合成／预设模拟。

[旧会议报告](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/meeting_brief.zh.md) · [现有受控实验](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/real_completion/) · [NLI原生指标复算](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/) · [痕迹机制验证](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/trace_validation/) · [五模型合成模拟（非真实结果）](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/simulated_study/)。

正确的结果等级是：已执行且可复核的真实模型运行、已有但缺逐例归档的历史总结、数学解析控制、真正拟合的合成机制试验、预设分数模拟。任何比较表都要按此顺序区分来源。
