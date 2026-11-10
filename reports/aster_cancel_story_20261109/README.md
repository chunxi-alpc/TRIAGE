# ASTER-CR：对消，而非擦除

**[当前中文论文正文](paper_story_latest.zh.md)**，研究目标为在保留完整输入时修正指南条件判断的干扰响应。它与docs/ASTER中的历史**表示编辑**研究属于不同方法路线，数值不能直接混用。

## 四类基线与五模型三数据集

第4.3节是**必测完整矩阵**：五个模型（Qwen3-8B、Llama-3.1-8B-Instruct、Gemma-2-9B-IT、Gemma-3-27B-IT、MedGemma-27B-Text-IT）×三个主数据集（MEGA-CDP、NLI4CT-2024、MedPIC）×13种外部基线及3种通用控制，合计240个基线比较单元。再加原模型、ASTER-CR线性及非线性，合计285个实验单元。两个27B模型不只作为附录或模拟。

[五模型全量方案与运行核查](meeting/full_factorial_20261110/README.zh.md) · [完整285单元状态表](meeting/full_factorial_20261110/matrix_status_285.csv) · [结构化方法与指标](meeting/full_factorial_20261110/matrix_spec.json) · [模型清单与比较目的](model_selection.zh.md)

当前已有Qwen3-8B全量与受控结果，以及五模型在MEGA、NLI4CT和程序生成三段论上的15组旧实验；它们与上述统一协议的任务、拟合方式和基线范围不同，**尚未完成240个同协议基线实测及285项完整矩阵**。MedPIC全467题逐例结果已补入[迁移研究归档](../../results/aster_cr_transfer_20261110.zip)，包含四种方法各467条整题记录、2,335条选项分数及配对结果。全部状态见[实验总表](../../docs/ASTER/experiments.zh.md)。

**示例填值与真实运行必须分开。** [论文附录D.5](paper_story_latest.zh.md#d5-五模型与四类基线的预设填表示例非模型实测)已直接预填三项主任务的五模型×19方法数值表；[全部十张预填表](meeting/full_factorial_20261110/prefill_example_tables.zh.md)、[285条预设示例数据](meeting/full_factorial_20261110/prefill_example_285_long.csv)和[假设参数](meeting/full_factorial_20261110/prefill_scenario_parameters.json)仅用于版式和指标检查，均**不是新增实测**。待运行状态以独立的[matrix_status_285.csv](meeting/full_factorial_20261110/matrix_status_285.csv)为准。

## 结果入口

论文第4.5节为MEGA受控结果，第4.6节是**仅Qwen3-8B**在三个完整任务上的已有结果；第4.7、4.8节分别报告NLI原生指标及MedPIC既有结果，MedPIC逐例文件此次已补录。附录B含完整指标和全量实验协议，附录C含实测、消融及机制验证，附录D单独收纳补充任务与合成／预设模拟。

[旧会议报告](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/meeting_brief.zh.md) · [现有受控实验](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/real_completion/) · [NLI原生指标复算](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/) · [痕迹机制验证](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/trace_validation/) · [五模型合成模拟（非真实结果）](https://github.com/chunxi-alpc/TRIAGE/tree/aster-cr-20261108/reports/aster_cancel_story_20261109/simulated_study/)。

结果分别保留真实模型运行、数学解析控制、实际拟合的合成机制试验和预设分数模拟的来源。MedPIC历史全量结果的逐例文件已补录；部分历史子集探索仍按各自评价范围记录。
