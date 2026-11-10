# 方法与实验总表

[项目首页](../../README.md) · [方法与公式](versions.zh.md) · [当前论文](../../reports/aster_cancel_story_20261109/paper_story_latest.zh.md)

核对日期：2026-11-10。本表合并服务器实验目录、实验分支和主分支已有记录，列出方法、规模、状态、结论及代码和结果位置。

## 方法演变

| 阶段 | 主要做法 | 后续变化 |
|---|---|---|
| CARE / ACRE | 定位证据、判断条件，探索适用性方向和参考切换 | 解析与证据定位有进展；复杂干预未稳定优于直接偏置 |
| ASTER早期表示干预 | 注意力、动作分数、动态融合、残差及结构修复 | 发展为头定位、配对子空间和表示编辑 |
| 闭式与可学习M3 | 硬筛选 → 连续衰减 → 同事实参考 → 映射与门控 → N/S/NS | 固定几何性质未保证任务收益，转向学习任务所需的修正 |
| ASTER-CR | 冻结模型，从选中头提取特征，学习线性或非线性分数修正 | 增加证据保护、对照、五模型和跨任务研究 |

## 近期分数修正实验

“近期归档”和“迁移归档”分别指本页末尾两个压缩包；源码、配置、报告及实际保存的结果收入包内，来源路径保存在包内说明中。

| 实验与方法 | 评价规模 | 完成状态与主要结果 | 材料 |
|---|---|---|---|
| CR / repair-v2定位、支持空间、读出与训练对照 | 11,512输入、8,730对；12组训练；开发420路径、原始测试806路径 | 已有开发、测试及后续优化记录。开发CR383/420，Full384/420，普通读出382/420，LoFiT386/420 | [原阶段分析](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md)、近期归档 |
| 独立普通读出与偏置校准 | 原始372节点、806路径 | 完成；二者均767/806，预测相同；Full768/806 | [代码](../../experiments/ordinary_readout.py)、[报告](../../results/ordinary_readout_20261108/REPORT.zh.md) |
| 配对拟合、证据保护、混合痕迹与秩对照 | MEGA受控2,293个A/P对、40个C/F对、806路径 | 已有实测；证据保护版本干扰方差降低约4.10%、证据响应保留约99.60%，路径767/806 | 近期归档：paired_fit、evidence_preserving、real_completion |
| 线性、非线性、分数条件修正与集成优化 | 开发选择、测试扩展及全量任务；另有原错误子集重测 | 实测已保存；完整测试与仅原错误子集分别记录，后者不代表全体准确率 | 近期归档：optimization、test_expansion、primary_full |
| Qwen全量MEGA与NLI冻结迁移 | 7,458动态路径；5,500条NLI声明 | 已完成归档运行；MEGA原模型7,037、线性7,034、平衡集成7,041；NLI对应4,073、4,031、4,048 | [论文4.6节](../../reports/aster_cancel_story_20261109/paper_story_latest.zh.md#46-三个完整任务总体冻结迁移的效用)、近期归档 |
| MedPIC冻结迁移与集合解码 | 官方467题、2,335选项；另有原错误题重测 | 已完成；原模型98/467、线性70/467、两种优化版均78/467；467题×4方法的整题结果及2,335选项分数已补录，后续子集另列 | 迁移归档：medpic；[历史汇总](../../reports/aster_cancel_story_20261109/paper_story_latest.zh.md#48-medpic证据保持与集合正确性) |
| 痕迹预测、H2及机制验证 | 原划分上的预测、方向、缩放和消融对照 | 实测与解析对照已保存；固定比例缩放为数学控制；生成数据上的真实拟合与预设分数模拟单独保存 | 近期归档：trace、trace_validation、toy、simulated_study |
| 五模型 × 三个旧任务 | Qwen3-8B、Llama-3.1-8B、Gemma-2-9B、Gemma-3-27B、MedGemma-27B；MEGA节点372及806路径、NLI原声明500、生成三段论200 | 15组均完成；干扰方差降低3.96%—27.08%，证据响应保留99.55%—100.94%；准确率有升有降 | [15组结果](../../results/aster_five_models_20261109.csv)、迁移归档：five_models |
| 跨任务A/B/C：冻结、校准与任务内适配 | NLI原声明500；MedPIC另分拟合／开发／测试，测试188题 | 完成；适配可降低干扰响应，未取得稳定准确率收益；不与全467题冻结迁移混排 | 迁移归档：cross_task_transfer |
| G0/G1/G2：截距、新增标签改变监督与删除保护 | NLI500声明、864改变对；MedPIC188题、110选项改变对 | 完成；NLI G2干扰方差降低92.01%，证据响应仅保留35.13%，正确366/500，原模型378/500；MedPIC部分变化仅来自截距 | 迁移归档：cross_task_followup |
| 新痕迹修复 | 重新构造并收集拟合／开发／最终数据 | 代码已补录，快照为collecting_fit；无最终结论 | [代码与启动状态](../../experiments/aster_cr/trace_repair_20261110.zip) |
| 五模型 × 三主任务 × 19方法 | 13项外部基线、3项控制、原模型、CR线性与非线性，共285单元 | 方案和调度代码已有，统一协议未完成；三主任务为MEGA、NLI4CT、MedPIC | [方案与状态矩阵](../../reports/aster_cancel_story_20261109/meeting/full_factorial_20261110/README.zh.md) |

五模型MEGA路径中，Llama与Gemma-2相对原模型提高，但都低于简单校准；其余模型也未显示稳定路径收益。

五模型旧实验按模型与任务分别拟合，不是在MEGA拟合后冻结迁移；其NLI保持关系统计范围也与新协议不同。因此15组完成与285项尚未完成同时成立。MEGA节点和完整路径、NLI原声明和全部声明、MedPIC选项与整题集合各保留自己的分母。

## 表示编辑与配套实验

| 方法或阶段 | 规模与状态 | 结论和入口 |
|---|---|---|
| 早期E0–E8、P0–P3、动作质量、动态融合、稀疏编辑及P1–P6 | 按原阶段记录，含部分完成与运行快照 | [30阶段、60报告历史索引](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md)逐项保留，不是30次独立验证 |
| A/B/C、头和层迁移、长输入、规则读取、旧M3及结构修复 | 各项结果或协议已保存 | 同一[历史索引](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md)收录工程修复和局部诊断 |
| 旧可学习表示与参考交换 | 官方7,458路径及参考交换100组等 | 旧no_js为6,653/7,458，提示与后续Full不同；参考交换未支持正确参考独有作用，详见历史索引 |
| Gemma层／头与四输入 | 100组确认完成 | 直接盲读94、整层91、零干预85、选头85；[报告与代码](../../experiments/gemma3_head_pilot/README.md) |
| 修订遮挡、关键事实及三数据集 | 2,994条遮挡评分；MedPAIR、MedGUIDE、MedEinst补充分析完成 | [遮挡分析](../../reports/semantic_masking_followup_20261105/README.md)、[三数据集](../../reports/three_dataset_analysis_20261105/README.md) |
| 闭式硬筛选 | 1,290输入、1,053对；610测试输入、512对、432路径，完成 | 配对宏平均97.08%，Full96.57%；路径401/432，Full404；[代码](../../experiments/aster_closed_form/EXISTING_DATA.zh.md)、[结果](../../results/aster_existing_data_20261106/REPORT.zh.md) |
| 连续衰减 | 同一主数据、569历史输入及432路径，完成 | 宏平均97.08%，路径399/432；[实现](../../experiments/aster_closed_form/soft_angle.py)、[结果](../../results/aster_soft_angle_20261106/REPORT.zh.md) |
| 未筛选方向与同事实参考 | 同一主数据及569历史输入，节点／配对评价完成 | 同事实编辑与直接参考在610输入上同判；[实现](../../experiments/aster_closed_form/reference_anchor.py)、[结果](../../results/aster_reference_anchor_20261106/REPORT.zh.md) |
| 辅助／自身参考可学习M3 | 五方法各7,458路径，完成 | 辅助参考7,072，Full7,068；差值95%来源区间含零；[代码](../../experiments/aster_learnable_m3/README.zh.md)、[结果](../../results/aster_learnable_m3_20261106/REPORT.zh.md) |
| 11月7日P0–P3：节点、选头及N/S/NS | 固定节点各45,627、变化节点32已补评；9配置训练及200题迁移完成 | paired_ns开发90%、90%、89%，选第1轮；迁移103/200，Full104/200；7,458路径与522对主诊断未完成，4,955题扩展未触发；[代码](../../experiments/aster_m3_followup/README.zh.md)、[阶段报告](../../results/aster_m3_followup_20261107/REPORT.zh.md) |
| M1双模型诊断 | 头修补、层贡献、残差及原文试测完成 | [代码](../../experiments/m1_diagnostics/README.zh.md)、[结果](../../results/m1_diagnostics_20261106/REPORT.zh.md) |

## CARE、ACRE与早期探索

既有文件保持原位；本次补入的旧分支源码、结果和报告统一放在[早期研究归档](../../experiments/legacy_research_20261110.zip)。

| 探索 | 状态、结果及用途 | 来源 |
|---|---|---|
| TRIAGE最早三模块与24例诊断 | 完成；无规则15/24，完整指南13/24，完整TRIAGE15/24，整体无净提升 | [历史说明](../../TRIAGE_MVP.md)、[诊断](../../results/algorithm_diagnostic/summary.json) |
| CARE V1–V3 | 结果已补录；V2第二批180/180，V3新规则116/192，直接基线159/192；设置不同，不据V2推断泛化 | 早期研究归档 |
| CARE V4–V6 | 结构化证据、解析修复、模型组合及证据定位；V6旧192例78.65%、新增100例73.00% | [V4](../../experiments/care_module1_v4/README.md)、[V5](../../experiments/care_module1_v5/README.zh.md)、[V6](../../experiments/care_module1_v6/README.md) |
| CARE O1/O2/O3、B1–B4及有符号诊断 | 历史实验完成；未建立学习子空间优于直接偏置的证据 | [三项观察](../../experiments/care_qwen3_4b_base_20261016/results/observations_summary.md)、[有符号诊断](../../experiments/care_module2_signed_20261019/README.md) |
| CARE可靠性、steering及因果注意力 | 已保存真实运行；可靠性未显示子空间额外收益；规则执行177/192，结构提示158/192，加干预159/192 | 早期研究归档 |
| A1语义判定、V8.1及U示例 | A1新72例192/216，A2为210/216；V8.1新集189/216与190/216但后者成本翻倍；U示例仅开发集改善，U识别下降 | [冻结接口](../../experiments/care_module1_semantic_v8/MODULE1_A1_FREEZE.zh.md)、早期研究归档 |
| N1–N5适用性方向后续 | N2路由0/8、文本4/8；N3路由20/32、均值25/32、文本27/32；N5路由9/24、文本11/24；N1/N4未触发 | 早期研究归档 |
| TrialGPT A1外部条款 | 1,015条全量；可评价960条中722正确（75.21%），另存U错误分析 | 早期研究归档 |
| 适用性子空间、在线读取及适用性决策 | 前两项为CPU小模型实验：24提示、6方法×4剂量；24提示×9配置；决策后续96提示 | 子空间各配置24/48、配对0/24；在线注意力变化未改善最终选择；决策优化49/96、均值51、随机最佳52；早期研究归档 |
| SEKA标准任务及临床迁移 | 标准任务多个5,000例运行完成；临床开发16例×7方法；goal为方案，execution有实测 | CounterFact Qwen8B编辑成功率98.02%，零强度51.50%；临床注意力0.48125，文本0.603125，融合0.58125；早期研究归档 |
| ACRE先导、同源参考及目标切换 | 代码、结果与停止结论已保留；未执行提案按方案记录 | [文档](../ACRE/README.md)、[同源诊断](../../results/acre_pilot/oracle_switch_diagnostic/README.md) |

## 本次补齐的归档

| 仓库文件 | 内容 |
|---|---|
| [legacy_research_20261110.zip](../../experiments/legacy_research_20261110.zip) | 早期CARE、适用性、SEKA等独立支线的源码、配置、原报告和已有结果 |
| [aster_cr_research_20261109.zip](../../results/aster_cr_research_20261109.zip) | CR实验分支的配对拟合、保护、优化、正式测试、错误重测、机制验证、模拟及相关源码 |
| [aster_cr_transfer_20261110.zip](../../results/aster_cr_transfer_20261110.zip) | 五模型旧15组、MedPIC、跨任务A/B/C及G0/G1/G2的源码和已有结果 |
| [trace_repair_20261110.zip](../../experiments/aster_cr/trace_repair_20261110.zip) | 尚未完成的新修复代码、配置与启动状态 |
| [aster_five_models_20261109.csv](../../results/aster_five_models_20261109.csv) | 五模型15组节点／声明／三段论正确数、MEGA路径和校准对照、干扰方差降低率及证据响应保留率，无需解压 |

包内保留目录结构，附来源与使用说明。模型原权重、可重算的大表示缓存和重复日志留在原服务器，不逐份复制到Git。材料收录与实验完成分别记录。

服务器为gcx@10.29.128.115，环境为/home/gcx/gcx261014。本次未启动实验；恢复某项实验先看其原说明。后续运行使用GPU 0、1，启动前另报规模、显存与预计时长。

[历史30阶段索引](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) · [早期文稿](archive/README.md) · [历史分支](../BRANCHES.md)
