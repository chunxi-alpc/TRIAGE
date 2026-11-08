# ASTER 实验索引与运行入口

[主页](../../README.md) · [方法变体](versions.zh.md) · [模块与训练目标](method-clarifications.zh.md)

本页按实验路线列出结果与运行入口。结果与运行状态核对时间：**2026-11-08 16:53:16 UTC**。

## 表示编辑

| 实验 | 规模 | 快照状态 | 结果与实现 |
|---|---|---|---|
| 硬筛选闭式 | 1,290 输入、1,053 配对；测试 610 输入／512 对，432 条自主路径 | 完成 | [报告](../../results/aster_existing_data_20261106/REPORT.zh.md)、[实现说明](../../experiments/aster_closed_form/EXISTING_DATA.zh.md) |
| 连续衰减 | 同一主集，另补 569 历史输入、357 开发／测试 A/P 修补对和 432 条路径 | 完成 | [报告](../../results/aster_soft_angle_20261106/REPORT.zh.md)、[实现](../../experiments/aster_closed_form/soft_angle.py) |
| 未筛选投影与参考 | 同一主集及 569 历史输入；评价节点与配对 | 完成 | [报告](../../results/aster_reference_anchor_20261106/REPORT.zh.md)、[实现](../../experiments/aster_closed_form/reference_anchor.py) |
| 辅助／自身参考 M3 | 200 训练、100 开发；五方法各 7,458 测试路径 | 完成 | [报告](../../results/aster_learnable_m3_20261106/REPORT.zh.md)、[实现](../../experiments/aster_learnable_m3/README.zh.md) |
| P3：N/S/NS 与 online_n | 四组训练；计划评价 522 测试对和 7,458 条路径 | 四组训练完成；paired_ns三轮开发90%、90%、89%，选第1轮；200题迁移试测完成，主测试与配对诊断未完成 | [阶段报告](../../results/aster_m3_followup_20261107/REPORT.zh.md)、[训练实现](../../experiments/aster_m3_followup/training.py) |

### 已完成结果

| 范围 | 完整阅读 | 硬筛选 | 连续衰减 | 未筛选＋均值 | 未筛选＋同事实 | 辅助参考学习M3 |
|---|---:|---:|---:|---:|---:|---:|
| 512测试对四类宏平均 | 96.57% | 97.08% | 97.08% | 77.56% | 96.95% | 此表未评 |
| 610唯一输入正确 | 601 | 603 | 602 | 513 | 601 | 此表未评 |
| 432自主路径正确 | 404 | 401 | 399 | 未评 | 未评 | 此表未评 |
| 7,458自主路径正确 | 7,068 | 7,045 | 未评 | 未评 | 未评 | 7,072 |

表中分别报告配对宏平均、唯一输入正确数和完整路径正确数。同事实直接参考在610输入上与同事实编辑同判；7,458路径中直接参考为6,445，自身参考学习M3为7,068。两阶段参考提示不同。

原全量报告使用2,000次来源bootstrap、种子20261105；P0直接差值使用20,000次、种子20261107。两份记录使用同一批预测、不同重采样设置。


## 分数读出

| 实验 | 数据与评价范围 | 快照状态 | 结果与实现 |
|---|---|---|---|
| ASTER-CR / repair-v2 | 2,782 候选节点、11,512 输入、8,730 对；原始 fit/dev/test 节点 761/199/372、路径 1,712/420/806 | Qwen fit/dev、定位、H2/H3、读出、消融、12组训练及4个开发对照分片完成；配置冻结，test运行；MedGemma与外部评价待完成 | [分析](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md)、[逐例结果](../../results/aster_cr_repair_v2_20261108/README.md)、[运行说明](../../experiments/aster_cr/README.zh.md) |
| 独立普通读出 | 761 原始 fit 节点训练；372 节点、806 路径测试 | 完成 | [报告](../../results/ordinary_readout_20261108/REPORT.zh.md) |

CR 开发配对宏平均：Full 78.26%、CR 79.09%、普通读出79.86%、LoFiT 79.17%；正确路径分别为384/420、383/420、382/420、386/420。CR低于普通读出与LoFiT。开发集用于选参，类别样本量与区间见分析报告。

冻结配置为CR `cr_r16_l0.01`、普通读出 `ordinary_r8`、Full模板；训练对照分别为LoReFT lr0.01、LoFiT lr0.001、LoRA lr0.001、最后一层残差偏移lr0.0001。各方法先通过开发路径下降不超过0.5个百分点的门槛，再比较配对宏平均。LoReFT三个学习率成绩并列，保存选择为0.01。开发对照中直接参考393/420、参考融合392/420；配对宏平均分别为78.59%、78.53%。这些均为开发成绩，test结果待完成。

独立普通读出以末层残差的 64 维 PCA 特征训练线性修正，测试路径与偏置校准均为 767/806，Full 为 768/806；线性读出与校准逐例预测一致。

## 配套分析与对照

| 实验 | 内容 | 快照状态与记录 |
|---|---|---|
| P0 直接统计 | 五方法、7,458 病例、440 来源的差值统计 | 完成；[配对结果](../../results/aster_m3_followup_20261107/p0/paired.json)、[报告](../../results/aster_m3_followup_20261107/p0/REPORT.zh.md) |
| P0 节点补评 | 45,627 标准路径节点；8例×4方法的变化节点记录 | 五方法固定节点各45,627/45,627完成，变化节点32/32；[阶段报告](../../results/aster_m3_followup_20261107/REPORT.zh.md) |
| P1 迁移 | MedGUIDE 校准200题、试测200题；条件扩展计划4,955题 | 200题试测完成：paired_ns 103/200、Full 104/200、bias 107/200；4,955题扩展未触发；[阶段报告](../../results/aster_m3_followup_20261107/REPORT.zh.md) |
| P2 选头 | 三组同层随机头、94候选头、1,152全头，共五组训练 | 五配置训练及200题迁移试测完成，主测试待完成；[头与参数表](../../results/aster_m3_followup_20261107/p2/head_controls.json) |
| M1 双模型诊断 | 头修补、层贡献、残差及两个数据集原文试测 | 完成；[报告](../../results/m1_diagnostics_20261106/REPORT.zh.md) |
| CR 训练对照 | LoReFT、LoFiT、LoRA、最后一层残差偏移，各三个学习率 | 12/12训练及开发评价完成，选参冻结；[运行说明](../../experiments/aster_cr/README.zh.md) |
| CR 外部评价 | NLI4CT 与 MedPIC 原生任务 | 数据构造完成，模型成绩待发布；[分析](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md) |

## 运行入口

服务器为 `gcx@10.29.128.115`，环境为 `/home/gcx/gcx261014`。各实验按原配置和运行目录恢复。

| 实验 | 代码入口 | 运行目录／说明 |
|---|---|---|
| 已完成可学习 M3 | `experiments.aster_learnable_m3.run`、`experiments.aster_learnable_m3.controller` | [完整运行说明](../../results/aster_learnable_m3_20261106/RUN.zh.md) |
| P0–P3 | `experiments.aster_m3_followup.controller` | `/home/gcx/TRIAGE-worktrees/aster-m3-p0-p3-20261107`；[运行说明](../../experiments/aster_m3_followup/README.zh.md) |
| CR / repair-v2 | `experiments.aster_cr.resume_repair` | `/home/gcx/aster_cr_repair_v2_run_20261108`；[恢复说明](../../experiments/aster_cr/README.zh.md) |

当前仅继续CR / repair-v2流程：GPU 0、1、5、6用于Qwen，MedGemma使用6卡；Qwen test完成后依次进行MedGemma及外部评价。旧P0–P3未运行，已有训练与试测结果保留。控制器复用已完成的训练和开发记录，运行稳定后停止持续监控。

## 更新与历史资料

新结果发布时更新本页状态、主页摘要和对应论文结果，保留方法名、数据划分、分母及快照时间。方法变化更新[版本说明](versions.zh.md)与[模块说明](method-clarifications.zh.md)。

[历史总分析](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md) · [历史实验索引](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) · [早期方法与旧文档](archive/README.md)
