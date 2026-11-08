# ASTER近期实验索引与运行入口

[首页](../../README.md) · [中文主稿](manuscript.zh.md) · [版本演变](versions.zh.md)

更新：2026-11-07。索引从闭式三模块开始。更早规则键编辑的协议逐字保存在[整理前实验协议](archive/20261107/docs__ASTER__experiments.zh.md.txt)，其数据、模块编号与约束仅按原版本理解。

## 按版本查结果

| 版本／实验 | 规模与指标 | 状态 | 结果 | 实现 |
|---|---|---|---|---|
| 起始闭式的现成数据整理版 | 1,290输入、1,053配对；测试610输入/512对；432自主路径 | 已完成 | [报告](../../results/aster_existing_data_20261106/REPORT.zh.md)、[选参](../../results/aster_existing_data_20261106/results/selection.json) | [现成数据说明](../../experiments/aster_closed_form/EXISTING_DATA.zh.md) |
| 连续夹角衰减 | 同一主集、569历史输入、357开发／测试A/P修补对、432路径 | 已完成 | [报告](../../results/aster_soft_angle_20261106/REPORT.zh.md) | [soft_angle.py](../../experiments/aster_closed_form/soft_angle.py) |
| 未筛选方向与参考 | 同一主集及569历史输入；无自主路径 | 已完成 | [报告](../../results/aster_reference_anchor_20261106/REPORT.zh.md) | [reference_anchor.py](../../experiments/aster_closed_form/reference_anchor.py) |
| 辅助／自身参考可学习M3 | 200训练、100开发；五方法各7,458测试路径 | 已完成 | [报告](../../results/aster_learnable_m3_20261106/REPORT.zh.md)、[计数和区间](../../results/aster_learnable_m3_20261106/comparison.json) | [代码说明](../../experiments/aster_learnable_m3/README.zh.md) |
| P0直接统计 | 同一五方法预测、7,458病例、440来源 | 已完成 | [配对结果](../../results/aster_m3_followup_20261107/p0/paired.json)、[解释](../../results/aster_m3_followup_20261107/p0/REPORT.zh.md) | [p0.py](../../experiments/aster_m3_followup/p0.py) |
| P0固定节点与变化节点 | 45,627标准路径节点；8例×4方法=32条变化节点记录 | 部分完成 | [阶段报告](../../results/aster_m3_followup_20261107/REPORT.zh.md)、[进度快照](../../results/aster_documentation_20261107/progress.json) | [evaluation.py](../../experiments/aster_m3_followup/evaluation.py) |
| P1迁移 | MedGUIDE校准200、试测200；条件扩展4,955 | 准备完成，本轮成绩待完成 | [配置](../../results/aster_m3_followup_20261107/configuration.json) | [scoring.py](../../experiments/aster_m3_followup/scoring.py) |
| P2选头 | 三个同层随机头、94候选头、1,152全头；五组训练 | 头和基准备完成，正式训练待完成 | [头与参数表](../../results/aster_m3_followup_20261107/p2/head_controls.json) | [bases.py](../../experiments/aster_m3_followup/bases.py) |
| P3配对训练 | N/S/NS及online_n；四组训练、522测试对和7,458路径 | paired_ns训练中，完整结果待完成 | [阶段报告](../../results/aster_m3_followup_20261107/REPORT.zh.md) | [training.py](../../experiments/aster_m3_followup/training.py) |
| M1双模型诊断 | 各64对头、64对层贡献、12对残差；两个数据集各200例 | 已完成；作为分析，不是新方法版本 | [报告](../../results/m1_diagnostics_20261106/REPORT.zh.md) | [代码](../../experiments/m1_diagnostics/README.zh.md) |

## 最新进度如何理解

2026-11-07 17:41（北京时间）只读快照：P0分片0完成1,997项，分片1记录800/1,996项；变化节点16/32条记录已保存；`paired_ns`处于第3轮。九个新配置尚无正式`complete.json`。微型端到端验证目录中的完成记录不属于正式训练。完整阶段报告在后续实验保存完成后更新，本次不等待训练结束。

阶段报告的0/7,458、0/200等指已保存记录数不足完整分母，不表示成绩为0。4,955题是条件扩展计划规模，未触发或未完成时不计作已评测。P0直接统计已经完成，固定节点和变化节点补评分别报告状态。

快照保存于[progress.json](../../results/aster_documentation_20261107/progress.json)。该文件记录观察时间和原运行目录，是本次整理的状态依据，不作为持续监控程序。

## 已完成结果的对应关系

| 范围 | 完整阅读 | 硬筛选 | 连续衰减 | 未筛选＋均值 | 未筛选＋同事实 | 辅助参考学习M3 |
|---|---:|---:|---:|---:|---:|---:|
| 512测试对四类宏平均 | 96.57% | 97.08% | 97.08% | 77.56% | 96.95% | 此表未评 |
| 610唯一输入正确 | 601 | 603 | 602 | 513 | 601 | 此表未评 |
| 432自主路径正确 | 404 | 401 | 399 | 未评 | 未评 | 此表未评 |
| 7,458自主路径正确 | 7,068 | 7,045 | 未评 | 未评 | 未评 | 7,072 |

四类宏平均、唯一输入准确率和自主路径准确率不能横向混成同一指标。同事实直接参考在610输入上与同事实编辑同判；7,458路径中直接参考为6,445，自身参考学习M3为7,068。两阶段参考提示不同。

原全量报告使用2,000次来源bootstrap、种子20261105；P0直接差值使用20,000次、种子20261107。两份记录复用同一批预测，区间差异不是新增运行或样本变化。

## 服务器代码与运行目录

所有运行在`gcx@10.29.128.115`，先激活`/home/gcx/gcx261014`。本页提供复现入口，本次整理未执行以下训练或评测命令。

| 用途 | 现有目录 |
|---|---|
| 原主工作区与已完成M3数据／权重 | `/home/gcx/TRIAGE` |
| 已完成M3结果 | `/home/gcx/TRIAGE/results/aster_learnable_m3_20261106` |
| 既有M1/M2源产物 | `/home/gcx/aster_existing_data_20261106` |
| 连续衰减／参考实验 | `/home/gcx/aster_soft_angle_20261106`、`/home/gcx/aster_reference_anchor_20261106` |
| P0–P3运行工作区 | `/home/gcx/TRIAGE-worktrees/aster-m3-p0-p3-20261107` |
| 本次独立整理工作区 | `/home/gcx/TRIAGE-worktrees/aster-version-manuscript-20261107` |

已完成M3代码入口为`experiments.aster_learnable_m3.run`，阶段参数为`--stage prepare/verify/train/evaluate`，训练模式`--mode reference/self`。主控制器是`experiments.aster_learnable_m3.controller`；运行配置、选择检查点及五方法完整预测已合入仓库。[详细运行说明](../../results/aster_learnable_m3_20261106/RUN.zh.md)

最新P0–P3入口：

```bash
ssh gcx@10.29.128.115
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-m3-p0-p3-20261107
python -m experiments.aster_m3_followup.controller
```

控制器使用现有记录续跑，原后台实验稳定运行时无需重复启动。已授权实验仅用GPU 0、1；新一轮启动前按用户要求汇报预计显存、规模和时长。详细过程写入对应运行目录日志。

P0–P3分支为`codex/aster-m3-p0-p3-20261107`，本次合入其`ca90006`及祖先中的代码、配置和阶段结果。后续该分支自动发布的新结果仍按同版本接入，勿以本次文档快照覆盖其运行目录。

## 文档后续更新位置

新结果形成后，更新本页状态、主稿第7–8节和首页摘要，并保留检查点名称、方法名与原分母。方法公式随实际实现变化更新；历史公式和数值继续留在[版本演变](versions.zh.md)。M1诊断进入机制分析；更早单层表示方法和CARE等仍由[历史索引](archive/README.md)进入。

[历史总分析](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md)与[30阶段、60份报告索引](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md)继续保留。其“30阶段”不代表30次独立验证。


## repair-v2 与普通读出（2026-11-08）

[分析报告](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md)和[阶段快照](../../results/aster_cr_repair_v2_20261108/README.md)汇总本次数据修复、Qwen P0/H2/H3、读出和消融。使用 2,782 个候选节点、8,730 对变体；原始节点分母保持 761/199/372，原始路径分母保持 1,712/420/806。四类训练对照、修复版 test、MedGemma 正式确认与外部模型评价尚未完成。

[独立普通读出](../../results/ordinary_readout_20261108/REPORT.zh.md)已经完成。仅用原始节点监督，测试路径 Full 768/806，偏置校准和线性读出均为 767/806，二者逐例预测一致。它与修复版开发集配对实验分别列出。
