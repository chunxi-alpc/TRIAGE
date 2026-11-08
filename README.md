# ASTER

<!-- repair-v2-results:start -->
## 新实验进展（2026-11-08）

repair-v2 已完成 2,782 个候选节点、11,512 个输入、8,730 对变体的数据修复，以及 Qwen 开发集诊断、读出和消融。训练对照正在 GPU 0、1、5、6 上运行；修复版 test、MedGemma 正式确认和外部模型评价尚未完成。

开发配对宏平均：Full **78.26%**，预选 repair-v2 读出 **79.09%**，普通读出 **79.86%**；完整开发路径分别为 **384/420、383/420、382/420**。主读出尚未超过普通读出，差值区间见分析。P/C/F 独立节点仍少，分别列数量与区间。

独立普通读出实验已完成：Full 测试路径 **768/806**，偏置校准与普通线性读出均为 **767/806**；两者逐节点、逐路径预测完全一致，没有获得超出简单校准的额外收益。

[新结果与分析](reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md) · [结果与逐例明细](results/aster_cr_repair_v2_20261108/README.md) · [数据修复报告](reports/aster_cr_repair_v2_20261108/DATA_REPAIR.zh.md) · [普通读出完整结果](results/ordinary_readout_20261108/REPORT.zh.md) · [执行说明](experiments/aster_cr/README.zh.md)
<!-- repair-v2-results:end -->


ASTER研究模型如何依据完整指南和患者事实判断当前条件，再由程序沿指南树执行路径。方法保留三模块：**M1定位干扰通路，M2估计干扰与信号子空间，M3学习表示编辑。**

**论文当前方法：P0–P3中的`paired_ns`。** 固定51个查询头和已有子空间，在干扰／信号联合空间中训练映射与门控，使用在线自身参考及配对监督，共78,051个可训练参数。代码已实现，正式训练和评测尚未完成。

[中文论文](docs/ASTER/manuscript.zh.md) · [起始闭式至今的版本对照](docs/ASTER/versions.zh.md) · [实验进度与运行入口](docs/ASTER/experiments.zh.md) · [模块与公式说明](docs/ASTER/method-clarifications.zh.md)

## 已完成的 M3 全量结果

上一阶段辅助参考／自身参考M3已完成MEGA-CDP全部7,458例测试。各方法共享模板和自主路径控制器。

| 方法 | 完整路径正确 | 正确率 | 相对完整阅读纠正／新增错误 |
|---|---:|---:|---:|
| 完整阅读 | 7,068/7,458 | 94.77% | 0/0 |
| 旧闭式M3 | 7,045/7,458 | 94.46% | 22/45 |
| 辅助参考M3 | 7,072/7,458 | 94.82% | 6/2 |
| 自身参考M3 | 7,068/7,458 | 94.77% | 3/3 |
| 直接读取参考 | 6,445/7,458 | 86.42% | 137/760 |

辅助参考相对完整阅读净增4例（+0.0536个百分点），P0来源重采样95%差值区间为[-0.0138,+0.1300]；相对旧闭式净增27例。以上是已完成版本的成绩，最新版`paired_ns`结果另行补入。

[完整结果与检查点](results/aster_learnable_m3_20261106/README.zh.md) · [P0直接配对统计](results/aster_m3_followup_20261107/p0/REPORT.zh.md)

## 实验进度

截至2026-11-07 17:41（北京时间）的[状态快照](results/aster_documentation_20261107/progress.json)：

| 阶段 | 状态 |
|---|---|
| P0 节点分析 | 直接统计完成；固定节点补评运行中；变化节点16/32条记录已保存 |
| P1 迁移评测 | 代码和数据已准备，本轮MedGUIDE结果待完成 |
| P2 选头对照 | 随机头／候选头／全头配置已准备，正式训练和评测待完成 |
| P3 N/S/NS配对训练 | paired_ns进入第3轮；九个新配置此时均未保存正式完成记录 |
| M1双模型诊断 | 已完成，纳入论文机制分析，未改变当前主方法头集 |

进度按原实验分支保存结果更新；部分记录不是完整成绩。主线展示最新方法与实际进度，不以“稳定优势”为收录条件。[阶段报告](results/aster_m3_followup_20261107/REPORT.zh.md) · [实验代码及运行说明](experiments/aster_m3_followup/README.zh.md)

## 文档与历史入口

[ASTER文档](docs/ASTER/README.md) · [历史实验总索引](reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) · [历史资料与整理前快照](docs/ASTER/archive/README.md) · [既有实验目录](experiments/README.md)

更早TRIAGE、ACRE、CARE和单层表示干预保留原路径；此前首页的数据集目录及历史说明保存在[整理前首页](docs/ASTER/archive/20261107/README.md.txt)。本次整理不运行训练或评测，不占用GPU，不修改原工作区或实验进程。
