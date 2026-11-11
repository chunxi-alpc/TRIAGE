# ASTER

研究模型如何依据完整指南和患者事实判断当前条件，再由程序执行诊疗路径。仓库同时保留 CARE、ACRE 和 ASTER 的历次方法与实验。

**[实验总表：全部路线、结果和完成状态](docs/ASTER/experiments.zh.md)** · [方法与公式](docs/ASTER/versions.zh.md) · [ASTER-CR 中文论文](reports/aster_cancel_story_20261109/paper_story_latest.zh.md)

## 方法路线

| 路线 | 主要探索 | 入口 |
|---|---|---|
| CARE / ACRE | 证据定位、条件真假判定、适用性方向、参考对照 | [早期实验汇总](docs/ASTER/experiments.zh.md#careacre与早期探索) |
| ASTER 表示编辑 | 闭式投影、参考设置、可学习 M3、N/S/NS 配对训练 | [方法演变](docs/ASTER/versions.zh.md#表示编辑) |
| ASTER-CR 分数修正 | 线性与非线性修正、证据保护、五模型和跨任务实验 | [近期实验](docs/ASTER/experiments.zh.md#近期分数修正实验) |

## 配对训练数据：2026-11-11

ASTER-CR的MEGA-CDP配对数据构建已完成：全部合格池6,345对，均衡集每类635对、共2,540对；四类均覆盖80个拟合来源。生成为Sol/high，独立审核为Sol/xhigh；尚未使用新数据训练模型。[构建报告与两张图](reports/aster_cr_pair_dataset_20261111/README.md) · [程序与复现](experiments/aster_cr_pair_dataset_20261111/README.md)。GitHub只收录程序、说明和汇总，数据及逐条记录保留服务器。

## 当前结果与进度

核对日期：2026-11-10。不同阶段的提示、训练数据和评价范围分别记录。

- 可学习 M3 五方法全量测试已完成：辅助参考7,072/7,458，完整阅读7,068/7,458。
- N/S/NS等9个新配置已训练并完成200题迁移试测；paired_ns为103/200，完整阅读104/200；7,458路径主测试和522对诊断未完成。
- ASTER-CR已有Qwen全量运行、优化、机制验证和跨任务结果。部分设置降低干扰响应，但尚未取得稳定的跨任务准确率收益。
- 五模型在MEGA节点、NLI4CT和程序生成三段论上的15组实验已完成；[逐组结果](results/aster_five_models_20261109.csv)。
- 后续五模型 × MEGA / NLI4CT / MedPIC的285项统一评测仍未完成；旧15组使用不同任务和协议。
- 新痕迹修复已有代码，归档时仍在收集拟合数据，无最终结果。

## 阅读顺序

先看[实验总表](docs/ASTER/experiments.zh.md)，需要方法细节再读[方法说明](docs/ASTER/versions.zh.md)或[论文](reports/aster_cancel_story_20261109/paper_story_latest.zh.md)。代码与结果分别从[实验目录](experiments/README.md)和[结果目录](results/README.md)进入。

历史报告保留原位置；2026-11-10补齐材料按主题压缩收纳，包内附来源和文件说明，见[归档对应表](docs/ASTER/experiments.zh.md#本次补齐的归档)。该次归档只整理已有材料；2026-11-11的数据构建另见上方入口。
