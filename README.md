# TRIAGE · ASTER

**当前主线：处置盲判与适用性门控注意力。** 本仓库以 ASTER（Action-blind Selection, Then Evidence-weighted Reveal）为研究主线：先根据患者事实和规则条件判断适用分支，再按分支置信度控制处置信息的读取。

> **研究状态：方案与待验证工作稿。** 2026-10-26 主线整理以新上传的论文草稿及六项主张为依据。原稿中的【预填】数值、机制定位、跨模型结果、消融、显著性、延迟与成本均不能作为已完成的实验发现。新主线稿使用待测结果表；原稿另行完整保留。已有历史代码与结果不自动构成新版 ASTER 的实现或证据。

[主线文档](docs/ASTER/README.md) · [论文工作稿](docs/ASTER/manuscript.zh.md) · [主张与证伪](docs/ASTER/claims-and-falsification.zh.md) · [实验计划](docs/ASTER/experiment-plan.zh.md) · [图稿](docs/ASTER/figures/README.md) · [旧支线索引](docs/BRANCHES.md)

## 研究问题与方法

条件化指南要求模型先判断“哪条分支适用”，再执行该分支的处置。本文拟检验：不适用兄弟分支上的熟悉处置是否会反过来影响分支判断。这一待验证现象称为处置先行的分支选择（TLBS）。

ASTER 的方法设计分为两个阶段：

1. **处置盲判。** 屏蔽判断位置对处置内容的读取，包括处置名称、选项和间接信息泄漏；依据患者事实及条件形成分支分布。完整版本增加只读取条件和病历的证据员。
2. **适用性门控执行。** 揭示处置，以对数偏置压低竞争处置，并在预先确定的层窗口中有限强调所选规则。置信差小则干预较弱；高置信错误仍可能被放大。

![ASTER 两阶段方法示意；设计图，不含实测结果](docs/ASTER/figures/fig02_aster.svg)

## 主线阅读入口

| 内容 | 文件 | 作用 |
| --- | --- | --- |
| 论文 | [论文工作稿](docs/ASTER/manuscript.zh.md) | 现象、机制假设、方法、评测与待填结果表 |
| 六项主张 | [主张—实验—证伪条件](docs/ASTER/claims-and-falsification.zh.md) | 明确什么结果会削弱或否定主张 |
| 评测协议 | [实验计划与指标定义](docs/ASTER/experiment-plan.zh.md) | 五卷配对、同信息比较、分母、锁定测试与统计原则 |
| 证据状态 | [证据与投稿检查](docs/ASTER/evidence-status.zh.md) | 区分来源陈述、待测假设和已核对的文档事实 |
| 数学与数值审计 | [原稿审计](docs/ASTER/audit.zh.md) | 公式边界、占位表一致性和待修问题 |
| 文献 | [引用核验清单](docs/ASTER/references-to-verify.zh.md) | 追踪原稿中尚未核验的文献与引文标记 |
| 图稿 | [四幅图及图注](docs/ASTER/figures/README.md) | SVG、PDF、PNG 与可复现绘图脚本 |
| 实现范围 | [代码与方案对应关系](docs/ASTER/implementation-status.zh.md) | 防止把旧三模块或早期 ASTER 实验等同于本稿 |
| 原始输入 | [原稿快照](docs/ASTER/sources/manuscript-20261026.txt) · [用户主张清单](docs/ASTER/sources/user-claims-20261026.md) | 可追溯的原始来源；不是结果发布 |

## 旧研究作为支线保存

整理前 `main` 的完整快照保存在 [`archive/pre-aster-main-20261026`](https://github.com/chunxi-alpc/TRIAGE/tree/archive/pre-aster-main-20261026)，固定提交为 [`0a831d1`](https://github.com/chunxi-alpc/TRIAGE/commit/0a831d1127c338119bc11712677170486d40aabb)。已有 38 条研究分支继续保留，逐一列于 [支线索引](docs/BRANCHES.md)。

原 CARE、ACRE 和 TRIAGE 三模块文档不再定义研究主线。旧文档路径提供归档跳转；`src/`、`configs/`、`examples/`、既有实验及结果文件继续保留，作为历史实现和复现资产。各目录中的“当前”“默认”“下一步”须按其原实验日期理解。

| 历史资产 | 入口 |
| --- | --- |
| 既有模块与流水线 | [模块一](src/module1/README.md) · [模块二](src/module2/README.md) · [模块三](src/module3/README.md) · [流水线](src/pipeline/README.md) |
| 旧实验与负结果 | [实验索引](experiments/README.md) · [结果索引](results/README.md) |
| 早期 ASTER 反事实实现 | [2026-10-22 实验目录](experiments/aster_medguide_20261022/README.md) |
| 旧模型运行命令 | [历史运行说明](docs/RUNNING.md) · [脚本索引](scripts/README.md) |

## 图稿与复现

四幅图已按英文论文双栏版式重绘：TLBS 虚构示例、ASTER 注意力算子、五卷配对决策树、解析门控曲线。提供 7 英寸宽的 PDF、可编辑 SVG、450 dpi PNG，以及带英文图注的[实际尺寸校样](docs/ASTER/figures/figures_review.pdf)和 [LaTeX 插图片段](docs/ASTER/figures/figures.tex)。前三幅是研究设计示意；第四幅由明确给定的概率代入公式计算，**不是实测性能曲线**。

绘图环境及命令见 [图稿说明](docs/ASTER/figures/README.md)。本次文档与图稿整理没有运行新的模型实验。后续实验须在独立开发集上确定配置，并在锁定测试集上统一重算指标；若实测结果不支持主张，应修改主张。
