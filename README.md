# TRIAGE

TRIAGE 研究指南规则的适用性判断、患者证据定位，以及受适用性约束的注意力干预。仓库同时保存三模块实现、CARE 独立实验和 ACRE 历史研究。

**当前模块一为 CARE V6。** 包级调用、命令行和统一流水线均使用 V6。模块二、三保留现有实现及校准产物；模块二最新研究为 2026-10-19 的有符号适用性诊断。项目仍处于组件验证和小规模研究阶段，尚未完成 V6 三模块端到端基准。版本来源与整理范围见 [版本记录](docs/REPOSITORY_STATUS.md)。

[模块一](src/module1/README.md) · [模块二](src/module2/README.md) · [模块三](src/module3/README.md) · [统一流水线](src/pipeline/README.md) · [实验索引](experiments/README.md) · [运行说明](docs/RUNNING.md)

## 三模块与当前状态

| 部分 | 职责 | 当前实现与边界 |
| --- | --- | --- |
| [模块一：适用性与证据](src/module1/README.md) | 将结构化规则和患者文本转换为适用性、证据位置与门控 | CARE V6；Phi-4 + TrialMatchAI LoRA 判断条件，数值比较处理支持字段，BioLinkBERT-MedNLI 定位证据。V6 不解析自由文本指南，也不进行规则检索 |
| [模块二：查询头定位](src/module2/README.md) | 通过查询头输出消融，估计适用性相关贡献 | 主流程保留 Qwen3-8B 的历史校准组件；CARE 子空间研究单独保存，未自动替换在线定位实现 |
| [模块三：门控干预](src/module3/README.md) | 按规则与证据门控编辑选定查询头的键向量，并保留输出投影抑制路径 | 复用原 rank=4 投影。普通 V6 输出的先验竞争权重为零；投影尚未针对 V6 重新校准 |
| [统一流水线](src/pipeline/README.md) | V6 图 → 词元位置映射 → Qwen3-8B 原始与干预推理 | 只接受 `module1.v6`。模块一独立加载模型；原始与干预生成共享最终 Qwen 实例 |

已有组件可连接并不等于新组合已经获得性能增益。模块一的数值主体与时间识别、合取规则证据门控仍有已知限制，详见 [模块一说明](src/module1/README.md#已知限制)。

## 从哪里开始

| 目标 | 入口 |
| --- | --- |
| 使用当前模块一 | [输入与 API](src/module1/README.md) · [结构化输入示例](examples/module1_v6_input.json) |
| 运行三模块流程 | [流水线配置](src/pipeline/README.md) · [服务器运行说明](docs/RUNNING.md) |
| 查看模块一最新结果 | [V6 实验导航](experiments/care_module1_v6/README.md) |
| 查看模块二最新结果 | [有符号适用性诊断](experiments/care_module2_signed_20261019/README.md) |
| 查找旧实验与失败记录 | [实验索引](experiments/README.md) · [结果索引](results/README.md) |
| 理解版本来源与研究演变 | [版本记录](docs/REPOSITORY_STATUS.md) · [ACRE 归档](docs/ACRE/README.md) |

## 当前入口示例

从仓库根目录运行。下列模型推理命令需要先按 [运行说明](docs/RUNNING.md) 配置服务器环境与本地模型路径。

```bash
# 查看参数，不加载模型
python -m src.module1 --help
python scripts/run_pipeline.py --help

# 单独运行模块一 V6
python -m src.module1 \
  --input examples/module1_v6_input.json \
  --output results/module1_v6/example.json

# 统一流水线：V6 + 原有组件与投影
python scripts/run_pipeline.py \
  --config configs/pipeline_minimal.json \
  --input examples/module1_v6_input.json \
  --output results/pipeline/v6_example/inference.json
```

输入包含 `guideline`、`patient`、`question`；V6 的 `guideline` 必须是预先结构化的规则。示例只演示接口格式，不属于实验数据。当前模型依赖入口为 [requirements-module1-v6.txt](requirements-module1-v6.txt)。三个模块一模型可通过 `CARE_MODULE1_V6_*` 环境变量指定；完整配置见 [模块一说明](src/module1/README.md)。

Python 调用：

```python
from src.module1 import create_module1_v6

module1 = create_module1_v6(device="cuda:0")
graph = module1.run(guideline=structured_rules, patient=patient, question=question)
assert graph["schema_version"] == "module1.v6"
```

历史 Qwen/MedCPT 模块一保留为 `LegacyModule1`、`run_legacy` 和命令行 `--legacy`。旧 `Module1` 类别名供历史脚本使用；新代码应使用明确的 V6 入口。旧 `module1.v1` 缓存不能直接传给当前流水线。

## 最新已保存结果

### 模块一 V6

V6 将 BioLinkBERT 的职责限制为证据定位与门控，不再由它覆盖适用性标签。最终方法 C 的已保存结果如下；旧集与新增集分别解释。

| 数据 | 适用性准确率 | 宏平均 F1 | 已知样本证据定位准确率 | 判定与证据同时正确 |
| --- | ---: | ---: | ---: | ---: |
| 旧 192 条，离线重合成 | 151/192（78.65%） | 0.7754 | 55.47% | 38.54% |
| 新增 100 条 | 73/100（73.00%） | 0.7339 | 37.68% | 25.00% |

来源：[保存指标](experiments/care_module1_v6/data/v6_metrics.json) · [逐例结果与复现边界](experiments/care_module1_v6/README.md)。旧 192 条复用已有 V5 条件与证据结果，并非重新运行 V6 模型。证据可靠性仍有限；本表不代表完整三模块流程的表现。

### 模块二有符号适用性诊断

2026-10-19 的 Qwen3-4B-Base 独立实验使用 384 个模拟病例、192 对和 12 个规则家族。干预使用参考适用性标签，因而不评价模块一预测错误。

| 方法 | 成对准确率 |
| --- | ---: |
| 无干预 | 37.50% |
| 原始 B4 正向放大 | 38.54% |
| 有符号 B4 | 39.06% |
| 有符号均值位移 | 39.06% |
| 直接有符号分数偏置 | 40.10% |

成对准确率要求一对病例的两侧均正确。**尚未建立学习方向优于直接分数偏置的证据。** 直接偏置的成对准确率点估计最高，但正确答案边际略低于无干预；方法间差异及不确定性见 [原结果报告](experiments/care_module2_signed_20261019/results/analysis/README.md)。该研究属于单模型、模拟病例的探索性诊断。

## 历史实验入口

| 阶段 | 保存内容与结论 | 入口 |
| --- | --- | --- |
| 模块一 V4/V5 | 证据修复、模型组合、解析器修复与失败条件重生成；同时保留改善和退化 | [V4](experiments/care_module1_v4/README.md) · [V5](experiments/care_module1_v5/README.md) |
| CARE 三项观察 | 临床来源规则的 O1 与虚构规则的 O2/O3；没有观察到假设中的稳定忠实性—误用权衡 | [实验目录](experiments/care_qwen3_4b_base_20261016/README.md) |
| 模块二 O2 复核与 B1–B4 | 头选择、数值混杂、家族留出与干预对照；未建立可靠学习子空间优势 | [O2 复核](experiments/care_qwen3_4b_base_20261016/results/module2_o2_reanalysis/README.md) · [B1–B4](experiments/care_qwen3_4b_base_20261016/results/module2_followup/README.md) |
| TRIAGE 原 24 例诊断 | 完整 TRIAGE 与无规则基线均为 15/24；纠正与退化各 4 例，未获得净准确率提升 | [结果索引](results/README.md) · [原摘要](results/algorithm_diagnostic/summary.json) |
| ACRE 同源参考对照 | 完整规则切换 66/80，固定增强 68/80；完整规则路线暂停 | [研究归档](docs/ACRE/README.md) |

各阶段的模型、样本、筛选条件和评价目标不同。历史指标按对应协议保留，不合并为整体性能结论。原始研究提案与已执行结果在索引中明确标注。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [src/](src/) | 当前三模块、流水线、历史兼容代码和独立 ACRE 实现 |
| [configs/](configs/) | 流水线参数、模型与数据位置、历史环境记录 |
| [examples/](examples/) | 当前结构化输入格式示例 |
| [experiments/](experiments/README.md) | 按实验版本保存的代码、协议、预测、指标与图表 |
| [results/](results/README.md) | 原流水线诊断、校准组件、投影及 ACRE 历史产物 |
| [docs/](docs/README.md) | 文档导航、运行说明、版本记录与研究归档 |
| [scripts/](scripts/README.md) | 当前推理入口和按版本说明的历史脚本 |
| [tests/](tests/) | 仓库接口检查及历史独立实验检查 |

## 运行与维护约定

模型实验在既有服务器运行，可用 GPU 为 **0、1、2、3、6、7**。启动前说明 GPU、预计显存、样本规模与时长；稳定运行后停止持续监控。服务器和环境记录见 [运行说明](docs/RUNNING.md)。普通静态检查与已存结果的离线复核不需要启动模型。

每轮实验完成后，将代码、配置、结果摘要、逐例或汇总指标、失败案例、数据划分、模型指纹与复现说明提交并推送到本仓库。只保存在服务器不算交付完成。新实验使用新目录，保留负结果和旧校验清单，并同步更新本首页及相关模块入口。

参考答案仅用于独立离线评价；使用参考标签的诊断必须单独标注。模型权重、完整环境、受限原文、凭据及无关运行日志依各轮约定留在服务器，不进入仓库。不能用文档整理替代模型或临床效果验证。
