# TRIAGE

TRIAGE 将指南规则检索、患者条件与证据判断、注意力头诊断及门控干预串联起来。三模块已经集成，目前处于小规模算法诊断与关键错误修复阶段，尚未运行完整基准。

项目在服务器 `gcx@10.29.128.115:/home/gcx/TRIAGE` 运行，复用环境 `/home/gcx/gcx261014`、现有 Qwen3-8B、MedCPT-MedPair、投影参数及缓存。当前阶段没有重新部署、下载新模型或训练模型。

## 最新关键实验：2026-10-15 正确标注下的目标切换诊断

已完成 E0/E1 并独立复算全部 80 例。相同参考关系与证据、共同 β=2 下，固定增强 OF **68/80、31/40 对、OAR 2/40**；切换增强 OS **66/80、30/40 对、OAR 3/40**，原正确均保持 64/64。主比较没有切换收益，按停止条件暂停当前完整规则路线；E2 自标注优化、E3 新前向和独立验证均未运行。

完整 Oracle 的 80 例掩码虽不同，但 OS 只是 OF 的子集，没有新增否定证据。仅参考目标规则的次要诊断在 40 例不适用病例真正改变目标，OS-target 比 OF-target 多纠正 3 例（65→68/80），保留为不可部署、知晓答案的局部机制信号，不能替代主比较。β=1 的完整 OF/OS 指标相同，未重新选择参数。

- [任务书](docs/ACRE/07-next-experiments.md)、[中文评审](docs/ACRE/08-oracle-switch-results.md)
- [新结果与复现](results/acre_pilot/oracle_switch_diagnostic/README.md)、[完整指标](results/acre_pilot/oracle_switch_diagnostic/oracle_factorial_metrics.json)、[独立复算](results/acre_pilot/oracle_switch_diagnostic/recompute_check.json)
- 历史 revision3 和旧模块不变。正常预测与 Oracle 来源隔离。全部结果仍属已观察开发集的探索性分析。

## 上一轮：2026-10-15 简化 ACRE 开发集预实验

完成 **40 对、80 例**可程序核验的反事实开发样本，覆盖 36 个新指南；旧筛选只得 4 对主要由条件覆盖、全文匹配和选项上限造成。JSON 与原文引用已修复，另外发现并修正受限解码对“不适用”的分词偏置，在同一冻结数据上重新运行 B0–B6。

共享 beta=2，动作同义表达审计修正后的结果：

| 方法 | 正确 | 成对正确 | 目标动作误用 |
| --- | ---: | ---: | ---: |
| B0 原始指南 | 64/80 | 29/40 | 4/40 |
| B1 整条增强 | 63/80 | 28/40 | 5/40 |
| B2 固定目标 | 64/80 | 29/40 | 3/40 |
| B3 ACRE | 64/80 | 29/40 | 3/40 |
| B4 结构化提示 | 60/80 | 24/40 | 7/40 |
| B5 标注直接决策 | 37/80 | 12/40 | 7/40 |
| B6 参考标注 | 66/80 | 30/40 | 3/40 |

目标规则适用性判断 60/80（75%），不适用误判适用 12/40（30%）；决定性证据召回 74/80。原文定位 100% 是 ID 机制保证，不代表语义正确。20 项 CPU 测试与 GPU 检查通过，80 例零偏置/空目标完全一致，实际目标及指标已独立复算。

**上一轮 B3 没有优于 B2。最新同源 Oracle 对照见下节；当前不进入独立验证或完整论文实验。** B6 有两例纠正；仅替换适用性状态的额外诊断为 67/80，仍属于参考信息诊断，优先改进适用性和相关性。噪声诊断已完成，片段替换没有改变最终增强集合，只能作为失败记录。37/40 对是资源/可实施性条件，没有临床专家盲审；未知真值和完整动作语义评价尚缺。动作别名计数是公开的事后指标修正，原始计数保留。

- [中文评审与完整复现说明](docs/ACRE/06-pilot-results.md)
- [开发数据审计](results/acre_pilot/round2/data_audit.json)、[动作别名审计](results/acre_pilot/round2/action_alias_audit.json)
- [最终指标](results/acre_pilot/round2/revision3/metrics.json)、[标注质量](results/acre_pilot/round2/revision3/annotation_metrics.json)、[工程验收](results/acre_pilot/round2/revision3/acceptance.json)
- [逐例完整结果](results/acre_pilot/round2/revision3/case_results.jsonl.gz)、[标注原始缓存](results/acre_pilot/round2/revision3/annotations.jsonl.gz)、[噪声结果](results/acre_pilot/round2/revision3/noise_metrics.json)
- 最终配置 `configs/acre_pilot_v3.json`；解压入口 `scripts/unpack_acre_pilot_artifacts.py`，运行入口 `scripts/run_acre_pilot.py`，离线复算 `scripts/verify_acre_records.py`。模型和环境均复用服务器现有版本。

初轮 24 例只作回归，初轮四对数据门槛记录见 [归档报告](docs/ACRE/archive/06-pilot-results-round1.md)。`results/acre_pilot/round2/` 根目录保留失败的 v2 解码实验，**最终结果仅以 revision3 为准**。以下旧阶段结果和模块保持原样。


## 当前结果：2026-10-14 小规模诊断

使用 MEGA-CDP **train_val 的 24 例**，覆盖 6 个指南；排除此前用于组件校准的指南 3。概率对照采用完整指南的全部去重动作加“证据不足”作为固定选项，不属于完整基准或自由文本准确率评估。无规则基线仍能看到动作选项，但没有规则条件。

| 版本 | 正确动作选择 |
| --- | ---: |
| 原始模型，无规则条件、无干预 | 15/24 |
| 原始模型，提供完整指南、无干预 | 13/24 |
| 正确规则片段引导，仅作诊断上界 | 24/24 |
| TRIAGE 提示，关闭注意力干预 | 16/24 |
| 完整 TRIAGE | 15/24 |

- **关键规则召回：6/24 → 24/24。** 本轮的小指南保留全部规则；这说明候选截断问题得到改善，不代表大指南的检索排序已经改善。
- **条件判断仍是主要问题。** 正确规则被判适用 15 例，另 9 例为未知。
- 完整流程相对无规则基线，正确动作与先验动作的对数概率差平均增加 **3.83**，正确动作概率平均增加 **3.11 个百分点**；但平均正确动作对数概率下降约 **0.33**，且准确率没有净提升。
- 完整流程纠正 **4 例**，同时使 **4 例退化**。单独注意力干预纠正 **0 例**、退化 **1 例**，正确动作概率平均下降约 **0.37 个百分点**。
- 候选头采用从零开始的编号：**层 35/头 8、层 12/头 24**。前者在 15 个活跃病例中 4 例正向、11 例为零；后者 7 例正向、6 例负向、2 例为零，不能认为现有干预具有稳定增益。
- **零门控和零强度：24/24 例与原始计算精确一致。**
- 对历史病例 5:1、9:1 另外检查了自由文本输出，两例完整 TRIAGE 动作均与对应规则一致；原始完整指南输出也正确，没有观察到注意力单独带来的决策纠正。

下一步优先改进条件与患者证据的匹配和复核，减少正确规则被判为未知的情况。当前结果不支持整体准确率已经提高的结论。

## 结果与文档入口

| 内容 | 文件 |
| --- | --- |
| 本轮最终摘要、指标定义与错误统计 | [summary.json](results/algorithm_diagnostic/summary.json) |
| 24 例决策、概率变化、规则召回和错误原因 | [cases.csv](results/algorithm_diagnostic/cases.csv) |
| 两例自由文本验证 | [free_generation.json](results/algorithm_diagnostic/free_generation.json) |
| 固定病例及完整指南输入 | [manifest.json](results/algorithm_diagnostic/manifest.json) |
| 独立离线参考路径与动作 | [references.json](results/algorithm_diagnostic/references.json) |
| 采样与数据来源审计 | [data_audit.json](results/algorithm_diagnostic/data_audit.json) |
| 逐例完整模块输出与干预记录 | [algorithm_diagnostic](results/algorithm_diagnostic/) 中的 `case_*.json` |
| 首轮错误及第二轮修复的历史结果 | [round1](results/algorithm_diagnostic/round1/)、[round2](results/algorithm_diagnostic/round2/) |
| 最初三模块集成记录，属于历史阶段 | [TRIAGE_MVP.md](TRIAGE_MVP.md)、[triage_mvp_report.json](results/triage_mvp_report.json) |
| 组件及校准来源 | [components.json](results/module2/components.json)、[module2/report.json](results/module2/report.json) |
| 复用投影及其元数据 | [projections.pt](results/module3/projections.pt)、[projections.json](results/module3/projections.json) |
| 条件判断回归与干预检查 | [diagnostic_module1_checks.json](results/diagnostic_module1_checks.json)、[module23_checks.json](results/diagnostic/module23_checks.json) |

首轮证据核对误把“有引用”当成“条件为真”，第二轮修复了真假和编号判断，最后修复了被阻断动作仍进入最终提示的问题。各阶段结果保留，最终结论以根目录下的 `summary.json`、`cases.csv` 为准，不能混用不同阶段。

**参考答案只用于离线评价，不进入常规推理、门控或注意力干预。** 唯一单列的诊断上界使用正确的官方规则片段，不输入参考答案正文或路径标签，也不参与门控计算。

## 原有研究设计文档

原仓库的 ACRE 研究设计文档继续保留。这些文件是方案与评审归档；条件—证据成对注意力偏置、完整基准和论文机制主张仍需后续验证，不能把设计方案视为本轮已实现的功能。

- [设计评审与近邻工作](docs/ACRE/01-design-review.md)
- [方法定义](docs/ACRE/02-method.md)
- [实验设计](docs/ACRE/03-experiments.md)
- [风险与实施计划](docs/ACRE/04-plan-and-risks.md)
- [参考文献核验清单](docs/ACRE/05-reference-checklist.md)

## 三模块实现

1. **模块一：** [src/module1](src/module1/) 使用问题和患者信息检索规则，默认候选数量为 12；不超过 24 个规则的小指南全部保留。条件判断使用明确的真假状态、稳定条件 ID、条件原文核对以及患者引用证据复核。未知和不适用规则的动作门控为零。
2. **模块二：** [src/module2](src/module2/) 对固定候选注意力头进行消融诊断。当前组件来自独立的非测试校准病例；没有进行大规模注意力头搜索。
3. **模块三：** [src/module3](src/module3/) 复用 rank=4 投影，仅编辑选中的查询头，使用现有门控控制键向量修改及先验输出抑制。默认强度仍为 `context_strength=0.1`、`prior_strength=0.1`。

[统一推理入口](src/pipeline/runner.py) 复用同一 Qwen 实例。最终提示保留条件判断和患者证据，但只展示已适用规则的动作；被阻断或未知分支不能作为建议依据。输出仍需进一步验证，尚未实现强制的动作语义一致性校验。

门控、相关性分数和证据置信度都是未校准的启发式权重。投影属于少量校准激活的 SVD，先验方向是注意力输出的近似子空间，不是已证明的独立先验方向。

## 服务器运行

```bash
ssh gcx@10.29.128.115
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE
```

**实验只在服务器运行。** 可用 GPU 为 **0、1、2、3、6、7**。正式启动前必须汇报计划使用的 GPU、预计显存、样本规模和时长；稳定运行后停止持续监控。

模型与数据路径见 [configs/models.json](configs/models.json)、[configs/datasets.json](configs/datasets.json)，依赖版本见 [configs/environment.lock.txt](configs/environment.lock.txt)。原始数据、模型权重、第三方仓库和环境保留在服务器。

输入 JSON 只包含 `guideline`、`patient`、`question`；可选 `input_source` 用于来源记录：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_pipeline.py \
  --input /absolute/path/input.json \
  --output results/pipeline/inference.json
```

模块一也可单独调用：

```python
from src import module1
result = module1.run(guideline=R, patient=P, question=Q)
```

`R` 支持文本、条款列表或 `{"rules": [...]}`，`P` 和 `Q` 为字符串。批量调用可创建 `module1.Module1(top_k=12)` 复用模型。患者引用记录原文字符区间 `[start,end)`，保留满足、不满足和未知三种状态。

本轮诊断入口为 [scripts/run_algorithm_diagnostic.py](scripts/run_algorithm_diagnostic.py)，支持 `--worker`、`--workers` 和 `--module1-cache-dir`。复用 `results/algorithm_diagnostic/round2` 可只重算最终概率与干预，避免重复运行模块一。离线汇总入口为 [scripts/summarize_algorithm_diagnostic.py](scripts/summarize_algorithm_diagnostic.py)；它只生成基础指标摘要与 CSV，不调用模型。已保存的最终摘要另含自由文本验证、资源和修复记录。

## 每次实验的上传要求

**每次实验完成后，必须将对应的结果和文档提交并推送到本仓库 `git@github.com:chunxi-alpc/TRIAGE.git`。只在服务器保存，不算实验交付完成。**

至少上传：

- 简洁的结果摘要、逐例或汇总指标、错误案例和原因；失败实验也要保留真实结果。
- 数据集和 split、样本选择与来源、运行入口和参数、随机种子、模型与组件版本或指纹。
- 相应代码或配置变更，以及必要的可复现产物；参考答案与普通推理输入分开保存。
- 更新本 README 的当前阶段、结果入口、主要结论、尚未解决的问题；必要时同步相关技术文档。

保留不同实验阶段，不用新结果覆盖旧实验依据。提交信息应说明实验内容；推送后核对远端提交。模型权重、完整数据、环境、凭据和无关下载日志不进入仓库。

仓库保存代码、配置、文档及可分析的实验产物；详细运行日志留在服务器对应结果目录。

