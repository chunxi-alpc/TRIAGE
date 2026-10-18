# TRIAGE

TRIAGE 将指南规则检索、患者条件与证据判断、注意力头诊断及门控干预串联起来。三模块已经集成，目前处于小规模算法诊断与关键错误修复阶段，尚未运行完整基准。

TRIAGE 主流程在服务器 `gcx@10.29.128.115:/home/gcx/TRIAGE` 运行，复用环境 `/home/gcx/gcx261014`、现有 Qwen3-8B、MedCPT-MedPair、投影参数及缓存。该流水线阶段没有重新部署、下载新模型或训练模型；新增 CARE 独立实验使用 Qwen3-4B-Base，见下一节。

## 最新模块一结果：2026-10-18 证据修复与新规则验证

在服务器 GPU 6 用 **Qwen3-8B** 实际完成一次固定测试：16 条新规则、32 个未参与前几轮的完整 MedGUIDE 病例，连同缺失、多值、否定和矛盾修改共 192 条、32 对反事实。

| 方法 | 判定正确 | 宏 F1 | 已知样本证据正确 | 判定与证据同时正确 |
| --- | ---: | ---: | ---: | ---: |
| 直接判定 | 136/192 | 0.7041 | 87/137 | 81/192 |
| 结构化，不核对证据 | 140/192 | 0.7472 | 56/137 | 68/192 |
| 修复后的完整结构化 | 140/192 | 0.7673 | 68/137 | 93/192 |

**结果仅为部分改善，尚未证明同时稳定提高准确率与证据可靠性。** 完整方案纠正 21 条，同时新增 21 条错误；已知样本的证据定位仍低于直接判定。原始完整病例正确数为 26/32，直接判定为 28/32；按规则分组的差值区间包含零。完整方案生成耗时 1331 秒，直接判定 79 秒。全部 24 条规则格式无效输出均计错，未剔除。

修复了引用格式兼容、原文片段定位和脆弱的关键词拒绝规则，并增加同一模型的全文证据核对。仍有 24 条规则格式错误、18 条抽取/核对错误、10 条因证据或状态约束损失正确判定。控制样本保留原病例全文，采用明确更正补充；它们不是独立临床病例，标签未经过医生复核。下一步应处理上述错误，并增加相同推理预算的直接判定核对对照。

[中文报告](experiments/care_module1_v4/REPORT.zh.txt) · [指标](experiments/care_module1_v4/data/metrics.json) · [逐例预测](experiments/care_module1_v4/data/model/predictions.jsonl) · [错误分析](experiments/care_module1_v4/data/error_categories.json) · [代码与复现](experiments/care_module1_v4/README.zh.txt)

6 项针对性回归测试通过，源数据重建哈希与公开指标复算通过。受来源许可限制，原病例和带原文引用的完整输出留在服务器；仓库提供标签、来源行号/哈希、去除引用后的记录与复现代码。该实验位于独立目录，不替代现有三模块流水线。

## 新增结果：2026-10-16 CARE 三项观察

新增独立的 **Qwen3-4B-Base CARE 实测实验**，含真实临床规则的观察一补充；见 [CARE 实验目录](experiments/care_qwen3_4b_base_20261016/)。

- 临床 O1：12 个规则家族、24 条真实规则、384 个模拟病例；宽松闭卷筛选合格 **153 例（适用 78、不适用 75）**，**25 个完整配对两侧均合格**，不设高置信度门槛。
- 三折按规则家族独立测试。原始模型、官方 SEKA、官方 PASTA 适用准确率均为 **33.3%（95% CI 19.5–50.7%）**；误用率为 **36.0%、33.3%、37.3%**，未观察到所假设的忠实性–误用权衡。
- O2/O3 保持虚构规则机制实验范围：适用性 Key AUROC **0.738 [0.660, 0.832]**；实际 BF16 局部分数反向变化 **8.91% [8.70%, 9.14%]**，不能称为临床验证。
- 数据来源、逐例输出、家族置信区间、代码和 **10 组 PNG/PDF/SVG 图**均已保存。完整服务器验收 **65 项检查通过**；患者模拟，无独立医生复核。

[更新报告](experiments/care_qwen3_4b_base_20261016/results/observations_summary.md) · [临床详细报告](experiments/care_qwen3_4b_base_20261016/results/obs1_clinical/clinical_report.md) · [逐例核验表](experiments/care_qwen3_4b_base_20261016/data/clinical/case_review.csv) · [完整数据 Release](https://github.com/chunxi-alpc/TRIAGE/releases/tag/care-qwen3-4b-base-20261016)

## 既有 TRIAGE 流水线结果：2026-10-14 小规模诊断

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

