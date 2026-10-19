# CARE 模块二：有符号适用性诊断

**2026-10-19 已完成的独立研究实验。** 本目录比较原始 B4、有符号 B4、有符号均值位移与直接注意力分数偏置。真实适用性标签作为诊断输入，用于研究模块二在理想适用性信息条件下的行为。**尚未建立学习方向优于直接分数偏置的证据。**

[模块二统一入口](../../src/module2/README.md) · [原始结果报告与图表](results/analysis/README.md) · [完整实验协议](PROTOCOL.md)

## 主要结果

模型固定为 Qwen3-4B-Base，使用 BF16。数据沿用前轮的 384 个模拟病例、192 对、12 个临床规则家族及三个家族划分。每折包含六个训练家族、两个开发家族、四个测试家族；同一家族与配对成员始终留在同一划分，每个家族测试一次。模型版本和设计细节见 [PROTOCOL.md](PROTOCOL.md)。

成对准确率要求同一对中适用与不适用病例均回答正确。下表取自[保存的指标](results/analysis/metrics_ci.csv)，变化以无干预为参考。

| 方法 | 成对准确率 | 变化（百分点） | 配对家族自助法 95% 区间（百分点） |
|---|---:|---:|---:|
| 无干预 | 37.50% | 0.00 | 参考 |
| 原始 B4 正向放大 | 38.54% | +1.04 | [−1.04, +3.12] |
| 有符号 B4 | 39.06% | +1.56 | [−0.52, +3.65] |
| 有符号均值位移 | 39.06% | +1.56 | [0.00, +3.12] |
| 直接有符号分数偏置 | 40.10% | +2.60 | [+1.04, +4.17] |

直接偏置的成对准确率点估计最高，但正确答案与最大错误答案的候选输出分数（logit）间隔均值略低于无干预（`0.532552` 对 `0.533529`）。学习方向与直接偏置的差异仍有不确定性；应结合[全部十项配对方法比较](results/analysis/paired_comparisons.csv)判断，而不只按表中点估计排序。

## 干预与控制条件

记原 B4 正交投影为 `P = UUᵀ`，动作片段的键为 `k`。真值适用时 `s = +1`，不适用时 `s = −1`。本轮方法定义如下，完整公式见[协议](PROTOCOL.md)。

| 方法 | 修改 |
|---|---|
| 原始 B4 | `k' = k + aPk` |
| 有符号 B4 | `k' = k + a·sPk` |
| 有符号均值位移 | `k' = k + a·s·d_app`；`d_app` 为训练配对中适用减不适用的动作键均值位移 |
| 直接分数偏置 | 在相同因果注意力位置加上 `s·tau` |

原 B4 的正交投影丢失位移的整体符号，在线编辑也没有适用性符号参数。键的正系数增量不保证每个查询—键分数都增加。有符号方法传入的真值标签只控制干预，不进入提示文本或答案选项。

四种方法共用 PR #3 每折在开发集选定的位置与秩、原始动作片段、四个关联查询头及相同因果位置。B4 从新提取的训练键重新拟合；均值位移也只用训练家族。每个输入的键编辑按旋转位置编码及 BF16 舍入后的实际分数变化均方根校准，属于近似匹配。

强度候选为 `0、0.01、0.05、0.1`，依据开发集选择每折共用强度，最终为 `0.01、0.1、0.1`。测试结果不参与选择；各方法独立开发集选择及完整固定强度结果另行保存。最大相对校准误差为 7.29%，详见 [calibration.csv](results/analysis/calibration.csv)。

## 结果与复现材料

| 内容 | 入口 |
|---|---|
| 人可读结果、四组图表 | [结果报告](results/analysis/README.md) |
| 主要指标、区间、全部方法比较 | [metrics_ci.csv](results/analysis/metrics_ci.csv)、[paired_comparisons.csv](results/analysis/paired_comparisons.csv) |
| 家族结果与强度敏感性 | [per_family.csv](results/analysis/per_family.csv)、[sensitivity.csv](results/analysis/sensitivity.csv) |
| 结构化摘要、选定预测 | [summary.json](results/analysis/summary.json)、[selected_predictions.jsonl.gz](results/analysis/selected_predictions.jsonl.gz) |
| 三折原始记录、选择、算子及来源 | [fold0](results/fold0/)、[fold1](results/fold1/)、[fold2](results/fold2/) |
| 旧 B4 独立复现、归档核验 | [validation](results/validation/)、[verification.json](results/verification.json) |
| 文件 SHA256、前轮源码指纹 | [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)、[pr3_source_hashes.json](pr3_source_hashes.json) |

上述材料记录的是完成的历史运行。整理增加本中文入口，保留原推断代码、报告、协议和清单。三折共保存 9,792 条主记录，部分为零强度复用，不能解释为 9,792 个独立病例。

## 离线复算

从**仓库根目录**执行以下命令。只需保存的预测记录与筛选表，不加载模型权重、不启动 GPU。当前 [analyze.py](analyze.py) 从 [diagnostic.py](diagnostic.py) 导入辅助函数，因此仍需安装 NumPy、Matplotlib、PyTorch、Transformers。记录版本可查 [运行来源](results/fold0/provenance.json)及[环境清单](../../configs/environment.lock.txt)。

```bash
M2_SIGNED_REPLAY=$(mktemp -d /tmp/triage-module2-signed.XXXXXX)

python experiments/care_module2_signed_20261019/materialize_journals.py

python experiments/care_module2_signed_20261019/analyze.py \
  --run experiments/care_module2_signed_20261019/results \
  --artifacts experiments/care_qwen3_4b_base_20261016 \
  --output "$M2_SIGNED_REPLAY/analysis"
```

[解压脚本](materialize_journals.py) 在原目录旁生成 `.jsonl`，保留提交的 `.jsonl.gz`；已有未压缩文件须与压缩内容完全一致。此步必需，因为原分析脚本读取 `predictions.jsonl`。`--artifacts` 指向仓库内前轮数据；该筛选表缺失时，分析脚本使用本目录的 [inputs/prior_screen.csv](inputs/prior_screen.csv)。新统计表与图写入临时目录，原结果不被覆盖。

完整模型重跑另需固定权重、原始规则与模拟病例，以及 PR #3 未在本目录重复保存的完整键缓存。[PROTOCOL.md](PROTOCOL.md) 记录模型修订、服务器路径、完整归档 SHA256、四卡启动条件和数值检查。重跑须按已有项目要求先报告资源计划，使用新输出目录，作为新一轮执行保存。

## 历史关联与结论边界

本轮来自 [PR #8](https://github.com/chunxi-alpc/TRIAGE/pull/8)，原始提交为 [`4b2f6f3`](https://github.com/chunxi-alpc/TRIAGE/commit/4b2f6f3ebf774aaecb786478b532fae13971e105)。它在当时 `main`（`c005b52`）上独立增加本目录，研究上承接 [PR #3 子空间实验](../care_qwen3_4b_base_20261016/results/module2_followup/README.md)。PR #3 的分支包含 [PR #2 O2 复核](../care_qwen3_4b_base_20261016/results/module2_o2_reanalysis/README.md)，PR #8 的分支未包含二者。本仓库整理后保留三个阶段，原始来源提交仍可追溯。

本轮在前轮结果已知后设计，是同一模型和数据上的探索性后续研究。10,000 次家族自助抽样条件于已拟合的三个外层折，没有包括重新拟合的不确定性；仅有 12 个家族，次要比较未做同时多重比较校正。零宽或触及零的经验区间不能证明总体效应为零或方法等效。

真值适用性诊断不评价模块一错误，也不等于已部署流程性能。无冲突准确率使用原 176 例稳定闭卷一致病例；保持率另以其中无干预已答对的 149 例为分母。模拟病例、单模型及小规模家族样本不支持临床效果结论。
