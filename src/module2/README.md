# 模块二：离线适用性子空间学习

**2026-10-27 新接口：开发树最小对 → 配对差分子空间 → 留出选头 → 冻结键位移。** 新接口不接收模块一的预测、匹配度或置信度。模块一仅在在线调用模块三时确定规则与系数。完整公式及限制见[方法说明](../../docs/ASTER/module2-applicability.zh.md)，实际实验见[独立报告](../../experiments/module2_applicability_20261027/README.md)。

| 新接口 | 职责 |
|---|---|
| [minimal_pairs.py](minimal_pairs.py) | 官方开发树的同父叶兄弟、正负中性三元组、处置与顺序平衡、按原指南划分与独立逻辑审计 |
| [extraction.py](extraction.py) | Qwen3 归一化后、旋转前的处置键；实际位置下的有效查询；GQA 汇总与严格词元对齐 |
| [applicability.py](applicability.py) | 按指南等权的非中心二阶矩、投影均值位移、留出读取与空集合选头；仅依赖 NumPy |
| [在线缓存位移](../module3/applicability_shift.py) | 在首答案查询前修改指定规则缓存键；支持直接分数偏置与实际注意力质量诊断 |
| [表示预实验](../../scripts/run_module2_preexperiment.py) · [因果预实验](../../scripts/run_module2_causal_pilot.py) | 分别保存模型指纹、划分、冻结选择、原始表示或预测和同强度对照 |

输出的位移是自然的投影后正负锚点差，不是单位范数向量，也不保证每个查询都同向读取。第二矩阵的谱识别变化子空间；适用性符号来自均值和独立验证。新的 `module2.applicability.v2` 不能传入旧 `module2.v1` 流水线；历史接口通过延迟导入保留，复现入口不变。

## 历史实现与结果

以下查询头定位及 CARE 结果保留原实验语境。它们不构成新接口的已完成验证。

### 旧查询头定位实现

| 入口 | 职责 |
|---|---|
| [localizer.py](localizer.py) | 在 `o_proj` 输入处消融查询头输出，比较适用与不适用病例的判断变化 |
| [data.py](data.py) | 构造非测试校准对，或读取已有模块一缓存构造证据反事实 |
| [公开接口](./__init__.py) | 提供 `build_calibration_pairs`、`build_cached_calibration_pairs`、`locate_heads`、`scale_query_heads` |
| [定位脚本](../../scripts/run_module2_smoke.py) | 复用本地 Qwen3-8B 权重，执行小规模定位并保存组件 |
| [统一推理入口](../pipeline/runner.py) | 加载已保存的组件，交由模块三执行门控干预 |

定位分数使用二选一适用性判断的 `log P(A) − log P(B)`。候选头须同时满足适用病例贡献为正、且该贡献大于不适用病例的误用贡献。默认检查三个层位置、每层四个查询头，最多保留四个符合条件的头；没有合格头时不设置回退头。实现和返回字段见 [localizer.py](localizer.py)。

统一流程当前接收 `module1.v6`，仍读取历史 [components.json](../../results/module2/components.json)，不会在每次推理时重新定位。独立定位脚本使用的 `build_cached_calibration_pairs` 仍要求 `module1.v1` 历史缓存；它不能直接用 V6 输出替代输入。V6 接入不表示模块二已重新校准或重新验证。

历史校准依据保存在 [calibration_pairs.json](../../results/module2/calibration_pairs.json) 和 [report.json](../../results/module2/report.json)。主流程模型配置见 [models.json](../../configs/models.json)。这些小规模适用性探针的定位分数不代表完整动作质量或临床效果。

### 2026-10-19 历史研究结论

[2026-10-19 有符号诊断](../../experiments/care_module2_signed_20261019/README.md) 使用固定 Qwen3-4B-Base，在 384 个模拟病例、192 对、12 个留出规则家族上比较干预方法。无干预、原始 B4、有符号 B4、有符号均值位移、直接分数偏置的成对准确率依次为 **37.50%、38.54%、39.06%、39.06%、40.10%**。成对准确率要求适用与不适用两例均答对，完整区间见[保存的指标表](../../experiments/care_module2_signed_20261019/results/analysis/metrics_ci.csv)。

**尚未建立学习方向优于直接分数偏置的证据。** 本轮把真实适用性标签作为干预的诊断输入，没有评价模块一的判断错误。病例为模拟病例，结果仅支持单模型探索性结论。主流程 `src/module2` 并未替换为这些研究方法。

## 研究阶段与历史入口

| 阶段 | 内容与有限结论 | 原始入口 |
|---|---|---|
| 2026-10-17，PR #2 | 对 O2 的 288 个层/KV 头位置复核；适用性解码可能混有数值或词汇信号，没有新模型前向 | [O2 统计复核](../../experiments/care_qwen3_4b_base_20261016/results/module2_o2_reanalysis/README.md) |
| 2026-10-17，PR #3 | B1–B4、随机投影、分数偏置及成对替换；没有可靠的 B3/B4 特异决策改善 | [子空间与干预对照](../../experiments/care_qwen3_4b_base_20261016/results/module2_followup/README.md) |
| 2026-10-19，PR #8 | 固定前轮位置与秩，加入真实适用性符号并匹配分数扰动；学习方向优势仍未建立 | [最新诊断](../../experiments/care_module2_signed_20261019/README.md) |

代码来源关系：PR #3 的提交 [`3da16af`](https://github.com/chunxi-alpc/TRIAGE/commit/3da16afb090a71db42843b9df29553f3e2610ceb) 包含 PR #2 的 [`8c5593e`](https://github.com/chunxi-alpc/TRIAGE/commit/8c5593e3d9e1a1bc9295bed916cb3e542198c574)。PR #8 的 [`4b2f6f3`](https://github.com/chunxi-alpc/TRIAGE/commit/4b2f6f3ebf774aaecb786478b532fae13971e105) 则从当时的 `main`（`c005b52`）独立新增实验目录。研究上承接 PR #3，不表示 Git 分支包含 PR #3 的文件。这些阶段在整理后的仓库中并列保留。

PR #2 的高解码 AUC 不能解释为有效干预；PR #3 的 O2 零强度主结果与非零强度补充结果须分别阅读。PR #8 复用同一批病例和家族划分，因此不是新的独立数据集验证。三个阶段均未修改本目录的定位实现。

## 从已保存数据复算

以下命令从**仓库根目录**执行，只复算已有统计量，不加载模型或启动 GPU。输出写入新建临时目录。最新有符号诊断的解压与复算步骤见[其目录说明](../../experiments/care_module2_signed_20261019/README.md#离线复算)。

| 分析入口 | 当前实际依赖 | 输入 |
|---|---|---|
| O2 统计复核 | Python 标准库 | 已提交的 `layer_head_stats.csv` |
| B1–B4 主指标分析 | NumPy、Matplotlib、PyTorch | `completed_20261017` 内压缩预测记录与元数据 |
| 最新有符号诊断 | NumPy、Matplotlib、PyTorch、Transformers | 本轮压缩预测记录，先解压 |

后两项的深度学习库依赖来自分析脚本的顶层导入。无需模型权重不等于无需这些 Python 包。版本记录见 [environment.lock.txt](../../configs/environment.lock.txt) 与各轮运行元数据。

```bash
M2_REPLAY=$(mktemp -d /tmp/triage-module2.XXXXXX)
M2_SOURCE=experiments/care_qwen3_4b_base_20261016
M2_ARCHIVE="$M2_SOURCE/results/module2_followup/completed_20261017"

python "$M2_SOURCE/scripts/reanalyze_o2_head_selection.py" \
  --output "$M2_REPLAY/head_selection.json"

python "$M2_SOURCE/scripts/analyze_module2_followup.py" \
  --run "$M2_ARCHIVE/clinical_fold0" "$M2_ARCHIVE/clinical_fold1" "$M2_ARCHIVE/clinical_fold2" \
  --artifacts "$M2_SOURCE" \
  --output "$M2_REPLAY/clinical"

python "$M2_SOURCE/scripts/analyze_module2_followup.py" \
  --run "$M2_ARCHIVE/o2" \
  --artifacts "$M2_SOURCE" \
  --output "$M2_REPLAY/o2"
```

临床复算须显式提供 `--artifacts`：原默认值是服务器路径，在普通检出中缺失时会省略筛选子组。分析脚本可直接读取 `.jsonl.gz`；临时父目录由 `mktemp` 提前创建。以上命令复算 O2 头选择及 B1–B4 主指标，完整的次要比较、供体分析和原始图表仍以[历史结果包](../../experiments/care_qwen3_4b_base_20261016/results/module2_followup/completed_20261017/)为入口。

## 保存边界与后续执行

历史推断代码、结果、协议和 SHA256 清单共同记录每轮实验依据。新增导航不重写它们，复算时使用独立输出目录。压缩记录支持统计复算；完整模型重跑还需要固定权重、原始键缓存和相应第三方源码，具体条件见各轮协议。

服务器模型实验执行约束沿用[仓库首页](../../README.md)。重新定位或重跑 GPU 实验前，应按既有项目要求报告资源计划，并单独保存新一轮参数、结果与局限。当前模块二结果不支持整体准确率已可靠提高的结论。
