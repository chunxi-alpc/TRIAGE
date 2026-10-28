# 模块二机制预实验 · 2026-10-28

本目录对齐 main `e3d2bfb8da539c15527d607c155af73e44788eeb` 的逐 token 子空间估计、单位方向、双状态查询筛选和指南前缀缓存协议。它重新运行真实预训练模型，不把 PR #38 的患者前置结果解释为当前协议结果。

[固定协议](PROTOCOL.zh.md) · [完整结果](RESULTS.zh.md) · [全部案例与筛选](results_analysis/CASE_STUDIES.zh.md)

## 研究问题

1. 方向能否在未参与拟合的规则上保持有向变化？
2. 离线患者前置学到的方向能否被在线指南前置查询读取？
3. 正负联合编辑是否分别将相反患者的读取推向各自正确规则？

这是模块二的机制实验，使用 oracle 规则符号。A/B 编号选择为次要观察；不包含完整模块一、处置生成或临床效果验证。复用的开发指南及旧划分已在前轮公开，本轮不是新的盲测数据。

## 复现

固定环境：Python 3.12、PyTorch 2.8.0+cpu、Transformers 4.57.6、NumPy 2.3.5。模型为 `Qwen/Qwen3-0.6B-Base` 的 revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。精确模型/分词器文件摘要和运行配置见 `results/fingerprint.json`。

### 只复算已完成的结果和图

从仓库根目录执行。先按 [RAW_MEASUREMENTS.json](RAW_MEASUREMENTS.json) 将下载的测量数组放到 `results/measurements.npz` 并核对 SHA256。将发布的压缩记录还原为分析输入：

```bash
gzip -dk experiments/module2_mechanism_20261028/results/causal/records.jsonl.gz
gzip -dk experiments/module2_mechanism_20261028/results/causal_boundary_readout/records.jsonl.gz
```

以下命令不加载预训练模型：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/triage-mpl
python scripts/analyze_module2_mechanism.py \
  --offline experiments/module2_mechanism_20261028/results/measurements.npz \
  --online experiments/module2_mechanism_20261028/results/measurements.npz \
  --operators experiments/module2_mechanism_20261028/results/operators.npz \
  --triplets experiments/module2_mechanism_20261028/inputs/triplets.jsonl \
  --selection experiments/module2_mechanism_20261028/results/selection.json \
  --causal experiments/module2_mechanism_20261028/results/causal_boundary_readout/records.jsonl \
  --status experiments/module2_mechanism_20261028/results/causal_boundary_readout/status.json \
  --conditions experiments/module2_mechanism_20261028/results/causal/conditions.json \
  --output experiments/module2_mechanism_20261028/results_analysis \
  --figures experiments/module2_mechanism_20261028/figures
```

该数组包含拟合所需的逐 token 二阶矩、片段均值及查询测量等充分统计量，不是模型完整序列的逐 token 激活转储；重新提取所有键需运行固定模型。分析图中的区间条件于已经冻结的方向，仅对 3 个确认指南做聚类重采样，属于描述性区间。

### 从固定权重重新提取并干预

安装 [requirements.txt](requirements.txt) 中的固定依赖，将 `MODEL_DIR` 设为固定修订的本地模型目录，并使用新的结果目录。下列 `M2_RUN_DIR` 不应指向本仓库已发布的结果：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/triage-mpl
M2_RUN_DIR=/tmp/triage-m2-reproduction
python scripts/run_module2_mechanism.py --model "$MODEL_DIR" --output "$M2_RUN_DIR"
python scripts/run_module2_mechanism_causal.py --model "$MODEL_DIR" --source "$M2_RUN_DIR"
python scripts/replay_module2_boundary_readout.py --model "$MODEL_DIR" --source "$M2_RUN_DIR"
python scripts/analyze_module2_mechanism.py \
  --offline "$M2_RUN_DIR/measurements.npz" \
  --online "$M2_RUN_DIR/measurements.npz" \
  --operators "$M2_RUN_DIR/operators.npz" \
  --triplets experiments/module2_mechanism_20261028/inputs/triplets.jsonl \
  --selection "$M2_RUN_DIR/selection.json" \
  --causal "$M2_RUN_DIR/causal_boundary_readout/records.jsonl" \
  --status "$M2_RUN_DIR/causal_boundary_readout/status.json" \
  --conditions "$M2_RUN_DIR/causal/conditions.json" \
  --output "$M2_RUN_DIR/analysis" --figures "$M2_RUN_DIR/figures"
```

最终答案诊断使用经过实际分词边界核验的 ` A`／` B` 续接，来源为 `causal_boundary_readout`。原 `causal` 裸字母分数及所有注意力记录仍保留，两次运行的对应前向逐条核对。具体原因和范围见[读出修正](READOUT_CORRECTION.zh.md)。

## 实现对应

| 实现 | 内容 |
|---|---|
| [mechanism_extraction.py](../../src/module2/mechanism_extraction.py) | 两种提示顺序、逐 token 键、实际 GQA/RoPE 查询、规则注意力 |
| [mechanism.py](../../src/module2/mechanism.py) | 逐 token 二阶矩、单位位移、当前 Sep、双状态选头、A/B 相对对照 |
| [guideline_prefix.py](../../src/module3/guideline_prefix.py) | 未编辑指南缓存、逐患者键平移、完整患者后缀及精确后段优化 |
| [分析脚本](../../scripts/analyze_module2_mechanism.py) | 全体分母、所有方法、成对案例选择、三张真实结果图 |

测试使用随机微型模型或解析数组验证实现，不作为科学结果。研究产物只来自上述固定预训练模型。检查点允许以相同源码/输入指纹恢复；如果运行恰在写入某个 NPZ 时被外部中断，应先移走损坏的单个检查点，再以原指纹恢复，不能混用不同配置的缓存。

发布文件的大小和 SHA256 见 [ARTIFACTS.json](ARTIFACTS.json)；运行覆盖与数值核验见 [RUN_VALIDATION.json](RUN_VALIDATION.json)。
