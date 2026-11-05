# 修订遮挡、关键事实与融合实验

[结果分析](../../reports/semantic_masking_followup_20261105/ANALYSIS.zh.md) · [冻结报告](../../reports/semantic_masking_followup_20261105/REPORT_full.zh.md) · [结果与快照](../../results/semantic_masking_followup_20261105/README.md)

本目录保存最终服务器运行代码及固定预测的补充分析脚本。MedGemma-27B 负责自动语义标注，Qwen3-8B 负责“是／否”候选评分。没有训练内部干预模块。固定 40 份指南、379 个统计节点和原 50 个构造条件；修复后 39 个构造节点可用。

## 只用保存分数复算，不需要 GPU

以下在服务器 `/home/gcx/TRIAGE` 仓库根目录运行；如使用其他检出目录，先切换到对应目录。快照包含复算所需代码、输入、自动标注、分数和开发／测试划分，不需要重新加载模型或完整原始数据集。

```bash
source /home/gcx/gcx261014/bin/activate
replay_dir=$(mktemp -d /home/gcx/semantic_masking_replay_XXXXXX)
python -m zipfile -e results/semantic_masking_followup_20261105/snapshot.zip "$replay_dir"
CUDA_VISIBLE_DEVICES='' python "$replay_dir/analyze_revision.py" --stage full
python experiments/semantic_masking_followup_20261105/summarize_predictions.py \
  --run-dir "$replay_dir" \
  --output "$replay_dir/results/supplementary_analysis.json"
```

复算输出位于新目录的 `results/`，不会覆盖仓库冻结结果。`analyze_revision.py` 按开发来源重新选择融合参数并计算全部主指标；`summarize_predictions.py` 只汇总固定预测，不拟合任何参数。它输出失败类型、概率响应、背景变化、代表病例、融合改对改错和每节点一对的融合 P_b。

已在服务器从发布快照执行一次主指标复算，全部指标与最终运行一致：[publication_replay.json](../../results/semantic_masking_followup_20261105/publication_replay.json)。主分析用到 NumPy 2.5.4、scikit-learn 1.9.1；补充分析只用 Python 标准库。原模型运行环境还包括 PyTorch 2.8.0、Transformers 4.57.6。

## 主要文件

| 文件 | 用途 |
|---|---|
| `predicates.py`、`revision.py` | 完整逻辑标签、受控病例变化、原文位置与输入构建 |
| `source_annotations.py`、`review_semantics.py`、`dependency_rules.py` | 原文语义依赖、节点对应与未知处理 |
| `conservative.py`、`recover_reviews.py` | 保守遮挡与已保存标注结果恢复 |
| `score.py` | 统一二分类分数、按完整输入与模型设置复用 |
| `analyze_revision.py` | 实际删除配对、主 P_b、开发集选参及单／双路融合 |
| `run_revision.py`、`run_followup.py` | 原服务器修订与定点补跑流程 |
| `summarize_predictions.py` | 发布时新增的固定预测补充汇总 |
| `tests/` | 已通过的 21 项实现测试 |

`config.json` 保留实际运行时的绝对数据和模型路径。重新标注、重新推理以及涉及原文的测试需要原服务器数据与模型；不应直接把历史启动脚本作为离线复算入口。历史抽样记录中的 80 个候选不等于最终构造数量，最终固定条件池为 50、有效条件为 39。完整配置、标签和逐例输入均在结果快照中。
