# 修订实验结果与逐例快照

[结果分析](../../reports/semantic_masking_followup_20261105/ANALYSIS.zh.md) · [运行报告](../../reports/semantic_masking_followup_20261105/REPORT_full.zh.md) · [CPU 复算命令](../../experiments/semantic_masking_followup_20261105/README.md)

最终运行：`/home/gcx/semantic_masking_followup_20261105`，2026-11-05 完成。自动预实验，未训练内部干预模块。2,994 条评分全部成功；实际新增 25 条，其余 2,969 条使用相同完整输入及模型设置的已有分数。

| 产物 | 内容 |
|---|---|
| [metrics_full.json](metrics_full.json) | Q、五种输入、关键事实、全部融合的主指标和分组结果 |
| [supplementary_analysis.json](supplementary_analysis.json) | 固定测试预测的失败类型、代表病例、融合改对改错与配对 P_b |
| `fusion_selection_full_*.json` | 两种参考输入的开发集选参、单路校准与逻辑回归配置 |
| `fusion_predictions_full_*.jsonl` | 原始与构造测试实例的各方法预测 |
| [scores.jsonl.gz](scores.jsonl.gz) | 2,994 条压缩分数：样本身份、标签、概率、对数分数、预测及复用来源 |
| [q_nodes_full.jsonl](q_nodes_full.jsonl) | 固定统计节点的依赖标注和有效分母信息 |
| [construction_specs.jsonl](construction_specs.jsonl) | 完整可执行条件、固定事实、指标变化与构造状态 |
| [input_changes_vs_prior.jsonl](input_changes_vs_prior.jsonl) | 相对前一修订的逐条输入变化 |
| [followup_summary.json](followup_summary.json) | 定点补跑范围、标注失败、新增推理与最终数量 |
| [snapshot.zip](snapshot.zip) | 完整可离线复算的输入与分数快照，约 12.2 MB |
| [publication_replay.json](publication_replay.json) | 从发布快照进行 CPU 复算的结果：全部指标一致 |
| [validation.txt](validation.txt) | 21 项服务器测试的执行记录 |

`snapshot.zip` 解压后包含运行代码、`config.json`、`tests/`、`data/` 和 `results/`，没有额外顶层目录。`data/edits_full.jsonl` 保存逐节点原文、删除与补回变化；`data/inputs_full.jsonl` 保存各模型输入；`results/scores_full_0.jsonl` 和 `results/scores_full_1.jsonl` 保留完整渲染输入及评分设置。仓库外层的 `scores.jsonl.gz` 是便于统计的精简导出，不重复保存提示文本。

快照不包含完整源数据集、模型权重或原始运行日志；它已包含复算本次保存分数所需的数据。发布时新增的分析文档和 `summarize_predictions.py` 位于仓库对应目录，不改写已完成运行的快照。

`sources.jsonl`、`sampling.json` 记录固定来源与划分；历史候选数和最终可构造条件数须按分析报告区分。`verification_*.json`、`completion_check.json` 与 `input_exclusions_full.jsonl` 保留运行检查和未入组原因。旧目录和旧分数保留，修订原因见分析第 7 节。
