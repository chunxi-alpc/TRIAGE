# 已训练 J 的零截距回放

J_zero 比 R0 净多走对 4 条自然路径和 1 条干扰路径。两项来源级 95% 区间均包括零。五种分数在 761 个自然节点上都答对 754 个，完整路径的差异来自动态回放涵盖的其他节点。

| 分数 | 自然路径正确数 / 1,712 | 干扰路径正确数 / 5,136 |
|---|---:|---:|
| R0 | 1,549 | 4,722 |
| R1 | 1,546 | 4,685 |
| J_zero | 1,553 | 4,723 |
| J_shared | 1,549 | 4,690 |
| J_independent | 1,550 | 4,687 |

固定 J 的修正量后，移除共享截距净多走对 4 条自然路径、33 条干扰路径；移除独立截距分别净多走对 3 条、36 条。其中独立截距下的干扰路径差为 +0.701 个百分点，95% 区间为 [+0.040,+1.398]。

J_zero 相对 R0 修复了 4 条自然路径的首次错误。沿正确答案方向的节点分数由 −0.447 至 −0.037 logit 移至 +0.114 至 +2.008。干扰路径修复 3 条，另有 2 条从正确变为错误，对应的首次错误节点由 +0.437、+0.088 移至 −0.041、−0.613 logit。

在仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m experiments.aster_m3_missing_controls_20261123.replay_a
```

[A_metrics.tsv](A_metrics.tsv) 给出三种评价单位的合并计数和准确率。[A_comparisons.tsv](A_comparisons.tsv) 给出全部五个对照的错误转移、净变化和 2,000 次来源自助法区间。区间和差值以比例为单位，乘以 100 得到百分点。[A_source_transitions.tsv](A_source_transitions.tsv) 给出各自来源的四向计数与净变化。每项对照按 rhs → lhs 记录转移。

[A_scores.jsonl.gz](A_scores.jsonl.gz) 保存 158,161 个折外输入的五种分数、归档修正量和自然节点评价标签。[A_paths.jsonl.gz](A_paths.jsonl.gz) 保存方法实际走过的节点、预测、叶节点和首错位置。[A_first_error_boundaries.jsonl.gz](A_first_error_boundaries.jsonl.gz) 同时记录调整前和调整后各自的首错节点、两种方法在该节点的分数，以及是否实际到访。路径全部正确时，对应首错字段为空。

[A_verification.json](A_verification.json) 记录归档逐条验证和配对差值断言。全部 27,392 条归档比较路径与重新回放的记录逐字段一致。5,003 个 A/P 和 1,342 个 C/F 配对在加截距前后的分数差最大浮点偏差为 1.42 × 10⁻¹⁴ logit。
