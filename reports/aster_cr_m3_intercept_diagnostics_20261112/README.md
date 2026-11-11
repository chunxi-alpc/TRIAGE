# ASTER-CR M3 截距校准与多选解码诊断

[中文报告](REPORT.zh.md) · [实验计划](EXPERIMENT_PLAN.zh.md)

服务器目录：`/home/gcx/aster-cr-diagnostics-e3cv-20261111/reports/aster_cr_m3_intercept_diagnostics_20261112/`

所有计算在服务器的gcx261014环境执行，没有本地实验、GPU推理或模型重训。

## 文件说明

- `configs/`：固定实验配置、原五折划分、独立截距及来源、逐折开发节点分数、MedPIC开发选项分数、校准参数和冻结选择。
- `predictions/<scope>/nodes.jsonl.gz`：逐节点各消融分数、预测、标签与来源；MedPIC含题目和选项ID。
- `predictions/<scope>/pairs.jsonl.gz`：配对分数差、修正差、全对、关系、证据方向与预算。
- `predictions/<scope>/paths.jsonl.gz`：预测自主分支回放的完整路径、叶节点和正确性。
- `predictions/medpic_*/questions.jsonl.gz`：逐题预测答案、金标准、以上均非与题型；`question_pairs.jsonl.gz`是去重整题配对。
- `medpic_candidate_scores/{dev,test}.jsonl.gz`：每道题及背景变体的全部合法集合、选项数量k、各方法得分和排名。
- `medpic_candidate_scores/ranking_changes_*.csv`：加截距前后的答案、大小和排序变化。
- `medpic_candidate_scores/native_pair_changes_*.csv`：保持/翻转配对涉及的题目和选项变化。
- `medpic_candidate_scores/none_competition_test.csv`：以上均非与最优其他集合的边际分数变化。
- `medpic_candidate_scores/size_distribution_*.csv`：候选集合大小及预测大小分布。
- `metrics.csv`：汇总指标；比例均用0–1。
- `source_bootstrap.csv`：估计、95%区间、有效重采样次数，包括差值、错误改对和正确改错次数的区间。
- `transitions.csv`：错误改对、正确改错、保持正确与保持错误。
- `figures/`：PNG/PDF图；`code/`：分析、复算及报告代码；`logs/`：详细运行日志（仅保留在服务器）。

scope中dev/test为原20/40来源，new60为已有60来源，cv_fold*_common为各折验证，cv_pooled_common为逐例合并；medpic_dev/test为原93/188题。不要同时合并各折和pooled，以免重复计数。汇总表包含补充题型分层，以目录根部三个CSV为完整结果。

## 从保存预测复算

在服务器仓库根目录执行：

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/aster-cr-diagnostics-e3cv-20261111
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python reports/aster_cr_m3_intercept_diagnostics_20261112/code/recompute.py
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python reports/aster_cr_m3_intercept_diagnostics_20261112/code/report.py
```

recompute.py只读取本目录保存的逐例预测，以原评价函数重建汇总，不读取原始特征、不加载模型。report.py从汇总表和候选集合记录生成中文报告、排序诊断与图。上述命令只更新本次目录的派生文件。

code/run.py记录完整生成流程，通常无需重跑。复算代码复用仓库中`experiments/aster_cr_retrain_paired_v2_20261111/evaluation.py`的指标定义。模型位于`/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111/models/E1/`；原逐例结果在同仓库`reports/aster_cr_diagnostics_e3cv_20261111/`。原文件保持不变。

## 口径

85/110是MedPIC选项级保持/翻转配对数，不是整题数。整题精确匹配、配对全对率、关系正确率分别统计。single/multiple按金标准答案大小划分，不改原合法集合。解码C/D/E的节点scores为底层选项分数，最终集合得分以candidate_scores为准。原截距不改变同一配对两端分数之差。

每项比较使用同一来源抽样权重，draws=2000，稀少分层的defined_draws可能较小。未把不显著当成非劣。历史测试已有使用记录，所有解码方案只作消融。
