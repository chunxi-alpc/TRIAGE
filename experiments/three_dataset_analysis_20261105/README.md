# 三数据集补充分析

仓库文档：[实验与结果总入口](../../reports/three_dataset_analysis_20261105/README.md)、[实验说明](../../reports/three_dataset_analysis_20261105/EXPERIMENTS.zh.md)、[结果分析](../../reports/three_dataset_analysis_20261105/ANALYSIS.zh.md)、[汇总结果与参数](../../results/three_dataset_analysis_20261105/README.md)。

自动预实验，复用已有标注和历史结果。按用户方案执行，零新增标注。所有计算均在 gcx@10.29.128.115 的 gcx261014 环境进行。

结果目录：/home/gcx/three_dataset_analysis_20261105/results

数据入口：/home/gcx/experiment_data_reuse_20261105/data_inventory.json

主报告：/home/gcx/three_dataset_analysis_20261105/results/REPORT.zh.md

依次使用以下入口：

1. prepare_medpair.py：复用原 build_prompt 和选项 token 映射，准备 Full/High 与共同底稿补回。540 个条件记录合并为 538 个唯一输入。
2. analyze_offline.py：旧 MedPAIR 删除对照、MedGUIDE、MedEinst，以及单列的历史补充结果。
3. score_medpair.py --stage pilot：GPU 0 的 20 道开发题，55 个输入。
4. score_medpair.py --stage full --shard 0 和 --shard 1：分别设置 CUDA_VISIBLE_DEVICES=0、1；复用试跑结果，只补缺失记录。启动前先汇报资源和时长并检查可用显存。
5. analyze_medpair.py：汇总本轮完整选项分数、补回对照和融合。
6. make_report.py：四张结果表、CSV 和中文报告。
7. python -m unittest test_analysis，以及 verify_results.py：方案要求的融合边界情况、选项映射、概率、样本配对和分组检查。

score_medpair.py 每个输入只调用一次原 score_answer_choices；通过前向钩子保留原始 logits，不增加第二次推理。使用 bfloat16、原始纯文本提示、带前导空格的答案字母；不使用聊天模板，不加载 adapter。完全相同输入通过字符串及候选集合比较复用，不计算哈希。

MedGUIDE：按 disease（指南树）排序后用 Python random.Random(42) 打乱，前 11 棵为开发集，其余 42 棵为测试集；149/651 个原始病例。A/B/C/D 遮挡输入相同时，基础错误率只计 A 一次。84 组 CS 全部来自测试集的一棵树，跨树区间留空。

MedPAIR：沿用 processed 数据中的 dev/test 标记（91/89）；补回为 31/29。高相关句与明确不相关句按 token 长度差最小选择，并列按高相关句、再按不相关句的原顺序选择。两种补回插在两个被删除句子中较早的位置，原句文本和句子编号不改。历史 High 的正确数为 56/89，本轮为 55/89；删除分析与本轮融合分开。

MedEinst：每侧独立取 46 候选最高分；原始分数文件路径、行号、病例对 ID、诊断及原划分逐条保存。恢复的数据以评分文件为来源；不重建或改写病例。M1 两诊断分数、Work3 自由生成结果和 token 均值评分独立列为补充。

融合：概率平均权重网格 0/0.25/0.5/0.75/1，并列选 Full 权重大者；两个 CAD 方向的 α 网格为 0/0.25/0.5/1/2/4，并列选较小值；共享双权重学习式融合用开发 NLL 加 0.01 L2 拟合，允许负权重。MedGUIDE 题内先平均变体，再对病例平均。仅开发集选参。

统计：2,000 次配对 bootstrap，种子 42；MedPAIR 按题、MedEinst 按病例对、MedGUIDE 按树。单棵树无法估计跨树区间。所有分子和分母保存于 JSON/CSV。

逐样本患者输入和分数留在服务器；本地交付只包含汇总报告、结果表和参数。Q% 暂不填。不训练内部干预模块，不混并不同协议的历史结果。

## 将已完成结果整理到仓库

在本目录执行 `python export_repository.py`。脚本只复制指定的汇总报告、CSV/JSON、参数和运行概况，并生成服务器文件索引；不调用模型，不修改原运行产物。报告保存在 `reports/three_dataset_analysis_20261105/`，汇总数据保存在 `results/three_dataset_analysis_20261105/`。默认运行目录为 `/home/gcx/three_dataset_analysis_20261105`，可用 `--run-root` 与 `--repo-root` 指定目录。

已完成的验证记录随结果保存。仅整理或修改文档时检查链接及文件完整性即可，无需重复 GPU 推理或全量统计。

## 服务器依赖与来源

本目录收录到 TRIAGE 后，实验输入和历史运行目录继续使用原位置。common.py 从 `/home/gcx/experiment_data_reuse_20261105/data_inventory.json` 找到原 MedPAIR 仓库 `/home/gcx/work3-migration-20261104/target`；prepare_medpair.py 和 score_medpair.py 从该仓库导入 ClinicalCase、build_prompt 与选项评分函数。MedGUIDE 和 MedEinst 复用 inventory 及脚本中记录的服务器历史分数。仅克隆 TRIAGE 不包含这些原始数据和依赖。

原 config.json 的 code_root 记录实际运行时的代码位置，保留原样。当前代码入口为 TRIAGE 的本目录；export_repository.py 默认写入所在仓库，可通过 `--repo-root` 指定其他位置。整理不重跑模型，也不复制患者输入和逐题输出。

MedGUIDE 离线复算还动态导入 `/home/gcx/TRIAGE-worktrees/qwen800-methods/experiments/aster_validation_20261022/mechanism_20261023/run.py` 的原分支提示函数。Work3 补充分析读取 Parquet，需要 pandas 和可用的 Parquet 引擎；本次 gcx261014 环境已满足。
