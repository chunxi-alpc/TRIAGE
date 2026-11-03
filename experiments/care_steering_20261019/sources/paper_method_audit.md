# 正式论文与官方代码核验

核验日期：2026-10-19。只做下载、静态审查；本审查未启动 GPU，也不包含本项目实验结果。

## 来源与固定版本

- 论文：[arXiv:2610.20067v1](https://arxiv.org/abs/2610.20067v1)，v1 提交于 2026-10-14 12:06:32 UTC；标题与用户提供的一致。
- [正式全文](https://arxiv.org/html/2610.20067v1) 明确链接 [官方仓库](https://github.com/AheadOFpotato/knowledge_selection_steering)。
- 仓库 commit：48148daf1179f9cfe201fd812c7dbbdc623aa74f，committer 时间 2026-10-14 13:32:29 UTC。
- 完整仓库保存于 sources/official_knowledge_selection_steering；关键文件副本及 SHA-256 见 official_raw/ 和 official_source_manifest.json；论文 HTML 为 paper_2610.20067v1.html。
- 已确认库中没有 data/ 下的 NQ-Swap / MacNoise 原始数据文件。IFEval vendor 有输入数据，不能替代事实冲突源集。

## 论文声称的方法

按全文 §3、附录 B–D：以原始问题和冲突上下文问题为对。用于方向估计的样本须原题及四种改写答对、冲突时采用注入答案。分别收集每层最后提示 token 的 resid_post，按两组均值余弦相似度最低选层，并排除最终输出层；方向 v = mean(clean) − mean(conflict)，不做单位长度归一化。只在所选层的冲突提示最后 token 加 αv。正系数偏向参数知识，负系数偏向上下文。train 估计方向，dev 选系数，test 最终评估；迁移不重新估计方向、层或系数。Qwen3-8B 的公布层编号为 21。NQ-Swap 划分 70/15/15；贪心生成、BF16、eager、max_new_tokens=200；匹配最多前 256 生成 token，both/neither 留在分母。

这些是论文描述，并不证明官方默认脚本完整执行了上述流程。

## 代码与论文的可核验差异

以下行号基于上述固定 commit，均为静态审查结果。

1. **干预位置与持续时间。** src/run_resid_projection_ablation.py:381–408 的 hook 将增量广播到整个 [batch, sequence, hidden] 张量（第 405 行），未限制最后提示 token，也未限制首次前向。第 571–577 行安装后，直到第 749–751 行才移除，覆盖 Stage1/Stage2 的生成。complex_tasks/src/run_resid_projection_ablation.py:749–779 同样广播；其 gate 仅判断 enabled。按论文实现“最后提示 token 一次干预”应明确标为论文方法实现，不是逐字运行官方 hook。

2. **训练筛选。** 核心 dump 的第 1226–1239 行只生成一次 clean 答案；第 1416–1420 行保存整个 stage1_correct_use_wrong_context 组，未在此要求 conflict 输出为注入答案。完整库另有 complex_tasks/src/generate_question_paraphrases.py 与 build_strict_conflict_subset.py；其默认分别为生成 5 个改写、至少 3 个改写且一致率 ≥0.8。不能声称官方库完全没有改写筛选，但这些默认参数不等于正文“原题 + 4 改写全部答对”。

3. **评估样本分母。** 核心 _run_one_mode 在已安装干预 hook 时重跑 clean（第 571–647 行），随后只对该 α 下仍通过 Stage1 的样本跑 Stage2（第 678–690 行）。分母为 len(pass_idxs)（第 753 行）；dev 打分也是 stage2_correct/stage2_runs（第 498–502 行）。因此不同 α 的测试分母可能不同。论文 Table 14 也展示随 α 改变的 valid-generation 数。可靠对比应预先固定评估集合，并把失败、空输出等保留在分母。

4. **迁移时选择系数。** 核心 cross_dataset 模式注释及参数支持使用完整 source NPZ，并在目标 manifest 上默认 50/50 划分 val/test；第 1219–1274 行在该 val 上选系数。complex_tasks/run/run_nqswap_dir_cross_taxonomy.sh 第 146–170 行调用未显式传 train_val_test，复杂 Python 默认 legacy（第 1841–1845 行），遍历目标数据系数。这不等于正文宣称的源 dev 选择后所有目标冻结。不能从静态审查推断作者最终表格究竟用了哪套私有命令。

5. **系数与符号。** core run/run_resid_projection_ablation.sh:31–33 默认 0..3、步长 0.2；all_final.sh:35–37 默认 0.2..3、步长 0.2。复杂任务脚本:74–76 默认 0..2、步长 0.5，并在第 166 行使用 --steer-toward context（内部反转 v）。所以复杂任务中的正数 α 对应正文 v 定义下的负数 α。没有可直接照抄到所有模型和任务的一项固定系数。Table 14 的不同目标列使用不同 selected α，与“固定源 dev”说明存在需要作者解释的不一致。

6. **提示格式。** 核心 dump 第 409–417 行是 raw Question:/Answer: 文本；复杂目录支持 chat template。tokenization、是否禁用 Qwen thinking、截断上限与输出长度应在本实验单独固定并记录。

## 可复现实施建议

本项目若以现有 ClashEval 替代 NQ-Swap，应称“论文方法在现有源数据上的小规模适配”，不称原论文数值复现。按 question 去重后分 train/dev/test；只在 train 筛方向与选层。对 source dev 选择 contextual-only 最优负系数（同分选更小绝对值），目标任务冻结。与正式默认密度一致可用 −3..0 步 0.2；稀疏网格须在结果前写入配置，并标记差异。instruction wrappers 不等于语义改写；若采用 wrappers，必须列为偏差。若足够严格合格的 train 样本不足，应报告失败，不把未筛选均值称为完整复现。

CARE 的患者适用性控制需另做真实模块预测和 oracle 控制的区分；使用金标适用性直接控制只能说明控制机制上限。此点来自实验设计要求，不来自上述论文。

## 尚未核验或完成

本审查没有执行官方表格全量复现，没有检验全部脚本的运行正确性，没有运行 GPU，也未验证 CARE 上传原文。论文和代码的差异已记录，但未经作者确认，不能断言哪条实现产生了原论文最终结果。
