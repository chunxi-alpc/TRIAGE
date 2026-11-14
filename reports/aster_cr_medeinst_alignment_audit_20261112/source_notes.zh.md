# 原始任务与证据来源

## MEGA-CDP 原生任务及 ASTER 改写

MEGA-CDP 原论文要求利用指南产生临床决策路径，包含单轮病例及多轮问诊两种设置，并用路径一致性评价。它并非原生的 A/P/C/F 二元配对基准。ASTER 从其决策树节点和病例构造“当前条件成立/不成立”的二元任务，再生成 A/P/C/F 配对。本次报告中的“MEGA F”专指这一项目内构造的患者事实反事实配对。[原论文](https://arxiv.org/html/2608.26592v1)

原始依据是服务器 `/home/gcx/TRIAGE/data/MEGA-CDP/train_val/` 下的 `decision_trees.jsonl`、`vignette_cases.jsonl`、`case_paths.jsonl`、`gt_paths.jsonl` 和指南 Markdown。666 对 F 的原始病例、条件节点、原始路径标签均已核对；修改侧标签仍属于合成反事实及既有审查依据，不是新患者的临床真值。

## MedEinst 原生任务及 ASTER 改写

MedEinst 原生目标是反事实鉴别诊断：从病例输出疾病名称，比较 control 与 trap 上的诊断表现。论文覆盖 49 种疾病，原始指标包括 control 准确率、两侧全对率和条件化的 Bias Trap Rate。构建过程使用 DDXPlus 证据及模型改写；论文所称抽样临床质量审查，不能自动转移为当前训练子集、当前单事实标准或本次审查的临床专家认证。[原论文](https://arxiv.org/html/2601.06636v1)

发布文件包含病例编号、control/trap、年龄、性别、叙述及疾病标签，没有逐病例 DDXPlus ID 映射、生成时疾病概率、所用证据条目及专家裁定记录。数据卡宣称 train 为 10689 对；本服务器实际原始文件是 10609 对、21218 行，test 为 5383 对、10766 行。公开页面当前显示总行数 31984，与本地两文件总行数一致；本报告按实际文件计数，不据此推断多出的 80 对去了哪里。[数据卡](https://huggingface.co/datasets/zhui711/MedEinst)

ASTER 将每对两侧发布诊断按字典序排序，作为共同的两个候选；第一个候选映射“是”，第二个映射“否”。这两个候选由配对真值构成，属于预先给定两候选的诊断比较，不能称作原生 49 类或开放诊断准确率。两侧候选相同，是实现事实；候选空间依据标签构造，是任务改写事实；两者均不能证明修改足以翻转真实诊断。

## 训练来源链

`raw/medeinst_train.jsonl` 的 10609 对 → 程序筛选 10111 对 → Qwen3-8B 有标签语义审核及确定性排除后 8228 对 → 按 148 个发布诊断变化类型、种子 20261112 抽取并冻结 4368 对 → `medeinst_train_selected.jsonl` → `fit_pairs.jsonl` 及其实际提示输入。

程序初筛共排除 498 对；原因计数可能重叠。原始 train/test 有 339 个 case_id 交集，已不在选中 4368 对内。历史 `agent_overrides.json` 是进一步排除，不是补充放行。历史 Qwen 审核直接展示两侧发布标签，而且提示明确允许多个事实变化；因此其“通过”回答的是更宽泛的相对诊断可解释性，不能替代本次单事实、先盲后揭示的标准。历史报告中的“已审核”不能读作“严格 MEGA F 已验证”。

所有 4368 对的两侧叙述与发布原始记录完全一致；二元标签映射、共同候选及实际提示均核对一致。底层 DDXPlus 映射和逐例生成标签依据仍缺失；本次医学判断使用发布叙述、固定候选与独立 AI 审查，不声称恢复了缺失证据。

## 实现证据索引

以下路径以原实验目录 `/home/gcx/aster-cr-retrain-paired-v2-20261111/` 为根：

- `reports/aster_cr_multisource_20261112/code/prepare.py`：原始配对、重复排除及 diagnosis_prompt。
- `reports/aster_cr_multisource_20261112/code/worker.py:26–35`：历史 Qwen 审核提示、可多事实及标签展示。
- `reports/aster_cr_multisource_20261112/code/lifecycle.py:9–58`：确定性排除、接受池、候选排序和二元标签。
- `reports/aster_cr_multisource_20261112/balanced_10006/code/build_balanced.py:11–58`：冻结 10006 对的准确来源。
- 同目录 `m1_impl.py:15–56,116–160`：归因、代表样本、排序和选头。
- 同目录 `training_impl.py:197–244`：M2 加权标准化、C/F 证据子空间、A/P 干扰子空间及正交处理。
- 同目录 `training_impl.py:262–310`：M3 仅回归 A/P，C/F 作为原分数修正约束，MEGA 节点校准截距。
- 同目录 `evaluation_impl.py:265–300`：保留率是有符号分数差比值，不是正确率。

用户指定的最后一个 `DATA_AUDIT.zh.md` 不在原实验工作树；已在 `/home/gcx/aster-cr-balanced-publish-20261112/reports/aster_cr_final_method_10006_20261112/DATA_AUDIT.zh.md` 定位并读取。其余五个入口均已读取。既有汇总仅用于定位，审计结论来自原始记录、实际代码和本次计算。
