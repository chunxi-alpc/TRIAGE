> Active scope updated by user: **500 held-out test cases**. See results/run500/REPORT.zh.md and worker500.sh. Earlier 3645-case and 200-case plans below are retained as history.

# ASTER 全量验证（2026-10-23）

本轮承接 `aster_medguide_20261022` 的真实 20 例先导，新增引导诊断、全量配对评价、错误适用信息实验和独立判断翻转实验。代码和已完成结果提交仓库；未完成任务只写运行状态，不预填准确率。最新实际状态见 [结果报告](results/REPORT.zh.md) 和 [机器状态](results/STATUS.json)。

## 数据和已运行诊断

服务器已有 MedGUIDE Parquet、原作者 all_paths.json，以及 Llama-3.1-8B-Instruct 和 Qwen3-8B 权重。7,747 条来源记录中，3,645 条满足原先已实现的 A/B/C 构造条件：3,822 条缺少选项对齐的兄弟路径，280 条缺少第三个动作。全量处理这 3,645 例，不按模型表现筛选。前 40 例为开发集（包含历史 20 例），其余 3,605 例用于正式评价；同时保留全体描述性结果。来源哈希、疾病分布及全部划分 ID 在 dataset_manifest.json。

A 保持规则，B 只改适用动作，C 只改兄弟动作；同例选项、患者、规则顺序一致。替换动作是否出现在其他分支只作标记，不额外删题。该成绩是可构造反事实子集的成绩，不是官方 MedGUIDE 原题成绩，也不是临床验证。

七档增益 `0,0.5,1,2,4,8,16` 已在两模型各 20 个开发病例上运行。测量真实最后决策位置、逐层平均的目标注意力质量和相对于均匀分配的质量；保存 B 的准确率与 logit 差、C 故意增强错误兄弟规则的负对照、g=0 与原生计算的一致性。增益依据开发集 **B 版首位置选项分数准确率**选定，同分选更小者：Llama=2，Qwen=0.5。这不等于自由生成准确率的最优增益。诊断结果、逐层注意力和曲线均在各模型 diagnostic/。

当前干预是 softmax 前的直接注意力分数偏置，作用于全部层、头和查询位置，**不是 SEKA 学习投影**。官方 SEKA 正对照单独运行，不用本诊断替代。

## 全量条件与解码

每模型主任务每例 52 行，合计 189,540 行；随后补全无符号消融每例 6 行，合并后每模型 211,410 行。

| 问题 | 条件 |
| --- | --- |
| 闭卷先验 | 无指南，同一套选项；模型各自划分 K/U |
| 未知适用性 | 原模型、整段指南增强、修改片段增强、CAD；A 无修改，修改片段增强在 A 上为零干预 |
| 已知适用性 | 完整指南＋规则序号注释、只保留适用规则、剪枝＋CAD、只增强适用规则、同时增强两规则、增强适用且抑制兄弟、注释＋有符号引导 |
| 错误适用性 | 对上述注入方法将适用规则与兄弟规则交换；原患者和评价标签不变 |
| 判断翻转 | 同一患者和末级文本，隐藏动作、展示原动作、展示替换动作 |

主任务所有条件均实际贪心生成至多 8 个新词元，Qwen 关闭思考模式。保存原始生成、解析答案，以及首位置每个选项的 logit 与全词表 log-probability。准确率以解析后的生成答案计算；无法解析计错，同时单列解析失败率和选项分数准确率。CAD 在每一步以相同已生成前缀计算 `2 log p(context) - log p(closed)`，不是只对最终选项做重排；CAD 最多 8 词元，遇 tokenizer EOS 停止，普通 generate 使用模型生成配置的 EOS。全部选项字母均验证为单词元。

K 为闭卷生成原答案正确，U 为其余；无效闭卷解析另行计数。B 主结果使用 K；A/C、全体和 U 作为补充。诊断增益扫描以选项分数为准，不能与主任务的生成准确率混合。已人工式逐条查看两模型各 20 条 B 基线输出（由 Codex 复核，不冒充独立人工标注）：明确答案提取均正确，Qwen 有 2 条与首位置选项分数最高项不同，因此两类指标分开保存。

0/10/20/30% 噪声使用种子 11/22/33，按病例哈希排序精确选取错误适用关系。同一病例的 A/B/C 和所有方法共享错误分配，不同噪声比例嵌套。贪心推理无随机采样，所以每个病例的正确/错误关系输出各计算一次，再精确重组噪声实验；不是从准确率估算。无符号同时增强两规则对交换关系不变，复用相同真实输出。

统计包括准确率、B 原答案选择率、C 被改兄弟动作误用率及 Wilson 区间，连续 logit 差，精确 McNemar、病例配对 bootstrap、疾病簇 bootstrap（2,000 次），以及对报告的基线比较作 Holm 校正。噪声下另配对比较有符号引导与注释/剪枝。疾病簇区间与病例区间同时报告，不把同树病例当独立规则来源。

判断实验不补造真假标签；只测动作展示是否引起判断变化。73 种末级文本由独立 agent 作语义研究标注、主 agent 复核：42 种确定型候选、14 种判断型候选、17 种动作/计划/含义不完整文本。公开保存逐病例标签与原文哈希，不发布受限原文，不称专家金标。动作/计划与不完整文本不混入两类型主比较。只有 K 组的原动作/替换动作可解释为与模型先验一致/冲突；U 组只称原动作/替换动作。无法解析的判断对单列，翻转率只在两边均有效的判断对上计算。低翻转不能直接证明不存在动机性推理。

## SEKA 正对照

服务器原有 SEKA 代码和 synthetic/pair_qa_new.jsonl；未找到原始 CounterFact 文件。Hugging Face 和 hf-mirror 下载连接超时，魔搭检索仅找到翻译版，未将其当同源基准。已改从原作者 `https://rome.baulab.info/data/dsets/counterfact.json` 后台断点下载（43 MB）。

`prepare_positive.py` 调用上游 PASTA 的原格式整理函数和真正的 en_core_web_sm；修正文件级 JSONL 换行，不替换句法解析器。记录原始数据和上游代码哈希，不能保证与 SEKA release 字节一致。已补装 spaCy 英文模型，官方 SEKA 评价入口能够正常加载帮助页。

两个 8B 模型分别用 200 条官方 synthetic 样本重建投影，构建与推理都显式 `last10`、min_diff=0.2、top_pct=0.9，避免全层投影错误映射至末十层。使用官方 CounterFact efficacy 指标，5,000:10,000 的 5,000 个样本，baseline 与 SEKA 成对评价；amplify_pos=1.56、batch_size=1、最大输入 300。该指标比较新旧目标首词元概率，**不是**生成字符串前缀准确率。模型版本不同且投影重新构建，属于同基准正对照，不宣称复现论文表格的具体数值。`positive.py` 保留为另外的 200 例原始格式 smoke 入口，正式队列不调用它。

正对照在各自 GPU 的主任务和无符号消融之后运行。若下载、格式整理、投影或官方评价失败，保留日志并明确写 positive_blocked，继续交付已经完成的主实验结果。

## 服务器命令与输出

所有实验在 115 服务器 `/home/gcx/TRIAGE-worktrees/aster-validation` 执行，环境 `/home/gcx/gcx261014`；GPU 0=Llama、1=Qwen。原始文本与详细日志保存在 `/home/gcx/aster_validation_20261022/private`。实验目录不含凭据。

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
RUN=/home/gcx/aster_validation_20261022/private
EXP=experiments/aster_validation_20261022
python "$EXP/prepare.py" --source /home/gcx/aster_medguide_20261022/private --output "$RUN"
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s "$EXP" -p 'test_*.py'
# 每个模型先运行 diagnostic；Llama 示例，Qwen 改模型路径并用 GPU 1。
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python "$EXP/run.py" \
 --input "$RUN/dev.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct \
 --phase diagnostic --limit 20 --output "$RUN/llama_diag.jsonl"
# diagnostic 文件已存在时直接从 worker 接续；新输出位置防止覆盖旧实验。
CUDA_VISIBLE_DEVICES=0 bash "$EXP/worker.sh" llama /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct
# finish_model.sh 参数为模型别名、模型目录、对应 worker PID、原始数据下载 PID。
# 它依次运行补充消融、合并统计、SEKA 投影构建、官方基准；具体命令均保存在脚本。
python "$EXP/collect.py"
```

主任务不覆盖既有输出；重跑需使用新目录或先明确归档旧文件。两轮只读选项分数的部分输出已单独归档，不混入正式生成结果。诊断期间只修正过描述性条件关键词标签，保留原始诊断输入版本和哈希说明；模型提示、输出和正式测试划分没有由此调整。

后台接续由 `finish_model.sh` 完成；`publish.sh` 等待两个模型结束后，收集本轮指定目录的安全数值结果、图、配置和指纹并提交推送。推送失败保留 delivery 日志，不声称交付成功。逐例数值结果压缩为 scores.jsonl.gz；完整生成文字、患者、指南、模型权重与详细运行日志不提交。未运行的可选方法和需要补充的条件标注在 STATUS.json 明列。

参考：[SEKA 官方实现](https://github.com/waylonli/SEKA)、[PASTA 数据整理](https://github.com/QingruZhang/PASTA/blob/main/evaluation/data.py)、[CounterFact 原始来源](https://rome.baulab.info/data/dsets/)。

### CAD output-length correction (2026-10-23)
Interim held-out analysis found Llama CAD frequently copied a condition and hit the eight-token cap before returning an answer. `repair_cad.py` reruns **all** CAD and prune_cad records (A/B/C, both rule assignments), rather than only failures, with 128 tokens and the model generation configuration's EOS set. It preserves the original outputs as `*_8token_original.jsonl`, records token counts and EOS/length stops, then replaces only those arms before final analysis. Other arms retain their original decoding; comparisons therefore use different output budgets and must disclose this. Unresolved parse failures remain errors. The running full experiments are unchanged. `repair_then_finish.sh` waits for each existing worker, performs the correction on its GPU, and resumes unsigned ablation, analysis and positive controls. No additional concurrent GPU model is started. First-position scores remain available separately.

### Interpretation correction (2026-10-23)
See `VERSION_AUDIT.zh.md` for the old-main/current-run comparison. K/U means observed original-action selection/non-selection on one closed-book prompt, not established parameter memory/knowledge absence. `original_action_selection_rate` and `original_action_wilson95` replace the misleading memory-named summary keys in future analysis. Existing historical summaries retain their original schema. The supplied context covers all 3–15 published paths in the dataset's disease entry, not the entire original clinical guideline. Option-order robustness has been queued; no result is yet available.


### Option-order sensitivity supplement (2026-10-23)
200 held-out IDs are selected by a fixed SHA256 rank, without using model outcomes; the selection is saved in option_order_design.json. Two nonidentity cyclic rotations move every option to a different letter, identically across A/B/C. Option text, patient, rules and rule order stay fixed. Gold and intervention labels are remapped with the same bijection. All main choice conditions plus unsigned are rerun (58 records per case, including unchanged judgment diagnostics excluded from order comparisons). CAD/prune_cad receive the same 128-token repair as the original-order run. Inference uses the already selected gains, without retuning. Each model runs after its entire existing main/CAD/unsigned/positive-control queue.

Results compare original versus rotated order on exactly those IDs, mapping predictions back to original action identities. Each rotation reports paired accuracy differences, valid-pair action agreement, both parse-failure counts and exact McNemar with Holm adjustment. Original-order K/U is fixed when comparing strata. Rotations are not treated as independent extra cases. This is a two-rotation sensitivity check, not exhaustive permutation invariance; judgment yes/no labels are not rotated. Raw completions stay private; numeric paired scores and summaries are published after both queues finish.


### Revised 500-case execution
Main results are reused for the first 500 IDs of the original fixed shuffled test split; all 40 development IDs are excluded. Original 3645-run partial outputs are preserved unchanged. CAD128 correction covers all nine CAD/prune-CAD arms per case. Unsigned, statistics and two option rotations use the same 500 IDs; gains are fixed from the earlier development run. The independent positive control uses CounterFact examples 5000:5500 with model-specific projections built from 200 synthetic pairs. GPU0 worker PID86135, GPU1 worker PID86139 at launch; private outputs under /home/gcx/aster_validation_20261022/private/run500. Commands: `CUDA_VISIBLE_DEVICES=0 bash experiments/aster_validation_20261022/worker500.sh llama /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct` and the analogous GPU1 Qwen3-8B invocation. Worker logs and status files record every stage.
