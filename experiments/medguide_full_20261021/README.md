# MedGUIDE 全量冻结 A1 实验

本实验固定使用 MedGUIDE-MCQA-8K 的全部 7,747 条记录。当前运行状态以 `results/summary.json` 为准；该文件尚未生成时不能声称全量实验已完成。

## 固定来源与隔离

- main 基线：`4e156cdecad12406f4c6d0adf888111fd07d4f22`。
- A1：PR #12 源提交 `8fb87d15576b88500f44a88f04799267d9f2b5ee`；遵守 `configs/care_module1_for_module2.json`。原提示（包括虚构场景措辞）、固定示例、NF4 Phi-4 + TrialMatchAI LoRA、batch=8、max_new_tokens=384、greedy、一次格式重试及 T/F/U/null 均不变。
- 数据：`MedGUIDE/MedGUIDE-MCQA-8K` revision `1460ec478280a7295040b92f7eb985327f77dcbe`，原始 train split 全部用作评估，不训练、不调参、不再划分筛选样本。7,747 是实际读取 Parquet 后核实的数量。
- 指南：论文链接的原作者仓库 `Submission-MedGUIDE-187A/data/all_paths.json`，包含55套完整指南候选路径；当前数据涉及其中53套。保存内容哈希作为固定版本。每例按 disease 取完整候选集合，绝不按该例正确 path 选取。
- 每条来源路径按原顺序写为临床经过与末端推荐动作，保留全部分支。该表示包含历史治疗过程和标题，属于原作者指南派生规则，不等同于专家独立标注的完整自然语言规则。
- 输入仅含 profile、固定提问、options、独立来源的全部候选规则。原始 prompt、answer、answer_text、病例 path、质量分数不进入被评估模块。
- 没有逐规则 T/F/U gold，不计算或宣称逐规则准确率。病例 path 只在评估程序中哈希分组；没有把 gold path 当成适用性标签。
- 源规则映射覆盖率与 A1 实际解析覆盖率、T/F 信号覆盖率分别统计。输入规则共57,408条，原始记录全部保留；失败算入完整分母。

## Qwen 对照

沿用历史 `Qwen3-4B-Base` 模型，使用 bfloat16、候选选项字母的下一词元 logits 评分。两组输入逐字相同，均含全部候选规则、病例和选项。

baseline 不干预；a1_attention 使用冻结 A1 状态，只在规则动作词元上施加 T=+1/F=-1，U/null 为零。沿用 PR #12 使用的层18–23、强度1、最后一个查询位置、direct attention 参数，不训练或调参。所有 A1 失败病例仍运行两组 Qwen；无信号对照应严格一致。`qwen_engine.py` 保留历史注意力实现，仅去掉未使用的旧数据渲染导入；新渲染与可变选项评分在 `qwen_run.py`。

同输入指的是两组 Qwen 输入一致；不是宣称与官方裸 MCQA prompt 或历史合成任务可直接比较。

## 资源与运行

仅在115服务器、gcx261014环境运行。使用 GPU 0/1/2/3，每卡一个分片，先执行 A1 再执行两个 Qwen 对照。预估 A1 每卡20–35GiB、Qwen10–20GiB、4–10小时；实际资源以模型运行 manifest 为准。运行稳定后不持续监控。

环境中的 bitsandbytes 使用既有 `/home/gcx/care_steering_20261019/.deps`。`environment.txt` 和各运行 manifest 记录实际依赖版本；`model_fingerprints.json` 保存全部模型权重与配置 SHA256。

服务器复算（从仓库根目录）：

```bash
source /home/gcx/gcx261014/bin/activate
python experiments/medguide_full_20261021/fetch_data.py
python experiments/medguide_full_20261021/prepare.py
nohup bash experiments/medguide_full_20261021/run.sh > experiments/medguide_full_20261021/private/run.log 2>&1 < /dev/null &
```

脚本可从已完成记录续跑。请使用独立输出目录进行从零重跑，不删除历史结果。`run.sh` 等待四个分片结束后自动汇总并提交、推送派生结果，不 merge。运行失败也输出完整分母报告并标注 incomplete。

## 输出与许可

数据卡许可为 TBD，说明用于学术研究；原数据、指南文字、原始提示与模型长文本均仅存服务器 private/，不提交或重新发布。公开到私有仓库的逐条派生结果仅含稳定行号、分组、哈希、预测字母/logits、正确性与状态，不含病例、指南或 gold 答案文本。

`results/summary.json`：完整分母、按癌种/指南/路径哈希指标、配对改善/损害、GPU消耗。`predictions.jsonl` 与 `failures.jsonl`：逐条派生结果与所有失败记录。`REPORT.zh.txt`：实际运行结论。没有独立规则真值，不能从MCQA正确率推断逐规则准确率。

预训练数据重叠无法排除；本轮未做任何训练。冻结输出上限可能导致多规则病例格式失败，这也是全量迁移的实际结果，不能丢弃这些病例或据结果修改设置。
