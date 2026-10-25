# FaithEval 与 AdaSEKA 校准来源重合审计

协调器已在 115 服务器下载三份官方固定对象，并核对 SHA256、大小及行数：CF 1000、IC 1500、UA 2492。官方 revision/文件指纹见 `work/faitheval_access/source_manifest.json`。下载成功不等于完成独立来源确认；本模块准备 CPU 审计，尚未在本地运行。

`audit_overlap.py` 只公开 ID 哈希、问题哈希、原始行号、来源类别和汇总。FaithEval 只读 `subset/qid/id/question` 列，完全不加载 context、choices、answer(s)、answerKey 或 justification。问题只在服务器内存中计算哈希，不返回方法开发者、不写入日志、聊天或可提交结果。所有 4992 条测试例原样保留，不生成筛选后测试集。

HotpotQA 使用已登记原始文件 SHA256 `26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316`。原作者固定 builder 明确使用 `data[:self.max_samples]`；脚本逐序读取其前 200 条，不打乱、不按有利问题另选。只取 `_id/question/context` 参与来源审计；答案和 supporting_facts 不用于重合匹配。

校准成员分为两层：

- **请求范围**：原文件前 200 条，作为保守的潜在曝光集合。
- **实际筛选记录**：读取已完成 build report 与匹配 SHA 的 selection trace，核对实际 `--max_samples 200`、源文件和 builder 指纹。用原作者 `assemble_texts` 函数重建两条提示，仅在服务器内计算 `text_h_sha256`、`text_hp_sha256`，用两者同时相等映射到原始行。没有模型前向；若用了 chat 模板，只在 CPU 加载本地 tokenizer。缺失/重复歧义或无法映射时记 unknown/ambiguous，不按 trace 第 N 行猜原始第 N 条。

报告可同时接收 instruct 与 Base 的实际 Hotpot build report，分别记录 included/rejected/not_observed/unknown。没有通过来源与 trace 核对时，`actual_calibration_membership_status` 为 unknown；不能把请求 200 条直接写成“200 条全部参与投影”。

匹配规则在读取测试前写定：`subset` 明确为 HotpotQA 的记录，其完整 `qid` 与原始 `_id` 相同，才记明确来源 ID 重合。只在其他/未知来源中碰巧同 ID 的记录单独列为候选。另比较原问题 UTF-8 SHA，以及 NFKC、casefold、空白折叠后的问题 SHA；不删除否定、数字、单位或标点。相同问题文本可提示内容重合，不能自动证明同一来源 ID。

未匹配记录标为 `unknown_no_observed_match`，不是“已证明独立”。零精确哈希重合也不能排除修改、转述、上游复用或来源标识缺失。汇总始终保留 `source_independence: unknown`，没有删题行为。

服务器命令示例（所有路径由协调器使用已存在的实际文件填写）：

```bash
PYTHONPATH="$OVERLAP:$EXTENSIONS" python -m unittest discover -s "$OVERLAP" -p 'test_overlap.py' -v
python "$OVERLAP/audit_overlap.py" --hotpot "$HOTPOT_JSON" --seka-root "$SEKA" --build-report "instruct=$HOTPOT_BUILD_REPORT" --faith-manifest "$ACCESS/source_manifest.json" --counterfactual "$CF_PARQUET" --inconsistent "$IC_PARQUET" --unanswerable "$UA_PARQUET" --output-dir "$OVERLAP_OUTPUT"
```

如有 Base 的已完成报告，可追加 `--build-report "base=$BASE_HOTPOT_BUILD_REPORT"`。脚本输出 `calibration_hashes.jsonl`、三份 `*.hashes.jsonl` 与 `summary.json`，全部不含问题/上下文/答案字符串。5 项合成测试不读真实数据；本轮未在本地执行。

## Counterfactual 原生选项修复

固定公开数据卡已经确认 CF 含 `choices.label/text`。论文 §3.1 说明该任务来自 ARC-Challenge，多选题有 3–5 个选项；§3.3 要求依据上下文作答。作者 README 的可执行示例只完整展示 Unanswerable/Inconsistent，没有给出 CF 选项逐字拼接代码；仓库完整脚本仍未发布。

已补 `work/extensions/faitheval_native.py`：CF 必须调用 `method_input(row, kind="counterfactual")` 读取公开 choices；保留作者发布的选项顺序与标签，不能按 answerKey 重排。提示附上完整 `Options` 区块，不再默默丢掉选项。answer、answerKey、justification 仍在方法接口之外；缺选项时明确报错。这个选项区块是**基于原生公开字段的项目适配**，不是经作者代码确认的逐字模板，`prompt_provenance` 明示这一点。没有读取任何实际 CF 测试题来设计它。

本次变化只补公开输入，不把该扩展说成临床动作盲 T/F/U 的同一机制，也没有预先得到 CF 模型分数。上一轮扩展 8 项与 MedPIC 15 项/50 行 fake 已由协调器在服务器验证；本次 choices 修改和 5 项新测试需重新执行对应测试后再登记通过。

依据：[原作者 Hotpot builder](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/src/custom_builders/adaptive/hotpot_qa_builder.py)、[原提示与筛选流程](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/src/model/projection_builder_base_adapsvd.py)、[FaithEval CF 固定卡](https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0/blob/e655f7c8750aabe431eeeda63fa5c019a2567b18/README.md)、[FaithEval 论文任务/评价说明](https://arxiv.org/html/2410.03727v2#S3.SS1)。
