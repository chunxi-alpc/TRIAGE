# MedCalc 与 FaithEval 原生扩展入口

2026-10-26；排在 MEGA/MedPIC 主任务之后。当前为代码准备，未在本地运行、未启动 GPU、未读取两者测试题。所有真实服务器验证由协调器执行，不能将以下接口或 fake 输出称为完成基准实验。

## MedCalc-Bench Verified

使用协调器已下载的提交 `801592132bfd833f049b517a4e013e00a5fc40fa`。服务器已有元数据为 10,538 train、1,100 test、55 one-shot；本轮没有重新读取数据正文。作者 README 建议采用 Verified 而非旧 v1.2，声明 CC-BY-SA 4.0；最终报告仍需写明这个固定提交及实际 CSV SHA，不能仅写“最新版”。

已提供可执行 `run_medcalc.py` 和独立 CPU 评价器 `evaluate_medcalc.py`。方法只接收 `Patient Note` 与 `Question`，以原作者 `direct_answer` 或 `zero_shot` 函数创建 system/user 消息。Calculator ID、Relevant Entities、GT answer/explanation、上下限、类别、Note ID 均不进入模型。身份和来源 hash 由外层保留，评价器才读取 Calculator ID 和标签。Note ID 不用于去重；同一来源的多条计算任务保留全部记录。

`OfficialMedCalc` 核对固定 Git blob，直接加载作者的上述两个提示函数、`extract_answer` 与 `check_correctness` 函数体。只加载这些函数，不执行作者模块顶层的模型导入、HF 登录、默认 test 文件读取或 one-shot 示例读取。未启用代码执行工具；模型只能按原生文字输入生成答案。

实际原生判定语义：

| 类型 | 固定源码行为 |
|---|---|
| 日期，ID 13/68 | 解析四位年份 `%m/%d/%Y` 后比较；README 的两位年份说明不能替代源码 |
| 周/天，ID 69 | 提取为数值二元组再比较；格式错误导致异常时保留并计错 |
| 整数型计算器 | 对解析结果执行 Python `round` 再与 GT 比较；不是字符串相等 |
| 小数型计算器 | `lower_limit <= answer <= upper_limit`，边界均包含 |
| 缺失或解析失败 | 保留原始行，正确率分子为零，分母不变 |

原作者解析器使用 `eval` 处理某些算式。这里保持原函数体不改，将其 `eval` 绑定到数值表达式解析器，支持标量算术、原整数取整、数值元组及常用 math/NumPy 数值函数，不执行生成内容中的任意 Python。超出此范围的表达式保留为 N/A/解析失败；因此应称“复用原作者数值正确率的安全执行适配”，不能声称与任意 Python 表达式路径完全等价。这不向模型额外提供算术工具，也不把工具增益算为注意力增益。

作者 `table_stats.py` 的整体标准差仍硬编码分母 1047，而本版本 test 为 1100。新评价器保留作者逐例正确率语义，按实际完整行数计算点估计，不沿用该旧标准差。输出每例来源 `(Note Type, Note ID)` hash、病例文本 hash，后续由既有跨方法/seed配对统计流程处理来源簇；没有将每题独立的二项误差当来源簇置信区间。

生成入口采用现有 Qwen 本地权重、原作者消息、贪心和不截断完整输入。Qwen 加载、关闭 thinking、32768 上下文属于本项目适配，作者旧 LLMInference 没有原生 Qwen 分支；不能声称原作者已验证该模型设置。草稿保留 4096 新词元预算，与作者开源生成路径的增量上限对应。正式预算仍由协调器冻结。当前实现是无工具原生基线，不冒充没有指南输入的“临床条件注意力方法”。

开发只读显式声明的 train 行；`medcalc_data.py` 要求实际 train CSV hash、`train_row_numbers`。正式 test 必须先有 `frozen:true`、时间戳、全部 seed 0/1/2、源码/模型/数据指纹，保留全部 1100 行。`medcalc_protocol.draft.json` 不会自动变成冻结文件。one-shot 的 55 个例子尚未审计或加载；这一赛道单独处理。

## FaithEval

固定仓库提交 `58d35840e3fbc7bf4b7672582a417b3b3a327dec` 的完整文件树只有 README、资源图与管理文档，没有完整评价脚本；README 仍写“将发布完整脚本”。代码许可证据按协调器审计为 Apache 2.0，README 对数据发布说明科研用途。三份 HF 数据在服务器下载报 APIConnectionError，本轮只访问 metadata endpoint 也失败，**实际数据文件、完整 gold schema 和 revision 仍未核实**。

`faitheval_native.py` 已实现从公开说明可确认的接口：只向方法传 `context` 与 `question`，使用 README 的原生任务消息；不传参考答案或上游来源元数据。Unanswerable 和 Inconsistent 直接实现 README 的归一化与有效词组包含判定，同时可选择 strict 单词组列表。

这里的 strict 仍是“归一化后是否包含规定词组”，不是整条输出的 exact match。例如非 strict 的 `not` 会命中更长词的一部分；本适配保留这一作者行为，不暗改定义以制造差异。论文 §3.3 将准确率定义为响应提及答案。Counterfactual 提供显式参考字符串列表后的包含匹配接口，标为依据论文的适配；真实 gold 列未核实前返回 blocked，不猜字段，也不宣称完整官方脚本已复现。

此扩展暂时只验证通用上下文忠实性，不把病例适用性 T/F/U 生搬到三种 QA 任务。FaithEval 包含 HotpotQA 来源，而 AdaSEKA 校准使用 HotpotQA 前 200 项。`source_overlap_report` 可比较已经核实命名空间的公开 source ID hash 与正文 hash，不输出正文；修改过的上下文可能不再逐字相等，没有匹配不能证明独立。缺少 source ID 时继续标记来源独立性未知，不能以随机切分或文本相似度猜出真实来源身份。

## 服务器最小验证

8 项合成 CPU 测试中，3 项会真正调用固定版本作者函数。设置官方代码目录后执行；不需要数据文件、模型或 GPU：

```bash
MEDCALC_OFFICIAL_ROOT="$MEDCALC" PYTHONPATH="$EXTENSIONS" python -m unittest discover -s "$EXTENSIONS" -p 'test_extensions.py' -v
PYTHONPATH="$EXTENSIONS" python "$EXTENSIONS/run_medcalc.py" --official-root "$MEDCALC" --config-lock "$EXTENSIONS/medcalc_protocol.draft.json" --output-dir "$MEDCALC_SYNTHETIC_OUTPUT" --split synthetic --engine fake --seed 0
```

第一项验证输入隔离、冻结前拒绝测试访问、日期/整数/小数边界/周天的作者函数、失败完整分母及 FaithEval 的实际包含语义。没有设置 `MEDCALC_OFFICIAL_ROOT` 时 3 项作者函数测试会明确跳过，不能把这种结果称为原函数验证通过。第二项仅处理两条内置虚构算术输入，实际模型调用为零，写出逐例预测/私有输出与运行清单。

真实 Qwen 或 train 开发随后由协调器先汇报资源再启动：`run_medcalc.py --split train --engine hf --model ... --model-fingerprint ...`，需要已写明 CSV hash 与训练行号的开发配置；本轮不执行。正式评价单独调用：

```bash
PYTHONPATH="$EXTENSIONS" python "$EXTENSIONS/evaluate_medcalc.py" --official-root "$MEDCALC" --config-lock "$MEDCALC_FROZEN_LOCK" --split test --predictions-private "$MEDCALC_RUN/predictions.private.jsonl" --output "$MEDCALC_METRICS"
```

正式多 seed/多方法应分别输出且保留错误后再统一比较；本评价入口一次只接收一个方法和 seed，不静默混合或覆盖重复 key。没有任何新增模型分数。

来源：[MedCalc README](https://github.com/nikhilk7153/MedCalc-Bench-Verified/blob/801592132bfd833f049b517a4e013e00a5fc40fa/README.md)、[原提示与解析](https://github.com/nikhilk7153/MedCalc-Bench-Verified/blob/801592132bfd833f049b517a4e013e00a5fc40fa/evaluation/run.py)、[原正确率](https://github.com/nikhilk7153/MedCalc-Bench-Verified/blob/801592132bfd833f049b517a4e013e00a5fc40fa/evaluation/evaluate.py)、[旧汇总](https://github.com/nikhilk7153/MedCalc-Bench-Verified/blob/801592132bfd833f049b517a4e013e00a5fc40fa/evaluation/table_stats.py)、[FaithEval README](https://github.com/SalesforceAIResearch/FaithEval/blob/58d35840e3fbc7bf4b7672582a417b3b3a327dec/README.md)、[FaithEval 论文评价定义](https://arxiv.org/html/2410.03727v2#S3.SS3)。精确版本和文件 SHA 另见 `source_registry.json`。
