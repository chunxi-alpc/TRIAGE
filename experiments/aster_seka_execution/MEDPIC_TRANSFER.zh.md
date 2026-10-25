# MedPIC 原生患者证据注意力迁移

2026-10-26。状态 `planned`：代码与合成测试已编写，未在本地运行，尚无本模块服务器测试或模型推理结果。本轮只读取原有代码、公开 schema 审计记录以及 PR #36 的 Goal/Work/spec；没有打开 467 道测试题或答案。协调器已确认采用下述单一迁移方案。原生基础接口实际位于 `work/data_audit/medpic_native_adapter.py`，不是 `work/clinical_dev/`。

## 固定的迁移含义

原生输入仍只有完整 `patient_vignette`、`question`、按原顺序提供的全部 `options`。没有额外指南、检索或模型生成规则。`answer`、GF/CF 分类、taxonomy、pair 信息不传给模型。

阶段 A 对每个原始选项判断 T/F/U：该选项对当前患者与问题是支持、违反/不适用、还是信息不足。A 会看到全部原生问题和候选选项，**因此不是 MEGA 的处置盲指南条件判断**。可检验的是患者证据注意力及两阶段调度的迁移，不能把它作为同一“动作盲条件机制”已泛化的证明。

每个选项用其原始文本与病例句子的词汇重合选择完整病例原句。直接复用现有 `clinical_core.lexical_patient_spans`，保留否定、数值和单位，不训练新选择器、不选 top-k、不生成规则。它只是可复现的词汇启发式，不是已经验证的临床支持判定；找不到重合时不凭空加入证据。

阶段 B 保留完整原始输入，附上每个选项的预测 T/F/U 和三类归一化概率，只对预测 T 的选项对应病例原句施加 SEKA。共享句子的词元只计一次。F、U 对应原文仍完整可见；没有 T 或没有证据时回退完整输入。失败的 A 显式记 U，但概率为空，不伪造为 U 概率 1。

保持原生多选输出 `FINAL_ANSWER: ["A", "C"]`。这个格式属于项目适配，不声称是作者逐字提示。A 的 T/F/U 取第一步三种标签的分数，其归一化数值不是经校准的临床置信度。完整词表中对应概率另存。

## 四个消融和基线

| 配置 | A 注意力 | B 注意力 | 用途 |
|---|---|---|---|
| A0B0 | 无 | 无 | 同信息、同预算文本两阶段基线 |
| A0B1 | 无 | 有 | 固定 A0 的执行阶段增量 |
| A1B0 | 有 | 无 | 固定 A1 的执行阶段基线 |
| A1B1 | 有 | 有 | 完整迁移方法 |

对于同一个 A，B0/B1 使用完全同一份 T/F/U、证据、提示及生成预算。A0/A1 使用完全相同的原生输入和预测 span。每个有 K 个选项的独立方法预算为 K 次单词元判断加一次最多 512 新词元生成；实现复用 A，真实总调用数为 2K+4，不能把缓存后的物理成本误当单个方法成本。各方法执行生成可能提前结束，实际前向、词元、耗时和显存分别记录。未得到的成本为 null，不是零。

原生单次无注意力基线沿用原有接口，也可通过 `execution_prompt(task, None)` 与同一个 SEKA 适配器的 `NO_HOOK` 路径运行，以排除加载/编码差异；它成本更低，另报。`direct_applicability_set` 是无 B 调用的直接映射对照，允许多选或空集合，不用 argmax 强行补一个选项。它**不是强校准概率融合**。现有概率融合需要迁移成逐选项多选语义、在独立开发数据上固定校准/权重及预算；目前尚未实现/验证，不能据这四个消融宣称超越最强基线。

没有新增参数搜索或多个方法变体。SEKA 投影、层、增益从协调器最终冻结的 MEGA 设置继承；本模块不修改现有核心或重新优化这些参数。

## 执行接口

`medpic_option_transfer.py` 不读取数据文件，不加载模型、不分配 GPU：

- `option_evidence`、`condition_prompt`、`execution_prompt` 只接收白名单 `NativeInput`。
- `run_factorial(task, sample_id=..., seed=..., lock=..., infer=...)` 运行四个固定消融；保留失败行，返回匿名可提交预测、仅服务器保留的全文 trace、A 结果和实际/独立方法两种成本。
- `seka_infer(existing_adapter, config)` 使用协调器已有的 `StageSEKAAdapter`；记录实际 forward、时间、峰值显存、模型输出与第一步分数。原生使用 user 聊天角色，MEGA 的 system 角色核心不改。运行时核对聊天模板产生的词元与共享 SEKA 编码完全相同。
- `_first_token_scores_tensor` 从各 trace 移出后统一放在 `numerical_score_tensors_save_separately`；在服务器用 `torch.save` 单独保存此项，再将其他项序列化 JSON。不要将张量转成长词表 JSON。原生题目与输出全文只能留服务器。
- `source_manifest()` 在服务器生成代码指纹；`require_lock` 验证时间戳、四个消融、seed 0/1/2、提示、选择器、配置和源文件。模型权重、tokenizer、环境、投影、SEKA 参数还需协调器外层运行清单核实；本接口不假称替代完整运行清单。

最短连接方式（实际测试集由外层先检查冻结，再读取；下面不给任何测试文件路径）：

```python
config = TransferConfig()
infer = seka_infer(existing_stage_seka_adapter, config)
task = native_input(row)  # 仅白名单三个字段进入方法
bundle = run_factorial(task, sample_id=row["instance_id"], seed=0,
                       lock=frozen_lock, infer=infer, config=config)
```

`protocol.draft.json` 明确 `frozen:false`，不能直接用于测试推理。完成下述实际项后由协调器生成公开冻结记录及真实指纹，不应把草稿改一个布尔值就认为已经完成独立确认准备。

## 负对照接口与尚缺的实际项

现有 SEKA 控制枚举支持零强度和反向；`attach_controls` 只从患者原文构造错误 span 与循环错位的选项—证据对应。内部沿用上游 `misaligned_rule` 枚举名，科学报告须称“选项—证据错位”，不能凭枚举名虚构指南规则。

控制范围须在最终聊天模板编码后构造。可将构造函数传入 `seka_infer(..., control_builder=...)`，它在模板处理后调用。错误 span 同词元数且与原 span 不重叠；错位同词元数且实际 mask 改变。无法构造时明确 unavailable，保留无注意力预测作为回退，不能宣称该项控制成功执行。B 的零强度/错位/反向固定 A1，与 A1B0 比较；A 控制固定其原生输入，与 A0 比较。完整词表第一步分数用于零强度数值漂移，不以最终答案偶然相同代替检查。

**目前仅核心四消融编排已经写好；控制构造接口已经写好，但所有控制的完整调用编排与结果文件导出仍需外层接入。** 正式冻结前实际缺项：

1. 服务器合成 CPU 测试、真实 Qwen tokenizer/聊天模板和 T/F/U 单词元检查；随后只用合成病例做小规模实际模型烟雾测试，保留全部输出。当前两项均未运行。
2. 外层 runner 接入全部负对照、原生单次与直接映射对照、强校准概率融合；同信息与实际预算对齐。不能把没有运行的融合记成强基线已经完成。
3. 继承最终 MEGA 方法设置，记录模型/投影/环境/输入/代码指纹、各类 seed、控制可用率、完整成本；确认最终比较族和全部冻结时间戳。不能从 MedPIC 错误里再调参数。
4. MedPIC 467 题全量固定分母，失败不删除。已有元数据核实 284 GF/183 CF，但 89 对精确配对与真实患者/来源簇依旧缺失。缺映射时可报告全量及 GF/CF 点估计；Pair、按真实患者/来源簇的置信区间及显著性明确 blocked，不能把每题或 seed 当独立患者。确定性 seed 重复不增加独立证据。

上述缺项是原任务要求的实际工作状态，不新增数据许可审批或额外实验审查步骤。没有任何 MedPIC 模型分数或注意力增益结论。

## 服务器 CPU 合成测试

协调器将本目录与现有三个依赖目录搬至服务器后，在已有 PyTorch 环境执行；命令不读取真实数据、不启动 GPU：

```bash
PYTHONPATH="$TRANSFER:$AUDIT:$CLINICAL:$METHOD_AGENT" python -m unittest discover -s "$TRANSFER" -p 'test_medpic_option_transfer.py' -v
PYTHONPATH="$AUDIT" python -m unittest discover -s "$AUDIT" -p 'test_medpic_native_adapter.py' -v
```

8 项新测试覆盖标签不进入方法、原句与否定保留、U 语义和共享 span 去重、user 模板/范围、四消融输入与预算一致、缓存成本、完整失败分母、错误 span 与不可用错位、未冻结拒绝运行。所有均为合成数据；本轮未在本地执行。

依据：[PR #36](https://github.com/chunxi-alpc/TRIAGE/pull/36)、[Goal](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-seka-goal-20261025/docs/ASTER_SOTA_GOAL/GOAL.zh.md)、[Work 总指令](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-seka-goal-20261025/docs/ASTER_SOTA_GOAL/WORK_ORDER.zh.md)、[固定版本 MedPIC 数据卡](https://huggingface.co/datasets/TIM0927/MedPIC-Bench/blob/9ef6db4f13865b14fc2e6be3f94dcfaf3a0cf983/README.md)。本轮未重新打开数据卡示例或测试正文；此前公开例题的被动曝光沿用原审计记录，不因此把测试标为完全未曝光。

扩展数据补充：协调器正在服务器获取 [MedCalc-Bench-Verified](https://github.com/nikhilk7153/MedCalc-Bench-Verified) 与 [FaithEval](https://github.com/SalesforceAIResearch/FaithEval)，暂不扩展本模块。FaithEval 含 HotpotQA 来源，而 AdaSEKA 校准也用 HotpotQA，未来需用公开 source ID/文本 hash 核查重叠或披露未知，不能先宣称来源独立。
