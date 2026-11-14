# 模块一 V7：完整证据证明与保守数值覆盖

本目录新增独立实验版本 `src/module1/care_v7.py`。V6 算法、历史预测和包级默认入口保持原样。本分支归档的可引用执行结果位于 `results/*_release_v2/`。

## 已完成与证据范围

| 工作 | 执行情况 | 可以支持的结论 |
|---|---|---|
| 充分证明门控 | 已实现，CPU 测试通过 | 门控按完整 `AND/OR/NOT` 与例外逻辑取证，修复部分必要条件无证据仍开启动作门控的反例 |
| 仅数值证据状态 | 已修复，模拟模型集成测试通过 | 明确数值证据可以形成支持／反驳状态，不再错误保留为缺失 |
| 数值主体、时效和单位专项 | 56 项开发回归已执行 | 覆盖明确当前患者、亲属及嵌套所有格、历史测量、单位与边界、多值及否定矛盾等已列出的情形 |
| 旧 192／新 100 门控复评 | 已读取冻结缓存执行 | 仅评价门控及证明内证据选择；不改变已有适用性标签，不重跑模型或数值抽取 |
| 新模型运行器 | 576 条机制输入已通过无模型结构预检查 | 已准备真实 Phi-4／TrialMatchAI 与 BioLinkBERT 推理入口；预检查的模型调用数为 0 |

**本阶段没有产生新的大模型或临床效果结果。** CPU 回归属于开发测试；缓存复评复用了旧模型输出。结构证明完整也不能保证自然语言推断证据在语义上正确。

## 方法变化

规则按“主条件且例外不成立”计算。已知状态下，枚举与该状态一致的子集最小充分证明。合取成立需要每个必要子项；析取成立可使用任一完整成立分支；否定保留原子条件实际方向；例外使用同样的逻辑。每个证明原子必须存在方向一致、区间有效且无反向冲突的证据。只有至少一份完整的真证明通过这些检查，规则动作门控才开启。未知不形成已知状态证明。证明数量超出限制时明确关闭门控，不截取不完整证明作为成功。

数值覆盖要求同一句话包含明确患者主体与当前时效标记，并将数值绑定到支持的字段表达。亲属、其他患者、嵌套所有格、历史或混合时效、否定及不确定数值均保守处理。跨句的当前否定／未知冲突也会阻止覆盖。单位使用十进制定点比例换算。字段、单位或表达不支持时返回未知并保留 TrialMatchAI 原判断。该策略是有限的正则规则，不是通用临床主体与时间解析器。

新增输出包含 `selected_proof`、`proof_evidence`、`proof_gate` 和 `proof_binding_gate`。原 `binding_gate` 仍是局部条件证据门控；需要完整证明的下游实验应明确采用 `proof_binding_gate` 与规则 `action_gate`。不自动替换现有流水线输入版本。权重仍未做概率校准。

## 结果入口

- [本次实验报告](REPORT.zh.md)
- [数据与评价协议](protocol.json)
- [56 项数值诊断与预测](results/cpu_release_v2/numeric_cases_and_predictions.jsonl)
- [CPU 指标](results/cpu_release_v2/metrics.json)与[17 项测试日志](results/cpu_release_v2/unit_tests.txt)
- [缓存门控复评指标](results/cached_gate_release_v2/metrics.json)
- [旧 192 逐例门控复评](results/cached_gate_release_v2/old_192_gate_predictions.jsonl)与[新 100 逐例门控复评](results/cached_gate_release_v2/new_100_gate_predictions.jsonl)
- [真实推理准备清单](results/inference_prepared_release_v2/run_manifest.json)

## 复现

在完整仓库根目录运行。输出目录必须是新目录，脚本拒绝覆盖已有结果。

```bash
python experiments/care_reliability_20261019/module1/run_cpu_diagnostics.py --output /tmp/care_numeric_new_run
python experiments/care_reliability_20261019/module1/replay_cached_gates.py --source-dir experiments/care_module1_v6/data --output /tmp/care_gate_new_run
python experiments/care_reliability_20261019/module1/run_inference.py --inputs experiments/care_reliability_20261019/data/mechanism_cases.jsonl --output /tmp/care_v7_precheck --version v7 --dry-run
```

真实推理需要实际可用的 GPU 与模型依赖。去掉 `--dry-run`，并将三个模型参数指向核验过的本地权重目录。例如：

```bash
python experiments/care_reliability_20261019/module1/run_inference.py \
  --inputs experiments/care_reliability_20261019/data/mechanism_cases.jsonl \
  --output /absolute/new_run_directory \
  --version v7 --device cuda:0 \
  --base-model /absolute/phi4_directory \
  --adapter-model /absolute/trialmatchai_adapter_directory \
  --evidence-model /absolute/biolinkbert_mednli_directory
```

V6 对照使用相同输入与生成参数，把版本改为 `v6` 并使用另一个输出目录。分片通过 `--shard-count` 与 `--shard-index` 明确指定；不得将只完成的分片写成完整实验。输出逐例区分正常解析、解析失败和技术错误。金标、参考结构与结构化事实不传入模型，仅允许结构化规则、患者原文与问题进入常规推理。

## 尚待实验

真实 V6／V7 同输入比较需要在可用模型环境执行。临床结论还需要按指南或规则族隔离的数据，以及实际完成并留痕的专业人员复核。门控策略应同时比较覆盖率、误开启率和标签／证据联合正确率。现有缓存结果不足以支持替换默认版本。
