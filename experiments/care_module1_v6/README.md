# CARE 模块一 V6：实验导航

当前运行入口为 [src.module1](../../src/module1/README.md)，输出 `module1.v6`。本目录保留 V6 提交时的预测、报告与脚本；[README.zh.md](README.zh.md) 和 [REPORT.zh.md](REPORT.zh.md) 是冻结阶段记录。

[项目首页](../../README.md) · [模块一使用说明与已知限制](../../src/module1/README.md) · [V5 历史记录](../care_module1_v5/README.md)

## 结果范围

V6 使用 TrialMatchAI 进行原子条件判断，数值比较可覆盖适用性，BioLinkBERT-MedNLI 用于证据定位。A 为 TrialMatchAI 单独判断，B 为 A 加数值比较，C 为 B 加证据定位；C 不再修改 B 的适用性标签。

| 数据与方法 | 适用性准确率 | 宏平均 F1 | 已知样本证据定位准确率 | 判断与证据同时正确 | 反事实成对准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 192 例，A | 150/192（78.13%） | 0.7709 | 55.47% | 38.02% | 65.63% |
| 旧 192 例，B/C | 151/192（78.65%） | 0.7754 | 55.47% | 38.54% | 68.75% |
| 新 100 例，A/B/C | 73/100（73.00%） | 0.7339 | 37.68% | 25.00% | 52.94% |

数值来自[冻结指标](data/v6_metrics.json)。旧 192 例通过已有 V5 条件判断、数值比较和证据结果重新合成，不是重新运行 V6 模型所得。新 100 例保存了 V6 逐例预测。评价脚本对 A/B/C 使用同一组已保存证据，因此表内证据指标不能作为三种独立证据定位方法的消融比较。B/C 标签相同属于设计约束，不是额外的性能增益。

旧集的数据限制见[数据协议](data/dataset_protocol.json)：包含受控文本变体，已用于方法开发，没有独立医生标注。新集来源与固定样本哈希见[来源清单](data/new_set_manifest.json)。这些结果属于模块一实验，不能替代 V6 与模块二、三联用的端到端验证。

## 文件入口

| 文件 | 内容 |
| --- | --- |
| [REPORT.zh.md](REPORT.zh.md) | 当时的实验报告与结论 |
| [data/v6_metrics.json](data/v6_metrics.json) | A/B/C 指标和门控统计 |
| [data/old_192_predictions_v6.jsonl](data/old_192_predictions_v6.jsonl) | 旧样本离线重合成结果 |
| [data/new_100_predictions_v6.jsonl](data/new_100_predictions_v6.jsonl) | 新样本逐例预测与证据区间 |
| [data/labels_public.jsonl](data/labels_public.jsonl)、[data/new_labels_100.jsonl](data/new_labels_100.jsonl) | 离线评价标签；不输入常规推理 |
| [data/v6_failure_samples.jsonl](data/v6_failure_samples.jsonl) | 保存的失败样本摘要 |
| [data/rules_v5.jsonl](data/rules_v5.jsonl)、[data/new_rules_100.jsonl](data/new_rules_100.jsonl) | 结构化规则 |
| [data/new_set_manifest.json](data/new_set_manifest.json) | 新集来源、样本 ID 及服务器输入哈希 |
| [tests/](tests/) | V6 和流水线字段契约测试 |

## 复现边界

[evaluate_v6.py](evaluate_v6.py) 是原服务器评价脚本。它硬编码 `/home/gcx/TRIAGE`，读取未入仓的 `data/new_inputs_100.jsonl` 和 `runs/new100/pred_*.jsonl`，并写入原结果路径。因此它不能直接从一个新检出的仓库完成离线复评，也不应被当作只读检查命令。

已入仓的标签、逐例预测和指标可以用于独立统计复核。重新生成预测还需要服务器病例输入和模型权重；输入保留范围与哈希由 [new_set_manifest.json](data/new_set_manifest.json) 标明。不要从预测或标签反推、补造原始病例。

[scripts_run_new_v6.py](scripts_run_new_v6.py) 与 [run_4gpu.sh](run_4gpu.sh) 保留当时模型路径、环境和 GPU 分片配置，属于历史运行记录。开展新实验时，应使用新的输出目录并记录实际模型、数据、设备和依赖，不覆盖本目录已保存的预测。

## 如何解读冻结说明

冻结说明中“找到明确主体、时间才数值覆盖”以及“证据缺失时关闭干预”的表述强于当前代码保证。现有数值提取仍可能匹配亲属或历史测量；合取规则的门控可能在部分条件缺证据时开启。仅数值证据还存在条件证据状态不一致的问题。准确的接口和限制以[模块一说明](../../src/module1/README.md#已知限制)为准，历史报告原文保留用于溯源。

本次仓库整理只修正入口、模型配置和导航，不改变 V6 算法或历史统计。
