# MEGA 构造修复 v2

运行环境为服务器 `/home/gcx/gcx261014`。数据和报告位于 `/home/gcx/aster_cr_repair_20261108`。旧版本 `/home/gcx/aster_cr_20261108` 及其原始结果保持不变。

本模块只处理数据，不启动训练、选参或模型评分。GPU 为 0、1、6。

## 入口

- `python -m experiments.aster_cr.repair.run_all`：按来源分配三个抽取进程，结束后构造、汇总并固定数据；已有规则按节点复用、已有病例事实按记录复用。
- `python -m experiments.aster_cr.repair.build --scope bulk`：只用已完成抽取构造数据，结果写入 `bulk/built`。部分抽取运行中不使用这项结果作为最终版本。
- `python -m experiments.aster_cr.repair.report`：更新报告，不评分。
- `python -m pytest -q experiments/aster_cr/repair/test_repair.py`：服务器 CPU 回归测试。

`status.json` 为 `data_complete_downstream_paused` 且存在 `data/frozen.json`，才表示全量数据构造完成。失败时停止在数据阶段，原因写入状态和日志，不启动下游。

## 输出与分母

`data/original_nodes.jsonl` 与 `data/path_cases.jsonl` 保留原始评价分母。`data/nodes.jsonl` 包含冻结来源中的 2,782 个受控实验候选节点。新增节点的 Full 输入为配对参照，`controlled_reference=true`，但不进入原始节点准确率分母；该分母由 `original=true` 决定。

全量结束后 `data/views.jsonl`、`data/pairs.jsonl` 使用原 A/P/C/F/F0 配对结构；A 的视图名保留 `remove_action`、`neutral_action`、`swap_action`。C 配对左侧直接使用未改动的原病例和原条件，不再使用自由改写的 C_base。

`data/rules.jsonl` 记录可用规则、事实与证据，`data/unavailable.jsonl` 分类别记录失败，`data/extraction_review.jsonl` 记录未知或抽取分歧。任何一类失败都不导致整节点其他类别丢弃。`data/variants.json` 同时报配对、独立节点和来源数。

小批修正及来源核对在 `pilot/review.jsonl` 和 `pilot/review_overrides.json`。其余记录为程序解析和模型抽取，不声称全量人工或临床审核。旧版中出现的模板复制不当作真实标签错误。

修复改变了受控输入。原 Full 输入相同且模型、评分协议相同时可复用分数和激活；最终清单见 `cache_reuse.json`。依赖修复配对的定位、训练、开发集选参均需重新计算。旧下游控制器仍保持暂停，不能把旧选参结果直接用于新 test。

## 外部任务

`external/nli4ct/items.jsonl` 包含原始临床试验对应章节、陈述、原题 ID 和发布标签；`pairs.jsonl` 保留发布关系，按实际标签是否相同分类。不得把 Numerical_contradiction 直接视作翻转。

`external/medpic/items.jsonl` 保留 467 道原始多选题。`groups.jsonl` 保留 90 组，包括一个三成员组；92 条边不能算作 92 个独立病例。`pairs.jsonl` 保存选项文字对应、共有病例文字与实际差异。未配对题保留原题评价。

预测文件为 JSONL，每行具有 `id` 和 `prediction`：

- NLI4CT 的 prediction 为 `Entailment` 或 `Contradiction`。
- MedPIC 的 prediction 为所有所选选项字母组成的 JSON 数组。

评分示例：

```bash
python -m experiments.aster_cr.repair.external --evaluate nli4ct --predictions /path/to/predictions.jsonl --output /path/to/results.json
python -m experiments.aster_cr.repair.external --evaluate medpic --predictions /path/to/predictions.jsonl --output /path/to/results.json
```

缺失或不合法预测按错误计入原分母；分别报告原题准确率、配对两端均正确、关系判断正确及整组均正确。数据仅用于独立验证，不并入 MEGA 四类宏平均、训练或选参，不替代路径评价。
