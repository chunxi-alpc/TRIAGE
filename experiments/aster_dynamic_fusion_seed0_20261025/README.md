# 最新完成状态

两模型各800条新前向已完成并验证，详见 [完成报告与分组对照](COMPLETED.zh.md)、[完整新前向报告](fresh_report.zh.md)、[匿名逐例答案](completed_predictions.csv)。独立新增数据验证未完成。以下旧阶段记录保留供追溯。

# 2026-10-25 实际执行复核补充

详见 [中文研究报告](REVIEW.zh.md)、[公平网格结果](balanced_cached.json)、[资源核对](resource_inventory.json)、[冻结候选](frozen_candidate.json)。
原两个脚本已在115服务器实际跑通，主要点估计复现。P3分种子退化/纠正累计已修复；历史后半层残差缺少对应零强度，下面原结果仅保留作历史探索性记录。
两模型新前向各800条已启动，完成状态以 [run_state.json](run_state.json) 为准；只有 complete_verified 表示完整结果已验证。
完成后交付 balanced_fresh.json、fresh_report.zh.md、execution.json 和 fresh_numeric/ 中的匿名逐例输出。
独立新指南树验证尚未完成：未发现可用的新独立标注病例集。

新增CPU复现：
    python -m unittest discover -s experiments/aster_dynamic_fusion_seed0_20261025 -p 'test_*.py'
    python experiments/aster_dynamic_fusion_seed0_20261025/balanced_analysis.py
    python experiments/aster_dynamic_fusion_seed0_20261025/render_report.py

新前向完成后，fresh_numeric/{qwen,llama}.jsonl.gz 可分别解压为一个新目录的 qwen/predictions.jsonl、llama/predictions.jsonl，再用 balanced_analysis.py --fresh-root 该目录 --output 新结果.json 进行无需模型权重的CPU复算。

---

# ASTER seed=0: 条件概率校准、动态融合与注意力残差

**状态：开发集离线复分析完成；没有启动新 GPU 前向；不是独立测试。**

## 数据与真实性

- Qwen：PR #32 完整 800 条真实模型推理记录；Git blob \`27e6880334a27be0962d2e2594ff08d39b52a24f\`
- Llama：PR #31 的 800 条开发集真实模型预测及条件概率；Git blob \`6f7c855dca45011299b03618eb70cfe952377f15\`
- 两模型均是 200 名患者 × NC/AC/IC/SW 四版本 × seed=0；仅 4 个开发树：aml_7、bre_1、pancreatic_11、vulvar_3
- \`data/\` 中是匿名数值派生文件，无病历、规则及选项原文；Qwen 包含零强度、动作分数偏置、质量守恒的 logits
- Qwen 发布的 fusion_0.75、direct_map、action_bias_2.0_all 逐例复算 0 不一致；Llama 原答案与概率 argmax 0 不一致

## 实验设计

以原模型分布 P0 和规则动作支持 S 为输入，比较：(a) 静态线性/几何融合与输出温度校准；(b) 基于支持度、差距、熵和原模型置信度的动态融合；(c) Qwen 注意力残差校正。注意力残差定义为已有动作注意力 logits 减去**相同数值路径的零强度 logits**，经 tanh 截断后在融合概率的对数域施加小幅修正。禁止用测试金标选择任何参数。

每棵开发树留出，余下三棵树通过宏平均准确率选参。固定网格：静态 1078 组，阈值动态 216 组，线性动态 48 组，Qwen 注意力残差 576 组，数值误差/动作错位/符号翻转对照各 144 组。下表为这些开发树的留一树预测合并后的成绩（不是 800 独立病例）。

| 方法 | Qwen 准确率 | Qwen 退化率 | Llama 准确率 | Llama 退化率 |
|---|---:|---:|---:|---:|
| 原模型 | 46.88% | 0 | 46.75% | 0 |
| 动作概率直接映射 | 54.13% | 15.88% | 54.75% | 16.75% |
| 原固定融合 0.75 | 56.50% | 1.50% | 55.50% | 2.63% |
| **增强静态温度校准** | **62.50%** | 3.25% | **63.88%** | 5.25% |
| 阈值动态融合 | 62.00% | 3.25% | 63.63% | 5.38% |
| 注意力残差 | **63.88%** | 5.88% | 未测试 | 未测试 |

Qwen 注意力残差跨树选择的净改善为 +11/800 (+1.375 个百分点)，四树描述性重采样区间约 [0,+2.75] 个百分点。**注意力残差没有证明统计稳健优势，且退化率增加 2.625 个百分点**。只用四棵树不能作强显著性结论。

## Qwen 负对照

| 注意力残差通道（各自交叉选参） | 准确率 |
|---|---:|
| 有效动作对齐残差 | 63.88% |
| 零强度数值漂移 | 61.88% |
| 残差循环错位到其他动作 | 60.63% |
| 残差符号随机翻转 | 61.38% |

在**相同融合参数**下，移除残差为 63.00%，保留对齐残差为 63.88%，仅净增 7/800。四树重采样区间 [-0.75,+2.50] 个百分点；反向干预下降至 59.50%。这些结果只能证明初步信息相关性，不能证明临床有效性或因果机制。

原 PR #32 的零强度数值控制仍改变 13/800 个预测；模型路径不是逐位等价，因此必须保留该对照。

## 复现与范围

在此分支的 repo checkout 中执行：

\`\`\`bash
python experiments/aster_dynamic_fusion_seed0_20261025/reproduce.py
\`\`\`

脚本只读取 \`data/\` 下的匿名数值文件，在 CPU 上完成配置选择，输出每折结果和汇总。运行不需 GPU、不修改 PR #31/#32 原始预测。

**剩余工作**：Qwen 新模型前向复核；Llama 真正的注意力残差实验；至少一组未参与方法设计的新增决策树和人工标注。不能把此前被使用过的 P3 测试结果继续称为全新独立测试。


## P3 exploratory post-hoc hard-anchor fusion

See [P3_EXPLORATORY.md](P3_EXPLORATORY.md) and [JSON metrics](p3_hard_anchor_exploratory.json). Using fixed development-selected parameters beta=0.8 and support temperature=0.25 on the previously inspected P3 action support vectors gives Qwen 73.375% accuracy / 4.842% regression and Llama 60.433% / 10.967% regression, at 100% coverage. Accuracy differences versus direct mapping have tree-bootstrap intervals crossing zero; regression rates fall. This is an exploratory post-hoc analysis, not a new confirmatory test, and it does not evaluate attention.
