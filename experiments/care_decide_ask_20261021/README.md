# CARE 模块二整改：决定还是追问（2026-10-21）

> **新研究方向：** 信息不完整不一定影响结果。先判断所有可能补全能否得到同一个合法动作；能则决策，不能则问真正会改变结果的属性。
>
> **状态：** 本目录是独立研究分支中的新方案和 **CPU 原型（19 项 CPU 单元测试已通过）**；不包含任何新增 GPU 模型效果，不代表临床有效性，也不替代冻结模块一。旧 PR #17 和旧神经干预结果保留为历史证据。

## 为什么转向

历史 CARE 键空间及二维残差干预在原实验里没有稳定超过简单方法；N3 中 baseline 21/32、二维 20/32、一维 mean 25/32、直接给定正确 D/Q 状态的文本策略 27/32。这个结果支持研究 **信息是否足够决策**，不支持“已经学到通用二维方向”。同类方向和探针门控也已有工作（见论文计划）。

## 一句话理解新方法

1. 模块一 A1 给每条规则的 T/F/U（**目前只有硬标签**）。
2. 结构化政策引擎列出与已知信息相容的世界，推导每种情况下应采取的合法行动。
3. 若所有可行世界都给出同一行动，则 **DECIDE**；否则 **ASK** 决定性的缺失属性；若信息矛盾或规则无法解释，**REVIEW** 人工检查。
4. **只有补充独立校准数据与可信 T/F/U 状态评分后**，才可从硬标签扩展到同时覆盖所有规则状态的保形预测集，获得其假设条件下的**边际**风险保证。

目前的原型支持布尔规则及明确优先级；自然语言指南 **无法直接原样执行**。真实数据需要独立审计的规则抽象/编译器、规则条件与属性之间的映射。不可把 A1 的自然语言输出当作这些结构化输入已经存在。

## 立即可执行（CPU-only）

从本目录运行：

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python -m care_decide_ask.cli run \
  --cases examples/demo_cases.jsonl --mode hard \
  --out /tmp/care_m2_demo_NEW.jsonl
```

示例为**手工构造的模拟病例和模拟 A1 状态**，不是模型结果：关键属性缺失时问 `eligible`，无关属性缺失时直接决策，未知但不影响动作时也直接决策。输出采用 `open('x')` 拒绝覆盖，不联网、不调用任何模型。

保形校准的 CPU 接口：

```bash
# 需另行准备 independent calibration.jsonl，每条记录有 truth 和 rule_probs；
# 如果按家族校准，每条还应提供 family_id。
PYTHONPATH=. python -m care_decide_ask.cli calibrate \
  --data /path/to/calibration.jsonl --alpha 0.05 \
  --unit family --out /path/to/new_calibration.json

# 运行数据必须包含 rule_probs + 评分来源；硬标签无法直接代替概率。
PYTHONPATH=. python -m care_decide_ask.cli run \
  --cases /path/to/scored_test_cases.jsonl \
  --mode conformal --calibration /path/to/new_calibration.json \
  --out /path/to/new_predictions.jsonl
```

**这些命令只是接口；还没有为真实 A1 取得可校准的概率分布，也没有在新测试集上运行这些实验。**

## 阅读顺序

- [论文怎么写（通俗版）](PAPER_OUTLINE.zh.md)
- [完整新实验协议和强基线](EXPERIMENT_PROTOCOL.zh.md)
- [方法、覆盖保证的准确含义](THEORY_AND_ASSUMPTIONS.md)
- [A1 和规则解析尚缺的接口](A1_SCORE_AND_POLICY_GAPS.md)
- [旧结果如何归档，不再重跑什么](LEGACY_EVIDENCE.md)
- [给 Codex 的分阶段实施任务](CODEX_NEXT_TASK.md)
- `care_decide_ask/`：可审计规则执行、候选追问、case/family 联合保形校准、严格 A1 硬标签适配及 CPU 命令。
- `tests/`：运行契约测试；`examples/`：可解释的玩具测试案例。

## 明确不要做的事

不再扩大二维向量的层、秩和强度搜索；不再做 N2 盲目来源切换和 N4 组合搜索；不把旧 N3 看过的测试集叫独立确认；不把 oracle D/Q 或硬 A1 标记成自动、已校准的高安全系统；不对医学患者提供临床决定。
