# CARE 模块二：仅新增未做机制的 N1–N5 实验（2026-10-20）

> **最新状态：候选 token / N5 拟合已修复，CPU 检查及获准的开发/测试补齐已完成；新增187次GPU前向。N1测试、N4联合实验按登记门槛停止。**
> **最终结果：[v3修复报告](runs/v3_repair_20261020T144100Z/REPORT.zh.md)。旧v1/v2结果保留供审计，不作为纠正后的最终结论。**
> **历史结果只引用、不重新跑；模块一固定为语义 A1，禁止修改、重训或以 V6/V7 冒充。**

> **后续N3开发检查已完成：[保留均值方向的两维基底](runs/v4_n3_mean_basis_20261020T151946Z/REPORT.zh.md)，新增48次前向，11/16；原主研究测试结果不变。**

## 研究命题

在 **不增加模型知识、不修改模型权重** 的前提下，学习可泛化的低维**选择接口**，检验其对规则相关性/适用性、来源策略、决策充分性及组合控制是否具有可定位的因果效应，以及相比简单提示、直接分数偏置和确定性流程是否具有**额外价值**。最终用冻结的模块一 A1 **预测**信号而非只用 oracle 控制，衡量上游错误传播。

已经做过的实验不可重做：B1–B4 / SEKA / 随机与均值空间、原 384 例有符号 B4 与 score bias、旧层段和强度搜索、V6/V7 门控、PR #11 结构提示/证据增强/规则执行、PR #10 的 8B 冲突方向测试、PR #12 的 24 例 A1 四臂对照，以及 2026-10-20 离线审计。请参见 [HISTORY_REUSE.md](HISTORY_REUSE.md)。

## 仅做这些新实验

| 阶段 | 尚未解决的问题 | 新数据的主测量 | 禁止重复 |
| --- | --- | --- | --- |
| **N1** | 规则**相关性 R** 与**适用性 A**能否独立控制？ | 同一机制 R×A 四格全对率，误选目标率 | 旧 B1–B4 网格 |
| **N2** | 上下文 C 与模型既有知识 P 能否同一输入**双向**切换并跨内容迁移？ | 接收实例 C/P 两向均正确率，IIA 供体复制审计 | PR #10 原 8B 方向实验 |
| **N3** | 缺未知信息何时必须追问 Q，何时可决断 D？ | 错误决定率、不必要追问率、追问目标正确率 | 把 U 直接当作 Q |
| **N4** | C/P、D/Q 与规则 T/F 控制能否组合而不串扰？ | 16 格完整块正确率、交叉串扰、轴顺序效应 | 旧 V6/V7 联合门控 |
| **N5** | 在**冻结 A1 真实预测**和错误下是否仍优于简单接口？ | 同 A1 信号下 vs 文本提示/直接偏置的增量 | PR #12 原 24 例四臂 |

执行顺序：**CPU 数据/真值/泄漏审计 → 新运行器最小 hook smoke → N1 → N2 与 N3 分别独立 → N4（须通过基础轴门槛）→ N5。** N5 可单独做已冻结 A1 信号接入准备；但最终效益结果不得用 oracle 冒充可部署信号。

## 文件入口

- [EXPERIMENT_PLAN.zh.md](EXPERIMENT_PLAN.zh.md)：完整 N1–N5 假设、可归因设计、强基线、门槛及失败解释。
- [HISTORY_REUSE.md](HISTORY_REUSE.md)：原始 PR 证据、已完成项和禁止重跑清单。
- [DATA_PROTOCOL.md](DATA_PROTOCOL.md)：600 条旧预备合成样本的地位、新评测生成/隔离规则、来源与补全真值。
- [ANALYSIS_AND_GATES.md](ANALYSIS_AND_GATES.md)：统计方法、主对比、因果控制、停止条件。
- [CODEX_TASK.zh.md](CODEX_TASK.zh.md)：交给 Codex 的端到端任务和验收定义。
- [CODEX_GPU_EXECUTION.md](CODEX_GPU_EXECUTION.md)：SSH/GPU 授权后逐阶段执行规范（模型运行命令由 Codex 实现并自检，禁止伪造入口）。
- [MATRIX.csv](MATRIX.csv)：精简 GPU 预算与实验矩阵。
- [HISTORICAL_INVENTORY.json](HISTORICAL_INVENTORY.json)、[FREEZE.json](FREEZE.json)：机器可读研究契约。

## 当前明确未完成

- 尚未重新生成完成机制隔离的新正式训练、开发、测试数据。
- 尚未实现或在 GPU 验证 N1–N5 的完整神经运行器；本次文档 PR 不声称已有可运行的 5 个新任务。
- 未运行新的 Qwen/Phi 前向；所有新结果状态为 `not_run`。
- 旧的 600 条合成研究前置样例及其 22 项 CPU 测试此前在独立本地包中验证，**本 PR 没有假装它们已经符合新的 N1–N5 因果设计，也没有把该本地 ZIP 当作已入库代码**。Codex 可以拿用户另附的 `CARE_module2_A1_preexperiments.zip` 做迁移起点，但必须完成本文规定的数据重构和测试。

## 冻结上游及证据边界

- A1：`configs/care_module1_for_module2.json`，source commit `8fb87d15576b88500f44a88f04799267d9f2b5ee`。
- T=+1，F=−1，U=0，解析技术失败=null；不能让 gold 或 downstream effect 参与信号筛选。
- Qwen/Qwen3-4B-Base revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`；权重、tokenizer 与来源校验必须与旧档一致。
- 实验是模拟机制研究；历史医学任务是探索性合成和特定来源数据，不代表临床有效性。
- **学习可解码方向、算子真的改变注意力、方法优于简单基线、真实 A1 端到端提升**是四个不同证据层级，不得相互替代。


## 当前有效入口：v2（2026-10-20 接手修复）

v1 的数据/算子问题及修复预登记见 `reports/v2_preflight/REPAIR_AND_FREEZE.zh.md`。
v1 GPU CLI 已禁用；原结果只作为失败日志，不是 N1 门槛结果。
CPU 已通过的新数据在 `data_v2_ready/`；模型实验必须使用 `python -m care_m2_v2.cli`。
运行示例（仅授权服务器，在 gcx261014 环境）：

```bash
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
python -m care_m2_v2.cli run --data experiments/care_module2_incremental_20261020/data_v2_ready --out experiments/care_module2_incremental_20261020/runs/UNIQUE_RUN_ID --push
```

`run` 按 S1→N1→N2→N3→N4（门槛）→N5 执行，逐阶段提交同一分支。
各阶段的 `STATUS.md`、原始 journal、summary、逐例结果为实际状态依据；启动并不代表已完成。
