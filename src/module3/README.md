# 模块三：指南缓存键平移与历史干预接口

## 当前正文对应的独立执行器（2026-10-28）

[guideline_prefix.py](guideline_prefix.py) 先无编辑预填充指南，为每个患者创建独立缓存视图，在指定 KV 头的编号与处置位置加入 `g R_j Delta`，再处理整个患者/问题/回答前缀。支持多个层头、双规则正负编辑、直接分数偏置对照及精确 suffix-tail 优化。冻结为 Transformers 4.57.6 的 eager Qwen3、默认 RoPE、batch 1，拒绝不受支持的缓存或架构。

真实验证与全部对照见[2026-10-28 机制实验](../../experiments/module2_mechanism_20261028/RESULTS.zh.md)。实验使用 oracle 规则符号，不能替代当前模块二处置盲评估的联合连接或实际处置生成验证。实验目录与方向脚本中的 `module2` 保留旧编号，现对应论文模块一；当前研究执行器不复用下方历史组件或 `module1.v6` 门控。

## 历史门控干预（下文保留原阶段含义）

模块三根据模块一的门控，在模块二选出的 Qwen3 查询头上执行低秩键向量编辑，并保留受冲突条件约束的注意力输出抑制通路。实现位于 [intervention.py](intervention.py)。当前算子和冻结投影沿用原 TRIAGE 最小实现；模块一的默认入口已更新为 V6，但尚不能将旧实验结果解释为 V6 联合效果。

## 公共接口

| 接口 | 输入与作用 |
| --- | --- |
| `GatedAttentionIntervention(model, components, projection_path, module1_output, token_mapping, context_strength=0.1, prior_strength=0.1)` | 在上下文管理器内启用干预，退出时恢复模型原始前向函数和钩子 |
| `build_projections(model, tokenizer, calibration_records, components, output_path, rank=4, max_length=2048, max_records=4)` | 从独立校准提示构建键与注意力输出的低秩投影，并写入 `.pt` 与 `.json` |
| `select_intervention_heads(components, projections, heads)` | 从已选头中构造子集视图，不重新定位或校准 |

`components` 可为已读入的对象或 JSON 文件路径。投影可为 `.pt` 路径或已读入的投影对象；`select_intervention_heads` 也支持这两种形式。头编号从零开始。正常业务调用通过[统一流水线](../pipeline/README.md)完成；算子本身使用字段契约，流水线负责强制模块一版本为 `module1.v6`。

模块一输入字段为 `applicability`、`action_gate`、`condition_evidence_relations`、`relation_weights` 和 `conflict_scores`。词元映射由 [tokens.py](../pipeline/tokens.py)生成，包含规则动作以及条件、证据的词元位置。患者引用必须对应原文的精确字符区间 `[start, end)`。

## 当前算子的实际含义

### 上下文通路

对适用规则，算子将规则动作门控与关系权重相乘，分配到条件和患者证据词元。动作词元使用动作门控。同一词元被多条规则选中时取最大权重。

随后在 Qwen3 的分组查询注意力中先展开共享键值头，再只修改选中的查询头。键向量经过低秩投影后按门控与强度相加，之后继续执行旋转位置编码和注意力计算。未选中的查询头不会被这一步直接改写。

这是词元显著性层面的谱编辑。它不为每个查询词元建立独立的条件—证据关系偏置，也不使用 `signed_weight` 实现正负关系的成对干预。

### 先验竞争通路

该通路在 `o_proj` 前减去选定头的部分输出投影。激活需要同时满足：规则适用、动作门控为正、决定性证据完整且置信字段不低于 `0.8`，以及 `prior_support=oppose`、`guided_support=support`、`status=disagreement` 的显式冲突记录。多个规则的抑制门控取最大值。

当前 V6 不执行旧版先验／指南决策比较。其普通输出的先验竞争权重为零，支持方向均为 `unknown`，因此不会激活这条通路。V6 的 `confidence` 字段为干预许可对应的 `0.0` 或 `1.0`，不能解释为校准概率。输出基仅是注意力输出的近似子空间，尚未识别为独立的先验方向。

不适用或未知规则、零动作门控均不触发干预。上下文和先验通路分别在对应强度为零时关闭。两个通路都关闭时保留原注意力实现，以便核对零门控、零强度的一致性。

## 冻结组件与投影

| 内容 | 当前产物与范围 |
| --- | --- |
| 注意力头 | [components.json](../../results/module2/components.json)：层 35／头 8、层 12／头 24 |
| 投影参数 | [projections.pt](../../results/module3/projections.pt) |
| 投影元数据 | [projections.json](../../results/module3/projections.json) |
| 校准来源 | MEGA-CDP `train_val` 的指南 3；旧模块一缓存对应的实际提示 |
| 投影方法 | `k_norm` 和选定头注意力输出的未中心化奇异值分解，秩为 4 |
| 计算检查 | [cpu_checks.json](../../results/module3/cpu_checks.json)、[gpu_checks.json](../../results/module3/gpu_checks.json) |
| 历史汇总 | [report.json](../../results/module3/report.json)、[TRIAGE_MVP.md](../../TRIAGE_MVP.md) |

这些资产属于 Qwen3-8B。加载时校验层数、查询头数、键值头数、头维度、选定头集合，以及元数据中的模型路径。模块一输出兼容字段，并不说明旧校准已经验证了 V6 门控下的效果；V6 的二值门控也不能与旧版连续权重等同解释。

### 路径与再校准

默认产物路径由统一入口相对仓库根目录定位。冻结投影中的 `model_config.name_or_path` 仍记录 `/home/gcx/TRIAGE/models/modelscope/Qwen3-8B`，算子要求它与加载模型的路径一致。仅移动模型目录可能触发身份校验失败。当前整理保留这一检查和全部冻结产物，不修改元数据以绕过校验。

若后续使用其他模型路径或构建新投影，应在独立实验中核对模型身份、固定非测试校准数据，并输出到新的目录。`build_projections` 拒绝显式标为测试划分的记录；调用者仍需负责样本隔离。它默认最多读取 4 条记录，每条最多 2048 个词元，不会在正常推理时自动构建投影。

## 计算检查

[run_module3_smoke.py](../../scripts/run_module3_smoke.py) 的 `--cpu` 分支使用小型随机 Qwen3，检查实际修改、未选头一致性、关闭门控与强度、未知冲突、低置信证据、缓存生成及退出后的恢复。它需要 PyTorch 和支持 Qwen3 的 Transformers，不加载预训练权重，也不是临床有效性测试。

脚本按当前工作目录写入 `results/module3/cpu_checks.json`。从仓库根目录使用以下方式，可将本次检查结果写入临时目录：

```bash
TRIAGE_CHECK_SOURCE="$(pwd)"
TRIAGE_CHECK_OUTPUT="$(mktemp -d)"
(
  cd "$TRIAGE_CHECK_OUTPUT"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$TRIAGE_CHECK_SOURCE" \
    python -m scripts.run_module3_smoke --cpu
)
```

不带 `--cpu` 的分支会加载服务器 Qwen3、重新构建投影并写入结果，属于模型实验。其输入和路径沿用历史阶段，不能用作无副作用检查。

现有 [24 例诊断](../../results/algorithm_diagnostic/)保留了收益与退化，也保留了零门控和零强度的精确一致性记录。它没有显示稳定的总体准确率增益；更不能据此声称更新模块一后已获得联合增益。[ACRE 历史研究](../../docs/ACRE/)使用的独立注意力偏置实验也不等同于本模块的实现。
