# 统一流水线

当前入口是 `TriagePipeline`：先调用[模块一 V6](../module1/README.md)生成规则适用性与证据表示，再由同一个 Qwen3-8B 实例完成无干预和有干预的最终推理。模块一另外加载 TrialMatchAI 的 Phi-4 基座、LoRA 适配器和 BioLinkBERT-MedNLI；它与最终生成模型分开。

V6 的接口已经接入流水线。仓库中原有的组件、投影和 24 例流水线诊断来自旧模块一，不能作为 V6 与模块二、模块三联合有效性的验证。当前整理不重新定位注意力头、不构建投影，也不重跑模型。

## 入口与文件

| 文件 | 用途 |
| --- | --- |
| [runner.py](runner.py) | V6 调用、最终生成、门控干预、零门控对照与结果记录 |
| [tokens.py](tokens.py) | 应用聊天模板后，精确映射条件、动作和患者证据的词元位置 |
| [run_pipeline.py](../../scripts/run_pipeline.py) | 读取输入 JSON 和配置，保存一次推理结果 |
| [pipeline_minimal.json](../../configs/pipeline_minimal.json) | 命令行默认配置，只有实际生效的字段 |
| [模块三说明](../module3/README.md) | 干预算子、冻结投影及兼容范围 |

## 输入契约

命令行输入为一个 JSON 对象，包含 `guideline`、`patient`、`question`，可选 `input_source` 记录来源。流水线的患者文本和问题必须是非空字符串。

V6 使用事先准备好的结构化规则。`guideline` 可为规则列表、`{"rules": [...]}`，或含 `conditions` 的单条规则对象。它不再调用旧版规则检索与自由文本解析，也不会按 `top_k` 截断规则。每条规则应给出条件及稳定的条件编号；若用于最终动作生成，还应提供非空的 `action`。复合条件使用 `AND`、`OR`、`NOT` 的显式逻辑树。完整要求见[模块一说明](../module1/README.md)。

以下是合成接口示例，不属于临床实验数据：

```json
{
  "guideline": {
    "rules": [{
      "rule_id": "resource_a",
      "text": "If resource A is available, follow pathway A.",
      "conditions": [{"condition_id": "available", "text": "Resource A is available."}],
      "logic": {"op": "AND", "args": ["available"]},
      "exceptions": [],
      "action": "Follow pathway A."
    }]
  },
  "patient": "The current facility has resource A available.",
  "question": "Which pathway is applicable?",
  "input_source": {"kind": "synthetic_interface_example"}
}
```

预先计算的模块一结果只能通过 Python API 的 `module1_output` 传入，且 `schema_version` 必须为 `module1.v6`。旧版 `module1.v1` 缓存会被明确拒绝。

## 命令行运行与配置

模型推理在既有服务器环境中运行。执行前按[项目运行要求](../../README.md)报告 GPU、显存、样本规模和预计时长。V6 另外加载模型，不能直接套用旧 Qwen 流水线的显存记录。

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_pipeline.py \
  --config configs/pipeline_minimal.json \
  --input /absolute/path/input.json \
  --output results/pipeline/new_v6_run/inference.json
```

配置优先级为：命令行显式选项、配置文件、内置默认值。模块一未显式配置的模型路径和缓存继续由公共 V6 工厂读取环境变量。

| 配置字段 | 命令行选项 | 默认行为 |
| --- | --- | --- |
| `components_path` | `--components` | `results/module2/components.json` |
| `projection_path` | `--projections` | `results/module3/projections.pt` |
| `device` | `--device` | 最终 Qwen 使用 `cuda:0` |
| `max_new_tokens` | `--max-new-tokens` | 命令行保留原有上限 `1284` |
| `module1_device` | `--module1-device` | `CARE_MODULE1_V6_DEVICE`；未设置时跟随最终 Qwen 设备 |
| `module1_cache` | `--module1-cache` | `CARE_MODULE1_V6_CACHE`；未设置时使用模型加载器默认缓存 |

配置版本为 `triage_pipeline_config.v1`。配置文件中的相对产物路径和缓存路径按仓库根目录解析；命令行中的相对路径按当前工作目录解析。`--config`、`--input` 和 `--output` 也按当前工作目录解析。默认配置位置由脚本定位，所以从其他目录调用脚本时仍能找到仓库配置。

V6 工厂统一读取 `CARE_MODULE1_V6_BASE_MODEL`、`CARE_MODULE1_V6_ADAPTER_MODEL`、`CARE_MODULE1_V6_EVIDENCE_MODEL`、`CARE_MODULE1_V6_DEVICE` 和 `CARE_MODULE1_V6_CACHE`。可以将模型相关变量设为已有本地目录。最终 Qwen 路径由 [models.json](../../configs/models.json) 的 `qwen3_8b` 项配置；不由上述 V6 变量控制。

`--top-k` 和构造函数的 `top_k` 仅为旧调用保留，传入时会提示它对 V6 无效。`model_key`、`context_strength`、`prior_strength`、`use_cache` 等不再作为未接入的配置字段保留；未知配置字段会报错。当前统一入口固定使用 Qwen3-8B、`context_strength=0.1`、`prior_strength=0.1`、确定性生成及 `use_cache=False`。

## Python API

```python
from src.pipeline import TriagePipeline

pipeline = TriagePipeline(
    device="cuda:0",
    module1_device="cuda:1",
    max_new_tokens=384,
)
result = pipeline.run(guideline=rules, patient=patient_text, question=question)
```

设备编号为进程可见的逻辑编号，受 `CUDA_VISIBLE_DEVICES` 影响。上述设备组合只是调用示例，实际使用以服务器资源计划为准。

Python 构造函数保留原有 `max_new_tokens=384` 默认值；命令行保留原有 `1284`。需要比较结果时显式固定该参数。Python API 不自动读取命令行配置文件。可通过 `module1=` 复用已加载的 V6 实例，通过 `backend=` 注入最终生成后端；模块一与最终 Qwen 的模型实例仍分别管理。

## 输出与门控语义

结果版本为 `triage_pipeline.v1`。主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `module1`、`module1_schema_version`、`module1_metadata` | V6 完整结果、版本及模块一运行元数据 |
| `module1_source` | `computed_live_v6` 或 `provided_output` |
| `baseline`、`intervened` | 同一提示下的无干预和有干预生成，含文本、解析结果及 `json_valid` |
| `token_mapping`、`prompt` | 实际提示与精确词元映射 |
| `gates`、`intervention_stats` | 输入门控及算子实际修改统计 |
| `comparison` | 下一词元分数变化及生成文本是否变化 |
| `zero_gate_control` | 默认执行的零门控精确一致性检查 |
| `artifact_identity` | 组件与投影文件的 SHA-256 及固定强度 |
| `inference_settings` | 生成参数及共享模型的范围 |

`baseline` 已包含 V6 的结构化判断，并非无规则基线。最终提示只展示适用规则的原文和动作，保留其他分支的条件与状态。V6 的动作门控控制是否进行注意力干预；它不等于最终提示中的动作过滤条件。`json_valid` 检查输出结构，不检验动作与临床证据的语义一致性。

V6 的普通结果将 `prior_competition_weight` 设为零，且不提供旧版的先验与指南决策冲突，所以模块三的先验抑制通路默认不激活。`check_prior_control=True` 只构造明确标记的合成前向对照，不能解释为患者中观察到的真实先验冲突。

## 检查与历史复现

以下检查使用标准库和替身对象，不加载模型，也不改写冻结结果：

```bash
python -m unittest discover -s tests -p 'test_pipeline_config.py' -v
python scripts/run_pipeline.py --help
```

已有 V6 契约检查位于 [test_pipeline_contract.py](../../experiments/care_module1_v6/tests/test_pipeline_contract.py)。其中完整调用测试使用替身干预算子；通过它不能证明数值干预有效。

[24 例诊断](../../results/algorithm_diagnostic/)、[最小集成记录](../../TRIAGE_MVP.md)及 `run_pipeline_smoke.py`、`run_pipeline_coverage.py` 保留旧阶段的数据和调用假设。它们不是当前 V6 的快速开始入口。历史复现应使用对应提交和独立输出目录；不能将旧模块一缓存直接传入当前流水线，也不能用新输出覆盖历史依据。
