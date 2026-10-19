# 模块一：规则适用性与证据定位

当前入口为 **CARE V6**，输出模式为 `module1.v6`。实现来自 [V6 集成提交](https://github.com/chunxi-alpc/TRIAGE/commit/2cc2e84d8f80988c456f14b1387892ef46624858)。本次整理统一了包级调用、命令行和流水线的模型配置入口，保留原有适用性计算、数值解析和门控算法。

[项目首页](../../README.md) · [统一流水线](../pipeline/README.md) · [V6 实验导航](../../experiments/care_module1_v6/README.md)

## 当前实现

V6 接收预先结构化的指南规则，逐项判断条件，并返回规则适用性、证据字符区间和注意力干预门控。

1. **条件判断。** Phi-4 加载 TrialMatchAI 的 LoRA，针对每个原子条件生成官方格式的分类。解析器只接受唯一目标条件对应的有效分类。纳入条件映射为真、假、未知（`T/F/U`）；排除条件的 `Not Violated` 保守映射为 `U`。
2. **数值比较。** 对已支持字段进行数值提取、单位换算和阈值比较。比较得到 `T/F` 时覆盖该原子条件；数值缺失或跨阈值矛盾时保留 TrialMatchAI 状态。主体和时间的识别仍有限，见“已知限制”。
3. **证据定位。** BioLinkBERT-MedNLI 对候选句与条件进行自然语言推断，记录支持、反驳和中立证据及其原文位置。它不改写最终条件适用性。
4. **规则合成。** 使用强克里尼三值逻辑计算 `AND`、`OR`、`NOT`。有排除条件时，规则状态为主条件与“排除条件不成立”的合取。
5. **下游接口。** 返回候选规则、条件—证据关系、适用性和门控字段。当前权重不是校准概率；普通 V6 输出的先验竞争权重为零。

V6 本身不调用 Qwen，不执行指南检索，不解析自由文本指南，也不训练模型。统一流水线仍使用 Qwen3-8B 完成后续生成；模型职责和验证范围见[流水线说明](../pipeline/README.md)。

## 输入契约

输入 JSON 顶层包含 `guideline`、`patient`、`question`。可从[结构化示例](../../examples/module1_v6_input.json)开始。

| 字段 | 要求 |
| --- | --- |
| `guideline` | 结构化规则列表、`{"rules": [...]}`，或含 `conditions` 的单条规则；不支持字符串形式的指南 |
| `rule_id` | 建议始终提供唯一的规则标识 |
| `text`、`action` | 规则原文与动作字符串；下游提示需要保留完整适用条件和动作 |
| `conditions` | 原子条件列表；每项提供唯一 `condition_id` 和非空 `text` |
| `logic` | 由条件 ID 构成的 `AND`、`OR`、`NOT` 树；省略时按所有主条件的 `AND` 处理 |
| `exceptions`、`exception_logic` | 排除条件及其逻辑；默认无排除条件，非空时默认使用 `OR` |
| `numeric` | 可选的数值约束，包含 `field`、`operator`、`value`，可选 `unit`；字段也可在条件顶层提供 |
| `patient` | 模块一支持病例文本或句子字符串列表；统一流水线使用病例文本字符串 |
| `question` | 非空问题字符串；V6 不根据问题重新检索或筛选候选规则 |

无条件规则必须明确设置 `explicitly_unconditional: true`。否定条件使用逻辑树的 `NOT` 节点，不使用 `negated: true`。字符区间采用 Python 字符索引 `[start, end)`，不使用字节偏移。原始病例、句子顺序和输出证据区间必须保持一致。

`validate_v6_input(guideline, patient, question)` 可独立执行无模型输入检查。它复制规则后复用现有结构检查，不修改调用方输入。包级 `run()`、模块一命令行和流水线实时入口在创建或加载模型前执行该检查；错误结构以 `ValueError` 或命令行参数错误报告。该检查不新增医学规则或逻辑树语义验证。

## Python 与命令行

一次性调用会延迟创建并复用默认 V6 实例：

```python
from src.module1 import run

graph = run(guideline=structured_rules, patient=patient_note, question=question)
assert graph["schema_version"] == "module1.v6"
```

需要明确控制模型实例时，使用共享工厂。批量处理应复用同一个实例：

```python
from src.module1 import create_module1_v6

engine = create_module1_v6(device="cuda:0")
graph = engine.run(guideline=structured_rules, patient=patient_note, question=question)
```

工厂还接受 `cache_dir`、`base_model`、`adapter_model`、`evidence_model`、`nli_batch_size` 和 `candidate_limit`。其中 `base_model`、`adapter_model`、`evidence_model` 可使用本地模型目录。若运行期间改变默认模型环境变量，先调用 `set_default_v6(None)` 清除已有默认实例，或显式创建新实例。

命令行默认运行 V6：

```bash
python -m src.module1 --help
python -m src.module1 \
  --input examples/module1_v6_input.json \
  --output results/module1_v6/example.json
```

第二条命令会加载模型并执行推理。模型实验遵循[项目首页](../../README.md)的服务器运行约定。

## 模型配置与依赖

包级 `run()`、命令行和统一流水线均使用 `create_module1_v6()`。配置优先级为：**显式参数 > 环境变量 > 默认模型 ID**。

| 环境变量 | 默认值或用途 |
| --- | --- |
| `CARE_MODULE1_V6_DEVICE` | `cuda:0`；流水线可显式指定模块一设备 |
| `CARE_MODULE1_V6_CACHE` | Hugging Face 模型缓存目录；未设置时使用库默认值 |
| `CARE_MODULE1_V6_BASE_MODEL` | `microsoft/phi-4` |
| `CARE_MODULE1_V6_ADAPTER_MODEL` | `majdabd33/trialmatchai-phi4-reasoning-lora` |
| `CARE_MODULE1_V6_EVIDENCE_MODEL` | `cnut1648/biolinkbert-mednli` |

离线复用服务器现有模型时，将三个模型变量指向对应本地目录，并使用 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。仅设置缓存目录不等于指定三个模型目录。TrialMatchAI 加载器会检查基础模型身份及有效 LoRA 权重，不会静默退回基础模型。

[requirements-module1-v6.txt](../../requirements-module1-v6.txt) 在根依赖上补充了四位量化所需的 `bitsandbytes`。其版本来自[已有 V5 环境记录](../../experiments/care_module1_v5/provenance/environment.json)。该文件用于补齐当前入口依赖，不代表 V6 新建环境的完整实测锁定记录。重现 V5 时使用其[历史依赖文件](../../experiments/care_module1_v5/requirements-v5.txt)；其中 `transformers` 版本与根依赖不同。

## 历史实现的显式入口

| 入口 | 实现与用途 |
| --- | --- |
| `src.module1.run`、`create_module1_v6`、`CAREModule1V6` | 当前 V6 |
| `src.module1.LegacyModule1`、`src.module1.run_legacy` | 历史 Qwen/MedCPT 模块一，输出 `module1.v1` |
| `src.module1.Module1` | 为旧脚本保留的 V1 类别名；新代码应使用明确命名 |
| `src.module1.care_v5`、`src.module1.compat_v5` | V5 实现及其历史输入适配；用于读取和重现 V5 实验 |

历史命令使用显式参数：

```bash
python -m src.module1 --legacy --top-k 12 \
  --input /absolute/path/legacy_input.json \
  --output results/module1_legacy/output.json
```

`--top-k` 只用于历史检索入口，V6 不接受该参数。V1/V5 输出不能直接作为当前统一流水线的 `module1_output`；流水线要求 `module1.v6`。V5 兼容说明中的“默认入口”属于当时版本记录，不适用于当前包入口。

## 实验与历史证据

| 阶段 | 入口 | 解释范围 |
| --- | --- | --- |
| V6 | [当前目录导航](../../experiments/care_module1_v6/README.md)、[冻结报告](../../experiments/care_module1_v6/REPORT.zh.md) | 旧 192 例由已有 V5 逐例结果重新合成；新 100 例有 V6 逐例预测。不是完整三模块端到端效果验证 |
| V5 与解析重试 | [V5 实验](../../experiments/care_module1_v5/README.md)、[重试报告](../../experiments/care_module1_v5/runs/v5_20261018/parser_v3_retry/RETRY_REPORT.zh.md) | 保留当时模型、解析器、证据融合和重试依据 |
| V4 | [V4 实验](../../experiments/care_module1_v4/README.md) | 早期规则与受控病例回归结果 |
| 原始 TRIAGE | [历史诊断摘要](../../results/algorithm_diagnostic/summary.json) | Qwen/MedCPT 模块一及原三模块小规模诊断 |

V6 的条件解析器与 [PR #6](https://github.com/chunxi-alpc/TRIAGE/pull/6) 中的文件一致。V6 与 V5/解析重试分支并非完整线性继承；本仓库按目录补齐历史实现和产物，并保留 V6 的当前入口。

## 已知限制

- **数值主体与时间尚不可靠。** 当前实现依赖字段别名和正则表达式，只有有限的撤回语句处理。它没有完整识别亲属信息、历史测量或当前时间范围。不能把数值覆盖描述为“已经验证主体与时间”。
- **规则门控没有完整检查证据证明。** 对合取规则，若多个条件均被判为真，一个条件有支持证据而另一个缺证据，规则动作门控仍可能开启。当前“决定性条件”字段也不是逻辑最小证明。证据不足并不保证所有相关规则干预都关闭。
- **仅有数值证据时存在状态不一致。** 如果 NLI 全为中立，数值证据可能已加入 `support_evidence` 或 `refute_evidence`，但该条件的 `evidence_status` 仍为 `missing`，相应条件门控关闭。
- **证据置信度未校准。** 关系置信度目前为 `0/1`，由条件门控决定。候选句检索和 NLI 预测仍可能定位过宽或冲突；长句超过模型预算时跳过，不截断后宣称已验证。
- **先验抑制路径默认未激活。** V6 的 `prior_competition_weight` 为零，先验与指南动作支持字段为 `unknown`。普通 V6 运行不能解释为已经使用了原流水线的先验冲突抑制。
- **实验与接口验证范围不同。** 保存的旧样本包含方法开发时使用过的受控样本；数据协议未声称独立医生复核。V6 接口可连接模块二、三，但旧投影和旧流水线结果不能证明新组合有效。

上述问题是现有算法的限制。本次入口整理没有改写冻结预测、统计或模型判断；后续算法修正需使用新的实验版本记录。

## 无模型接口检查

从仓库根目录运行：

```bash
python -m unittest discover -s tests -p 'test_module1_interfaces.py' -v
```

这些检查使用替身模型验证输入前置检查、工厂配置、实例复用、V6 默认命令行和显式历史入口，不加载模型、不调用 GPU、不改写实验结果。原有 V6 算法与流水线契约检查保留在[实验测试目录](../../experiments/care_module1_v6/tests/)。
