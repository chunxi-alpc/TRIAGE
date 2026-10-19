# 运行说明

[项目首页](../README.md) · [模块一](../src/module1/README.md) · [统一流水线](../src/pipeline/README.md) · [脚本索引](../scripts/README.md)

当前模块一输出 `module1.v6`，必须提供预先结构化的规则。旧自由文本指南和 `module1.v1` 缓存用于历史实现，不能直接替代 V6 输入。

## 环境与模型职责

| 用途 | 模型或资产 | 配置来源 |
| --- | --- | --- |
| V6 原子条件判断 | Phi-4 + TrialMatchAI LoRA，四位量化加载 | `CARE_MODULE1_V6_BASE_MODEL`、`CARE_MODULE1_V6_ADAPTER_MODEL` |
| V6 证据定位 | BioLinkBERT-MedNLI | `CARE_MODULE1_V6_EVIDENCE_MODEL` |
| 统一流水线最终生成 | Qwen3-8B | [configs/models.json](../configs/models.json) 中 `qwen3_8b` |
| 历史组件与投影 | 已保存的查询头和 rank=4 投影 | [configs/pipeline_minimal.json](../configs/pipeline_minimal.json) |
| CARE 模块二独立研究 | Qwen3-4B-Base 及其冻结实验资产 | [实验说明](../experiments/care_module2_signed_20261019/README.md) |

根 [requirements.txt](../requirements.txt) 保留原环境依赖；[requirements-module1-v6.txt](../requirements-module1-v6.txt) 增补 V6 模型加载需要的 `bitsandbytes`。重建环境时可使用后者，但它不是完整 V6 运行环境的新实测锁定文件。V5 等历史复现使用对应实验中的原版本记录，不能混用不同 `transformers` 版本。

仓库记录的执行位置为 `gcx@10.29.128.115:/home/gcx/TRIAGE`，已有环境为 `/home/gcx/gcx261014`。本次整理没有连接服务器验证其当前状态或重新配置模型。

```bash
ssh gcx@10.29.128.115
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE
```

在已授权的服务器环境中核对三个模型目录后配置：

```bash
export CARE_MODULE1_V6_BASE_MODEL=/absolute/path/to/phi-4
export CARE_MODULE1_V6_ADAPTER_MODEL=/absolute/path/to/trialmatchai-lora
export CARE_MODULE1_V6_EVIDENCE_MODEL=/absolute/path/to/biolinkbert-mednli
export CARE_MODULE1_V6_DEVICE=cuda:0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

这些模型目录是需替换的占位路径。仅设置缓存目录不等于指定本地模型目录；离线变量也不能替代缺少的模型文件。工厂的显式参数优先于环境变量，完整说明见 [模块一配置](../src/module1/README.md#模型配置与依赖)。

## 结构化输入

[examples/module1_v6_input.json](../examples/module1_v6_input.json) 给出可读取的格式示例。它是虚构规则与病例，只用于说明接口；没有附带模型预测或性能结果。

输入 JSON 含：

- `guideline`：包含规则 ID、原文、原子条件、逻辑树和动作的结构化对象或列表。
- `patient`：病例文本字符串，字符区间依此原文计算。
- `question`：当前判断问题。
- `input_source`：可选的来源记录，供流水线保存。

数值约束写在原子条件的 `numeric` 中，包含 `operator`、`value` 和适用的 `unit`；字段名称通过条件顶层的 `field` 或 `numeric.field` 提供。模块一的完整输入要求和数值处理限制见 [输入契约](../src/module1/README.md#输入契约)。

## 单独运行模块一

```bash
python -m src.module1 --help

CUDA_VISIBLE_DEVICES=0 python -m src.module1 \
  --input examples/module1_v6_input.json \
  --output results/module1_v6/example.json
```

`--help` 只显示参数。第二条命令会加载模型并执行推理。批量推理应通过 `create_module1_v6()` 复用同一个实例。

历史检索式模块一通过 `--legacy` 或 `LegacyModule1` 显式调用；`--top-k` 只用于该历史模块。运行原端到端诊断时，还需使用其对应版本的流水线，见 [脚本索引](../scripts/README.md)。

## 运行统一流水线

```bash
python scripts/run_pipeline.py --help

CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py \
  --config configs/pipeline_minimal.json \
  --input examples/module1_v6_input.json \
  --output results/pipeline/v6_example/inference.json
```

该流程实时生成 V6 图，再对同一最终提示执行原始和干预生成。模块一独立加载 Phi-4、LoRA 和 NLI 模型；两种最终生成共用 Qwen3-8B。设备分配、配置字段和显式参数优先级见 [流水线说明](../src/pipeline/README.md)。可用 GPU 为 0、1、2、3、6、7；启动前需说明计划用卡、显存、样本量和时长。

旧投影与组件保留模型身份校验。产物路径可以配置，不代表权重、模型名称或任意新投影都可互换。正常运行不从待评价病例自动估计投影，也不将参考答案输入常规门控。

## 离线检查与实验复现

无模型接口检查可以在普通 Python 环境执行：

```bash
python -m unittest discover -s tests -p 'test_module1_interfaces.py' -v
python -m unittest discover -s tests -p 'test_pipeline_config.py' -v
```

这类检查验证配置、接口与路由，不能替代真实模型推理或干预算子的数值验证。更多当前检查见 [流水线说明](../src/pipeline/README.md)。

| 任务 | 能否从已存产物复核 | 入口与限制 |
| --- | --- | --- |
| V6 指标 | 已有公开标签与逐例预测可用于独立复算 | [V6 导航](../experiments/care_module1_v6/README.md)；原 `evaluate_v6.py` 依赖服务器私有输入并写回原结果，不是通用只读入口 |
| O2 头选择复核 | 可用标准库重算保存的逐头统计 | [模块二说明](../src/module2/README.md)；显式指定新输出路径 |
| B1–B4 后续统计 | 可读取压缩预测复算 | [模块二说明](../src/module2/README.md)；需显式设置 `--artifacts`，保留原子组统计 |
| 有符号诊断统计 | 解压已存预测后复算 | [实验说明](../experiments/care_module2_signed_20261019/README.md)；原分析脚本仍依赖 PyTorch、Transformers 等库，但统计复算不加载模型 |
| 历史 GPU 重新运行 | 依赖原模型、数据和部分大张量 | 使用对应冻结协议、服务器或完整归档，并写入新实验目录 |

模型实验在既有服务器运行；静态检查、接口替身测试和已存结果的统计复核不需要占用 GPU。重跑时保留失败、样本选择、模型指纹和参数，不覆盖旧预测、图表或校验清单。
