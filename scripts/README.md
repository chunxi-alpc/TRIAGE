# 脚本索引

[项目首页](../README.md) · [运行说明](../docs/RUNNING.md) · [实验索引](../experiments/README.md)

脚本按其对应版本使用。新模块二研究以 A1 作为模块一固定信号来源；旧 `python -m src.module1` 和原流水线仍是 V6，以便历史复现。

## 当前入口

| 入口 | 作用 |
| --- | --- |
| [run_module1_a1.py](run_module1_a1.py) | **新模块二研究默认**：轻量 A1 规则 T/F/U；输入示例见 [A1](../examples/module1_a1_input.json) |
| `python -m src.module1` | 历史 V6 CLI；参数见 [模块一说明](../src/module1/README.md) |
| [run_pipeline.py](run_pipeline.py) | 结构化输入 → V6 → 原 Qwen3-8B 门控推理；配置见 [流水线说明](../src/pipeline/README.md) |

完整模型推理只在既有服务器和指定环境运行。先核对模型路径、组件与投影，再启动；格式示例见 [module1_v6_input.json](../examples/module1_v6_input.json)。

## 历史 TRIAGE 组件与诊断

| 脚本 | 对应阶段与用途 |
| --- | --- |
| [run_module1_smoke.py](run_module1_smoke.py)、[test_module1_logic.py](test_module1_logic.py) | 旧 `Module1` / `module1.v1` 的模型冒烟与逻辑回归 |
| [run_module2_smoke.py](run_module2_smoke.py) | 从旧模块一缓存定位查询头，不是 V6 的重新校准 |
| [run_module3_smoke.py](run_module3_smoke.py) | 门控干预的历史 CPU/GPU 检查 |
| [run_pipeline_smoke.py](run_pipeline_smoke.py)、[run_pipeline_coverage.py](run_pipeline_coverage.py) | 旧缓存或自由文本输入的历史集成脚本，未迁移为 V6 输入协议 |
| [run_algorithm_diagnostic.py](run_algorithm_diagnostic.py) | 旧 24 例算法诊断；关联历史模块一缓存与产物 |
| [summarize_algorithm_diagnostic.py](summarize_algorithm_diagnostic.py) | 已存诊断输出的基础指标汇总；会写回 `--results` 目录。先复制完整诊断输入到独立目录，再指定 `--results`，避免覆盖原结果及后来补充的摘要字段 |
| [run_diagnostic_free_generation.py](run_diagnostic_free_generation.py) | 历史自由文本补充，不能直接当当前 V6 端到端验证 |

旧 `module1.v1` 缓存不能直接传给当前 V6 流水线。需要复现旧端到端模型行为时，应在独立检出中使用 [整理前主分支提交](https://github.com/chunxi-alpc/TRIAGE/commit/c005b52078c7682edff86535454890d59095f5c2)，并复用该阶段的模型和数据；当前目录仅保留其脚本与结果依据。

## 独立实验

- CARE 模块一 V4–V6：脚本随各 [实验目录](../experiments/README.md) 保存。
- CARE 三项观察、O2 统计复核与 B1–B4 后续：见 [模块二说明](../src/module2/README.md)。
- CARE 有符号诊断：见 [实验入口](../experiments/care_module2_signed_20261019/README.md)。
- `*acre*` 脚本：独立 ACRE 先导、审计、统计与复现；对应 [归档索引](../docs/ACRE/README.md)。

## 环境与数据准备

[activate.sh](activate.sh)、[install_env.sh](install_env.sh) 及 `download_*`、`verify_*`、`validate_*` 脚本保留原服务器和历史数据准备流程。配置来源为 [模型清单](../configs/models.json)、[数据清单](../configs/datasets.json) 和 [历史环境记录](../configs/environment.lock.txt)。运行这些脚本前需核对实际目标路径；普通统计复核不需要重复下载或重建环境。
