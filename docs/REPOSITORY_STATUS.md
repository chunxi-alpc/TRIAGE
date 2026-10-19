# 版本与整理记录

[项目首页](../README.md) · [文档导航](README.md) · [实验索引](../experiments/README.md)

核对日期：2026-10-19。本记录描述本次整理采用的固定来源版本，后续提交仍以 Git 历史为准。

## 当前版本

| 部分 | 当前入口 | 版本与状态 |
| --- | --- | --- |
| 模块一 | [src/module1](../src/module1/README.md) | CARE V6；包级 `run`、命令行和统一流水线使用同一模型构造入口 |
| 模块二 | [src/module2](../src/module2/README.md) | TRIAGE 查询头消融定位；已有组件来自旧模块一校准缓存 |
| 模块三 | [src/module3](../src/module3/README.md) | 原有门控键编辑和输出投影抑制；保留原投影与强度 |
| 统一流水线 | [src/pipeline](../src/pipeline/README.md) | 消费 `module1.v6`，复用原有 Qwen3-8B 组件和投影；尚无完整 V6 端到端基准 |
| CARE 模块二研究 | [最新实验](../experiments/care_module2_signed_20261019/README.md) | Qwen3-4B-Base 独立诊断，使用参考适用性标签；未替换在线模块二、三 |
| ACRE | [归档索引](ACRE/README.md) | 保留先导实验与完整规则路线的停止结论 |

## 来源提交

整理从主分支 `c005b52078c7682edff86535454890d59095f5c2` 开始。下列来源的代码与实验文件统一进入目录导航；同名当前接口采用 V6，首页重新组织。已有结果、预测、图表和校验清单保持来源内容。

| 来源 | 固定提交 | 纳入内容 |
| --- | --- | --- |
| 整理前主分支 | [c005b52](https://github.com/chunxi-alpc/TRIAGE/commit/c005b52078c7682edff86535454890d59095f5c2) | TRIAGE 原三模块、24 例诊断、CARE 三项观察 |
| 模块一 V6，PR #7 | [2cc2e84](https://github.com/chunxi-alpc/TRIAGE/commit/2cc2e84d8f80988c456f14b1387892ef46624858) | V6 实现、V6 预测与报告、统一流水线适配 |
| 模块一解析器 V3 复评，PR #6 | [f84dc58](https://github.com/chunxi-alpc/TRIAGE/commit/f84dc58fc39e9c332b91ac97abbf60f2db62985a) | V4/V5 实验、失败条件重生成、V5 历史兼容实现 |
| 模块二 B1–B4 后续，PR #3 | [3da16af](https://github.com/chunxi-alpc/TRIAGE/commit/3da16afb090a71db42843b9df29553f3e2610ceb) | B1–B4 对照、家族留出结果，以及已继承的 O2 二次统计复核 |
| 模块二有符号诊断，PR #8 | [4b2f6f3](https://github.com/chunxi-alpc/TRIAGE/commit/4b2f6f3ebf774aaecb786478b532fae13971e105) | 独立有符号 B4、均值位移与直接分数偏置实验 |
| ACRE 后续评审，PR #1 | [808f3b8](https://github.com/chunxi-alpc/TRIAGE/commit/808f3b866c27bc150dab47c35fb88713099a0d04) | V6 来源未包含的目标切换评审与交叉核对归档 |

其他阶段的来源为 [V4 / PR #4](https://github.com/chunxi-alpc/TRIAGE/commit/6a0f49841ace42939c3ad41047fd714890181afa)、[V5 / PR #5](https://github.com/chunxi-alpc/TRIAGE/commit/f7d3828b4a4d368da9edeb5d6da977b5601b3b9e)、[O2 复核 / PR #2](https://github.com/chunxi-alpc/TRIAGE/commit/8c5593e3d9e1a1bc9295bed916cb3e542198c574)。相关内容由表中后续来源一并保留。此表记录文件来源，不表示各来源分支中的方法均已成为默认算法。

## 需要明确的版本关系

1. **V6 是模块一的最新实现。** V6 已复制继承 TrialMatchAI 解析器及部分 V5 工具，但没有完整包含 V4/V5 实验目录。本次补齐历史实验与兼容实现，保留 V6 默认入口。
2. **模块二最新分支没有包含全部前序实验。** 有符号诊断从原主分支单独建立；B1–B4 后续分支已包含 O2 复核。两组新增目录均需保留。
3. **主流程与独立研究使用不同模型。** 当前模块一用 Phi-4、TrialMatchAI LoRA 和 BioLinkBERT-MedNLI；最终推理与历史组件用 Qwen3-8B；CARE 模块二独立实验用 Qwen3-4B-Base。
4. **旧产物不是 V6 的重新校准结果。** `results/module2`、`results/module3` 和原 24 例诊断仍有历史有效性，但不能据此声称 V6 流水线性能已复现。
5. **历史评价脚本保留原运行上下文。** 部分脚本依赖服务器路径、未入仓原始输入或大张量。各模块说明区分统计复算、模型重跑和仅可读取的归档。

## 本次整理范围

统一模块说明、实验和结果索引、首页导航；修复 V6 命令行入口与模型配置传递；使统一流水线配置实际生效并说明兼容参数。数值判定、证据融合、干预算子与冻结实验统计不在本次算法修改范围内。

模块一现有证据门控和数值抽取的实现限制见 [模块一说明](../src/module1/README.md)。这些限制属于后续算法工作，不能用文档整理替代其验证。

本次不重新运行模型，不新增性能结果。工程检查只覆盖接口、配置、静态完整性和明确列出的离线复核。

## 本次检查

| 检查 | 结果与范围 |
| --- | --- |
| 模块一接口 | 8 项标准库测试通过，覆盖 V6 默认入口、历史入口、模型配置与前置输入校验 |
| 流水线接口与配置 | 11 项标准库测试通过，覆盖配置覆盖、路径、空环境变量回退、无效输入和旧图拒绝 |
| 原 V6 基本检查 | `python experiments/care_module1_v6/tests/test_v6.py` 中 3 项检查通过 |
| 文档与语法 | 17 份新增或更新 Markdown 的 339 个相对链接均有效；所检出 90 个 Python 文件语法通过，JSON 解析及 `git diff --check` 通过 |
| 模块一保存指标 | 从旧 192 条和新增 100 条公开预测、标签独立重算 A/B/C 准确率及宏平均 F1，与保存指标一致 |
| 模块二保存指标 | 从已选预测重算五种方法的成对正确数，分别为 72、74、75、75、77，对应每种方法 192 对 |
| O2 统计复核 | 对照 143 个数值，最大绝对差约 `2.61e-15`；输入 Git blob 与 SHA-256 均核对，来源描述和解释文字按各版保留 |
| 算法保持 | AST 对比确认 V6 原模型、数值提取、融合和门控定义保持原样；仅新增配置、校验辅助函数并调整入口及版本错误信息 |

接口测试使用替身对象。当前检查环境未安装 PyTorch、Transformers，没有执行完整历史测试集、随机模型数值干预或真实模型推理。保存结果的统计复核没有重新生成预测，也没有覆盖冻结产物。
