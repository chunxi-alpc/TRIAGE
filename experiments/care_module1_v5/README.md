# CARE 模块一 V5：历史实验与解析器修复

[实验索引](../README.md) · [当前模块一 V6](../../src/module1/README.md) · [V4](../care_module1_v4/README.md)

本目录保留 V5 的模型组合、解析失败、离线修复和失败条件重生成。当前包级入口已经使用 V6；本目录的报告与结果按各自运行阶段保留。

## 优先阅读

1. [解析器 V3 失败条件重生成报告](runs/v5_20261018/parser_v3_retry/RETRY_REPORT.zh.md)：本目录最后一次复评。
2. [解析器 V2 离线复评](PARSER_V2_REPORT.zh.md)：从缓存的原始输出重新解析。
3. [初始完整报告](REPORT.zh.md) 与 [原运行说明](README.zh.md)：理解初始失败及模型组合路线。

## 阶段与结论

| 阶段 | 条件解析成功 | TrialMatchAI 单独规则判定 | 组合结果与限制 |
| --- | ---: | ---: | --- |
| 初始 V5 | 75/492 | 60/192 | 初始解析失败影响覆盖率，原结果保留 |
| 解析器 V2 离线修复 | 423/492 | 129/192 | 数值与缓存 MedNLI 组合为 98/192；覆盖率提升没有带来组合改善 |
| 解析器 V3 重生成 | 492/492 | 150/192 | TrialMatchAI + BioLinkBERT 为 96/192；证据定位仍为 76/137 |

表格依据 [V2 报告](PARSER_V2_REPORT.zh.md) 与 [V3 报告](runs/v5_20261018/parser_v3_retry/RETRY_REPORT.zh.md)。各行使用对应阶段的模型输出或缓存，不应将不同阶段的数值拼为一条方法结果。V3 仅重生成 V2 后仍失败的 69 条条件，其余 423 条输出保留。

该数据用于方法开发与探索性复评，不能作为独立临床验证。V6 调整的是融合方式，使 BioLinkBERT 不再覆盖适用性真假标签；V6 结果另见 [V6 实验目录](../care_module1_v6/README.md)。

## 其他资料

- [解析错误分析](ANALYSIS_PARSE.zh.md) 与 [判定错误分析](ANALYSIS_ERRORS.zh.md)。
- [详细错误分析](analysis/ERROR_ANALYSIS.zh.md)。
- [环境及失败处理](SETUP_AND_FAILURES.zh.md) 与 [历史依赖](requirements-v5.txt)。
- [探索性补充](exploratory/README.zh.md)。
- [V5 兼容接口说明](../../src/module1/V5_COMPATIBILITY.zh.md)：历史阶段说明，当前默认入口以 [模块一 README](../../src/module1/README.md) 为准。

完整原始模型输出和部分原始病例保留在服务器。复评输出、哈希、代码和公开记录留在本目录；本次整理不重生成模型输出或修改已冻结的结果。
