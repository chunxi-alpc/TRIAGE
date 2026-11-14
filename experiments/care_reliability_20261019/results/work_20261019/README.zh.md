# 原服务器真实实验结果

[给老师的简报](TEACHER_REPORT.zh.md)；[机器可读摘要](summary.json)；[联合完整指标](joint_metrics.json)；[资源预算与实际消耗](resource_plan.json)。

全部实验在 `gcx@10.29.128.115` 执行，未在本地运行模型或测试。模块二/Qwen 使用 `/home/gcx/gcx261014`，Phi-4/NLI 使用原 `/home/gcx/n/.venv`。GPU 为 0、1、2、3、6、7，4、5 未使用。

## 原始数据归档

- `module2_raw_runs.tar.gz`：首次运行与数值修复后的全部六项正式运行、逐例日志、零控制、失败捕获和无答案数值诊断。
- `joint_raw_runs.tar.gz`：因扩卡中止的两片尝试与完整六片正式运行，包含 1,090 个原始生成、请求对应关系、V6/V7 原始结构、2,496 条答案、768 次零控制、模型来源和运行日志。
- `preparation_raw.tar.gz`：开始时已有的统一准备检查。

各归档的文件大小与 SHA256 在 [archive_manifest.json](archive_manifest.json)。所有服务器原始目录均保留：

- `/home/gcx/care_reliability_runs/module2_work_01`、`module2_work_02`
- `/home/gcx/care_reliability_runs/joint_formal_work_01`、`joint_formal_work_02`

归档使用相对目录，可解压到任意新目录后按各模块分析器读取。联合评分可运行 `joint/evaluate.py`，输入冻结 `data/mechanism_cases.jsonl` 和解压后的 `joint_formal_work_02/joint/predictions.jsonl`。模块二各模式使用 `module2/analyze.py` 读取解压后的 `module2_work_02/legacy384_regression` 或 `sealed_mechanism`。不需要重新生成模型输出。

## 结果使用范围

所有原定病例均保留；解析失败、未知状态和无操作并未被删除。参考结构仅为诊断。V7 的规则状态准确率有次要改善，但未改善主要成对终点；学习方向未显示相对直接偏置的主要收益。具体限制和用户提出的过严规则问题见 [限制分析](../../checks/work_verification/threshold_review.zh.md)。
