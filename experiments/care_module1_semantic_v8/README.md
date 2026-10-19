# CARE 模块一 V8：轻量语义适用性识别

本实验在 `gcx@10.29.128.115` 实际运行，独立分支基于 PR #11 的 `f0c3c28a30096bdda37c021f6db9de85754cb8e7`。历史代码、分支和输出未覆写。结果与结论见 [REPORT.zh.md](REPORT.zh.md)，给老师的简报见 [TEACHER_REPORT.zh.md](TEACHER_REPORT.zh.md)。

## 方法与数据

- **A0**：Phi-4 + 原 TrialMatchAI LoRA，直接判断每条完整自然语言规则。每病例一次请求同时输出全部三条规则的 T/F/U、原文摘录和一句依据。
- **A1**：相同权重、量化与推理参数，增加三个固定训练演示：T/F 最小对比、例外/否定、信息不足。第一个演示含两个患者叙述，因此三个演示共四个患者示例；来源均为历史 train 家族，与 dev/confirmation 家族分离。
- **A2 历史**：直接复用 PR #11 修复后的 192 例原子判定及三值聚合结果，不重新运行模型；96 例 dev 没有对应缓存，不补跑。
- **A2 新数据**：新病例提供可用的作者预解析规则结构，实际执行原子命题判定并复用修复后的三值聚合。具体实现差异与调用预算见 A2 说明和运行清单。

历史 96 例 dev 用于选择默认语义版本，优先比较规则准确率，其次 F→T 数量、宏 F1，完全相同则选 A0。选型规则在结果前保存在 `protocol.json`，结果后固定于 `selection.json`。A0/A1 都运行完整历史 192 例和新 72 例。历史 192 例曾用于开发，只称历史回归探索，不称新的独立确认。

新 72 例由 12 个新语义主题组成，每家族两对只改变关键条件的 T/F 病例及两个 U 场景，共 216 个规则判断。病例、独立参考标签和预测前冻结记录分别位于 `data/new72_cases.jsonl`、`data/new72_gold.jsonl`、`data/manifest_new72.json`。参考标签由作者指定的事实真假及可执行三值逻辑产生，不使用正在评估的模型制定。数据是人工构造的虚构管理分流情境；新增的是语义主题，三值逻辑运算沿用已有定义。只有 Codex 作者文本阅读检查，没有独立临床专家审核。

## 输入与失败处理

语义预测输入只从正向字段清单构造：问题、患者文本、全部规则 ID 和自然语言规则。历史规则从既有指南逻辑无损展开为自然语言，包含完整相邻分支；没有把患者结构化事实、参考状态、参考证据或正确动作送入预测。示例的已知输出仅用于固定训练演示。

每条输出包含 `rule_id,state,evidence_quote,reason_short`。字符位置由程序根据原始规则 ID 和患者原文确定，动作不由模型重生成。文本证据无法定位仅记诊断，不修改 T/F/U、不阻断判定。解析失败最多一次简短格式重试；再次失败保存空状态和失败原因，不转换为医学 U。所有病例进入分母，不按效果剔除。

A2 获得预解析原子命题和逻辑树；历史 A2 还保留原有数值解析优先逻辑。A0/A1 获得等价规则语义与相邻分支，但无机器可执行结构。比较同时包含算法差异和输入结构优势，不能将差值全部归因于模型语义能力。

## Qwen 接入

统一接口为 `S={rule_id,state,action_char_span,optional_evidence_char_span}`。在 Qwen 的完整实际提示分词后映射 token 位置，保留所有规则和动作，U 不产生动作分数偏置，技术失败保持独立标记。仅沿用 PR #11 的 C 配置：18–23 层、强度 1、最后一个查询位置；不重新搜索层、头或强度。

在提前冻结的 24 个历史病例上运行基线、仅 V8 结构提示、仅干预、二者组合，保留同病例 PR #11 记录。此部分用于检验接口和决策变化。结构提示收益不能归因于注意力；Qwen 本身仍收到历史预解析指南，不能据此声称整个系统已实现纯自然语言输入。

## 复现

均在服务器 worktree 根目录执行。已有结果默认保留，语义运行器跳过已有病例；如需重新实验，应另建输出目录。

```bash
source /home/gcx/gcx261014/bin/activate
python experiments/care_module1_semantic_v8/prepare_historical.py

CUDA_VISIBLE_DEVICES=0 /home/gcx/n/.venv/bin/python experiments/care_module1_semantic_v8/semantic_runner.py \
  --cases experiments/care_module1_semantic_v8/data/semantic_dev_inputs.jsonl \
  --arm A0 --output experiments/care_module1_semantic_v8/results/dev_A0
# A1 使用相同命令与参数，将 arm/output 改为 A1，可用 GPU1。

python experiments/care_module1_semantic_v8/analyze.py --select \
  --new-cases experiments/care_module1_semantic_v8/data/new72_cases.jsonl

CUDA_VISIBLE_DEVICES=0 bash experiments/care_module1_semantic_v8/run_followup.sh A0 \
  experiments/care_module1_semantic_v8/data/new72_cases.jsonl
# A1 同理，使用 GPU1。

python experiments/care_module1_semantic_v8/analyze.py \
  --new-cases experiments/care_module1_semantic_v8/data/new72_cases.jsonl

CUDA_VISIBLE_DEVICES=2 /home/gcx/n/.venv/bin/python experiments/care_module1_semantic_v8/a2_new72.py
CUDA_VISIBLE_DEVICES=3 python experiments/care_module1_semantic_v8/qwen_smoke.py \
  --predictions experiments/care_module1_semantic_v8/results/confirmation_A1/predictions.jsonl --variant A1
python experiments/care_module1_semantic_v8/build_reports.py
```

精确参数、环境版本、模型来源、提示、每次输出、重试与耗时均保存在 `results/*/manifest.json` 和 `results/*/predictions.jsonl`。语义推理细节见 `INFERENCE.zh.md`；Qwen 和 A2 的运行记录随对应结果保存。A2与Qwen运行器遇到已有结果会拒绝覆盖；重跑应使用新的输出目录，或在另一个独立worktree中进行。

## 指标定义与产物

`analysis/metrics.json` 保存规则准确率、T/F/U 宏 F1、完整混淆矩阵（额外 ERROR 列）、F→T、U 召回、有效输出率、配对状态和证据定位统计。`summary.csv` 为紧凑表，`family_metrics.json` 为家族结果，`rule_predictions.jsonl` 为完整逐规则评价，`paired_comparison.json` 为同病例 A0/A1 对 A2 的改变明细。

配对准确率要求目标规则在一对病例中分别正确输出 T 和 F；另保存所有发生 T/F 翻转的规则对及整例全部规则正确指标。历史全部配对包含不同改写版本，不能当成完全独立样本。证据定位率只表示非空摘录是否能在患者原文找到，不代表证据完整或推断成立。耗时以实测批处理时间为准，逐例时间为分摊值，不能当作在线单请求延迟。历史 A2 复用缓存，不能与本轮语义版做公平的全流程运行时间比较。
