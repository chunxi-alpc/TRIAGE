# TrialMatchAI 失败条件定向重生成与复评

生成状态：completed；只重跑原先失败的 69/492 条条件（截断 29、目标不明确 40），原 423 条成功输出逐字保留。模型调用只使用本地 Phi-4 + TrialMatchAI LoRA；Qwen3 未调用。生成提示版本 `retry_exact_single_condition_v1`，上限 4096 token，批大小 8，使用 4 张 A100。

## 解析结果

| 指标 | 重跑前 | 重跑后 |
|---|---:|---:|
| 成功解析 | 423/492 (85.98%) | 492/492 (100.00%) |
| 真正截断 | 29/492 (5.89%) | 0/492 (0.00%) |
| 目标不明确 | 40/492 (8.13%) | 0/492 (0.00%) |

69 条重跑中严格解析成功 69/69；没有放宽原文/标签核验。重跑后全体解析状态：`{"accepted": 492}`。

## 条件判断与规则适用性

| 指标 | 重跑前 | 重跑后 |
|---|---:|---:|
| 原子条件准确率 | 381/492 (77.44%) | 431/492 (87.60%) |
| 原子条件宏 F1 | 0.7293 | 0.8308 |

| 规则级方法 | 重跑前准确率 / 宏 F1 | 重跑后准确率 / 宏 F1 |
|---|---:|---:|
| TrialMatchAI 单独 | 60/192 (31.25%) / 0.2251 | 150/192 (78.12%) / 0.7709 |
| TrialMatchAI + 数值比较 | 63/192 (32.81%) / 0.2461 | 122/192 (63.54%) / 0.6370 |
| TrialMatchAI + BioLinkBERT（缓存证据） | 104/192 (54.17%) / 0.5254 | 96/192 (50.00%) / 0.4902 |
| V4 已保存 Qwen3 对照 | 136/192 (70.83%) / 0.7041 | 136/192 (70.83%) / 0.7041 |

新 TrialMatchAI 准确率相对历史 Qwen3 为 +7.29 个百分点。加入 BioLinkBERT 后相对单独 TrialMatchAI -28.12 个百分点；证据定位 76/137 (55.47%)，判定和证据同时正确 48/192。证据关系沿用同一批已缓存 MedNLI 输出，本次没有重跑 BioLinkBERT，因此不能把证据指标变化归因于重生成。

本次 4 卡并行墙钟 0.9 分钟；69 条输出分摊生成 GPU 工作 1.2 GPU 分钟。单卡峰值显存：GPU 4: 13.2 GiB allocated / 13.3 GiB reserved, GPU 5: 13.2 GiB allocated / 13.3 GiB reserved, GPU 6: 13.2 GiB allocated / 13.3 GiB reserved, GPU 7: 13.2 GiB allocated / 13.3 GiB reserved.

## 重跑后仍未严格解析的条件

无。

## 产物

- 逐条件解析及冻结记录：`experiments/care_module1_v5/runs/v5_20261018/parser_v3_retry/conditions_public.jsonl`、`experiments/care_module1_v5/runs/v5_20261018/parser_v3_retry/REPARSE_FREEZE.json`。
- 原子条件评价和错误：`experiments/care_module1_v5/runs/v5_20261018/parser_v3_retry/condition_eval/condition_metrics.json`、`experiments/care_module1_v5/runs/v5_20261018/parser_v3_retry/condition_eval/condition_predictions_public.jsonl`。
- 规则级指标和逐例错误：`experiments/care_module1_v5/runs/v5_20261018/parser_v3_retry/metrics.json`、`experiments/care_module1_v5/runs/v5_20261018/parser_v3_retry/errors_public.jsonl`。
- 私有重跑原文与合并输出：`experiments/care_module1_v5/runs/v5_retry_20261018_final/private/`；不包含在公开预测文件中。

192 条数据已用于方法开发，属于同一集合的探索性回归，不是独立验证。
