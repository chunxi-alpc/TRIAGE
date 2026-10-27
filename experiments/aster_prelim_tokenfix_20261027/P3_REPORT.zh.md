# P3 分层干预实测

状态：**COMPLETED**。只用 169 个 SW 开发病例。

| 模型 | 方法 | 准确率 % | 相对明判变化（百分点） | 差距恢复比例 |
|---|---|---:|---:|---:|
| qwen | blind | 86.98 | +1.78 | 1.000 |
| qwen | visible | 85.21 | +0.00 | 0.000 |
| qwen | neutral | 92.90 | +7.69 | 4.333 |
| qwen | patch_K | 92.90 | +7.69 | 4.333 |
| qwen | patch_V | 92.90 | +7.69 | 4.333 |
| qwen | patch_KV | 92.90 | +7.69 | 4.333 |
| qwen | knock_1_4 | 85.21 | +0.00 | 0.000 |
| qwen | knock_2_5 | 85.21 | +0.00 | 0.000 |
| qwen | knock_3_6 | 85.21 | +0.00 | 0.000 |
| qwen | knock_4_7 | 85.21 | +0.00 | 0.000 |
| qwen | knock_5_8 | 85.21 | +0.00 | 0.000 |
| qwen | knock_6_9 | 85.21 | +0.00 | 0.000 |
| qwen | knock_7_10 | 85.21 | +0.00 | 0.000 |
| qwen | knock_8_11 | 85.21 | +0.00 | 0.000 |
| qwen | knock_9_12 | 85.21 | +0.00 | 0.000 |
| qwen | knock_10_13 | 85.21 | +0.00 | 0.000 |
| qwen | knock_11_14 | 85.21 | +0.00 | 0.000 |
| qwen | knock_12_15 | 85.21 | +0.00 | 0.000 |
| qwen | knock_13_16 | 85.21 | +0.00 | 0.000 |
| qwen | knock_14_17 | 85.21 | +0.00 | 0.000 |
| qwen | knock_15_18 | 85.21 | +0.00 | 0.000 |
| qwen | knock_16_19 | 85.21 | +0.00 | 0.000 |
| qwen | knock_17_20 | 85.21 | +0.00 | 0.000 |
| qwen | knock_18_21 | 85.21 | +0.00 | 0.000 |
| qwen | knock_19_22 | 85.21 | +0.00 | 0.000 |
| qwen | knock_20_23 | 85.21 | +0.00 | 0.000 |
| qwen | knock_21_24 | 85.21 | +0.00 | 0.000 |
| qwen | knock_22_25 | 85.21 | +0.00 | 0.000 |
| qwen | knock_23_26 | 85.21 | +0.00 | 0.000 |
| qwen | knock_24_27 | 85.21 | +0.00 | 0.000 |
| qwen | knock_25_28 | 84.62 | -0.59 | -0.333 |
| qwen | knock_26_29 | 85.21 | +0.00 | 0.000 |
| qwen | knock_27_30 | 85.21 | +0.00 | 0.000 |
| qwen | knock_28_31 | 85.21 | +0.00 | 0.000 |
| qwen | knock_29_32 | 85.21 | +0.00 | 0.000 |
| qwen | knock_30_33 | 85.21 | +0.00 | 0.000 |
| qwen | knock_31_34 | 85.21 | +0.00 | 0.000 |
| qwen | knock_32_35 | 85.21 | +0.00 | 0.000 |
| qwen | knock_33_36 | 85.21 | +0.00 | 0.000 |
| qwen | knock_all | 84.62 | -0.59 | -0.333 |
| llama | blind | 69.82 | +16.57 | 1.000 |
| llama | visible | 53.25 | +0.00 | 0.000 |
| llama | neutral | 60.95 | +7.69 | 0.464 |
| llama | patch_K | 55.62 | +2.37 | 0.143 |
| llama | patch_V | 60.95 | +7.69 | 0.464 |
| llama | patch_KV | 60.95 | +7.69 | 0.464 |
| llama | knock_1_4 | 53.25 | +0.00 | 0.000 |
| llama | knock_2_5 | 53.25 | +0.00 | 0.000 |
| llama | knock_3_6 | 53.25 | +0.00 | 0.000 |
| llama | knock_4_7 | 53.25 | +0.00 | 0.000 |
| llama | knock_5_8 | 53.85 | +0.59 | 0.036 |
| llama | knock_6_9 | 53.25 | +0.00 | 0.000 |
| llama | knock_7_10 | 53.25 | +0.00 | 0.000 |
| llama | knock_8_11 | 53.25 | +0.00 | 0.000 |
| llama | knock_9_12 | 53.25 | +0.00 | 0.000 |
| llama | knock_10_13 | 53.25 | +0.00 | 0.000 |
| llama | knock_11_14 | 53.25 | +0.00 | 0.000 |
| llama | knock_12_15 | 53.25 | +0.00 | 0.000 |
| llama | knock_13_16 | 53.25 | +0.00 | 0.000 |
| llama | knock_14_17 | 53.25 | +0.00 | 0.000 |
| llama | knock_15_18 | 53.25 | +0.00 | 0.000 |
| llama | knock_16_19 | 53.25 | +0.00 | 0.000 |
| llama | knock_17_20 | 53.25 | +0.00 | 0.000 |
| llama | knock_18_21 | 53.25 | +0.00 | 0.000 |
| llama | knock_19_22 | 53.25 | +0.00 | 0.000 |
| llama | knock_20_23 | 53.25 | +0.00 | 0.000 |
| llama | knock_21_24 | 53.25 | +0.00 | 0.000 |
| llama | knock_22_25 | 53.25 | +0.00 | 0.000 |
| llama | knock_23_26 | 53.25 | +0.00 | 0.000 |
| llama | knock_24_27 | 53.25 | +0.00 | 0.000 |
| llama | knock_25_28 | 53.25 | +0.00 | 0.000 |
| llama | knock_26_29 | 53.25 | +0.00 | 0.000 |
| llama | knock_27_30 | 53.25 | +0.00 | 0.000 |
| llama | knock_28_31 | 53.25 | +0.00 | 0.000 |
| llama | knock_29_32 | 53.25 | +0.00 | 0.000 |
| llama | knock_all | 53.85 | +0.59 | 0.036 |

窗口选择：`{"qwen": {"best_four_layer_window": "knock_1_4", "best_accuracy_pct": 85.20710059171597, "baseline_gap_pp": 1.7751479289940875, "best_recovery_fraction": 0.0, "selected_layers": "all", "selection_scope": "exploratory development selection; threshold heuristic is not proof of localization"}, "llama": {"best_four_layer_window": "knock_5_8", "best_accuracy_pct": 53.84615384615385, "baseline_gap_pp": 16.568047337278117, "best_recovery_fraction": 0.03571428571428593, "selected_layers": "all", "selection_scope": "exploratory development selection; threshold heuristic is not proof of localization"}}`。

解释范围：

- 敲除仅切断最后一个判断位置对两条终端处置的注意力；选项与其他位置仍可传递信息。它检验直接读取路径，不声称消除所有间接路径。
- K/V 从中性代号运行修补到熟悉处置运行；修补诱饵段及必要的分词边界词元，段外 token ID 完全一致。修补发生在 k_proj/v_proj 输出，Qwen 的 K 归一化与 RoPE 继续正常执行。
- P3 使用 1/2 的下一词元概率读出，另行报告与 P1 自由生成的一致率。恢复比例用 P3 自己的盲判／明判差距作分母，不混用分母。
- 线性探针逐层在 3 个开发树训练、剩余开发树评估；全程未读取测试树。探针可解码不等于因果证据。
- 旧版的词元对齐和布尔掩码失败记录保留于原实验目录；本轮从头运行，不复用旧结果。
- 不能找到足够恢复的窗口时，按用户规则改用全层；不把预期的第 13–22 层或 K/V 恢复比例填入结果。
