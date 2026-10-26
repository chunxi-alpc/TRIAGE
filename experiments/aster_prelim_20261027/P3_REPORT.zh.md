# P3 分层干预实测

状态：**COMPLETED**。只用 200 个 SW 开发病例。

| 模型 | 方法 | 准确率 % | 相对明判变化（百分点） | 差距恢复比例 |
|---|---|---:|---:|---:|
| qwen | blind | 81.00 | +11.50 | 1.000 |
| qwen | visible | 69.50 | +0.00 | 0.000 |
| qwen | neutral | 93.50 | +24.00 | 2.087 |
| qwen | patch_K | 86.00 | +16.50 | 1.435 |
| qwen | patch_V | 94.00 | +24.50 | 2.130 |
| qwen | patch_KV | 93.50 | +24.00 | 2.087 |
| qwen | knock_1_4 | 69.50 | +0.00 | 0.000 |
| qwen | knock_2_5 | 69.50 | +0.00 | 0.000 |
| qwen | knock_3_6 | 69.50 | +0.00 | 0.000 |
| qwen | knock_4_7 | 69.50 | +0.00 | 0.000 |
| qwen | knock_5_8 | 69.50 | +0.00 | 0.000 |
| qwen | knock_6_9 | 69.50 | +0.00 | 0.000 |
| qwen | knock_7_10 | 69.50 | +0.00 | 0.000 |
| qwen | knock_8_11 | 69.50 | +0.00 | 0.000 |
| qwen | knock_9_12 | 69.50 | +0.00 | 0.000 |
| qwen | knock_10_13 | 69.50 | +0.00 | 0.000 |
| qwen | knock_11_14 | 69.50 | +0.00 | 0.000 |
| qwen | knock_12_15 | 69.50 | +0.00 | 0.000 |
| qwen | knock_13_16 | 69.50 | +0.00 | 0.000 |
| qwen | knock_14_17 | 69.50 | +0.00 | 0.000 |
| qwen | knock_15_18 | 69.50 | +0.00 | 0.000 |
| qwen | knock_16_19 | 69.50 | +0.00 | 0.000 |
| qwen | knock_17_20 | 69.50 | +0.00 | 0.000 |
| qwen | knock_18_21 | 69.50 | +0.00 | 0.000 |
| qwen | knock_19_22 | 69.50 | +0.00 | 0.000 |
| qwen | knock_20_23 | 69.50 | +0.00 | 0.000 |
| qwen | knock_21_24 | 69.50 | +0.00 | 0.000 |
| qwen | knock_22_25 | 69.50 | +0.00 | 0.000 |
| qwen | knock_23_26 | 69.50 | +0.00 | 0.000 |
| qwen | knock_24_27 | 69.50 | +0.00 | 0.000 |
| qwen | knock_25_28 | 69.50 | +0.00 | 0.000 |
| qwen | knock_26_29 | 69.50 | +0.00 | 0.000 |
| qwen | knock_27_30 | 69.50 | +0.00 | 0.000 |
| qwen | knock_28_31 | 69.50 | +0.00 | 0.000 |
| qwen | knock_29_32 | 69.50 | +0.00 | 0.000 |
| qwen | knock_30_33 | 69.50 | +0.00 | 0.000 |
| qwen | knock_31_34 | 69.50 | +0.00 | 0.000 |
| qwen | knock_32_35 | 69.50 | +0.00 | 0.000 |
| qwen | knock_33_36 | 69.50 | +0.00 | 0.000 |
| qwen | knock_all | 69.50 | +0.00 | 0.000 |
| llama | blind | 64.50 | +12.00 | 1.000 |
| llama | visible | 52.50 | +0.00 | 0.000 |
| llama | neutral | 64.50 | +12.00 | 1.000 |
| llama | patch_K | 57.00 | +4.50 | 0.375 |
| llama | patch_V | 67.50 | +15.00 | 1.250 |
| llama | patch_KV | 64.50 | +12.00 | 1.000 |
| llama | knock_1_4 | 52.50 | +0.00 | 0.000 |
| llama | knock_2_5 | 52.50 | +0.00 | 0.000 |
| llama | knock_3_6 | 52.50 | +0.00 | 0.000 |
| llama | knock_4_7 | 52.50 | +0.00 | 0.000 |
| llama | knock_5_8 | 52.50 | +0.00 | 0.000 |
| llama | knock_6_9 | 52.50 | +0.00 | 0.000 |
| llama | knock_7_10 | 52.50 | +0.00 | 0.000 |
| llama | knock_8_11 | 52.50 | +0.00 | 0.000 |
| llama | knock_9_12 | 52.50 | +0.00 | 0.000 |
| llama | knock_10_13 | 52.50 | +0.00 | 0.000 |
| llama | knock_11_14 | 52.50 | +0.00 | 0.000 |
| llama | knock_12_15 | 52.50 | +0.00 | 0.000 |
| llama | knock_13_16 | 52.50 | +0.00 | 0.000 |
| llama | knock_14_17 | 52.50 | +0.00 | 0.000 |
| llama | knock_15_18 | 52.50 | +0.00 | 0.000 |
| llama | knock_16_19 | 52.50 | +0.00 | 0.000 |
| llama | knock_17_20 | 52.50 | +0.00 | 0.000 |
| llama | knock_18_21 | 52.50 | +0.00 | 0.000 |
| llama | knock_19_22 | 52.50 | +0.00 | 0.000 |
| llama | knock_20_23 | 52.50 | +0.00 | 0.000 |
| llama | knock_21_24 | 52.50 | +0.00 | 0.000 |
| llama | knock_22_25 | 52.50 | +0.00 | 0.000 |
| llama | knock_23_26 | 52.50 | +0.00 | 0.000 |
| llama | knock_24_27 | 52.50 | +0.00 | 0.000 |
| llama | knock_25_28 | 52.50 | +0.00 | 0.000 |
| llama | knock_26_29 | 52.50 | +0.00 | 0.000 |
| llama | knock_27_30 | 52.50 | +0.00 | 0.000 |
| llama | knock_28_31 | 52.50 | +0.00 | 0.000 |
| llama | knock_29_32 | 52.50 | +0.00 | 0.000 |
| llama | knock_all | 52.50 | +0.00 | 0.000 |

窗口选择：`{"qwen": {"best_four_layer_window": "knock_1_4", "best_accuracy_pct": 69.5, "baseline_gap_pp": 11.5, "best_recovery_fraction": 0.0, "selected_layers": "all", "selection_scope": "exploratory development selection; threshold heuristic is not proof of localization"}, "llama": {"best_four_layer_window": "knock_1_4", "best_accuracy_pct": 52.5, "baseline_gap_pp": 12.0, "best_recovery_fraction": 0.0, "selected_layers": "all", "selection_scope": "exploratory development selection; threshold heuristic is not proof of localization"}}`。

解释范围：

- 敲除仅切断最后一个判断位置对两条终端处置的注意力；选项与其他位置仍可传递信息。它检验直接读取路径，不声称消除所有间接路径。
- K/V 从中性代号运行修补到熟悉处置运行；修补诱饵段及必要的分词边界词元，段外 token ID 完全一致。修补发生在 k_proj/v_proj 输出，Qwen 的 K 归一化与 RoPE 继续正常执行。
- P3 使用 1/2 的下一词元概率读出，另行报告与 P1 自由生成的一致率。恢复比例用 P3 自己的盲判／明判差距作分母，不混用分母。
- 线性探针逐层在 3 个开发树训练、剩余开发树评估；全程未读取测试树。探针可解码不等于因果证据。
- 第一次启动在词元对齐断言处停止，没有形成模型结果；失败日志保留。修正后重新从头运行。
- 不能找到足够恢复的窗口时，按用户规则改用全层；不把预期的第 13–22 层或 K/V 恢复比例填入结果。
