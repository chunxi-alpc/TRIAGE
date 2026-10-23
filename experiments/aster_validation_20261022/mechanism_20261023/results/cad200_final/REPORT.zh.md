# Qwen同批200题：CAD与机制实验联合汇总

更新时间：2026-10-23T19:57:02.790049+08:00。固定200题，K76/U124，40开发题不计入。

CAD候选过滤版及三个顺序均已完整结束。

|方法|A|B|C|D|
|---|---:|---:|---:|---:|
|existing_baseline|65.5%|41.5%|67.0%|21.0%|
|existing_signed|75.0%|48.5%|74.5%|27.5%|
|visible_prune|80.0%|49.0%|82.0%|47.5%|
|hidden_prune|77.5%|57.5%|77.5%|64.0%|
|hidden_top2|82.5%|53.5%|85.0%|55.5%|
|existing_prune|99.0%|83.0%|99.0%|89.0%|
|CAD8|61.0%|52.0%|64.0%|未运行|
|CAD128|61.0%|52.0%|64.0%|未运行|
|CAD_filtered_original|65.0%|45.0%|65.5%|未运行|
|CAD_filtered_left|63.5%|42.5%|61.0%|未运行|
|CAD_filtered_right|60.5%|44.0%|63.0%|未运行|

## CAD解析失败与真实选择错误
|版本/条件|正确|解析失败|有效但选错|
|---|---:|---:|---:|
|CAD8/cad/A/False|122|1|77|
|CAD8/prune_cad/A/False|196|0|4|
|CAD8/prune_cad/A/True|9|0|191|
|CAD8/cad/B/False|104|1|95|
|CAD8/prune_cad/B/False|184|0|16|
|CAD8/prune_cad/B/True|7|0|193|
|CAD8/cad/C/False|128|1|71|
|CAD8/prune_cad/C/False|196|0|4|
|CAD8/prune_cad/C/True|13|0|187|
|CAD128/cad/A/False|122|1|77|
|CAD128/prune_cad/A/False|196|0|4|
|CAD128/prune_cad/A/True|9|0|191|
|CAD128/cad/B/False|104|1|95|
|CAD128/prune_cad/B/False|184|0|16|
|CAD128/prune_cad/B/True|7|0|193|
|CAD128/cad/C/False|128|1|71|
|CAD128/prune_cad/C/False|196|0|4|
|CAD128/prune_cad/C/True|13|0|187|
|CAD_filtered_original/cad/A/False|130|0|70|
|CAD_filtered_original/prune_cad/A/False|196|0|4|
|CAD_filtered_original/prune_cad/A/True|14|0|186|
|CAD_filtered_original/cad/B/False|90|1|109|
|CAD_filtered_original/prune_cad/B/False|174|0|26|
|CAD_filtered_original/prune_cad/B/True|6|0|194|
|CAD_filtered_original/cad/C/False|131|0|69|
|CAD_filtered_original/prune_cad/C/False|196|0|4|
|CAD_filtered_original/prune_cad/C/True|21|0|179|
|CAD_filtered_left/cad/A/False|127|0|73|
|CAD_filtered_left/prune_cad/A/False|198|0|2|
|CAD_filtered_left/prune_cad/A/True|12|0|188|
|CAD_filtered_left/cad/B/False|85|0|115|
|CAD_filtered_left/prune_cad/B/False|180|0|20|
|CAD_filtered_left/prune_cad/B/True|0|0|200|
|CAD_filtered_left/cad/C/False|122|0|78|
|CAD_filtered_left/prune_cad/C/False|198|0|2|
|CAD_filtered_left/prune_cad/C/True|14|0|186|
|CAD_filtered_right/cad/A/False|121|0|79|
|CAD_filtered_right/prune_cad/A/False|199|0|1|
|CAD_filtered_right/prune_cad/A/True|13|0|187|
|CAD_filtered_right/cad/B/False|88|0|112|
|CAD_filtered_right/prune_cad/B/False|188|0|12|
|CAD_filtered_right/prune_cad/B/True|3|0|197|
|CAD_filtered_right/cad/C/False|126|0|74|
|CAD_filtered_right/prune_cad/C/False|199|0|1|
|CAD_filtered_right/prune_cad/C/True|10|0|190|

每行分母均为200；正确、解析失败、有效但选错互斥且合计200。CAD无D/CS输出，不能填入其分数或计算相同BCD/BCDCS调和平均。左右重排与原顺序分开列出，不混合报告。
机制结论：分支与动作错误并存；K组动作诱导有害翻转尚未达到校正后的预设注意力门槛，未执行阻断。隐藏动作裁剪在D上优于可见预测裁剪，但A/C非全面优势，仍低于金标。CS2例仅作描述。
完整机制结果与逐题统计见../priority200/qwen/full_parser_v2。CAD逐题统计见cad_scores.jsonl；同批过滤前后配对检验和疾病簇区间见summary.json。Llama保持保存暂停。
