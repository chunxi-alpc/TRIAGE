# CounterFact 可选方法列表

2026-10-26。此次仅扩展`counterfact_entity_ci.py`的实际方法选择，原bootstrap、配对和文件哈希核验不变。此前`entity_ci_v2`已在服务器通过；本次参数扩展及3项新测试待服务器运行，旧结果不改写。

不传`--methods`时仍保持原四方法：seka、matched_zero、unsteered、pasta，旧命令完全兼容。

Qwen3-8B-Base当前只有三种真实结果时，用：

```bash
python "$AUDIT/counterfact_entity_ci.py" --data "$COUNTERFACT_JSONL" --methods seka matched_zero unsteered --seka "$BASE_SEKA_EXPORT" --matched-zero "$BASE_ZERO_EXPORT" --unsteered "$BASE_UNSTEERED_EXPORT" --output-dir "$NEW_BASE_ENTITY_CI_DIR"
```

只计算SEKA对两种已完成对照；PASTA、CAD、Spotlight在报告中列为planned，不填不存在的结果。不允许选择了方法却缺少结果路径，也不默默忽略显式传入但未选择的方法结果。所有实际分数和负结果保留；程序核对所有所选方法的非空模型文件指纹完全一致，防止混合Base和非Base。

```bash
python -m unittest discover -s "$AUDIT" -p 'test_counterfact_method_selection.py'
```

复制修改版到新的运行版本目录，避免热替换正在执行进程已导入的统计模块。新报告记录新的源码SHA，既有运行/结果保持原样。
