# ASTER 参考实现与运行入口

[论文主稿](../../docs/ASTER/manuscript.zh.md) · [接口与实现约定](../../docs/ASTER/supplementary.zh.md) · [实验协议](../../docs/ASTER/experiments.zh.md)

## 实现范围与编号对应

论文已调整为“模块一学习方向、模块二判断适用性、模块三执行编辑”。这次整理不改变运行代码、公共类名、输出字段或已有脚本名称。

| 论文模块 | 当前范围 |
|---|---|
| 一：对比适用性子空间学习 | 方法已定义；方向估计与专家构建尚未实现 |
| 二：处置盲适用性评估（ABAA） | 已有输入构建、受限分类和单温度拟合参考实现 |
| 三：适用性门控的规则键编辑 | 已有 `rule_gains` 映射；实际键编辑、缓存协议和完整树运行器尚未实现 |

`run_aster_module1` 的 module1 是历史编号，现执行论文模块二。`aster.action_blind.v1` 格式保持不变。旧 `src/module1`、`src/module2`、`src/module3` 和 `src/pipeline` 属于历史实现，不因文档重排而变成新版系统。没有新增预训练模型运行或性能结果。

## 复用同一骨干

```python
from src.aster import ActionBlindBranchClassifier, HFLabelScorer, rule_gains

# 参数由非测试资料确定；分类必须先于任何键编辑。
model.eval()
classifier = ActionBlindBranchClassifier(
    HFLabelScorer(model, tokenizer),
    temperature=temperature,
    probability_threshold=probability_threshold,
    margin_threshold=margin_threshold,
)
result = classifier.run(guideline, patient, question, options)
gains = rule_gains(
    result["groups"][0],
    g_plus=g_plus,
    g_minus=g_minus,
    eligible=eligible,  # 调用者核验输入与祖先路径；函数不遍历树。
)
```

`HFLabelScorer` 复用已加载模型，不下载或加载第二个骨干。真实评分需要 PyTorch、Transformers 与已准备好的模型；无模型契约测试不证明所有聊天模板已经适配。`gains` 只是计划增益，不是编辑后的键或答案。

## 输入与输出

参见[输入示例](../../examples/aster_action_blind_input.json)。每组要求 `group_id`、`mutually_exclusive:true`、`ancestor_conditions` 和含 `rule_id`／`condition` 的 `rules`。互斥声明由数据提供者核验。分类器默认排除 `action`、原始 `question` 和 `options`；必要的非处置语境通过 `task_context` 与 `task_context_verified=True` 显式提供。既往治疗和时间事实保留。

每组支持 1–25 条实际规则，另加 NONE。过长输入、非法分数、标签边界不合格等直接报错，不静默截断或伪造 NONE。输出包含完整概率、`prediction`、`probability`、`margin`、`entropy`、`abstain`、`intervention_allowed` 和 `prompt_sha256`；`include_logits=True` 额外保存分数。

`prediction` 始终保留最高类别；并列总是弃权。`abstain=True` 对应论文 $a_g=1$，`eligible` 对应独立的 $e_g$。高置信 NONE 也不编辑。`intervention_allowed` 仍受可达性约束，不能直接当作实际编辑日志。提示文本哈希不替代模型、模板和词元指纹。

## 温度拟合

```python
from src.aster import fit_temperature

# 使用独立金标，不使用分类器自身 prediction 充当 label。
fit = fit_temperature(calibration_records, split="calibration")
temperature = fit["temperature"]
```

记录格式示例为 `{"logits":{"R1":2.0,"R2":0.5,"NONE":-1.0},"label":"R1"}`，仅演示格式。可混合不同组大小，只拟合一个温度。默认搜索范围 `[0.05,20]` 是优化范围而非已校准配置。函数接受 `development`、`calibration` 并拒绝 `test`，但名称声明不替代真实隔离。弃权阈值另由开发评价确定，当前无自动阈值调优器。

## 命令与检查

```bash
python -m scripts.run_aster_module1 \
  --input examples/aster_action_blind_input.json \
  --output /tmp/aster_module2_output.json \
  --model "$ASTER_BACKBONE_PATH" \
  --device cuda:0 \
  --temperature "$ASTER_TEMPERATURE" \
  --probability-threshold "$ASTER_PROBABILITY_THRESHOLD" \
  --margin-threshold "$ASTER_MARGIN_THRESHOLD"
```

环境变量须先设置为真实本地模型路径与开发阶段冻结参数。此命令只执行 ABAA，不学习方向、不修改键、不生成最终处置。

```bash
python scripts/check_aster_markdown_math.py
python -m unittest discover -s tests -p 'test_aster*.py' -v
python -m scripts.run_aster_module1 --help
```

契约测试使用可控分数与分词替身，覆盖输入遮蔽、病史保留、标签、概率、弃权、校准和增益错误输入。源码检查不等于性能、实际缓存编辑或临床有效性验证。
