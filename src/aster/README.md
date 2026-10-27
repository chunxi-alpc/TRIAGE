# ASTER 当前模块一参考实现

[项目首页](../../README.md) · [模块一确定版](../../docs/ASTER/module1-spec.zh.md) · [实现范围](../../docs/ASTER/implementation-status.zh.md)

本目录实现 **处置盲分支分类、单温度拟合、模块一到模块三的规则增益映射**。它独立于历史 CARE／V6 接口，不生成处置，不抽取患者原子，不加载外部医学核查器。独立模块二与缓存编辑实现见[模块二说明](../module2/README.md)和[真实预实验](../../experiments/module2_applicability_20261027/README.md)；本目录的分类器尚未与其连接成完整流水线。

## 复用同一骨干

```python
from src.aster import ActionBlindBranchClassifier, HFLabelScorer, rule_gains

# model / tokenizer 为执行阶段使用的同一已加载对象。
model.eval()
classifier = ActionBlindBranchClassifier(
    HFLabelScorer(model, tokenizer),
    temperature=temperature,                 # 开发资料拟合并冻结
    probability_threshold=probability_threshold,
    margin_threshold=margin_threshold,
)
result = classifier.run(guideline, patient, question, options)
gains = rule_gains(
    result["groups"][0],
    g_plus=g_plus,
    g_minus=g_minus,
    eligible=True,  # 树结构必须由调用者先核验祖先可达性
)
```

应在安装任何模块三干预前运行分类器。`HFLabelScorer` 不下载、不加载第二个模型；直接用同一对象读取合法标签 logits。真实前向需要 PyTorch、Transformers 及已经准备好的模型权重。当前无模型契约测试不证明所有模型模板已完成适配。

## 输入与默认遮蔽

使用[示例 JSON](../../examples/aster_action_blind_input.json)。`guideline.groups` 的每组包含 `group_id`、`mutually_exclusive:true`、`ancestor_conditions` 和 `rules`。规则需要 `rule_id`、`condition`；可以附带 `action`，但分类器不读取它。

原始 `question` 和 `options` 被整个排除。必要的非处置语境只能由独立 `task_context` 加 `task_context_verified=True` 显式提供。患者既往用药等事实完整保留。模型只看到中性代理标签；原始编号在程序中映回。

`mutually_exclusive` 是数据提供者声明，不是自然语言互斥性证明。输入切分、条件或患者中的推荐泄漏、树依赖仍须数据质控。超过 25 个规则、多词元代理、边界变化、过长上下文、非法分数都会报错；没有静默截断或伪造 NONE。

## 输出与弃权

返回 `aster.action_blind.v1`，包含 `groups` 与策略元数据。每组有完整概率、`prediction`、`probability`、`margin`、自然对数 `entropy`、`abstain`、`intervention_allowed`、`prompt_sha256`。`include_logits=True` 额外保存原始分数。

`prediction` 永远保留最大概率标签。NONE 和弃权独立；高置信 NONE 也关闭干预。精确并列总是弃权。`rule_gains` 对通过策略且可达的组产生选中正增益、兄弟负增益；其他组全零。它不编辑注意力，也不自动处理子树。

## 校准一个全局温度

```python
from src.aster import fit_temperature

# 每条记录含全部合法规则 logits 与独立金标。
# 来自非测试树；不能用 M1 自己的 prediction 作为 label。
fit = fit_temperature(calibration_records, split="calibration")
temperature = fit["temperature"]
```

记录格式为 `{"logits":{"R1":2.0,"R2":0.5,"NONE":-1.0},"label":"R1"}`。这些数字只演示格式，不是实验结果。可以混合不同分支数的组，但仅拟合一个温度。返回拟合前后 NLL、样本数、搜索边界及边界解标记；默认搜索范围 `[0.05,20]` 仅为优化范围，应在开发阶段核验。

函数接受 `development` 或 `calibration`，拒绝 `test`；该名称声明不替代真实数据隔离。阈值选择仍由独立开发评价完成，当前不提供自动阈值调优器。正温度不改变 argmax，不提供覆盖或伤害认证。

## 命令行

从仓库根目录运行；环境变量应先设为实际本地模型路径及开发集冻结参数：

```bash
python -m scripts.run_aster_module1 \
  --input examples/aster_action_blind_input.json \
  --output /tmp/aster_module1_output.json \
  --model "$ASTER_BACKBONE_PATH" \
  --device cuda:0 \
  --temperature "$ASTER_TEMPERATURE" \
  --probability-threshold "$ASTER_PROBABILITY_THRESHOLD" \
  --margin-threshold "$ASTER_MARGIN_THRESHOLD"
```

该独立命令只读取本地或已有缓存的模型。完整系统应采用前面的对象注入方式，避免重复加载。它没有运行模块二、修改注意力或生成最终处置。

## 无模型契约检查

```bash
python -m unittest discover -s tests -p 'test_aster*.py' -v
python -m scripts.run_aster_module1 --help
```

检查覆盖处置交换、原问题／选项排除、病史保留、祖先条件、标签边界、概率与弃权、校准和增益错误输入。这些检查使用可控分数及分词替身，不下载或运行预训练模型，也不说明临床性能已经验证。
