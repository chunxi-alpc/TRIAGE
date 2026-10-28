# ASTER 参考实现与运行入口

[论文主稿](../../docs/ASTER/manuscript.zh.md) · [接口与实现约定](../../docs/ASTER/supplementary.zh.md) · [实验协议](../../docs/ASTER/experiments.zh.md)

## 实现范围与编号对应

论文已调整为“模块一学习方向、模块二判断适用性、模块三执行编辑”。这次整理不改变运行代码、公共类名、输出字段或已有脚本名称。

| 论文模块 | 当前范围 |
|---|---|
| 一：对比适用性子空间学习 | 已有逐 token 配对二阶矩、归一化方向、双状态选头与共享专家的独立实现；固定小模型机制预实验完成 |
| 二：处置盲适用性评估（ABAA） | 已有输入构建、受限分类和单温度拟合参考实现 |
| 三：适用性门控的规则键编辑 | 已有 `rule_gains` 映射及独立 Qwen3 指南前缀缓存键编辑器；完整树运行器及实际模块二到生成的联合验证待完成 |

`run_aster_module1` 的 module1 是历史编号，现执行论文模块二。`aster.action_blind.v1` 格式保持不变。新增 `src/module2/mechanism*` 与 `run_module2_mechanism` 保留冻结时的旧编号，现执行论文模块一。`src/module3/guideline_prefix.py` 是当前指南缓存协议的独立执行器；其余旧定位、投影和 `src/pipeline` 保留历史接口，不因文档重排而变成新版联合系统。

## 独立机制实现与已完成运行

| 入口 | 实际范围 |
|---|---|
| [mechanism.py](../module2/mechanism.py) | 逐 token 配对二阶矩、方向、分离度、双状态选头和成组符号诊断 |
| [mechanism_extraction.py](../module2/mechanism_extraction.py) | 患者前置离线键与指南前置在线查询；实际 GQA／RoPE 坐标 |
| [guideline_prefix.py](../module3/guideline_prefix.py) | 独立指南缓存、多头双规则键平移、整个患者后缀和经过核验的后段优化 |
| [运行与分析入口](../../experiments/module2_mechanism_20261028/README.md) | 固定模型／数据／算子、完整配置覆盖、三图、所有案例与可复算选例 |

[机制预实验](../../experiments/module2_mechanism_20261028/RESULTS.zh.md)已完成 Qwen3-0.6B-Base 的 144 次离线和 48 次在线表示前向、216 条实际干预以及同条件的 216 条答案边界修正重放。12/12 个配对配置均改善局部读取比，原始均值和直接偏置同样有效；规范 A/B 读出所有配置均为 12/24 正确、1/12 配对全对且无预测翻转。研究证据限于 3 个确认指南、显式 Boolean 患者和金标编辑符号，实际处置盲预测、自由生成与完整系统尚未验证。

统计与执行器、最小对、边界读出的检查分三个实际批次完成，分别为 31、10、7 项。24 个零编辑和 72 个代表非零配置的完整／后段比较，logits、attention、query 最大误差均为 0；24 个完整提示／分段基线的全词表最大误差为 3.37e−5，低于 1e−4。研究运行使用固定预训练权重，微型随机模型仅用于代码检查。详细覆盖与摘要见 [RUN_VALIDATION.json](../../experiments/module2_mechanism_20261028/RUN_VALIDATION.json)。

[后续答案读出诊断](../../experiments/rule_readout_diagnostic_20261028/RESULTS.zh.md)完成相同 24 提示上的八条件、192 条记录。固定键增益 +32 与直接分数偏置 +8 分别为 15/24 和 14/24 正确，但正向减反向的平均正确标签间隔均为负；显式正确／兄弟 ID 注释分别为 24/24 和 0/24。该探索显示标签可以改变，同时限制了从局部读取到答案效用的推断；默认参数、头集及原冻结结果未改。

新增运行器 [run_rule_readout_diagnostic.py](../../scripts/run_rule_readout_diagnostic.py)与[分析脚本](../../scripts/analyze_rule_readout_diagnostic.py)保留完整分母、正反方向、候选概率及实际读取质量。8 项新代码检查通过；48 个首轮条件重放的最大 logit／attention 误差分别为 5.72e−6／1.38e−7，在预设容差内；24 个完整／后段比较误差均为 0。运行状态见[诊断核验记录](../../experiments/rule_readout_diagnostic_20261028/results/status.json)，这些检查不扩充独立指南数。

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
