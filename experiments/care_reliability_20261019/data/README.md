# 新合成机制测试集

本目录提供新编的可执行逻辑机制集。**它不是临床数据集，不是新的临床指南，也未经临床专家复核。** 年龄、肿物径和复发间隔仅作为模拟字段。所有决策都是虚构的 Cedar、Maple、Juniper 路由或请求复核，不涉及治疗推荐。

本集在模型前向执行前冻结，使用新编的可执行合成规则。其 schema 和病例措辞曾用于开发阶段的回归检查与修复，包括亲属主体识别。因此，本集不是对全部方法开发过程保持盲法的独立确证，也不是临床验证。这里的确认划分表示预先固定的家族保留集，不表示全程未暴露于方法开发。病例、参考标签和划分未根据模型预测或干预收益删改。

## 已冻结的设计

共 24 个新规则家族、576 例。12 个家族用于训练，4 个用于开发，8 个用于确认。规则家族的定义、划分与文件摘要写入 `family_split.json` 和 `manifest.json`。所有病例配对与表面变体均保留在同一划分。生成器不读取既往病例、模型预测、模型置信度或干预收益。

每个家族包含不同的 AND、OR、NOT 组合以及完整的例外结构。目标规则与互补规则分别导向 Cedar 与 Maple；例外成立时导向 Juniper。缺失、矛盾或来源不合格的事实按未知处理。无法建立唯一动作时请求复核。

每个家族有 3 个基础适用性对比，每对只改变一个当前患者原子。每对另有单位等价变换、主体及历史干扰与顺序变化，共 18 例。另含缺失、只有历史值、只有亲属值、当前记录矛盾 4 例，以及事实不全但仍有充分逻辑证明的 2 例。表面变体是稳健性重复，不是新的独立规则家族。

确认划分共 192 例。其中 144 例组成 72 对，涉及 8 个家族；仅标准表述构成的主要分析为 48 例、24 对。其余 48 例用于未知状态和充分证明诊断。样本量小，家族留出仍不等于自然临床语言或开放世界泛化。

## 文件与接口

| 文件 | 用途 |
| --- | --- |
| `mechanism_cases.jsonl` | 完整训练、开发与确认数据 |
| `module2_sealed.jsonl` | 确认划分中的完整 T/F 配对；兼容旧模块二候选字段 |
| `family_split.json` | 不依赖模型结果的固定家族划分与逻辑树 |
| `manifest.json` | 生成来源、范围、数据摘要、分析约定与限制 |
| `schema.json` | 病例接口 |
| `build_mechanism_set.py` | 可确定性重建数据的生成器 |

模块一预测只接收 `rules`、`patient`、`question`。`reference_structure`、`structured_facts` 和 `gold_*` 只供参考标注、显式参考结构诊断及事后评分，禁止传入 V6/V7 预测器。`joint.context.prediction_input` 使用正向字段白名单保证这个边界。

模块二的固定目标是 `R_TARGET` 对应的 `Route Cedar`，不会按当前正确答案改变。`spans.action` 为完整提示中的字符半开区间。`target_answer` 为目标动作的候选字母，配对双方相同。所有 T/F 行满足 `applicable == (gold_answer == target_answer)`。`outer_fold` 是家族名散列对旧三折资产的固定分配，不是使用新标签调参的依据。

完整提示始终按问题、患者、全部规则、全部选项排列。不适用分支、例外分支和未知分支均保留。

## CPU 核验

从仓库根目录运行：

```bash
python experiments/care_reliability_20261019/data/build_mechanism_set.py
python -m unittest discover -s experiments/care_reliability_20261019/joint -p 'test_*.py' -v
python experiments/care_reliability_20261019/joint/audit_cpu.py --output /tmp/care_joint_numeric_audit.json
```

完整性测试包括真值表、单位转换、时效与主体、矛盾、配对最小变化、家族隔离、完整上下文、充分证明与禁止参考答案进入预测。额外的 V7 数值兼容诊断仅调用确定性比较器，不调用语言模型。所有确认样本原样保留，不根据该诊断删选数据。数值兼容正确不能证明 TrialMatchAI、医学自然语言推断或端到端决策正确。
