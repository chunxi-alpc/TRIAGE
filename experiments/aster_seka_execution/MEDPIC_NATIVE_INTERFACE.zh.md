# MedPIC 执行前接口与限制

2026-10-26。本次仅使用此前审计的公开schema、官方数据卡说明及合成例编写接口，**没有重新打开467题、题目正文或标签**。新增代码/测试尚未运行。固定数据revision为`9ef6db4f13865b14fc2e6be3f94dcfaf3a0cf983`；此前服务器元数据已核实467题，284 GF和183 CF。目标是先冻结方法，再执行完整外部评价。

## 原生输入究竟有什么

方法输入只允许三个字段：

| 字段 | 数据类型 | 模型用途 |
|---|---|---|
| `patient_vignette` | 字符串 | 患者证据 |
| `question` | 字符串 | 需要回答的问题 |
| `options` | 选项字母到文本的字典 | 全部候选选项，保持发布顺序 |

`instance_id`仅用于日志、稳定seed和评价对齐。`answer`、`benchmark_task_family`、`taxonomy_*`及任何配对标识不传给方法。已核查schema/元数据**没有独立官方指南字段或可直接供方法读取的指南文档/规则映射**；不能将任务名称“guideline_following”解释为每题附有官方指南。仅凭schema也不能判断题目字符串是否引用个别规则；执行时应原样提供这些合法原生字段。

原生任务是多选集合，不是“预测一个最可能字母”。每题独立零样本输入、简短解释、指定最终答案字段；集合须完全正确才得分。新适配采用`FINAL_ANSWER: ["A", "C"]`这一冻结的输出格式，**这是本项目适配模板，不声称取得了作者逐字推理代码/提示**。该格式及原生指标与公开说明一致，但不据此宣称重现作者全部模型设置。

## 可直接使用的最小适配

`medpic_native_adapter.py`不打开数据文件，提供：

- `native_input(row)`：白名单投影，只将上述三项交给方法。
- `render_native(task)`：构造原生提示，返回患者文本精确字符范围。`prepare_chat_tokens`转换为相同聊天模板的输入词元和患者范围mask；这个mask本身不注入注意力。
- `hf_native_generator(model, tokenizer, DecodeConfig(...))`：最小HF文本基线，模型已由协调器加载；不下载、不另启GPU。默认非thinking、512新token、上下文32768、贪心解码，是**待冻结的项目预算**，不是已核实的论文默认值。超预算时保留错误，不截断或删除病例。
- `predict_native_row(...)`：在配置锁为原生协议且冻结后调用。仅给生成callback传`PreparedPrompt`及稳定seed，不传原始行。输出分成可提交的预测/成本记录，以及仅服务器保存的prompt和原始模型输出。
- `parse_final_answer`：只解析唯一且位于末尾的指定字段；不从解释里的字母猜答案。无效、多重字段、未知选项和模型错误均保留，交既有评价器按错误计入全467题分母。

调用接口示意（真正载入数据必须在配置锁检查后，由独立执行器完成）：

```python
config = DecodeConfig(max_new_tokens=512, context_limit=32768, do_sample=False)
generate = hf_native_generator(model, tokenizer, config)
# lock: frozen/frozen_at/methods/seeds/protocol/prompt_template_sha256,
#       plus frozen model, source, decode and method fingerprints.
public_row, private_trace = predict_native_row(
    row, generate, method="native_full_text", seed=0, lock=lock
)
```

原生锁的`protocol`固定`medpic_native_v1`，`seeds`为0/1/2，`prompt_template_sha256=fingerprint(INSTRUCTION)`。应同时保存方法源码、模型/投影、解码配置、全部对照和比较族的指纹；若设置`decode_config_sha256`，适配器会核对callback实际声明配置。每个原始seed和题目ID派生稳定的case RNG seed，同病例跨方法一致；不依赖答案或任务家族。贪心下三seed可能完全一致，不将它们当三个独立患者。

## 患者证据注意力能做什么

原生注意力适配可在病例范围中选择患者事实，或根据原生问题/选项预测相关证据。选择器必须在MEGA train_val或合成数据上完成设计，外部评价后不改规则。接口已提供患者范围与相同输入词元，现有已审计的SEKA/PASTA钩子可在该范围工作；此文件只准备基线和接口，**没有宣称原生MedPIC注意力方法已经实现或运行**。

所有方法须使用同样的患者/问题/选项、相同预算和输出解析。患者证据的同路径零强度、反向、错误对齐应继续使用相同原生提示。若选择器产生额外患者条件文字或中间判断，文本两阶段基线也获得同样文字，额外前向/词元成本全部计入。不能用隐藏答案、GF/CF家族、人工配对标签或gold规则选择患者证据。

## 原生与增加规则的开卷实验分开

原生协议不凭空增加指南。若从参数知识生成规则、动作映射，或从外部资料检索文档，就属于单列的`medpic_augmented_open_book_v1`。

`render_augmented_open_book`仅提供单独的接口，要求声明`model_derived_rules`或`retrieved_document`。生成规则明确标为“非官方指南”；检索文本也不会由适配器自动认证为官方指南。该接口没有读取或生成任何测试题规则，原生预测函数拒绝用开卷配置锁冒充原生运行。

在看测试内容前冻结的是**规则生成器/检索策略、动作映射算法、提示、预算及评价方式**；不是提前读取测试题生成逐题规则。冻结后的逐题派生内容应完整保存来源和指纹，并给同信息文本基线使用同一份内容。只有“同一派生内容+普通文本”与“同一派生内容+注意力”的匹配比较才支持注意力增益；原生闭卷与增加规则的开卷分数不能混排宣称原生SOTA。生成或检索成本、额外判断/执行调用全部记录。

## 当前评价限制

完整467题的原生精确集合准确率、GF/CF分组点估计可在冻结后计算；既有`evaluate_medpic.py`已准备该评分流程。实际metadata没有可核实的89对链接，Pair仍blocked，不猜配。真实患者/规则来源簇也未得到；缺映射时区间/显著性保留blocked，不能把每题当独立患者。病例文字或任务类别相似不作为真实配对证据。

新增6项合成合同测试覆盖输入白名单、指定字段多选解析、原生/开卷分离、callback不接触原始行、失败记录保留和协议锁隔离：

```bash
python -m unittest discover -s "$AUDIT" -p 'test_medpic_native_adapter.py'
```

这些测试不读取真实数据，也不运行模型。完整外部执行必须等待主方法与全部种子冻结；当前没有MedPIC模型分数。

依据：[官方固定版本数据卡](https://huggingface.co/datasets/TIM0927/MedPIC-Bench/blob/9ef6db4f13865b14fc2e6be3f94dcfaf3a0cf983/README.md)、[论文原生评价设置](https://arxiv.org/html/2608.03028v1)，以及协调器已执行并归档的服务器字段/Pair元数据审计。本轮未重开原始测试文件。
