# 标准任务逐例统计与导出

2026-10-26。以下新脚本仅准备完成，**尚未执行**；所有运行交协调器在115服务器CPU进行，无模型前向，不改任何方法参数。只分析已有标准实现结果，不是临床独立确认。没有用已汇报的效果选择区间算法、样本或方法设置。

## CounterFact efficacy

`counterfact_entity_ci.py`读取官方`counterfact.jsonl`的零基5000:10000标准slice，取其公开`entity`字段，以精确字段值的SHA256分组。重复实体的全部事实一起抽样，不将每条事实假定为独立样本。输出只有sample/entity哈希、行号、数值，无实体名称或病例文本。

输入兼容已有`work/independent_audit/export_counterfact.py`产物，每种方法目录必须含通过审计的`audit.json`、`case_metrics.jsonl`和`predictions.jsonl`。核对文件哈希、benchmark、5000条唯一ID与官方slice完全一致、每题一个efficacy提示及二元分数。保留所有既有正负结果。原方法分数完全来自文件，不硬编码已汇报均值。

固定比较SEKA对matched_zero、unsteered和PASTA；计算原生ES差值、纠正数、退化数及全5000题分母对应比率。固定分析随机种子0，10000次整实体percentile bootstrap；所有比较共用抽样，簇内保留配对。每次统计用重采样的总成功差除以总病例数，保持原生病例加权，不改成实体等权。95%区间为各比较的边际区间，**不作多重检验校正或临床显著性声明**。同entity被当作相关簇，不声称不同表面名字的实体映射已经彻底消歧。

```bash
python "$AUDIT/counterfact_entity_ci.py" --data "$COUNTERFACT_JSONL" --seka "$SEKA_EXPORT" --matched-zero "$ZERO_EXPORT" --unsteered "$UNSTEERED_EXPORT" --pasta "$PASTA_EXPORT" --output-dir "$RUN/entity_ci" --bootstrap-samples 10000 --bootstrap-seed 0
```

产物`entity_ci.json`与`paired_cases.jsonl`。本工具不新增模型seed复现，不声称forward输入已匹配；后者引用独立审计已有trace。CAD和Spotlight仅列planned，不填数值。Paraphrase是原病例多提示的平均分，不送入此二元efficacy工具，也不改写为二元正确/错误。

## BiasBios

`export_biasbios.py`读取官方`result.json`、实际处理后JSONL及成功退出的运行manifest。默认5000:10000，逐行核对原ID、源职业标签，以及官方完整context构建出的提示（允许官方marker版本）；原源码先确保context以句号结束，再加两个换行和补全提示。不会按重复名字合并或删除行。

导出sample_id为SHA256(处理文件SHA、split、绝对行号、原ID)。重复名字的记录因此保持不同sample_id。单独的`source_name_id_sha256`只用于展示重名关系，不宣称是已核实患者身份。原始姓名、prompt、generation、传记不导出；职业标签和top-k职业列表保留以核查指标。

主指标严格按[官方evaluate.py第365–368行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L365)计算：`predictions[0] == target`为top1，`target in predictions`为topk，k取实际`metrics.k`。分别与保存均值比较，不臆造`accuracy`等字段。另核对top-k概率顺序与target已出现时的概率一致性；只验证已保存列表与聚合，不声称从全词表logits重新推导了职业排序。辅助fluency/consistency明确标为saved，不冒称重新计算；非有限数转null并计数，不删除行。

```bash
python "$AUDIT/export_biasbios.py" --raw "$BIAS_RESULT" --processed-data "$BIAS_JSONL" --run-manifest "$BIAS_RUN_MANIFEST" --output-dir "$RUN/biasbios_export" --start 5000 --end 10000
```

产物`predictions.jsonl`及`audit.json`，包含实际指标、重名统计、文件/源码/模型哈希和原运行成本。生成token数量没有原token ID时记null，不以解码文本重新分词猜测。

4项新增合成测试涵盖不等簇大小的正确分母、保留负效应、重名保留与top1/topk复核、同名不同源顺序错位检测：

```bash
python -m unittest discover -s "$AUDIT" -p 'test_standard_exports.py'
```

输入来源：[CounterFact标准slice与entity预处理](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/counterfact/preprocess.py)，blob`9f9cfd74390e6952cf9fc300d28a3c35996ff424`；[实际context拼接](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/utils/pasta_utils.py#L81)。其他BiasBios指标/转换依据见`SEKA_BIASBIOS_PROTOCOL.zh.md`。
