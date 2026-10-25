# BiasBios 两方法配对比较

2026-10-26。协调器已报告SEKA与matched-zero的标准5000例运行均成功退出，SEKA独立导出为top1=0.9118、top3=0.9748。本脚本不硬编码这些数字，也不据此改方法。新增`compare_biasbios_paired.py`与4项合成测试尚未执行，只供服务器CPU只读计算；既有结果保持原样。

读取两份已verified的匿名导出目录，固定标准`train[5000:10000]`全部5000条。按唯一sample_id与绝对行号配对，复核实际k=3，再报告top1、top3准确率、SEKA−matched-zero差值、纠正/退化数量与比例，以及完整逐例差值。重名记录不合并、不删除。所有负结果保留。

可传实际处理后数据文件，使用`context`全文精确哈希，将完全相同的处理后传记放在同一bootstrap簇。10000次整簇重采样，seed0，保持原生病例加权分母。它是**重复文本来源分组，不是已知人身份**；同一个人可能有不同文本，因此该区间不能叫患者簇置信区间。不提供数据文件时保留完整点估计，区间明确未计算，不用名字替代真实患者映射。

提示匹配先比较全部5000条prompt SHA。若提供原run_manifest，再验证原forward trace并识别313个cached prefill批（batch16，末批8）。每批只比最初input_ids和attention_mask；按批次顺序配对，不要求全局call号相同。模型生成后的token、解码步数和EOS提前结束可以不同，不纳入输入一致性要求。显式attention_mask=None只与None匹配，缺失记录记unknown。无法确认prefill布局时保留统计，只把该条件标为blocked；不会因可选trace不可用丢弃逐例差值。

```bash
python "$AUDIT/compare_biasbios_paired.py" --seka-export "$BIOS_SEKA_EXPORT" --zero-export "$BIOS_ZERO_EXPORT" --processed-data "$BIOS_PROCESSED_JSONL" --seka-run-manifest "$BIOS_SEKA_MANIFEST" --zero-run-manifest "$BIOS_ZERO_MANIFEST" --output-dir "$NEW_COMPARISON_DIR"
```

生成`comparison.json`、`paired_cases.jsonl`及`prefill_comparison.jsonl`。只输出匿名行ID、文本哈希、数值和成本来源，不输出传记或姓名。报告区分“统计完成”与“输入匹配条件通过”，不把标准实现验证写成临床独立确认。现有GPU耗时/显存从两份导出审计记录带入，不新增模型前向。

```bash
python -m unittest discover -s "$AUDIT" -p 'test_biasbios_paired.py'
```

4项测试覆盖解码后缀/步数不同仍可匹配、prefill真实差异、mask缺失与None区分，以及重名元数据不导致行合并。没有本地执行或重新读取临床test。
