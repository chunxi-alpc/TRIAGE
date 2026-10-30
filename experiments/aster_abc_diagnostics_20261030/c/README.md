# C 独立分类/层级门控/任务诊断

仅在服务器执行，现有官方运行目录只读。先读 protocol.md。输出目录应使用新名称，脚本拒绝覆盖已有运行。

```bash
ssh gcx@10.29.128.115
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
python -m unittest experiments.aster_abc_diagnostics_20261030.c.test_c -v
python -m experiments.aster_abc_diagnostics_20261030.c.audit \
  --run /home/gcx/aster_manuscript_20261030/clinical01 \
  --output /home/gcx/aster_abc_diagnostics_20261030/c/cpu02
```

可选GPU诊断须先汇报资源并遵守协调的排程；完整confirm43，174次生成，GPU1一张A100，预计20–45GB、8–25分钟。可用--limit调整独立诊断样本量；不选择官方参数。

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.aster_abc_diagnostics_20261030.c.gpu_oracle \
  --run /home/gcx/aster_manuscript_20261030/clinical01 \
  --cpu-audit /home/gcx/aster_abc_diagnostics_20261030/c/cpu01 \
  --output /home/gcx/aster_abc_diagnostics_20261030/c/oracle02
```

CPU输入契约：run/frozen.json、cases/confirm.jsonl，以及confirm0/units/*和confirm1/units/*的classification.json和各method.json。病例需case_id/source_id/guideline/gold_rules/gold_leaf，分类需raw_decisions[group_id].logits或None；技术失败不得编码为语义NONE。群组标签只评价gold_rules明确给定的金标路径，不给非路径组伪造语义标签。

GPU另需冻结expert和本地Qwen3-8B权重；只用既有分类缓存，无新临床标签。predicted和oracle采用同样的分块注意力、探针设置和生成预算。旧predicted产物只用于CPU描述，不能与新oracle当作纯路由因果对照。零编辑两路首logit和输出必须相等。所有oracle输出标记diagnostic_only。

CPU输出summary.json（路径门控、方法误差、配对来源簇区间）、routing_cases.json、paired_outcomes.json、scoring_differences.json。GPU输出manifest.json/reference.json/records.jsonl/status.json/summary.json。records含生成内容，仅存服务器，不公开患者文本。两个来源簇的CI仅探索性描述，不能给跨来源稳定性强结论。

补充参考路径覆盖/风险（不为非路径组伪造标签）：

```bash
python -m experiments.aster_abc_diagnostics_20261030.c.risk \
  --run /home/gcx/aster_manuscript_20261030/clinical01 \
  --output /home/gcx/aster_abc_diagnostics_20261030/c/risk02
```
