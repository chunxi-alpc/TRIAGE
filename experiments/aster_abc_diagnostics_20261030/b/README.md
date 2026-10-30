# B 独立编辑机制诊断
所有命令均在服务器执行，在诊断工作树根目录运行。
输入只读：clinical01/frozen.json、analysis/expert.json、review/accepted_pairs.jsonl、collect0/collect1 中的已采集张量。
默认confirm，按来源排序后每簇取1配对；当前4配对来自2簇，因此实际GPU诊断2配对、6状态、60方法记录。
临床范围只有已规则核验的年龄根节点；医生独立裁定三态对为0。

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
python -m unittest experiments.aster_abc_diagnostics_20261030.b.test_b
python -m experiments.aster_abc_diagnostics_20261030.b.saved_responses --output /home/gcx/aster_abc_diagnostics_20261030/b/new_saved_cpu
CUDA_VISIBLE_DEVICES=0 /home/gcx/aster_manuscript_20261030/venv-overlay/bin/python -m experiments.aster_abc_diagnostics_20261030.b.run_gpu --output /home/gcx/aster_abc_diagnostics_20261030/b/new_gold_root --route gold_root --max-sources 8 --steps 17
python -m experiments.aster_abc_diagnostics_20261030.b.summarize_replay --run /home/gcx/aster_abc_diagnostics_20261030/b/new_gold_root --output /home/gcx/aster_abc_diagnostics_20261030/b/new_gold_summary
```

predicted_root 使用相同命令加 --route predicted_root 与新的输出目录，不能覆盖gold_root结果。
新增候选集必须按根节点rule_id、source_id、split、states、verification、guideline输入契约经独立裁定后先采集；不能直接将通用临床候选伪装成现有年龄规则数据。
模型权重位置由原只读data.MODEL使用，保持Qwen3-8B revision 2602814与原专家一致。

输出：
- preflight.json：预定分母、实际来源、方法、冻结增益。
- records.jsonl：逐状态/方法汇总及失败；未知状态独立核验。
- detail/*.pt：逐层/KV头/共享查询头/规则目标词元的实际键增量分数、理论分数、attention与同一query撤销编辑的attention，baseline共享token。
- summary.json：探索性来源簇汇总；summarize_replay追加配对差异CI、覆盖率、分位置统计。
- status.json：完成/停止状态，真实时间与显存。
17步中抽取0/1/2/4/8/16六个query位置。位置0决定首生成token，位置1是已强制输入baseline首token后的查询。
所有方法共用baseline token轨迹，因此这里只检验局部机制，不把这些输出当成各方法自由生成的任务准确率。

CPU检查6/6通过。GPU真实启动与结果以协调者及独立运行目录status为准。
