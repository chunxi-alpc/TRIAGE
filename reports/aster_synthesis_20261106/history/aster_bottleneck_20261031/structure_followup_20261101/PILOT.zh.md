> Source snapshot: /home/gcx/aster_bottleneck_20261031/structure_followup_20261101/PILOT.zh.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# 结构约束与只抑制编辑：开发集实验

仅开发100例；复用模块二direct:0.0，均值方向、280个头及增益5.989389629154409固定。
全部结构化条件使用完整原始规则树约束贪心生成，不强制遵循模块二路径；终点处置由规则读取。
8个条件×100例，零编辑自由生成100例复用旧结果；另700个条件单元包含小批次。
主要指标为路径与处置同时正确。当前不是独立验证结果。

| 条件 | 已完成 | 路径 | 处置 | 两者同时正确 | 无效 | 平均秒/例 |
|---|---:|---:|---:|---:|---:|---:|
| zero_structured | 2 | 2 | 2 | 2 | 0 | 5.92 |
| mean_free | 2 | 1 | 2 | 1 | 0 | 7.23 |
| mean_structured | 2 | 2 | 2 | 2 | 0 | 4.28 |
| suppress_free | 2 | 2 | 2 | 2 | 0 | 7.73 |
| suppress_structured | 2 | 2 | 2 | 2 | 0 | 4.07 |
| clip_free | 2 | 2 | 2 | 2 | 0 | 1.42 |
| clip_structured | 2 | 2 | 2 | 2 | 0 | 0.38 |
| zero_free | 2 | 1 | 2 | 1 | 0 | 4.80 |

所有对照均以预测路径执行；不改变原120例报告。自由生成耗时包含旧诊断前向，不能据此声称部署速度优势。

| 配对比较（联合指标） | 配对例数 | 纠正 | 退化 | 净变化 |
|---|---:|---:|---:|---:|
| zero_structured vs zero_free | 2 | 1 | 0 | 1 |
| mean_structured vs mean_free | 2 | 1 | 0 | 1 |
| suppress_structured vs suppress_free | 2 | 0 | 0 | 0 |
| clip_structured vs clip_free | 2 | 0 | 0 | 0 |
| suppress_free vs mean_free | 2 | 1 | 0 | 1 |
| suppress_structured vs mean_structured | 2 | 0 | 0 | 0 |
| mean_structured vs zero_structured | 2 | 0 | 0 | 0 |
| suppress_structured vs zero_structured | 2 | 0 | 0 | 0 |
| mean_structured vs clip_structured | 2 | 0 | 0 | 0 |
| suppress_structured vs clip_structured | 2 | 0 | 0 | 0 |
