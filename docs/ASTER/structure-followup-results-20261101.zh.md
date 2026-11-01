# 结构约束与只抑制编辑：开发集实验

仅开发100例；复用模块二direct:0.0，均值方向、280个头及增益5.989389629154409固定。
全部结构化条件使用完整原始规则树约束贪心生成，不强制遵循模块二路径；终点处置由规则读取。
8个条件×100例，零编辑自由生成100例复用旧结果；另700个条件单元包含小批次。
主要指标为路径与处置同时正确。当前不是独立验证结果。

| 条件 | 已完成 | 路径 | 处置 | 两者同时正确 | 无效 | 平均秒/例 |
|---|---:|---:|---:|---:|---:|---:|
| zero_structured | 100 | 82 | 82 | 82 | 0 | 3.86 |
| mean_free | 100 | 63 | 70 | 60 | 4 | 4.10 |
| mean_structured | 100 | 75 | 75 | 75 | 0 | 1.90 |
| suppress_free | 100 | 71 | 75 | 70 | 1 | 4.08 |
| suppress_structured | 100 | 77 | 77 | 77 | 0 | 1.76 |
| clip_free | 100 | 71 | 74 | 70 | 1 | 2.31 |
| clip_structured | 100 | 76 | 76 | 76 | 0 | 0.39 |
| zero_free | 100 | 65 | 67 | 65 | 1 | 3.34 |

所有对照均以预测路径执行；不改变原120例报告。自由生成耗时包含旧诊断前向，不能据此声称部署速度优势。

| 配对比较（联合指标） | 配对例数 | 纠正 | 退化 | 净变化 |
|---|---:|---:|---:|---:|
| zero_structured vs zero_free | 100 | 18 | 1 | 17 |
| mean_structured vs mean_free | 100 | 18 | 3 | 15 |
| suppress_structured vs suppress_free | 100 | 12 | 5 | 7 |
| clip_structured vs clip_free | 100 | 7 | 1 | 6 |
| suppress_free vs mean_free | 100 | 13 | 3 | 10 |
| suppress_structured vs mean_structured | 100 | 7 | 5 | 2 |
| mean_structured vs zero_structured | 100 | 7 | 14 | -7 |
| suppress_structured vs zero_structured | 100 | 4 | 9 | -5 |
| mean_structured vs clip_structured | 100 | 6 | 7 | -1 |
| suppress_structured vs clip_structured | 100 | 8 | 7 | 1 |
