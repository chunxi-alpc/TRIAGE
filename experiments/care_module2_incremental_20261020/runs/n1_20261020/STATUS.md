# N1 status

`gpu_evaluated`（新数据，单次 frozen test evaluation）

N1 test 使用 8 个留出逻辑家族、32 行、384 次选项 forward。开发集预先选择 beta=0.0；四种方法均为 8/32 行正确，四格 family 全对为 0/8。显式文本、直接 bias 和低维 route 没有超过无干预基线，因此按协议停止 N1 的空间/头搜索，不进入旧扫描。

模型无法在该新合成任务上区分三类答案，结果应解释为任务能力/机制负结果，不作 L1/L2 成功结论。
