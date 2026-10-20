# N1–N5 状态

`cpu_verified`

本阶段只生成新数据、结构化 oracle、分组划分和泄漏审计，没有启动模型或 GPU 前向。数据生成器、N1 四格、N2 中性接收者、N3 合法补全集合、N4 16 格和 prompt allowlist 已通过 CPU 检查。

当前 CPU 数据是正式机制数据的冻结候选；N2 的 `synthetic_prior` 与真实参数知识轨严格分开，N5 的 PR12 A1 精确连接在 GPU 阶段前由不可变源文件完成。

后续状态只能按协议更新为 `gpu_smoke_passed`、`gpu_evaluated`、`blocked` 或 `not_run_gate_fail`，不得把 CPU 验证写成神经效果。
