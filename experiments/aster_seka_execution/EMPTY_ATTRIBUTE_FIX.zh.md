# CounterFact 空属性最小修复

协调器只按字段结构核查固定 5000 例：relative index 2093 / id 7093、relative index 3594 / id 8594 的 attribute 为空；对应 prior 非空，前缀有效。本修复没有读取这些案例的模型分数。

- CAD 不使用 attribute。空属性返回空跨度列表，完整上下文与原问题照常进入真实双路推理，公式和原 efficacy 指标不变。
- Spotlight 空跨度使用同一完整输入的真实 no-hook 生成，记录 `no_support` 或 `control_unavailable`、`intervention_applied=false`、`effective_control=no_hook`；不制造证据，不称有效注意力对照。其真实输出保留在完整固定分母。
- 没有 BOS 补造、错误概率插补或吞掉一般异常。其他未预见异常仍保留 partial 后失败，不能把失败行冒称实测概率。
- 新增两个合成边界测试，尚未由本代理运行。原 15 项加这两项共 17 项。

需同步三个文件：`baseline_operators.py`、`run_counterfact_baselines.py`、`test_baseline_operators.py`。CAD 与 Spotlight 入口本身是一进程一方法；协调器应把两方法放入各自的服务器作业，不能让 Spotlight 启动依赖 CAD 的成功退出。这里不启动或分配 GPU。
