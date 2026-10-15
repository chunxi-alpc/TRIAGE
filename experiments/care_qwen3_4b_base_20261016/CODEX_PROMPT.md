# 复制给 Codex 的一句指令

请在当前项目中先阅读 `AGENTS.md`、`CODEX_TASK.md` 和 `EXPERIMENT_PROTOCOL.md`。使用 `Qwen/Qwen3-4B-Base` 并复用官方 SEKA 仓库的实现，完成 CARE 三个动机观察：O1 规则忠实性与误用权衡（需真实闭卷冲突筛选）、O2 相同规则在相关性与适用性 2×2 条件下的 Key 表征变化、O3 连续带符号 Key 编辑的权重–注意力 logit 方向一致性。实现真实 Hugging Face 模型运行器，输出可追踪的原始记录、聚类置信区间和 PNG/SVG 学术图。先生成并验证逻辑测试数据，再取得来源核实的临床样本；选择头、投影和超参数时不能使用测试集。严格检查 Qwen3 的 QK-Norm、RoPE 和 GQA 头映射；不得生成虚构结果或将空白模板当论文图。若 4B 未观察到足够合格的真实先验冲突样本，仅如实报告合格样本数、筛选标准、阻塞原因和状态，不擅自更换模型。最终运行测试并汇总每个假设 supported / not_observed / inconclusive / blocked，附命令、环境版本、模型 revision、实际结果和全部图。不要只给出代码，要真正执行；缺少资源时明确报告阻塞原因。
