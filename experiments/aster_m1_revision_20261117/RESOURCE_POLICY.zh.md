# 修补资源设置更新（2026-11-17）

用户追加指示：“没有修补预算”。所有七个模型的确认修补前向次数均不设上限；完整运行95%累计归因质量对应的候选头。

本文件和更新后的 CODEX_EXECUTION.zh.md 取代初始 PROTOCOL.zh.md 中的200000次修补上限。初始协议、configuration.json 和 identity.json 保留原样，作为本轮已提取缓存的身份记录。每个模型的当前设置保存为 effective_configuration.json，selection_plan.json 和 resource_preflight.json 记录实际候选数量、前向估算及 null 上限。

候选与确认比例、来源划分、数值容差、M2、M3、开发/留出评价和模型扩展顺序保持既定设置。M3 的逐对 C/F 约束 delta=0.2923349 保持不变。

Qwen 已完成的1823个 fit 输入保留；统一队列使用 --resume 接续剩余输入。全部阶段日志和正负结果继续提交至当前PR分支。
