CARE V5 保持 `module1.run(guideline, patient, question)` 和 `Module1(config_path=None, device="cuda:0", top_k=12, cache_dir=None, backend=None).run(...)` 的调用形式。默认类和函数均不调用 Qwen。旧实现仅通过 `LegacyModule1` / `run_legacy` 显式访问。`backend` 仅用于读取最终生成模型的设备，不调用其方法。

默认配置为 `experiments/care_module1_v5/config.json`，可用 `CARE_MODULE1_V5_CONFIG` 指定。模型优先使用配置的 `model_paths`、`CARE_MODULE1_V5_BASE_MODEL` / `ADAPTER_MODEL` / `EVIDENCE_MODEL` 环境变量，或服务器已有的 `/home/gcx/models/care_module1_v5/` 模型目录。设备、缓存和规则表可分别用 `CARE_MODULE1_V5_DEVICE`、`CARE_MODULE1_V5_CACHE`、`CARE_MODULE1_V5_RULES` 指定。已构建的 `CAREModule1V5` 也可直接注入 `Module1(engine=...)`。

结构化指南直接使用。文本指南只允许在离线规则表的 `text` 或 `source_text` 中精确匹配到唯一规则，无匹配即拒绝；不启动模型解析，也不按患者答案选择规则。`top_k` 仅保留旧构造接口，不截断调用者给出的规则。

输出的 `schema_version` 为现有模块二、三接受的 `module1.v1`，`implementation_version` 为 `module1.v5`，同时保留 V5 的 `rules` 结构。`decisive_condition_ids` 记录使完整“前提且无例外”表达式成立、不成立或仍未知的条件；成立时列入 `supporters`，不成立时列入 `blockers`。OR 分支只把能决定其结果的分支纳入证明。

`confidence` 是证据分数，不是经过校准的正确概率：无冲突且有确定性数值证据时为 1；其他已知条件取与最终状态同向的实际 MedNLI 概率最大值；未知、冲突或没有同向证据时为 0。相关证据完整保留原文字符区间，未知候选句单独保留。binding gate 等于该分数，适用规则的 action gate 为 1。

V5 不生成或比较闭卷与指南 Qwen 决策，因此旧 `conflict_scores` 的动作冲突信号标记为未知，`prior_competition_weight` 为 0。TrialMatchAI 与证据模型之间的真实冲突保留于条件和 `model_evidence_conflicts`，不冒充闭卷决策冲突。模块三仍可读取适用状态与证据，但需要闭卷动作冲突的先验抑制路径没有可用信号。

兼容字段不会补造缺失的指南内容。当前 V4 规则没有完整原始治疗动作，输出会在 `metadata.missing_action_rule_ids` 明列缺动作的规则。需要调用者提供合法来源的动作文本和位置后，才适合完整治疗决策流程。

数值优先级在运行前固定：明确、未撤回、单位可转换且比较结果一致的数值由程序判定；NLI 不因自身数值推断意见不同就推翻该结果，分歧仍写入冲突字段。未解决的多值跨阈值、缺失或无法解释的数值保持未知。非数值条件的 TrialMatchAI 与 MedNLI 已知结论冲突时保留未知。
