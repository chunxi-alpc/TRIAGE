# P6 数据、缓存与审计字段

准备约定，尚未实现或验收运行器。病例文本和生成结果留在服务器受控目录；仓库保存协议、指纹、去标识汇总和审计。

## 配对记录

每条记录包含 pair_id、guideline_id、source_case_id、source_split、source_revision、patient_facts、question、完整 native_guideline 和 decoy_guideline、edited_node_id、原生/诱饵参考路径或可接受路径集合、条件与处置字符区间、复杂结构标记及理由。模型运行前保存内容指纹与评分配置指纹。

核查每对患者事实、问题、规则条件相同；变化仅限目标节点处置与对应参考路径。保留候选总数、排除原因和人工核验结果，不能按方法表现删除病例。review_template.csv 的 reviewer_id、reviewed_at、verdict 由真实审阅人填写；未填写为待核验。

## 共享盲判

同一输入只产生一份盲判缓存。键应涵盖输入、模型、tokenizer、判断提示、解码配置及实现版本；载荷保存分支编号、概率、证据位置和生成原文。TS-Note、TS-Prune、TS-SEKA、ASTER-self 均引用相同 blind_cache_id 和 evidence_hash。不同方法不得自行重判。

原生/诱饵盲判若实际输入相同，可复用同一缓存；必须以输入指纹核对，不能仅根据 pair_id 假定相同。判断阶段×执行阶段消融中，每种判断条件的执行变体复用该条件对应的缓存。

## 生成与干预日志

每次输出记录 pair_id、variant、method、blind_cache_id、evidence_hash、完整输出及 output_hash、参考路径和 scorer_config_hash、生成预算、解码设置、输出 token 数、预填充 token 数、延迟、停止原因、解析状态、重复判定、目标 span、实际干预 span/层和命中结果。零强度与无引导使用同一输入和输出格式；记录解析失败并计失败。

同一个（病例、输出、参考路径、评分配置）只评分一次，其余方法复用 score_cache_id。解析器或评分器修订后产生新版本，不能只对不利结果重算旧分数。

## 汇总

分别报告原生/诱饵 PathCost、BFR、DA，以及相对 Vanilla 的原生改错率和纠正率。报告重复、触顶、解析失败、规则片段语义正确率、覆盖率、实际干预比例、token 和端到端成本。按 guideline_id 聚类抽样，每次保留完整病例配对；160 对不是 320 个独立患者。未运行指标填 null 或 NOT_RUN，不填 0。

最低数据门槛、稳定性、原生不受损、诱饵收益、最强同信息文本基线差值和成本规则均见 protocol.json。这里不增加协议之外的审批门槛。
