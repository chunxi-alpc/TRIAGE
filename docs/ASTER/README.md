# ASTER 主线文档

[项目首页](../../README.md) · [文档导航](../README.md) · [历史支线](../BRANCHES.md)

**模块一规格：2026-10-27；模块二规格、实验协议与机制预实验：2026-10-28。** ASTER 为 *Action-blind Selection, Then Eigen-subspace Re-routing*。模块一采用处置盲适用性评估（Action-Blind Applicability Assessment, ABAA）；三模块分别判断适用性、离线学习方向和在线编辑规则键。本轮新增按当前逐 token 公式与指南前缀协议运行的机制结果，正式主任务的实验章节、基线与待填表继续保留。

## 阅读顺序

1. [设计决策与版本比较](design-update-20261027.zh.md)：哪些修改更合理、哪些保证必须撤下。
2. [论文工作稿](manuscript.zh.md)：保留 TLBS 与五卷评测，对齐三模块方法、实验章节及理论边界；[模块二补充材料](module2-supplement.zh.md)记录三状态构造、谱估计、双状态选头及验证目标。公式使用 GitHub 兼容的排版，检查脚本见[数学公式检查](../../scripts/check_aster_markdown_math.py)。
3. [模块一确定版](module1-spec.zh.md)：作为当前模块一的规范，不再采用原子抽取或证据员链条。
4. [六项主张与证伪](claims-and-falsification.zh.md)：将表征假设与执行器贡献分开验证。
5. [实验章节与执行计划](experiment-plan.zh.md)：确定正文第 5–7 节、两项主任务、三类基线和主要比较。配套[数据协议](dataset-protocol.zh.md)、[基线协议](baseline-protocol.zh.md)、[指标公式](experiment-metrics.zh.md)、[结果表与代填段落](experiment-tables.zh.md)、[历史实测结果](historical-results.zh.md)。
6. [新版代码](../../src/aster/README.md)、[实现状态](implementation-status.zh.md)与[证据状态](evidence-status.zh.md)：明确已经实现与仍待验证的范围。
7. [引用核验](references-to-verify.zh.md)：已有原始来源与尚未核对的项目。
8. [模块二机制预实验](../../experiments/module2_mechanism_20261028/README.md)：固定协议、真实模型结果、方向／迁移图、全部病例对与示范选择；对应[模块二补充材料](module2-supplement.zh.md)的三项验证。

模块一输出只进入模块三；其开发集输出可用于温度及策略选择，不用于拟合模块二子空间。取消保形预测和 LTT 后，主方法没有有限样本覆盖或伤害认证结论。

## 已完成的模块二机制测量

`Qwen3-0.6B-Base` 已完成 144 次患者前置离线前向与 48 次指南前置在线前向。12 个开发指南构成 96 个目标三元组；拟合／选择／确认指南为 6／3／3。选择集冻结的四个层／KV 头为 `(13,4)`、`(23,6)`、`(6,2)`、`(22,7)`，确认有向差分正向数为 21/24、22/24、19/24、16/24，两种在线状态各自均为 24/24 的目标平均正响应。

已完成 24 个物理患者提示 × 9 个固定配置的 216 条联合编辑记录。预设增益 4 的谱投影联合编辑使全部 12 个配置的两名相反患者都提高正确／兄弟规则注意力质量比，几何平均提升约 4.827 倍。按真实前缀边界修正后的单 token 答案读出，全部配置仍为 12/24 正确、1/12 配对全对，预测翻转为 0，未取得答案收益。

本轮支持这批规则上的共享方向、在线读取迁移和局部控制。原始均值方向的确认平均余弦更高，因此不声称谱投影优越；患者为显式布尔条件、确认仅含 3 个指南，且沿用此前公开的开发划分。[完整结果](../../experiments/module2_mechanism_20261028/RESULTS.zh.md)保留全部对照与精选案例，未接入模块一预测或验证处置生成。根仓库中其他历史实验继续保持原模型、输入与算子的含义。

## 来源与历史

[2026-10-26 原稿](sources/manuscript-20261026.txt)与[六项原始主张](sources/user-claims-20261026.md)继续保存。新增[三模块讨论稿来源](sources/three-module-proposal-20261027.txt)记录本次上传的中间方案；其中的原子抽取、保形预测、认证及部分理论强断言已被本次确定版修改，不能将来源稿当作现行规范。

[旧图](figures/README.md)、[原稿审计](audit.zh.md)及[旧编辑记录](editorial-notes.zh.md)保留历史语境。2026-10-26 的完整主线可通过[固定提交](https://github.com/chunxi-alpc/TRIAGE/tree/781ee863784c60f1f6710ba5eede22bc8ee6b93c)查看。当前方法图使用主稿和首页中的三模块流程图，旧图 2／图 4 不再代表当前算法。

所有经验性结果仍须以真实模型运行、锁定数据和逐例产物支持。源码契约测试不能替代性能、校准或临床有效性评价。
