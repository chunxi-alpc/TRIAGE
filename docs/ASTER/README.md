# ASTER 主线文档

[项目首页](../../README.md) · [文档导航](../README.md) · [历史支线](../BRANCHES.md)

**当前版本：2026-10-27。** ASTER 更新为 *Action-blind Selection, Then Eigen-subspace Re-routing*。模块一确定为处置盲适用性评估（Action-Blind Applicability Assessment, ABAA）；三模块分别判断适用性、离线学习方向和在线编辑规则键。

## 阅读顺序

1. [设计决策与版本比较](design-update-20261027.zh.md)：哪些修改更合理、哪些保证必须撤下。
2. [论文工作稿](manuscript.zh.md)：保留 TLBS 与五卷评测，重写三模块方法及理论边界。 公式使用 GitHub 兼容的排版，检查脚本见 [数学公式检查](../../scripts/check_aster_markdown_math.py)。
3. [模块一确定版](module1-spec.zh.md)：作为当前模块一的规范，不再采用原子抽取或证据员链条。
4. [六项主张与证伪](claims-and-falsification.zh.md)：将表征假设与执行器贡献分开验证。
5. [实验计划](experiment-plan.zh.md)：M1-only 强基线、同信息比较、校准与选择性干预。
6. [新版代码](../../src/aster/README.md)、[实现状态](implementation-status.zh.md)与[证据状态](evidence-status.zh.md)：明确已经实现与仍待验证的范围。
7. [引用核验](references-to-verify.zh.md)：已有原始来源与尚未核对的项目。

模块一输出只进入模块三；其开发集输出可用于温度及策略选择，不用于拟合模块二子空间。取消保形预测和 LTT 后，主方法没有有限样本覆盖或伤害认证结论。

## 来源与历史

[2026-10-26 原稿](sources/manuscript-20261026.txt)与[六项原始主张](sources/user-claims-20261026.md)继续保存。新增[三模块讨论稿来源](sources/three-module-proposal-20261027.txt)记录本次上传的中间方案；其中的原子抽取、保形预测、认证及部分理论强断言已被本次确定版修改，不能将来源稿当作现行规范。

[旧图](figures/README.md)、[原稿审计](audit.zh.md)及[旧编辑记录](editorial-notes.zh.md)保留历史语境。2026-10-26 的完整主线可通过[固定提交](https://github.com/chunxi-alpc/TRIAGE/tree/781ee863784c60f1f6710ba5eede22bc8ee6b93c)查看。当前方法图使用主稿和首页中的三模块流程图，旧图 2／图 4 不再代表当前算法。

所有经验性结果仍须以真实模型运行、锁定数据和逐例产物支持。源码契约测试不能替代性能、校准或临床有效性评价。
