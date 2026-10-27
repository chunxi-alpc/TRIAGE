# 文档导航

[项目首页](../README.md) · [历史支线](BRANCHES.md)

## 当前主线：2026-10-27 三模块 ASTER

[主线索引](ASTER/README.md)是当前研究入口。模块一固定为同一骨干模型的处置盲分支分类，模块二离线学习适用性方向，模块三在线平移规则键。

- [论文工作稿](ASTER/manuscript.zh.md)：研究问题、三模块方法、数学边界、待测结果。
- [模块一确定版](ASTER/module1-spec.zh.md)：输入白名单、受限 logits、温度、NONE 与弃权。
- [设计决策与版本比较](ASTER/design-update-20261027.zh.md)：为何采用最新方案。
- [实验协议](ASTER/experiment-plan.zh.md)：同信息基线、开发／测试隔离、覆盖与错误率。
- [新版代码说明](../src/aster/README.md)：模块一及规则增益接口。
- [实现状态](ASTER/implementation-status.zh.md)：参考代码与未完成端到端系统的边界。

## 历史内容

[2026-10-26 ASTER 固定版本](https://github.com/chunxi-alpc/TRIAGE/tree/781ee863784c60f1f6710ba5eede22bc8ee6b93c)采用两阶段注意力掩码、证据员和对数门控。旧图、审计与编辑记录保留其时间语境。

更早版本见 [archive/pre-aster-main-20261026](https://github.com/chunxi-alpc/TRIAGE/tree/archive/pre-aster-main-20261026) 和[支线索引](BRANCHES.md)。旧 CARE／ACRE 及 `src/module1`、`src/module2`、`src/module3`、`src/pipeline` 的接口不代表当前 ASTER 已实现。
