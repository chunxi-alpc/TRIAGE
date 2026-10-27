# 主线与归档记录

[项目首页](../README.md) · [文档导航](README.md) · [全部支线](BRANCHES.md)

## 2026-10-27：处置盲分支分类与三模块主线

根据最新决定，模块一固定为同一骨干模型的处置盲组内分类、单温度校准和概率／间隔弃权。模块二仅以独立金标最小对学习方向，模块三在线生成规则正负增益并按设计平移键。方法名 ASTER 的展开改为 Action-blind Selection, Then Eigen-subspace Re-routing。

主稿、模块一确定版、六项主张、实验计划、证据状态和导航同步更新。新增 `src/aster` 模块一与增益接口、命令行入口、示例及 26 项契约测试。测试没有使用预训练模型；模块二、实际模块三和多层树运行器仍待实现。保形集合、LTT、证据员与原子概率融合不再属于默认方法。

本次修改以 [781ee863](https://github.com/chunxi-alpc/TRIAGE/commit/781ee863784c60f1f6710ba5eede22bc8ee6b93c) 为父版本。该固定提交保留改版前全文及图稿。历史源代码、配置、实验和结果不覆盖；新版入口与旧 CARE／V6 接口分开标识。完整范围见[实现状态](ASTER/implementation-status.zh.md)与[设计决策](ASTER/design-update-20261027.zh.md)。

## 2026-10-26：ASTER 文档成为主线

依据本次上传的完整草稿与六项主张，主线改为处置盲判与适用性门控注意力。新文档位于 [docs/ASTER](ASTER/README.md)。主稿保留方法和实验设计，经验性结果改为待测；原稿原样存档。新增四幅图及可复现绘图、算术核对脚本。

整理以提交 [0a831d1](https://github.com/chunxi-alpc/TRIAGE/commit/0a831d1127c338119bc11712677170486d40aabb) 为父版本。该提交已保存为归档分支 [archive/pre-aster-main-20261026](https://github.com/chunxi-alpc/TRIAGE/tree/archive/pre-aster-main-20261026)。原有 38 条非主线分支保留；索引记录整理时的每条分支及其固定提交。

CARE、ACRE 与原三模块叙述转为历史支线；旧顶层文档提供归档跳转。现有源代码、配置、输入示例、实验预测、指标和校验清单继续保存，本次不重写这些实验资产。索引中的历史内容按原阶段解释。

## 2026-10-26 核验范围

检查原稿字节一致性、文档链接、表格结构、占位表的派生算术、图中公式与假定参数、SVG／PDF／PNG 渲染及提交内容范围。未运行模型，未核实试点数据，未声称完成锁定测试或统计显著性验证。具体发现见[原稿审计](ASTER/audit.zh.md)。

## 更早的整理记录

2026-10-19／20 的版本来源、A1／V6 关系及历史检查记录见[原文固定版本](https://github.com/chunxi-alpc/TRIAGE/blob/0a831d1127c338119bc11712677170486d40aabb/docs/REPOSITORY_STATUS.md)。

