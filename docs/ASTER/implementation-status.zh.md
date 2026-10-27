# 论文设计与既有实现的关系

[主线索引](README.md) · [历史支线](../BRANCHES.md)

2026-10-26 的主线整理仅涉及文档与图稿。2026-10-27 新增独立的离线模块二和 Qwen3 缓存位移适配器，见[方法说明](module2-applicability.zh.md)与[预实验记录](../../experiments/module2_applicability_20261027/README.md)。这仍不是完整 ASTER 阶段掩码、证据员和端到端系统的验收；旧三模块也未被自动转换为新方法。

| 部分 | 仓库已有入口 | 与本稿的关系 |
| --- | --- | --- |
| 离线适用性子空间 | [模块二](../../src/module2/README.md) · [预实验](../../experiments/module2_applicability_20261027/README.md) | 新增开发树最小对、旋转前键差分、留出选头和空集合拒绝；不使用模块一伪标签 |
| 新缓存位移适配器 | [applicability_shift.py](../../src/module3/applicability_shift.py) | 首答案查询前编辑指定规则键；与直接分数偏置对照；Qwen3 架构验证，不等于完整 ASTER 实现 |
| TRIAGE／CARE 模块一、二、三 | [模块一](../../src/module1/README.md)、[模块二](../../src/module2/README.md)、[模块三](../../src/module3/README.md) | 历史适用性、定位与投影干预实现；保留原接口和复现用途 |
| 旧统一流水线 | [流水线说明](../../src/pipeline/README.md) | 原版本契约不因文档主线调整而自动变成 ASTER 两阶段协议 |
| ASTER 早期反事实实验 | [2026-10-22 目录](../../experiments/aster_medguide_20261022/README.md) | 既有 A/B/C 构造与实验入口；不是本稿五卷、证据员与阶段控制的完整验收 |
| 新论文设计 | [主稿](manuscript.zh.md) | 待实现核对、开发验证与锁定测试 |
| 设计图生成 | [绘图脚本](../../scripts/generate_aster_figures.py) | 可复现生成图稿；不运行模型，也不读取占位实验成绩 |
| 原稿算术核对 | [审计脚本](../../scripts/audit_aster_placeholders.py) | 仅检查原稿已显示数字的一致性；不是性能评测程序 |

旧实验的预测、指标、配置和负结果保持原样；主线导航仅重新标明其身份。后续若复用代码，需要逐项验证角色与阶段掩码、间接泄漏、分支编号概率读出、多组规则、缓存和数值边界。不能仅将旧类或旧结果改名为 ASTER。
