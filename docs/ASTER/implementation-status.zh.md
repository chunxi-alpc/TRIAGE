# 当前主线的实现状态

[主线索引](README.md) · [模块一确定版](module1-spec.zh.md) · [代码说明](../../src/aster/README.md) · [证据状态](evidence-status.zh.md)

**更新：2026-10-27。** 保留当日主线新增的模块一、单温度拟合和规则增益接口，并加入独立的模块二学习及模块三缓存平移实现。Qwen3-0.6B-Base 的真实表示提取和单目标因果对照均已完成，1,152 条干预条件记录完整、无阻断或缺失。局部注意力发生变化，但全部预设剂量的受约束分支准确率保持 50%、配对全对率保持 0%；证据及范围见[结果报告](../../experiments/module2_applicability_20261027/RESULTS.zh.md)。模块一驱动的完整流水线仍未验收。

## 当前代码与完成范围

| 组件 | 新入口 | 状态与边界 |
|---|---|---|
| 条件视图与分支分类 | [action_blind.py](../../src/aster/action_blind.py) | 字段白名单、中性代理标签、完整合法 logits、温度缩放、概率／间隔／熵、弃权；真实骨干适配待实测 |
| 同骨干评分适配 | `HFLabelScorer(model, tokenizer)` | 注入已有模型；实际模板边界单词元验证；拒绝截断与思考接口混用；无模型下载 |
| 温度校准 | [calibration.py](../../src/aster/calibration.py) | 一个全局温度的开发集 NLL 拟合，记录搜索边界；没有自动阈值选择或风险认证 |
| 模块一到模块三 | [gains.py](../../src/aster/gains.py) | 对分布及摘要作一致性检查，输出规则正负或零增益；可达性由调用者提供 |
| 独立运行 | [run_aster_module1.py](../../scripts/run_aster_module1.py) · [输入示例](../../examples/aster_action_blind_input.json) | 本地模型前向入口；参数必须显式指定；不运行模块二或模块三 |
| 模块二数据与提取 | [minimal_pairs.py](../../src/module2/minimal_pairs.py) · [extraction.py](../../src/module2/extraction.py) | 开发树最小三元组、原指南划分、处置／顺序平衡、严格词元对齐、归一化后旋转前处置键与 GQA 有效查询 |
| 模块二估计与选择 | [applicability.py](../../src/module2/applicability.py) | 指南等权差分二阶矩、投影均值、留出读取、稳定性和空集合筛选；不接受模块一伪标签 |
| 独立模块三执行 | [applicability_shift.py](../../src/module3/applicability_shift.py) | 首答案之前的缓存平移、直接分数偏置及实际注意力诊断；本轮仅单层单目标、金标符号与 A/B 评分 |
| 等价后段执行 | [applicability_tail.py](../../src/module3/applicability_tail.py) | 固定单编辑层下复用更低层计算；逐提示核验全词表分数及注意力完全一致；不是多层／自由生成加速验收 |
| 新模块一连接与联合执行 | [方法边界](module2-applicability.zh.md) | M1→M3 适配、多目标联合编辑、多层联合编辑及完整自由生成仍待整合验证 |
| 多层树调度 | [模块一规格](module1-spec.zh.md) | 祖先条件进入提示；只沿确认路径激活下游组的调度器尚待实现 |

## 本次实际验证

```bash
python -m unittest discover -s tests -p 'test_aster*.py' -v
python -m scripts.run_aster_module1 --help
```

**26 项无模型测试通过。** 测试位于 [test_aster_action_blind.py](../../tests/test_aster_action_blind.py)。覆盖处置与选项变化时提示不变、病史与祖先条件保留、NONE／弃权区分、概率与熵、单词元边界、非法标签、禁截断、校准及规则增益。额外回归覆盖非法规则编号、矛盾摘要、极端有限分数的 NLL 汇总，以及共享模型被切回训练模式。

上述模块一测试使用可控分数和分词替身，验证程序契约。独立模块二增加 [34 项实现与统计测试](../../experiments/module2_applicability_20261027/results/tests.txt)及 [7 项后段执行测试](../../experiments/module2_applicability_20261027/results/tail_validation_tests.txt)。这些检查与真实预训练模型结果分别记录，不能互相替代。

模块二预实验使用 PyTorch 2.8.0+cpu、Transformers 4.57.6 和固定修订的 Qwen3-0.6B-Base。表示阶段完成 144 次真实前向，在 12 个开发指南的 96 个三元组上拟合、选择与确认，最终保留四个零起始第 17 层 KV 头。本轮使用普通因果掩码、患者在前、合成布尔事实与局部兄弟指南；没有运行新版模块一的真实分类或校准，也不构成其模型适配验收。完整模型／数据摘要、原始记录及统计见[独立实验](../../experiments/module2_applicability_20261027/README.md)。

输入中的 `mutually_exclusive=true` 与 `task_context_verified=true` 是调用者声明，不是自动语义证明。任意原始指南的条件／处置切分、患者文本中的建议泄漏检查、完整模型与数据指纹记录仍由数据适配器和实验运行器负责。`prompt_sha256` 只标识条件提示文本，不包含模型修订、聊天模板或分词器身份。

## 历史代码不能替代当前实现

| 历史入口 | 保存用途 |
|---|---|
| [src/module1](../../src/module1/README.md) | CARE／A1／V6 的原接口与复现；其默认工厂和命令仍按历史版本解释 |
| [src/module2](../../src/module2/README.md) 中的旧定位接口 | 保留旧组件构建；新增 `applicability` 接口单独标识，不能互换组件格式 |
| [src/module3](../../src/module3/README.md) 中的旧投影和门控 | 保留原实验算子；新的缓存平移入口与其分开，不能直接加载旧投影作为平移方向 |
| [src/pipeline](../../src/pipeline/README.md) | 旧 V6 图结构流水线；不接受新版概率接口作为等价输入 |
| [早期 ASTER 实验](../../experiments/aster_medguide_20261022/README.md) | 原实验协议、预测及负结果，保持原版本含义 |
| [旧图稿](figures/README.md) | 2026-10-26 的掩码／证据员／对数门控设计；图 2、图 4 已退出当前主线 |

新版模块一入口是 `src/aster` 和 `python -m scripts.run_aster_module1`；独立模块二与因果入口是 `scripts/run_module2_preexperiment.py` 及 `scripts/run_module2_causal_tail_pilot.py`。旧代码与冻结结果不更名、不覆盖，也不用于宣称新版联合效果已经完成。直接分数偏置继续作为同信息、同支持和匹配强度的必要对照。
