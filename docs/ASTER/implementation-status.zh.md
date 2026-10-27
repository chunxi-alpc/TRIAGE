# 当前主线的实现状态

[主线索引](README.md) · [模块一确定版](module1-spec.zh.md) · [代码说明](../../src/aster/README.md) · [证据状态](evidence-status.zh.md)

**更新：2026-10-27。** 本次完成三模块研究规格、模块一参考实现、单温度拟合和规则增益接口。没有运行预训练骨干模型，也没有完成模块二及真实注意力干预的端到端实现。

## 当前代码与完成范围

| 组件 | 新入口 | 状态与边界 |
|---|---|---|
| 条件视图与分支分类 | [action_blind.py](../../src/aster/action_blind.py) | 字段白名单、中性代理标签、完整合法 logits、温度缩放、概率／间隔／熵、弃权；真实骨干适配待实测 |
| 同骨干评分适配 | `HFLabelScorer(model, tokenizer)` | 注入已有模型；实际模板边界单词元验证；拒绝截断与思考接口混用；无模型下载 |
| 温度校准 | [calibration.py](../../src/aster/calibration.py) | 一个全局温度的开发集 NLL 拟合，记录搜索边界；没有自动阈值选择或风险认证 |
| 模块一到模块三 | [gains.py](../../src/aster/gains.py) | 对分布及摘要作一致性检查，输出规则正负或零增益；可达性由调用者提供 |
| 独立运行 | [run_aster_module1.py](../../scripts/run_aster_module1.py) · [输入示例](../../examples/aster_action_blind_input.json) | 本地模型前向入口；参数必须显式指定；不运行模块二或模块三 |
| 模块二 | [论文规格](manuscript.zh.md) | 金标最小对、差分谱估计、可读性选头与冻结方向尚待实现 |
| 实际模块三 | [论文规格](manuscript.zh.md) | 规则位置映射、归一化后／旋转前键平移、GQA、缓存隔离和退出恢复尚待实现 |
| 多层树调度 | [模块一规格](module1-spec.zh.md) | 祖先条件进入提示；只沿确认路径激活下游组的调度器尚待实现 |

## 本次实际验证

```bash
python -m unittest discover -s tests -p 'test_aster*.py' -v
python -m scripts.run_aster_module1 --help
```

**26 项无模型测试通过。** 测试位于 [test_aster_action_blind.py](../../tests/test_aster_action_blind.py)。覆盖处置与选项变化时提示不变、病史与祖先条件保留、NONE／弃权区分、概率与熵、单词元边界、非法标签、禁截断、校准及规则增益。额外回归覆盖非法规则编号、矛盾摘要、极端有限分数的 NLL 汇总，以及共享模型被切回训练模式。

测试使用可控分数和分词替身。它们验证程序契约，不证明 Qwen、Llama、Mistral 的真实模板、概率校准或分支准确率已经验收。本次环境没有可用的 PyTorch／Transformers 及预训练模型，因此未运行真实 HF 前向、临床性能或端到端实验。

输入中的 `mutually_exclusive=true` 与 `task_context_verified=true` 是调用者声明，不是自动语义证明。任意原始指南的条件／处置切分、患者文本中的建议泄漏检查、完整模型与数据指纹记录仍由数据适配器和实验运行器负责。`prompt_sha256` 只标识条件提示文本，不包含模型修订、聊天模板或分词器身份。

## 历史代码不能替代当前实现

| 历史入口 | 保存用途 |
|---|---|
| [src/module1](../../src/module1/README.md) | CARE／A1／V6 的原接口与复现；其默认工厂和命令仍按历史版本解释 |
| [src/module2](../../src/module2/README.md) | 旧定位与组件构建；不能视为本文金标最小对方向估计 |
| [src/module3](../../src/module3/README.md) | 旧投影和门控干预；不能直接加载为本文平移方向 |
| [src/pipeline](../../src/pipeline/README.md) | 旧 V6 图结构流水线；不接受新版概率接口作为等价输入 |
| [早期 ASTER 实验](../../experiments/aster_medguide_20261022/README.md) | 原实验协议、预测及负结果，保持原版本含义 |
| [旧图稿](figures/README.md) | 2026-10-26 的掩码／证据员／对数门控设计；图 2、图 4 已退出当前主线 |

当前新版入口是 `src/aster` 和 `python -m scripts.run_aster_module1`。旧代码与冻结结果不更名、不覆盖，也不用于宣称新版联合效果已经完成。
