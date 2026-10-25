# ASTER-SEKA 最小阶段封装（实现准备）

日期：2026-10-25。状态：代码已写；本代理没有运行本地或远程测试，没有启动 GPU，没有使用开发/测试标签选参数。G1 尚未确认通过，不能标记方法实验已完成。

`aster_seka_adapter.py` 复用已加载的官方 `SEKALLM` 或 `AdaptiveSEKALLM`，没有重新实现谱编辑算子。源码依据为 `waylonli/SEKA@aab9cfed80dc012fb9e83f8bdefb8675c33f40e7` 的 `src/model/seka_llm.py` 与 `adaptive_seka_llm.py`。调用其真实 `generate(ids, steer, steer_mask, attention_mask, return_raw, ...)` API。

## 本次具体增量

- 两阶段独立输入结构。判断阶段只接收病史、逐条条件及自动选择器的 span；结构中没有候选动作、选项、完整指南或金标字段。执行阶段必须保留原始病史、原始问题、原指南、预测 T/F/U 和原始公开输出选项；信息不会被压缩为 T/F/U。多条规则的分数不跨规则 softmax；unknown 不等于 false。
- 严格字段校验可拒绝结构化 gold/action 数据混入判断阶段，但不能证明自由文本条件里没有处置泄漏。规则提取器、span 选择器与真实语义盲判还需独立实现和审查，不能用本封装的 schema 测试替代。
- 条件/执行分别开关，形成 2×2。执行阶段的固定 T/F/U 比较必须由上层复用同一 `ExecutionRequest`；本封装不偷偷重新预测。
- 原生单选、多选集合和路径生成均保留文本输出；没有把多选或路径任务变成 argmax，也没有内置未经核实的评价器。
- 输入不插入高亮标记，直接传词元 mask 给官方算子。所有配置使用同一无新增标记文本和相同 tokenizer 参数；原指南本来的 Markdown 保留。高亮按字符区间与词元相交关系映射，共享/重复 span 自然去重；这是明确的输入适配，不宣称与上游 marker 边界启发式完全一致。
- 检查保存的 layer 顺序与实际选层完全相同，检查 Qwen3 的 query/KV 头整除关系、投影形状、有限值及 dtype。SEKA 清单路径必须与加载配置一致；AdaSEKA 按专家顺序核对实际已加载的 U/S 张量。AdaSEKA 上游用整头 NaN 表示被筛掉的头，本封装保留并统计这种标记，只拒绝部分 NaN 或 Inf。

## 对照语义

`no_hook` 真正关闭 hook；`matched_zero` 保留官方计算路径而将全部增益设为 0；`reverse` 只翻转增益，保持投影；`wrong_span` 使用自动生成且与原 span 不交的等词元数位置；`misaligned_rule` 使用上层明确排列错位的规则 span。后两种仍使用原 prompt 和相同预算。

错位/错误 span 需由上层提供，封装验证 span 来源、规则到 span 的关联改变、最终 mask 改变及高亮词元数匹配。允许同一规则集合内部重新关联。若全规则支持导致 mask 不变，该对照不构成实际干预，显式报错并保留病例，不能从结果分母删去。上层需记录控制构造失败。封装不把随意选择的文本包装成临床规则解析结果。

无预测支持时保留全部原输入并记录 no-support fallback。该病例没有实际 hook；不能把它计为 matched-zero 数值验证成功。所有调用结束或异常后清除 hook、恢复强度，避免阶段间状态污染。AdaSEKA 固定单例调用，避免上游批内首例路由共享问题。

## 输出与待完成项

每次返回生成文本、输入/词元/mask hash、来源 commit、投影 hash、选择器指纹、阶段开关、增益、token 数、耗时、dtype 和生成 seed。checkpoint 完整权重 hash、数据 hash、峰值显存、总调用账本与下游指标由协调器运行清单补齐。当前输出默认贪心和原始文本提示；将来若增加 Qwen chat template，必须冻结新 prompt 版本并对各配置统一使用，不能悄悄改变。

需要数值比较时可设 `capture_first_token_scores=True`，额外返回 CPU tensor `first_token_scores`。上层将该张量另存数值文件后再序列化 JSON 清单。它是 Hugging Face generation scores（经过已配置 logits processor），不冒称未经处理的模型 logits。no_hook/zero_hook 必须保持相同 generation config；开启 scores 会增加显存和耗时，三条路径须一致开启并记录成本。

`test_aster_seka_adapter.py` 提供 15 个合成 CPU 契约测试，覆盖字段隔离、2×2、同输入 no-hook/zero-hook、反向与状态恢复、等词元数错误/错位控制、层匹配、AdaSEKA 合法筛头标记与部分 NaN 区分、多选/路径契约、无支持回退及异常清理。测试使用 API 替身，因此只能验证调度契约。尚未执行这些测试，也尚未证明真实 Qwen 的 RoPE/GQA/cache/padding/数值一致性；后者必须在研究服务器实际 smoke 后报告。

服务器执行契约测试命令：

```bash
cd /path/to/method_agent
python -m unittest -v test_aster_seka_adapter
```

真实 smoke 只需将 `operator` 换为官方已验证、同 checkpoint 投影的 `SEKALLM` 实例，构造合成 `ConditionRequest`，用同一 PreparedPrompt 依次运行 no_hook / matched_zero / normal。不能把此 smoke 的模型输出当临床性能结论。
