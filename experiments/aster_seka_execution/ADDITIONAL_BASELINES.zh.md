# 原生路径附加基线

状态：`planned`。模块、入口和 15 项合成测试已写入，尚未运行。不会把旧选择题结果称作 MEGA 路径效果，也没有进行新测试集选参。代码只读取相同冻结 train_val 输入及主开发运行的规则/TFU 私有缓存；原生生成仍使用官方完整任务模板、原病史和统一输出预算。

## 旧结果与新生成实现的区别

PR33 的静态公式是 `(1-beta) * P0 + beta * S`，并分别对 P0、S 作温度变换；冻结参数为 beta=.75、T0=1、Ts=.25。旧 S 是**答案选项支持分布**。动态公平比较使用 `beta_t = clip(beta + .25 * (2 * margin(S) - 1), 0, 1)`，margin 在温度变换前计算。来源：[原公式](https://github.com/chunxi-alpc/TRIAGE/blob/a8c1e49c0fbda941282daf25908004407eb0b1bf/experiments/aster_dynamic_fusion_seed0_20261025/reproduce.py)、[公平动态比较](https://github.com/chunxi-alpc/TRIAGE/blob/a8c1e49c0fbda941282daf25908004407eb0b1bf/experiments/aster_dynamic_fusion_seed0_20261025/balanced_analysis.py)、[冻结参数](https://github.com/chunxi-alpc/TRIAGE/blob/a8c1e49c0fbda941282daf25908004407eb0b1bf/experiments/aster_dynamic_fusion_seed0_20261025/frozen_candidate.json)。旧结果未证明稳定的独立注意力增益，不能由本模块的存在改变该结论。

这里明确标记为**完整词表双路生成扩展**：

- 基础路：完整原指南、原病史、原任务问题和 A0 预测 TFU。
- 支持路：完全相同的信息，另重复 A0 预测为真的规则原文段。没有新答案、参考路径或额外指南；无预测支持时原样使用基础路。
- 每个输出 token 分别计算两路完整词表分布。静态/动态用上述混合公式，动态 margin 改为支持路完整词表最高两个 token 的原始概率差。支持温度不参与 margin 定义。
- 两路拥有独立 KV cache，每步把**同一个混合后选出的 token**送入两路。生成正常的完整路径文本，不把动作选项概率硬映射为路径。
- 默认参数是历史参数迁移，**不是已经完成 MEGA 温度校准**。若将来提供校准配置，必须同时提供独立的 train_val calibration manifest；入口核对哈希和指南组不重合。这里不拟合配置，也不读裁判参考或测试结果。

## 可执行方法

| 方法名 | 实际行为 |
|---|---|
| `text_two_stage` | 复用主运行 A0B0 的真实路径输出，注明来源和哈希，新生成成本为 0，原运行成本仍计入方法比较。 |
| `text_emphasis` | 同信息完整指南+TFU，再强调预测支持原文，一路生成。 |
| `top1_crop` / `top2_crop` | 仅按已保存 A0 的每规则 P(T) 排序，保留前 1/2 个完整规则段；不看正确答案。 |
| `applicable_crop` | 所有条件已确定时保留预测 T 的规则段；有 U 或没有支持时保留原完整指南。 |
| `fusion_static` / `fusion_dynamic` | 完整词表、两路 cache、每 token 两次真实模型前向。 |
| `fusion_static_zero` / `fusion_dynamic_zero` | 仍运行两路和同一公式，但混合权重为 0；与相同基础温度的原分布比较。 |
| `cad` / `cad_zero` / `cad_reverse` | `(1+alpha)*context_logits-alpha*prior_logits`，alpha 分别为 1/0/−1；先验路移除指南，保留同一原病例、任务模板与 TFU，生成共享后缀。关闭候选筛选。 |
| `spotlight` 及控制 | 论文公式的 query 分块实现，完整上下文和 KV 保留；见下节。 |
| `direct_action_mapping` | 仅有一个预测为真、无显式例外且其余条件不为 U 的已引用规则时，输出单步条件→原文动作及唯一原文结论。否则保留 `mapping_unresolved` 空输出，不杜撰多节点顺序或“指南不适用”。这只是未验证提取映射的机械基线，不是已恢复的临床决策树执行器。 |

裁剪是有信息删除的基线，不是同信息注意力对照。直接映射的覆盖和失败率必须按全部固定病例报告；不能只计算能映射的子集。没有规则真值或例外判断真值被加入算法。

## 长上下文 Spotlight

已知本批原始输入范围为 9,187–55,145 token，不能使用完整 `[heads,Q,K]` eager 矩阵。`chunked_spotlight.py` 将 query 固定分为 128 个 token 一块，每块保留全部 key/value，按原论文逐 query/head/layer 计算证据质量，低于 .1 时加 `log(.1/current)`。这不删 token，不近似为组级偏置，也不改成强制达到目标的 log-odds 公式。

`spotlight_no_hook`、`spotlight_zero`、正常、错误跨度和反向全部使用相同分块 attention 函数。这里 no-hook 的准确含义是**同分块内核、不加证据偏置**；不能宣称等同于原 SDPA 内核。对小合成 Qwen 写了分块/原整块公式、零强度/同内核无偏置、缓存和原 stock eager 的数值测试，等待服务器执行。临床效果只与同内核无偏置比较；SDPA 到分块 eager 的差异单列。

上游 Qwen 的共享 `[1,1,Q,K]` 因果 mask 仍是二次内存，55k 时 BF16 理论约 6 GiB；一块 32×128×55k FP32 分数理论约 .84 GiB。还需权重、KV、激活和临时张量。它减少内存峰值，不减少注意力的二次计算量；这些尺寸不是实测显存或耗时。运行器不输出完整 attention 矩阵，只记录层数、块数和实际最大分数张量元素数。

主运行的 YaRN 配置自动继承，当前为 factor=2.0、original_max_position_embeddings=32768、rope_type=yarn，总上下文 65,536；输出上限 1,024。所有方法、负对照、两条解码路径使用相同运行时位置编码配置，并另写实际模型配置与指纹。标准 CounterFact 程序不受影响。

## 数值、失败与成本

双路生成保留 Hugging Face 主路径生成循环和停止条件。临时 forward hook 在任何 logits processor 之前复制原始分数，逐步核对与默认 generation scores 一致；若模型存在非恒等 processor，明确报错，不能默默改 CAD 公式。输出后核对实际生成 token 与混合决策完全一致、实际前向次数为 token 数的两倍。两路都计算完整词表，没有只算候选动作。

零融合使用 log 概率分数，no-fusion 原分数可能相差任意常量，因此数值控制比较**归一化后的完整词表 log 概率**和 argmax，不把常量平移当作漂移。每个含零融合/CAD 控制的病例另运行一次单 token 基础参考，成本写入 `numeric_probes.private.jsonl`。Spotlight 控制若未显式列入无偏置臂，入口自动运行一条相同预算的分块无偏置参考，并完整记录其额外成本。

每例记录完整生成文本、全部输出 token、每步融合权重/差距/选定 token/两路 argmax、首步数值控制、实际前向数、耗时、显存和所有来源指纹。模型异常的已发生前向和部分双路生成记录也保留。主运行抽取、A0 条件判别成本通过文件哈希和缓存键关联，不能只报告附加生成耗时。所有失败行保留固定分母；程序不计算临床准确率，也不自行标 `complete_verified`。

## 服务器入口

协调器先验证合成测试并汇报资源，再启动选定方法；没有默认偷偷运行全部 18 种。输入 `PRIMARY_RUN` 必须是真实主开发运行，不能用计划目录代替。

```bash
cd "$CLINICAL_DEV"
PYTHONPATH="$METHOD_AGENT:$BASELINE_ADAPTERS:$PYTHONPATH" python -m unittest -v test_native_baselines

CUDA_VISIBLE_DEVICES=0 python run_mega_additional_baselines.py \
  --dataset-root "$MEGA_DATA" --input-manifest "$RUN/trainval_manifest.json" \
  --official-root "$MEGA_CODE" --primary-run "$PRIMARY_RUN" \
  --model "$MODEL" --model-fingerprint "$MODEL_FINGERPRINT" \
  --adapter-dir "$METHOD_AGENT" --reader-dir "$DATA_AUDIT" --baseline-dir "$BASELINE_ADAPTERS" \
  --methods text_two_stage text_emphasis top1_crop top2_crop applicable_crop \
    fusion_static fusion_dynamic fusion_static_zero fusion_dynamic_zero cad cad_zero cad_reverse \
    spotlight_no_hook spotlight spotlight_zero spotlight_wrong spotlight_reverse direct_action_mapping \
  --output-dir "$RUN/clinical_extra_seed0"
```

独立评价仍使用已核对的官方单轮 parser 和路径/结论裁判；每行包含其必需的 language 与身份字段。私有输出和指南引用不提交 PR。只有实际生成、解析和评价完成后，才能给出哪个方法最优以及是否有独立注意力增益。
