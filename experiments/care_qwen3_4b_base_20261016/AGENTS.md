# Codex 项目指令：CARE 三项观察预实验

## 任务目标
执行 `CODEX_TASK.md`，只在 `Qwen/Qwen3-4B-Base` 上跑三个**真实**观察实验，基于真实模型输出绘制论文级图表。不得自动替换其他模型；如 4B 无法观察到足够先验冲突，应客观记录结果和样本筛选情况。不需实现完整 CARE 或训练大模型。

## 必须遵守
1. **绝对不得编造实验结果**：无 GPU、权重下载失败、方法不兼容、无真实冲突样本时，要将任务标记为 `blocked`，保留日志，不得生成假数据图；`figures/templates` 仅是空白版式参考，不能在论文中当作结果。
2. **先阅读** `EXPERIMENT_PROTOCOL.md`，然后按 `CODEX_TASK.md` 顺序实施。首先 `git clone https://github.com/waylonli/SEKA.git external/SEKA`，记录准确提交哈希、依赖版本和模型修订版。不要默默更换 SEKA 算子。
3. `data/derived/pilot.jsonl` 是**非真实临床**的规则逻辑测试数据。可用于 O2/O3 工程探索；O1 的“模型先验冲突”只能使用满足闭卷筛选及来源审查的数据。若未获足够合格样本，明确报告无法验证 O1。
4. 数据必须按 `group_id`（同一规则模板的全部样例）划分投影构建、开发、测试，不能把同一规则家族的变体跨集合。模型选择、头和超参只用开发集，不能按测试集效果挑头。
5. **O1**：至少无干预、官方 SEKA、官方 PASTA（不兼容时标注 `PASTA-equivalent` 并验证数学等价）；相同输入、高亮 span、候选动作和可比层头。区分 `A_app`、`A_non`、误用率、先验回退；必须报告 eligible 样本数量。
6. **O2**：对同一规则固定文本，分别控制问题相关性和患者适用性，至少 2×2；患者变更带来的数字扰动可能是混杂因素，须有 same-side 值变化作为对照。读取同词元同层同 KV 头的 Key；PCA 与线性探针两者都做。
7. **O3**：明确对齐 RoPE/QK-norm、GQA 头映射；原始 SEKA 的 `k_norm` hook 在 RoPE **之前**，公式中的 `z` 和 `delta_a` 必须在与真实 attention logits 一致的坐标系中计算。比较 `lambda*omega*z` 和固定 Q 下的实际 QK logit 差异，进行数值测试。不要将 `P^-` 解释为临床负权重。
8. 输出：`results/obs{1,2,3}/`、`figures/` 内实测 PNG+SVG、`results/manifest.json`、`results/observations_summary.md`、测试日志和无法完成项目的 blocker 清单。
9. 不泄漏答案：不编辑候选项位置；不能把规则适用性标签直接写入输入（除作为额外有标签基线）。不使用测试集为 SEKA 建投影。
10. 只使用模拟病例或许可范围内的数据，不提供个体医疗建议。所有输出须注明是不是 synthetic-only。

## 代码质量
- Python 3.10–3.12、PyTorch、Transformers 与已有库；类型标注、可复现 seed、日志及错误检查。
- 不得用 `except: pass` 隐瞒方法运行失败；失败样例写明理由。
- 推断优先 `torch.inference_mode()`；4B 显存不够时优先 BF16、batch=1、缩短输入、按层分批记录 Key/Query 并及时转存 CPU；不能自动更换模型或使用随机权重冒充预训练模型。
- 绘图只从保存的原始结果读取，不在绘图函数里合成假数值；模板单独保存 `figures/templates/`。

## 模型固定约束
- 本轮仅使用 `Qwen/Qwen3-4B-Base`；`model.config` 应报告 36 层、32 个 Q 头、8 个 KV 头、每头 128 维。遇到实际配置不一致应停止并核查修订版本，不得把其他模型的硬编码沿用到 4B。
- 建议单卡显存 24 GB 或以上（BF16，batch=1）；权重本身约 8 GB，采样长度、KV cache、临时张量和表征记录还需额外空间。
