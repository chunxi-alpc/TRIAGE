# Codex 执行任务：CARE / SEKA 三个动机观察（4B 预实验）

请自动执行以下任务，在可联网、有合适 CPU/GPU 的开发环境中实现、运行、测试、绘图。**先保证科学有效性，再追求出图**。详细协议见 `EXPERIMENT_PROTOCOL.md`。已有 `scripts/build_minimal_pairs.py`、`scripts/plot_observations.py`、单元测试和 YAML 配置；补齐真正的模型执行器并运行，不要把版式模板当结果。

## P0. 环境与来源
- 安装 `requirements.txt`；安装稳定版 PyTorch（匹配 CUDA）。下载 `Qwen/Qwen3-4B-Base`，先做 `model.forward` 烟雾测试。优先 BF16 CUDA；若设备不支持 BF16，应显式报告并评估 FP16/FP32 资源要求，不要静默改成量化模型。官方 Qwen3-4B-Base 配置为 36 层、32 Q 头、8 KV 头、head_dim=128。模型参数 BF16 约占 8 GB，建议单卡 24 GB 或以上用于完整实验，batch=1、约 1024 输入 token 作为初始设置（超出则逐项测显存）；不能将此数字当作保证。
- 克隆 https://github.com/waylonli/SEKA ，阅读 `src/model/seka_llm.py`、`src/model/projection_builder_base.py` 和 `pastalib/pasta.py`；在 manifest 写 `git rev-parse HEAD`、模型 revision、transformers/torch 版本、硬件、seed、配置、时长。
- SEKA 的现有 `SEKALLM` 在 Qwen3 的 `k_norm` 注册 forward hook。检查 Qwen3 `QK-Norm -> RoPE -> attention` 的实际流程，记录所用 Transformers 实现版本。GQA：4B 32 查询头、8 KV 头，明确 4:1 对应，每个 KV 头映射到 4 个 Q 头；头数不许硬编码为旧模型值。
- 默认无需模型训练；仅使用 `Qwen/Qwen3-4B-Base`。如 O1 合格的真实先验冲突样本不足，将 O1 标为 `inconclusive` 或 `not_testable`，不得擅自使用其他模型、构造虚假临床冲突或者事后筛选“想要”的结果。若显存不足，先减少 batch、长度、按层采样和 CPU 卸载，并保留耗时/显存测量。

## P1. 构建对照数据（先完成）
- 运行 `python scripts/build_minimal_pairs.py --n-groups 60 --seed 42 --out data/derived/pilot.jsonl`，得到 60 个虚构规则组，每组 4 个状态。验证分组、gold、患者属性、字符 span、互斥动作无误。它们仅供机制测试，不能证明真实知识冲突。
- 加载 MedGUIDE (`MedGUIDE/MedGUIDE-MCQA-8K`) / MedCalc (`ncbi/MedCalc-Bench`) 作为临床候选源。**先审查格式和授权**，不能假定 `path` 是完整可执行树。人工核验或符号可执行的规则才能进入真实临床数据。可输出 `data/curated_candidates.jsonl` 供用户审查，未核验者不得标为真实冲突。
- O1 的重要组：同一规则片段 `R*` 的**适用**和**不适用**患者对，另有两个分支动作的反事实修改。应避免不适用病例的标准答案等于闭卷答案而造成先验回退“白拿分”。
- 闭卷先验筛选：同模型、不提供指南、相同病人及备选动作列表，记录 `y_prior`/其置信情况；只有 `y_prior != y_guideline` 且预测稳定、经来源核验的样本才进入“真正先验冲突”子集；同时报告总体条件分支数据结果和筛选通过数量。
- 采用按规则模板或临床指南 ID 分组切分（参考配置 60/20/20），用于谱投影/超参/测试。不得把最小对照对分开。

## P2. O1：忠实–误用权衡
- 至少运行 `baseline`、`SEKA`、`PASTA` 三种方法。相同目标是被修改的 `R*` **动作片段**，不包括候选选项。可以额外运行 `whole-guideline` 高亮消融。
- 使用官方 SEKA projection builder 适配本项目三元提示生成投影：语境 `R` 与相关/无关问题构成同规则对照；只能用 train 组。官方 SEKA 的正/负投影都从统计得到，禁止把原始负投影误称为不适用规则投影。
- PASTA 头选择与 SEKA 的 KV 头对齐：每个 KV 头对应一组 Q 头，必须解释匹配方式。PASTA 在当前 Transformers 版本不兼容时，应显式修复或实现公式等价的 `PASTA-equivalent`，并验证与原公式输出一致；不能报告成官方 PASTA。
- 候选判定：对 A/B/C/D 使用受控词元概率或完整字符串条件对数似然，记录策略并在所有方法中保持一致；别混用自由解码和受限选择。至少记录 `group_id,case_id,method,subset,correct,misapplied,prior_reversion,prior_eligible,split,model,seed`。
- `results/obs1/predictions.csv`；按规则组 bootstrap 95% CI。O1 主图：`A_app` 对 `A_non` 的散点及 `ΔA_app, ΔA_non, Δmisapplication` 点估计/区间图。
- 拒绝“强行得到权衡”：若无统计支持，summary 明确 `not_observed`；当真实 prior eligible=0 时 `not_testable`。

## P3. O2：Key 表示的相关性与适用性
- 同规则同片段、**2×2** 问题相关性×患者适用性：`RA`相关适用、`RN`相关不适用、`IA`无关适用、`IN`无关不适用；全部四种在规则之前提供问题和病例，保证因果可见。
- 增加 `same_side` 数值扰动对照（仍适用或仍不适用）以区分数字变化与适用性跨阈值变化。
- Hook `k_norm` 输出 Key（**pre-RoPE**），记录关键动作片段完全相同的词元，位置偏差与长度；对齐失败须跳过并记录。按同一 token span 聚合规则级 Key，或保存 token-level 并复现论文的成对 arrows。
- 初次扫所有 36 层×8 KV 头（每头 128 维）；应边抽取边流式写盘或转到 CPU，严禁将所有前向激活常驻 GPU；选图示层头时只能用 dev 数据的相关性稳定度指标。选定后再用 test 数据绘图。
- 输出 `results/obs2/key_embeddings.npz`，字段：`keys[N,4,L,H,D]`、`group_ids[N]`、`layer_ids[L]`、`states=['RA','RN','IA','IN']`、`split[N]`；另存 `layer_head_stats.csv`，包含相关偏移幅度、适用偏移幅度、方向一致性、跨规则线性探针 AUROC。避免同规则 train/test 混放。
- 画 `figures/O2_pca_pairwise.png`（两面板共享 PCA 坐标，箭头）、`figures/O2_layer_head_heatmaps.png`。图注不可声称 PCA 分离即因果证明；若适用性可解码，明确报告。

## P4. O3：连续权重与注意力方向不一致
- 使用 **train** 构造的谱相关性投影 Π，测试组用 oracle `s,c,r` 得到 `ω`（O3 初期允许 `c=r=1` 与非退化连续 `c,r` 两组，只比较是否幅度相关）；仅对非零目标 tokens 统计。
- 记录 `q_T` 与 `k_t`：对指定头按当前 Transformer 实现完成 QK 归一化、RoPE 及 GQA 头映射。设 `u_t=Πk_t`（projection 坐标系应与 Key 的定义一致），`z=q_T^T R_t(u_t)/sqrt(d_k)`；若 Π 已在 post-RoPE 空间构造，不得重复旋转。
- **固定原始 q_T** 做局部验证：原始 `a=q_T^T R_t(k_t)/sqrt(d_k)`，改 Key 得 `a'`，核对 `a'-a ≈ λ ω z`，float32 误差应达到预设容差；此处验证的是 attention logits 不是 softmax 概率，也不是最终模型完整因果效应。
- GQA：每个 KV 头有多个查询头，必须分别记录，而不是只记录组内平均值；模型配置若映射变化自动适配。
- `results/obs3/logits.csv` 必备列 `group_id,case_id,layer,kv_head,query_head,token_idx,omega,z,a_before,a_after,delta_a,lambda,split,model`。保存负 z 比例、`sign(Δa) != sign(ω)` 比例、按层头热图。
- 自动输出 `figures/O3_z_histogram.png`、`figures/O3_weight_delta_scatter.png`、`figures/O3_mismatch_heatmap.png`，标注有效样本数及零邻域处理。不应事先把所有 z 设为负或选择 z<0 的头来“证明”假设。

## P5. 图表与复现
- 调用 `python scripts/plot_observations.py --results results --out figures`。绘图脚本缺失数据时必须报 missing，不得自动生成随机数。
- `figures/templates/` 是**不含数据的版式图**；真正论文图放 `figures/` 根目录。每张出 PNG（>=250 dpi）+ SVG，英文坐标标题，显示 n/置信区间。
- 输出 `results/observations_summary.md`，每个观察写 `supported / not_observed / inconclusive / blocked` 和判据、样本数、限制；不要强行解释为成功。
- 执行 `pytest -q` + 模型烟雾测试 + 端到端跑通至少 2 组数据 + 路径及 token-span 的自检 + 公式误差验证。最终报告准确命令、Git commit/hash、模型 revision、时长、显存与缺失结果。

## 最终验收
- 三项观察均有原始数据、真实图、清晰的状态或阻塞说明。
- O1 不把“分支逻辑测试”错称“参数先验冲突”。
- O2 无 token 对齐泄漏、无看 test 选最漂亮 head。
- O3 logits 数学等价单元测试通过，pre-/post-RoPE 坐标无误。
- **不要训练大模型**；若得不到结果，详述客观原因，不编造或用示意替换。
