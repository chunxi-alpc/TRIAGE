# CARE 三观察：预注册式实验协议（先导版）

研究对象：具有条件及互斥分支的临床指南。样例输入 `Q（问题）+ X（患者）+ G（指南）+ D（决策指令）`。规则 `R=(condition, action, exception)`。

## 总体设计
- **模型**：统一固定 `Qwen/Qwen3-4B-Base`（36 层、32 Q 头、8 KV 头、head_dim=128），不自动降级或升级。它属于单个 Base 模型的探索性实验；阴性或阳性结果都不应外推为其他模型的普遍结论。
- **数据**：pilot generator 是虚构流程规则，**不含真实医学知识**；O1 的临床冲突结论仅来自合法来源、可执行规则、被闭卷筛选的子集。MedGUIDE 公开 path 不保证直接恢复整棵树。
- **重复性**：每个规则 group 下的版本和病例不跨 split；同模型、同输入顺序、同答案集合。训练/校准不更新模型权重，谱投影仅使用 train。
- **预注册状态**：supported/not_observed/inconclusive/blocked，不能根据结果改判据；默认探索性实验不预设统计功效。

## O1 忠实性–误用
- 最小反事实对：规则 `C -> A, otherwise B`；改写 `C -> C', otherwise D'`。每个规则有 `condition true`/`condition false` 的患者对，两个分支的推荐动作均与既有动作不同（双冲突控制）。高亮固定为被修改的同一个规则动作 span，跨患者不变。
- 真正的参数知识冲突必须筛选：闭卷预测 `y_prior`（无 G）；gold 在有 G 下由符号规则执行器确定；只把 `y_prior != y_gold` 的**有效且稳定**案例计入 `conflict_eligible`，其余称 conditional-rule-following。
- `A_app = mean(correct | rule applicable, eligible)`；`A_non = mean(correct | rule not applicable, eligible)`；`M_non = mean(selected_inapplicable_action | nonapplicable, eligible)`；`P_revert = mean(predicted == y_prior | eligible)`。
- `H_Acc = 2*A_app*A_non/(A_app+A_non)`（分母 0 时定义为 0），且必须报告两个分项。Bootstrap 按 group 重采样，使病例配对不被打散。模型基线、SEKA 和 PASTA 按相同分组比较，训练之外的超参固定。
- **判据**：某方法相对于 baseline 的 `ΔA_app>0` 与 `ΔM_non>0`，并报告不确定性；如果 `A_non` 下降但 `M_non` 不升，不可认定是规则误用。

## O2 Key 相关性与适用性
- 四状态：`RA`、`RN`、`IA`、`IN`，每个模板同一规则正文和目标 token。
- 在相关性上比较 `RA-IA` 与 `RN-IN`；在适用性上比较 `RA-RN` 与 `IA-IN`；交互作用 `(RA-RN)-(IA-IN)`。
- 提取 **pre-RoPE key** 与 SEKA builder 坐标保持一致，索引 target span 需 offsets 严格对齐。投影分析除了 PCA，还需显示距离（相同归一化标准）、位移一致性 `||sum δ|| / sum ||δ||` 与 GroupKFold linear-probe AUC。对数值变化做 same-side 对照。
- 每层头的统计不应根据 test 结果选择可视化目标；先用 dev 的 relevance-consistency 选择，再测试。
- **可反证**：若 Key 中有强适用性线性可解码信息，只能得出“内部已有适用性但最终输出未必利用”，不再声称“适用性不被编码”。

## O3 连续权重的方向不一致
- `k' = k + λ ω Πk` 是关键被测算子（原始 SEKA 的谱正负 P± 不对应规则适用性 ±）。
- 对被干预层和头、决策 query T：`u_t = Πk_t`，`z = (R_T(q_T)^T R_t(u_t))/sqrt(d_k)`，`Δa = λ ω z`。注意 `q_T` 可定义为已旋转后查询；不能额外重复施加 R_T。建议在代码中统一 `q_rope`、`k_pre_rope` 和 `u_rope` 名称。
- 计算 `sign_mismatch = (ω≠0 and z<0)`（近零阈值例外），并用 `a_after-a_before` 独立检查公式；只测单层局部固定 q，不能把跨层完整运行变化当代数等式。
- 公式等价性是实现单元测试；**核心实证量**为真实模型 Key 与真实 query 所诱导的负 z 频率及幅度。`P` 必须由 train contrastive 数据构造。
- GQA 时一个 KV 头对应多个 Q 头，逐头记录，均值只能作为额外汇总。

## 真实样本与局限
- 临床原始 guideline/病例的授权、原子条件正确性需审查。当前包不包含 NCCN 受限指南正文，不自动生成处方和剂量建议。
- 虚构数据有金标适用性但**没有保证具备参数先验冲突**，因此只能完成数据生成/表示变化/算术一致性预检，不能作为临床结论。
