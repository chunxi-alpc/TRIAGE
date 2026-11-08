# ASTER 模块与训练目标

[主页](../../README.md) · [方法变体](versions.zh.md) · [实验索引](experiments.zh.md)

表示编辑修改注意力头的输出；ASTER-CR 从冻结表示中学习判断分数的修正量。本页按实现说明两条路线。

## 表示编辑：M1、M2、M3

| 模块 | 做法 | 实现 |
|---|---|---|
| M1：头定位 | 归因预筛与双向精确修补；可学习 M3 和 P3 主配置复用既有 51 头 | [m1.py](../../experiments/aster_closed_form/m1.py) |
| M2：子空间估计 | 用配对激活差的未中心化二阶矩估计干扰基 U 和信号基 S；既有总秩分别为 244、356 | [operators.py](../../experiments/aster_closed_form/operators.py)、[bases.py](../../experiments/aster_m3_followup/bases.py) |
| M3：表示编辑 | 闭式版本用固定投影；可学习版本训练坐标映射和门控 | [m3.py](../../experiments/aster_learnable_m3/m3.py)、[training.py](../../experiments/aster_m3_followup/training.py) |

既有基的配置来自能量目标 0.90/0.95、每类秩上限 4/8/16 和 M3 夹角 15°/30°/45°的 18 组比较。开发指标并列后按既定规则选取 0.95、4、45°。连续衰减与参考投影继续使用这组基；闭式公式见[方法变体](versions.zh.md)。

### 可学习编辑与参考

```math
z'_h=z_h+g_hV_h(W_hr_h-V_h^\top z_h),\qquad
g_h=\sigma(a_h^\top[V_h^\top z_h;W_hr_h]+b_h).
```

| 代码名 | 编辑空间 V | 参考 r | 参数量 | 推理末位置评分 |
|---|---|---|---:|---|
| learned_reference | U | 原病历＋当前及祖先条件的无干预头激活 | 31,771 | 参考与主输入各一次 |
| learned_self | U | 完整输入的无干预头激活 | 31,771 | 同前缀两次 |
| paired_n | U | 本次前向当前头输出，停止参考梯度 | 31,771 | 一次 |
| paired_s | S | 同上 | 46,331 | 一次 |
| paired_ns | orth([U,S]) | 同上 | 78,051 | 一次 |
| online_n | U | 同上；仅用原病例监督 | 31,771 | 一次 |

只编辑最终位置的选中头，较早层的编辑会影响后层表示。W 初始化为 V 的转置，门控初值为 0.01。辅助参考使用原病历和条件；历史“同事实参考”则按删除规则构造提示，两者的具体输入见[方法变体](versions.zh.md)。

### 训练目标

单节点损失为 CE + 0.1 × 原模型正确指示 × KL + 0.001 × 平均头相对编辑量。KL 随全部节点／病例平均。原病例最多取首、中、末 3 个节点，病例内平均，累积 8 个任务后更新参数。

配对版本对两侧的完整单节点损失取平均，再增加关系损失：A/P/F_same 使用 JS，C/F 使用两侧标签定向的间隔损失，间隔为 1；各配对类型等权。`online_n` 和前一阶段两种 M3 使用原病例损失。完整公式见[主稿训练目标](manuscript.zh.md#55-训练目标与推理流程)。

### 算子性质

硬筛选闭式满足 ES=S 和幂等性；连续衰减保持纯信号方向。这里的“保持”指输入位于信号空间时保持原向量。全部方向正交投影与可学习编辑的性质见[版本对照](versions.zh.md#算子与参考对照)。可学习编辑的更新位于 V 所张成的空间，信号坐标变化量作为诊断记录。

## ASTER-CR：支持空间与分数读出

### 选头与支持空间

CR 在 repair-v2 的 fit 数据上重新选头和估计基。A/P 为处置文本／无关背景变化，C/F 为条件／事实变化，F0 为事实改变但标签保持的配对。

定位先按梯度与头激活差做归因，每类取 64 个候选头，再用双向精确修补排序。A/P 与 C/F 的头排名交替合并；头数沿用既有配置。实现见 [localize](../../experiments/aster_cr/mechanisms.py)。

将选中头的末位置输出拼接为 h。均值 μ 与逐维标准差 s 来自 fit 的 full 视图，标准差下限为 10⁻⁵。以标准化后的配对差做 SVD：C/F 得到信号基 S，A/P 得到初始干扰基 U，再移除 U 在 S 中的分量并正交化：

```math
\tilde h=(h-\mu)\oslash s,\qquad
S=\operatorname{basis}(\Delta_{C/F}),\qquad
U=\operatorname{basis}(\Delta_{A/P}),\qquad
N=\operatorname{orth}((I-SS^\top)U).
```

配对差按 s 缩放，SVD 使用未中心化差分矩阵。实现见 [supports](../../experiments/aster_cr/mechanisms.py) 与 [basis / orth_nuisance](../../experiments/aster_cr/features.py)。

H2 在开发集检查干扰坐标对分数变化的预测能力（R² > 0），以及 C/F 上的信号扰动比（≤ 0.1）。H3 在原模型错误节点上比较表示探针和打乱标签对照，用于决定是否启用信号分量。当前快照中 H2 通过、H3 未通过，读出使用干扰基 N。

### 读出公式

```math
x=V^\top\tilde h,\qquad
\ell'=\ell+w^\top x+b,\qquad
\hat y=\mathbf 1[\ell'>0].
```

ℓ 为基础模型的二分类分数差；V 取 N、S 或二者拼接，取决于诊断结果。模型与支持空间固定，训练 w 和 b。预选配置 `cr_r16_l0.01` 使用 16 维 N，共 17 个可训练参数。推理时从一次基础模型前向取得分数和头表示，再计算修正量。实现见 [project / optimize / predict](../../experiments/aster_cr/readout.py)。

### 训练与选参

读出用 LBFGS 优化，最多 90 次迭代。损失由五部分组成：

| 项 | 实现 |
|---|---|
| 分类 | fit 输入上的二元交叉熵 |
| 权重正则 | λ × mean(w²) |
| 保持型配对 | A/P/F0 各类平均的两端修正后分数差平方 |
| 翻转型配对 | C/F 各类平均的 max(0, 1 − (y右−y左)(ℓ′右−ℓ′左))² |
| 正确预测保持 | 原模型预测正确的 fit 输入上，原二分类概率到修正后概率的平均 KL |

后三项默认权重均为 1。候选秩为 4/8/16/full，正则系数为 0.001/0.01/0.1。先要求开发路径准确率下降不超过 0.5 个百分点，再按 A/P/C/F 配对宏平均和参数量选择。每对两端均正确才计正确，F0 单列。该阶段的配置和结果见[CR 开发分析](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md)。

### 普通读出与校准

| 对照 | 特征与训练 |
|---|---|
| CR 实验内的普通读出 | 同一组头的标准化表示做普通 PCA，使用分类损失与权重正则；其 `ordinary_r8` 和参数匹配配置分别有 9、17 个参数 |
| 偏置校准 | 只训练 b，ℓ′=ℓ+b |
| 独立普通读出 | 只用 761 个原始 fit 节点；最后一层残差标准化后做 64 维 PCA，训练 65 参数线性修正 |

CR 的支持空间来自配对差；实验内普通读出的 PCA 来自 fit 输入表示。独立普通读出另用末层残差和原始节点监督，结果见[独立报告](../../results/ordinary_readout_20261108/REPORT.zh.md)。
