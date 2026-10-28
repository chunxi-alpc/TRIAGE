# ASTER 实验评价指标与统计口径

[论文工作稿](manuscript.zh.md) · [实验计划](experiment-plan.zh.md) · [结果表](experiment-tables.zh.md)

**版本：2026-10-28。状态：评价协议，尚非实验结果。** 本文将答案正确性、分支适用性、路径忠实性、选择性干预与计算成本分别评价。除明确引用的既有指标外，以下组合和诊断指标是本研究的预定定义。

## 1. 统计单位与输出处理

以 $b$ 表示病例底稿，以 $t$ 表示来源指南或决策树，以 $v\in\{\mathrm{NC,IC,AC,SW,CS}\}$ 表示试卷。$D_v$ 是在运行模型前锁定的该卷纳入集合，$N_v=|D_v|$。同一底稿的变体保留配对关系。同一指南的所有病例保留聚类关系。

令 $y_i$ 为根据所给规则得到的答案，$\widehat y_i^m$ 为方法 $m$ 的预测，定义

$$
c_i^m=\mathbf{1}\{\widehat y_i^m=y_i\}.
$$

输入不符合预先规定的数据质量标准时，在冻结清单中记录排除原因。模型运行失败、超时、无法解析或输出非法答案均保留在已锁定分母中，答案正确性记为 0，并单列失败率。未实现或尚未运行的方法写“待填”，不能写 0。金标缺失的项目写“不适用”，不能用猜测标签计算指标。

MedGUIDE 采用单选答案完全匹配。MedPIC 使用选项集合：只有 $\widehat Y_i=Y_i$ 才正确；选项顺序不影响匹配，漏选或多选均为错误。答案集合归一化器在测试前冻结。不能用“命中任意一个正确选项”替代集合完全匹配。[MedPIC 原文](https://arxiv.org/html/2608.03028v1)

所有准确率以 0–1 值计算，表格乘以 100 展示。两种准确率之差的展示单位为百分点。训练重复运行报告均值与种子标准差；测试抽样不确定性另报告置信区间，二者不混写。

## 2. 五卷答案正确性

### 2.1 分卷准确率与平衡指标

$$
\mathrm{Acc}_v^m=\frac{1}{N_v}\sum_{i\in D_v}c_i^m.
$$

原生及不适用处置改写构成非冲突评价，适用处置改写、处置交换及条件漂移构成冲突评价：

$$
\mathrm{NCA}^m=\frac{\mathrm{Acc}_{NC}^m+\mathrm{Acc}_{IC}^m}{2},
\qquad
\mathrm{CA}^m=\frac{\mathrm{Acc}_{AC}^m+\mathrm{Acc}_{SW}^m+\mathrm{Acc}_{CS}^m}{3}.
$$

$$
H^m=\frac{2\,\mathrm{NCA}^m\,\mathrm{CA}^m}
{\mathrm{NCA}^m+\mathrm{CA}^m}.
$$

两项均为 0 时规定 $H=0$。三者均越高越好。$H$ 用于衡量两类表现的平衡，不能代替逐卷结果，也不能单独证明原生能力保持。若某一整卷不可构造，则 $H$ 暂不定义，不擅自将五卷公式改成四卷公式。

主表采用分卷宏平均。另报告按题数加权的微平均与各卷分母。计算使用未取整的原始计数，最后统一保留两位小数。

### 2.2 答案退化率与纠正率

令 $D_0=D_{NC}\cup D_{IC}$，$m=0$ 表示同一骨干的未干预基线。定义无条件答案退化率（历史稿称“伤害率”）与纠正率：

$$
\mathrm{HR}^m=
\frac{\sum_{i\in D_0}c_i^0(1-c_i^m)}{|D_0|},
\qquad
\mathrm{CR}^m=
\frac{\sum_{i\in D_0}(1-c_i^0)c_i^m}{|D_0|}.
$$

HR 越低越好，CR 越高越好。“伤害”只表示本评测中由对变错，不能解释为临床不良事件。相同样本及权重下有

$$
\mathrm{Acc}_{D_0}^m-\mathrm{Acc}_{D_0}^0=\mathrm{CR}^m-\mathrm{HR}^m.
$$

这是微平均恒等式。两卷不等大时，它不能直接替代分卷宏平均的 $\Delta\mathrm{NCA}$。另报条件退化率 $\sum c_i^0(1-c_i^m)/\sum c_i^0$，必须标明其不同分母；基线全错时该条件率未定义。

### 2.3 诱饵采纳率

令 $d_i$ 为 SW 构造时预先指定的诱饵答案标签，其对应处置与金标处置不同；$d_i$ 与 $\widehat y_i^m$ 使用相同的答案编码，则

$$
\mathrm{DA}_{SW}^m=
\frac{1}{N_{SW}}\sum_{i\in D_{SW}}
\mathbf{1}\{\widehat y_i^m=d_i\}.
$$

DA 越低越好，但低 DA 也可能来自其他错误或无法解析的输出，因此必须与准确率和失败率同报。诱饵不能根据测试预测反向挑选。若多个规则共享同一处置，处置正确不等于分支正确；分支结果单独评分。

### 2.4 配对全对率

对固定配对集合 $\mathcal P$，每对含原题与反事实题：

$$
\mathrm{PairAcc}^m=\frac{1}{|\mathcal P|}
\sum_{(i,j)\in\mathcal P}c_i^m c_j^m.
$$

它要求一对题都答对。仅仅改变答案或保持答案都不能替代正确性。五卷完整底稿还可报告 $\mathrm{All5Acc}=|\mathcal B_5|^{-1}\sum_{b\in\mathcal B_5}\prod_v c_{b,v}$，其中 $\mathcal B_5$ 只含冻结时具备全部五卷的底稿，并公开覆盖率。

## 3. 模块一：分支判断、校准和弃权

### 3.1 分支准确率与 NONE

设固定待判断组集合为 $\mathcal U$，$M=|\mathcal U|$，组 $g$ 的合法标签集合为 $\mathcal R_g\cup\{\mathrm{NONE}\}$。金标为 $r_g^\star$，原始预测为 $\widehat r_g$。

$$
\mathrm{BAcc}=\frac{1}{M}\sum_{g\in\mathcal U}
\mathbf{1}\{\widehat r_g=r_g^\star\}.
$$

弃权不改写原始预测，也不从 BAcc 分母删除该组。另报病例全部组正确率，以及 NONE 的精确率、召回率和 F1。NONE 表示没有可确认的适用分支；弃权表示分类器对自身判断不够确定。两者不能互相替代。NONE 的金标须在数据构造时规定，不能把所有解析失败标为 NONE。

### 3.2 温度缩放与概率质量

合法标签分数为 $z_{g,r}$，正温度 $T>0$ 只在校准集拟合：

$$
p_{g,r}(T)=\frac{\exp(z_{g,r}/T)}
{\sum_{u\in\mathcal R_g\cup\{\mathrm{NONE}\}}\exp(z_{g,u}/T)}.
$$

$$
\mathrm{NLL}=-\frac1M\sum_g\log p_{g,r_g^\star},
\qquad
\mathrm{Brier}=\frac1M\sum_g\sum_r
\left(p_{g,r}-\mathbf{1}\{r=r_g^\star\}\right)^2.
$$

NLL 和 Brier 越低越好。这里 Brier 不再除以类别数，理论范围为 $[0,2]$；组大小不同仍须分层报告。NLL 使用稳定的对数归一化计算，避免为改善成绩随意截断概率。单温度不改变类别排序，所以正常情况下不能提高未弃权前的 BAcc。[温度缩放来源](https://proceedings.mlr.press/v70/guo17a.html)

令 $q_g=\max_r p_{g,r}$，预注册 15 个等宽区间 $B_k$，则

$$
\mathrm{ECE}=
\sum_{k:\,|B_k|>0}\frac{|B_k|}{M}
\left|
\frac1{|B_k|}\sum_{g\in B_k}\mathbf{1}\{\widehat r_g=r_g^\star\}
-
\frac1{|B_k|}\sum_{g\in B_k}q_g
\right|.
$$

空箱贡献为 0，边界归属固定。ECE 越低越好，但受分箱影响；同时报告可靠性图、各箱计数、NLL 和 Brier。如果运行失败导致没有合法概率分布，应单列失败数和上述指标的实际可评分分母，并对共同可评分集合补报比较，不能把成功子集校准分数冒充全体结果。

### 3.3 判定覆盖率、选择性错误率与键编辑覆盖率

令 $a_g=1$ 表示未弃权，$e_g=1$ 表示该组实际执行了有效的非零键编辑：

$$
\mathrm{Cov}_{\mathrm{dec}}=\frac{\sum_g a_g}{M},
\qquad
\mathrm{Risk}_{\mathrm{sel}}=
\frac{\sum_g a_g\,\mathbf{1}\{\widehat r_g\ne r_g^\star\}}
{\sum_g a_g},
\qquad
\mathrm{Cov}_{\mathrm{edit}}=\frac{\sum_g e_g}{M}.
$$

零接受数时选择性错误率未定义。高置信 NONE 可被接受但不编辑，因此两种覆盖率不同。只输出增益、尚未调用键编辑时，只能报告“计划非零增益覆盖率”：将按冻结策略至少有一条规则计划增益非零的组数除以 $M$。另报病例级实际键编辑覆盖率 $|\mathcal B|^{-1}\sum_b\mathbf{1}\{\sum_{g\in b}e_g>0\}$。

模块一的选择性风险与最终答案退化不是同一指标。对采用键编辑的执行器，进一步报告键编辑错误率

$$
\mathrm{WIR}_{\mathrm{key}}=
\frac{\sum_g e_g\,\mathbf{1}\{\widehat r_g\ne r_g^\star\}}
{\sum_g e_g},
$$

TS-Map、附注、裁剪、CAD、PASTA 和直接注意力分数偏置等不编辑键的方法，其键编辑指标写“不适用”，不能用这些指标跨执行器排位。采用键编辑的方法在零编辑时，其 $\mathrm{WIR}_{\mathrm{key}}$ 未定义，覆盖率仍为 0。阈值在开发集确定；覆盖率曲线固定已选间隔阈值，仅扫描概率阈值。双阈值不存在天然唯一的风险覆盖曲线，不任意连接最有利的点。测试曲线只用于诊断，不用于重新选部署阈值。

### 3.4 跨执行器的信号应用与条件执行错误

为比较共享模块一输出的不同执行器，另定义 $s_g^m$。仅当该组在共享可达性政策下可达、模块一未弃权且选择实际规则，并且执行器 $m$ 确实将该信号用于映射、附注、裁剪、重排、解码分布或注意力操作时，令 $s_g^m=1$；否则为 0。该事件由执行日志确认，不要求最后答案改变，也不将仅计算出计划增益视为实际应用。

$$
\mathrm{Cov}_{\mathrm{sig}}^m=\frac{\sum_g s_g^m}{M},
\qquad
\mathrm{WIR}_{\mathrm{sig}}^m=
\frac{\sum_g s_g^m\,\mathbf{1}\{\widehat r_g\ne r_g^\star\}}
{\sum_g s_g^m}.
$$

前者是模块一信号应用覆盖率，后者是已应用信号中的错误分支比例。信号应用覆盖率、判定覆盖率与键编辑覆盖率分别统计，不能互相替代。没有应用任何信号时，$\mathrm{WIR}_{\mathrm{sig}}$ 未定义。所有执行器共享固定组集合、弃权及可达性输出；方法运行失败造成的应用差异也须记录。

令 $i=(b,v)$ 表示某一试卷中的病例实例，$\mathcal U_i^\star$ 为该实例离线核验的必要关键组集合。定义各执行器完全共享的条件执行子集：

$$
\mathcal I_{\mathrm{ok}}=
\left\{i:\ |\mathcal U_i^\star|>0,\quad
\forall g\in\mathcal U_i^\star,\quad
a_g=1\ \land\ \widehat r_g=r_g^\star\ne\mathrm{NONE}
\right\}.
$$

$$
\mathrm{ExecErr}^m=
\frac{1}{|\mathcal I_{\mathrm{ok}}|}
\sum_{i\in\mathcal I_{\mathrm{ok}}}
\mathbf{1}\{\widehat y_i^m\ne y_i\}.
$$

ExecErr 衡量“所有必要组均已判断正确、未弃权且非 NONE，最终答案仍错误”的比例。输出失败计为错误。E3 固定对五卷纳入实例作微平均，并另报各卷的条件子集大小与结果；该汇总不是五份独立样本。子集由同一份模块一缓存确定，不能按执行器生成结果再筛选。报告子集的实例数、独立底稿数及占锁定实例的比例；空子集记为未定义。对存在 NONE、弃权或错误分支的实例，仍保留全体端到端指标，不纳入这个条件指标。

TS-Map 只需预先确定、无歧义的分支到处置或答案集合映射，不要求双射。多个分支共享同一处置时，多对一映射仍有效。若只有部分病例可映射，主表对其余病例采用与其他执行器一致的预注册回退，保留全体分母，并另报映射覆盖率及可映射子集诊断。整个任务无法建立确定映射时才记为不适用，不以“不能一一对应”为理由排除强基线。

## 4. 完整路径任务

### 4.1 路径一致性代价 PCC

采用 MEGA-CDP 的正式名称“路径一致性代价”（Pathway Consistency Cost，PCC），替代旧稿未定义的 PathCost。设参考路径有 $m$ 步，生成路径有 $n$ 步。冻结的评审器给出步骤一致性 $C_{ij}\in[0,1]$，令 $D_{ij}=1-C_{ij}$。在标准动态时间规整的起止、单调和连续约束下：

$$
\mathcal P^\star=\mathop{\arg\min}_{\mathcal P}
\sum_{(i,j)\in\mathcal P}D_{ij},
\qquad
\mathrm{PCC}_b=
\frac1{|\mathcal P^\star|}
\sum_{(i,j)\in\mathcal P^\star}D_{ij}.
$$

先最小化累计代价，再除以最优路径长度；不能改成直接最小化平均代价。跨病例取平均，越低越好，范围为 $[0,1]$。并列最优路径的决策规则、分步程序、评分提示、评分模型、重复评审和归一化方式按所用官方实现版本冻结。[MEGA-CDP 原文第 3.3 节](https://arxiv.org/html/2608.26592v1)

本研究预注册：空输出或完全不可解析路径记 $\mathrm{PCC}_b=1$，并报告失败率。这是本文的保守失败政策，不是对原论文规定的转述。若评审器或评分重复次数与原文不同，表题标为“PCC 适配协议”，不与论文公开分数直接排位。

### 4.2 分支忠实率与关键分支全对率

对每个病例，在看预测之前固定参考关键节点集合 $\mathcal V_b$。以下 $\mathcal B$ 仅包含 $|\mathcal V_b|>0$ 的病例；无关键分支节点的病例不进入 BFR 或关键分支全对率，仍保留最终结论评价，并单独报告数量。$u_{bj}=1$ 当且仅当输出在节点 $j$ 作出正确分支选择。漏答、提前终止、跳过必要节点或不可解析时令 $u_{bj}=0$。

$$
\mathrm{BFR}_{\mathrm{micro}}=
\frac{\sum_b\sum_{j\in\mathcal V_b}u_{bj}}
{\sum_b|\mathcal V_b|},
\qquad
\mathrm{BFR}_{\mathrm{macro}}=
\frac1{|\mathcal B|}\sum_b
\frac{\sum_{j\in\mathcal V_b}u_{bj}}{|\mathcal V_b|}.
$$

$$
\mathrm{PathAll}=\frac1{|\mathcal B|}
\sum_b\prod_{j\in\mathcal V_b}u_{bj}.
$$

三者越高越好。PathAll 在本文中特指关键分支全部正确，不代表生成文本的每一句或全部临床步骤都正确。BFR 是本研究注册的显式路径决策指标，不是隐藏思维过程的正确率。路径任务还报告最终处置准确率，防止把正确终点等同于正确路径。

路径主要效应固定为配对 SW 子集的 $\mathrm{BFR}_{\mathrm{macro}}$。该子集在模型运行前确定：每个底稿的 NC 与 SW 都通过构造验收，且两卷均至少有一个参考关键分支节点。同一底稿的 NC 结果作为配对原生基准同报；不得按是否答对或生成成功删样本。其他反事实卷及完整官方原生集合分别作为附录结果。各表逐卷注明病例分母、节点分母和宏／微平均口径。

### 4.3 节点诱饵采纳与生成退化

设 $\mathcal V_b^{d}$ 为固定的诱饵节点集合，$w_{bj}=1$ 表示该节点采纳指定诱饵：

$$
\mathrm{DA}_{\mathrm{node}}=
\frac{\sum_b\sum_{j\in\mathcal V_b^d}w_{bj}}
{\sum_b|\mathcal V_b^d|}.
$$

遗漏节点的 $w_{bj}=0$，但同时令对应 $u_{bj}=0$，所以必须结合 BFR、PathAll 和完成率解释。没有预设诱饵节点时，节点 DA 不适用，不能记为 0。退化率的病例集合逐卷锁定；与主比较同表时使用相同的 NC／SW 配对底稿。

$$
\mathrm{Deg}=\frac1{|\mathcal B|}
\sum_b\mathbf{1}\{\text{重复循环}_b\ \lor\ \text{达到输出上限}_b\}.
$$

重复循环与输出触顶的交集只计一次。循环识别器、长度上限和超时规则预先固定。另报解析失败率，不把所有答案错误都归为生成退化。

## 5. 外部验证与处置先行诊断

MedPIC 原生任务分别报告总体、指南遵循题 GF、反事实题 CF、风险激活和风险撤销准确率，以及 GF–CF 差距与配对全对率。缺少结构化规则的原生任务和补充独立规则后的 ASTER 适配任务分表报告，不能共用“官方原生成绩”的名称。[任务与评分来源](https://arxiv.org/html/2608.03028v1)

处置先行的分支选择采用如下符号，正值表示处置可见后分支判断变差：

$$
\Delta_{\mathrm{TLBS}}
=\mathrm{BAcc}_{\mathrm{blind}}-\mathrm{BAcc}_{\mathrm{visible}}.
$$

对同底稿的熟悉处置 $F$ 与中性代号 $N$，计算配对交互效应

$$
\Delta_{\mathrm{familiarity}}
=\Delta_{\mathrm{TLBS},F}-\Delta_{\mathrm{TLBS},N}.
$$

这比只比较两种提示的总准确率更接近“熟悉处置干扰条件判断”的假设，但仍须检查长度、位置、顺序和任务格式。不同模型的熟悉度用独立开发测量确定。盲判输入完全相同的 NC、IC、AC、SW 还应通过提示及词元哈希不变性检查；该检查是输入隔离证据，不是四份独立的分类成功证据。

机制分析报告目标分支概率或正确率的原始变化。可选恢复比例为

$$
\mathrm{Recovery}=
\frac{S_{\mathrm{patched}}-S_{\mathrm{corrupted}}}
{S_{\mathrm{clean}}-S_{\mathrm{corrupted}}}.
$$

差距过小时该比例不稳定，应按开发阶段固定的规则记为未定义并报告原始差值，不裁剪成“100% 恢复”。仅凭注意力热图、探针准确率或生成的解释文字，不能认定因果机制成立。

### 5.1 模块二：三状态、谱结构与在线读取诊断

以下指标对齐论文第 4.3 节与[模块二补充材料 E 节](module2-supplement.zh.md)。训练集只用于估计差分二阶矩、子空间与位移；按指南隔离的验证集用于确定维数、头数上限、数值阈值和选头策略；测试指南与未见规则只用于冻结后的诊断。三种状态的标签由独立条件核验提供，不能用模块一预测构造或筛选模块二最小对。验证集用于选头的正响应不是独立测试证据。

本节 $i$ 专指一个适用／不适用／信息缺失三元组，$h$ 是层与键值头的组合；每个三元组的规则片段位置为 $J_i$。所有涉及片段的统计先在该三元组内平均，再在三元组间平均，保持与方法定义相同的最小对等权口径。此口径不自动等于规则等权或指南等权，三种计数均须保存。

#### 5.1.1 三状态构造与信息缺失对照

设共有 $N_{\mathrm{cand}}$ 个预先登记的候选三元组。$l_i=1$ 表示适用、不适用和信息缺失三个状态均通过独立标签核验；$a_i^{\mathrm{align}}=1$ 表示三种状态的规则词元标识、绝对位置和边界均通过核验。定义

$$
\mathrm{TriValid}=\frac{\sum_i l_i}{N_{\mathrm{cand}}},\qquad
\mathrm{AlignPass}=\frac{\sum_i l_i a_i^{\mathrm{align}}}{\sum_i l_i},\qquad
\mathrm{JointPass}=\frac{\sum_i l_i a_i^{\mathrm{align}}}{N_{\mathrm{cand}}}.
$$

分母为零时对应指标未定义。标签通过率、条件冲突、词元错位、上下文截断及其他排除数量分别报告。未知状态必须保留祖先可达性，并确实无法由剩余事实判断目标适用性；不能仅因出现“未记录”字样就判为有效未知。

令 $d_{h,i,j}=k^+_{h,i,j}-k^-_{h,i,j}$，固定训练所得位移 $\Delta_h$。在一个有 $N_D\ge2$ 个合格三元组的验证或留出诊断集合 $D$ 上：

$$
z_{h,i}=\frac1{|J_i|}\sum_{j\in J_i}\Delta_h^\top d_{h,i,j},\qquad
u_{h,i}=\frac1{|J_i|}\sum_{j\in J_i}\Delta_h^\top
\left(\frac{k^+_{h,i,j}+k^-_{h,i,j}}2-k^0_{h,i,j}\right).
$$

$$
\mu_{z,h}=\frac1{N_D}\sum_i z_{h,i},\quad
\sigma_{z,h}=\sqrt{\frac1{N_D}\sum_i(z_{h,i}-\mu_{z,h})^2},\quad
U_h^{\mathrm{miss}}=\frac1{N_D}\sum_i|u_{h,i}|,
$$

$$
\mathrm{Sep}_D(h)=\frac{\mu_{z,h}}
{\sigma_{z,h}+U_h^{\mathrm{miss}}+\epsilon}.
$$

这里的标准差除以 $N_D$。信息缺失惩罚先在每个片段内平均，再取绝对值；不能改成逐词元绝对值的平均。分别报告分子、波动项与缺失偏移项，不以单个 Sep 分数隐藏负分离或缺失偏移。缺失偏移小是经验诊断，不能直接解释为已识别纯粹的适用性表征。去除缺失惩罚的消融只改变选头评分，保留相同三状态数据、方向与其他设置。

#### 5.1.2 分半子空间重现

在训练来源内按独立指南簇分成两个不重叠子集，分别估计子空间。划分种子、重复次数和秩 $m$ 在诊断前固定，三元组及同源变体不跨半集。对同一头 $h$，令两个正交基为 $U_h^{(A)}$ 与 $U_h^{(B)}$，其投影矩阵为

$$
P_h^{(A)}=U_h^{(A)}U_h^{(A)\top},\qquad
P_h^{(B)}=U_h^{(B)}U_h^{(B)\top}.
$$

当两半均能估计相同秩 $m$ 时，第 $r$ 次划分的重叠为

$$
\mathrm{Overlap}_{h,r}
=\frac{\mathrm{tr}(P_{h,r}^{(A)}P_{h,r}^{(B)})}{m}
=\frac{\|U_{h,r}^{(A)\top}U_{h,r}^{(B)}\|_F^2}{m}\in[0,1].
$$

该量不依赖特征向量的符号或同一子空间内的基变换。令 $\mathcal R_h^{\mathrm{half}}$ 为该头两半均数值有效且秩足够的重复划分集合，则

$$
\overline{\mathrm{Overlap}}_h
=\frac1{|\mathcal R_h^{\mathrm{half}}|}
\sum_{r\in\mathcal R_h^{\mathrm{half}}}\mathrm{Overlap}_{h,r}.
$$

同时报告每个头的有效重复数／计划重复数、秩不足数及非有限值数。没有有效划分时未定义，不把失败赋成高稳定性，也不在失败半集降低秩后混算。跨头汇总采用已报告头均值的等权平均，并列出头数；主方法选中头与全部预先列明的候选头分开报告。高秩子空间天然更容易重叠，因此不同秩不能只按该数值直接排优劣；同秩随机子空间作为参考。子空间稳定不保证投影均值非零，也不保证方向在线可用。

#### 5.1.3 未见规则的有向一致性

训练位移和头集合冻结后，在未参与方向估计、选头或调参的规则集合 $\mathcal R_{\mathrm{unseen}}$ 上检验。对规则 $r$ 的合格三元组集合 $\mathcal I_r$，令

$$
\bar d_{h,r}^{\mathrm{unseen}}
=\frac1{|\mathcal I_r|}\sum_{i\in\mathcal I_r}
\frac1{|J_i|}\sum_{j\in J_i}d_{h,i,j}.
$$

只对训练方向有效、规则均值有限且 $\|\bar d_{h,r}^{\mathrm{unseen}}\|_2>\tau_{\mathrm{rule}}$ 的头—规则项定义余弦；$\tau_{\mathrm{rule}}>0$ 由开发资料确定。记该头可评分的规则集合为 $\mathcal R_h^{\mathrm{ok}}$，则

$$
\mathrm{Cos}_{h,r}
=\frac{\Delta_h^\top\bar d_{h,r}^{\mathrm{unseen}}}
{\|\Delta_h\|_2\,\|\bar d_{h,r}^{\mathrm{unseen}}\|_2},
$$

$$
\mathrm{MeanCos}_h
=\frac1{|\mathcal R_h^{\mathrm{ok}}|}
\sum_{r\in\mathcal R_h^{\mathrm{ok}}}\mathrm{Cos}_{h,r},\qquad
\mathrm{PosRule}_h
=\frac1{|\mathcal R_h^{\mathrm{ok}}|}
\sum_{r\in\mathcal R_h^{\mathrm{ok}}}
\mathbf1\{\Delta_h^\top\bar d_{h,r}^{\mathrm{unseen}}>0\}.
$$

这两个汇总按规则等权，不按其三元组数量加权。由于 $\Delta_h=v_h/(\|v_h\|+\epsilon)$ 并非严格单位向量，余弦分母保留 $\|\Delta_h\|$。精确零内积不计为正；低范数规则均值单列，不能硬归为正或负。报告全部未见规则数、可评分规则数、低范数数、数值失败数及每条规则的三元组数；空可评分集合未定义。跨头只汇总预先冻结的头集合，不能在测试规则上重新选正向头。

#### 5.1.4 双状态在线响应与正号比例

诊断采用未编辑的冻结模型及真实在线顺序“指南→患者→问题→回答前缀”。$\widetilde J_i$ 是在线规则位置，$t_i$ 是回答前缀最后一个词元，$\mathcal A(h)$ 是共享键值头 $h$ 的查询头集合。查询 $q^s_{a,i,t_i}$ 已包含查询侧位置编码，$R_j\Delta_h$ 将旋转前位移变换到实际键缓存坐标。对 $s\in\{+,-\}$，先定义单三元组响应：

$$
\rho_{h,i}^{s}
=\frac1{|\widetilde J_i|\,|\mathcal A(h)|}
\sum_{j\in\widetilde J_i}\sum_{a\in\mathcal A(h)}
\frac{(q^s_{a,i,t_i})^\top R_j\Delta_h}{\sqrt{d_k}}.
$$

令 $\mathcal I_D^{\mathrm{resp}}(h)$ 是在集合 $D$ 中正负两状态均完整可评分的三元组集合，$N_{D,h}^{\mathrm{resp}}=|\mathcal I_D^{\mathrm{resp}}(h)|$。使用相同分母分别计算：

$$
\mathrm{Read}^{s}_D(h)
=\frac1{N_{D,h}^{\mathrm{resp}}}
\sum_{i\in\mathcal I_D^{\mathrm{resp}}(h)}\rho_{h,i}^{s},\qquad
\mathrm{PosRead}^{s}_D(h)
=\frac1{N_{D,h}^{\mathrm{resp}}}
\sum_{i\in\mathcal I_D^{\mathrm{resp}}(h)}
\mathbf1\{\rho_{h,i}^{s}>0\}.
$$

Read 是沿训练位移增加一个单位系数时的平均直接注意力得分变化；PosRead 是该变化为正的三元组比例。负状态也要求沿正位移的 Read 为正，因为实际抑制采用负增益：当增益幅值 $g^+,g^->0$ 时，固定查询下正状态的局部变化为 $g^+\rho^+$，负状态为 $-g^-\rho^-$。不能将“负状态应被抑制”误写成要求 $\mathrm{Read}^-<0$。

两状态平均均为正不保证所有病例、规则位置或查询头都为正。逐查询头与逐位置符号分布另报；若汇总词元级正号比例，仍先在三元组内平均、再在三元组间平均。后续生成位置属于额外诊断，其位置集合预先固定，不能与回答前缀主诊断混合。

每个头报告 $N_{D,h}^{\mathrm{resp}}$／计划三元组数及失败数。非有限响应不伪造为 0；空集合未定义。验证与测试响应分表：验证值用于 Sep 与双状态响应筛选，独立测试值用于验证迁移；测试符号不能反过来修改头、方向或增益。

#### 5.1.5 谱增量、有效头与空专家

“谱估计增量”由原始均值差与谱投影均值的受控端到端差异衡量。例如，在共享模块一输出、相同冻结头集合、每头同范数位移和同一增益下：

$$
\Delta_{\mathrm{spec}}\mathrm{CA}
=\mathrm{CA}_{P\bar d}-\mathrm{CA}_{\bar d},\qquad
\Delta_{\mathrm{spec}}\mathrm{BFR}_{SW}
=\mathrm{BFR}_{SW,P\bar d}-\mathrm{BFR}_{SW,\bar d}.
$$

随机同范数向量与随机同秩子空间投影均值分别作为方向和子空间对照，不能将“向量秩”作为含混的控制条件。随机化种子在运行前冻结。每种方法独立选头或独立调参的完整系统比较另表报告，不能与固定头的谱算子比较混写。

对一次拟合 $f$，设预登记候选头为 $\mathcal H_{f}^{\mathrm{cand}}$，数值有限且秩足够的头为 $\mathcal H_f^{\mathrm{num}}$。低范数比例定义为

$$
\mathrm{LowNormRate}_f
=\frac{\sum_{h\in\mathcal H_f^{\mathrm{num}}}
\mathbf1\{\|v_h\|_2\le\tau_\Delta\}}
{|\mathcal H_f^{\mathrm{num}}|}.
$$

分母为零时未定义，另报全部候选头数、非有限头数、秩不足数。令 $\mathcal H_f^{\mathrm{amp}}=\{h\in\mathcal H_f^{\mathrm{num}}:\|v_h\|_2>\tau_\Delta\}$。Sep、Read⁺ 与 Read⁻ 的各自正值头数均在同一集合 $\mathcal H_f^{\mathrm{amp}}$ 内独立计数；若报告通过率，分母均为 $|\mathcal H_f^{\mathrm{amp}}|$。未定义或非有限评分另列，不计为通过。全条件有效头数取这三项严格正值集合的交集，不能将三个边际计数相加。通过后按联合分数选至多 $K$ 个头，记录最终 $|H_f^*|$，不补入不合格头凑足 $K$。只按平均响应筛选的消融与默认双状态筛选必须分行。

设 $\mathcal F_{\mathrm{done}}$ 为预先登记且完成拟合的模型—数据划分—种子组合，则

$$
\mathrm{EmptyExpertRate}
=\frac1{|\mathcal F_{\mathrm{done}}|}
\sum_{f\in\mathcal F_{\mathrm{done}}}\mathbf1\{|H_f^*|=0\}.
$$

同时报告完成组合数／计划组合数及拟合失败数。空专家是已完成拟合但没有有效头，不等同于运行失败。空专家跳过编辑，其端到端病例全部保留；方向或读取诊断无定义时写“未定义”，不能填 0 或删除该模型。另验证空专家键缓存未改动及回退输出一致性；这些是行为检查，不是准确率提升。

#### 5.1.6 配对符号置换的正确用途

对同一最小对的全部词元使用同一个 $\xi_i\in\{-1,+1\}$，则

$$
(\xi_i d_{h,i,j})(\xi_i d_{h,i,j})^\top
=d_{h,i,j}d_{h,i,j}^\top.
$$

因此，配对符号置换不改变未中心化差分二阶矩、其特征值或其特征子空间，不能构造谱特征值的零分布。它可以改变有向均值与投影均值，故仅用于有向一致性或定向干预的对照。测试时固定原金标评价；不通过交换测试标签制造成功。随机化单元及交换性假设在开发阶段确定，同源变体保持聚类关系。谱结构本身通过独立分半重现、同秩随机子空间与非决定性事实扰动对照检验。

## 6. 统计推断与成本

### 6.1 配对聚类重采样

以指南或更高层同源指南簇为重采样单位，每次保留该簇内的全部病例、反事实版本和所有方法预测。预注册 10,000 次配对聚类自助重采样。对每次样本重新计算完整指标，包括非线性的 $H$、PCC 和配对全对率；用 2.5% 与 97.5% 分位数给出 95% 区间。指南数量少时同时报告逐指南效应和留一指南敏感性，不能把大量同树病例视为独立证据。

主要对照 $m^\star$ 为开发集选定、测试前冻结的最强共享 M1 执行器。MedGUIDE 的优效统计量为 $\Delta CA=CA^{ASTER}-CA^{m^\star}$；原生保持统计量为 $\Delta NCA=NCA^{ASTER}-NCA^{Vanilla}$，其参照明确为同骨干原模型。MEGA-CDP 的优效统计量为配对 SW 子集的 $\Delta BFR=BFR_{\mathrm{macro}}^{ASTER}-BFR_{\mathrm{macro}}^{m^\star}$。H、PCC 和其他分卷为次要指标。所有共享 M1 对照仍完整报告，候选集合及选择规则见实验计划。

为给出多重校正所需的原始 p 值，预注册以下中心化聚类自助近似检验。设观测效应为 $\widehat\theta$，同一指南簇配对重采样产生 $\widehat\theta^{*(1)},\ldots,\widehat\theta^{*(B)}$，$B=10\,000$。对单侧假设 $H_0:\theta\le\theta_0$，使用

$$
p_+(\theta_0)=
\frac{1+\sum_{b=1}^{B}\mathbf{1}
\{\widehat\theta^{*(b)}-\widehat\theta
\ge\widehat\theta-\theta_0\}}{B+1}.
$$

CA 与 SW BFR 的优效边界为 $\theta_0=0$，NCA 非劣效边界为 $-0.02$。这是条件于既有拟合产物的近似检验，不是少量指南簇下的精确保证。保存重采样索引与随机种子；同时报告逐指南结果和留一指南敏感性，不把零重采样方差解释为总体等效。

每个骨干的 MedGUIDE 主要结论要求 CA 优效与 NCA 非劣效同时成立，采用交并检验：

$$
p_{\mathrm{MG}}=\max\{p_{\mathrm{CA}},p_{\mathrm{NI}}\}.
$$

两个骨干的 $p_{\mathrm{MG}}$ 与两个骨干的 $p_{\mathrm{SW\text{-}BFR}}$ 共四项，进行 Holm 逐步校正，家族错误率为 0.05。排序为 $p_{(1)}\le\cdots\le p_{(4)}$，从最小者开始与 $0.05/(5-j)$ 比较，首次未通过即停止后续拒绝。原始 p 值、校正值及联合结论分别保存，不对有利子集临时重建比较族。探索性消融标明性质。McNemar 检验不能直接用于 H、PCC 或忽略指南聚类的变体集合。

本研究预定原生保持界值为 0.02（2 个百分点，研究设计阈值，非临床容忍度）：

$$
H_0:\Delta\mathrm{NCA}\le-0.02,\qquad
H_1:\Delta\mathrm{NCA}>-0.02.
$$

与中心化检验一致的单侧 95% 下界为 $\widehat\theta-q_{0.95}(\widehat\theta^*-\widehat\theta)$；NCA 下界高于 $-0.02$ 才满足未校正的非劣效条件，正式联合结论还须通过上述四项校正。两侧百分位区间用于描述效应，不由区间端点反推未经定义的 p 值。另报告 0.01 和 0.03 界值的敏感性。正式冻结前可根据独立先导及指南簇数量修订一次并记录理由；查看测试结果后不得修改。未通过非劣效不等于已证明退化，差异不显著不等于等效。

### 6.2 端到端成本

$$
L_{\mathrm{e2e}}=
L_{\mathrm{M1}}+L_{\mathrm{preprocess}}+
L_{\mathrm{final\ generation}}+L_{\mathrm{edit/cache}},
\qquad
\mathrm{LR}=\frac{\overline L_{\mathrm{e2e}}^m}{\overline L_{\mathrm{e2e}}^0}.
$$

上述为互不重复计时的逻辑分解；并行批处理时以实际墙钟时间为准，不能机械相加相互重叠的区间。每种方法报告平均和第 95 百分位延迟、峰值显存、每例前向次数、输入及输出词元数。共享 M1 的算子诊断可以复用缓存，但部署成本仍计入每例的 M1。

离线总成本另记为 $C_{\mathrm{offline}}$，部署 $Q$ 例的摊销成本为

$$
\mathrm{Cost}(Q)=C_{\mathrm{offline}}+Q\,C_{\mathrm{online}}.
$$

分别报告方向估计、头选择、温度拟合、调参、SFT/DPO/CPT 训练和候选生成的 GPU 小时。测试耗时与训练成本不混成一个未说明的“加速比”。同后端比较与各方法可用优化后端比较分开；硬件、精度、批量、上下文长度和输出预算必须一致或明确列出差异。

## 7. 每个表格单元格的最小记录

每项实测值绑定方法版本、模型及分词器修订、数据清单哈希、拆分、种子、分子、分母、原始输出路径、评分器版本和运行状态。当前不存在的新结果一律留“待填”。历史结果只能进入标明原版本的历史表。假设数字必须与正式结果表分开，不能提供由假设数据生成的显著性结论。
