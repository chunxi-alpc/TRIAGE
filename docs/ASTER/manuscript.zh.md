# 对消，而非擦除：用干扰痕迹修正大模型的指南条件判断

## 摘要

指南条件判断应由条件定义和患者事实决定，后续处置与无关背景会改变模型的判断分数。ASTER-CR保留完整输入，以A/P配对估计输入相关干扰贡献，以C/F配对约束证据响应，并独立校准决策阈值。理论分析刻画混合痕迹下干扰对消与证据保持的共同可行条件。主训练采用MEGA-CDP的2,540对均衡A/P/C/F配对；NLI4CT-2024和MedPIC用于冻结跨任务评价。Qwen3-8B受控留出上，16维线性ASTER-CR将A/P干扰方差降低4.10%，保留99.60%的C/F有向证据响应，证据修正RMS为0.2539 logit，动态路径全对为767/806。MEGA官方路径准确率为94.31%，NLI正式测试准确率为73.29%，MedPIC整题集合全对率为14.99%。80个来源的五折验证中，E1的A/P方差降低率为3.498%。这些结果确立了ASTER-CR在受控配对上的工作点：以低幅度输入相关修正降低干扰响应，并保持较高的决定性证据响应。

## 1 引言

指南条件判断的正确性取决于规则条件和患者事实。模型同时读取后续处置与背景文本时，非决定性内容也可能改变判断分数。忠实推理因此要求模型在决定性证据保持不变时维持判断，并在决定性证据改变时相应更新判断。这一要求与解释忠实性（Jacovi & Goldberg, 2020）及反事实不变性（Veitch et al., 2021）相联系。

在指南节点上，当前真值由条件界定 C 和决定性事实 F 确定，即 \(y=g(C,F)\)。后续处置与当前判断相关，但不参与前置条件的真值定义。历史 Qwen3-8B 结果显示，仅改写处置或无关背景时，判断分数变化的中位数约为 0.5 logit；改变决定性事实时约为 50.8 logit [R5]。非决定性内容产生的较小分数变化也会影响接近零阈值的节点；历史被编辑改变的节点，其原始分数集中在 0、0.25 或 0.5 [R8]。

输入删除会同时改变模型可用的信息。历史 275 个可统计节点中，71 个需要条件句之外的定义；遮挡处置后，完整路径正确数由 404/432 降至 394/432 [R2]。ASTER-CR保留完整输入，直接估计非决定性内容对当前判断分数的贡献。

本文将目标定义为决策分数层面的条件不变性。设模型表示为 \(h\)，后续读出为 \(G\)。在固定证据时，方法需要使处置变化对当前决策分数的贡献保持稳定；表示本身可以继续编码处置信息。该目标作用于最终决策函数，适用于线性和非线性修正。非线性概念擦除扩展了函数表达能力（Ravfogel et al., 2022；Akbari et al., 2025）。本文进一步研究痕迹中可用于估计判断相关干扰贡献的信息，以及证据约束下的共享修正函数。

ASTER-CR估计需要从判断分数中消去的干扰贡献，并限制修正对证据变化的响应。与证据方向正交的痕迹可以携带其他方向上的干扰信息，因此修正器预测配对的总分数差。A/P配对监督干扰贡献的估计，C/F配对约束证据响应，差分消去共享常数项，独立校准确定决策阈值。痕迹提供跨证据条件共享的预测信息；线性读出构成当前主方法，神经网络扩展贡献估计函数的表达能力。

本文开展四项工作。第一，形式化指定判断的条件不变性，并区分决策分数修正与表示层概念擦除。第二，区分痕迹的直接传导效应与对总干扰贡献的预测，给出混合痕迹下对消与证据保持的共同可行条件。第三，将A/P差分目标、C/F证据预算与独立阈值校准落实为可分离检验的算法。第四，在MEGA-CDP、NLI4CT-2024和MedPIC上评价干扰响应、证据响应与原生任务表现，并通过组件消融分析方法机制。

## 2 相关工作

### 2.1 行为忠实性与输入干预

解释忠实性关注解释与实际决策过程的联系（Jacovi & Goldberg, 2020）。模型可能受到解释未充分揭示的偏置影响（Turpin et al., 2023），也可能呈现推理内容效应（Lampinen et al., 2024）。本文将规则规定的输入干预转化为两类配对评价：A/P配对保持决定性证据和标签不变，检验判断对无关变化的稳定性；C/F配对改变条件或决定性事实，检验判断对证据变化的响应。该设计将反事实不变性研究（Veitch et al., 2021）落实为可测的分数差与配对正确率。

处置遮挡、仅保留条件和证据抽取构成输入层对照，用于评价输入删除对条件定义及任务信息的影响。方法消融、输入层对照和外部算法基线分别承担机制分析、信息保留和方法比较的论证职责。

相关方法按主要作用位置及优化目标组织。第4.3节列出主实验比较方法；函数容量和修正位置的内部比较见第5.2节。

### 2.2 输出校准、监督读出与对比解码

输出层方法包括分数缩放、偏置估计和输入相关修正。温度缩放用一个正温度调整置信度（Guo et al., 2017），本文以固定正比例缩放作为整体幅度控制。上下文校准（CC）从无内容参考估计标签偏置（Zhao et al., 2021）；批量校准（BC）利用无标签输入的平均输出估计偏置（Zhou et al., 2024）。本文的仅截距校准使用自然训练输入的任务标签，普通监督线性读出重新学习分类方向。这些方法分别提供幅度、偏置和任务监督读出的比较基准。

上下文感知解码（CAD）组合同一模型在有、无上下文时的输出，强化上下文相对于模型先验的影响（Shi et al., 2024）。DExperts组合基础模型、专家和反专家的输出以控制生成属性（Liu et al., 2021）。CAD提供条件对比分数基线；DExperts代表需要额外专家训练的解码控制路线。

AxBench在Gemma-2-2B和9B上统一比较提示、微调、线性表示方法、监督引导和稀疏自编码器，报告提示在行为引导任务上的表现，以及微调和线性方法在不同任务上的优势（Wu et al., 2025）。据此，本文将监督线性读出和参数微调纳入Qwen3-8B主实验比较，与表示干预方法共同评价。

ASTER-CR以A/P配对差分监督输入相关分数修正，并以C/F配对约束证据响应。固定截距在配对差分中抵消；统一正比例缩放按相同比例改变干扰与证据响应。两类控制为输入相关修正提供明确的比较参照。

### 2.3 概念擦除

概念擦除包含多种优化路线。INLP迭代移除线性分类方向（Ravfogel et al., 2020）；R-LACE通过带秩约束的线性对抗优化学习投影（Ravfogel et al., 2022a）；LEACE给出闭式线性擦除（Belrose et al., 2023）。谱属性移除（SAL）依据表示与概念的交叉协方差移除高协变方向，其核化版本kSAL扩展到核特征空间（Shao et al., 2023）。

非线性概念擦除包括核化对抗擦除（Ravfogel et al., 2022b）、核化率失真方法KRaM（Chowdhury et al., 2023）、基于类别条件分布匹配的非线性LEOPARD（Saillenfest & Lemberger, 2025），以及交替训练编码器与核空间变换的Obliviator（Akbari et al., 2025）。FaRM研究率失真目标下的公平表示，并包含使用任务标签的设置（Chowdhury & Chaturvedi, 2022）；完全概念擦除的信息论极限也已有专门研究（Chowdhury et al., 2025）。

本文将概念擦除作为表示层参照，重点评价指定判断分数对无关变化的响应，以及对决定性证据变化的保留。该任务目标由A/P和C/F配对指标共同刻画。

### 2.4 表示干预与表示微调

表示干预可以使用固定方向，也可以依输入决定是否干预。ITI沿估计的注意力头方向调整激活（Li et al., 2023）；对比激活相加（CAA）用成对行为示例的残差激活差估计干预向量（Rimsky et al., 2024）；条件激活引导（CAST）用输入激活识别条件，再选择性施加行为向量（Lee et al., 2025）。三者分别提供共享方向、成对方向和条件触发的表示干预参照。

LoFiT学习局部表示调整（Yin, Ye & Durrett, 2024）；LoReFT学习低秩表示干预，同一研究中的DiReFT提供简化的表示变换对照（Wu et al., 2024）。本文以共享痕迹和配对监督构造线性及多层感知机表示编辑，用于分析函数容量与干预位置。

近年的表示引导研究进一步发展了成对监督和动态干预。Stolfo et al.（2025）研究指令差分激活向量及其跨模型迁移；AxBench建立提示、微调和表示干预的统一基准（Wu et al., 2025）。RePS以双向偏好优化学习表示引导（Wu et al., 2025）；PSR从输入激活估计位置相关的引导系数（Heyman & Vandeputte, 2026）。这些方法分别提供成对偏好学习和动态引导强度的设计参照。

Cheng et al.（2026）通过多词元激活修补研究拒绝行为中的表示引导，报告引导向量主要与注意力OV回路交互。该机制结果为本文M1的注意力头定位和OV路径分析提供研究线索。

Venhoff et al.（2026）比较基础模型与思考模型检查点，分析RL训练、SFT蒸馏及混合训练对模型行为的影响；Xu（2026）的预印本比较Gemma-2、Llama-3.1和Qwen-2.5的Base/IT匹配检查点，报告指令暴露程度与方向的行为可干预性相关。这些研究支持通过同家族匹配检查点分析后训练对表示行为的影响。

### 2.5 参数高效权重微调

LoRA用低秩矩阵参数化权重更新（Hu et al., 2022）。DoRA将权重幅值和方向分开，并对方向采用低秩适配（Liu et al., 2024）。FourierFT学习权重增量的部分傅里叶系数，再重建权重更新（Gao et al., 2024）。三者分别代表低秩更新、幅值—方向分解和频域参数化。

权重微调与表示干预按参数作用位置区分。成本评价同时记录训练参数、更新模块、训练量、搜索开销、选头、标准化和子空间存储。ASTER-CR的M3线性校正层包含16个输出系数；完整方法成本按全部模块统计。

## 3 方法

### 3.1 任务定义与配对

输入包括指南决策树、患者病例和当前节点。模型只判断当前条件是否成立。记 $z_+(x)$、$z_-(x)$ 分别为模型对答案词元“是”和“否”的输出 logit；将“是”编码为 $1$、“否”编码为 $0$。定义冻结原模型的二元判断分数及其预测：
```math
\ell_0(x)=z_+(x)-z_-(x),\qquad
\hat y_0(x)=\mathbf 1[\ell_0(x)>0].
```
因此，$\ell_0(x)>0$ 表示原模型预测当前条件成立。下文以 $\ell_0$ 表示原模型分数，以 $\ell_m$ 表示评价方法 $m$ 的分数。

**配对类型。**

以下仅为任务示意，不是数据集原文。当前条件为“eGFR ≥ 30 mL/min/1.73m²”，患者 eGFR 为 34，真值为“是”；两条分支分别给出继续或调整用药的后续处置。

| 类型 | 干预对象 | 标签 | 预期响应 |
|---|---:|---:|---:|
| A 处置改写 | 分支后的处置文字 | 不变 | 保持稳定 |
| P 背景改写 | 与当前节点无关的病史、画像 | 不变 | 保持稳定 |
| C 条件改写 | 条件阈值或定义，如阈值从 30 改为 45 | 翻转 | 跟随真值改变 |
| F 事实改写 | 决定性事实，如数值从 34 改为 26 | 翻转 | 跟随真值改变 |

**表1：** 指南判断的反事实配对类型。A/P保持决定性证据，C/F改变真值；F0另行统计。

F0 表示事实数值改变但仍在阈值同侧，属于同真值、不同证据的配对。本文以保持证据的 A/P 配对定义主方法，F0 作为额外的同侧分数稳定扩展单列比较。某条信息的角色由当前节点决定：年龄可能在一个节点上无关，在年龄条件节点上成为决定性事实。

“真值不变”首先要求配对在语义上成立。处置改写若同时改变条件定义，就不是合格的A配对。配对语义有效性与实际审核覆盖在数据说明及局限性中明确区分。

### 3.2 M1–M3：干扰痕迹与受约束修正

ASTER-CR由相互衔接的模块组成：M1定位与当前判断相关的注意力头；M2根据反事实激活差分估计证据与干扰子空间，并构造内部痕迹；M3利用痕迹学习受证据约束的分数修正；M4在M1–M3冻结后进行任务校准。下文以模型索引 \(m\) 表示模型相关配置。注意力头集合、候选集合规模、子空间秩、正则强度和证据预算均由相应模型的训练／开发流程确定，不作为跨模型固定常数。

**M1：注意力头定位。** 令 \(h_{j}^{L}\) 与 \(h_{j}^{R}\) 分别表示配对 \(j\) 两端在某注意力头上的激活，\(\ell_0(x)\) 为原模型判断分数。对每一类配对 \(q\in\{A,P,C,F\}\)，先在训练来源上计算梯度归因分数。记
\[
g_{j,h}^{L}=\nabla_{h_h}\ell_0(x_j^L),
\qquad
a_{j,h}=\left|
\left\langle h_h(x_j^R)-h_h(x_j^L),\,g_{j,h}^{L}\right\rangle
\right|.
\]
其中，配对权重 \(\alpha_j\) 根据原模型误判情况及其距决策边界的程度预先确定。每类配对对注意力头 \(h\) 的加权归因分数为
\[
A_{q,h}=
\frac{\sum_{j\in\mathcal J_q^{\mathrm{train}}}\alpha_j a_{j,h}}
{\sum_{j\in\mathcal J_q^{\mathrm{train}}}\alpha_j}.
\]
按 \(A_{q,h}\) 对注意力头进行排序，并保留模型配置规定的候选集合。梯度归因用于筛选候选头，随后以双向激活修补进行精确验证：在左侧输入的前向计算中，将候选头的激活替换为右侧输入对应激活；再在右侧输入的前向计算中进行反向替换。定义
\[
\begin{aligned}
IE_{j,h}^{L\leftarrow R}
&=\ell(x_j^L;\,h_h\leftarrow h_h(x_j^R))-\ell_0(x_j^L),\\
IE_{j,h}^{R\leftarrow L}
&=\ell(x_j^R;\,h_h\leftarrow h_h(x_j^L))-\ell_0(x_j^R).
\end{aligned}
\]
这里的 \(\ell(x;\,h_h\leftarrow h_h(x'))\) 表示在输入 \(x\) 的前向计算中，仅将指定注意力头的激活替换为输入 \(x'\) 对应的激活，再读取最终判断分数。对按来源平衡抽取的训练配对子集，计算双向修补效应的加权平均幅度：
\[
R_{q,h}=
\frac{\sum_{j\in\mathcal P_q^{\mathrm{rank}}}
\alpha_j\bigl(
|IE_{j,h}^{L\leftarrow R}|+
|IE_{j,h}^{R\leftarrow L}|
\bigr)/2}
{\sum_{j\in\mathcal P_q^{\mathrm{rank}}}\alpha_j}.
\]
候选头据此重新排序。A与P的排名通过秩聚合形成干扰侧排序，C与F的排名形成证据侧排序；最终交替合并两侧排序、去除重复项，并按模型配置确定的头集合规模截取。这样，M1同时利用梯度信息进行高效筛选，并利用双向真实激活修补确认候选头对最终分数的实际影响。所有归因、修补排名和最终头集合均只由训练来源确定。

**M2：证据保护的干扰痕迹提取。** 令 \(\mathcal H_m\) 为M1选出的模型相关注意力头集合，\(h_m(x)\in\mathbb R^{d_m}\) 为最终答案位置上这些注意力头输出的拼接向量。对训练配对 \(j\)，记左、右端输入为 \(x_j^L,x_j^R\)。令
\[
\mathcal J_{\mathrm{train}}
=\mathcal J_{AP}^{\mathrm{train}}\cup
\mathcal J_{CF}^{\mathrm{train}}
\]
为训练配对索引集合，\(\pi_j\geq0\) 为预先确定并归一化的配对权重，满足 \(\sum_{j\in\mathcal J_{\mathrm{train}}}\pi_j=1\)。权重用于平衡配对类别、来源、病例及病例内变体。令 \(\mathcal V_{\mathrm{train}}\) 为所有训练配对端点去重后的集合。将每个配对权重平均分配给其左右端点，对唯一端点 \(v\) 定义
\[
\rho_v=\frac12\sum_{j\in\mathcal J_{\mathrm{train}}}
\pi_j\left(
\mathbf 1[v=x_j^L]+\mathbf 1[v=x_j^R]
\right),
\qquad
\sum_{v\in\mathcal V_{\mathrm{train}}}\rho_v=1.
\]
据此估计训练端点的加权均值和逐维标准差：
\[
\begin{aligned}
\mu_m&=\sum_{v\in\mathcal V_{\mathrm{train}}}\rho_v h_m(v),\\
\sigma_{m,a}&=\max\left(
\epsilon_m,\,
\sqrt{\sum_{v\in\mathcal V_{\mathrm{train}}}
\rho_v(h_{m,a}(v)-\mu_{m,a})^2}
\right),\\
\tilde h_m(x)&=(h_m(x)-\mu_m)\oslash\sigma_m.
\end{aligned}
\]
其中 \(\epsilon_m\) 是防止数值退化的模型相关下限。对训练配对 \(j\)，定义标准化激活差
\[
\Delta\tilde h_{m,j}
=\tilde h_m(x_j^R)-\tilde h_m(x_j^L).
\]
证据子空间 \(S_m\) 与初始干扰子空间 \(U_m\) 分别由C/F和A/P配对差分估计：
\[
\begin{aligned}
S_m&=\operatorname{basis}_{k_m}
\left(\{\Delta\tilde h_{m,j}:j\in\mathcal J_{CF}^{\mathrm{train}}\}\right),\\
U_m&=\operatorname{basis}_{k_m}
\left(\{\Delta\tilde h_{m,j}:j\in\mathcal J_{AP}^{\mathrm{train}}\}\right).
\end{aligned}
\]
具体而言，\(\operatorname{basis}_{k_m}\) 先将每个配对差分行乘以对应配对权重的平方根，再对加权差分矩阵进行截断奇异值分解，保留由训练／开发流程确定的有效右奇异向量，并剔除数值零方向。随后将干扰子空间投影到证据子空间的正交补：
\[
N_m=\operatorname{orth}\bigl((I-S_mS_m^\top)U_m\bigr),
\qquad
t_m(x)=N_m^\top\tilde h_m(x).
\]
其中 \(I\) 为相应维度的单位矩阵，\(\operatorname{orth}\) 表示对投影残差再次正交化并剔除数值零方向。\(N_m\) 张成干扰变化子空间中与主要证据变化子空间正交的部分；\(t_m(x)\) 是供M3读取的内部痕迹坐标。注意，正交投影仅依据训练配对估计的主要变化子空间降低线性重叠，最终修正对证据响应的保护仍由M3中的C/F约束直接承担。标准化参数、\(S_m\)、\(U_m\) 和 \(N_m\) 均只使用相应训练来源估计。

**M3：证据约束的干扰贡献对消。** 对每个训练配对 \(j\)，定义原模型分数差和痕迹差：
\[
\Delta\ell_{0,j}
=\ell_0(x_j^R)-\ell_0(x_j^L),
\qquad
\Delta t_{m,j}=t_m(x_j^R)-t_m(x_j^L).
\]
线性修正函数为 \(c_{w_m}(t)=w_m^\top t\)。拟合修正系数时，在A/P配对上最小化修正后的分数差，并在C/F配对上逐对限制修正量：
\[
\begin{aligned}
\hat w_m=\arg\min_{w_m}\quad&
\sum_{j\in\mathcal J_{AP}^{\mathrm{train}}}
\omega_j\left(
\Delta\ell_{0,j}+w_m^\top\Delta t_{m,j}
\right)^2
+\lambda_m\|w_m\|_2^2,\\
\text{subject to}\quad&
|w_m^\top\Delta t_{m,j}|\leq\delta_m,
\qquad j\in\mathcal J_{CF}^{\mathrm{train}}.
\end{aligned}
\]
其中，\(\omega_j\) 为A/P回归权重，\(\lambda_m\) 为模型相关正则强度，\(\delta_m\) 为模型相关证据改动预算。A/P目标使固定证据下的干扰分数差趋近于零；C/F约束限制修正项对有效证据变化所产生的额外分数差。零修正始终是该约束问题的可行解。训练约束直接保证的是训练配对上的修正预算，留出来源上的证据响应、超预算情况及双端判断正确性仍需独立评价。

**M3输出中心化。** 为将输入相关修正与后续任务校准分开，使用M2定义的训练端点权重计算修正项的加权均值：
\[
\begin{aligned}
\mu_{c,m}
&=\sum_{v\in\mathcal V_{\mathrm{train}}}
\rho_v\,\hat w_m^\top t_m(v),\\
c_m(x)&=\hat w_m^\top t_m(x)-\mu_{c,m},\\
s_m(x)&=\ell_0(x)+c_m(x).
\end{aligned}
\]
因此，训练端点上的加权平均修正为零。中心化只减去输入无关的常数，不改变任意配对的分数差，也不改变C/F训练约束。最终M3不包含可学习的全局共享截距；整体阈值和概率校准由M4独立处理。非线性修正函数作为扩展时，仍以A/P两端函数值之差拟合干扰分数差，并以C/F两端函数值之差施加证据保护约束。

### 3.3 M4：任务感知校准与决策

M1–M3冻结后，M4在训练来源的自然样本上独立估计任务校准参数。对MEGA类二元判断，采用逻辑再校准：
\[
p(y=1\mid x)=\operatorname{sigmoid}
\left(\alpha_m+\beta_m s_m(x)\right),
\qquad \beta_m>0.
\]
校准参数仅由训练来源的自然节点标签估计；开发集用于预先规定的配置选择，测试标签仅用于最终评价。仅偏置、仅温度及联合校准作为消融设置，用于区分整体阈值变化、分数缩放和输入相关修正的作用。

对于MedPIC等集合预测任务，M4按照任务原生计分协议，将M3产生的选项分数映射为合法答案集合的能量，再在合法集合上归一化并选择预测集合。集合大小先验、特殊选项偏置和温度均由任务开发集确定。对于NLI等冻结迁移任务，按照预先指定的迁移协议使用M3分数，不在目标测试集重新拟合校准参数。各任务的校准形式、开发集范围和参数设置在实验部分分别说明。

### 3.4 M1–M4整体算法

给定冻结模型 \(f_m\)、训练来源、A/P与C/F反事实配对，以及模型相关配置，ASTER-CR按以下顺序训练：

1. **M1定位注意力头。** 对A、P、C、F配对分别计算激活差分与梯度的内积归因分数；对候选头执行双向激活修补；按干扰侧与证据侧分别排序并交替合并，得到模型相关注意力头集合 \(\mathcal H_m\)。
2. **M2构造内部痕迹。** 提取 \(\mathcal H_m\) 的答案位置激活，估计训练端点标准化参数；分别对C/F和A/P加权激活差分矩阵进行截断SVD，得到 \(S_m\) 和 \(U_m\)；将 \(U_m\) 投影到 \(S_m\) 的正交补并正交化，得到 \(N_m\) 和痕迹函数 \(t_m(x)\)。
3. **M3拟合受约束修正。** 计算A/P原模型分数差与痕迹差，以带正则的加权最小二乘拟合 \(w_m\)，并对每个C/F训练配对施加修正预算；再计算训练端点加权中心化常数，得到 \(s_m(x)\)。
4. **M4任务校准。** 固定M1–M3，在训练来源自然样本上拟合任务规定的校准器，并冻结所有参数。
5. **推理。** 对新输入运行冻结模型，读取原始判断分数和所选注意力头激活，计算 \(t_m(x)\)、\(s_m(x)\)，再应用冻结的M4校准器得到最终判断。

所有模型相关的头集合规模、候选集合规模、子空间秩、正则强度、证据预算及任务校准参数，均在相应训练／开发流程中确定并于测试前冻结。方法定义不依赖某个模型的固定头数、固定特征维度或固定配对数量。

### 3.5 核心理论结论

为了说明何时能同时去掉干扰并保留证据，考虑局部加性模型。令 $e\in\mathbb R^{d_e}$ 表示决定性证据坐标，$\nu\in\mathbb R^{d_\nu}$ 表示干扰坐标，$t\in\mathbb R^{d_t}$ 表示可读痕迹；令 $D\in\mathbb R^{d_t\times d_\nu}$、$H\in\mathbb R^{d_t\times d_e}$ 分别描述干扰和证据进入痕迹的线性映射，$\boldsymbol\gamma\in\mathbb R^{d_\nu}$ 表示原分数对干扰坐标的系数。局部模型写为
```math
\ell_0(e,\nu)=f(e)+\boldsymbol\gamma^\top\nu,
\qquad
t(e,\nu)=D\nu+He.
```
对痕迹施加线性修正 $w^\top t$，并允许一个输入无关常数 $\kappa$，得到
```math
\ell_w(e,\nu)
=\ell_0(e,\nu)+w^\top t(e,\nu)+\kappa
=f(e)+(\boldsymbol\gamma+D^\top w)^\top\nu+w^\top He+\kappa.
```
精确对消干扰且保持原证据响应要求
```math
D^\top w=-\boldsymbol\gamma,\qquad H^\top w=0.
```
令 $Z\in\mathbb R^{d_t\times r}$ 的列构成 $\ker(H^\top)$ 的正交单位基，则 $w=Za$，其中 $a\in\mathbb R^r$。上述两个条件可同时满足，当且仅当
```math
\boldsymbol\gamma\in\mathrm{range}(D^\top Z).
```
这说明痕迹可以混入证据信息；关键在于是否仍存在既能识别判断相关干扰、又不响应证据的读出方向。

局部几何下界也刻画了严格保持证据方向的代价。令 $u_e$、$u_n$ 分别为同一表示点 $h_0$ 处的单位证据切向和单位干扰切向，两者夹角为 $\theta$。若可微表示编辑 $T$ 的雅可比矩阵 $J_T(h_0)$ 满足 $J_T(h_0)u_e=u_e$ 且 $J_T(h_0)u_n=0$，则
```math
\|I-J_T(h_0)\|_2\geq\csc\theta.
```
其中 $\|\cdot\|_2$ 表示谱范数。该结论只适用于同一点严格保留原证据切向、同时消去干扰切向的局部约束；若允许证据重编码或仅要求最终输出保持，应按相应任务约束评价。

一般函数修正要求需要抵消的标量贡献可由同一个痕迹函数确定。增加函数容量能拟合更复杂的关系，但不能补足特征中缺失的信息。完整命题、证明及反例见附录 A。

## 4 实验

### 4.1 实验设置

本文的主训练配置为 E1 / Dataset A：2,540 对 MEGA-CDP 配对，A/P/C/F 各 635 对，全部来自拟合集的 80 个指南来源。E1 于 2026-11-11 根据独立开发集选择并冻结；选择过程未使用历史留出集、官方测试集或跨任务评价标签。E0 为历史训练基线，E2 和 E3 为 Dataset B 的数据规模及权重消融，不作为主配置。原 10,006 对混合来源配置及其 M4 扩展仅作为补充历史实验报告，不构成本文主训练设置。

实验分别考察三个方面：保持决定性证据时的干扰抑制、改变证据时的响应保留，以及原生任务正确性。前两者是行为机制指标，后者衡量方法是否改善实际判断，不能相互替代。我们在MEGA-CDP上开展受控配对与动态指南路径评价，并将冻结修正迁移至NLI4CT-2024和MedPIC。所有完整路径均依据模型实际预测自主遍历，不使用金标路径替代推断。

本节报告已经完成的Qwen3-8B实验。主干模型冻结，采用原实验固定提示与二元标签分数差。Qwen3-8B线性实例的具体头集合、痕迹秩、正则强度和证据预算在本节作为该模型的实验配置报告；这些数值不构成方法定义。其他模型分别依据自身内部激活、训练来源和开发集确定注意力头集合、标准化参数、子空间秩及修正参数，具体配置见对应模型的实验记录。第4.5—4.8节保留旧版结果，第4.9节单列2026-11-11新配对训练、五折和收益拆分；各自使用原评价分母。

### 4.2 数据集与划分

本文采用一个同任务主训练数据集和两个冻结迁移评价数据集。MEGA-CDP 是唯一的核心训练数据来源；NLI4CT-2024 和 MedPIC 用于跨任务评价，不参与 M1–M3 的训练或超参数选择。MedPIC 预先划定的 93 题开发集仅用于其独立的 M4 集合校准；188 题冻结测试集不参与校准。

| 数据集 | 任务 | 本文评价范围与主要指标 |
|---|---|---|
| MEGA-CDP | 根据患者事实和指南条件逐节点判断 | 拟合集 80 个指南来源；开发集 20 个来源；历史受控留出 40 个来源、2,293 个 A/P 对、40 个 C/F 对及 806 条动态路径；另评价官方 7,458 条动态路径 |
| NLI4CT-2024 | 临床试验报告与声明的文本推断 | 正式测试 5,500 条声明（500 条原声明），4,136 组标签保持关系、864 组标签翻转关系；报告准确率、宏平均 F1 及关系响应 |
| MedPIC | 患者特异的用药规则判断 | 官方 467 道题、2,335 个选项；单列既有 93 题开发集和 188 题冻结测试；报告整题答案集合正确率及选项级结果 |

**表2：数据集与评价单位。** MEGA 受控留出与官方测试属于不同总体；NLI 和 MedPIC 的原生标签关系未经逐对语义核验时，不等同于本文定义的纯 A/P 干扰或 C/F 证据干预。

**主训练配对。** 主实验使用数据版本 20261111-sol-agents-v2 的均衡子集 Dataset A。该数据仅从 MEGA-CDP 的 80 个拟合来源构建，包含 2,540 对配对样本：

| 类型 | 数量 | 干预对象 | 标签关系 |
|---|---:|---|---|
| A | 635 | 仅改写当前节点之后的处置 | 保持 |
| P | 635 | 仅改写与当前节点无关的背景信息 | 保持 |
| C | 635 | 改写当前指南条件或定义 | 翻转 |
| F | 635 | 改写患者的一项决定性事实 | 翻转 |
| **合计** | **2,540** | — | **1,270 对保持，1,270 对翻转** |

Dataset A 按指南来源、语言和决策树复杂度分层抽样；在各层内优先选择不同病例，再选择额外配对。四类均覆盖全部 80 个拟合来源；A、P、C、F 的不同病例数分别为 635、622、635 和 635。该设计使四类样本数量相等，并降低同一病例的重复贡献。完整合格池 Dataset B 共 6,345 对，仅用于第 4.9 节的数据规模与权重消融，不作为主训练配置。

**质量控制与数据边界。** 配对构建包括独立智能体语义审核，以及对数据结构、来源、标签关系、编辑范围、重复样本和来源隔离的确定性检查。本轮没有临床专家逐对审核；已准备的 150 条人工抽查清单尚未实际审核。因此，本文将其描述为经过独立语义审核和程序验证的研究配对数据，不称为临床专家认证数据。训练、开发和受控留出按指南来源隔离，分别包含 80、20 和 40 个来源。历史留出和官方测试均有既往使用记录，不作为全新独立确认。

NLI4CT 和 MedPIC 均只按各自原生任务报告。NLI 标签保持不自动表示证据未变，标签翻转也不自动表示本文定义的 C/F；MedPIC 同侧选项关系同理。MEGA 以完整路径和叶节点均正确为成功条件；NLI 在 500 条原声明上报告宏平均 F1，并对全部 5,500 条声明计分；MedPIC 以 467 题的选项集合全对为主指标，其中开发题和冻结测试题分别报告。详细划分、来源与计分定义见附录 B。

### 4.3 基础模型与比较方法

主实验均以冻结的Qwen3-8B为基础。比较方法包括原模型、仅截距校准、固定比例缩放、监督线性读出、LEACE、ITI、LoFiT和LoRA。ASTER-CR以16维线性修正作为主方法；函数容量与修正位置的扩展比较见第5.2节。

| 对照类型 | 主比较方法 | 论证职责 |
|---|---|---|
| 原始与校准控制 | 原模型、仅截距、固定比例缩放（解析控制） | 区分输入相关修正、阈值偏置与整体分数幅度变化 |
| 监督读出与概念擦除 | 监督线性读出、LEACE | 比较任务监督读出与线性概念擦除 |
| 表示与权重干预 | ITI、LoFiT、LoRA | 比较激活干预、表示微调和权重微调 |
| 本文主方法 | ASTER-CR（16维线性） | 检验A/P差分修正与C/F证据约束的联合效果 |

**表3：主实验比较方法及其论证职责。** 表4在同一MEGA受控留出上报告各方法的分数响应、证据保持和动态路径结果。各方法的训练监督与实现配置见附录B。

### 4.4 评价指标

我们从三个方面评价模型忠实性：对无关信息保持稳定、对决定性证据变化作出正确响应，以及完成原始任务。连续分数指标使用任务级校准前的分数；准确率和路径指标使用包含M4在内的完整推理流程。

**分数与配对。** 原模型分数记为 $\ell_0(x)$，方法 $m$ 在任务校准前的分数记为 $\ell_m(x)$。对第 $i$ 组反事实配对，左右输入为 $x_i^L,x_i^R$。定义两端分数差及其修正量：

\[
\Delta\ell_{0,i}=\ell_0(x_i^R)-\ell_0(x_i^L),\qquad
\Delta\ell_{m,i}=\ell_m(x_i^R)-\ell_m(x_i^L),\qquad
\Delta c_{m,i}=\Delta\ell_{m,i}-\Delta\ell_{0,i}.
\]

其中，$\Delta\ell_{0,i}$ 是原模型的配对分数差，$\Delta\ell_{m,i}$ 是修正后分数差，$\Delta c_{m,i}$ 是方法改变的配对分数差。

**干扰不变性（A/P）。** A/P配对保持决定性证据和标签不变，只改变无关背景或后续处置。以修正模型在这些配对上的分数差均方根衡量干扰响应：

\[
\mathrm{RMS}_{AP}(m)
=\sqrt{\frac{1}{N_{AP}}\sum_{i\in\mathcal I_{AP}}
(\Delta\ell_{m,i})^2}.
\]

该值越小，模型对无关变化的响应越弱。干扰方差降低率（VR）作为辅助指标，完整定义见附录B.1。

**证据敏感性（C/F）。** C/F配对改变条件或患者的决定性事实，使金标准标签翻转。令 $d_i=y_i^R-y_i^L$，证据响应保留率定义为

\[
\mathrm{ER}(m)
=100\times
\frac{\sum_{i\in\mathcal I_{CF}}d_i\Delta\ell_{m,i}}
{\sum_{i\in\mathcal I_{CF}}d_i\Delta\ell_{0,i}}.
\]

同时报告证据修正RMS，以衡量方法对原模型C/F配对分数差的改动幅度：

\[
\mathrm{RMS}_{\mathrm{corr},CF}(m)
=\sqrt{\frac{1}{N_{CF}}\sum_{i\in\mathcal I_{CF}}
(\Delta c_{m,i})^2}.
\]

ER衡量全部C/F配对上的有向证据响应总量相对原模型的保留比例；证据修正RMS衡量方法对证据配对分数差的改变量级。C/F双端全对率进一步评价两端预测与金标准标签的一致性。方向退化和超预算比例用于描述逐对证据响应的安全边界，完整定义见附录B.1。

**端到端任务效用。** MEGA-CDP报告完整动态路径正确率；路径全对要求模型从根节点出发，按自身预测动态遍历的带标签节点序列及终点叶节点均与金标准一致。NLI4CT-2024报告原始声明宏平均F1和正式测试准确率；MedPIC报告整题答案集合完全正确率。所有任务分别报告，不合并为单一总分；方法差值按数据来源分组进行配对重采样。

### 4.5 MEGA受控比较：选择性分数修正

本节报告MEGA-CDP来源隔离受控留出上的主比较结果，包括2,293对A/P配对、40对C/F配对和806条动态路径。表4集中呈现原始与校准控制、外部比较方法和ASTER-CR线性主方法；函数容量与高维集成结果见第5.2节及附录C.3。

| 类别 | 方法 | A/P干扰响应RMS↓ | C/F证据响应保留率ER（%，参考100） | C/F证据修正RMS↓ | C/F双端全对↑ | 动态路径全对↑ |
|---|---|---:|---:|---:|---:|---:|
| 参照 | 原模型 | 2.6282 | 100.00 | 0.0000 | 19/40 | 768/806 |
| 解析控制 | 固定比例缩放（a=0.95） | 2.4967 | 95.00 | 1.8149 | 19/40 | 768/806 |
| 校准控制 | 仅截距校准 | 2.6282 | 100.00 | 0.0000 | 19/40 | 767/806 |
| 监督读出 | 监督线性读出 | 1.3591 | 31.06 | 25.3734 | 20/40 | 737/806 |
| 概念擦除 | LEACE | 2.5524 | 98.73 | 1.1118 | 19/40 | 767/806 |
| 表示干预 | ITI | 1.1752 | 8.14 | 33.4525 | 0/40 | 45/806 |
| 表示微调 | LoFiT | 2.4911 | 91.93 | 4.6130 | 19/40 | 769/806 |
| 权重微调 | LoRA | 2.3588 | 50.09 | 20.8974 | 20/40 | 773/806 |
| 本文主方法 | ASTER-CR（16维线性） | 2.5730 | 99.60 | 0.2539 | 19/40 | 767/806 |

**表4：MEGA-CDP同一来源隔离受控留出上的主比较结果。** A/P为2,293对，C/F为40对，动态路径为806条；RMS单位均为logit。ER以原模型100%为参照。来源：[线性方法与外部对照指标](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/metrics.json)、[共同计分器](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/evaluate.py)。

ASTER-CR的主结果位于低幅度、证据保持的修正区间。16维线性配置将A/P干扰方差降低4.10%，保留99.60%的C/F有向证据响应，证据修正RMS为0.2539 logit。相较固定比例缩放，ASTER-CR的C/F证据修正RMS降低86.0%，ER提高4.60个百分点，体现了输入相关修正对干扰贡献的选择性估计。相较LEACE，ASTER-CR的ER高0.86个百分点，证据修正RMS由1.1118降至0.2539 logit；两种方法对应不同的干扰抑制幅度。

在完整动态路径上，ASTER-CR全对767/806；C/F双端全对为19/40。该结果与连续分数指标共同刻画主方法在受控MEGA评价中的表现。

固定比例缩放将原分数乘以0.95且不加截距。根据已归档的A/P原响应RMS=2.62815198、C/F原响应RMS=36.29733329，可解析得到干扰RMS=2.4967、证据修正RMS=1.8149、VR=9.75%和ER=95.00%。正比例缩放保持零阈值下的二元预测，因此配对全对和动态路径结果与原模型一致。[解析控制完整计算](../../reports/aster_cancel_story_20261109/analytic_scale_control_095.json)。

40个C/F配对中，线性方法有11对超过拟合预算，最大改动为0.5569 logit；方向退化数为0。A/P双端全对为2,266/2,293，原模型为2,258/2,293，仅截距校准为2,267/2,293。

### 4.6 三个完整任务的冻结迁移结果

| 方法 | MEGA官方路径全对：7,458例 | NLI正式声明准确率：5,500条 | MedPIC整题集合全对：467题 |
|---|---:|---:|---:|
| 原模型 | 94.36%（7,037/7,458） | 74.05%（4,073/5,500） | 20.99%（98/467） |
| 固定比例缩放（a=0.95；解析控制） | 94.36%（7,037/7,458） | 74.05%（4,073/5,500） | 20.99%（98/467） |
| 仅截距校准 | 94.37%（7,038/7,458） | 73.33%（4,033/5,500） | 未归档 |
| ASTER-CR（线性） | 94.31%（7,034/7,458） | 73.29%（4,031/5,500） | 14.99%（70/467） |
| ASTER-CR（99维强保护集成） | 94.36%（7,037/7,458） | 73.49%（4,042/5,500） | 16.70%（78/467） |
| ASTER-CR（99维平衡集成） | 94.41%（7,041/7,458） | 73.60%（4,048/5,500） | 16.70%（78/467） |
| 仅痕迹MLP（探索） | 94.45%（7,044/7,458） | 73.45%（4,040/5,500） | 未归档 |

**表5：Qwen3-8B在三个完整任务总体上的冻结评价。** MEGA、NLI和MedPIC分别按7,458条路径、5,500条声明和467道题计分。固定比例缩放由正比例变换保持类别预测的性质解析得到。MedPIC逐例记录见[迁移研究归档](../../results/aster_cr_transfer_20261110.zip)。表中列出各任务完整评价总体上具有归档预测的配置；受控806路径上的外部基线结果见表4及附录B.5。

MEGA官方测试中，原模型路径准确率为94.36%，ASTER-CR线性配置为94.31%，99维平衡配置为94.41%。平衡配置比原模型多4条正确路径，差值为0.054个百分点，来源簇配对区间为[−0.114, 0.212]个百分点；强保护配置的总正确数与原模型相同，包含26例纠正和26例新增错误。

NLI正式测试中，原模型准确率为74.05%，ASTER-CR线性配置为73.29%，99维平衡配置为73.60%。MedPIC整题集合全对率分别为20.99%（原模型）、14.99%（线性配置）和16.70%（两种99维配置）。第4.7和4.8节进一步报告这两个任务的原生关系指标与选项级证据响应。

### 4.7 NLI原生指标

| 方法 | 原题宏平均F1（%）↑ | 正式测试准确率（%）↑ | 标签保持一致率（%）↑ | 条件翻转忠实性（%）↑ | 条件成功数/合格关系数 | 全部翻转对双端全对↑ |
|---|---:|---:|---:|---:|---:|---:|
| 原模型 | 75.57 | 74.05（4,073/5,500） | 78.36 | 79.11 | 534/675 | 534/864 |
| 固定比例缩放（a=0.95；解析） | 75.57 | 74.05（4,073/5,500） | 78.36 | 79.11 | 534/675 | 534/864 |
| 仅截距校准 | 74.27 | 73.33（4,033/5,500） | 77.54 | 75.86 | 528/696 | 528/864 |
| ASTER-CR（线性） | 74.27 | 73.29（4,031/5,500） | 77.47 | 75.86 | 528/696 | 528/864 |
| ASTER-CR（99维强保护集成） | 74.93 | 73.49（4,042/5,500） | 78.41 | 76.71 | 527/687 | 527/864 |
| ASTER-CR（99维平衡集成） | 75.14 | 73.60（4,048/5,500） | 78.36 | 77.63 | 531/684 | 531/864 |
| 仅痕迹MLP（探索） | 74.93 | 73.45（4,040/5,500） | 78.36 | 76.71 | 527/687 | 527/864 |

**表6：NLI4CT-2024原生任务指标。** 原声明500条，正式测试5,500条，标签保持关系4,136对，标签翻转关系864对。预测由归档结果重新计分；固定比例缩放由解析性质得到。[复算脚本](https://github.com/chunxi-alpc/TRIAGE/blob/0694980caa971260bff8f78962d446c5ba4a51cf/reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/nli_recomputed_metrics.py)、[完整计数](https://github.com/chunxi-alpc/TRIAGE/blob/0694980caa971260bff8f78962d446c5ba4a51cf/reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/nli_recomputed_metrics.json)。

正比例缩放将连续分数差统一乘以0.95，原声明宏平均F1和分类结果保持不变。ASTER-CR线性配置在翻转关系上的ER为99.48%，证据修正RMS为0.2926 logit；99维平衡配置分别为97.38%和1.4982 logit。原模型的宏平均F1和准确率为75.57%和74.05%，线性配置为74.27%和73.29%，平衡配置为75.14%和73.60%。表6同时呈现连续关系响应与原生分类结果，分别刻画证据响应和任务推断表现。

### 4.8 MedPIC：选项级证据响应与整题集合表现

| 方法 | 翻转证据响应保留率（%），参照100 | 翻转证据修正RMS↓ | 翻转选项对双端全对↑ | 选项准确率（%）↑ | 整题集合全对↑ |
|---|---:|---:|---:|---:|---:|
| 原模型 | 参照 | 参照 | 25/244 | 68.65 | 20.99%（98/467） |
| 固定比例缩放（a=0.95；解析） | 95.00 | 0.05×原始翻转响应RMS† | 25/244 | 68.65 | 20.99%（98/467） |
| ASTER-CR（线性） | 100.45 | 0.1179 | 31/244 | 65.82 | 14.99%（70/467） |
| ASTER-CR（99维平衡集成） | 93.82 | 0.9937 | 32/244 | 67.02 | 16.70%（78/467） |
| ASTER-CR（99维强保护集成） | 96.91 | 0.4968 | 33/244 | 66.85 | 16.70%（78/467） |

**表7：MedPIC选项级证据响应与整题集合结果。** 全部467题、2,335个选项；既有配对映射包含244个标签翻转选项对。选项对双端全对、选项准确率和整题集合全对分别按配对、选项和题目计分。MedPIC逐例分数见[迁移研究归档](../../results/aster_cr_transfer_20261110.zip)。RMS单位为logit。

线性配置的翻转证据响应保留率为100.45%，证据修正RMS为0.1179 logit，翻转选项对双端全对为31/244；选项准确率为65.82%，整题集合全对为70/467。99维平衡配置和强保护配置的整题集合全对均为78/467，分别对应93.82%和96.91%的证据响应保留率。

MedPIC的选项级响应与整题集合解码刻画不同层级的任务行为。216个标签保持选项对同时改变患者事实，体现原生选项关系中的患者信息变化。历史合法集合解码使用93题开发集拟合校准，并在开发集外的324道固定错误题上评价；188题冻结测试和任务内适配结果见[实验总表](experiments.zh.md)。

### 4.9 新配对训练、五折验证与数据及权重消融（2026-11-11）

本节比较同一 MEGA-CDP 任务内的三种训练配置。**E1 / Dataset A 是主配置**，使用 2,540 对配对样本（A/P/C/F 各 635 对），并采用类别与来源均衡权重。E2 使用完整合格池 Dataset B（6,345 对）及类别、来源均衡权重；E3 使用相同 Dataset B，但采用自然样本权重。E2 和 E3 均作为消融，不替代 E1。E0 复用历史全量模型及各折 M1/M2，并补齐缺失的五折 M3 和截距，作为历史基线；E0 与新组的差异包含训练流程变化。各组固定 Qwen3-8B、51 个注意力头、16 维子空间、正则系数 λ=0.01 和证据预算 δ=0.2923349；A/P 回归权重总和为 4,887，A 与 P 各占一半，全部训练 C/F 配对施加逐对硬约束。[数据构建报告](../../reports/aster_cr_pair_dataset_20261111/REPORT.zh.md)；[全量训练与开发集选择](../../reports/aster_cr_retrain_paired_v2_20261111/REPORT.zh.md)。

E1 与 E2 的主配置选择仅依据 20 个来源的开发集，按自然完整路径准确率、自然节点准确率、C/F 方向退化数、C/F 超预算比例、A/P 方差降低率的顺序比较，最后以较小的均衡数据集作为平局规则。E1 的自然路径为 385/420，E2 为 384/420；两者自然节点均为 196/199，C/F 方向退化数均为 0；E1 有 1/60 对超预算，E2 为 0/60。由于自然路径准确率是预先规定的首要选择指标，且 E1 在开发集多正确 1 条自然路径，因此选择 E1。历史留出、官方测试、NLI4CT 和 MedPIC 标签均未用于该选择。该选择不表示 E1 在每个指标上都优于 E2，也不构成测试集上的显著性结论。

E0—E3 各有五个来源验证折，每折使用 64 个来源训练、16 个来源验证；每折均独立重新估计 M1、M2 和 M3。合并五折未参与对应训练的预测后，覆盖 80 个验证来源、761 个自然节点、1,712 条自然路径和 5,136 条干扰路径。为保证各组配对指标可比，五折验证均在相同的冻结 Dataset B 配对评价池上计算；这不改变 E1 的训练集仍为 Dataset A。以下区间按指南来源执行 2,000 次配对重采样，每次重新计算合并指标；不平均五折 VR 或区间。

为分离修正和校准，分别评价原始分数ℓ、仅配对修正ℓ+c、原截距诊断ℓ+b、独立截距基线ℓ+b₀及完整方法ℓ+c+b。b₀用原始分数和同一761个自然拟合节点拟合，每折仅使用其64个训练来源；目标任务直接使用源参数。原b在修正后分数上拟合，故ℓ+b只用于诊断。

| 版本 | A/P分数差RMS | VR | C/F修正RMS | ER | 方向退化 | 超预算 | 自然节点 | 自然路径 | 干扰路径 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|原模型|2.6165|0.000%|0.0000|100.000%|0|0.000%|99.080%|90.479%|91.939%|
|独立截距基线|2.6165|0.000%|0.0000|100.000%|0|0.000%|99.080%|90.304%|91.219%|
|E0完整|2.5455|5.059%|0.4120|99.567%|1|46.796%|99.080%|90.304%|91.258%|
|E1仅配对|2.5665|3.498%|0.1225|99.984%|0|1.863%|99.080%|90.479%|91.959%|
|E1完整|2.5665|3.498%|0.1225|99.984%|0|1.863%|99.080%|90.304%|91.316%|
|E2完整|2.5653|3.719%|0.1136|99.981%|0|1.118%|99.080%|90.304%|91.258%|
|E3完整|2.5706|3.201%|0.1164|100.001%|0|1.341%|99.080%|90.304%|91.199%|

**五折冻结配对池的合并指标。** 自然节点正确率只使用有标签节点，完整路径按各版本的分数自主遍历全部可达节点。验证C/F共1,342对，超预算不按训练约束过滤。原模型与截距版本的配对差相同，配对分数指标因而不变。

**配对修正与独立校准。** E1的A/P方差降低率为3.498%，95%区间[0.747%, 6.359%]，显示分数稳定性改善。相对独立截距基线，E1自然节点差值为0.000个百分点[0.000, 0.000]，自然路径为0.000[−0.170, +0.172]，干扰路径为+0.097[0.000, +0.214]。三个区间均包含0，尚未显示稳定的额外正确率收益。自然路径相对该基线纠正1例、误伤1例，干扰路径纠正7例、误伤2例；原模型与E1完整的自然路径分别为1549/1712、1546/1712。E1验证C/F的25/1342对超预算，方向退化数为0；新增E3五折全部训练C/F约束满足δ+1e−7。

**均衡权重。** E3−E2的自然节点及自然路径差值均为0.000个百分点，区间均为[0.000, 0.000]；干扰路径为−0.058个百分点[−0.141, +0.020]。当前未确认均衡权重的稳定正确率优势。这一比较同时改变类别与来源权重，不能分别归因。E3−E1的干扰路径为−0.117个百分点[−0.223, −0.021]，但两组还改变了数据规模，不能归为单一权重效应。

**MedPIC保持关系的下降环节。** 本轮目标任务采用已有映射子集：NLI原声明500条及其1,500个保持、864个翻转对；MedPIC本轮映射子集188题（历史全量为467题）、85个原生保持对、110个翻转对和1,880个背景对。这些分母与第4.6—4.8节的历史全量不同。

| MedPIC版本 | 整题正确数/188 | 原生保持关系正确率（85对） | 原生翻转关系正确率（110对） | 背景保持关系正确率（1880对） | 平均选中选项数 | “以上均非”选择比例 |
|---|---:|---:|---:|---:|---:|---:|
|原模型|70/188|83.53%|17.27%|92.66%|1.218|60.64%|
|独立截距基线|66/188|69.41%|31.82%|95.11%|1.394|56.38%|
|E1仅配对|70/188|90.59%|12.73%|93.62%|1.191|61.70%|
|E1完整|66/188|69.41%|30.91%|95.43%|1.378|56.91%|

E1加入截距相对仅配对修正使85个原生保持对的关系正确率下降21.176个百分点，95%区间[−31.522, −10.869]；18对由正确变错误，没有错误改对。完整方法和独立截距基线的该项正确率均为69.41%。合法答案集合按选项分数之和排序，对每个选项统一加b会使包含k个选项的集合增加k·b，因此截距可以改变多选集合与“以上均非”的排序。本次下降主要发生在这一环节；背景1880对的保持关系应单列，不能据原生85对概括所有保持配对。

本轮开发集、历史40来源、已有60来源及两目标任务的完整拆分、纠正与误伤、来源区间和模块诊断见[完整报告](../../reports/aster_cr_diagnostics_e3cv_20261111/REPORT.zh.md)、[指标与区间](../../reports/aster_cr_diagnostics_e3cv_20261111/source_bootstrap.csv)及[逐例预测入口](../../reports/aster_cr_diagnostics_e3cv_20261111/README.md)。这些评价总体均有历史使用记录，本轮属于既有数据上的对照；新增五折预测仅保证验证来源未参与对应折训练。


### 4.10 补充历史实验：混合来源配置与 M4 校准（非主训练设置）

本节保留已完成的 10,006 对混合来源实验，作为方法扩展的历史记录，不纳入本文主训练配置或核心证据。该配置由 MEGA-CDP 的 A/P/C 配对与 MedEinst 的 4,368 对 F 配对组成。随后开展的任务兼容性审计确认，进入该配置的 MedEinst 提示均为固定两候选鉴别诊断任务，而非 MEGA-CDP 的指南条件真值判断；对 4,368 对进行程序普查，严格同任务兼容类别为 0。另对 600 对进行分层 AI 语义审查，结果为 S3 512 对、S4 81 对、S2 7 对。该审查没有临床专家参与，不能解释为医学标签错误率；但足以说明这些配对不能作为本文核心 F 类的同任务监督。因此，混合来源结果仅作探索性补充，不用于支持主任务中的证据保护结论。[MedEinst 任务兼容性审计](https://github.com/chunxi-alpc/TRIAGE/blob/52f3338/reports/aster_cr_medeinst_alignment_audit_20261112/REPORT.zh.md)。

该历史配置增加 M4 校准后，原报告记录的五折干扰路径差值为 +0.448 个百分点，95% CI [-0.135, +1.072]；自然路径差值为 +0.350 个百分点，95% CI [-0.459, +1.088]。Brier 分数由 0.00841344 降至 0.00718277，log loss 由 0.0968616 降至 0.0335027。原报告中的 E3→E4 指混合来源配置下的校准扩展，与第 4.9 节 MEGA-only 的 E3 数据权重消融不是同一实验编号体系，故不作直接合并解释。

MedPIC 冻结 188 题的历史整题正确率由 36.70% 变为 39.36%，差值区间包含零；官方 467 题包含开发校准题，只作完整覆盖的描述性评价。NLI4CT 官方 5,500 条结果及该历史配置的全部消融、置信区间和错误转移见[补充实验报告](../../reports/aster_cr_final_method_10006_20261112/REPORT.zh.md)。

## 5 机制分析与消融

### 5.1 证据约束对修正行为的作用

| 设置 | 干扰方差降低率（%）↑ | 证据响应保留率（%），参考100 | 证据修正RMS↓ | 动态路径全对↑ |
|---|---:|---:|---:|---:|
| ASTER-CR（线性） | 4.10 | 99.60 | 0.2539 | 767/806 |
| 去掉截距校准 | 4.10 | 99.60 | 0.2539 | 769/806 |
| 去掉证据约束 | 30.61 | 91.67 | 3.3240 | 769/806 |
| 去掉证据正交投影 | 2.16 | 100.10 | 0.1826 | 767/806 |
| 随机子空间，种子11 | 1.53 | 99.80 | 0.1506 | 767/806 |
| 痕迹与证据坐标合并，32维 | 4.03 | 99.56 | 0.2740 | 767/806 |
| 32维且去掉证据约束 | 84.74 | 9.33 | 33.9492 | 460/806 |

**表8：证据约束与子空间设计的组件比较。** RMS单位为logit；各配置使用相同受控留出。[归档指标](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/metrics.json)。

证据约束控制干扰修正对C/F响应的影响。主线性方法的VR为4.10%，ER为99.60%，证据修正RMS为0.2539 logit。去掉证据约束后，VR升至30.61%，ER降至91.67%，证据修正RMS增至3.3240 logit；32维无约束配置的VR达到84.74%，ER为9.33%，动态路径全对为460/806。该对照显示，证据预算将干扰抑制限制在证据响应可承受的修正范围内。

正交投影与逐对约束分别作用于特征构造和最终分数修正。投影减少证据变化进入干扰特征，逐对约束限制C/F配对上的实际改变量。去掉正交投影后，ER为100.10%，VR为2.16%；保留投影但去掉逐对约束时，VR和ER呈现更大的偏离。两项机制共同确定修正方向和修正幅度。

截距校准保持所有连续配对指标不变，并将路径全对从767/806调整为769/806。M3决定输入相关分数修正，M4单独调整决策阈值。

### 5.2 函数容量与修正位置

| 位置与函数 | 修正系数，不含独立截距 | 干扰方差降低率（%）↑ | 证据响应保留率（%），参考100 | 证据修正RMS↓ | 路径正确数或种子范围 |
|---|---:|---:|---:|---:|---:|
| 输出线性修正 | 16 | 4.10 | 99.60 | 0.2539 | 767/806 |
| 输出MLP，16维输入 | 576 | 14.59±0.24 | 98.49±0.17 | 0.6491±0.0479 | 770—771/806 |
| 线性表示编辑 | 256 | 3.48±1.11 | 99.69±0.11 | 0.1642±0.0576 | 各种子均767/806 |
| MLP表示编辑 | 1,056 | 13.12±0.17 | 98.43±0.20 | 0.6704±0.0783 | 768—769/806 |

**表9：共享16维痕迹和拟合配对下的函数容量与修正位置比较。** 三种训练式配置报告三个训练种子的均值±样本标准差；线性输出为固定凸优化解。路径列报告种子间范围。各方法自然节点均为368/372；所有行另有一个独立截距，RMS单位为logit。[逐种子结果](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/seed_summary.json)。

在共享16维痕迹下，提高函数容量扩大了可估计的干扰贡献范围。输出MLP的VR为14.59%，ER为98.49%；MLP表示编辑的VR为13.12%，ER为98.43%。线性输出修正的VR为4.10%，ER为99.60%；线性表示编辑的VR为3.48%，ER为99.69%。这些结果呈现了函数容量、修正位置与证据改动之间的定量关系。

输出MLP与表示MLP具有相近的干扰降低率和证据响应保留率。输出修正直接作用于决策分数；表示编辑通过修改末层残差影响后续读出。两种位置均可实现输入相关干预，具体选择由目标修正幅度和系统部署位置决定。

99维平衡集成结合16维干扰痕迹、16维证据坐标、64维残差主成分和3维分数特征，并采用集成训练与证据保护损失。其结果见附录C.3，与16维输出MLP共同展示了丰富特征和保护策略对修正曲线的影响。

### 5.3 痕迹传导与总干扰预测

在固定有效证据、仅改变后续处置的A/P配对中，记原模型完整分数差和16维痕迹坐标差为
\[
z_i=\Delta\ell_{0,i}
=\ell_0(x_i^R)-\ell_0(x_i^L),\qquad
\Delta t_i=t(x_i^R)-t(x_i^L).
\]
令 $h_i^L=h(x_i^L)$、$h_i^R=h(x_i^R)$ 为两端选中注意力头的拼接激活，$\Delta\tilde h_i=(h_i^R-h_i^L)\oslash\sigma$。将标准化激活差投影到干扰痕迹子空间并还原原坐标，得到
\[
\Delta h_i^N=\sigma\odot(NN^\top\Delta\tilde h_i).
\]
其中，$\odot$ 表示逐元素乘法。记 $\ell_{\mathrm{patch},i}^{N}$ 为在左端输入的最终答案位置，将选中头激活从 $h_i^L$ 改为 $h_i^L+\Delta h_i^N$ 后继续前向得到的“是／否”logit差；直接修补效应定义为
\[
q_i=\ell_{\mathrm{patch},i}^{N}-\ell_0(x_i^L).
\]
对开发A类配对上的任意量 $v_i$，记 $\mathrm{RMS}(v)=\sqrt{N_A^{-1}\sum_i v_i^2}$，其中 $N_A=|\mathcal I_{\mathrm{dev},A}|$；$\bar z_{\mathrm{dev}}$ 表示该开发配对集合中 $z_i$ 的算术平均。$\mathrm{RMS}(q)/\mathrm{RMS}(z)$ 衡量该投影方向的直接修补效应相对于完整A/P分数差的幅度。

令 $\mathcal I_{\mathrm{fit},AP}$ 为拟合集A/P配对索引，$\mathcal I_{\mathrm{dev},A}$ 为来源隔离开发集A类配对索引。直接修补效应的决定系数为
\[
R^2_{\mathrm{patch}}
=1-\frac{\sum_{i\in\mathcal I_{\mathrm{dev},A}}(z_i-q_i)^2}
{\sum_{i\in\mathcal I_{\mathrm{dev},A}}(z_i-\bar z_{\mathrm{dev}})^2}.
\]
在拟合集上用岭回归从痕迹坐标差预测完整分数差：
\[
\hat a
=\arg\min_a
\sum_{j\in\mathcal I_{\mathrm{fit},AP}}
(z_j-a^\top\Delta t_j)^2+\lambda_r\|a\|_2^2,
\qquad
\hat z_i=\hat a^\top\Delta t_i,
\]
并在来源隔离的开发A类配对上计算。拟合集使用4,887个A/P配对，开发集A类包含1,194个配对。
\[
R^2_{\mathrm{trace}}
=1-\frac{\sum_{i\in\mathcal I_{\mathrm{dev},A}}(z_i-\hat z_i)^2}
{\sum_{i\in\mathcal I_{\mathrm{dev},A}}(z_i-\bar z_{\mathrm{dev}})^2}.
\]
该回归使用 $\lambda_r=0.01$，不额外拟合截距，也不对痕迹坐标再次标准化。直接修补效应刻画痕迹方向对分数的即时传导；岭回归刻画痕迹坐标对完整干扰分数差的预测能力。

| 同版真实机制诊断 | A：1,194对、20来源 | A/P：1,198对、20来源 |
|---|---:|---:|
| 痕迹直接修补RMS／完整干扰效应RMS | **13.91%** | 13.90% |
| 16维痕迹坐标预测完整分数差的 $R^2$ | **0.4063** | 0.4069 |

**表10：痕迹直接传导与总干扰预测。** 16维痕迹坐标在开发A类配对上的预测决定系数为0.4063；直接修补效应RMS为完整干扰效应RMS的13.91%。两项指标分别刻画预测信息和直接传导幅度。[回归记录](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/paired_fit/metrics.json)。

16维痕迹坐标解释了开发A类配对约40.63%的干扰分数差变异，直接修补效应占完整干扰效应RMS的13.91%。这表明痕迹包含超出其直接传导幅度的预测信息。M3据此从痕迹估计完整干扰贡献，再按C/F证据预算限制修正量。

### 5.4 随机对照与来源泛化

我们在同一Qwen3-8B表示缓存上比较选中痕迹、随机头和随机子空间，并在拟合来源内部执行五折交叉验证。随机对照各使用50个固定种子，预测目标均为原模型的A/P分数差。

| 对照 | 预测决定系数 \(R^2\) | 评价范围 |
|---|---:|---|
| 选中51头的16维正交痕迹 | 0.4063 | 开发集A类1,194对 |
| 随机16维子空间（50种子均值） | 0.3115 | 同一开发集 |
| 随机51头（50种子均值） | 0.0182 | 同一开发集 |
| 打乱训练分数差（50种子均值） | −0.0062 | 同一开发集 |
| 固定证据子空间的32维痕迹 | 0.5197 | 同一开发集 |
| 16维痕迹的小型网络（3种子均值） | 0.4486 | 同一开发集 |
| 每折重新选头与拟合的完整来源五折 | 0.3454 | 拟合80来源的留折A/P配对 |

**表11：痕迹预测对照与来源泛化结果。** 开发集A类包含1,194对；完整来源五折覆盖80个拟合来源。[完整实验报告](../../reports/aster_cancel_story_20261109/meeting/trace_validation/REPORT.zh.md)。

选中痕迹在开发集上的预测决定系数为0.4063，高于随机16维子空间的均值0.3115和随机51头的0.0182。完整来源五折的合并决定系数为0.3454，80个来源中62个来源的来源内决定系数为正。结果显示，痕迹中的干扰预测信息可跨来源复现。

### 5.5 方法作用范围

ASTER-CR修正的是非决定性输入对当前判断分数的贡献，判断语义仍由基础模型对指南条件和患者事实的理解决定。MEGA受控留出中，C/F双端全对为19/40；40个C/F配对中有11对超过拟合预算。两项结果界定了当前证据约束的评价范围：配对监督能够控制已观测证据改动，跨来源预算泛化需要更广泛的C/F覆盖。

配对分数差与动态路径分别评价局部响应和完整推理流程。M4校准改变决策阈值，因而单独报告其对自然节点和动态路径的影响。

## 6 结论

ASTER-CR提出一套分数级干扰修正框架：保留完整输入，以A/P配对学习输入相关的干扰贡献，以C/F配对约束证据响应，并独立校准决策阈值。理论分析给出混合痕迹下干扰对消与证据保持的共同可行条件，将指定判断的修正转化为可测量的分数级目标。

在Qwen3-8B的MEGA受控留出上，16维线性ASTER-CR将A/P干扰方差降低4.10%，保留99.60%的C/F有向证据响应，证据修正RMS为0.2539 logit，动态路径全对为767/806。痕迹分析显示，16维坐标解释了开发A类配对约40.63%的干扰分数差变异；80来源五折的合并决定系数为0.3454。MEGA官方路径准确率、NLI正式测试准确率和MedPIC整题集合全对率分别为94.31%、73.29%和14.99%。

本文的核心贡献是将干扰抑制、证据保持和阈值校准分解为可独立估计与评价的模块。ASTER-CR由此提供一种输入相关、证据受约束的决策分数修正方法，为医学指南条件判断的忠实性评价与干预建立了可复现的实验框架。

## 局限性

当前语言模型实测以Qwen3-8B为主。MEGA受控留出包含40个指南来源，C/F配对40对、覆盖12个来源；P、F与F0配对数量较少。配对语义审核由独立智能体完成，尚未进行临床专家逐对审核。

外部方法沿用各自归档的训练监督、超参数和搜索配置，训练预算与证据保护强度存在差异。NLI声明与变体共享基础试验材料；MedPIC选项级和整题级指标具有不同统计单位。当前结果主要界定Qwen3-8B及现有数据划分下的机制表现和任务迁移范围。

## 伦理说明

本文研究模型判断依赖，不提供临床决策建议。数据按发布条款使用；生成病例仅用于研究，不用于实际临床决策。配对的医学语义与标签仍需专业核验，完成状态如实报告。新增NLI复算清单仅保存公开记录标识、标签及配对关系，不复制临床文本。

## 参考文献

- Jacovi, A. & Goldberg, Y. (2020). *Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?* [ACL](https://aclanthology.org/2020.acl-main.386/).
- Veitch et al. (2021). *Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests*. [原文](https://arxiv.org/abs/2106.00545).
- Turpin et al. (2023). *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html).
- Lampinen et al. (2024). *Language models, like humans, show content effects on reasoning tasks*. [PNAS Nexus](https://doi.org/10.1093/pnasnexus/pgae233).
- Chen et al. (2026). *Benchmarking Clinical Decision Pathway Adherence in Large Language Models*. [原文](https://arxiv.org/abs/2608.26592).
- Jullien et al. (2023). *NLI4CT: Multi-Evidence Natural Language Inference for Clinical Trial Reports*. [EMNLP](https://aclanthology.org/2023.emnlp-main.1041/).
- Jullien, M., Valentino, M. & Freitas, A. (2024). *SemEval-2024 Task 2: Safe Biomedical Natural Language Inference for Clinical Trials*. [任务论文](https://aclanthology.org/2024.semeval-1.271/)、[勘误](https://aclanthology.org/2024.semeval-1.271e1.pdf)。
- Belrose et al. (2023). *LEACE: Perfect linear concept erasure in closed form*. [原文](https://arxiv.org/abs/2306.03819).
- Ravfogel et al. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection*. [ACL](https://aclanthology.org/2020.acl-main.647/).
- Ravfogel, S., Vargas, F., Goldberg, Y. & Cotterell, R. (2022b). *Adversarial Concept Erasure in Kernel Space*（PDF 题名 *Kernelized Concept Erasure*）. [EMNLP](https://aclanthology.org/2022.emnlp-main.405/).
- Akbari, R., Afshari, M. & Boddeti, V. N. (2025). *Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure*. [NeurIPS 正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/d55f39791f04745b2e0c8abebf3dd5d7-Paper-Conference.pdf)，[作者全文](https://arxiv.org/html/2603.07529v1).
- Li et al. (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*. [原文](https://arxiv.org/abs/2306.03341).
- Zhao et al. (2021). *Calibrate Before Use: Improving Few-Shot Performance of Language Models*. [ICML](https://proceedings.mlr.press/v139/zhao21c.html).
- Wu et al. (2024). *ReFT: Representation Finetuning for Language Models*. [原文](https://arxiv.org/abs/2404.03592).
- Yin, Ye & Durrett (2024). *LoFiT: Localized Fine-tuning on LLM Representations*. [原文](https://arxiv.org/abs/2406.01563).
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. [原文](https://arxiv.org/abs/2106.09685).
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. [ICML](https://proceedings.mlr.press/v70/guo17a.html).
- Zhou et al. (2024). *Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering*. [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/003e438cf04e3caf0a5c908495e484fe-Abstract-Conference.html)、[作者全文](https://arxiv.org/html/2309.17249v3).
- Shi et al. (2024). *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding*. [NAACL](https://aclanthology.org/2024.naacl-short.69/)、[作者代码](https://github.com/xhan77/context-aware-decoding).
- Liu, A. et al. (2021). *DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts*. [ACL](https://aclanthology.org/2021.acl-long.522/)、[作者代码](https://github.com/alisawuffles/DExperts).
- Ravfogel, S., Twiton, M., Goldberg, Y. & Cotterell, R. D. (2022a). *Linear Adversarial Concept Erasure*. [ICML](https://proceedings.mlr.press/v162/ravfogel22a.html)、[R-LACE作者代码](https://github.com/shauli-ravfogel/rlace-icml).
- Shao, S., Ziser, Y. & Cohen, S. B. (2023). *Gold Doesn't Always Glitter: Spectral Removal of Linear and Nonlinear Guarded Attribute Information*. [EACL及修订稿](https://aclanthology.org/2023.eacl-main.118/)、[SAL作者代码](https://github.com/jasonshaoshun/SAL).
- Chowdhury, S. B. R. et al. (2023). *Robust Concept Erasure via Kernelized Rate-Distortion Maximization*. [NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/hash/86bd650f85480c595ecab29081a3774e-Abstract-Conference.html)、[KRaM作者代码](https://github.com/brcsomnath/KRaM).
- Saillenfest, A. & Lemberger, P. (2025). *Nonlinear Concept Erasure: A Density Matching Approach*. ECAI 2025. [作者全文](https://arxiv.org/html/2507.12341v2)、[非线性LEOPARD作者代码](https://github.com/ToineSayan/non-LEOPARD).
- Chowdhury, S. B. R. & Chaturvedi, S. (2022). *Learning Fair Representations via Rate-Distortion Maximization*. [TACL](https://aclanthology.org/2022.tacl-1.67/).
- Chowdhury, S. B. R. et al. (2025). *Fundamental Limits of Perfect Concept Erasure*. [AISTATS](https://proceedings.mlr.press/v258/chowdhury25a.html)、[作者代码](https://github.com/brcsomnath/PEF).
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. [ACL](https://aclanthology.org/2024.acl-long.828/)、[CAA作者代码](https://github.com/nrimsky/CAA).
- Lee, B. W. et al. (2025). *Programming Refusal with Conditional Activation Steering*. [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html)、[CAST作者代码](https://github.com/IBM/activation-steering).
- Liu, S.-Y. et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. [ICML](https://proceedings.mlr.press/v235/liu24bn.html)、[作者代码](https://github.com/NVlabs/DoRA).
- Gao et al. (2024). *Parameter-Efficient Fine-Tuning with Discrete Fourier Transform*. [ICML](https://proceedings.mlr.press/v235/gao24o.html)、[FourierFT作者代码](https://github.com/Chaos96/fourierft).
- Stolfo, A., Balachandran, V., Yousefi, S., Horvitz, E. & Nushi, B. (2025). *Improving Instruction-Following in Language Models through Activation Steering*. [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c3262a4c965ba9888f120d4f9e13478-Abstract-Conference.html)、[作者代码](https://github.com/microsoft/llm-steer-instruct)。
- Wu, Z. et al. (2025). *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*. [ICML 2025](https://proceedings.mlr.press/v267/wu25a.html)。
- Wu, Z., Yu, Q., Arora, A., Manning, C. D. & Potts, C. (2025). *Improved Representation Steering for Language Models (RePS)*. [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb1ef82926376d252dde00d5dd909f4b-Abstract-Conference.html)。
- Heyman, G. & Vandeputte, F. (2026). *Steer Like the LLM: Activation Steering that Mimics Prompting*. [ICML 2026](https://proceedings.mlr.press/v306/heyman26a.html)。
- Cheng, S. S., Wiegreffe, S. & Manocha, D. (2026). *What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal*. [EMNLP 2026](https://aclanthology.org/2026.emnlp-main.79/)。
- Venhoff, C., Arcuschin, I., Torr, P., Conmy, A. & Nanda, N. (2026). *Base Models Know How to Reason, Thinking Models Learn When*. [ICML 2026](https://proceedings.mlr.press/v306/venhoff26a.html)。
- Xu, S. (2026). *From Geometry to Behavior: How Instruction Exposure Unlocks Latent Rhetorical Directions in LLMs*. [arXiv预印本](https://arxiv.org/abs/2610.37685)。
- 官方模型资料：[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)、[Gemma-2-9B-IT](https://huggingface.co/google/gemma-2-9b-it)、[MedGemma-27B](https://huggingface.co/google/medgemma-27b-text-it)、[Gemma-3-27B-IT](https://huggingface.co/google/gemma-3-27b-it)。

**仓库内证据**（chunxi-alpc/TRIAGE）

- [R2] ASTER 全部实验汇总与问题定位. https://github.com/chunxi-alpc/TRIAGE/blob/742e6b32c59fade4ddd2639314785209e85f1006/reports/aster_synthesis_20261106/ANALYSIS.zh.md
- [R3] 未筛选方向的全局均值与同事实参考对照. https://github.com/chunxi-alpc/TRIAGE/blob/742e6b32c59fade4ddd2639314785209e85f1006/results/aster_reference_anchor_20261106/REPORT.zh.md
- [R5] 冻结 M1：机制诊断与原文基础模型试测. https://github.com/chunxi-alpc/TRIAGE/blob/742e6b32c59fade4ddd2639314785209e85f1006/results/m1_diagnostics_20261106/REPORT.zh.md
- [R6] ASTER-CR 实现：experiments/aster_cr/mechanisms.py、readout.py. https://github.com/chunxi-alpc/TRIAGE/tree/742e6b32c59fade4ddd2639314785209e85f1006/experiments/aster_cr
- [R8] ASTER 中文主稿. https://github.com/chunxi-alpc/TRIAGE/blob/742e6b32c59fade4ddd2639314785209e85f1006/docs/ASTER/manuscript.zh.md

**当前方法与结果。** 有约束拟合记录见[证据保护目录](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/evidence_preserving/)，受控结果见[完整计分目录](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/)，新增NLI缓存计分见[复算材料](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/)。实际合成机制拟合见[实验归档](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/simulated_study/)，不合入真实结果。MedPIC全467题的四种方法整题记录、2,335条选项分数及配对结果见[迁移研究归档](../../results/aster_cr_transfer_20261110.zip)，包内aster_medpic_20261109目录保留原报告和逐例文件。

- MedPIC-Bench. *Evaluating Counterfactual Sensitivity to Patient Information in Medication-Safety Reasoning*. [论文](https://arxiv.org/abs/2608.03028)、[作者数据卡](https://huggingface.co/datasets/TIM0927/MedPIC-Bench)。
- Hao et al. (2025). *MedPAIR: Measuring Physicians and AI Relevance Alignment in Medical Question Answering*. [作者项目](https://medpair.csail.mit.edu/)、[论文](https://arxiv.org/abs/2505.24040)。
- Chen, W., Zhu, Z., Huang, G. & Wang, W. (2026). *MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis*. [论文](https://arxiv.org/abs/2601.06636)、[官方说明](https://github.com/zhui711/MedEinst/blob/main/README.md)。
- Li et al. (2025). *MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models*. [论文](https://arxiv.org/abs/2505.11613)、[数据](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K)。
- Jin et al. (2024). *Matching patients to clinical trials with large language models*. [Nature Communications](https://doi.org/10.1038/s41467-024-53081-z)、[专家条款数据](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)。

## 附录 A 详细证明与理论条件

标量贡献的识别条件说明对消何时成立，局部几何下界解释保留原证据方向时的编辑代价。方法的实际效果与成本通过同目标对照评价。

### A.1 命题 1：保留证据切向的擦除具有局部几何代价

为了分析线性和非线性擦除在相同局部约束下的代价，设 $u_e$、$q$ 为正交单位向量，$u_e$ 属于证据切空间，单位干扰方向 $u_n$ 与其夹角为 $\theta$：
```math
u_n=\cos\theta\,u_e+\sin\theta\,q,\qquad
0<\theta\leq\pi/2.
```
在线性编辑 $E$ 中，若 $Eu_e=u_e$ 且 $Eu_n=0$，则
```math
\|I-E\|_2\geq\csc\theta.
```
对任意在 $h_0$ 可微的非线性编辑 $T$，令 $J_T(h_0)$ 为其在 $h_0$ 处的雅可比矩阵。若它在同一点保持证据切向且消去干扰切向，即 $J_T(h_0)u_e=u_e$、$J_T(h_0)u_n=0$，同样有
```math
\boxed{\|I-J_T(h_0)\|_2\geq\csc\theta.}
```
其中 $\|\cdot\|_2$ 为谱范数。证明：由 $J_Tu_n=0$、$J_Tu_e=u_e$ 得 $J_Tq=-\cot\theta\,u_e$，故 $(I-J_T)q=q+\cot\theta\,u_e$，其范数为 $\csc\theta$。线性情形取 $J_T=E$ 即得。

若允许误差 $\|J_Tu_e-u_e\|_2\leq\varepsilon_e$、$\|J_Tu_n\|_2\leq\varepsilon_n$，则三角不等式给出
```math
\|I-J_T\|_2\geq
\frac{[1-\cos\theta\,\varepsilon_e-\varepsilon_n]_+}{\sin\theta}.
```
其中 $[a]_+=\max(a,0)$。该命题说明，增加非线性并不自动消除“同点保持原有证据切向、同时压平干扰切向”的局部约束。它给出编辑改变量的局部最坏方向增益；实际误差取决于数据、误差方向及后续计算。只保持最终任务分数而允许证据表示重编码的编辑，采用的是更弱的保留条件，应通过对应任务目标评价。

$\theta=45^\circ$ 时，精确下界为 $\csc(45^\circ)=\sqrt 2\approx1.414$。历史约 1.412 的算子值 [R2] 仅作为不同实测条件下的近似参照，不作严格数值验证。

### A.2 命题 2：混合痕迹下的对消与证据保留

为了明确痕迹混入证据后何时仍可使用，考虑第3.4节的局部模型
```math
\ell_0(e,\nu)=f(e)+\boldsymbol\gamma^\top\nu,
\qquad
t(e,\nu)=D\nu+He.
```
其中，$e$、$\nu$ 分别为证据和干扰坐标，$D$、$H$ 分别描述干扰和证据进入痕迹的线性映射。加入线性修正及输入无关常数 $\kappa$ 后，
```math
\ell_w(e,\nu)
=\ell_0(e,\nu)+w^\top t(e,\nu)+\kappa
=f(e)+(\boldsymbol\gamma+D^\top w)^\top\nu+w^\top He+\kappa.
```
同时完全对消干扰、保持原证据响应，要求
```math
D^\top w=-\boldsymbol\gamma,\qquad H^\top w=0.
```
取 $Z$ 为 $\ker(H^\top)$ 的正交单位基，则 $w=Za$。上述两个条件可同时满足，当且仅当
```math
\boxed{\boldsymbol\gamma\in\mathrm{range}(D^\top Z).}
```
证明：$H^\top w=0$ 等价于 $w=Za$；代入干扰条件后，方程 $D^\top Za=-\boldsymbol\gamma$ 可解恰好等价于上述值域条件。最小范数解可写为
```math
w^*=-Z(D^\top Z)^+\boldsymbol\gamma,
```
其中 $(\cdot)^+$ 表示 Moore–Penrose 伪逆，$\|\cdot\|_\infty$ 表示向量最大绝对分量范数。

**痕迹混入证据并不自动使对消失效。** 所需条件是仍存在一个对证据不响应、又能识别判断相关干扰贡献的读出方向。$H=0$ 时，条件退化为 $\boldsymbol\gamma\in\mathrm{range}(D^\top)$，痕迹不必恢复全部干扰变量。若干扰与证据在可读特征中完全重合，精确保留可能只允许 $w=0$；在上述线性修正与精确证据保留要求下，此时无法取得非零对消。

**配对分别监督两个条件。** 对配对 $j$，记 $\Delta e_j=e(x_j^R)-e(x_j^L)$、$\Delta\nu_j=\nu(x_j^R)-\nu(x_j^L)$。在固定证据的 A/P 配对中，$\Delta t_j=D\Delta\nu_j$、$\Delta\ell_{0,j}=\boldsymbol\gamma^\top\Delta\nu_j$，因而差分损失监督 $\boldsymbol\gamma+D^\top w$。在固定干扰的证据配对中，$\Delta t_j=H\Delta e_j$，限制 $w^\top\Delta t_j$ 则直接限制所覆盖证据变化上的修正响应。

实际训练将 C/F 痕迹差按行堆叠为矩阵 $E_{CF}$，其第 $j$ 行为 $\Delta t_j^\top$，逐对预算写为 $\|E_{CF}w\|_\infty\leq\delta$。该约束给出训练配对的逐对响应改动界，并由独立留出的证据配对检验迁移。若 $E_{CF}$ 满列秩且 $\delta=0$，则只能有 $w=0$；报告非零预算下的干扰降低和证据改动，可以明确展示两者的实际权衡。

**只识别与判断相关的干扰仍是较弱目标。** 所需的是 $\boldsymbol\gamma^\top\nu$，而非重建全部 $\nu$；表示也可继续保存处置信息。岭惩罚进一步控制小奇异值与痕迹噪声带来的系数放大。本文的结构选择保持原前向，通过受约束的标量估计修复判断。

### A.3 预测能力与方差降低的关系

为了将诊断与最终效果对齐，令原配对分数差为 $z_i=\Delta\ell_{0,i}$，由痕迹预测并试图移除的干扰分数差为 $p_i$，修正后分数差（残差）为 $r_i=z_i-p_i=\Delta\ell_{m,i}$。对线性 M3，$p_i=-\hat w^\top\Delta t_i$；配对中心化常数在两端相减时抵消。中心化决定系数与干扰方差降低率分别定义为
```math
R^2=1-\frac{\sum_i r_i^2}{\sum_i(z_i-\bar z)^2},
\qquad
\mathrm{VR}_{\%}
=100\left[1-\frac{\mathrm{Var}(r)}{\mathrm{Var}(z)}\right].
```
这里的方差采用总体方差约定，且要求 $\mathrm{Var}(z)>0$；$\bar z$、$\bar r$ 分别表示相应配对集合中 $z_i$、$r_i$ 的算术平均。两者满足
```math
\mathrm{VR}_{\%}
=100\left(R^2+\frac{\bar r^{\,2}}{\mathrm{Var}(z)}\right).
```
因此，残差均值接近零时，VR 的百分数数值接近 $100R^2$；二者不是两个独立的机制证据。开发集预测关系在留出测试上保持，才支持对消关系的迁移。

### A.4 共享非线性修正的成立条件

为了将对消扩展到一般函数，要求分数中需要移除的贡献可由同一个痕迹函数确定：存在共享函数 $a$ 和仅依赖证据的 $f_*$，使
```math
\ell_0(e,\nu)=f_*(e)+a(t(e,\nu)).
```
取修正函数 $c_\phi(t)=-a(t)$，则修正后分数为 $\ell'=f_*(e)+\kappa$，其中 $\kappa$ 是输入无关常数，实现固定证据时的干扰不变性。痕迹只需确定这个标量贡献；神经网络增加对 $a$ 的拟合能力。

共享函数的存在是信息条件。例如 $\ell_0(e,\nu)=e+e\nu$、$t(e,\nu)=\nu$，且 $e$ 可取正负值时，同一干扰变化在不同证据条件下产生相反的分数变化，单个 $c_\phi(t)$ 无法同时抵消。此时需要检验痕迹是否还提供了区分这些贡献的信息。现有信息不足以保证共享修正函数可跨所有证据条件使用；附录 D.1 仅在受控合成条件下检验了可加情形。

为了同时保持原分数在固定干扰下的证据响应，还需 $a(t(e,\nu))$ 在固定 $\nu$ 时不随 $e$ 改变。局部模型中 $H=0$ 且分数可加时满足这一要求。若目标包含证据与干扰的交互，去掉交互贡献也可能改变原 C/F 响应，因此 ER 与逐对方向、正确率共同解释。

对一组 C/F 配对，记两端证据坐标为 $e_L,e_R$。A/P 配对监督干扰修正的变化，C/F 约束限制在固定干扰坐标时由证据变化引起的修正差，即 $c_\phi(t(e_R,\nu))-c_\phi(t(e_L,\nu))$。全局加法常数由任务级校准处理。有限配对所覆盖的变化具有直接约束，留出证据配对用于检验同一函数的迁移。线性实例的特征中心平移只产生常数，一般非线性实例保持训练时的标准化。

表示编辑也能实现与输出修正相同的分数函数。同监督对照比较达到相同不变性和证据保留时的数据、计算及实现代价。痕迹噪声与不同条件下的迁移通过留出预测评价。

### A.5 非线性分离、信息可辨识性与决策修复

线性子空间中证据与干扰的方向重叠，并不构成非线性不可分的证明。一个由坐标乘积或更复杂的交互决定的属性，可能不能由单个线性判别边界准确恢复，却能由神经网络恢复。因此，本文不把历史方向重叠诊断[R2]解释为“所有线性模型均无法分离”，更不推出“非线性分离不可能”。但能够识别干扰与能够在保留证据时安全删除干扰，是两个不同命题；是否有必要删除，则是第三个、由任务决定的问题。

**表达能力与可用信息必须区分。** 例如 ℓ(e,ν)=e+eν 且 t(e,ν)=ν，在不同 e 下，同一 ν 变化对分数产生不同幅度甚至不同方向的影响；任何只以 t 为输入的共享修正 c(t)，不论是线性函数还是神经网络，都不能同时准确抵消所有条件下的贡献。相反，如果增加的痕迹坐标提供了 e 或相关交互信息，一个非线性 cφ(t) 就可能实现线性函数无法实现的修正。这说明更高的函数容量有价值，但不能凭空补足缺失的可辨识信息。

更极端地，若可读表示只有 h=e+ν，且 e、ν∈{0,1}，那么 h=1 同时对应 (e,ν)=(1,0) 和 (0,1)。仅凭 h 无法唯一恢复期望的证据判断 e；换用任意复杂的非线性函数也不能解除这个输入歧义。这只是信息不足的反例，不意味着实际大模型的完整表示具有同样的歧义，亦不排除读取其他层或其他特征后恢复可辨识性。

**非线性表示编辑仍须明确保留约束。** 第 A.1 节的雅可比下界适用于在同一点严格保留原证据切向、同时消去干扰切向的可微编辑。它不是所有非线性编辑的无条件不可能定理：允许证据重编码、改变保护对象或只要求输出不变时，约束可能更弱，非线性方法也可能取得更好效果。因此，应在同一任务指标和可比较的证据保护预算下检验，而不是凭函数类别断言优劣。

**修正函数的非线性与干预位置彼此独立。** 即使神经网络能把两类信息分开，我们仍可以用该网络估计最终分数应减去多少，而无需写回隐藏状态。第5.2节用同一 16 维痕迹比较输出 MLP 与末层表示编辑，附录C.3进一步扩大特征和证据保护训练；两者共同检验“更多表达能力是否有益”与“为什么修正必须放在表示层”这两个独立问题。

## 附录 B 评价定义、实现与结果来源

### B.1 完整指标与计分约定

**分数与配对方向。** 二元任务固定“是/否”或“蕴含/矛盾”的正负映射，用两个答案词元的 logit 差作为分数。原模型分数记为 $\ell_0(x)$，方法 $m$ 的任务级校准前分数记为 $\ell_m(x)$。对第 $i$ 组配对，左、右输入为 $x_i^L,x_i^R$，原模型与方法分数差及修正差分别为
```math
\begin{aligned}
\Delta\ell_{0,i}&=\ell_0(x_i^R)-\ell_0(x_i^L),\\
\Delta\ell_{m,i}&=\ell_m(x_i^R)-\ell_m(x_i^L),\\
\Delta c_{m,i}&=\Delta\ell_{m,i}-\Delta\ell_{0,i}.
\end{aligned}
```
二元正类预测使用严格大于零的阈值；等于零归负类。若使用 M4，则最终预测按校准后概率阈值 $0.5$ 决定。共享截距在配对分数差中抵消。真实模型分数以 logit 计，合成分数以生成模型的分数单位计，二者不直接合并。MedPIC 的机制配对在对齐选项层面使用二元分数；其最终任务决策则在完整合法答案集合上进行。

**干扰指标。** 主文的干扰 RMS 对合格 A/P 配对求均方根：
```math
\mathrm{RMS}_{AP}(m)
=\sqrt{\frac{1}{|\mathcal I_{AP}|}
\sum_{i\in\mathcal I_{AP}}(\Delta\ell_{m,i})^2}.
```
辅助 VR 定义为
```math
\mathrm{VR}(m)
=100\left[
1-\frac{\mathrm{Var}_{i\in\mathcal I_{AP}}(\Delta\ell_{m,i})}
{\mathrm{Var}_{i\in\mathcal I_{AP}}(\Delta\ell_{0,i})}
\right].
```
两种方差均为同一配对集合的中心化总体方差。VR 可为负；原方差为零时未定义。干扰差分预测的决定系数定义为
```math
R^2(m)
=1-\frac{\sum_{i\in\mathcal I_{AP}}(\Delta\ell_{m,i})^2}
{\sum_{i\in\mathcal I_{AP}}
(\Delta\ell_{0,i}-\overline{\Delta\ell_0})^2}.
```
其中，预测要移除的干扰分数差为 $\Delta\ell_{0,i}-\Delta\ell_{m,i}=-\Delta c_{m,i}$，修正后的分数差 $\Delta\ell_{m,i}$ 是预测残差。该决定系数与 VR 相差残差均值项，不作为另一项独立机制发现。VR 不度量平均偏移，这也是正文优先报告 RMS 的原因。

**证据响应与改动。** 对标签翻转配对，定义 $d_i=y_i^R-y_i^L\in\{-1,+1\}$、$r_{0,i}=d_i\Delta\ell_{0,i}$、$r_{m,i}=d_i\Delta\ell_{m,i}$。有向响应保留率为
```math
\mathrm{ER}(m)
=100\frac{\sum_{i\in\mathcal I_{CF}}d_i\Delta\ell_{m,i}}
{\sum_{i\in\mathcal I_{CF}}d_i\Delta\ell_{0,i}}.
```
它是有向响应总和的比值，不是逐对比值均值。原分母为非正、零或接近零时，不能把保留比例解释为正确证据利用，应同时给出原量纲响应。100%仅表示保留平均响应。例如两对原响应均为1，修改后为3和−1，平均保留仍为100%，但一对已反向。

证据修正 RMS 为
```math
\mathrm{RMS}_{\mathrm{corr},CF}(m)
=\sqrt{\frac{1}{|\mathcal I_{CF}|}
\sum_{i\in\mathcal I_{CF}}(\Delta c_{m,i})^2}.
```
补充归一化量定义为
```math
\mathrm{DU}(m)
=\frac{\mathrm{RMS}_{\mathrm{corr},CF}(m)}
{\mathrm{RMS}_{AP}(0)},
```
其中 $\mathrm{RMS}_{AP}(0)$ 是原模型在 A/P 配对上的干扰 RMS；分母为零时 DU 未定义。DU 乘以 100 后以百分比展示，不与 ER 混淆。另报告 $|\Delta c_{m,i}|$ 的中位数、第 95 百分位和最大值，以及超预算比例
```math
\mathrm{Exceed}_{\delta}(m)
=\frac{1}{|\mathcal I_{CF}|}
\sum_{i\in\mathcal I_{CF}}
\mathbf1[|\Delta c_{m,i}|>\delta].
```
这些指标使用原训练预算，不根据测试响应重新定阈值。正文仅呈现已有记录支持的数字；尚未重算的尾部指标不填示例实测。

**方向退化。** 定义原模型和修正后方法的有向响应 $r_{0,i}$、$r_{m,i}$。方向退化数为
```math
N_{\mathrm{deg}}(m)
=\sum_{i\in\mathcal I_{CF}}
\mathbf1[r_{0,i}>0\ \land\ r_{m,i}\leq0],
```
方向退化比例以全部有标签 C/F 配对为分母：
```math
\mathrm{Deg}_{CF}(m)
=\frac{N_{\mathrm{deg}}(m)}{|\mathcal I_{CF}|}.
```
该分母不是原模型正向响应配对数；因此它表示全部 C/F 配对中发生方向退化的比例。方向正确与双端分类全对不是同一事件。没有观察到方向退化也不保证幅度保留、原错误修复或总体风险为零。

**任务与配对正确率。** 二元配对全对要求 $\hat y_m(x_i^L)=y_i^L$ 且 $\hat y_m(x_i^R)=y_i^R$。C/F 双端配对全对率按全部具有两端金标准标签的合格配对计算；MedPIC 选项对使用两个二元标签，病例对则要求两端预测答案集合分别与金标集合完全相同，后者称集合配对全对率，两种分母不得混用。整题集合全对率以 467 题为分母，不由选项准确率推算。F0 只保持二元真值、改变证据数值，判断一致率不等于正确率。

MEGA 控制器由每种方法从根节点按自身分数动态执行。路径正确要求执行无错误、带标签节点序列与参考路径一致、目标叶节点正确。叶节点正确率只看终点，自然节点准确率只看指定节点；三者不是同一指标。对固定共享节点序列的合成全对率不称作真实动态路径全对率。

**NLI 固定口径。** 设 $\mathcal O$ 为官方 500 条原声明，$\mathcal P$ 为 4,136 个标签保持关系，$\mathcal A$ 为 864 个标签翻转关系。宏平均 F1 为两个类别 F1 的等权平均：
```math
\mathrm{MacroF1}
=\frac12\sum_{k\in\{0,1\}}
\frac{2TP_k}{2TP_k+FP_k+FN_k}.
```
本次两个类别均有金标支持；计分脚本保存各类真阳性、假阳性、假阴性及支持数。一般情形的零分母必须明确实现规则，不静默删除类别。

保持一致率为
```math
C(m)=\frac{1}{|\mathcal P|}
\sum_{i\in\mathcal P}
\mathbf1[\hat y_m(x_i^L)=\hat y_m(x_i^R)].
```
令 $\mathcal A_m^+=\{i\in\mathcal A:\hat y_m(x_i^L)=y_i^L\}$，按任务论文定义的条件翻转忠实性为
```math
F_{\mathrm{cond}}(m)
=\frac{1}{|\mathcal A_m^+|}
\sum_{i\in\mathcal A_m^+}
\mathbf1[\hat y_m(x_i^R)\ne\hat y_m(x_i^L)].
```
二元翻转下，其分子等于全部翻转关系中的双端正确数，但分母不同。条件分母随方法变化，也可能因覆盖减少而提高比值；因此正文同时报告 $|\mathcal A_m^+|$、原题 F1 及无条件配对全对。它与连续分数方向退化的 $N_{\mathrm{deg}}$ 也不同。

任务论文规定宏平均 F1 与条件忠实性，后续勘误修正了预测一致性实现及相关成绩。公开脚本的固定版本仍存在 F1 平均方式、忠实性筛选及一致性调用差异。本文明确采用上述定义，独立复算，不声称执行了该脚本或复现官方榜单。参见[任务论文](https://aclanthology.org/2024.semeval-1.271.pdf)、[勘误](https://aclanthology.org/2024.semeval-1.271e1.pdf)、[公开脚本固定版本](https://github.com/ai-systems/Task-2-SemEval-2024/blob/7f32fa6c7db43e577f22e9fc6c28eef1cc6d223e/evaluate.py)。

**不确定性。** 受控 MEGA 按 40 个指南来源成组配对重采样 2,000 次，种子 11；同一次抽样用于全部方法，再计算差值区间。证据配对仅覆盖 12 个来源，稀疏分项会有未定义重采样，需报告有效次数。官方 MEGA 按归档同源簇计分。原 NLI 准确率区间按试验编号组合，配对响应区间按基础声明家族；二者分组不同，共享试验造成的更高层依赖尚未完全处理。新补算 NLI 指标没有置信区间，不从已有准确率区间推导。

不同随机训练配置分别计算指标与种子间差异，来源自助重采样区间则衡量另一种不确定性，两者不混用。公式中的上横线表示在相应评价配对集合上的算术平均，$\mathrm{Var}$ 均表示中心化总体方差。三个任务不合并总分；既有区间包含零只能说明未建立差异，不能解释为方法等效。

### B.2 数据、冻结配置与复算

| MEGA划分 | A：处置 | P：背景 | C：条件 | F：翻转事实 | F0：同侧事实 |
|---|---:|---:|---:|---:|---:|
| 拟合 | 4,869 | 18 | 66 | 69 | 82 |
| 开发 | 1,194 | 4 | 29 | 31 | 29 |
| 受控留出 | 2,283 | 10 | 33 | 7 | 6 |

**表B1：既有配对清单。** F0不进入主方法A/P目标。拟合、开发、受控留出分别有80、20、40个来源及1,712、420、806条路径；自然校准、开发、留出节点分别为761、199、372。A/P合计4,887个拟合对，C/F合计135个拟合对。划分记录见[修订前固定稿](https://github.com/chunxi-alpc/TRIAGE/blob/742e6b32c59fade4ddd2639314785209e85f1006/reports/aster_cancel_story_20261109/paper_story_latest.zh.md)附录B.3，受控结果见[当前归档](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/metrics.json)。

留出A/P中的A占2,283对、覆盖40来源，P仅10对、5来源；C为33对、11来源，F为7对、2来源，C/F合并12来源；F0为6对、2来源。没有因样本量小而补造配对或显著性。

线性方法读取51个选中头，标准化与子空间仅在拟合集估计。16个修正系数和一个截距之外，还需保存头索引、均值、标准差及子空间基。每次在线判断一次主干前向；离线定位、配对提取与子空间估计成本不能省略。已有缓存小模型训练计时不包含这些成本，不作为端到端加速证据。

**新增NLI计分的复现。** 四个预测分片固定为仓库提交 `742e6b32c59fade4ddd2639314785209e85f1006` 的归档。脚本验证其Git内容哈希、57,252条跨划分缓存记录、六方法各5,500条测试预测、逐条金标一致性及已有全测试正确数。官方关系清单提取自 `gold_test.json`，原文件内容哈希为 `38de30f1e7b83fc9668a086a18ea0e05a38959a8`，仅保留标识、标签及干预关系。

以下复算依赖研究分支提交 `08fc9dc6ef65236f697c10278648a024c89541a0` 中的预测缓存、脚本与关系清单。先检出该提交，再从仓库根目录执行：

```bash
python reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/nli_recomputed_metrics.py \
  --predictions-dir reports/aster_cancel_story_20261109/meeting/primary_full/nli4ct_all \
  --manifest reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/nli_official_test_manifest.json \
  --output /tmp/nli_recomputed_metrics.json
```

该操作只读已有预测并计算统计，不加载主干、不拟合、不选择方法。结果内保留完整定义、整数分子分母、类别统计与输入来源。

### B.3 零值为何出现、如何报告

| 情形 | 产生原因 | 本文处理 |
|---|---|---|
| 原模型自比较、共享截距的修正差 | 自比较恒等式，或常数在两端抵消 | 主表写“参照”“解析不变”，表注给出零值原因；保留原始归档 |
| 有限样本未出现方向退化 | 所评价配对中没有观察到该事件 | 删除全零结果列，在正文给出观察范围；不宣称总体风险为零 |
| 原方法在固定原错误池中的正确数 | 错误池按该方法答错定义 | 只作选集说明，不把该行当独立方法发现 |
| 尚未运行、无配对、分母为空 | 缺失、不适用或未定义 | 分别写明状态，不能写成数值零 |
| 标准差显示为零 | 种子结果相同，或舍入精度不足 | 相同计数直接说明；舍入情形给更高精度或仅列均值并链接原始记录 |
| 冻结LoReFT分数与原模型一致 | 可能涉及适配器、干预位置或训练效果 | 移出主比较，先核验实现，不按理论恒等式解释 |

**表B2：零值处理规则。** 不把真实零值改成正数，不通过删除失败样本美化结果。表内省略的是冗余数值展示，事实与原始记录仍可追溯。

零预算还有具体理论价值：当前拟合证据差分矩阵为135×16、秩为16，最小奇异值5.508；要求所有证据差分修正严格为零会迫使 $w=0$。这解释了为何需允许小幅预算，不必在预算表重复一个全零行。线性方法在有限留出中没有新增方向退化的观察，则是另一种零，不能混为同一个证明。

### B.4 结果类型与可追溯性

| 结果类型 | 本文内容 | 可复查材料 |
|---|---|---|
| 已归档真实运行的汇总 | MEGA受控比较、消融及官方路径；NLI连续分数与完整声明结果 | [受控结果](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/)、[官方任务结果](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/primary_full/) |
| 真实预测的新增计分 | NLI原题宏平均F1、预测一致率、条件翻转忠实性 | [计分脚本、关系清单与指标](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/experiment_review_20261109/) |
| 真实运行，逐例文件已补录 | MedPIC冻结迁移、历史解码及截距探索 | [迁移研究归档](../../results/aster_cr_transfer_20261110.zip)中的aster_medpic_20261109；不同子集沿用各自分母 |
| 实际合成拟合 | 已知数据生成机制下训练修正函数 | [非线性机制脚本](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/simulated_study/controlled_nonlinear.py)、[混合痕迹脚本](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/evidence_preserving/mixed_trace_control.py) |
| 文献报告 | 数据集定义、方法背景与原始协议 | 参考文献原始论文；不移植为本文同设置结果 |

**表B3：证据层级。** 已归档汇总的核对不等于重新运行主干；新增计分也不增加独立实验样本。历史模拟表和搜索记录保留于原始结果文件，本文只呈现与研究问题直接对应的实测。

### B.5 已评价方法的实现与可比性边界

正文受控实验的已评价方法包括仅截距、普通监督读出、LEACE、ITI、LoFiT、LoRA和ASTER-CR线性实例，所有方法均使用归档的Qwen3-8B评分及相应运行配置。MEGA受控评价为806条动态路径，官方完整评价为7,458条路径；各方法的实际覆盖范围不相同，不能将受控路径成绩当作官方成绩。[受控计分与逐方法指标](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/)保留实现与参数信息。

固定比例缩放是根据原模型分数解析计算的控制，不是额外训练的温度缩放模型。历史LoReFT缓存与原模型完全一致，尚不能确认干预是否有效加载，因此不将其列入已验证外部方法的性能排名。不同历史方法使用的监督信息和搜索预算未完全统一，本文相关比较属于观察性对照而非严格等预算排行榜。





## 附录 C 补充实测、探索性结果与实现核查

### C.1 同版痕迹诊断与历史动机

| 同一16维痕迹的测量 | A：1,194对、20来源 | A/P：1,198对、20来源 |
|---|---:|---:|
| 修补效应RMS与总效应RMS之比 | 13.91% | 13.90% |
| 总体绝对修补幅度与总效应之比 | 20.41% | 20.38% |
| 拟合后差分预测决定系数 | 0.4063 | 0.4069 |

**表C1：同一版本、同一批配对上的传导与预测。** 回归只在拟合集训练，开发集评价；修补为单向替换。它们分别度量干预传导与可预测信息，不合成单一优势分数。使用痕迹修补效应本身预测A总差分的决定系数为0.0786，坐标回归则为0.4063。[同版回归记录](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/paired_fit/metrics.json)直接支持坐标回归指标；传导比及0.0786沿用[修订前固定稿](https://github.com/chunxi-alpc/TRIAGE/blob/742e6b32c59fade4ddd2639314785209e85f1006/reports/aster_cancel_story_20261109/paper_story_latest.zh.md)附录C.1的诊断汇总，不声称本次由该JSON重新验证了全部修补统计。

原动机中的输入遮挡结果属于历史协议：275个可统计节点中71个需要条件句之外的定义，遮挡处置后完整路径正确数由404/432降至394/432 [R2]。它说明无条件删除有风险，但尚未在表4的全部同版输入上完成匹配比较，不能直接与806路径结果相减。历史全局均值表示编辑、人工同事实参考和多层传播结果同样用于提出问题，不能当作已完成同监督的擦除比较 [R2, R3]。

### C.2 预算、基线和同痕迹实现

| 特征 | 归一化拟合预算 | 留出VR（%）↑ | 留出ER（%），参考100 | 留出证据修正RMS↓ |
|---|---:|---:|---:|---:|
| 16维 | 0.05 | 3.07 | 99.82 | 0.1266 |
| 16维 | 0.10 | 4.10 | 99.60 | 0.2539 |
| 16维 | 0.25 | 5.64 | 99.14 | 0.5607 |
| 16维 | 0.50 | 9.38 | 98.22 | 0.9933 |
| 16维 | 1.00 | 17.36 | 96.35 | 1.6394 |
| 16维 | 2.00 | 26.71 | 93.63 | 2.6094 |
| 16维 | 无约束 | 30.61 | 91.67 | 3.3240 |
| 32维 | 0.10 | 4.03 | 99.56 | 0.2740 |
| 32维 | 0.25 | 7.38 | 98.73 | 0.6530 |
| 32维 | 无约束 | 84.74 | 9.33 | 33.9492 |

**表C2：既有预算扫描的受控留出结果。** 归一化预算为实际预算除以拟合A/P原始RMS，RMS单位为logit。预算为零时当前满秩约束只允许零修正，已在附录B.3解释，故不重复该行。没有将插值点混入本表。[完整指标](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/metrics.json)。

训练预算不是测试上界。开发60个C/F对中，线性主方法有11对超预算，最大改动0.5158；受控留出40对中则有11对超预算，最大0.5569。两组分母不同。开发集无约束修正RMS为3.0869，约束后为0.2029；对应VR为40.71%和4.58%。这些开发结果与表C2留出结果分别报告。

| 补充对照 | 干扰RMS↓ | 证据ER（%），参考100 | 证据修正RMS↓ | 路径全对↑ |
|---|---:|---:|---:|---:|
| 参数匹配的监督线性读出 | 1.2938 | 30.70 | 25.8109 | 734/806 |
| 闭式ASTER，历史内部方法 | 2.5499 | 99.88 | 0.5896 | 771/806 |
| 末层偏移 | 2.6266 | 99.96 | 0.0172 | 769/806 |

**表C3：补充冻结对照。** 与表4共用受控输入及计分器，RMS单位为logit。闭式ASTER不是额外外部算法家族。普通读出使用归档的8维配置，参数匹配配置单列。LoRA、LoFiT、LoReFT历史学习率分别为0.001、0.001、0.01，末层偏移为0.0001，本次没有重搜或重训。[评估实现](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/evaluate.py)。

**LoReFT实现核查。** 现有测试缓存的分数及汇总与原模型一致，证据改动为零。该结果无法区分适配器未生效与训练后确实无变化，故不放入主表当有效强基线。需核对适配器加载、干预层与位置、实际参数变化和训练日志，再报告新的独立评价；不能人为给它加微小扰动以消除零值。

**同痕迹训练细节。** 表9的三个训练式模型均使用种子11、22、33，各训练600步，学习率0.001。每步64个A/P对，全部135个C/F对参与保护，拟合后只按训练C/F检查修正幅度并缩放，随后校准截距。表示编辑形式为 $h'=h+Vg(t)$，其中 $h$ 为决策位置的原末层残差向量，$V\in\mathbb R^{d_h\times16}$ 的列为拟合 A/P 末层残差差分得到的 16 个正交主方向，$t$ 为输入的痕迹坐标，$g(t)\in\mathbb R^{16}$ 为学习到的方向系数，$h'$ 为编辑后的残差向量；冻结原最终归一化与输出行计算分数变化，并锚定原缓存。它只验证当前输出的末层编辑行为，不证明完整概念擦除。[训练代码](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/train.py)、[验证记录](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/verification.json)。

### C.3 扩展配置与官方MEGA

| 配置 | 受控干扰RMS↓ | 受控VR（%）↑ | 受控ER（%），参考100 | 受控证据修正RMS↓ | 受控路径全对↑ |
|---|---:|---:|---:|---:|---:|
| 16维线性 | 2.5730 | 4.10 | 99.60 | 0.2539 | 767/806 |
| 99维强保护集成 | 2.5196 | 8.02 | 99.88 | 0.1201 | 767/806 |
| 99维平衡集成 | 2.4274 | 14.59 | 99.77 | 0.2403 | 768/806 |

**表C4：探索性扩展的保护—干扰权衡。** RMS单位为logit，仍是已使用的受控集合。[扩展完整结果](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/optimization/metrics.json)。

扩展搜索包含540个线性拟合和334次小网络训练，共874个训练配置；后续还评价幅度、阈值与有限集成。99→128→1网络每个12,928个系数，三个网络等权集成后共38,784个系数及一个截距。平衡配置使用1,600步训练、A/P差分目标、C/F均方惩罚和自然节点分类损失，再按开发集保护档确定幅度。它没有采用线性方法的同一个逐对硬约束优化。

最初按开发路径选择的截距在受控留出上泛化较差，后续统一补报搜索时已经保存的拟合自然节点交叉熵截距。该补报发生在首次扩展测试之后，属于校准消融，不称新的独立测试。全部预存截距及配置见[校准消融记录](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/optimization/selected_calibration_ablation.json)。

| 方法 | 官方路径正确数/7,458 | 官方路径准确率（%）↑ | 准确率95%来源区间 | 纠正/新增错误 | 相对原模型差值95%区间，百分点 |
|---|---:|---:|---:|---:|---:|
| 原模型 | 7,037/7,458 | 94.36 | [93.44, 95.20] | 参照 | 参照 |
| 仅截距校准 | 7,038/7,458 | 94.37 | [93.45, 95.21] | 45/44 | [−0.31, 0.31] |
| ASTER-CR（线性） | 7,034/7,458 | 94.31 | [93.39, 95.18] | 43/46 | [−0.34, 0.24] |
| 99维强保护集成 | 7,037/7,458 | 94.36 | [93.46, 95.21] | 26/26 | [−0.21, 0.20] |
| 99维平衡集成 | 7,041/7,458 | 94.41 | [93.51, 95.24] | 20/16 | [−0.11, 0.21] |
| 仅痕迹MLP | 7,044/7,458 | 94.45 | [93.55, 95.28] | 23/16 | [−0.09, 0.27] |

**表C5：官方MEGA完整路径结果。** 区间沿用归档来源簇配对重采样；正确数相同不等于逐例一致。官方7,458条路径的C/F与F0配对评价尚未完成，不能挪用受控集合补齐。[官方完整指标](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/primary_full/mega_official_test/metrics.json)。

官方测试涉及440个发布指南单元与418个既有同源簇，二者分组不同。归档另标出未记录输入使用的7,286条路径，平衡配置与原模型分别为6,877和6,872条正确；该分层仍未确立收益，也不证明不存在预训练接触。开发集36条错误路径中34条的首次错误分数绝对值超过5，19条集中于同一PMA时间条件来源，提示小幅对消难以修复部分高置信语义错误，但该诊断不替代逐例原因核验。

### C.4 NLI连续响应与记录范围

| 方法 | 保持关系分数差RMS | 翻转ER（%），参考100 | 翻转证据修正RMS↓ | 全部翻转对正确方向 | 新增方向退化 |
|---|---:|---:|---:|---:|---:|
| 原模型 | 16.9688 | 参照 | 参照 | 749/864 | 参照 |
| 仅截距校准 | 16.9688 | 解析不变 | 解析不变 | 749/864 | 解析不变 |
| ASTER-CR（线性） | 16.8750 | 99.48 | 0.2926 | 748/864 | 2/749 |
| 99维平衡集成 | 16.5619 | 97.38 | 1.4982 | 750/864 | 1/749 |
| 99维强保护集成 | 16.7597 | 98.69 | 0.7491 | 750/864 | 未观察到 |

**表C6：冻结迁移的连续分数诊断。** RMS单位为logit。保持关系4,136对，翻转关系864对；原正向响应749对，因此新增退化的分母为749。“未观察到”表示这749个合格关系中事件计数为零，有限样本并不证明总体风险为零。该方向分母与表6原声明答对的条件忠实性分母不同。仅截距的浮点微小差异按解析恒等式解释，不写成经验微小改动。[归档原生配对指标](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/primary_full/nli4ct_all/pair_summary/metrics.json)。

NLI原始划分为训练1,700条、开发200条、练习测试2,142条、正式测试5,500条。跨划分记录合计9,542，但仅有9,342个不同原ID。本文不将开发与练习的重复计算为新增测试证据。新增脚本只在正式测试500原题及对应变体上计算表6指标；另外一种使用NLI训练标签拟合残差或截距的结果属于任务内适配，见C.6。

### C.5 MedPIC既有汇总与解码探索

| 方法 | 保持选项对的响应RMS | 方法引起的配对改动RMS↓ | 保持选项对全对↑ |
|---|---:|---:|---:|
| 原模型 | 4.5870 | 参照 | 163/216 |
| ASTER-CR（线性） | 4.6249 | 0.0996 | 145/216 |
| 99维平衡集成 | 4.4393 | 0.5752 | 156/216 |
| 99维强保护集成 | 4.5046 | 0.2876 | 152/216 |

**表C7：MedPIC既有标签保持选项对。** RMS单位为logit，配对仍改变患者信息，响应RMS不按纯干扰指标排名。与表7一样，本表保留既有运行数字，逐例文件已补录至[迁移研究归档](../../results/aster_cr_transfer_20261110.zip)，本次未重算指标。

原结果目录记录为 `/home/gcx/aster_medpic_20261109`，包括 `option_results.jsonl`、`question_results.jsonl`、`pair_results.jsonl` 和 `metrics.json`。当前仓库的该历史结果主要依赖已归档汇总与补录的逐例记录；原服务器文件及部分病例配对映射尚缺少独立复核证据，因此不将90组、92个病例对和460个选项对的映射视作已经逐项验证。

合法集合解码仅允许“以上皆非”单选，或不含该项的其他非空集合。原模型与修正方法使用相同规则。历史校准另按病例组划出93题开发集，各方法单独拟合截距；评价仅覆盖开发集外、按旧线性方法事先定义的324个错误题，不能当作剩余全部测试总体。

| 方法 | 合法解码后全对/324 | 再校准后全对/324 | 校准纠正/新增错误 | 翻转错误选项对：解码后全对 | 翻转错误选项对：校准后全对 |
|---|---:|---:|---:|---:|---:|
| 原模型 | 96/324 | 96/324 | 2/2 | 28/177 | 21/177 |
| ASTER-CR（线性） | 90/324 | 96/324 | 10/4 | 37/177 | 21/177 |
| 99维平衡集成 | 93/324 | 99/324 | 9/3 | 37/177 | 21/177 |
| 99维强保护集成 | 90/324 | 96/324 | 9/3 | 37/177 | 21/177 |

**表C8：MedPIC固定错误池的既有探索汇总。** 分母分别为324题与177个翻转错误选项对。校准少量改善整题却降低该翻转池的双端正确，不宜只挑整题数字。旧方法独立解码前在自身错误池上的正确数由选集定义为零，不重复该列。

### C.6 原错误池与目标任务适配

| 固定池 | 方法 | 修复数/复测数 | 剩余错误 | 额外监督或限制 |
|---|---|---:|---:|---|
| MEGA旧线性错误路径 | 原模型 | 46/424 | 378 | 共用已评价输入 |
| MEGA旧线性错误路径 | 99维平衡集成 | 39/424 | 385 | 冻结MEGA修正 |
| MEGA旧线性错误路径 | 难配对继续训练、开发缩放 | 7/424 | 417 | 仅原拟合来源扩充节点 |
| MEGA旧线性错误路径 | 稳健损失继续训练、开发缩放 | 29/424 | 395 | 仅原拟合来源扩充节点 |
| NLI旧线性错误声明 | 原模型 | 98/1,469 | 1,371 | 冻结迁移 |
| NLI旧线性错误声明 | 99维平衡集成 | 47/1,469 | 1,422 | 冻结迁移 |
| NLI旧线性错误声明 | NLI监督残差适配 | 207/1,469 | 1,262 | 使用NLI训练标签 |
| NLI旧线性错误声明 | 线性修正加NLI截距 | 213/1,469 | 1,256 | 使用NLI训练标签 |
| NLI旧线性错误声明 | 平衡修正加NLI截距 | 214/1,469 | 1,255 | 使用NLI训练标签 |

**表C9：事先按旧线性方法错误定义的条件修复。** 原方法在自身错误池中当然没有正确例，因此不把其定义性零列为新结果。MEGA的424来自旧线性方法，与原模型官方全量421个错误不是同一错误池。新适配未复测原正确全集，不能由这些数推断全量净收益。[冻结方法复测](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/primary_full/error_retest/)、[新训练与适配](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/primary_full/hard_optimization/)。

NLI监督残差在336个原翻转错误对中的ER为40.10%，证据修正RMS为9.1447，新增方向退化18对；它修复较多错例，也明显压缩证据响应。仅截距适配可修复213或214条声明而不改变连续配对响应，进一步说明阈值作用必须独立分析。上述适配用NLI训练集1,700条标签，监督残差由200条开发集选择配置；不能称冻结迁移。原正确全集未重新评价，纠正数不能代表净收益。

### C.7 成本与归档边界

已有小模型训练复用主干缓存，没有重新加载8B主干。历史记录的峰值小模型显存约0.43 GiB、600步训练约2秒，仅反映该缓存实现，不含主干前向、配对生成、头定位、子空间估计或外部基线训练。该数值不包含离线资源、保存状态、在线额外延迟及显存，因此不能把17个输出系数等同于整个方法的资源开销。

完整历史预算点和细分结果可通过[修订前固定稿](https://github.com/chunxi-alpc/TRIAGE/blob/742e6b32c59fade4ddd2639314785209e85f1006/reports/aster_cancel_story_20261109/paper_story_latest.zh.md)、[受控结果目录](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/)、[优化目录](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/optimization/)及[官方任务目录](https://github.com/chunxi-alpc/TRIAGE/tree/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/primary_full/)复核。

### C.8 痕迹有效性验证的已完成补充结果

历史痕迹验证在旧版MEGA拟合集上完成了50种子随机对照、4/8/16/32维容量实验、16维小型网络预测以及五折重新选头的来源泛化验证。具体输入、训练配置和逐来源预测见[完整验证报告](../../reports/aster_cancel_story_20261109/meeting/trace_validation/REPORT.zh.md)，其中随机和来源验证的参数、评价分母均与正文第5.4节一致。

| 机制验证 | 实测结果 | 主要解释 |
|---|---:|---|
| 16维选中痕迹的开发集A类预测 \(R^2\) | 0.4063 | 含有可预测干扰分数变化的信息 |
| 50种子随机16维子空间的平均 \(R^2\) | 0.3115 | 选中痕迹超过随机平均，但并非超过全部种子 |
| 50种子随机51头的平均 \(R^2\) | 0.0182 | 头定位具有额外预测价值 |
| 固定证据子空间的32维痕迹 \(R^2\) | 0.5197 | 预测信息受表示维数限制 |
| 16维小型网络三种子平均 \(R^2\) | 0.4486 | 非线性预测有一定增益 |
| 完整来源五折合并A/P \(R^2\) | 0.3454 | 62/80个来源的 \(R^2\) 为正 |

**表C10：** 真实预测与来源泛化结果。前五行均来自开发集A类配对，第六行来自80个拟合来源的留折A/P预测，因此评价总体不同。预测探针的结果不能代替约束后的任务效用。旧版Qwen修正和路径结果见正文第4—5节；新数据训练、20个来源验证折及收益拆分见第4.9节，并由[实验框架](experiments.zh.md)跟踪。

## 附录 D 受控合成机制实验

本附录只保留实际运行、具有固定生成机制和独立测试划分的合成拟合结果，用于检验函数容量与混合痕迹下的可行边界；它们不构成真实语言模型的性能证据。合成结果不用于推断真实医学任务中的改进幅度。



### D.1 实际执行的合成机制实验

本节实际在CPU上训练修正函数并评价独立生成的测试数据。数据、隐藏表示和固定读出均为合成；这些结果不属于真实语言模型或医学数据集评测，也不代表官方擦除算法。

**函数表达能力与修正位置。** 证据与干扰独立，16维痕迹的首坐标为带噪干扰。固定读出含一个痕迹二次项，分别设置近线性和非线性干扰。比较线性岭回归、16→32→1输出网络，以及相同规模的表示编辑网络。后者修改表示并清除痕迹坐标，再执行固定读出；清除这些坐标不等于证明全部干扰信息不可恢复。两个网络共享初始化、数据、批次和600步优化预算。每场景使用三个种子，拟合、开发、测试配对数分别为1,000、300、1,000，开发与测试不参与选型。

| 方法 | 修正系数数目，不含截距 | 近线性场景干扰方差降低率（%）↑ | 非线性场景干扰方差降低率（%）↑ |
|---|---:|---:|---:|
| ASTER-CR（线性） | 16 | 98.53±0.05 | 7.52±1.01 |
| ASTER-CR（16维MLP） | 576 | 98.59±0.09 | 96.48±0.35 |
| MLP表示编辑与痕迹移除 | 576 | 98.60±0.07 | 96.55±0.32 |

**表D1：** 独立合成测试集的均值±样本标准差。基准的相对降低率由定义确定，不重复列行。非线性场景中线性估计不足，两个网络均能有效降低干扰；结果没有建立输出修正优于表示编辑的结论。

所有方法的证据响应保留率约为100%，来自痕迹不含证据、证据配对只改变证据的构造，不能作为真实任务证据保护的验证。复现材料：[训练脚本](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/simulated_study/controlled_nonlinear.py)、[逐种子结果](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/simulated_study/controlled_nonlinear_results.csv)。

**混合痕迹与证据约束。** 令原分数为两倍证据加干扰，16维痕迹同时含二者并加入标准差0.03的噪声。可解场景的干扰方向为第一坐标，证据方向同时覆盖第一、第二坐标。每个种子拟合使用5,000个干扰对和200个证据对，独立测试使用2,000和1,000对。正则系数、归一化预算分别固定为0.01、0.1，种子为11、22、33。

| 方法 | 干扰方差降低率（%）↑ | 证据响应保留率（%），参考100 | 证据修正与原干扰RMS之比（%）↓ |
|---|---:|---:|---:|
| 无证据约束的ASTER-CR | 99.91 | 60.03 | 114.91 |
| 证据约束修正 | 99.85 | 99.65 | 4.11 |
| 已知证据方向的参考方法 | 99.85 | 100.03 | 3.89 |

**表D2：** 三个种子的均值。原汇总中部分标准差四舍五入为零，本表统一仅列均值，不把显示精度内的微小波动解释为完全稳定；逐种子精度保留于结果文件。

无约束回归几乎消除全部干扰，却削弱证据响应；可解场景中加入约束可保留大部分干扰收益。当干扰方向与证据方向完全重合时，受约束方法的干扰方差降低率为13.15%，证据响应保留率为96.60%，已知证据方向的参考方法也没有正的干扰降低收益。该负结果说明优化不能补足不可辨识的信息。训练配对全部满足预算，而可解场景测试配对满足相同预算的比例为97.8%–99.2%，训练保证仍不等于测试保证。复现材料：[脚本](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/evidence_preserving/mixed_trace_control.py)、[逐种子结果](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/evidence_preserving/mixed_trace_results.csv)、[完整指标](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/evidence_preserving/mixed_trace_metrics.json)。
