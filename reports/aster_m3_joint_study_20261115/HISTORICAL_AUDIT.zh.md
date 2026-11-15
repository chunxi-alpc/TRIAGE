# ASTER-CR 第三模块联合监督实验：历史等价性审计

**结论：已归档实验中没有与本轮 S/J 完全等价的监督对照。正式 E1 五折的 R0、R1、R2 参数及预测可以复用；S/J 需要按照当前数据、表示、任务标签和校准条件补齐。**

本文件记录代码与归档产物的只读审查，下文历史数字均为已归档结果的引用。本轮 S/J 的执行状态和新增实测结果以同目录 `status.json` 与 `REPORT.zh.md` 为准。经用户提供的堡垒机和119服务器连接路径，已进入指定115服务器并激活 `gcx261014`，服务器预检查已找到正式缓存、五折参数、761个自然节点、2,540对训练数据及1,712个路径病例。

| 项目 | 记录 |
|---|---|
| 仓库 | `chunxi-alpc/TRIAGE` |
| 本次读取的 main 提交 | `95859d5c2bb610bf9f151a64ba405751431b833c` |
| 实施分支 | `experiment/aster-m3-joint-20261115` |
| 审计日期 | 2026-11-15 |
| 旧扩展代码归档提交 | `08fc9dc6ef65236f697c10278648a024c89541a0`，来自当前论文中的历史代码链接 |
| 操作范围 | 读取历史代码、配置和报告；本文件为新增产物 |
| 真实训练与评价 | 本次审计未运行 |
| 服务器数据状态 | 2026-11-15服务器预检查通过；332,860条缓存分数记录，五折具体头特征在运行投影时核对 |

## 1. 本轮比较需要满足的条件

本轮研究固定 E1 / Dataset A 的 2,540 对配对、冻结 Qwen3-8B、80 来源的原五折划分。每折使用相应 64 个训练来源拟合得到的 51 个头与 16 维痕迹，对余下 16 个来源评价。S/J 使用相同自然训练标签、相同 16 维线性容量、相同正则项和逐对 C/F 预算 0.2923349 logit；J 新增 A/P 配对目标。

修正采用

\[
c_w(t)=w^\top(t-\mu_{\mathrm{train}}),\qquad
q_w(x)=\ell_0(x)+c_w(t(x)).
\]

训练期间以同一训练自然节点拟合的原模型截距 \(b_0\) 为固定分类参考。评价分别使用共享 \(b_0\) 和各方法自然训练节点独立拟合的 \(b_w\)。所有中心化、校准和选择统计均来自当前外折的训练来源。

## 2. 历史实验与本轮对照的对应关系

| 历史实验 | 数据与监督 | 特征与容量 | 约束、正则和校准 | 是否可直接作为 S/J |
|---|---|---|---|---|
| 普通监督残差读出 | 761 个原始自然拟合节点；80/20/40 来源训练、开发和历史测试 | 末层判定位置、最终 RMSNorm 前的残差；训练集标准化和 64 维白化 PCA | 无 C/F 约束；平均 BCE 加 \(\lambda\|w\|^2/2\)；b 与 w 联合训练；开发路径选正则 | 否，维度、表示、约束、截距与选择协议不同 |
| 参数匹配普通读出 `ordinary_matched` | 旧协议 6,727 个拟合视图；包含配对侧任务标签 | 16 维普通子空间 O，17 个训练参数；O 与干扰痕迹 N 不同 | 无 C/F 硬约束；\(0.01\operatorname{mean}(w^2)\)；b 与 w 联合训练 | 否，参数个数相同不足以构成相同表示与监督的对照 |
| 监督痕迹修正 `label_linear_b0.05/0.1/0.25` | 扩展拟合集 11,722 自然节点、35,166 干扰节点及 A/P 对；135 个旧 C/F 对 | 旧 51 头和 16 维 N；在痕迹上再次按训练自然节点标准化 | 自然与干扰 BCE 各占 0.5；可加 AP 和原正确节点保护；预算等于档位乘训练 AP RMS；训练不加 b0；训练后独立校准 | 否，数据、M1/M2、监督、预算、正则尺度和校准均有差异 |
| 99 维非线性强保护/平衡集成 | 旧配对池与自然节点标签 | 32 维 N/S、64 维残差和 3 个分数特征；三个 99→128→1 网络集成 | CF 软惩罚；开发 CF RMS 档位决定修正幅度；开发路径选截距，后补训练自然节点截距消融 | 否，表示、容量、约束和搜索协议不同 |
| 正式 E1 五折完整方法及独立截距 | 当前 E1 配对、原折内自然校准节点 | 每折对应 51 头、16 维 N | 原 E1 AP 目标和硬预算；原自然节点截距或独立 b0 | 可复用 R0/R1/R2；新增评分入口需验证已保存分数和动态路径 |

### 2.1 普通监督残差读出

[ordinary_readout.py](../../experiments/ordinary_readout.py) 第 20—31 行实现平均 BCE、权重平方惩罚和未惩罚截距；第 43—46 行构造 64 维白化 PCA；第 68 行记录 761 个原始拟合节点、正则候选 0.01/0.1/1、开发路径选型和无配对数据的协议。第 99、179 行调用 `experiments.aster_cr.paths.execute` 进行动态路径评价。

[历史报告](../../results/ordinary_readout_20261108/REPORT.zh.md) 给出训练/开发/测试节点 761/199/372，开发/测试路径 420/806。它回答残差监督读出能否超过标量校准的问题，不能替代当前外层五折中的 S。

### 2.2 参数匹配的监督线性读出

[aster_cr/readout.py](../../experiments/aster_cr/readout.py) 第 12—18 行分别定义普通子空间 O 与 N/S 投影，第 99—103 行从训练头激活拟合 O。第 19—47 行联合优化 b 与 w，使用 BCE、`reg * mean(w²)` 及可选软损失，没有当前逐对预算的硬约束。

`results/aster_cr_repair_v2_20261108/qwen/readout_training.json` 记录 6,727 个训练视图和 1,685 个开发视图。相同目录的 `readout_candidates.json` 中，`ordinary_matched` 为 rank=16、17 参数、regularization=0.01，invariance/sensitivity/preservation 权重均为 0。它与本轮 S 的痕迹表示和自然节点监督集合不同。

### 2.3 监督痕迹修正

[trace_repair_followup/train.py](../aster_cancel_story_20261109/meeting/trace_repair_followup/train.py) 的具体实现如下。

| 位置 | 已核对的实现 |
|---|---|
| 第 19—22、48—49 行 | 全量使用旧 `qwen/supports_r16.npz`；折内使用旧 `trace_validation/exp3/full/fold0…4/support_and_ridge.npz` |
| 第 114—129 行 | 加入旧拟合集的 135 个 C/F 配对 |
| 第 164—168 行 | label-linear 使用 16 维 N；小网络另外加入原始分数 |
| 第 169—178 行 | 分别构造自然、干扰和配对样本；损失按来源、病例、变体、节点层级均衡 |
| 第 185—186 行 | 自然训练节点的加权均值和标准差用于再次标准化痕迹 |
| 第 203—227 行 | 自然与干扰分类损失、AP 平方损失、原正确节点保护及正则；拟合后独立自然截距 |
| 第 289—335 行 | 五折验证用于选择 AP/clean 权重及网络训练步数 |

其线性目标可写为

\[
\begin{aligned}
L_{\mathrm{old}}={}&\tfrac12\sum_{i\in\mathrm{natural}}a_i\,\mathrm{BCE}(\ell_i+z_i^\top w,y_i)
+\tfrac12\sum_{i\in\mathrm{interference}}d_i\,\mathrm{BCE}(\ell_i+z_i^\top w,y_i)\\
&+\lambda_{AP}\frac{\sum_j u_j(\Delta\ell_j+\Delta z_j^\top w)^2}{r_{AP}^2}
+\lambda_{\mathrm{clean}}\frac{\sum_{i\in\mathrm{basecorrect}}v_i(z_i^\top w)^2}{r_{AP}^2}
+0.01\,\frac{\|w\|^2}{16},
\end{aligned}
\]

其中各组样本权重分别归一化，\(z=(t-\mu)/\mathrm{sd}\)，\(r_{AP}\) 是当前训练来源 AP 原始分数差的无权 RMS。线性硬约束为

\[
|\Delta z_j^\top w|\leq \mathrm{budget}\cdot r_{AP}.
\]

训练期间没有固定 \(b_0\)。训练后才对自然节点拟合 b。`training/full_fit_input_validation.json` 保存 11,722 个自然节点、35,166 个干扰节点、35,166 个 AP 对和 135 个 CF 对。

`training/models.json` 记录的全量预算为：

| 旧预算档位 | 实际 logit 预算 | 最终 label-linear 选择 |
|---:|---:|---|
| 0.05 | 0.18000212608892718 | lambda_AP=0，lambda_clean=0 |
| 0.10 | 0.36000425217785437 | lambda_AP=0，lambda_clean=0 |
| 0.25 | 0.9000106304446359 | lambda_AP=0，lambda_clean=0 |

旧报告使用来源等权的完整路径统计；当前正式评价器采用来源簇重采样后重新合并路径计数。旧五折用于选择超参数，其选后结果不能作为本轮新的外层验证结果。

### 2.4 99 维非线性修正

当前论文第 995—997 行链接到归档提交 `08fc9dc6ef65236f697c10278648a024c89541a0`。本次读取了该提交的 [score_conditioned/search.py](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/optimization/code/score_conditioned/search.py)、[selection/select.py](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/optimization/code/selection/select.py) 及 [selected_portable.json](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/optimization/selected_portable.json)。

`search.py` 第 17—32 行构造 GELU 网络和 99 维特征：16 维 N＋16 维 S＋64 维末层残差投影＋\(\ell/10,|\ell|/10,\tanh(\ell/4)\)。第 45—53 行在旧拟合自然节点上估计 b0；第 66—82 行以 AP 均方差、CF 均方或 hinge 惩罚、自然 BCE 和自然修正幅度项训练，采用 AdamW、weight_decay=0.001。所选三个网络使用 seed=11/22/33、width=128、steps=1600、beta=10000、ce=1、jitter=0.1。

`selection/select.py` 第 91—100 行使用开发 CF RMS 档位与 ER 确定幅度，再按开发路径选择截距。强保护与平衡幅度分别为 0.21822606185792146 和 0.4364521237158429。三个网络均匀集成后包含 38,784 个网络系数及一个截距。

论文记录的搜索包含 540 个线性拟合与 334 次网络训练。后补训练自然节点交叉熵截距发生在首次扩展测试之后，归类为历史校准消融。这些结果提供容量和校准背景，不承担本轮同容量 S/J 比较的论证职责。

## 3. R0/R1/R2 可复用的历史结果

以下数字直接引用 [ASTER-CR 收益拆分与 E3 五折验证报告](../aster_cr_diagnostics_e3cv_20261111/REPORT.zh.md)，**属于历史五折结果复评记录，不是本轮新增实测结果**。合并分母为 80 个来源、1,712 条自然路径和 5,136 条干扰路径。

| 本轮对应方法 | 历史名称 | 自然正确路径 | 干扰正确路径 |
|---|---|---:|---:|
| R0 | raw | 1549 / 1712 | 4722 / 5136 |
| R1 | independent_intercept | 1546 / 1712 | 4685 / 5136 |
| R2 | E1_full | 1546 / 1712 | 4690 / 5136 |

归档的 R2−R1 自然路径差值为 0.000 个百分点，95% 来源配对区间为 [−0.170,+0.172]；干扰路径差值为 +0.097 个百分点，区间为 [+0.000,+0.214]。干扰区间下限没有严格大于 0。该比较使用 2,000 次来源簇重采样，每次重新计算合并指标。

R2 相对 R1 的错误转移完整保存如下：

| 层级 | 错误改对 | 正确改错 | 保持正确 | 保持错误 |
|---|---:|---:|---:|---:|
| 自然路径 | 1 | 1 | 1545 | 165 |
| 干扰路径 | 7 | 2 | 4683 | 444 |

归档的 E1 五折 AP 方差降低率为 3.498%，CF 修正 RMS 约 0.1225 logit，有向响应保留率约 99.984%；1,342 个验证 CF 对中有 25 对超预算，方向退化计数为 0。训练逐对预算和验证超预算率需要分别报告。

独立 b0 的来源与拟合样本由 `reports/aster_cr_diagnostics_e3cv_20261111/independent_intercepts.json` 保存；[score_variant.py](../../experiments/aster_cr_diagnostics_e3cv_20261111/score_variant.py) 第 8—27 行使用原模型分数为固定偏移，最小化无权平均自然节点 BCE。

| 外折 | 自然拟合节点数 | b0，按历史报告显示精度 | 训练来源数 |
|---|---:|---:|---:|
| fold1 | 607 | −0.080842 | 64 |
| fold2 | 615 | 5.848431 | 64 |
| fold3 | 615 | −0.044125 | 64 |
| fold4 | 604 | 1.259180 | 64 |
| fold5 | 603 | 5.847193 | 64 |

实现应读取归档文件中的完整精度参数及 `calibration_ids`，不能用本表的四舍五入值重新生成预测。

## 4. 新入口需要保留的评分和目标定义

### 4.1 中心化与截距重建

原 E1 在 M2 的原始头坐标标准化后直接计算 \(t^\top w\)，没有再次减去自然训练痕迹均值。原完整评分为

\[
s_{E1}=\ell_0+t^\top w+b_{\mathrm{old}}.
\]

统一采用 \(c_w(t)=w^\top(t-\mu_{\mathrm{train}})\) 后，保持原完整分数的截距为

\[
b_{\mathrm{centered}}=b_{\mathrm{old}}+w^\top\mu_{\mathrm{train}}.
\]

此变换不重训 E1，也不改变配对分数差。共享 b0 条件下，中心化 E1 的新评分应作为单独消融保存；原 E1 完整分数的复现对应上式的截距重建。

### 4.2 浮点评分入口

[training.py](../../experiments/aster_cr_retrain_paired_v2_20261111/training.py) 第 258—260 行按 float32 头坐标计算 M2 投影，再转 float64。[evaluation.py](../../experiments/aster_cr_retrain_paired_v2_20261111/evaluation.py) 第 52—96 行以 float64 折叠

\[
\alpha=(Nw)/\mathrm{scale},\qquad
\mathrm{offset}=b-\mu_{\mathrm{head}}^\top\alpha,
\]

随后计算 `ell + raw @ alpha + offset`。原 E1 已保存分数的复现以正式评价入口为依据，避免把训练投影的浮点差异解释为模型变化。

### 4.3 正则与损失归一化

正式 E1 的 `training.py` 第 278—289 行使用 AP 权重总和 4,887，实际目标等价于

\[
\sum_j \omega_j(\Delta\ell_j+\Delta t_j^\top w)^2+0.01\|w\|_2^2,
\qquad \sum_j\omega_j=4887.
\]

求解时将二次项和线性项共同除以

\[
4887\max\!\left(\frac{\sum_j\omega_j\Delta\ell_j^2}{4887},1\right).
\]

这是整个求解目标的共同缩放，保留原回归与正则的相对权重。归档目标数值由第 323 行写入 `weighted_regression_objective`。

本轮 S/J 的任务和 AP 损失分别归一化，并采用明确的 \(\lambda_w\|w\|^2\)。应保存完整公式、权重、归一化常数、各项优化值及实际正则系数。数值同为 0.01 时，需说明它相对于归一化损失的尺度；R2 继续保留原协议参数。S/J 之间必须保持这一尺度完全一致。

### 4.4 动态路径与来源统计

正式 `evaluation.py` 第 324—375 行包含动态分支执行及路径预测，第 101—145 行实现共享来源重采样，第 378 行起组合自然节点、配对和路径评价。新入口应复用这些定义。历史监督扩展报告中的来源等权路径百分比不能直接填入正式合并路径表。

## 5. 数据路径和访问状态

以下为从代码和归档读取到的路径线索，均未确认服务器现状。路径映射应在恢复 SSH 后读取并验证，先复用现有数据和模型。

| 资产 | 已知入口 | 当前状态 |
|---|---|---|
| 正式 E1 五折模型 | 正式重训产物的 `models/E1/fold1…5/m2.npz`、`m3.npz`、`model.json` | 仓库代码和报告可读取；服务器文件未核实 |
| 正式来源划分 | 正式重训产物的 `folds.json` | 需要按文件保持原划分 |
| E1 配对与自然校准节点 | 正式重训产物的 `data/E1_pairs.jsonl`、`data/natural_views.jsonl` | 缓存位置和访问状态未核实 |
| 独立截距及样本 ID | `reports/aster_cr_diagnostics_e3cv_20261111/independent_intercepts.json` | 归档入口已确认；新运行中仍需对照使用相同样本 |
| 分数与路径归档 | 诊断报告 `evaluation/` 下的预测、`scores_by_variant.npz` 和 `score_manifest.json` | 历史报告确认保存；服务器当前位置未核实 |
| 旧监督扩展缓存 | `/home/gcx/aster_trace_repair_20261110` | 历史代码默认路径，未核实 |
| 旧痕迹和模型缓存 | `/home/gcx/aster_cr_repair_v2_run_20261108` | 历史代码默认路径，未核实 |
| 旧扩展冻结配置 | `/home/gcx/aster_next_20261109/frozen_methods.json` | 历史代码引用路径，未核实 |

## 6. 本轮交付中的结果资格

R0/R1/R2 上述数字、旧监督痕迹结果、99 维扩展、历史 40/60 来源、官方 MEGA 和跨任务结果均已有使用记录。本轮可以重放和核对它们，并明确标为历史复评。

S/J 使用原 80 来源外层五折能够评价当前修改在对应留出来源上的表现；它们属于已使用研究来源上的新方法实验。参数选择必须限制在每个外折的训练来源内，并说明内部选择条件于冻结的 M1/M2 表示。冻结方案后的新来源才承担后续独立确认。

当前能够确定的是需要补齐同表示、同容量、同监督与同校准的 S/J 对照。任务收益、AP 独立贡献、自然非劣和是否进入 J2 均应由本轮实际运行结果判断。本次审计没有这些新增结果。
