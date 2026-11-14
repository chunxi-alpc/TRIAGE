# ASTER-CR 近期外部方法补充实验结果

2026-11-14。Qwen3-8B 上完成 RePS、PSR-MSE、CAA、INLP 和 DoRA 五种新增方法，共 11 次冻结测试评价。DoRA、RePS、PSR-MSE 各报告种子 11、22、33；CAA 和 INLP 为确定性拟合，各报告一次。

原始数据：[11次逐运行主表](section4_5_new_methods.csv) · [5种方法均值与种子标准差](section4_5_method_summary.csv) · [逐方法测试产物](evaluation/test/)。

## 核心发现

- **DoRA 提高完整路径正确率并减少 A/P 分数波动。** 三个种子的自然路径正确率为 97.188 ± 0.312%，相对原模型缓存参照提高 1.902 个百分点；干扰路径正确率为 96.912 ± 0.357%。A/P RMS 从 2.628 降至 1.361，VR 为 70.652%。同时，C/F ER 为 0.5212，C/F 修正 RMS 为 21.170，双端全对率为 59.167%。
- **PSR-MSE 保持接近原模型的 C/F 差分，并取得路径正确率增益。** 三种子的 C/F ER 为 0.9913，修正 RMS 为 0.574；自然与干扰路径正确率分别为 95.988 ± 0.612% 和 96.843 ± 0.096%。新增参数为 8,193 个。种子 22 的七项任务评价指标与原模型缓存参照相同；种子 11、33 的自然路径分别纠正 8、9 例，新增错误均为 0。
- **INLP 的最终预测接近原模型。** 自然路径纠正 1 例、新增错误 0 例，干扰路径正确性保持原值；C/F 修正 RMS 为 0.0244。CAA 的自然和干扰路径正确率分别为 95.037% 和 96.443%，对应纠正/新增错误为 5/7 和 5/12。
- **RePS 在当前 E1 任务适配与固定训练配置下呈现较大的种子差异。** 自然路径正确率为 47.270 ± 35.493%，三个种子的范围为 7.444%—75.558%。种子 33 的 A/P VR 为 97.557%，同时 C/F ER 为 0.0092、C/F 双端全对率为 5.000%、自然路径正确率为 7.444%。该组实测说明，应联合读取稳定性、证据差分和路径正确率。全部种子保留在正式表中。

以上结论采用统一测试分母。表中的“±”为三个种子的样本标准差；来源级 95% 区间保存在各方法 statistics.json 和 statistics.csv 中。

## 数据、评分与评价协议

主训练来源为 E1 / Dataset A，版本 20261111-sol-agents-v2：2,540 对，A/P/C/F 各 635 对，对应 4,637 个唯一 prompt。拟合、开发和测试按指南来源划分。基础模型参数保持冻结，DoRA 训练新增适配参数。

| 范围 | 来源数 | A/P 对 | C/F 对 | 自然路径 | 干扰路径 |
|---|---:|---:|---:|---:|---:|
| 拟合集 E1 | 80 | 1,270 | 1,270 | — | — |
| 开发集 | 20 | 1,198 | 60 | 420 | 1,260 |
| 冻结受控测试 | 40 | 2,293 | 40 | 806 | 2,418 |

测试 C/F 40 对来自 12 个来源，均具有两端标签。开发集另有 29 对 F0，测试另有 6 对 F0，作为独立诊断保留。每个病例包含一条自然路径和三种 A 类干扰路径。

本报告使用既有 40 来源 MEGA-CDP 受控留出，属于有历史使用记录的评价集合。MEGA-CDP 官方 7,458 条路径是另一个评价范围。数据清单和来源划分见 [准备统计](prepared/summary.json)及[输入引用](prepared/input_reference.json)。

模型采用服务器既有 Qwen3-8B 快照，路径为 /home/gcx/TRIAGE/models/modelscope/Qwen3-8B。该快照未声明不可变上游 revision，配置中保留 model_revision=null。运行环境为 Python 3.12.3、PyTorch 2.8.0、Transformers 4.57.6、PEFT 0.21.2；主干使用 BF16、SDPA，关闭 thinking，标签输出投影采用 FP32。模型分数 s 是“是”与“否”的 logit 差，s>0 判为“是”。

所有层、秩、方向、训练预算和选择规则均在测试前固定；强度和 INLP 迭代数按各方法已记录的开发规则确定。完整路径由各方法自己的预测逐节点执行，正确条件为所有访问决策及最终叶节点与标准路径一致。原模型参照复用相同输入和评分协议的历史缓存。

### 指标定义

对每一配对，d=s右−s左 表示原模型差分，d′=s′右−s′左 表示适配后的差分，y左/y右 为二元真值。

| 指标 | 计算与读法 |
|---|---|
| A/P RMS | sqrt(mean(d′²))，单位为 logit；反映标签保持配对的分数差幅度。 |
| VR | 1−Var(d′)/Var(d)，使用中心化方差；正值表示方差减少，负值表示方差增加。表中乘100显示为百分比。 |
| C/F ER | mean((y右−y左)d′) / mean((y右−y左)d)，表中保留比例值；1表示平均有向证据差分保持原值。 |
| C/F 修正 RMS | sqrt(mean((d′−d)²))，单位为 logit；记录方法对证据改变配对差分施加的改变量。 |
| C/F 双端全对率 | 一对中左右两端均判断正确的比例。 |
| 自然/干扰路径正确率 | 对应范围内完整路径正确的比例，分母分别为806和2,418。 |
| 纠正/新增错误 | 同一病例、同一变体下，相对原模型的“错误→正确”及“正确→错误”计数。 |

原模型缓存参照：A/P RMS=2.628152，VR=0%，C/F ER=1，C/F 修正 RMS=0，C/F 双端全对为19/40（47.500%）；自然路径768/806（95.285%），干扰路径2339/2418（96.733%）。

## 第4.5节结果

### 方法均值与种子差异

CAA、INLP 为一次确定性运行；其余方法为三个种子的均值±样本标准差。ER 保留比例值，准确率和 VR 使用百分比。

| 方法 | 运行数 | A/P RMS | VR (%) | C/F ER | C/F 修正 RMS | C/F 双端全对 (%) | 自然路径 (%) | 干扰路径 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoRA | 3 | 1.361 ± 0.539 | 70.652 ± 18.811 | 0.5212 ± 0.1058 | 21.170 ± 2.504 | 59.167 ± 1.443 | 97.188 ± 0.312 | 96.912 ± 0.357 |
| PSR-MSE | 3 | 2.584 ± 0.039 | 3.283 ± 2.843 | 0.9913 ± 0.0076 | 0.574 ± 0.501 | 47.500 ± 0.000 | 95.988 ± 0.612 | 96.843 ± 0.096 |
| INLP | 1 | 2.628 | -0.043 | 1.0002 | 0.024 | 47.500 | 95.409 | 96.733 |
| CAA | 1 | 2.752 | -9.657 | 0.9955 | 1.076 | 47.500 | 95.037 | 96.443 |
| RePS | 3 | 3.909 ± 2.952 | -201.697 ± 260.806 | 0.8417 ± 0.9410 | 28.233 ± 8.969 | 31.667 ± 23.629 | 47.270 ± 35.493 | 46.264 ± 34.347 |

### 全部11次运行

| 方法 / 种子 | A/P RMS | VR (%) | C/F ER | C/F 修正 RMS | C/F 双端全对 (%) | 自然路径 (%) | 干扰路径 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAA | 2.752 | -9.657 | 0.9955 | 1.076 | 47.500 | 95.037 | 96.443 |
| DoRA / 11 | 1.595 | 63.546 | 0.6061 | 19.156 | 60.000 | 97.519 | 97.229 |
| DoRA / 22 | 1.743 | 56.429 | 0.5547 | 20.380 | 60.000 | 96.898 | 96.526 |
| DoRA / 33 | 0.744 | 91.980 | 0.4027 | 23.973 | 57.500 | 97.146 | 96.981 |
| INLP | 2.628 | -0.043 | 1.0002 | 0.024 | 47.500 | 95.409 | 96.733 |
| PSR-MSE / 11 | 2.561 | 4.953 | 0.9881 | 0.797 | 47.500 | 96.278 | 96.898 |
| PSR-MSE / 22 | 2.628 | 0.000 | 1.0000 | 0.000 | 47.500 | 95.285 | 96.733 |
| PSR-MSE / 33 | 2.562 | 4.896 | 0.9859 | 0.924 | 47.500 | 96.402 | 96.898 |
| RePS / 11 | 5.449 | -322.084 | 0.6532 | 18.432 | 40.000 | 58.809 | 56.741 |
| RePS / 22 | 5.773 | -380.565 | 1.8627 | 30.233 | 50.000 | 75.558 | 74.152 |
| RePS / 33 | 0.506 | 97.557 | 0.0092 | 36.033 | 5.000 | 7.444 | 7.899 |

### 纠正与新增错误

每一行对应独立运行；跨种子计数保留各自的806条自然路径和2,418条干扰路径分母。

| 方法 / 种子 | 自然路径全对数 | 自然纠正 | 自然新增错误 | 干扰路径全对数 | 干扰纠正 | 干扰新增错误 | C/F 双端全对数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAA | 766 | 5 | 7 | 2332 | 5 | 12 | 19 |
| DoRA / 11 | 786 | 27 | 9 | 2351 | 56 | 44 | 24 |
| DoRA / 22 | 781 | 25 | 12 | 2334 | 52 | 57 | 24 |
| DoRA / 33 | 783 | 25 | 10 | 2345 | 54 | 48 | 23 |
| INLP | 769 | 1 | 0 | 2339 | 0 | 0 | 19 |
| PSR-MSE / 11 | 776 | 8 | 0 | 2343 | 9 | 5 | 19 |
| PSR-MSE / 22 | 768 | 0 | 0 | 2339 | 0 | 0 | 19 |
| PSR-MSE / 33 | 777 | 9 | 0 | 2343 | 10 | 6 | 19 |
| RePS / 11 | 474 | 12 | 306 | 1372 | 11 | 978 | 16 |
| RePS / 22 | 609 | 12 | 171 | 1793 | 24 | 570 | 20 |
| RePS / 33 | 60 | 1 | 709 | 191 | 0 | 2148 | 2 |

逐来源统计、区间和配对记录均保存在 [evaluation/test/](evaluation/test/) 各方法目录；原始精度见 [section4_5_new_methods.csv](section4_5_new_methods.csv)。

## 实现来源、任务适配与冻结配置

正式方法名称分别为 CAA-E1-task-adapted、INLP-task-adapted、DoRA-E1-SFT、RePS-E1-task-adapted 和 PSR-MSE-E1-task-adapted；后三者的逐种子名称附加 seed11、seed22、seed33。Codex01 的官方实现和兼容性审计副本见 [upstream_audit/](upstream_audit/)。

| 方法 | 实际复用的官方核心 | 固定版本 |
|---|---|---|
| RePS | AxBench PreferenceVectorIntervention、AdditionIntervention、preference_loss | [41c8332543e5a631f9a8c0a9df38799893ace758](https://github.com/stanfordnlp/axbench/tree/41c8332543e5a631f9a8c0a9df38799893ace758) |
| PSR-MSE | FocusedSteeringModule、mse_on_aligned_activations | [3d916c618d146c5d657f055e432a432b0fa493c6](https://github.com/Nokia-Bell-Labs/steer-like-the-llm/tree/3d916c618d146c5d657f055e432a432b0fa493c6) |
| CAA | generate_save_vectors_for_behavior、add_vector_from_position | [5dabbbd9a0bca5f25e174501e959de378806aa48](https://github.com/nrimsky/CAA/tree/5dabbbd9a0bca5f25e174501e959de378806aa48) |
| INLP | get_debiasing_projection 及原作者分类器接口 | [e1edcc19d808108ab71cbb3afb0389db0206a7eb](https://github.com/shauli-ravfogel/nullspace_projection/tree/e1edcc19d808108ab71cbb3afb0389db0206a7eb) |
| DoRA | PEFT LoraConfig(use_dora=True)、DoraLinearLayer | 实际实现 PEFT 0.21.2；作者参考 [7e2f10abbe8efe212c8fca1d983ae1d04ef13a18](https://github.com/NVlabs/DoRA/tree/7e2f10abbe8efe212c8fca1d983ae1d04ef13a18) |

源码快照和许可证保存位置见 [官方来源记录](official_sources.json)。该文件记录源码准备时的状态；本报告的完成依据为各方法冻结测试 statistics.json 和 status.json。

### 表示引导

- **CAA**：固定第18层（从0计数）。每个 E1 唯一 prompt 使用正确答案和相反答案构成行为对，执行官方正负回答激活差分均值。Qwen 适配序列为原 prompt+单 token 答案+EOS，官方−2索引指向答案；干预从用户结束标记开始，覆盖助手模板和答案位置。开发网格为−1、−0.5、0、0.5、1，按A/P/C/F双端正确率宏平均选择，最终强度1。方向包含4,096个数值，使用全部4,637个prompt。配置见 [caa/configuration.json](caa/configuration.json)、[caa/frozen.json](caa/frozen.json)。
- **RePS**：固定第18层、秩1；每个prompt的正确/错误答案构成双向偏好，保留官方 scaled_simpo 损失、orig_add/orig_sub 分支和原长度缩放。学习率0.04，AdamW，1轮、batch1；干预覆盖除首token外的全部位置。开发强度候选0.5、1、2，以二元负对数似然选择；种子11/22/33分别选0.5/1/1。训练保存4,096维方向和1个偏置，推理沿用官方 AdditionIntervention 对方向加权。配置和选择记录见 [reps/seed11/](reps/seed11/)、[seed22](reps/seed22/)、[seed33](reps/seed33/)。
- **PSR-MSE**：采用 S-PSR_MSE，固定第18层、秩1、answer_only位置；训练对齐最终prompt token和教师答案token，模仿各层教师激活。保留官方未归一化MSE与0.1正则项，学习率0.001、AdamW、1轮、batch1。教师额外指令要求依据决定性事实和当前条件作答，使用拟合集标签保留教师回答正确的4,475/4,637个prompt（排除162个），教师缓存跨种子共享。开发候选0.5、1、2，以二元负对数似然选择；种子11/22/33分别选2/0.5/2。配置见 [psr/seed11/](psr/seed11/)、[seed22](psr/seed22/)、[seed33](psr/seed33/)，实际样本数见 [教师统计](psr/teacher_fit/summary.json)。

### 概念擦除与参数高效微调

- **INLP**：擦除概念定义为“指南叶节点处置或结论文本的可见性”，依据实际文本中的删除/中性占位标记生成标签。固定原prompt末token、末层final norm前的4,096维表示，保留原RMSNorm及输出读出。官方迭代分类与零空间投影使用逻辑回归；开发候选迭代数1/4/8，按任务端点正确率、概念探针正确率及较少迭代数选择，最终迭代1次、擦除秩1。开发概念探针平衡正确率68.938%，证据探针平衡正确率95.976%。投影采用等价低秩存储。配置及定义见 [inlp/selection.json](inlp/selection.json)。
- **DoRA**：使用全部36层的q_proj/v_proj，秩8、alpha16、dropout0；新增参数4,018,176。目标为正确单答案token的全词表交叉熵；A/P/C/F类别等权，类别内来源等权，两端权重合并到唯一prompt。AdamW学习率0.0001、1轮、micro-batch1、梯度累积8、580次优化器更新、warmup3%，启用梯度检查点。配置在开发/测试前固定。三种子配置见 [seed11](dora/full/seed11/frozen_configuration.json)、[seed22](dora/full/seed22/frozen_configuration.json)、[seed33](dora/full/seed33/frozen_configuration.json)。

这些实验使用E1任务答案、概念标签或任务教师定义监督，均在名称和配置中记录适配。五种方法保留各自训练目标，额外A/P一致性或C/F保持损失均未加入。ITI、SEA、LEACE、LoRA沿用项目既有实验，本批新增结果范围为上述五种方法。

## 参数、训练与推理成本

主干参数量为8,190,735,360。下表“新增/训练/保存”依次对应主表 parameter_count、trainable_parameter_count、stored_parameter_count；INLP的训练数值指线性分类与投影拟合，CAA为直接均值估计。单位分别为参数或数值元素。

| 方法 | 新增 / 训练 / 保存 |
|---|---:|
| DoRA | 4,018,176 / 4,018,176 / 4,018,176 |
| PSR-MSE | 8,193 / 8,193 / 8,193 |
| INLP | 4,096 / 4,096 / 4,096 |
| CAA | 4,096 / 0 / 4,096 |
| RePS | 4,097 / 4,097 / 4,097 |

训练时间与测试推理时间分列。测试推理按本方法实际访问的唯一UID计时，包含prompt查询、分词和模型回调，模型加载不计入。各方法路径不同，访问UID数也不同。

| 方法 / 种子 | 训练秒 | 训练 GPU·h | 训练峰值 GiB | 测试推理秒 | 测试 GPU·h | 每唯一输入 ms | 底层前向次数 | 记录推理峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAA | 1194.06 | 0.3317 | 17.195 | 3864.06 | 1.0734 | 192.933 | 40068 | 16.401 |
| DoRA / 11 | 3150.44 | 0.8751 | 19.031 | 5531.10 | 1.5364 | 275.810 | 40120 | 16.658 |
| DoRA / 22 | 3144.39 | 0.8734 | 19.031 | 5523.20 | 1.5342 | 275.911 | 40048 | 16.658 |
| DoRA / 33 | 3145.86 | 0.8739 | 19.031 | 5526.53 | 1.5351 | 276.037 | 40054 | 16.658 |
| INLP | 1072.30 | 0.2450 | — | 3775.22 | 1.0487 | 188.544 | 40058 | 16.401 |
| PSR-MSE / 11 | 2274.79 | 0.6319 | 18.834 | 3974.02 | 1.1039 | 198.413 | 40070 | 18.834 |
| PSR-MSE / 22 | 2274.60 | 0.6318 | 18.833 | 3969.87 | 1.1027 | 198.275 | 40056 | 18.833 |
| PSR-MSE / 33 | 2275.71 | 0.6321 | 18.834 | 3976.56 | 1.1046 | 198.520 | 40074 | 18.834 |
| RePS / 11 | 3997.92 | 1.1105 | 17.613 | 3478.53 | 0.9663 | 187.836 | 37046 | 17.613 |
| RePS / 22 | 3994.50 | 1.1096 | 17.614 | 3699.59 | 1.0277 | 190.917 | 38768 | 17.614 |
| RePS / 33 | 3994.02 | 1.1095 | 17.615 | 3244.22 | 0.9012 | 193.120 | 33604 | 17.615 |

成本记录的具体范围：

- CAA训练1194.06秒，包括未干预prompt前向、正负答案激活和方向估计。开发五强度选择另耗1414.39秒。其原始prompt表示同时提供给INLP。
- INLP直接CPU拟合、探针与开发选择190.19秒，独立运行计入共享特征提取882.12秒，合计1072.30秒。共享特征阶段峰值显存未单独记录，表中保留空值；共享特征费用在合计实际多方法成本时去重。
- PSR每个种子的训练成本按独立运行口径包含1109.97秒教师准备。实际三种子执行时教师缓存构建一次，额外两个教师准备计费在整批实际成本合计时扣除。
- RePS开发选择分别826.34、826.26、824.82秒；PSR分别863.61、861.44、863.76秒，另列于各冻结配置。DoRA使用预先固定的单配置。
- RePS和PSR的推理峰值字段沿用当前进程累计allocated峰值，可包含此前训练阶段；DoRA在评价阶段重置，INLP逐请求重置，CAA使用独立评价进程记录。按需UID评价时间与固定长度单请求延迟分别报告。

### DoRA独立固定长度计时

DoRA种子11还完成E11的40次独立测量：4个固定开发prompt、512/2048 token两档，每档预热3次，随后5轮×4个输入；batch1、BF16、SDPA，CUDA同步计时。统一队列实际分配NVIDIA A100-SXM4-80GB物理GPU4。下表为该工作负载的单请求均值与实测单请求峰值。

| 主输入token | 端到端延迟 ms | 单请求峰值 GiB |
|---|---:|---:|
| 512 | 155.015 | 15.670 |
| 2,048 | 323.988 | 16.086 |

输入截断用于固定长度计时，任务准确率使用前述完整prompt评价。E11的配置、逐次测量和统计由发布包保存在 [efficiency/dora_seed11/](efficiency/dora_seed11/)。

## 产物与复现入口

代码位于 [experiments/aster_codex03_external_20261112/](../../experiments/aster_codex03_external_20261112/)。主表由已完成的测试统计生成，所有数值保留原始CSV精度。正式结果按以下位置组织：

| 内容 | 位置 |
|---|---|
| 11次完整主表 | [section4_5_new_methods.csv](section4_5_new_methods.csv) |
| 5方法种子汇总 | [section4_5_method_summary.csv](section4_5_method_summary.csv) |
| 逐例、逐对、完整路径、来源区间、成本和实现信息 | [evaluation/test/](evaluation/test/)各方法目录 |
| 完整开发路径评价 | [evaluation/dev/](evaluation/dev/)，共5次：CAA、INLP及DoRA三种子 |
| RePS/PSR开发强度选择分数 | 各seed目录的dev_selection_scores与frozen.json |
| 官方版本与许可证位置 | [official_sources.json](official_sources.json)、[upstream_audit/](upstream_audit/) |
| 数据版本、划分和输入引用 | [prepared/summary.json](prepared/summary.json)、[prepared/input_reference.json](prepared/input_reference.json) |

发布包对逐例JSONL和大型CSV使用gzip压缩，保留相同目录结构及.gz后缀；顶层主表CSV、统计JSON和配置可直接读取。原始模型、完整数据、激活与适配器文件保存在服务器路径，按各配置中的原始路径查找。

服务器工作目录为 /home/gcx/aster-codex03-external-20261112，环境为 /home/gcx/gcx261014。复现采用冻结配置和对应种子；GPU训练或推理由统一队列分配已授权卡。

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/aster-codex03-external-20261112
export PYTHONPATH=/home/gcx/aster_codex03_external_assets_20261112/python:/home/gcx/aster-codex03-external-20261112
# 只重新汇总现有结果，使用CPU
python -m experiments.aster_codex03_external_20261112.finalize
```

训练与冻结评价的入口如下；各入口使用报告目录中已记录的配置，种子11/22/33分别运行。

| 方法 | 模块与入口 |
|---|---|
| CAA | caa：--mode train → --mode select → --mode evaluate --split dev/test |
| INLP | inlp：fit及run；使用CAA导出的原始prompt表示与已冻结selection.json |
| DoRA | dora：run-seed --seed 11/22/33 --output reports/aster_codex03_external_20261112/dora/full |
| RePS | reps：all --seed 11/22/33 --output 对应seed目录 |
| PSR | psr：先teacher生成一次教师缓存，再all --seed 11/22/33 --teacher-dir 指向共享教师缓存 |

统一模块前缀为 python -m experiments.aster_codex03_external_20261112.；具体参数、输出路径和依赖顺序保存在服务器统一队列 /home/gcx/aster-gpu-queue-20261113/codex03_jobs.json。冻结测试统计可由 evaluate 模块直接读取已保存分数重建。
