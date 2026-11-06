# ASTER 连续夹角衰减固定对照报告

本轮仅替换夹角权重，固定原数据、51 个头、单类秩上限 4、能量目标 0.95、45° 设置、信号空间及参考均值。直接从保存基构建算子，没有重新拟合、筛样本或搜索参数。

测试集主指标：完整阅读 96.57%，硬筛选 97.08%，连续衰减 97.08%。
测试集 C/F 唯一输入相对完整阅读的变化为 C：纠正 0、新增错误 0；F：纠正 0、新增错误 0。历史完整阅读路线：纠正 0、新增错误 6。
432 例路径：连续衰减正确 399 例；相对完整阅读纠正 4、退化 9，相对硬筛选纠正 4、退化 6。

## 固定方法与必要验证

令 (I−SSᵀ)U = Q diag(s) Vᵀ，τ = sin45°，D = UV diag(s / max(s², τ²)) Qᵀ，z′ = z − D(z−μ)。
夹角不小于 45° 时与硬筛选相同；30° 方向擦除自身分量的 50%，15° 约 13.4%，零夹角编辑为零。信号保持是保存隐藏空间上的代数性质，输出判断另行检验。
硬筛选使用 120 个方向，连续衰减保留 244 个保存方向，按自身擦除比例求和为 166.834。这没有改变每类秩上限。
保存精度下，最大信号保持误差 5.46e-08；最大算子范数 1.411382 < √2；高夹角部分相对旧算子的最大误差 8.2e-08。
每头干扰基残留的相对 Frobenius 范数均值为 0.4893；最大非幂等残差 0.5443。残留和非幂等为本方法允许的结果。
夹角解析检查、既有算子检查、零干预一致和目标头／最终位置检查已通过。两个 GPU 上首个输入的自然分数均与保存基线一致；自主路径控制器的根节点起步与恢复执行检查通过。
逐头结果见 [operator_residuals.csv](results/operator_residuals.csv)，逐方向权重见 [operator_directions.csv](results/operator_directions.csv)。

## 主实验

评分覆盖全部 1,290 个现成输入、1,053 对；拟合／开发／测试输入数分别为 561／119／610。下表为测试集，全部划分和逐输入结果见 CSV。
PA 要求配对两侧均正确；PA_P 对临床背景与治疗偏好等权，主指标为 A/P/C/F 四类 PA 的平均值。单路校准为原保存参考，没有重新训练。

| 方法 | PA_A | PA_P | PA_C | PA_F | 宏平均 | 全部测试输入准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 99.00% | 95.92% | 92.86% | 98.50% | 96.57% | 601/610（98.52%） |
| aster | 99.00% | 97.96% | 92.86% | 98.50% | 97.08% | 603/610（98.85%） |
| soft_angle | 99.00% | 97.96% | 92.86% | 98.50% | 97.08% | 602/610（98.69%） |
| calibrated_full | 99.00% | 97.96% | 92.86% | 98.50% | 97.08% | 603/610（98.85%） |

| 范围 | 对照 | 纠正 | 退化 |
| --- | --- | ---: | ---: |
| fit | aster_vs_full | 1 | 4 |
| fit | soft_angle_vs_full | 1 | 5 |
| fit | soft_angle_vs_aster | 0 | 1 |
| dev | aster_vs_full | 0 | 0 |
| dev | soft_angle_vs_full | 0 | 0 |
| dev | soft_angle_vs_aster | 0 | 0 |
| test | aster_vs_full | 2 | 0 |
| test | soft_angle_vs_full | 2 | 1 |
| test | soft_angle_vs_aster | 0 | 1 |
| all | aster_vs_full | 3 | 4 |
| all | soft_angle_vs_full | 3 | 6 |
| all | soft_angle_vs_aster | 0 | 2 |

| 划分 | C/F 唯一输入 | 输入数 | 连续衰减相对 full 纠正／新增错误 | 相对硬筛选纠正／新增错误 |
| --- | --- | ---: | ---: | ---: |
| fit | C | 26 | 0/1 | 0/1 |
| fit | F | 316 | 0/0 | 0/0 |
| dev | C | 2 | 0/0 | 0/0 |
| dev | F | 80 | 0/0 | 0/0 |
| test | C | 28 | 0/0 | 0/0 |
| test | F | 400 | 0/0 | 0/0 |
| all | C | 56 | 0/1 | 0/1 |
| all | F | 796 | 0/0 | 0/0 |

测试配对宏平均持平，但逐输入并非全部不变。新增 P 输入错误所在配对在硬筛选下已因另一侧判错而失分，因此不会再次降低该配对指标。全部划分的 C/F 新增错误另列，不能把测试子集的结果推广为全部输入都无损伤。

主表与按来源重采样的测试区间见 [main_effects.json](results/main_effects.json)、[main_comparison.csv](results/main_comparison.csv)。全部输入及变化见 [main_inputs.csv](results/main_inputs.csv)，翻转见 [main_prediction_flips.csv](results/main_prediction_flips.csv)。

## 双向联合修补

保留开发 59 对、测试 298 对 A/P 配对，新增 D_soft(z右−z左) 的双向联合修补。恢复比例仍为两个方向归一化效应的平均值；仅 |TE|≥0.1 时计算比例，其他配对保留原始效应。没有按恢复结果筛选配对。

| 划分 | 类型 | 分量 | 全部对数 | 可归一化对数 | 平均恢复比例 |
| --- | --- | --- | ---: | ---: | ---: |
| dev | A | full_head | 40 | 36 | 85.27% |
| dev | A | nuisance_orthogonal | 40 | 36 | 39.83% |
| dev | A | screened_orthogonal | 40 | 36 | -13.49% |
| dev | A | oblique | 40 | 36 | -12.45% |
| dev | A | soft_angle | 40 | 36 | 5.09% |
| dev | P | full_head | 19 | 16 | 75.73% |
| dev | P | nuisance_orthogonal | 19 | 16 | 48.37% |
| dev | P | screened_orthogonal | 19 | 16 | 4.89% |
| dev | P | oblique | 19 | 16 | 12.81% |
| dev | P | soft_angle | 19 | 16 | 13.55% |
| test | A | full_head | 200 | 185 | 87.71% |
| test | A | nuisance_orthogonal | 200 | 185 | 63.08% |
| test | A | screened_orthogonal | 200 | 185 | 11.88% |
| test | A | oblique | 200 | 185 | 15.26% |
| test | A | soft_angle | 200 | 185 | 9.98% |
| test | P | full_head | 98 | 92 | 84.11% |
| test | P | nuisance_orthogonal | 98 | 92 | 69.34% |
| test | P | screened_orthogonal | 98 | 92 | 11.85% |
| test | P | oblique | 98 | 92 | 14.42% |
| test | P | soft_angle | 98 | 92 | 20.67% |

测试 A：连续衰减相对硬筛选斜投影的恢复比例变化为 -5.28 个百分点；未经筛选的正交分量为 63.08%。

测试 P：连续衰减相对硬筛选斜投影的恢复比例变化为 +6.26 个百分点；未经筛选的正交分量为 69.34%。

完整五种分量对照与逐对双向分数见 [mechanism_comparison.csv](results/mechanism_comparison.csv)、[mechanism_all_pairs.jsonl](results/mechanism_all_pairs.jsonl.gz)。

## 历史事实、偏好及删除补回

仅补连续衰减的 569 个输入评分，沿用各自原系统提示、用户提示和标签。504 个事实／偏好输入来自 28 个节点的三条路线；另有 65 个删除补回输入。此处 full 表示该路线原提示下的无干预模型。

| 路线 | 方法 | PA_F | PA_P | 输入准确率 |
| --- | --- | ---: | ---: | ---: |
| full | full | 25/56（44.64%） | 59/84（70.24%） | 119/168（70.83%） |
| full | aster | 22/56（39.29%） | 57/84（67.86%） | 114/168（67.86%） |
| full | soft_angle | 21/56（37.50%） | 56/84（66.67%） | 113/168（67.26%） |
| direct | full | 26/56（46.43%） | 60/84（71.43%） | 120/168（71.43%） |
| direct | aster | 21/56（37.50%） | 57/84（67.86%） | 115/168（68.45%） |
| direct | soft_angle | 22/56（39.29%） | 57/84（67.86%） | 116/168（69.05%） |
| condition_ancestors | full | 17/56（30.36%） | 54/84（64.29%） | 111/168（66.07%） |
| condition_ancestors | aster | 18/56（32.14%） | 54/84（64.29%） | 111/168（66.07%） |
| condition_ancestors | soft_angle | 15/56（26.79%） | 53/84（63.10%） | 109/168（64.88%） |

| 路线 | 数值状态 | 对照 | 输入数 | 纠正 | 新增错误 |
| --- | --- | --- | ---: | ---: | ---: |
| full | below | soft_angle_vs_full | 56 | 0 | 1 |
| full | below | soft_angle_vs_aster | 56 | 0 | 1 |
| full | above | soft_angle_vs_full | 56 | 0 | 3 |
| full | above | soft_angle_vs_aster | 56 | 0 | 0 |
| full | same_side | soft_angle_vs_full | 56 | 0 | 2 |
| full | same_side | soft_angle_vs_aster | 56 | 0 | 0 |
| full | exact_threshold | soft_angle_vs_full | 12 | 0 | 0 |
| full | exact_threshold | soft_angle_vs_aster | 12 | 0 | 0 |
| direct | below | soft_angle_vs_full | 56 | 0 | 1 |
| direct | below | soft_angle_vs_aster | 56 | 0 | 0 |
| direct | above | soft_angle_vs_full | 56 | 0 | 3 |
| direct | above | soft_angle_vs_aster | 56 | 1 | 0 |
| direct | same_side | soft_angle_vs_full | 56 | 0 | 0 |
| direct | same_side | soft_angle_vs_aster | 56 | 0 | 0 |
| direct | exact_threshold | soft_angle_vs_full | 12 | 0 | 0 |
| direct | exact_threshold | soft_angle_vs_aster | 12 | 0 | 0 |
| condition_ancestors | below | soft_angle_vs_full | 56 | 0 | 4 |
| condition_ancestors | below | soft_angle_vs_aster | 56 | 0 | 4 |
| condition_ancestors | above | soft_angle_vs_full | 56 | 1 | 0 |
| condition_ancestors | above | soft_angle_vs_aster | 56 | 0 | 0 |
| condition_ancestors | same_side | soft_angle_vs_full | 56 | 1 | 0 |
| condition_ancestors | same_side | soft_angle_vs_aster | 56 | 2 | 0 |
| condition_ancestors | exact_threshold | soft_angle_vs_full | 12 | 0 | 0 |
| condition_ancestors | exact_threshold | soft_angle_vs_aster | 12 | 0 | 0 |

exact_threshold 指保存数值恰好等于条件中的数值阈值；below／above／same_side 沿用原构造标签。分数接近零与数值恰好等于条件阈值分别统计。

| 删除／补回路线 | full | aster | soft_angle |
| --- | ---: | ---: | ---: |
| condition_ancestors | 9/11（81.82%） | 9/11（81.82%） | 9/11（81.82%） |
| conservative | 10/11（90.91%） | 10/11（90.91%） | 10/11（90.91%） |
| direct | 11/11（100.00%） | 11/11（100.00%） | 10/11（90.91%） |
| full | 10/11（90.91%） | 10/11（90.91%） | 10/11（90.91%） |
| irrelevant_restore | 10/10（100.00%） | 10/10（100.00%） | 10/10（100.00%） |
| necessary_restore | 11/11（100.00%） | 11/11（100.00%） | 11/11（100.00%） |

事实、偏好、数值分层、删除补回及纠正／退化见 [historical_fact_preference.csv](results/historical_fact_preference.csv)、[historical_numeric_strata.csv](results/historical_numeric_strata.csv)、[historical_deletion_restore.csv](results/historical_deletion_restore.csv)、[historical_corrections.json](results/historical_corrections.json)。

## 分数阈值与共同偏移

逐对同时记录两侧分数、配对差分变化与共同偏移：共同偏移 = (左侧分数变化 + 右侧分数变化) / 2。配对差分不变不能推出两侧预测不变。
开发集完整阅读分数绝对值 ≤4 的输入为 0 个；该分组仅作描述，不删样本或参与选参。不能单凭开发准确率持平判断保护有效。
主实验的全部 1,053 对及历史每条路线的全部配对都已记录分数变化，见 [main_pair_shifts.csv](results/main_pair_shifts.csv)、[historical_pair_shifts.csv](results/historical_pair_shifts.csv)。统计见 [threshold_analysis.json](results/threshold_analysis.json)。
所有主实验、历史输入与自主路径共同节点上的预测翻转合并于 [all_prediction_flips.csv](results/all_prediction_flips.csv)。路径分叉后只有一方访问的节点另列，不虚构另一方法的分数。

## 自主路径

连续衰减对全部 432 例从 G0 根节点重新执行，标准路径仅用于执行后的判分。原 100 例是嵌套子集。完整阅读和硬筛选复用既有自主执行结果。

| 范围 | 方法 | 完整路径正确率 | 相对 full 纠正／退化 | 相对硬筛选纠正／退化 |
| --- | --- | ---: | ---: | ---: |
| all_432 | aster | 401/432（92.82%） | 2/5 | 0/0 |
| all_432 | full | 404/432（93.52%） | 0/0 | 5/2 |
| all_432 | soft_angle | 399/432（92.36%） | 4/9 | 4/6 |
| original_100 | aster | 95/100（95.00%） | 0/1 | 0/0 |
| original_100 | full | 96/100（96.00%） | 0/0 | 1/0 |
| original_100 | soft_angle | 94/100（94.00%） | 1/3 | 1/2 |

路径变化、首个分歧条件与分数见 [path_changed_cases.csv](results/path_changed_cases.csv)；全部共同节点上的翻转见 [path_node_flips.csv](results/path_node_flips.csv)。未共同访问节点见 [path_unshared_nodes.csv](results/path_unshared_nodes.csv)。

## 分开判断

- 测试 A 修补恢复比例相对硬筛选斜投影降低；连续衰减的 PA_A 为 99.00%，硬筛选为 99.00%。
- 测试 P 修补恢复比例相对硬筛选斜投影提高；连续衰减的 PA_P 为 97.96%，硬筛选为 97.96%。
- 测试 C 的 28 个唯一输入，相对完整阅读本组诊断未观察到新增损伤；纠正 0 个。
- 测试 F 的 400 个唯一输入，相对完整阅读本组诊断未观察到新增损伤；纠正 0 个。
- 历史 full 路线的 168 个事实／偏好输入，相对原模型出现 6 个原本正确输入的新错误，纠正 0 个；恰好等于数值阈值的 12 个输入新增错误 0 个。
- 历史 direct 路线的 168 个事实／偏好输入，相对原模型出现 4 个原本正确输入的新错误，纠正 0 个；恰好等于数值阈值的 12 个输入新增错误 0 个。
- 历史 condition_ancestors 路线的 168 个事实／偏好输入，相对原模型出现 4 个原本正确输入的新错误，纠正 2 个；恰好等于数值阈值的 12 个输入新增错误 0 个。
- 路径相对完整阅读净变化 -5/432，相对硬筛选净变化 -2/432。

以上判断仅对应实际评价的输入与既有标签。代数信号保持不能推出所有数值阈值判断安全；本轮正负结果全部保留，没有根据测试结果调整参数。

## 执行与交付

仅 GPU 0、1 各一份 Qwen3-8B，bfloat16；评分总耗时 26.03 分钟，峰值保留显存分别为 17.82／18.23 GiB。
主实验 1,290 输入、机制 357 对、历史 569 输入及路径 432 例均完整，无缺分数或路径技术失败。原始输入、完整路径和运行日志保留在服务器。
实现位于 code/experiments/aster_closed_form/，固定配置见 configuration.json。执行与复算说明见 README.zh.md。
服务器目录：/home/gcx/aster_soft_angle_20261106。
