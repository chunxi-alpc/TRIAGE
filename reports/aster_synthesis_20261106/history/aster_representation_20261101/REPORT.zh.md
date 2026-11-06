> Source snapshot: /home/gcx/aster_representation_20261101/REPORT.zh.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# 处置盲表示引导的内部干预实验

新增训练200例／20来源，原年龄最小对继续监督条件翻转；开发100例选配置，新100例／10来源确认。
骨干冻结。所有方法使用相同合法路径与终点处置读取。原120例不作为本轮独立确认。

## development

| 方法 | 原指南联合正确 | 交换处置联合正确 | 交换前后路径一致率 |
|---|---:|---:|---:|
| zero | 82/100 | 79/100 | 0.820 |
| static | 75/100 | 74/100 | 0.860 |
| fusion | 85/100 | 83/100 | 0.880 |
| blind_lookup | 80/100 | 80/100 | 1.000 |
| ordinary | 91/100 | 84/100 | 0.910 |
| no_js | 94/100 | 93/100 | 0.970 |
| context | 85/100 | 83/100 | 0.920 |
| blind_readout | 86/100 | 86/100 | 1.000 |
| full | 91/100 | 94/100 | 0.960 |

| 完整方法相对对照 | 视图 | 纠正 | 退化 | 来源等权差值及95%区间 |
|---|---|---:|---:|---|
| zero | original | 10 | 1 | 0.090 [0.040, 0.150] |
| zero | swap | 16 | 1 | 0.150 [0.050, 0.260] |
| static | original | 18 | 2 | 0.160 [0.070, 0.260] |
| static | swap | 20 | 0 | 0.200 [0.150, 0.280] |
| fusion | original | 8 | 2 | 0.060 [0.010, 0.110] |
| fusion | swap | 12 | 1 | 0.110 [0.020, 0.220] |
| blind_lookup | original | 14 | 3 | 0.110 [0.040, 0.190] |
| blind_lookup | swap | 14 | 0 | 0.140 [0.080, 0.200] |
| ordinary | original | 2 | 2 | 0.000 [-0.030, 0.030] |
| ordinary | swap | 10 | 0 | 0.100 [0.040, 0.170] |
| no_js | original | 1 | 4 | -0.030 [-0.080, 0.010] |
| no_js | swap | 1 | 0 | 0.010 [0.000, 0.030] |
| context | original | 6 | 0 | 0.060 [0.030, 0.090] |
| context | swap | 11 | 0 | 0.110 [0.040, 0.190] |
| blind_readout | original | 10 | 5 | 0.050 [-0.020, 0.110] |
| blind_readout | swap | 10 | 2 | 0.080 [0.040, 0.120] |

| 原指南：盲判查表／完整上下文零干预 | 病例数 | 完整方法正确数 |
|---|---:|---:|
| 正确／正确 | 71 | 70 |
| 正确／错误 | 9 | 7 |
| 错误／正确 | 11 | 11 |
| 错误／错误 | 9 | 3 |
## confirmation

| 方法 | 原指南联合正确 | 交换处置联合正确 | 交换前后路径一致率 |
|---|---:|---:|---:|
| zero | 68/100 | 61/100 | 0.790 |
| static | 71/100 | 67/100 | 0.820 |
| fusion | 83/100 | 79/100 | 0.890 |
| blind_lookup | 78/100 | 78/100 | 1.000 |
| ordinary | 87/100 | 81/100 | 0.890 |
| no_js | 91/100 | 84/100 | 0.870 |
| context | 73/100 | 72/100 | 0.790 |
| blind_readout | 79/100 | 79/100 | 1.000 |
| full | 86/100 | 83/100 | 0.910 |

| 完整方法相对对照 | 视图 | 纠正 | 退化 | 来源等权差值及95%区间 |
|---|---|---:|---:|---|
| zero | original | 21 | 3 | 0.180 [0.100, 0.270] |
| zero | swap | 23 | 1 | 0.220 [0.120, 0.330] |
| static | original | 18 | 3 | 0.150 [0.100, 0.200] |
| static | swap | 21 | 5 | 0.160 [0.090, 0.230] |
| fusion | original | 8 | 5 | 0.030 [-0.010, 0.070] |
| fusion | swap | 8 | 4 | 0.040 [-0.010, 0.090] |
| blind_lookup | original | 16 | 8 | 0.080 [0.020, 0.140] |
| blind_lookup | swap | 14 | 9 | 0.050 [-0.020, 0.110] |
| ordinary | original | 6 | 7 | -0.010 [-0.080, 0.060] |
| ordinary | swap | 7 | 5 | 0.020 [-0.030, 0.080] |
| no_js | original | 1 | 6 | -0.050 [-0.080, -0.020] |
| no_js | swap | 4 | 5 | -0.010 [-0.060, 0.040] |
| context | original | 15 | 2 | 0.130 [0.050, 0.210] |
| context | swap | 12 | 1 | 0.110 [0.040, 0.190] |
| blind_readout | original | 15 | 8 | 0.070 [0.010, 0.140] |
| blind_readout | swap | 13 | 9 | 0.040 [-0.030, 0.110] |

| 原指南：盲判查表／完整上下文零干预 | 病例数 | 完整方法正确数 |
|---|---:|---:|
| 正确／正确 | 58 | 56 |
| 正确／错误 | 20 | 14 |
| 错误／正确 | 10 | 9 |
| 错误／错误 | 12 | 7 |

## 条件翻转

训练年龄对与已暴露确认年龄对分别报告；后者只作机制诊断，不称新的独立确认。

| 方法 | 数据 | 原指南两态均正确 | 处置交换两态均正确 |
|---|---|---:|---:|
| zero | age_train | 5/8 | 6/8 |
| zero | age_diagnostic | 2/2 | 0/2 |
| static | age_train | 7/8 | 7/8 |
| static | age_diagnostic | 2/2 | 2/2 |
| fusion | age_train | 6/8 | 6/8 |
| fusion | age_diagnostic | 2/2 | 1/2 |
| blind_lookup | age_train | 8/8 | 8/8 |
| blind_lookup | age_diagnostic | 2/2 | 2/2 |
| ordinary | age_train | 8/8 | 8/8 |
| ordinary | age_diagnostic | 2/2 | 2/2 |
| no_js | age_train | 8/8 | 8/8 |
| no_js | age_diagnostic | 2/2 | 2/2 |
| context | age_train | 8/8 | 7/8 |
| context | age_diagnostic | 2/2 | 2/2 |
| blind_readout | age_train | 8/8 | 8/8 |
| blind_readout | age_diagnostic | 2/2 | 2/2 |
| full | age_train | 8/8 | 8/8 |
| full | age_diagnostic | 2/2 | 2/2 |

## 结论规则

比较完整方法与零干预、直接盲判查表、标签融合、普通低秩、同样训练的上下文条件信息、盲表示分类器及无不变性损失对照。
直接盲判与盲表示分类器在每个节点均作Y/N二选一，Unknown概率只作诊断，避免用拒答人为削弱对照。
上下文对照把当前节点的T/F/U概率写入问题，并训练同参数量的普通低秩模块；盲表示分类器参数量匹配，不读取完整指南表示。
互补表按完整路径计数，两侧路径都错时仍可能存在节点层面的互补，不能据此直接认定新推理能力。
开发或确认若无额外收益，应如实报告未获支持；结构整理收益不算内部编辑收益。
时间包含表示提取及诊断，需单独测量部署成本，不能直接声称提速。
