# ASTER 三模块瓶颈实验报告

初轮状态：complete；后续阶段状态：complete。

43例、2来源为已暴露开发诊断；120例、12新来源用于验证，根分支74∶46。所有金标路线结果仅用于算子诊断。
主线保留方向学习、处置盲评估和键编辑；均值差默认，谱为对照。

## 1. 开发集算子对照

| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |
|---|---:|---:|---:|---:|
| zero | 43 | 20 | 19 | 0 |
| spectral025 | 43 | 20 | 19 | 0 |
| mean025 | 43 | 21 | 20 | 0 |
| clip | 43 | 43 | 43 | 0 |
| neutral | 43 | 38 | 40 | 0 |
| hard | 43 | 40 | 40 | 0 |
| hard_siblings | 43 | 23 | 21 | 0 |
| bias1 | 43 | 21 | 32 | 0 |
| bias2 | 43 | 25 | 40 | 0 |
| bias4 | 43 | 14 | 35 | 0 |
| bias8 | 43 | 0 | 0 | 0 |

## 2. 新来源算子验证

| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |
|---|---:|---:|---:|---:|
| zero | 120 | 78 | 84 | 0 |
| clip | 120 | 111 | 120 | 0 |
| hard | 120 | 90 | 116 | 0 |
| mean025 | 120 | 77 | 84 | 0 |
| bias2 | 120 | 85 | 112 | 0 |

## 3. P0连续诊断

下列比值仅比较同一注意力打分空间，不能与输出logit边际直接比较。词元和头不作为独立样本。

| 条件 | 位移上界／零编辑打分标准差中位数 | 预期符号比例均值 | 实际绝对位移均值 |
|---|---:|---:|---:|
| spectral025 | 0.0972 | 0.9830 | 0.043412 |
| mean025 | 0.0973 | 0.9859 | 0.043323 |

## 层窗口与选择性重算

| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |
|---|---:|---:|---:|---:|
| window0 | 43 | 21 | 20 | 0 |
| window6 | 43 | 21 | 21 | 0 |
| window12 | 43 | 16 | 18 | 0 |
| window18 | 43 | 28 | 28 | 0 |
| window24 | 43 | 18 | 26 | 0 |
| window30 | 43 | 23 | 23 | 0 |
| repair10 | 43 | 39 | 42 | 0 |
| repair20 | 43 | 38 | 43 | 0 |

## 头覆盖

| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |
|---|---:|---:|---:|---:|
| heads4 | 43 | 28 | 35 | 0 |
| heads8 | 43 | 26 | 36 | 0 |
| heads16 | 43 | 20 | 30 | 0 |
| heads32 | 43 | 26 | 33 | 0 |
| heads48 | 43 | 28 | 28 | 0 |

## 开发集匹配位移键对照

| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |
|---|---:|---:|---:|---:|
| matched_bias | 43 | 28 | 35 | 0 |
| matched_mean | 43 | 25 | 28 | 0 |
| least_squares_mean | 43 | 25 | 28 | 0 |
| matched_spectral | 43 | 24 | 27 | 0 |
| matched_lowfreq | 43 | 27 | 29 | 0 |

## 模块二

新100例开发集选择：direct:0.0。新来源不参与选择。

| 方法 | 新100例开发完整路径 | 新来源完整路径 | 新来源接受节点 | 新来源错误接受节点 |
|---|---:|---:|---:|---:|
| legacy:0.0 | 21/100 | 50/120 | 415 | 42 |
| calibrated:0.0 | 36/100 | 64/120 | 492 | 72 |
| direct:0.0 | 46/100 | 69/120 | 482 | 19 |
| direct:0.1 | 45/100 | 69/120 | 482 | 19 |
| direct:0.2 | 42/100 | 68/120 | 474 | 13 |

模块二直接读取原始病历与规则条件，不抽取结构化事实、不拆原子条件。只比较原分类器、NONE校正及直接T/F/U判断；停止门控后保留已接受祖先。

## 开发预测路径执行

| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |
|---|---:|---:|---:|---:|
| legacy_0.0_clip | 43 | 22 | 25 | 0 |
| legacy_0.0_hard | 43 | 23 | 26 | 0 |
| legacy_0.0_bias | 43 | 22 | 25 | 0 |
| legacy_0.0_mean | 43 | 20 | 19 | 0 |
| calibrated_0.0_clip | 43 | 26 | 28 | 0 |
| calibrated_0.0_hard | 43 | 26 | 27 | 0 |
| calibrated_0.0_bias | 43 | 19 | 26 | 0 |
| calibrated_0.0_mean | 43 | 21 | 20 | 0 |
| direct_0.0_clip | 43 | 23 | 26 | 0 |
| direct_0.0_hard | 43 | 25 | 27 | 0 |
| direct_0.0_bias | 43 | 23 | 26 | 0 |
| direct_0.0_mean | 43 | 20 | 19 | 0 |
| calibrated_marginal3 | 43 | 21 | 27 | 0 |

## 新来源预测路径执行

| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |
|---|---:|---:|---:|---:|
| legacy_0.0_clip | 120 | 70 | 82 | 0 |
| legacy_0.0_hard | 120 | 66 | 81 | 0 |
| legacy_0.0_bias | 120 | 62 | 81 | 0 |
| legacy_0.0_mean | 120 | 78 | 84 | 0 |
| calibrated_0.0_clip | 120 | 69 | 79 | 0 |
| calibrated_0.0_hard | 120 | 65 | 79 | 0 |
| calibrated_0.0_bias | 120 | 62 | 81 | 0 |
| calibrated_0.0_mean | 120 | 79 | 85 | 0 |
| direct_0.0_clip | 120 | 87 | 96 | 0 |
| direct_0.0_hard | 120 | 78 | 98 | 0 |
| direct_0.0_bias | 120 | 75 | 91 | 0 |
| direct_0.0_mean | 120 | 77 | 84 | 0 |
| calibrated_marginal3 | 120 | 66 | 86 | 0 |

## 新来源配对变化

| 条件 | 纠正 | 退化 | 来源等权差 | 来源重采样95%区间 |
|---|---:|---:|---:|---|
| validation/bias2 | 21 | 14 | 0.0583 | [-0.041666666666666664, 0.15833333333333335] |
| validation/clip | 36 | 3 | 0.2750 | [0.11666666666666668, 0.425] |
| validation/hard | 25 | 13 | 0.1000 | [-0.016666666666666666, 0.21666666666666667] |
| validation/mean025 | 0 | 1 | -0.0083 | [-0.025000000000000005, 0.0] |
| validation/zero | 0 | 0 | 0.0000 | [0.0, 0.0] |
| predicted_validation/calibrated_0.0_bias | 14 | 30 | -0.1333 | [-0.25833333333333336, -0.008333333333333331] |
| predicted_validation/calibrated_0.0_clip | 18 | 27 | -0.0750 | [-0.22520833333333332, 0.07520833333333223] |
| predicted_validation/calibrated_0.0_hard | 16 | 29 | -0.1083 | [-0.225, 0.016666666666666663] |
| predicted_validation/calibrated_0.0_mean | 1 | 0 | 0.0083 | [0.0, 0.025000000000000005] |
| predicted_validation/calibrated_marginal3 | 15 | 27 | -0.1000 | [-0.2333333333333333, 0.025000000000000005] |
| predicted_validation/direct_0.0_bias | 16 | 19 | -0.0250 | [-0.09166666666666669, 0.05000000000000001] |
| predicted_validation/direct_0.0_clip | 26 | 17 | 0.0750 | [-0.041666666666666664, 0.19999999999999998] |
| predicted_validation/direct_0.0_hard | 20 | 20 | 0.0000 | [-0.09166666666666667, 0.09166666666666666] |
| predicted_validation/direct_0.0_mean | 0 | 1 | -0.0083 | [-0.025000000000000005, 0.0] |
| predicted_validation/legacy_0.0_bias | 13 | 29 | -0.1333 | [-0.25833333333333336, -0.008333333333333333] |
| predicted_validation/legacy_0.0_clip | 17 | 25 | -0.0667 | [-0.19999999999999998, 0.06666666666666667] |
| predicted_validation/legacy_0.0_hard | 14 | 26 | -0.1000 | [-0.2085416666666667, 0.016666666666666673] |
| predicted_validation/legacy_0.0_mean | 0 | 0 | 0.0000 | [0.0, 0.0] |

## 错误门控曲线

ε为每个参考决策位置的扰动概率，每条路径最多在首次扰动处改错兄弟或停止；错误兄弟后保留合法但未确认的子树。开发3个种子，新来源2个非零剂量、1个种子；共享扰动不视为新增独立病例；同一病例、算子和路径的确定性重复复用已生成结果，并记录来源。

| 集合 | 类型 | ε | 算子 | 完整路径准确率 |
|---|---|---:|---|---:|
| errors_dev | stop | 0.1 | bias | 0.5271 |
| errors_dev | stop | 0.1 | clip | 0.7364 |
| errors_dev | stop | 0.1 | hard | 0.6589 |
| errors_dev | wrong | 0.1 | bias | 0.3178 |
| errors_dev | wrong | 0.1 | clip | 0.5039 |
| errors_dev | wrong | 0.1 | hard | 0.4574 |
| errors_dev | stop | 0.2 | bias | 0.4806 |
| errors_dev | stop | 0.2 | clip | 0.6512 |
| errors_dev | stop | 0.2 | hard | 0.5504 |
| errors_dev | wrong | 0.2 | bias | 0.2171 |
| errors_dev | wrong | 0.2 | clip | 0.2791 |
| errors_dev | wrong | 0.2 | hard | 0.2558 |
| errors_dev | stop | 0.3 | bias | 0.4729 |
| errors_dev | stop | 0.3 | clip | 0.5504 |
| errors_dev | stop | 0.3 | hard | 0.4729 |
| errors_dev | wrong | 0.3 | bias | 0.1395 |
| errors_dev | wrong | 0.3 | clip | 0.1318 |
| errors_dev | wrong | 0.3 | hard | 0.1085 |
| errors_dev | stop | 0.5 | bias | 0.4574 |
| errors_dev | stop | 0.5 | clip | 0.5194 |
| errors_dev | stop | 0.5 | hard | 0.4651 |
| errors_dev | wrong | 0.5 | bias | 0.1085 |
| errors_dev | wrong | 0.5 | clip | 0.0388 |
| errors_dev | wrong | 0.5 | hard | 0.0310 |
| errors_validation | stop | 0.2 | bias | 0.5667 |
| errors_validation | stop | 0.2 | clip | 0.7250 |
| errors_validation | stop | 0.2 | hard | 0.6833 |
| errors_validation | wrong | 0.2 | bias | 0.2833 |
| errors_validation | wrong | 0.2 | clip | 0.3333 |
| errors_validation | wrong | 0.2 | hard | 0.3000 |
| errors_validation | stop | 0.5 | bias | 0.5833 |
| errors_validation | stop | 0.5 | clip | 0.6583 |
| errors_validation | stop | 0.5 | hard | 0.6667 |
| errors_validation | wrong | 0.5 | bias | 0.1500 |
| errors_validation | wrong | 0.5 | clip | 0.0750 |
| errors_validation | wrong | 0.5 | hard | 0.0583 |

## 模块一修订实验

模块一以新100例／10来源选参，旧43例仅提议候选。结果详见 [模块一报告](module-one-results-20261031.zh.md)。七种组合搜索和LEACE不在本轮队列。

## 资源与解释

使用GPU 0、1、5、6；记录到的最大峰值显存为31.37GB。各阶段时长详见汇总JSON。初轮计时包含诊断，不据此声称部署速度优势。

没有独立祖先治疗历史时不伪造历史条件化对照。模型三值判断不作为医生金标。未覆盖的临床语义和额外训练方法保留为后续工作。

所有已有结果和失败记录保留；新来源数据仅用于验证，未用于参数选择。
