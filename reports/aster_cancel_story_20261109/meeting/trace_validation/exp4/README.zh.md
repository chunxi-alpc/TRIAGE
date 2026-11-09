# 实验4：真实缓存上的分数修正

只使用真实Qwen3-8B缓存，不使用合成数据。输入根目录默认为 `/home/gcx/aster_cr_repair_v2_run_20261108`。所有方法只在原拟合集训练，开发集与测试集用于冻结后的评价。测试集此前已被旧ASTER-CR方案评价使用，不属于全新独立测试。

## 运行与重放

在服务器仓库根目录激活 `/home/gcx/gcx261014/bin/activate` 后运行：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/aster_cancel_story_20261109/meeting/trace_validation/exp4/run_repair.py
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp4/replay_saved.py
```

默认写入本目录；可用 `--out` 指定另一目录重跑。大投影缓存默认写入仓库外的 `/home/gcx/aster_trace_validation_cache_20261109`。通过 `--root`、`--cache` 可指定具有相同原始缓存的其他服务器路径。

重放入口只加载保存参数，重算全部开发/测试逐节点、逐对及路径节点分数，与保存CSV比较，不训练、不进行新的语言模型推理。`replay_check.json` 记录检查结果，并补存拟合集逐对预测。

## 预先固定的配置

- 沿用拟合来源确定的51个注意力头、拟合标准化及16维正交干扰子空间。
- 监督为拟合集A/P配对的右侧分数减左侧分数；F0不纳入干扰训练。
- 线性修正最小化AP分数差残差平方和，加 `0.01 * ||w||²`，配对回归无截距。全局分数偏置在修正模型固定后，只使用拟合集自然节点标签校准。
- 证据预算固定为0.05、0.1、0.25，不根据开发/测试结果选择最佳预算。每个拟合C/F配对的修正差绝对值不得超过 `预算 * 拟合集AP原始差RMS`。
- 随机对照使用种子11的6528→16高斯矩阵QR正交投影，所选注意力头空间、拟合标准化、监督和证据预算均相同。
- 神经网络为16→32(tanh)→1，577个网络参数，另有1个校准偏置，使用种子11、22、33。只通过拟合集内部来源划分选择学习率、惩罚和轮数，再在完整拟合集重训。候选学习率0.003/0.01，惩罚10/100，最多500轮。
- 神经网络使用证据超额惩罚，再按拟合证据最大修正差统一缩放，保证与线性方法相同的拟合最大差预算。开发/测试不重新缩放，超额数量单独报告。
- 神经网络内部选择沿用预先冻结的完整拟合集子空间；内部验证属于冻结子空间条件下的函数选择，没有重新选头或估计子空间。

## 指标与解释

VR为 `1-var(修正后AP分数差)/var(原始AP分数差)`，采用总体方差。ER为 `mean(标签方向×修正后CF分数差)/mean(标签方向×原始CF分数差)`。证据RMS为C/F配对两侧修正量之差的均方根，单位为判断分数logit。方向退化统计原始标签方向响应为正、修正后不为正的配对。

自然节点正确率采用 `variant=full, original=True`。动态路径从指南首节点开始，按修正分数正负选择分支，执行到实际叶子，要求节点序列、判断与叶子都正确。路径使用已有各可达节点的真实原模型特征缓存，不以静态节点正确率替代。

预算只保证拟合集逐对约束。相同预算不等于开发/测试实际证据RMS相同。神经网络在留出来源上明显超过训练证据预算，较高VR应连同实际RMS评价。

## 产物

- `config_and_metrics.json`：配置、来源、覆盖、归属检查、训练信息和运行开销。
- `linear_models.npz`、`nn_*.pt`：冻结参数。
- `neural_training_selection.json`：拟合集内部来源划分、候选比较、最终选择。
- `main_results.csv`、`summary.csv`、`path_summary.csv`：主指标、类别明细与动态路径汇总。
- `pair_predictions.csv.gz`、`fit_pair_predictions.csv.gz`：开发/测试和拟合集逐对预测。
- `node_predictions.csv.gz`、`path_node_predictions.csv.gz`：逐完整节点和逐可达路径节点分数。
- `path_results.csv.gz`：逐路径实际执行结果。
- `repair_tradeoff.png/pdf`：证据修正RMS与干扰方差降低率的权衡。
- `replay_check.json`：保存参数重放检查。
- `run.log`、`replay.log`：详细日志。

大CSV采用gzip无损压缩，可用Python gzip或pandas读取；重放入口兼容普通和压缩CSV。原模型与阈值校准具有相同配对差，因而VR、ER、证据RMS相同，但节点及动态路径正确率可能不同。
