# 独立复核：准备检查与真实模型结果

14 组准备检查全部通过，两个 PyTorch 适配器检查实际运行；这不等于 GPU 实验完成。准备记录及静态划分复核见 [preparation_static_review.json](preparation_static_review.json)，原日志保留在服务器 `/home/gcx/care_reliability_runs/preparation_work_01`。统一入口未传模型路径导致内部联合预检为 blocked；补充原模型路径后，初始化前预检通过。

## 已完成的模块二真实实验

最终六任务为服务器 `/home/gcx/care_reliability_runs/module2_work_02`，覆盖旧 384 例和新机制 144 例、共同基线加 11 项干预。6,336 条记录、6,905 次模型前向、588 次实际零控制均完成验收。模型权重与原记录一致，配置与原环境核验通过，训练投影和均值方向复现误差均为 0。最终旧集 4,608 条预测和候选分数与修复前成功运行完全一致。

| 数据及主要方法 | 成对正确 | 成对准确率 |
|---|---:|---:|
| 旧集基线 | 72 / 192 | 37.50% |
| 旧集直接偏置 | 77 / 192 | 40.10% |
| 旧集校准 B4 | 71 / 192 | 36.98% |
| 旧集校准均值 | 74 / 192 | 38.54% |
| 新机制主分析：全部 12 条件 | 0 / 24 | 0% |

旧集校准 B4 减直接偏置为 −3.13 个百分点，97.5% 家族配对区间为 −5.21 至 −1.04；校准均值的差为 −1.56，区间为 −3.65 至 +1.04。没有显示学习方向相对直接偏置的额外收益。新集主差和经验区间均为 0，是所有方法都未完成整对判断造成的，不能称为等效。

新机制全部 144 例中，校准 B4 有 128 例无操作（88.9%），校准均值有 54 例无操作（37.5%）；无操作记录保留在分母。保留的校准编辑均满足当前层、最后提示查询上的实际符号检查，但这不保证答案改善。

最终机器可读验收见 [旧集](module2_final_legacy_verification.json)和[新集](module2_final_sealed_verification.json)。统计由独立的成对计数与固定 10,000 次家族抽样重算，全部日志另经正式完整性验收器检查。

## 首次失败及修复范围

首次新集运行在实际分数均方根预算误差 15.4404% 时停止，原文件未覆盖。固定查询、键与方向的独立 CPU 重算发现合法比例点误差为 13.3465%，证实 BF16 非单调数值路径使单一路径二分漏点。最终改为仅在二分超限时进行固定 4,097 点数值搜索；没有使用答案，也未更换方向、种子、样本或放宽 15% 门槛。六任务以统一源码重跑，数值回退最终仅触发两条新集记录。详见 [数值失败独立证据](module2_scalar_failure_independent_review.json)；原资产在服务器 `/home/gcx/care_reliability_runs/module2_work_01/numerical_failure_diagnostic`。

## 结构缓存与联合结果状态

精确请求缓存只使用患者文本、条件文本和条件类型，与真实 TrialMatchAI 提示一致；V6/V7 只共享相同请求的生成输出。输入白名单为 guideline、patient、question，参考状态、结构化事实和答案不进入预测器。不可变来源记录在恢复写入之前核对，连续两次恢复拒绝且旧记录不变的测试已通过。见 [结构缓存静态复核](structure_cache_static_review.json)。

模块一与联合正式运行现已完成，覆盖全部 192 例。直接汇总已有正式指标见[联合简报](joint_final_summary.zh.md)及[机器可读摘要](joint_final_summary.json)。768 次实际零控制通过；主要 24 对中参考结构正确 15 对，V6/V7 均为 0 对。具体门控问题和 U 来源见[阈值审阅](threshold_review.zh.md)，不把保守映射、截断和解析拒绝混称为模型语义未知。

## 解释边界与复算

旧集为参考适用性和动作位置给定的模拟病例回归。新集是虚构路由规则及模拟观察，只有 8 个测试家族；措辞曾参与解析器开发修复，因此不属于全程盲法的独立确证，更不能称为临床验证。家族区间条件于旧训练折，不含重新拟合不确定性。自由生成、KV 缓存和证据预算仍未验证。

`recheck_module2.py` 不加载模型或使用 GPU。可在服务器原环境内，从仓库根目录对解压后的完整日志执行：

```bash
python experiments/care_reliability_20261019/checks/work_verification/recheck_module2.py \
  --run-root /home/gcx/care_reliability_runs/module2_work_02/legacy384_regression \
  --compare-with /home/gcx/care_reliability_runs/module2_work_01/legacy384_regression \
  --output /tmp/care_legacy_independent_recheck_new.json
```

新集将 `--run-root` 改为 `module2_work_02/sealed_mechanism` 并去掉 `--compare-with`。输出文件必须尚不存在。

