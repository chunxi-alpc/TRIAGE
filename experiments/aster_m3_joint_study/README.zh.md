# ASTER 第三模块联合监督与决策校准

复用 E1 的五折 51 头和 16 维 M2 参数，在缓存上训练线性 S/J。所有实验在服务器 `gcx@10.29.128.115` 的 `gcx261014` 环境执行；无需 GPU。

在仓库根目录执行：

```bash
source /home/gcx/gcx261014/bin/activate
bash experiments/aster_m3_joint_study/run_server.sh
```

`run_server.sh` 先运行合成测试，再完成 5 折实验和报告。默认历史路径均在启动时实际检查。路径迁移时可直接使用模块入口：

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m experiments.aster_m3_joint_study.run preflight \
  --e1-root /actual/path/to/reports/aster_cr_retrain_paired_v2_20261111 \
  --cases-file /actual/path/to/data/path_cases.jsonl \
  --diagnostic-root /actual/path/to/reports/aster_cr_diagnostics_e3cv_20261111 \
  --path-map /old/cache/root=/actual/cache/root
# 用完全相同参数将 preflight 改为 run。
python -m experiments.aster_m3_joint_study.run report
```

输出默认位于 `reports/aster_m3_joint_study_20261115/`。`status.json` 记录完成或阻塞状态；缺失缓存写入 `missing_assets.json`；协议冲突直接停止。首轮入口不启动新的头提取、激活修补或 J2。

首轮完成后的可达性诊断和独立J2探索使用单独入口。本轮诊断已发现原证据约束下不可纠正的首错，实际J2配置及结果保存在 `j2/`：

```bash
python -m unittest experiments.aster_m3_joint_study.test_training \
  experiments.aster_m3_joint_study.test_evaluation \
  experiments.aster_m3_joint_study.test_bounds \
  experiments.aster_m3_joint_study.test_j2 -v
python -m experiments.aster_m3_joint_study.diagnose reports/aster_m3_joint_study_20261115
python -m experiments.aster_m3_joint_study.j2_run reports/aster_m3_joint_study_20261115
python -m experiments.aster_m3_joint_study.report reports/aster_m3_joint_study_20261115
python -m experiments.aster_m3_joint_study.package_results reports/aster_m3_joint_study_20261115
python -m experiments.aster_m3_joint_study.package_results reports/aster_m3_joint_study_20261115/j2
```

J2沿用各折J的AP权重，只在原内层来源划分选择CF方向损失权重0.1、1、10，共20次线性拟合。训练CF有向修正界为 `-δ ≤ (y_right-y_left)*Δc ≤ 2δ`；原绝对δ保证随之改变。J2的全部比较均属后续探索。打包入口无损压缩逐例输出，服务器保留原始文件。

## 固定协议

- 训练：Dataset A / E1 的 2,540 对，80 来源的原五折64/16划分；仅当前折训练来源的自然节点提供任务标签和截距校准。
- 评价：沿用现有共同 Dataset B 配对池；自然路径和三个干扰路径使用原动态评分器。E1训练数据和共同机制评价池分别标明。
- 每外折在64个训练来源内按固定种子分为48/16：S一个候选，J的三个候选为0.1、1、10；每折选择一个J后，S/J各在64来源重拟合，总计30次线性拟合。内部选择条件于已用64来源拟合的M1/M2；外层16来源不参与训练或选参。
- 自然任务损失为均值BCE，除以同一训练集原分数加固定b0的BCE；A/P类别与来源均衡权重和为1，平方损失除以训练原始配对差平方均值与1的较大值。L2项固定为 `0.01*sum(w*w)`。各项尺度及实际值均保存。
- 新中心为自然训练痕迹的均值；S/J相同。原E1截距转为 `b_old+mu_train@w` 以复现已存独立校准分数；共享b0的R2是另存的中心化诊断。
- J的选择先要求相对内层R1自然路径下降不超过0.5个百分点，再按干扰路径、证据方向退化、超预算率、CF RMS和较小AP权重排序。若没有候选满足自然要求，明确记录并按自然正确率优先选一个候选完成诊断。
- 首要比较为独立截距J−S，随后J−R1。每步同时满足干扰优势与自然非劣；其余比较为探索性。使用原2000次来源配对重采样，不平均折级指标。

## 文件

`training.py`：归一化目标、梯度、约束求解、截距与选择规则。`data.py`：缓存读取、折内来源检查、旧分数与路径复现。`evaluation.py`：原评价器适配、逐来源指标、四向错误转移、首错margin与区间。`report.py`：中文报告和结果图。`test_training.py` / `test_evaluation.py`：合成测试；真实归档复现由运行入口执行并写入 `verification.json`。

所有历史资产只读。历史测试来源、官方MEGA和跨任务结果均不参与本轮训练或新选参。80来源已在历史研究中使用，本轮结果属于既有来源复评，正式确认需冻结后新来源。
