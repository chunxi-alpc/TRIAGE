# 实验 1–2 与参考结果复算

全部计算在服务器 gcx@10.29.128.115 的 gcx261014 环境执行。输入是真实 Qwen3-8B repair-v2 分数、注意力头表示和干预缓存；没有合成数据，没有新增大模型前向。岭回归、神经网络训练及子空间重估属于新增真实训练。

完整重跑：

~~~bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-cr-20261108
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp12/audit_reference.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp12/run.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp12/replay.py
~~~

可设置 TRACE_EXP12_OUTPUT=/绝对路径/新目录，将 run.py 结果写入新目录；replay.py 使用同一个变量。共享特征缓存由上级 shared.py 管理，原始缓存不修改。

主要文件：

- config.json：数据规模、种子、维度、标准化及指标配置。
- reference_audit.json：由原始逐对干预记录与旧无截距回归预测复算 51 头、16 维、1,194 对、13.91%、0.4063、0.0786。
- reference_reproduction.json：从真实头表示重新拟合旧回归，比较旧支持空间与本次重估结果。
- metrics.csv、summary.csv：每次运行结果和按种子汇总。
- pair_predictions.csv.gz：每个方法、每个种子的全部 fit/dev 配对目标和预测。行以 method、seed、split、pair_id 标识。
- bootstrap_summary.csv、bootstrap_draws.npz：2,000 次指南来源成组自助重采样。主方法与随机对照的差值使用各随机种子 R² 的均值，未将 50 个预测平均成集成模型。
- selected_support_r*.npz、fixedS16_support_r*.npz、*_model.npz：子空间和回归参数。
- nn_tuning.json、nn_inner_support.npz、mlp16_seed*.pt、*_scaler.npz：内部选择记录与最终三次神经网络训练参数。
- prediction_comparison.png/.pdf：开发 A 配对预测比较，误差条为种子标准差。
- replay_check.json：已保存参数重放与已保存逐对开发预测的实际一致性验证。

预测定义是原始分数差 ell(right)−ell(left)。fit A/P 共 4,887 对；主要开发指标为 1,194 个 A 配对，A/P 共 1,198 对另列。岭回归固定 alpha=0.01、无截距，不对投影后的特征另行标准化。原始头坐标的均值、标准差仅由 fit 的 full 视图估计；S 来自 fit C/F，U 来自 fit A/P，N 为 U 去除 S 方向后正交化所得。随机子空间和随机头使用种子 11–60，随机头从全部 1,152 个头均匀抽取 51 个，不额外排除选中头。

Q2 的主要维度比较为 fixedS16_N4、fixedS16_N8、selected_N16、fixedS16_N32：固定 S16 和原始坐标标准化，仅改变 U/N 维度。selected_N4/8/32 是同时改变 S/N 维度的次要历史兼容比较，不能解释为纯痕迹维度效应。有效秩均已记录。

神经网络为 16→32→1，ReLU，577 个可训练参数，输入为配对痕迹差。超参数及训练轮数只用 fit 内按来源分出的训练/验证来源选择：内部训练来源单独重估坐标标准化和 S/N 子空间。注意力头保持全 fit 冻结，因此内部选择是条件于已有选头的比较。最终三个种子 11/22/33 均在全 fit 配对上重训；没有用 dev/test 调参。该差分网络是预测信息诊断，不能直接当作单输入分数修正函数，实验 4 的可部署网络另行训练。

R² = 1−Σ(target−prediction)²/Σ(target−mean(target))²；RMSE 为预测残差均方根。13.91% 是 RMS(直接痕迹修补效应)/RMS(原始总分数差)。0.0786 将直接修补效应本身视作预测量计算 R²，没有重新拟合“修补效应→总分数差”的回归。旧 paired_fit 的 APF0 回归另含 82 个 F0 配对，其 A R² 为约 0.4093，不应与 A/P-only 的 0.4063 混用。

nn_refit.py 与 supplement_capacity.py 用于在已完成对照结果上只重算神经网络或补充固定 S16 维度比较；普通完整复现只需 run.py。运行成本分别保存在 config.json、nn_refit.json、capacity_supplement.json；ridge 的 fit_seconds 仅计系数拟合，随机特征的 feature_seconds 另计标准化、子空间和投影开销。所有任务为 CPU 运算，GPU/显存占用为零。
