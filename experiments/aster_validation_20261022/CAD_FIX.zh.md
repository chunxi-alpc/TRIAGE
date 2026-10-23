# CAD 不输出选项的原因与修正

服务器真实诊断发现：未经候选过滤的全词表 CAD 在第一步可将上下文概率极低、闭卷概率更低的 IF 排到选项字母之前。例如 medguide_05282 A 中，普通上下文首选 B；IF 的 log p(context)=-9.526、log p(closed)=-21.340，其上下文概率只有首选词的0.00848%，但2 log p(context)-log p(closed)使其胜出。随后进入复述规则的生成路径。延长到128 token不会改变这个第一步选择。

这解释了部分格式失败，不等同于证明全部失败来自同一原因。提示两路除是否提供规则外也有任务前缀差异；本轮修正保持提示不变，单独检验解码修正，不把提示效应混入同一对照。

新增条件：

1. **CAD128 候选过滤，自由生成**：每一步先排除 p(context,token)<0.1*max p(context,token) 的候选，再在余下词元上取 CAD 最高分；保留128-token上限和模型EOS配置。beta=0.1在失败诊断后固定，未以准确率调参；这不是独立验证过的最佳超参数。不使用金标选择词元。过滤只改变生成候选，保存的首位置选项分数仍为原始CAD分数。
2. **合法选项约束对照**：仅允许题目合法字母，取保存的首位置分数最大值并结束。普通baseline/prune接受完全相同约束。该计算可以精确复用已保存logits，不伪称新增GPU推理，也不称为自由生成准确率。

真实失败样本验证：3个Llama病例的9个CAD A/B/C输出中，原CAD128有2/9可解析，过滤后9/9可解析；正确数从0/9到4/9。18个prune_cad条件可解析数仍为16/18。样本专门从失败中选取，不能作为总体效果估计。明细见 results/run500/cad_diagnosis。

完整补跑：两模型×500题×9个CAD/prune_cad条件×3种选项顺序（原顺序、left、right），每模型13,500条过滤版输出。cad_fix_worker.sh 等待原 worker500.sh 完成后使用同一GPU；不停止原实验和重排。所有原CAD8、未过滤CAD128和新过滤版独立保留。analyze_cad_fix.py 配对比较准确率、解析失败、长度截断及动作一致性，并报告配对统计。合法选项约束结果单独报告。报告完成判据已新增cad_fix状态，不能在其排队时声称所有实验完成。

复现：`CUDA_VISIBLE_DEVICES=0 bash experiments/aster_validation_20261022/cad_fix_worker.sh llama /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct 86135`；Qwen在GPU1，模型路径为 /home/gcx/TRIAGE/models/modelscope/Qwen3-8B，前序PID86139。首次运行的实际PID保存在服务器 private/run500/cad_fix_pids。诊断、生成及私有日志均留服务器。
