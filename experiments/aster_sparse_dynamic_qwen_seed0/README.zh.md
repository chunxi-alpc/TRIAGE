# Qwen3-8B seed=0 稀疏动态注意力复现

完整结果见 [REPORT.zh.md](REPORT.zh.md)。本目录基于 PR #33 和 #34，保留其全部历史记录。

## 服务器完整前向

运行环境：115服务器，`/home/gcx/gcx261014`；GPU 0、1、6。默认模型是 `/home/gcx/TRIAGE/models/modelscope/Qwen3-8B`，默认原数据与缓存位于 `/home/gcx/aster_dynamic_review_20261025`。私有病例不在仓库中。

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-sparse-dynamic-qwen
python -m unittest discover -s experiments/aster_sparse_dynamic_qwen_seed0 -p 'test_*.py' -v
bash experiments/aster_sparse_dynamic_qwen_seed0/launch.sh
python experiments/aster_sparse_dynamic_qwen_seed0/analyze.py
python experiments/aster_sparse_dynamic_qwen_seed0/render_report.py
```

默认输出根目录已经含本次结果，脚本遇到已有分片目录会报错以保留原始记录。新的复现实验应使用一个新输出目录，并启动三条命令：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/aster_sparse_dynamic_qwen_seed0/run_gpu.py --shard 0 --out /home/gcx/aster_sparse_repeat
CUDA_VISIBLE_DEVICES=1 python experiments/aster_sparse_dynamic_qwen_seed0/run_gpu.py --shard 1 --out /home/gcx/aster_sparse_repeat
CUDA_VISIBLE_DEVICES=6 python experiments/aster_sparse_dynamic_qwen_seed0/run_gpu.py --shard 6 --out /home/gcx/aster_sparse_repeat
python experiments/aster_sparse_dynamic_qwen_seed0/analyze.py --out /home/gcx/aster_sparse_repeat
```

三条前向命令可分别放入后台并行执行。按排序后的患者轮流分片，四卷始终同行；无完整重跑旧800条基线。首次核对所有病例输入、缓存条件键、动作支持和映射，失败会明确退出。现有源码不支持自动跳过未完成行；中断记录需要保留，并在新目录重试。

## 无病例、无GPU的匿名结果复算

CPU环境需要 Python、numpy、torch、transformers。使用当前环境版本见结果中的 manifest。不需要模型权重或私有病例：

```bash
python -m unittest discover -s experiments/aster_sparse_dynamic_qwen_seed0 -p 'test_*.py' -v
python experiments/aster_sparse_dynamic_qwen_seed0/analyze.py \
  --numeric experiments/aster_sparse_dynamic_qwen_seed0/numeric/predictions.jsonl \
  --metadata experiments/aster_sparse_dynamic_qwen_seed0/results.json \
  --destination /tmp/aster_sparse_reproduced
```

此命令只复算公开匿名数值；不是新的GPU实验。原始结果含每种方法全部27组候选及其800条预测，不只保存胜出配置。配对统计以整棵开发树为组，包含同患者四卷。CPU复算输出应与仓库结果完全一致。

## 实验配置

- 主配置：层18/22/26/30，头0/8/16/24，动态强度1，组偏置上界2。
- 结构消融：层28/30/32/34，或头0/16。强度消融为0.5。
- 每个结构都有相同计算路径的零强度前向。
- 主配置另有动作支持循环错位、反向、患者固定符号翻转、同位置静态偏置；这些都是实际GPU前向。
- 融合网格与旧实验相同，每种方法27组。主配置不通过比较各结构消融后挑选。
- 零强度残差、去掉残差和数值漂移对照分别报告，不混同。

只使用已有开发树。没有新增独立数据；已查看P3不能重新充当独立验证。新分支不覆盖main，不删除旧负结果。
