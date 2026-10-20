# A1 + 针对 U 的对照示例：dev96

仅比较原96例开发集，A1直接使用已完成缓存。新示例是两个T/F/U对照组，共6段新编训练叙述：OR与未知例外。系统指令、目标输入、输出字段顺序及生成参数均沿用原A1。

在原服务器仓库根目录运行：

```bash
source /home/gcx/gcx261014/bin/activate
CUDA_VISIBLE_DEVICES=0 /home/gcx/n/.venv/bin/python experiments/care_module1_u_examples_dev/runner.py \
  --arm A1_U \
  --cases experiments/care_module1_semantic_v8/data/semantic_dev_inputs.jsonl \
  --output experiments/care_module1_u_examples_dev/results/dev_A1_U
python experiments/care_module1_u_examples_dev/analyze.py
```

完成的病例会跳过。需要真正重新推理时使用新的输出目录，不能覆盖现有原始结果。训练示例构建器、冻结记录、输入长度比较、模型调用记录和分析代码均保存于本目录。

本轮没有安排历史192例或任何72例扩展实验。
