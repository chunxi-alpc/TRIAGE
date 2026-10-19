# A0/A1 语义推理实现

A0 与 A1 均直接对患者的一组完整自然语言规则判断 T/F/U。一个病例的全部 3 条规则及相邻分支进入同一个提示；模型不生成动作位置，规则 ID 用于下游映射原始动作。没有检索、患者事实抽取、自动三值聚合、置信度或证据门控。

使用与 PR11 修复原子接口相同的 Phi-4 与 TrialMatchAI LoRA：

- 本体：`/home/gcx/models/care_module1_v5/phi-4`（来源 `microsoft/phi-4`）。
- 适配器：`/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora`（来源 `majdabd33/trialmatchai-phi4-reasoning-lora`）。
- 环境：`/home/gcx/n/.venv/bin/python`。
- 调用仓库原有 `TrialMatchAIAdapter.from_pretrained`，保持 NF4 4bit、double quant、bfloat16 compute；未重复计算权重哈希。
- 两版都固定 batch size 8、greedy、num_beams 1、每病例最多 384 个新 token，先填入相同的 JSON 开始字符 `{"rules":[`。所有规则一次生成，非逐规则运行。

A1 增加固定的 3 个训练演示。第一个演示是一对仅改变年龄的 T/F 病例，第二个展示否定与例外，第三个展示信息不足 U；因此共 3 个演示、4 个患者文本。来源仅为 `mechanism_01_conjunction` 和 `mechanism_03_negation` 两个 train 家族，均不在 dev 或 confirmation 家族中。完整示例及训练病例 ID 保存在 `data/fewshot_examples.json`，在读取 dev 结果之前冻结。A0 不含任何示例。

历史规则只有结构化指南原文，所以预处理把指南条件和 AND/OR/NOT/例外语义展开为完整自然语言规则。此过程只处理 G，不读取患者事实或标签；模型不接收原子 ID、规则树、数值比较元数据。新病例使用作者写好的整条自然规则文本。所有模型输入始终采用正向字段白名单：问题、患者原文、规则 ID 与完整自然规则。训练示例的参考输出属于固定 few-shot 示范，待预测 dev/confirmation/new72 的金标与结构化患者事实从不进入预测输入。

JSON 解析只检查完整规则列表、ID、T/F/U 及两个字符串字段。第一次失败允许一次简短格式重试，保留原提示和原输出；没有参考答案反馈。第二次仍失败则状态为 `parse_failure`，各规则 state 为 null，不转写成 U。证据 quote 在患者文本中精确定位，不匹配或空 quote 都不会修改或阻止 T/F/U。

每例输出为 `predictions.jsonl` 的一行，保留 case_id、arm、status、rules、原始提示和输出、生成 token 数、重试、调用次数、实际生成时间。时间为实际 GPU batch wall time 均摊到病例，manifest 另外记录总墙钟时间、模型加载时间、显存峰值和所有版本。最大调用预算为每病例 1 次主请求与最多 1 次重试，即最多 768 个输出 token；batch 中每个病例仍是独立请求。日志保存在相应结果目录，不覆盖历史产物。

## 重现

所有命令在服务器独立 worktree 根目录执行。首次运行先生成无标签历史输入和冻结示例：

```bash
/home/gcx/n/.venv/bin/python experiments/care_module1_semantic_v8/prepare_historical.py
CUDA_VISIBLE_DEVICES=0 /home/gcx/n/.venv/bin/python experiments/care_module1_semantic_v8/semantic_runner.py --cases experiments/care_module1_semantic_v8/data/semantic_dev_inputs.jsonl --arm A0 --output experiments/care_module1_semantic_v8/results/dev_A0
CUDA_VISIBLE_DEVICES=1 /home/gcx/n/.venv/bin/python experiments/care_module1_semantic_v8/semantic_runner.py --cases experiments/care_module1_semantic_v8/data/semantic_dev_inputs.jsonl --arm A1 --output experiments/care_module1_semantic_v8/results/dev_A1
```

开发集固定默认版本后，将输入切换为 `data/semantic_confirmation_inputs.jsonl` 与新 72 例文件，输出到新的对应结果目录；两版本均运行完整集合。运行器遇到已有结果会按 case_id 跳过，不重复推理或改写旧记录。
