# CARE 模块一 V8.1

以 [PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12) 的 `8fb87d15576b88500f44a88f04799267d9f2b5ee` 为基础。实验仅在原服务器运行，使用原 Phi-4 + TrialMatchAI LoRA。

- A1：完全沿用 V8 的原始语义判断及训练示例。
- B1：在一次请求中检查完整条件、例外、否定、未知信息及相邻规则，直接返回最终 T/F/U。
- B2：A1 初判后增加一次完整条件复核，由模型返回最终 T/F/U；没有程序规则改写状态。

不加入 RAG、置信度或复杂规则门控，不重新训练。开发集仅使用原 96 例选择默认方案。随后比较历史 192 例、V8 的新 72 例，最后验证本轮新增的 72 例自然语言合成病例。原 A1 输出直接复用，也作为 B2 的第一轮，因此没有重复已完成的推理。

实验设计见 [protocol.json](protocol.json)，开发集选择见 `selection.json`，最终结果和结论见 `REPORT.zh.md`。`results/` 保留所有新增推理的逐例提示、原始输出及成本；原 A1 的提示和输出位于上一版实验目录。`analysis/` 保存完整逐规则比较、错误和成本分析。

原运行环境先执行 `source /home/gcx/gcx261014/bin/activate`。Phi 推理沿用 PR #12 的 `/home/gcx/n/.venv/bin/python`，因为原实验环境不含 bitsandbytes，使用既有专用环境可保持原模型加载和推理依赖一致。

所有病例都是合成分流情境，规则中的动作是实验定义。历史数据只用于回归比较；新增自然语言病例也不构成临床验证。

## 复现命令

在原服务器仓库根目录运行：

```bash
bash experiments/care_module1_v8_1/run_experiment.sh
```

入口脚本使用 GPU 0、1、2，每卡单个模型实例，batch 8、最多生成 384 tokens。先并行运行开发集 B1/B2，再执行 `analyze.py --select` 冻结选择；随后依次对 192 例和原新 72 例并行运行 B1/B2；最后并行运行自然语言 72 例 A1/B1，完成后用 A1 输出运行 B2。分析结果统一写入本目录的 `analysis/`。运行日志写入各结果目录的 `shell.log`，阶段事件写入 `events.jsonl`。

历史 A1 不重复推理，分析直接读取 `../care_module1_semantic_v8/results/{dev,confirmation,new72}_A1/predictions.jsonl`。B2 的 `--first-pass-cache` 读取对应 A1 的原始输出，复核提示保留原始病例和全部规则。自然语言集使用本轮产生的 `results/natural72_A1/predictions.jsonl`。完整推理成本包含复用的首轮；`added_*` 字段只计新增调用、tokens 和耗时。

单独运行一臂示例：

```bash
source /home/gcx/gcx261014/bin/activate
CUDA_VISIBLE_DEVICES=1 /home/gcx/n/.venv/bin/python \
  experiments/care_module1_v8_1/runner.py \
  --arm B2 \
  --cases experiments/care_module1_semantic_v8/data/semantic_dev_inputs.jsonl \
  --first-pass-cache experiments/care_module1_semantic_v8/results/dev_A1/predictions.jsonl \
  --output experiments/care_module1_v8_1/results/dev_B2
```

重复执行同一入口会跳过已有病例，并从 `stages.jsonl` 恢复完成的阶段；不要同时启动两个进程写入同一结果目录。若需要从头复算，应先将现有 `results/` 与 `selection.json` 归档，再运行入口。历史 A1 缓存仍需保留，保证复核和对照使用同一首轮结果。

完整原始输出以 `results/*/predictions.jsonl` 为准。`stages.jsonl` 是运行中的重复检查点，留在服务器用于断点续跑，不重复提交；运行日志会打包保存在 `artifacts/`。报告和逐例分析均由服务器上的实际输出生成。
