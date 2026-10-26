# CAD / Spotlight 最小基线适配

状态：`planned`。文件已写入，但本代理没有在本地或服务器运行测试、模型或实验。由协调器在服务器完成合成测试及 16 例小试后更新状态；运行器不自行标记 `complete_verified`。

## 身份和固定参数

- CAD：按照作者仓库 `xhan77/context-aware-decoding@1281d7fcd0e7e49d786c78d3170a3048dd9f6dc2` 的公式 `(1 + alpha) * context_logits - alpha * prior_logits` 编写 Qwen 适配；不是作者原程序直接运行。固定 `alpha=1`，关闭候选筛选。上下文路径使用 SEKA 官方完整 CounterFact 提示，先验路径只保留同一原问题，不读答案来改写问题。
- Spotlight：按照 [EACL 2026 论文](https://aclanthology.org/2026.eacl-long.174/) 编写 `paper_reimplementation`。未确认作者完整官方代码。每个层、每个 query、每个头分别计算选中证据的注意力总量；低于固定目标 `.1` 时加 `log(target/current)`。只给可见证据 key 加偏置。这里没有用强行达到目标质量的 log-odds 公式。
- 默认种子 0；只接受预先约定的 0/1/2。上述参数在读取新标准测试结果前确定，入口不提供效果搜索。
- 目标环境为 transformers 4.51.3 的 Qwen3 eager。服务器 Qwen3-8B 是检查点适配，不是论文的 Qwen3-8B-Base。torch 2.8 环境差异由总清单披露。

## 标准指标与输入

`run_counterfact_baselines.py` 直接导入固定 SEKA 源码的 `load_dataset` 和 `efficacy`，保留大小写三个目标首 token 取最大概率、`Previously ... Currently ...` 提示、旧事实、batch=1、chat=False。上下文 token 不插入或删除标记；原文字符区间映射为 token mask。

官方 `benchmarks/counterfact/evaluate.py:408–409` 的 `counterfact_efficacy` 强制 `max_new_tokens=1`，paraphrase 也转入该 wrapper。这里标准入口仅实现 efficacy，每条每路也生成 **1 个 token**，没有额外生成 64 token。Spotlight 沿用 Hugging Face generation scores。CAD 两路均调用同配置的单 token generate，读取 raw logits；先检验每路 raw logits 与 generation scores 完全相同再融合。若模型启用了会改变分数的 logits processor，明确报错，保留失败记录；没有默默把处理后的 logits 冒充原 CAD 公式。CAD 的融合 token 是融合分数的 argmax，不是两路各自生成的 token 之一的投票。

官方 CounterFact 的属性跨度和替代事实本来就是任务提供的输入；这里只用于该标准基线。不能把这一标准 benchmark 的属性输入解释为临床任务可以使用答案或真实适用规则选择跨度。

## 控制与记录

- CAD：`no_hook` 只执行上下文一路；`matched_zero` 依旧执行两路，alpha=0；`normal` alpha=1；`reverse` alpha=-1。两路前向成本逐例计入。
- Spotlight：`no_hook` 使用原 Qwen eager；`matched_zero` 进入同一注意力适配路径但 gain=0；`normal` gain=1；`reverse` gain=-1；`wrong_span` 选确定性的前若干非证据 token，和正确证据等 token 数且不重叠。所有比较复用相同原提示。
- Spotlight 沿用 transformers 的 Q/K/V 投影、RoPE、cache、GQA repeat、因果 mask、输出投影；只临时注册注意力函数，离开上下文时恢复。逐 query/head 的计算使用 float32 logsumexp，避免小概率先求和再取对数的下溢。zero 与 no_hook 的实际数值等价必须由服务器测试及真实小试验证。
- 每例保存预测 token、文本、三种目标 token id、目标/旧答案 log-prob、正确与否、提示/跨度哈希、源码/数据/模型指纹、实际顶层 forward hook 计数、耗时、显存峰值。`matched_zero` 记录完整词表首步分数的最大绝对漂移和 argmax 是否改变。逐例保留，不能删除错误跨度无法构造等失败病例。
- 运行目录不得覆盖；中途异常保留已有预测和失败清单。`execution_status=finished` 只表示程序退出完成，独立审计以后才可把 `task_state` 改成 `complete_verified`。按树/患者分组的统计由总评价程序负责；此标准 CounterFact 入口把官方 case id 保留供配对统计。
- 这不是临床路径或多选解码器；CAD 这里只提供标准单 token 接口，临床原生生成需要独立的双 cache 实现与验证。

## 服务器命令

仅协调器分配 GPU 后执行；这里没有启动服务器任务。`MODEL_FINGERPRINT` 为总模型清单 SHA256，不是仅目录名。

```bash
cd "$ADAPTER_ROOT"
python -m unittest -v test_baseline_operators

CUDA_VISIBLE_DEVICES=0 python run_counterfact_baselines.py \
  --seka-root "$SEKA_ROOT" --model "$MODEL" \
  --model-fingerprint "$MODEL_FINGERPRINT" \
  --data-path "$COUNTERFACT_JSONL" --output-dir "$RUN_ROOT/cad_smoke16" \
  --method cad --subset 0:16 --seed 0

CUDA_VISIBLE_DEVICES=0 python run_counterfact_baselines.py \
  --seka-root "$SEKA_ROOT" --model "$MODEL" \
  --model-fingerprint "$MODEL_FINGERPRINT" \
  --data-path "$COUNTERFACT_JSONL" --output-dir "$RUN_ROOT/spotlight_smoke16" \
  --method spotlight --subset 0:16 --seed 0
```

默认 CAD 4 组控制共 7 次前向/例，16 例共 112 次；Spotlight 5 组共 5 次前向/例，16 例共 80 次。合成测试还包含一个随机小 Qwen 模型的 CPU 测试，验证真实 Qwen3 GQA/RoPE/cache 进入适配、零强度一致、恢复注册表；它不读取研究测试数据，也不验证效果。生产参数仍固定 `.1`，合成测试另用 `.5` 只是保证随机小模型出现低于阈值的分支。
