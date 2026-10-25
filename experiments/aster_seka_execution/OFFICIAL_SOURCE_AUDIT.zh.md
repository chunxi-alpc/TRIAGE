# 官方基线源码核查

日期：2026-10-25。本文是源码和公开来源审计；本代理没有启动推理或测试。服务器执行结果由协调器另行登记。

## SEKA / AdaSEKA

- 固定官方代码 `waylonli/SEKA@aab9cfed80dc012fb9e83f8bdefb8675c33f40e7`，Apache-2.0。投影构建器和 `data/synthetic/pair_qa_new.jsonl` 均真实存在。缺少预制投影不能视为无法复现。
- 官方实验底座是 **Qwen3-8B-Base**。协调器核查的现有模型为 Qwen3-8B，故本机首轮必须标记 `adapted_checkpoint`，不能把与论文分数差异当成同检查点复现误差。
- [论文附录 F / 表 7](https://arxiv.org/html/2603.01281v1#A6) 的 Qwen3-8B-Base 配置：CounterFact `top_pct=.85,min_diff=.12,g_pos=2.4,g_neg=0`；BiasBios `.998,.12,.6,.3`；AdaSEKA `min_diff=.1`，CounterFact `g=3`、BiasBios `g=.5`。官方开发范围 4500:5000，测试范围 5000:10000，贪心生成。论文报告 SEKA 的 CounterFact ES/PS 为 99.08/98.96，BiasBios accuracy 为 88.74；这些仅为发表结果。
- 真实 [CounterFact 入口](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/eval_fact_gen.py) 写出 `efficacy.json` / `efficacy_metrics.json` 和相应 paraphrase 文件。README 关于统一 `metric_result.json` 的叙述不能替代检查实际产物。
- [CounterFact 评价器](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/counterfact/evaluate.py) 比较首个生成词元上目标事实与旧事实的概率；各取 lower/capitalize/upper 三种形式的最大概率。ES 不是文本精确匹配准确率。paraphrase 按原始 case 聚合多个提示，不能把多个提示当独立患者。

### 必须明确保留的上游实现细节

1. [seka_llm.py](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/src/model/seka_llm.py#L148) 读取投影 `layers` 后没有用它映射层号，而按所选层的顺序取张量。`builder --layers all` 与推理 `--layers last10` 会错配。无需更改算子的首跑方案是构建和推理都设 `last10`；使用已建全层投影时应另存明确切片产物并登记衍生 hash。
2. 同文件第 215 行附近：提供负投影时计算 `(g_pos*positive+g_neg*negative)/2`；即使 `g_neg=0` 也除以 2。带 `--neg` 与不带该选项不是同一强度。标准命令应保留该参数，与论文/上游示例一致。
3. [builder](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/src/model/projection_builder_base.py) 默认 `from_pretrained` 未设 dtype，通常加载 FP32。协调器服务器实查 pinned Git 仓库内合成文件只有 **100 行 pairs，即 200 组 triplet**；每组 3 次提取，完整源文件最多 600 次前向。即使传 `--max_samples 200` 也被截到 100 行，输出名称与清单必须按实际数写。跳过答案 span 未对齐的 triplet 时必须记录实际次数。首轮 16 行只是 smoke。
4. Qwen 键提取和注入在 `k_norm` 后、RoPE 前，使用 KV 头。调度必须只暴露分配的 GPU；SEKA 模型包装器在看到多卡时会自动跨全部可见卡分配。
5. [adaptive_seka_llm.py](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/src/model/adaptive_seka_llm.py#L388) 的官方路由取批内第 1 个样本的最后词元，整批共用路由；额外预计算前向没有传 padding mask，并取 `hidden_states[layer+1]` 再送当前层的 layernorm/q_proj。先用 batch_size=1 的无 padding 官方运行，另外把批处理修正标为适配；不能声称原代码逐例路由已验证。
6. pinned 仓库没有 README 所述 `scripts/` 和 `adaptive-seka-config/`。真实 AdaSEKA 输入是平坦 JSON `{expert_name: path_to_svd_pt}`；SVD 文件需要 `layers,U_matrices,singular_values`。不同专家必须对应实际不同校准来源，不可把同一合成文件重复构建后称为多任务专家。
7. adaptive SVD 构建器有异常时退回单位矩阵的路径。应保留日志并核对是否发生回退，不能把无声回退当成功谱分解。该构建器以整头 `U` 和 `S` 全 NaN 标记筛掉的头，这属于官方约定；应统计跳过头数，区分部分 NaN/Inf 数值错误。synthetic adaptive builder 每行仅取 context_1，故现有 100 行最多 100 个正/中性对、200 次前向，不能称完成论文 200 实例专家。

### 环境

官方 requirements 记录 torch 2.7.0、transformers 4.51.3、datasets 3.5.1，但同时锁定很旧的 numpy/scipy/sklearn；不宜整套覆盖现有实验环境。协调器提供的现有环境为 torch 2.8.0+cu128 / transformers 4.57.6，计划在独立环境固定 transformers 4.51.3 与 tokenizers 0.21.4，其他差异必须记录。

[Transformers 4.51.3 setup.py](https://github.com/huggingface/transformers/blob/v4.51.3/setup.py) 要求 Python>=3.9、tokenizers>=0.21,<0.22。[tokenizers 0.21.4](https://pypi.org/project/tokenizers/0.21.4/) 有 cp39-abi3 Linux wheel，可安装于 CPython 3.12。[Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py) 的 `k_norm` 输出形状与官方 SEKA 钩子相符。

最少导入依赖：builder 需要 torch/transformers/numpy/sklearn/matplotlib/tqdm；CounterFact 增加 datasets/dataclasses_json/scipy/nltk/anchoring（anchor 即使关闭也被顶层导入）；BiasBios 增加 spacy。仓库自带 pastalib，避免已安装 PyPI 包覆盖其 Qwen 移植。原生 CounterFact efficacy/paraphrase 不要求 spaCy 模型或 NLTK 文本生成评价资源。

## 标准数据实际路径

官方 Hugging Face 数据仓库是 `waylonli/SEKA-datasets`，已从[提交页面](https://huggingface.co/datasets/waylonli/SEKA-datasets/commit/91bf3f67fb2044c7e81514e103a962d6ef76c230)确认当前 revision `91bf3f67fb2044c7e81514e103a962d6ef76c230`。[实际目录](https://huggingface.co/datasets/waylonli/SEKA-datasets/tree/main/pasta_bench) 与 README 目录示例不同：

- `pasta_bench/counterfact.jsonl`，43.4 MB，efficacy/paraphrase 所需的处理后数据；其父目录可直接作为 `--data_path`。
- `pasta_bench/BIOS.pkl`，450 MB，须经转换后形成 `biasbios.json`。不能声称 HF 已提供处理后的 JSON。
- `attribute_snippets.json`、`idf.npy`、`tfidf_vocab.json` 仅为 generation 扩展所需，首轮不必下载。

[BiasBios 转换器](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/reformat_dataset.py) 默认先取原始前 50000 条，再用 spaCy `en_core_web_sm` 和固定过滤转换；评估 5000:10000 是处理后的行号。其 `json.dump` 循环漏写换行，输出连续 JSON 对象。可仅加 `handle.write('\n')` 修复文件格式，登记修复 diff 和处理后文件 hash，保留全部既有筛选逻辑。原始 BIOS 使用许可由数据代理确认，不重分发原文。

## PASTA

上游 [QingruZhang/PASTA](https://github.com/QingruZhang/PASTA/tree/56a9138d2d071c9ab2860c72be596da9723ca793) 当前固定为 `56a9138d2d071c9ab2860c72be596da9723ca793`，MIT。原作者 README 支持 LLaMA/LLaMA2/GPT-J。SEKA 自带的 [pastalib/pasta.py](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/pastalib/pasta.py) 明确加入 Qwen3 和 Gemma3，应命名“SEKA 作者提供的 PASTA Qwen 适配”，不是原 PASTA checkpoint 复现。

现有 Qwen3-8B-Base 头文件不能直接沿用到 Qwen3-8B。可使用同仓库 `pastalib/profiler/synthetic_qa_pasta_profile_builder.py` 在合成校准上重建 query-head 配置；其源码按 GQA 分组重复 KV 键，不等于原 PASTA 逐头下游性能搜索，需单独标明头选择策略。推理 `alpha=1` 是同路径零强度；注意力实现应固定并进行与无 hook 的数值比较。

## CAD

官方 [xhan77/context-aware-decoding](https://github.com/xhan77/context-aware-decoding/tree/1281d7fcd0e7e49d786c78d3170a3048dd9f6dc2) 固定 `1281d7fcd0e7e49d786c78d3170a3048dd9f6dc2`。正式 `group_decode_fileio.py` 用同一模型的完整上下文/无上下文两路，权重 `(1+alpha,-alpha)`，两路候选过滤默认 `filter_top_p=filter_top_p_prior=1.0`。Qwen 可以实现同一公式的两缓存路径，但需标为适配并核对原代码，不把旧候选过滤版本称官方结果。

`demo.ipynb` 用 `len(context_ids)` 截取上下文，而该变量形状是 `[batch,seq]`，所以它取的是批大小，不是上下文词元数。不能照搬这个演示。pinned 根目录未见 LICENSE 文件；引用源码来源，避免把其源码复制到 TRIAGE 后自行赋予其他许可证。

## Spotlight

已检查 [EACL 2026 正式论文](https://aclanthology.org/2026.eacl-long.174.pdf) 及作者信息，尚未确认作者发布的完整代码。公开检索出现第三方 Hugging Face 实现，不应冒称官方。

论文方法对每个 query、每层、每个头求高亮集合 attention mass；低于目标时加 `log(target/current)`，重新 softmax。它不是强制精确达到目标的 log-odds 校正，也不是 PR #35 的动作组差距偏置。后续可按论文复现，明确命名 `paper_reimplementation`，并保留计算路径零强度及不同 span 的对照。

## 当前可执行结论

官方 SEKA 源码与合成投影构建所需文件可获取，可推进实际 Qwen3-8B 投影及标准评价适配。精确论文复现还需 Qwen3-8B-Base 权重、标准数据、匹配参数与环境记录。上述源码审计不能替代真实模型输出；完整标准评测、强临床基线及独立注意力贡献均尚需服务器执行证据。

协调器随后回报的服务器实测：16 行 builder 完成 96 次前向、38.06 秒、30.56 GiB；完整 Git 合成文件 100 行完成 600 次前向、62.67 秒、30.61 GiB，80 个所选 KV 头中 79 个非零、1 个零投影。CounterFact 首 16 例官方 efficacy 入口成功退出、16/16，仅证明小试可运行。这些运行由协调器负责原始日志与独立复核，不能用这里的转述代替运行清单。HF 标准全量数据仍不可访问，尚无完整四专家 AdaSEKA bank。
