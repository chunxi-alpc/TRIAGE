# 官方基线的下一步实际入口

核查：2026-10-25 至 26 日。以下是可交协调器执行的命令说明；本代理未运行实验。每条命令由协调器设置单卡 `CUDA_VISIBLE_DEVICES`，从 pinned SEKA 仓库根目录运行。`MODEL` 是实际 Qwen3-8B，因此全部结果仍属 checkpoint 适配。

## PASTA 可行版本

最快可行的是 SEKA 作者附带的 Qwen PASTA 加合成相关性选头：

```bash
python -m pastalib.profiler.synthetic_qa_pasta_profile_builder \
  --model "$MODEL" --data data/synthetic/pair_qa_new.jsonl \
  --layers all --max_samples 100 --min_diff 0.10 --top_pct 0.90 \
  --output_dir "$OUT/pasta_heads_source100"
```

输出 `<model_basename>_head_config.json`，其头索引是按 GQA 重复到 query 头的配置。实际源文件 100 行/200 triplet，最多 600 次前向。按协调器 full-source SEKA 的 62.67 秒实测，额外考虑全层、32 个 query 头的矩阵分解与缓存，估计 1–5 分钟、31–40 GiB；这是估计，需记录实际值。

评价入口：

```bash
python -m benchmarks.eval_fact_gen \
  --model "$MODEL" --data_path "$CF_DIR" --output_dir "$OUT/pasta_smoke16" \
  --example_subset 0:16 --benchmarks efficacy --add_unmediated_fact True \
  --batch_size 1 --max_new_tokens 64 --pasta \
  --head_config "$OUT/pasta_heads_source100/${MODEL##*/}_head_config.json" \
  --pasta_alpha 0.01 --scale_position exclude
```

应标记 `SEKA-author Qwen PASTA adaptation + hybrid head selection`，不是原 PASTA 逐头性能搜索。`alpha=1` 是同路径零强度；同一 attention backend 也要跑无 hook。若 eager/SDPA 自动切换，要记录并比较匹配 backend。

原始头搜索的公开脚本 `pastalib/profiler/synthetic_profiler.py` 在 main 内硬编码任务路径、1000 行、k=500；`profile_counterfact.sh` 又依赖 pinned 仓库中不存在的 `pastalib/config/train/<model>`。不能直接声称原头搜索已一键可复现。其可调用函数 `profile_heads` 会逐层逐 query 头评估；Qwen3-8B 若 36 层×32 头，100 行×2 提示意味着 230400 次序列生成，成本与上述 600 次校准前向不同。后续若跑该版本，需先改为实际训练路径、登记 k/alpha/token预算，并用单头实际吞吐估计完整成本，不用混合选头冒充它。

## AdaSEKA 四个真实专家

真实来源是 synthetic、CounterFact、BiasBios、HotpotQA，见论文附录 B 和 `src/custom_builders/adaptive/` 四个构建器。不是 README 例子里的 pronchange。各专家独立目录，真实 SVD 文件名含 `0.1mindiff`。

已存在的 Git 合成文件只能提供 100 行；其 adaptive builder 仅使用每行 context_1，每个输入正/中性各一次前向，最多 200 次。它不包含两个方向的 200 个 adaptive 样本，不能混淆计数。

```bash
python -m src.custom_builders.adaptive.synthetic_qa_builder_adaptive \
  --model "$MODEL" --data data/synthetic/pair_qa_new.jsonl \
  --output_dir "$OUT/adaptive/synthetic_source100" \
  --max_samples 100 --min_diff 0.1 --top_pct 0.9 --layers last10 --save-svd --svd-only

python -m src.custom_builders.adaptive.counterfact_qa_builder \
  --model "$MODEL" --data "$CF_DIR" --example_subset 0:200 --benchmark_name efficacy \
  --output_dir "$OUT/adaptive/counterfact_train200" \
  --max_samples 200 --min_diff 0.1 --top_pct 0.9 --layers last10 --save-svd --svd-only

python -m src.custom_builders.adaptive.biasbios_qa_builder \
  --model "$MODEL" --data "$BIOS_JSON" --example_subset 0:200 \
  --output_dir "$OUT/adaptive/biasbios_train200" \
  --max_samples 200 --min_diff 0.1 --top_pct 0.9 --layers last10 --save-svd --svd-only

python -m src.custom_builders.adaptive.hotpot_qa_builder \
  --model "$MODEL" --data "$HOTPOT_TRAIN_JSON" \
  --output_dir "$OUT/adaptive/hotpotqa_train200" \
  --max_samples 200 --min_diff 0.1 --top_pct 0.9 --layers last10 --save-svd --svd-only
```

四个已存在文件再生成 flat JSON（下面仅示意路径结构，文件存在前不得把该配置标 ready）：

```json
{
  "synthetic": "/actual/output/adaptive/synthetic_source100/Qwen3-8B_0.1mindiff_pos_svd.pt",
  "counterfact": "/actual/output/adaptive/counterfact_train200/Qwen3-8B_0.1mindiff_pos_svd.pt",
  "biasbios": "/actual/output/adaptive/biasbios_train200/Qwen3-8B_0.1mindiff_pos_svd.pt",
  "hotpotqa": "/actual/output/adaptive/hotpotqa_train200/Qwen3-8B_0.1mindiff_pos_svd.pt"
}
```

官方 adaptive builder 会生成完整注意力矩阵，检查正提示对答案 span 的注意力是否达到中性提示的 1.2 倍；筛选前后数和实际前向数必须单列。整头 NaN 是被筛头标记，部分 NaN/Inf 则是异常。无任何有效样本时 `torch.cat` 可能失败，保留该失败，不擅自取消筛选后仍叫原实现。短合成/CounterFact 样本预计数分钟；HotpotQA 长上下文可能需要更多显存，必须先根据训练样本 token 长度估计，不能默默截断。

当前 HF 原数据在服务器不可访问，完整标准 split 和 BiasBios/HotpotQA 专家依赖尚未满足。只构建 synthetic 或 synthetic+CounterFact 可验证加载/路由，但须标 `single-expert smoke` 或 `reduced-bank smoke`，不能称完成四专家 AdaSEKA。论文描述固定 seed42 随机抽200实例，而发布构建器默认直接取前 N 行；该差异必须记录，后续取样清单应独立冻结。

评价的真实选项是 `--adaptive-seka --adaptive-expert-path CONFIG --adaptive_amplify_factor 3.0 --top_k_singular 5 --combination_method weighted_top_k --layers last10`，CounterFact 其余设置与 SEKA 相同。先 batch_size=1，官方额外预计算前向和路由成本单独计入。
