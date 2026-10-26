# 临床原生路径注意力基线入口

状态：`planned`。源码已准备，尚未运行临床任务。协调器服务器首轮 4 个 tiny 合同测试中 3 个通过，PASTA 作者完整 hook 与分块比较失败，日志已保留。定位到初稿用普通半开重叠转换 token 范围，而作者 `find_token_range` 使用闭区间字符边界，恰好在边界时会包含前置 token；已改为直接调用作者转换并按已选原文位置计算 occurrence，待服务器复测，没有提高容差。本地未执行任何实验或测试。这里只读取已经冻结的 MEGA `train_val` 16 例输入与 primary 的 A0 条件输出，不读取测试题或参考路径。

`run_attention_baselines.py` 复用 primary A0 的逐规则 T/F/U、相同患者原文、完整指南、官方问题和输出模板，重建提示并核对 primary A0B1 的完整提示哈希。所有新方法使用相同 BF16 模型、YaRN2、65,536 上下文和 primary 冻结的生成预算，生成完整原生路径。SEKA 执行阶段对应已有 A0B1，直接复用逐例输出，新增前向为零。

PASTA 使用 SEKA 作者仓库 `aab9cfed80dc012fb9e83f8bdefb8675c33f40e7` 内的真实 Qwen 移植和未改动的 `PASTA.edit_multisection_attention`。固定 `alpha=0.01, scale_position=exclude`，逐段保留预测证据区间，完整 key 维度与原提示长度不变，仅按 128 个 query 分块调度。所有 token 的因果关系、缓存、QKV、RoPE 与作者多段 mask 加法保留。应标为“SEKA 作者 Qwen PASTA 移植，另有 query 分块执行适配”，不能称原始 PASTA 原生支持 Qwen，也不能把当前有限校准头配置称为论文完整头搜索。`pasta_no_hook` 使用相同分块计算但不改 mask；`pasta_zero` 真正调用作者算子，alpha=1。没有可用证据时明确回退原输入，不伪造跨度。

PASTA 与 SEKA 共享相同预测证据原文区间，但 PASTA 保留作者原始 `find_token_range` 的字符到 token 边界规则，可能在恰好重合的字符边界包含前一个 token。实际 token 范围与 mask 哈希逐例记录；不宣称两个作者算子的 token mask 必然完全相等。

AdaSEKA 使用真实 `AdaptiveSEKALLM` 和现有 `StageSEKAAdapter(kind=adaseka)`。固定 last10、top_k_singular=5、weighted_top_k、temperature=1，四个真实来源专家，分别运行预先指定的强度 0.5 和 3。路由参数沿用 G1 已冻结预设；在任何临床运行前将初稿默认 3 改为 G1 的 5，不是观察临床效果后的调参。单次路由使用作者原始 hidden states、查询、系数与投影运算。仅在 `output_hidden_states=True,use_cache=False` 的路由预计算中传入 `logits_to_keep=1`，避免生成不被路由读取的全位置词表 logits；全部 hidden states 保留。BF16 与 primary 匹配，明确区别于 G1 作者默认 FP32 标准实验。nohook 不路由；zero 仍执行路由及零增益投影。

入口要求协调器完成的 `clinical_attention_artifact_provenance.json`。按已存在 schema 核对实际模型所有文件、原项目 `sha256(json.dumps(mapping,sort_keys=True))` 指纹、config/YaRN、PASTA 构建 manifest、四专家独立审计与实际投影文件。旧 32k 配置的专家不能凭相同张量形状通过。源码、运行时配置和构建材料哈希均写入新 protocol。

每例保留原生输出、token 数、首 token 分数漂移、输入 token 哈希、成功/失败前向计数、路由前向计数、耗时和峰值显存。数值对照核对真实输入哈希，不写死“相同输入”。模型调用失败仍保留该例该方法错误行和已尝试成本。解析/外部判分未在此入口执行，完成生成只标记 `verification_pending`，不自动宣称 `complete_verified`。共享 A0 条件与指南抽取成本引用 primary 原始记录；头/专家构建成本引用实际构建审计。

服务器合同测试（不能在本地执行；SEKA checkout 必须为上述 pin）：

```bash
PYTHONPATH="$SEKA_ROOT:$RUNTIME/method_agent:$RUNTIME/clinical_attention_baselines" \
  "$PYTHON" -m unittest discover -s "$RUNTIME/clinical_attention_baselines" -p 'test_author_attention.py' -v
```

四个测试分别验证：真实 tiny Qwen 的原作者完整 PASTA mask 与分块/缓存生成一致；相同 kernel 下 nohook 与真实作者零强度算子一致；真实 Ada 类加载四个合成专家、BF16、零强度与额外路由前向；tiny BF16 Qwen 的全部 hidden states 逐位相等及最后 token logits 的实测误差。它们不证明完整 55k 长度可运行或临床有效。

真实开发入口（变量由协调器指向实际服务器文件，输出必须为新目录）：

```bash
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" "$PYTHON" "$RUNTIME/clinical_attention_baselines/run_attention_baselines.py" \
  --dataset-root "$MEGA_DATA" --input-manifest "$FROZEN_INPUTS" --official-root "$MEGA_SOURCE" \
  --primary-run "$PRIMARY_RUN" --seka-root "$SEKA_ROOT" --model "$YARN_MODEL" \
  --model-fingerprint "$MODEL_MAPPING_SHA256" --pasta-heads "$YARN_PASTA_HEADS" \
  --adaseka-experts "$YARN_EXPERT_CONFIG" --artifact-provenance "$ARTIFACT_PROVENANCE" \
  --methods seka_execution_reuse pasta_no_hook pasta_zero pasta adaseka_no_hook adaseka_zero adaseka_05 adaseka_3 \
  --output-dir "$NEW_OUTPUT_DIR"
```

可分开运行 PASTA 和 Ada 方法，避免一组任务影响另一组；各自仍执行并记录本组 nohook 数值参照。所有真实结果和失败都保留，不根据开发结果改变本文件固定强度。
