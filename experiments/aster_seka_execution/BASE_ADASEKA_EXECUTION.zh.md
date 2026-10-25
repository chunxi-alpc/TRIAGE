# Qwen3-8B-Base AdaSEKA 标准评测准备

状态：`planned`。本文件和命令生成器只做执行准备，本代理没有运行本地测试、SSH 或 GPU。协调器已排队构建 Base 四专家；构建和独立审计完成不代表下面的推理已完成。既有 Qwen3-8B instruct 输出继续保留，不能更名为 Base 结果。

## 已核对的官方入口

代码固定 `waylonli/SEKA@aab9cfed80dc012fb9e83f8bdefb8675c33f40e7`。CounterFact 使用 [eval_fact_gen.py 第60–84行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/eval_fact_gen.py#L60)，BiasBios 使用 [eval_bias_gen.py 第54–77行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/eval_bias_gen.py#L54)。两者实际选项相同：

```text
--adaptive-seka --adaptive-expert-path /actual/expert_config.json
--adaptive_amplify_factor 3.0或0.5或0.0
--top_k_singular 5 --combination_method weighted_top_k --layers last10
--batch_size 1
```

CounterFact 固定强度3，BiasBios固定0.5，零强度均为0。沿用 G1 的 top_k=5，作者类 temperature 默认1；CLI没有 temperature 选项，不添加不支持的参数。两入口都没有给 Ada 模型加载传 `torch_dtype`，此前实际为 FP32，必须由新运行的 `first_forward_parameter_dtype` 再确认。normal/zero 均使用同一个官方 Ada 类及默认精度，不需要 dtype wrapper。专家加载也转换成模型实际精度。

官方 router 每次生成前额外调用一次模型，只读取 batch0 的最后位置来计算整个 batch 的投影，因此必须 batch1。不能拿 batch16 吞吐除以16冒充逐例路由成本。标准 Base 任务不使用临床 YaRN 模型、BF16 adapter、路由 logits 存储改写或临床 query 分块；这里保留原作者路径。

四专家必须来自同一 Base 原始配置：`runtime/adaseka_base_synthetic100`、`adaseka_base_counterfact200`、`adaseka_base_biasbios200`、`adaseka_base_hotpot200`。独立审计完成后产生 `runtime/adaseka_base_audit/expert_config.json`。生成器读取真实审计的专家哈希和构建 model_path，不用临床 YaRN 或旧 instruct bank 替代。

## 分阶段任务和固定规模

| 阶段 | 方法 | 固定slice | 样本数 | 新token上限 | 每臂模型调用 |
|---|---|---|---:|---:|---:|
| 可运行性与成本 | Bio Ada0.5、Ada0、FP32普通参考 | 4500:4516 | 各16 | 64 | Ada各最多1040；普通最多1024 |
| 完整标准结果 | CounterFact Ada3，efficacy+paraphrase | 5000:10000 | 5000原始case | 每提示1 | 按相同数据的实际改写数预计30000 |
| 完整同精度对照 | CounterFact Ada0、FP32普通参考，efficacy | 5000:10000 | 各5000 | 1 | Ada0为10000；普通5000 |
| 完整标准结果 | Bio Ada0.5、Ada0 | 5000:10000 | 各5000 | 64 | Ada各最多325000 |

小试不替代任何完整5000例。Bio小试只使用既定验证区间的前16行；它的职业候选集合可能少于28类，不能拿其准确率与论文正式结果相比。完整 Bio 普通 FP32 参考没有无依据地加入当前长队列；当前只安排16例以核对零强度的同精度差异。若要声称全5000例完全无数值漂移，仍需要相应证据，16例不能证明全量。

BiasBios64-token上限来自官方README；其 top1/top3 实际按首个生成步的职业大小写变体首token概率取最大值，候选标签由当前slice确定；后续生成用于fluency/consistency并产生真实成本。不能改为1token后仍称完整官方生成协议。保持官方TF-IDF全处理文件拟合、slice参考文本和相同预处理。详见现有 `work/data_audit/SEKA_BIASBIOS_PROTOCOL.zh.md`。官方输出为 `result.json`。

当前服务器 SciPy 缺少旧 `csr_matrix.A`，固定作者 [BiasBios scorer](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py) 仍调用它。因此所有 Bio normal/zero 使用既有 `biasbios_compat.py`，临时将 A 恢复为 `toarray()` 别名并最终恢复类定义；不改作者源码、TF-IDF 数值或指标公式。FP32 nohook 在外层套 `dtype_load_compat`，其 upstream 为 `biasbios_compat`，再进入相同官方入口。每个新输出目录分别保留 `scipy_compat_report.json`，nohook 另保留 `dtype_load_report.json`。该适配已用于先前 instruct Bio 的实际运行，但本次 Base 任务仍须真实执行，不能沿用旧结果称已完成。

CounterFact `efficacy` 官方 wrapper 强制只生成1token，对目标事实和旧事实的大小写变体首token分数比较，输出 `efficacy.json` 和 `efficacy_metrics.json`，不是生成文本精确匹配。`paraphrase` 先按case用set去重提示，再调用相同efficacy wrapper并按原case聚合。队列固定 `PYTHONHASHSEED=0` 保留顺序；原case是统计分组单位。full 默认 Ada3 同一进程明确传 `--benchmarks efficacy paraphrase`，必须完成ES/PS两个论文指标，不将PS降为可选项，也不重复启动独立normal PS。zero和FP32普通参考均跑完整5000例efficacy，提供同精度配对控制。

## 资源和调度依据

新队列拟用 GPU1 单张80GB A100，必须同时等待已有 Base 队列425321及 parser-v2 临床注意力父队列427918退出。已停止的v1父队列不能作为新任务的完成依赖；其余临床融合和判分仍按协调器既有安排，不抢占。生成器只写JSON，不启动任务；协调器在正式启动前继续向用户报告资源和预计时长。

实际执行器位于实验根目录 `p/run_official_job.py`，不在产物目录 `p/runtime/`。生成器显式接受并检查 `--runner`，`--runtime` 继续指数据/产物目录；所有JSON命令保存实际解释器、执行器、source、model、GPU、输出目录和全部官方参数，不依赖混淆的目录约定。

8B FP32权重约30GiB，另需KV、路由全部hidden states、全位置logits、专家和临时张量；规划先预留40–60GiB，但这不是Base Ada Bio实测峰值。长传记与官方未裁剪路由logits会影响峰值，16例不能证明全split最高显存。

目前没有 Ada Bio batch1 实测耗时。已知SEKA Bio batch16的5000例20032前向/1141.58秒/18.01GiB属于不同精度、不同路由与批大小，**不用于推算 Ada Bio**。先以16例实际成功前向、EOS长度、总耗时和峰值估计完整成本；保留上限每臂325000次调用。生成器可接收成功的 positive/zero smoke manifest，给出 `smoke_elapsed*5000/16` 的粗略排期值；该值包含模型加载和哈希固定开销，且验证16例长度不代表全量，只用于资源安排。不能把没有测得的小时数写成实测结论。

## 最少复现命令

以下须从固定官方仓库根目录、通过既有服务器环境执行。`MODEL` 必须为真实 Qwen3-8B-Base 原始配置，`EXPERTS` 为完成审计的上述flat JSON。正式执行优先使用生成器输出的 `run_official_job.py` 命令，以保存完整指纹、输入哈希、实际前向数和显存。

```bash
"$PYTHON" -m biasbios_compat --report "$NEW_BIOS_SMOKE_DIR/scipy_compat_report.json" -- \
  --model "$MODEL" --data_path "$BIOS_JSON" \
  --output_dir "$NEW_BIOS_SMOKE_DIR" --example_subset 4500:4516 --batch_size 1 --max_new_tokens 64 \
  --adaptive-seka --adaptive-expert-path "$EXPERTS" --adaptive_amplify_factor .5 \
  --top_k_singular 5 --combination_method weighted_top_k --layers last10

"$PYTHON" -m benchmarks.eval_fact_gen --model "$MODEL" --data_path "$CF_DIR" \
  --output_dir "$NEW_CF_DIR" --example_subset 5000:10000 --batch_size 1 --max_new_tokens 1 \
  --benchmarks efficacy paraphrase --add_unmediated_fact True --adaptive-seka --adaptive-expert-path "$EXPERTS" \
  --adaptive_amplify_factor 3 --top_k_singular 5 --combination_method weighted_top_k --layers last10
```

Bio完整任务把slice换为5000:10000，其他预算/参数不变；零强度把相应gain改0并使用新目录，CF零强度明确只传 `--benchmarks efficacy`。FP32普通参考去掉Ada参数，保留相同slice/输入/批大小/输出预算；Bio 的包装顺序为 `dtype_load_compat --torch-dtype float32 --upstream-module biasbios_compat --report NEW_DTYPE_REPORT -- --report NEW_SCIPY_REPORT -- <官方参数>`。CounterFact 的 dtype wrapper upstream 仍为 `benchmarks.eval_fact_gen`，同样只传efficacy。dtype wrapper 只覆写模型加载dtype，不更改注意力或生成算子，并记录实际参数dtype。无需给普通参考加literal marker；实际输入是否与Ada去marker后一致由已保存forward trace核对。执行包装模块时，其所在目录须在现有 PYTHONPATH 中。

生成可审查队列（不启动实验）：

```bash
"$PYTHON" "$PREP/prepare_base_adaseka_queue.py" --phase smoke \
  --python "$PYTHON" --runner "$EXPERIMENT_ROOT/run_official_job.py" --runtime "$RUNTIME" \
  --source "$SEKA_ROOT" --model "$BASE_MODEL" \
  --counterfact-dir "$CF_DIR" --biasbios "$BIOS_JSON" \
  --expert-config "$RUNTIME/adaseka_base_audit/expert_config.json" --expert-audit "$BASE_EXPERT_AUDIT" \
  --output-root "$RUNTIME" --gpu 1 --wait-for-pids 425321 427918 --plan "$NEW_SMOKE_QUEUE_JSON"
```

完成小试后改 `--phase full`，新plan文件，并可传入 `--smoke-positive-manifest`、`--smoke-zero-manifest` 做资源估计。新的实际后继PID由协调器提供；生成器不自行启动full，也不通过小试结果挑强度。队列使用现有互斥GPU锁、逐任务新日志、失败继续保留下一个独立任务。

## 完成仍需的证据

1. Base四专家实际构建退出及独立audit；100合成行来源限制、其他源200请求的筛选/NaN/回退头数和实际前向。
2. 16例同FP32小试与zero/nohook输入和分数差异；随后两任务完整5000例真实输出及所有失败，不能删失败或重复姓名行。
3. 逐例概率/生成、实际前向/耗时/显存、源码/数据/模型/投影指纹及独立导出和按原case/传记来源分组置信区间。官方保存字段只能支持其已有分数检查，不能据此声称完整词表logits逐位一致。
4. full 的 CounterFact efficacy/paraphrase 与 Bio 全部仍为 `planned`，直到真正运行与独立核验。环境仍是已记录的独立PyTorch2.8/Transformers4.51.3，区别于作者requirements的PyTorch2.7。
5. 即使Base完整输出完成，发布合成文件仅100行、官方builder取首N而论文描述随机200，仍须披露为发布代码下的校准限制；不能用“Base检查点一致”掩盖校准和环境差异。

这些标准实验不证明临床原生路径或多选反事实的独立注意力增益，后者仍依赖已经排队的匹配信息/预算临床实验。
