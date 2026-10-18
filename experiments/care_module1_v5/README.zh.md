# CARE 模块一 V5

原实验和 24 条新合成探索样本已完成。解析器 V2 先将原 492 条输出提高到 423/492 可严格解析；随后只重生成其中 69 条失败条件，V3 达到 492/492 严格解析，原子条件准确率 431/492（87.60%），纯 TrialMatchAI 规则准确率 150/192（78.12%）。加入缓存 BioLinkBERT 证据后的规则准确率为 96/192（50.00%），历史 Qwen3 直接判定为 136/192（70.83%）。因此，TrialMatchAI 单独判定已超过旧 Qwen3，但当前组合方案仍不适合作为正式模块一。原实验结果保留在 [初始实验报告](REPORT.zh.md)，解析器 V2 见 [解析器 V2 报告](PARSER_V2_REPORT.zh.md)，失败条件重生成和 V3 复评见 [V3 报告](runs/v5_20261018/parser_v3_retry/RETRY_REPORT.zh.md)。

V5 在 [src/module1/care_v5.py](../../src/module1/care_v5.py) 实现。`from src import module1` 后，`module1.run(guideline, patient, question)` 使用 V5；首次调用时加载指定的 TrialMatchAI 适配器和 MedNLI 模型，不导入旧 Qwen 后端。可选地用 `module1.set_default_v5(module1.CAREModule1V5.from_pretrained(device="cuda:6"))` 预先初始化。默认 `module1.Module1` 类同样使用 V5；旧实现只能显式通过 `module1.run_legacy` 或 `module1.LegacyModule1` 使用。

TrialMatchAI 适配器是 Phi-4 上的 PEFT LoRA 生成模型，不是独立分类头。V5 按官方源码固定版本使用聊天模板及 JSON 输出格式，每次请求只含一个原子条件，批量大小为 8、生成上限 2048 token；保存原始生成。仅从目标条件的 Classification 读取标签，截断或解析失败记为 U。批量时间按每批实际请求数分摊，属于累计生成工作量，不是单请求延迟；多卡并行时也不是端到端墙钟。入组标签映射为：`Met→T`、`Not Met→F`、`Unclear/Irrelevant→U`。例外条件按排除条件映射：`Violated→T`、`Not Violated→U`。任何未知标签都映射为 U；适配器加载失败会停止，不会回退基础 Phi-4 或 Qwen3。BioLinkBERT 根据实际模型配置读取标签映射；蕴含/矛盾/中立分别保留为支持/反驳/未知。只有找到明确数值及单位时才运行确定性阈值判断；有多个未撤回数值时设为 U。

患者文本按句子拆分并保存 Python 字符位置 `[start,end)`。BioLinkBERT 只接收筛选出的候选句和单个条件。超过 512 token 的候选句会记为未知并跳过，不会静默截断；完整 TrialMatchAI 输入若超出上下文也会停止该样本。明确标记为撤回的旧记录不用于证据筛选；未撤回的相互矛盾报告继续作为冲突保留。

## 数据和限制

当前实验使用 V4 的 192 条探索性回归样本、16 条结构化规则和已保存的 Qwen3 直接判定。V4 数据参与了方法设计，故不能声称独立测试。数据包括 32 个原始合成病例和每个病例的控制变体；标签没有经过临床专家复核。V5 公共目录只保留标签、样本 ID、证据句编号/字符区间和模型分数；病例原文、完整模型生成和带原文引用的模块输出只写入服务器 `runs/<run_id>/private/`，不会提交。

另有 [24 条新规则合成探索样本](exploratory/README.zh.md)，使用此前未用的 4 条完整结构树规则，运行前冻结，具有明确动作位置；这些仍不是临床独立金标。

V4 提供的是研究用谓词改写，没有完整原始指南文本、治疗动作文本或其原文区间。`data/rules_v5.jsonl` 的条件区间指向规范化规则包中的离线条件，不冒充原始指南位置；动作为空并在输出中显式保留。要支持正式下游动作决策，仍需提供有来源许可的原始指南动作片段及独立验证标签。

评价分别保留纯 TrialMatchAI 标签与三值逻辑（方法 A）、增加确定性数值比较（A+数值）、以及在 A+数值上加入 BioLinkBERT-MedNLI（方法 B），另读取历史 Qwen3 直接判定（方法 C）。这样可以单独看出数值比较和证据模型各自带来的变化。

## 服务器复现

在服务器仓库工作树根目录执行。以下命令只应在服务器运行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
/home/gcx/.local/bin/uv pip install --python /home/gcx/n/.venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple peft bitsandbytes
bash experiments/care_module1_v5/scripts/download_models.sh

/home/gcx/n/.venv/bin/python experiments/care_module1_v5/scripts/validate_model_configs.py \
  --base /home/gcx/models/care_module1_v5/phi-4 \
  --adapter /home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora \
  --evidence /home/gcx/models/care_module1_v5/biolinkbert-mednli \
  --output experiments/care_module1_v5/data/model_validation.json

/home/gcx/n/.venv/bin/python experiments/care_module1_v5/scripts/prepare_public_data.py \
  --v4-labels /home/gcx/care_module1_20261018_v4/data/labels.jsonl \
  --v4-predictions /home/gcx/care_module1_20261018_v4/data/model/predictions.jsonl \
  --v4-pairs /home/gcx/care_module1_20261018_v4/data/pairs.jsonl \
  --pairs-out experiments/care_module1_v5/data/pairs_public.jsonl \
  --labels-out experiments/care_module1_v5/data/labels_public.jsonl \
  --qwen-reference-out experiments/care_module1_v5/data/v4_qwen_direct_public.jsonl

/home/gcx/n/.venv/bin/python experiments/care_module1_v5/scripts/prepare_rules.py \
  --manifest /home/gcx/care_module1_20261018_v4/data/rules_manifest.jsonl \
  --output experiments/care_module1_v5/data/rules_v5.jsonl

/home/gcx/n/.venv/bin/python -m pytest experiments/care_module1_v5/tests -q
bash experiments/care_module1_v5/scripts/run_all_v5.sh
```

不要把 `runs/*/private/` 或病例原文加入版本控制。模型加载、推理及输出都强制使用 CUDA；没有 CPU、基础 Phi-4 或 Qwen3 回退。结果报告由 `scripts/generate_report.py` 从 `run_status.json`、指标和错误文件生成。

## V4 对照

V4 保存的直接判定为 136/192（70.83%）、宏 F1 0.7041、已知样本证据准确率 87/137（63.50%）、联合正确 81/192、反事实成对正确 22/32、生成耗时 79.2 秒。V5 使用同一批样本进行探索性回归，同时保存 TrialMatchAI 独立判定和加入证据模型后的结果。

运行前先执行 `scripts/resume_phi4_modelscope.py`，可续传与 HF 摘要一致的 Phi-4 权重分片；适配器与 MedNLI 可用 `scripts/download_small_models.py` 下载，均逐文件校验。实际下载记录保存在公开 provenance 和服务器模型目录。完整单卡顺序复现可使用 `bash experiments/care_module1_v5/scripts/run_suite.sh`。本次实际运行根据用户要求由单卡转为四卡：GPU 6 已刷盘的 256/492 个条件原始输出归档保留，余下 236 个条件用 GPU 4/5/6/7 并行；原始完整 batch=8、批内顺序、提示、模型适配器和生成上限保持不变。最后只加载 BioLinkBERT，严格复用全部 492 条原始生成进行证据处理。


## 本次四卡续跑与证据重放

下列命令概要用于服务器已有冻结输入及已完成原始输出的情况。必须先确保原生成进程已经停止，旧结果已归档，且 `existing` 文件只含完整、已刷盘批次。脚本会检查输入和条件身份、原始批次边界及摘要，不读取标签；现有 worker 输出不允许覆盖。`WORK` 是服务器私有目录，含病历与模型原始生成，不提交仓库。四个 worker 独立加载指定 Phi-4 + TrialMatchAI 适配器。

```bash
V5PY=/home/gcx/n/.venv/bin/python
V5EXP=/home/gcx/TRIAGE-v5-pr4/experiments/care_module1_v5
V5WORK=/home/gcx/care_module1_v5_parallel_private
V5SERIAL="$V5EXP/runs/v5_serial_initial"
V5OUT="$V5EXP/runs/v5_parallel_replay"

"$V5PY" "$V5EXP/scripts/parallel_trialmatch.py" prepare \
  --inputs /home/gcx/care_module1_20261018_v4/data/private/inputs.jsonl \
  --rules "$V5EXP/data/rules_v5.jsonl" \
  --existing "$V5SERIAL/private/trialmatch_precomputed.jsonl" \
  --work-dir "$V5WORK" --batch-size 8 --world-size 4

for V5RANK in 0 1 2 3; do
  V5GPU=$((V5RANK + 4))
  "$V5PY" "$V5EXP/scripts/parallel_trialmatch.py" worker \
    --work-dir "$V5WORK" --rank "$V5RANK" --device "cuda:$V5GPU" \
    > "$V5WORK/worker_rank${V5RANK}.log" 2>&1 &
done
wait

"$V5PY" "$V5EXP/scripts/parallel_trialmatch.py" merge \
  --work-dir "$V5WORK" --output "$V5WORK/merged_private.jsonl"

V5SOURCE_ARGS=(--source-status "$V5SERIAL/run_status.json")
for V5RANK in 0 1 2 3; do
  V5SOURCE_ARGS+=(--source-status "$V5WORK/worker_rank${V5RANK}_status_private.json")
done
"$V5PY" "$V5EXP/scripts/replay_evidence.py" \
  --merged "$V5WORK/merged_private.jsonl" \
  "${V5SOURCE_ARGS[@]}" \
  --parallel-manifest "$V5WORK/timing_summary_private.json" \
  --device cuda:6 --expected-count 192 --output "$V5OUT"
```

合并命令须在所有 worker 完成后执行；任何失败、缺失、重复或身份不匹配都会停止合并。`replay_evidence.py` 不加载基础模型或适配器，不重新判断条件，不调用 Qwen3；它校验来源状态及合并文件摘要，并保留原始 TrialMatchAI 输出。新的 24 条合成集用同一模型接口单独运行和评价。

`run_status.json` 中 `replay_wall_seconds`/`elapsed_seconds` 仅表示当前证据重放阶段墙钟；`trialmatch_generation_gpu_seconds` 表示原先单卡与四卡生成按条件分摊的累计调用工作秒，含主机开销，并非 CUDA 内核时间。`parallel_workers_wallclock_seconds` 是最早 worker 进程开始到最晚结束的墙钟，含模型加载。四卡每张卡的加载、进程耗时和显存峰值见 `parallel_manifest.parallel_workers`。证据重放显存只含 BioLinkBERT，不能作为 Phi-4 + LoRA 显存。最初单卡状态另保留 `runs/v5_serial_initial/run_status.json`；该状态被中断，未记录的峰值不补估。

评价仍使用 `scripts/evaluate.py`。最终报告生成时可传入 `--initial-serial-status`、`--exploratory-metrics` 和 `--exploratory-status`，分别保留单卡、四卡、证据重放和新探索集的真实计时。报告表格中的方法耗时是累计生成工作秒加对应后处理墙钟，不是四卡端到端等待时间；Qwen3 的历史生成秒数来自不同的单卡规则级和批量协议，不能据此声称等价硬件提速。


## TrialMatchAI 解析器 V2 离线复评

修复后解析器从 JSON 对象中提取单个目标条件，不读取 `Final Decision` 代替条件标签；只接受目标数组中与请求条件严格相同且唯一的原文项和官方标签；仅允许空白重排，大小写和标点必须一致。代码围栏、附加说明、非目标数组和其 Irrelevant 空占位不参与目标判定。真正达到 token 上限、目标原文不匹配/不唯一、目标标签明确为 `Unclear` 分别记录为截断、目标不明确和医学未知。JSON 语法本身不修补；以前成功的标签须保持不变。

所有解析输出与 492 条原始生成 SHA256 核验后写入独立的 `runs/v5_20261018/parser_v2_final/`。解析和指标复算均不调用模型或 GPU；规则级 BioLinkBERT 组合结果重用初次实验保存的 NLI 关系及证据位置。解析器代码、配置和结果见 [PARSER_V2_REPORT.zh.md](PARSER_V2_REPORT.zh.md)。`condition_eval` 使用服务器 V4 源读观察值做金标条件评价；该文件只在解析器冻结之后由评价脚本读取，不是推理输入。

在服务器仓库根目录重现时，将输出目录设为尚不存在的新目录：

```bash
V5PY=/home/gcx/n/.venv/bin/python
V5RUN=experiments/care_module1_v5/runs/v5_20261018/parser_v2_recheck
"$V5PY" experiments/care_module1_v5/scripts/reparse_trialmatch.py --output "$V5RUN"
"$V5PY" experiments/care_module1_v5/scripts/evaluate_reparsed_conditions.py \
  --conditions "$V5RUN/conditions_public.jsonl" \
  --predictions "$V5RUN/predictions_public.jsonl" \
  --reference-facts /home/gcx/care_module1_20261018_v4/data/labels.jsonl \
  --reparse-freeze "$V5RUN/REPARSE_FREEZE.json" --output "$V5RUN/condition_eval"
"$V5PY" experiments/care_module1_v5/scripts/evaluate.py \
  --labels experiments/care_module1_v5/data/labels_public.jsonl \
  --pairs experiments/care_module1_v5/data/pairs_public.jsonl \
  --predictions "$V5RUN/predictions_public.jsonl" \
  --qwen-reference experiments/care_module1_v5/data/v4_qwen_direct_public.jsonl \
  --output "$V5RUN/metrics.json" --errors "$V5RUN/errors_public.jsonl"
"$V5PY" experiments/care_module1_v5/scripts/generate_reparse_report.py --run "$V5RUN"
```

## TrialMatchAI 失败条件定向重生成（V3）

解析器审计发现 69 条失败输出：29 条真正达到 2048 token 上限，40 条没有唯一匹配的目标条件原文。为提高解析成功率，使用同一套 Phi-4 + TrialMatchAI LoRA，只重生成这 69 条；已有 423 条成功输出保持原文哈希不变。生成提示要求只输出一个紧凑 JSON、逐字复制目标条件，最大生成长度提高到 4096 token。任务使用 GPU 4、5、6、7 四张 NVIDIA A100 80GB，批大小 8；每卡峰值约 13.2 GiB，四卡墙钟约 58 秒。

69/69 条重生成均通过严格目标条件和官方标签核验，重跑后 492/492 条解析成功；29 条截断和 40 条目标不明确全部清零。条件级准确率由 381/492（77.44%）升至 431/492（87.60%），宏 F1 由 0.7293 升至 0.8308。规则级纯 TrialMatchAI 由 60/192（31.25%）升至 150/192（78.12%），比历史 Qwen3 高 7.29 个百分点；反事实成对正确为 21/32。

BioLinkBERT 的证据输出沿用前一轮已缓存结果，没有重新运行证据模型。组合方案由重跑前 104/192（54.17%）变为 96/192（50.00%），已知样本证据定位仍为 76/137（55.47%），判定和证据同时正确为 48/192。因此，解析修复显著改善了 TrialMatchAI 本身，但不能证明 BioLinkBERT 已改善证据定位，也不能据此把组合方案定为正式模块一。

公开结果、配置和复现文件位于 `runs/v5_20261018/parser_v3_retry/`；完整病例和 69 条原始重生成只保存在服务器私有目录 `runs/v5_retry_20261018_final/private/`。重生成脚本为 `scripts/rerun_failed_trialmatch.py`。本次仍使用 V4 的 192 条方法开发数据，属于探索性回归，不是独立验证。
