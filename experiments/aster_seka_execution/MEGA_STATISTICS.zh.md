# MEGA 临床开发结果的独立 CPU 汇总

本目录不调用模型、不发送裁判请求、不读取数据集问题或参考答案。只对已完成裁判评估的逐例原生分数、冻结病例/指南分组、生成及构建记录进行汇总。本地没有运行测试。协调器已报告三个合成测试在 115 服务器通过，日志为 `runtime/test_summarize_mega.log`；尚无真实临床汇总。以下为服务器命令。

```bash
PYTHONPATH="$EXPERIMENT_ROOT" python "$EXPERIMENT_ROOT/test_summarize_mega.py" -v
python "$EXPERIMENT_ROOT/summarize_mega.py" --statistics-dir "$EXPERIMENT_ROOT" --spec /absolute/path/summary_spec.json --output-dir /absolute/path/new_summary
```

服务器实际采用扁平 p 目录，本文 `$EXPERIMENT_ROOT` 指该目录，内含全部脚本及 `paired_statistics.py`；实际入口使用 `--statistics-dir "$EXPERIMENT_ROOT"`。输出目录必须新建，失败也保存 `summary.json`，不会覆盖先前失败。这里没有预填的实际结果。

spec 路径可为绝对路径，或相对 spec 文件。每个方法显式列出评估名称、原生成名称和条件分支，避免 A0/A1 的同名融合方法被合并。例子是输入格式，不代表实际任务已完成：

```json
{
  "freeze_manifest": "/run/frozen_development.json",
  "primary_run": "/run/primary",
  "methods": [
    {
      "name": "A1B1",
      "generation_method": "A1B1",
      "prediction": "/run/primary/A1B1.predictions.private.jsonl",
      "condition": "A1",
      "evaluation_dir": "/run/evaluation_primary",
      "evaluation_method": "A1B1",
      "calibration_jobs": ["seka_projection"]
    },
    {
      "name": "A1_fusion_static",
      "generation_method": "fusion_static",
      "prediction": "/run/extra_A1/fusion_static.predictions.private.jsonl",
      "condition": "A1",
      "evaluation_dir": "/run/evaluation_extra",
      "evaluation_method": "A1_fusion_static",
      "calibration_jobs": ["seka_projection"]
    }
  ],
  "comparisons": [{"method": "A1B1", "reference": "A1_fusion_static"}],
  "calibration_jobs": {
    "seka_projection": {
      "family": "seka",
      "run_manifest": "/run/projection_job/run_manifest.json",
      "scope": "Actual shared SEKA positive/negative projection construction"
    }
  },
  "generation_processes": [{"run_manifest": "/run/primary_job/run_manifest.json"}]
}
```

`name` 是本次公开汇总名称；`generation_method` 必须匹配预测行；`evaluation_method` 必须匹配 `evaluation_config.json` 的实际映射。后二者省略时等于 name。`condition` 为 A0、A1 或 full_guideline 的 null，必须匹配生成协议。每个方法都必须显式提供 calibration_jobs 数组，没有则为 []。构建 family 使用 seka、pasta、adaseka、fusion；四专家通常为四个独立 adaseka job。A1 条件方法需要计入 SEKA 构建，PASTA/Ada 方法另计对应构建，失败构建可登记在全局表中，状态与成本照实保留。历史 PR33 参数转用如没有可核验的原校准成本，会明确列出 fusion 成本缺失；不能由 [] 变成免费校准。

统计使用全部冻结病例，Acc 为原生连续归一化分数且越高越好，PCC 为原生 DTW 路径成本且越低越好。解析/裁判失败由评估器记固定最差分数，汇总核对这些值，不删病例；若 case_metrics 缺行、重复、指纹变动或裁判尚未结束，则停止并记录失败，不自造裁判结果。均值与方法差的 95% 区间按完整来源指南组重采样；只有一组时保留点估计并说明区间不可计算。连续指标只给改善/退化/不变，不称二元纠正/退化。仅实际 seed=0 开发结果，不冒充种子 0/1/2 的冻结独立评估。

成本有三种独立口径，不相加：

- 方法应承担的推理：实际 B 输出（复用时追溯原输出）、所选 A 条件预测、每份指南抽取缓存一次。完整批次内这等价于将抽取成本均摊到共享该缓存的病例。记录内层 elapsed 与外层 wall 同时存在时只取 enclosing wall，不重复相加。
- 实际生成调用总账：各已提供运行目录的所有计划方法，包括负对照、额外 nohook 和数值探针、条件控制；每个真实文件行仅记一次，复用行按新增调用 0 记账。它不等于各方法承担成本之和。
- 构建、裁判、整个生成进程成本：分别按真实 manifest/评估目录唯一登记。进程时间含加载、哈希等，与逐调用时间重叠；不能再次相加。未提供进程 manifest 时不估算开销。逐例显存取保存的最大值，不能恢复首次 reset 前加载峰值。

缺失的尝试次数、耗时或条件成本保持 null/缺失标记，不由完成次数推定尝试次数。共享记录副本不会重复计入成本。数值零强度只汇总实际保存的漂移标量及输入一致性记录；完整词表分数未保存时不声称独立复算。A0/A1 方法比较明确标出条件信息是否相同；本地 Qwen 裁判与论文 GPT-5.4 Mini 不同，也可能与生成模型共享检查点，所有输出明确禁止独立模型验证的结论。

三个合成测试覆盖连续指标方向与全部失败分母、按指南共享抽取与 A1 成本、复用不免费且总账不重复、缺失裁判行/预测 hash 改变拒绝、数值诊断缺失及未知尝试次数不补造。

## 全部临床队列的统一入口

原生裁判程序源码位于 `work/data_audit/mega_development_evaluation.py`，服务器发布为 `$EXPERIMENT_ROOT/mega_development_evaluation.py`，没有另建评分实现。新增 `summarize_all_mega.py` 直接兼容它的 `evaluation_config.json`、`evaluation_report.json` 和 `case_metrics.jsonl`，并读取三种生成器的四份实际 protocol：primary 11 臂、extra A0 18 臂、extra A1 6 臂、attention A0 8 臂。共 43 方法，每方法固定 16 例，共 688 方法×病例记录。协调器已在 115 服务器执行原 3 项及新增 2 项测试，全部通过，后两项耗时 0.370 秒。本地没有运行测试，尚无真实临床评分。

在全部生成和原生裁判结束后，只需一个 CPU 命令：

```bash
python "$EXPERIMENT_ROOT/summarize_all_mega.py" --statistics-dir "$EXPERIMENT_ROOT" \
  --freeze-manifest "$FROZEN16" --primary-run "$PRIMARY" \
  --additional-a0-run "$EXTRA_A0" --additional-a1-run "$EXTRA_A1" \
  --attention-run "$ATTENTION_A0" \
  --evaluation-dir "$EVAL_PRIMARY" --evaluation-dir "$EVAL_EXTRA_A0" \
  --evaluation-dir "$EVAL_EXTRA_A1" --evaluation-dir "$EVAL_ATTENTION" \
  --cost-spec "$EXPERIMENT_ROOT/clinical_all43_cost_spec.json" --output-dir "$NEW_ALL43_SUMMARY"
```

裁判可合在一个目录或分成更多目录，重复提供 `--evaluation-dir` 即可；程序按实际预测文件 SHA 找到已评估别名，不要求改写旧裁判文件。统一名称使用 `primary__`、`extra_A0__`、`extra_A1__`、`attention_A0__` 前缀，避免两个条件分支的 fusion 等名称冲突。若同一文件被重复评估且存在多个候选，会优先匹配同一统一别名；仍有歧义则要求显式 `--evaluation-map`，不会按分数选择。有必要时该文件格式为 `{统一名称: {"evaluation_dir": "实际目录", "evaluation_method": "已有别名"}}`，相对路径以该文件为基准。

若尚未启动裁判，可在相同生成目录参数下使用 `--prepare-evaluation-map-only` 和新输出目录。它只写 `evaluation_predictions.json` 的 43 个别名/路径及 `--prediction` 参数数组，供协调器传给既有原生裁判；不会开始判分，也不生成临床统计结果。

cost-spec 继续使用 `calibration_jobs` 和 `generation_processes`；可加 `family_jobs`（如 `{"seka":["seka_projection"],"pasta":["pasta_profile"],"adaseka":["ada_syn","ada_cf","ada_bio","ada_hotpot"]}`），自动按方法真实依赖归属。单个方法需要不同归属时用 `method_calibration_jobs` 覆盖。未提供成本证据会明确记录缺失，不能成为零校准成本。一个 extraction cache 在每个方法的归属成本中只记一次；实际总账中只按物理文件/行记一次，不将 43 份归属总计相加。失败准备行缺少测量时新增成本也保持未知，而不是省略后伪称零成本。

固定比较计划覆盖所有方法与同 A 条件的 B0、A0/A1 二维消融、所有额外基线与同条件 B1、PASTA/Ada 的各自 nohook/zero。不可用控制和错误病例保留其原状态与完整分母。SEKA execution reuse 是已有 A0B1 输出，不能当作另一个独立方法实验证据；若对相同输出再次随机采样裁判，评分变化可能来自裁判随机性。

新增服务器测试命令：

```bash
PYTHONPATH="$EXPERIMENT_ROOT" python "$EXPERIMENT_ROOT/test_summarize_mega.py" -v
PYTHONPATH="$EXPERIMENT_ROOT" python "$EXPERIMENT_ROOT/test_summarize_all_mega.py" -v
```

第二项通过真实统一 CLI 构造全部 43×16 个合成记录，覆盖三种生成 schema、A0/A1 区分、失败/不可用控制、复用成本及总账唯一计数；另验证缺少一个裁判臂明确失败、准备映射模式不启动裁判。不会运行本地测试或读取实际临床分数来调整方法。

## 补齐剩余原生裁判结果的 GPU0 队列

`evaluate_remaining_mega.py` 的 prepare 模式只写队列。实际 execute 由协调器通过原 GPU 锁运行器启动，等待 primary 评价 parent 428868 和 attention parent 428866 退出后，才读取完成的 43 方法映射。运行目录固定为：

- `$RUNTIME/mega_dev16_seed0_gpu01`
- `$RUNTIME/mega_dev16_additional_gpu01_A0` 和 `..._A1`
- `$RUNTIME/mega_dev16_attention_gpu01`
- 原 primary 评价：`$RUNTIME/mega_dev16_primary_gpu01_evaluation`

准备命令（不启动实验）：

```bash
PYTHONPATH="$EXPERIMENT_ROOT" "$PYTHON" "$EXPERIMENT_ROOT/evaluate_remaining_mega.py" prepare \
  --experiment-root "$EXPERIMENT_ROOT" --runtime-root "$RUNTIME" \
  --python "$PYTHON" --source "$SEKA_ROOT" --model "$JUDGE_MODEL" \
  --freeze-manifest "$FROZEN16" --train-val-root "$MEGA_SNAPSHOT/train_val" \
  --official-parser "$MEGA_ROOT/mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py" \
  --primary-evaluation-manifest "$PRIMARY_JUDGE_RUN_MANIFEST" \
  --gpu 0 --wait-for-pids 428868 428866 \
  --output-dir "$RUNTIME/mega_dev16_remaining_native_gpu0_job" \
  --plan "$EXPERIMENT_ROOT/mega_dev16_remaining_native_gpu0_queue.json"
```

生成的 JSON 已含实际 GPU0 job 命令与以下队列启动参数；本代理没有执行它：

```bash
"$PYTHON" "$EXPERIMENT_ROOT/run_baseline_queue.py" "$EXPERIMENT_ROOT/mega_dev16_remaining_native_gpu0_queue.json"
```

执行时核对 primary 已完成、同冻结输入及外层完整裁判模型文件指纹。原 11 臂按预测文件 SHA 复用；其余最多 32 个文件仅在整份文件字节完全相同时复用旧结果或新的唯一代表。所有 43 方法仍保留 16 例，`reuse_manifest.json` 和 `evaluation_map.json` 逐方法记录来源。没有按得分、相似文本或近似输出复用。原预测和裁判文件均不改写。

唯一新文件传给原 `mega_development_evaluation.py` 和现有 `mega_local_qwen_client.py`；实际 judge 参数逐项沿用 primary config，包括 5 次 PCC/5 次 Acc、temperature .7、top_p .8、top_k 20、非思考、输出/context预算、种子与 bootstrap 次数。不会更换评分法。失败和缺失预测仍交原生 evaluator 按固定 16 分母处理。

预算上限为 32×16=512 方法病例、5120 次正常裁判请求；原最多 3 次传输尝试对应 15360 次请求上界。若每次输出上限为4096，理论上界为 20,971,520 个生成 token（不计重试）或 62,914,560（含最多重试）；这是预算上限，不是预测实耗。prepare 能读取已有 primary 配置时采用实际 token 上限，否则暂列原生默认4096；execute 再按实际唯一文件数和 primary 参数更新预算。

资源只用 GPU0 一张80GiB A100，规划显存20–40GiB；它是8B BF16裁判的预留范围，不能冒充实测最高峰。primary 完成后保存其真实显存及 `primary_elapsed×new_unique_methods/11` 粗略时间外推，未完成时预计秒数保留 null，不虚造耗时。长路径、失败比例、重试与模型加载会影响外推。当前不使用 GPU6。

execute 不在工作结束后偷偷重复裁判。它会生成 `summary_invocation.json`，指向统一 CPU 汇总入口、原 primary 与新 native_remaining 两个目录、精确复用映射及 `$EXPERIMENT_ROOT/clinical_all43_cost_spec.json`；协调器可直接执行其中 `python_argv`。旧 primary 裁判成本只计一次，唯一新文件成本由原 evaluator 记录，复用方法新增判分成本为0。结果仍属于本地 Qwen 开发裁判，不能称独立模型确认。

新增准备/复用合同测试（服务器执行，本地未运行）：

```bash
PYTHONPATH="$EXPERIMENT_ROOT" "$PYTHON" "$EXPERIMENT_ROOT/test_evaluate_remaining_mega.py" -v
```

测试仅用伪文件，核对43方法保留、精确SHA复用、原裁判参数接线、GPU0依赖队列及请求预算，不会调用模型、裁判或子进程。实际 worker 参数只传四个指定生成目录和源码 `--reader-dir`，不把整个 runtime 根目录交给运行器递归登记。
