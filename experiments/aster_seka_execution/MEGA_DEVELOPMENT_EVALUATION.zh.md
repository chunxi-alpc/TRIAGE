# MEGA train_val 独立评价入口

2026-10-26。协调器已确认：MEGA读取器9项测试与统计工具6项测试在服务器通过；真实train_val冻结16例，覆盖候选源的1815个有病例指南组；`entity_ci_v2`计算完成。本次另增的评价入口、客户端与7项测试**尚未执行**，不改此前已验证文件，也没有读取临床test。

新增文件：`mega_development_evaluation.py`、`mega_local_qwen_client.py`、`test_mega_development_evaluation.py`。默认使用同进程HF本地Qwen裁判，不联网；API接口作为可选替代保留。通过协调器的`run_official_job`在空闲且加锁的单卡运行，生成和裁判分阶段加载。调用之前仍由协调器汇报GPU、预算和预计时长。

## 官方协议与固定分母

入口显式要求`--train-val-root`与先前冻结manifest，只读取train_val的病例文件指纹及`gt_paths.jsonl`作为评价参考。模型生成结果必须与冻结样本身份匹配；缺失生成行补为失败，重复或额外身份导致核验失败，不默默去重。每种方法均保留冻结16例分母。

直接调用固定官方commit的`parse_record`、路径/结论`evaluate_record`；调用原`average_and_normalize_matrices`和`dynamic_time_warping`计算PCC，以及原`normalize_accuracy_scores`计算Acc。逐个检查实际官方源码/模板与固定Git对象一致，不复制改写提示或指标公式。只计算主DTW PCC与Acc，辅助OT不在本入口内。

- 每个病例、每项指标预期5次Qwen评分，temperature=0.7、top_p=0.8、top_k=20、thinking=false，上限4096新token，串行调用。
- 官方裁判代码遇到一次解析失败会终止该项余下评分、标记skipped；本入口保留这种行为，记录实际完成数和所有已发生请求。未完成5次或生成/解析/裁判失败，PCC记1、Acc记0，不缩小分母。
- Acc是1–5裁判评分经`(score−1)/4`归一化后的均值。纠正/退化只报告**连续分数改善/下降的病例数**；二元纠正/退化为null，不假造准确/错误阈值。
- 比较按冻结指南文档将全部路径放入同一簇，10000次整簇bootstrap，病例加权，固定统计seed0。5次裁判评分留在病例内，不算5个患者。区间是开发样本上的边际95%区间，不是独立临床确认或多重检验显著性结论。

## 本地HF裁判与成本

`LocalQwenClient`实现官方函数需要的`chat.completions.create`接口。实际调用本地Qwen3的`apply_chat_template(enable_thinking=False)`和采样生成；要求只暴露协调器分配的一张GPU。使用现有检查点和隔离的transformers4.51.3环境，`local_files_only=True`，不会下载权重或联网。

本地适配在模型加载后固定一次torch CPU/CUDA裁判seed0，并按冻结方法顺序、病例顺序、路径后结论的顺序处理；上游API没有声明seed，因此这是明确登记的复现设置，不宣称逐字复现原SGLang随机输出。上下文默认32768，超预算作为失败记录，不截断参考或预测。主论文裁判是GPT-5.4 Mini；此处使用作者提供的Qwen替代轨道。若生成和裁判使用同一套Qwen权重，只有角色/阶段分离，**不称独立模型验证**。

成本包括每次请求尝试及官方重试、成功返回的实际输入/输出token数、每次耗时、全评价耗时、本地Qwen真实forward尝试/完成数和进程显存峰值。全局显存峰值不会按请求重置，保留整个过程最高值。外层`run_official_job`另保存GPU锁用时、所有权重分片指纹和forward输入哈希；输出token ID只保存在私有原始返回中。API模式禁用SDK内部重试，使每个实际客户端尝试可见；API无法自行测服务显存，必须另附服务运行清单，未知项不填假数。

## 服务器运行

先做CPU合成测试，指定固定官方parser位置后应运行全部7项；省略变量会明确跳过4项原生函数测试，不能宣称全通过：

```bash
ASTER_MEGA_OFFICIAL_PARSER="$MEGA_CODE/mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py" python -m unittest discover -s "$AUDIT" -p 'test_mega_development_evaluation.py'
```

以下是模块参数示意，由协调器放入现有单卡运行包装器；`--source`指现有已固定Git工作目录，`PYTHONPATH`需要包含`$AUDIT`以导入新模块。`$JUDGE_RUN`和`$EVAL_OUT`使用独立新目录。

```bash
PYTHONPATH="$AUDIT:$PYTHONPATH" python "$RUNNER" --gpu "$ASSIGNED_GPU" --source "$SEKA_CODE" --model "$QWEN_MODEL" --output "$JUDGE_RUN" --module mega_development_evaluation --seed 0 -- --train-val-root "$MEGA_DATA/train_val" --freeze-manifest "$FROZEN_MANIFEST" --official-parser "$MEGA_CODE/mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py" --prediction "baseline=$BASE_PREDICTIONS" --prediction "aster=$ASTER_PREDICTIONS" --reference baseline --model "$QWEN_MODEL" --output-dir "$EVAL_OUT"
```

可先只传一份`--prediction`核验评价链路，此时无配对比较。无需先搭建API服务。本地HF模式需要numpy、torch、transformers及openai包；openai仅因为官方模块顶层导入而需要，本地客户端不调用网络。

**目录参数注意**：已有运行包装器会递归哈希输入目录。这里传`train_val`而非整个dataset snapshot，传官方parser单文件而非整个仓库，防止外层指纹扫描遍历test。源GT只进入评价进程。方法参数、seed和生成预算不由评价器修改。

## 产物范围

可提交：`evaluation_config.json`、`evaluation_report.json`、`case_metrics.jsonl`、`judge_attempts.jsonl`，以及外层匿名运行清单。它们保存身份哈希、数值、源码/数据指纹和成本，没有指南全文或参考路径文本。

`private/`权限0700，包含全部原始裁判请求/返回、解析结果、原生裁判结果和原始token IDs；它们留在服务器供审计，**不提交PR**。原始调用失败和负结果同样保留。评价完成状态只表示计算与分母完整，不表示模型效果良好或所有裁判调用均成功。
