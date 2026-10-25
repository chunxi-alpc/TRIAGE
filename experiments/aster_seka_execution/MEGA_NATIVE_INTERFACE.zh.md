# MEGA 原生单轮开发接口

2026-10-26。代码与调用已按官方源码检查；本文件新增读取器及4项合成测试**尚未执行**，应由协调器在115服务器验证。先前6项MedPIC/统计测试已在服务器通过。这里没有生成或评价结果。

官方代码固定为 `nchennnn/MEGA-CDP@15a3a12e73a9169a870bff2ac431ec2e9968b8c7`；数据固定为 `nchennnn/MEGA-CDP@8f058c88ce1ed43693785b435ad0ff99907a2890`。内部科研使用决定见 `usage_decision.json`。正式许可证仍为TBD，不再分发原文。

## 输入与隔离

官方生成脚本读取 `train_val/vignette_cases.jsonl`；每行所需字段如下，表中只列结构，无病例内容。

| 字段 | 类型/用途 |
|---|---|
| `text` | 病例字符串，提供给模型 |
| `filename` | 指南文件标识，定位 `train_val/guideline_md/{filename}/{filename}.md` |
| `language` | 语言，默认en；zh使用中文官方模板，其余使用英文模板 |
| `case_path_id`，缺省回退`guideline_id` | 仅日志和评价对齐标识 |
| `path_index`，缺省回退`path_id` | 仅日志和评价对齐标识 |
| `id` | 可选原始标识，缺省为非空行序号 |

`mega_trainval_reader.py`只打开上述病例文件和指南；不打开`gt_paths.jsonl`、`decision_trees.jsonl`、`case_paths.jsonl`或test。方法函数只收`MethodInput(guideline, vignette, language)`；`TaskIdentity`在外层用于日志/评价。原始行即使包含gold字段也不会复制到方法输入或暂存输入。模型不接收参考树、参考路径、gold答案或gold规则选择。

指南文档SHA256作为开发分组，完全相同的文档合并一组。split seed固定0；按哈希固定80%组用于校准、其余用于开发，默认从开发部分固定抽16条；比例和批量在第一次冻结前可由协调器明确改变。组别及样本只由文档和身份哈希确定，无标签、长度或性能筛选。实际manifest生成后不可覆盖。该分组依据已发布文档，**不证明不同版本、译本或MEGA与MedGUIDE的来源独立**。

manifest只含ID、组别、数量、实际文件/输入/模板SHA256及配置。`--stage-jsonl`可生成官方脚本接受的最小病例文件，权限0600，**只保存在服务器私有实验目录，不提交PR**。预期revision另行登记；真实checkout HEAD仍由协调器记录，不把预期版本字符串当实际checkout证明。

方法适配入口：

```python
manifest = json.loads(manifest_path.read_text())
for identity, task in iter_method_tasks(dataset_root, manifest):
    prompt = render_official_prompt(task, official_root, manifest)
    # predict receives the prompt/task only; identity is retained by the logger.
    output = predict(prompt)
```

## 官方提示、预算与种子

原始英文/中文模板位于`mega_cdp/template/template_{en,zh}/`；在固定checkout读取，按原代码用`{Guideline}`和`{Case}`替换。SGLang后端将完整提示作为一条`system`消息。输出要求顺序临床决策步骤和单一结论；没有gold树占位符。

原作者启动器默认上下文131072、生成上限32768、TP1、显存比例0.85。生成脚本默认不传temperature/top_p/top_k/enable_thinking，因此随模型/服务配置变化；**没有seed参数**。开发如缩短预算，应冻结并标明“匹配预算开发设置”，检查每条完整prompt能否容纳，不能静默截断或丢弃超长样本。完整指南本身允许直接使用；只取gold路径涉及段落则增加了特权信息。

以下是已核对参数名的最短接口，变量均由协调器设为服务器实际路径。它们只调用已启动服务，不负责选择GPU或新启动模型。原生生成命令保留作者采样行为，不能凭目录名声称其是seed0。ASTER适配运行器必须实际设置并记录seed0，以及后续冻结的0/1/2；若后端不提供已验证的seed控制，应如实记录。

```bash
python "$AUDIT/mega_trainval_reader.py" --config "$TRIAGE/configs/datasets.json" --official-root "$MEGA_CODE" --manifest "$RUN/trainval_manifest.json" --stage-jsonl "$RUN/private_vignettes.jsonl" --batch-size 16

python "$MEGA_CODE/mega_cdp/scripts/inference/predict_singleturn_sglang.py" --model-name "$SERVED_MODEL" --backend sglang --base-url "$MODEL_URL" --input-jsonl "$RUN/private_vignettes.jsonl" --guidelines-root "$MEGA_DATA/train_val/guideline_md" --output-jsonl "$RUN/predictions.jsonl" --max-tokens "$GEN_MAX_TOKENS" --no-resume

python "$MEGA_CODE/mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py" --experiment-name "$RUN_NAME" --input-jsonl "$RUN/predictions.jsonl" --output-jsonl "$RUN/parsed.jsonl"
```

所有数据路径均需显式传入；原脚本缺省会读test。输出`predictions.jsonl`行结构为原身份字段加`llm_output: str, model, status`，错误行保留身份和error。解析后`llm_output`为`{clinical_decision_process: list[str], clinical_decision_outcome: str}`，并保留`parsed_success`。不要丢弃生成/解析失败行。

## 本地Qwen裁判与原生指标

官方入口 `mega_cdp/scripts/evaluate/singleturn_evaluate_qwen3_8b_sglang.sh`。生成结束后由独立评价过程读取train_val参考，方法进程不读取。已运行的裁判服务需使用官方`mega_cdp/template/qwen3_nonthinking.jinja`或确认等效禁用thinking。示例复用服务：

```bash
RUN_SERVER=0 PORT="$JUDGE_PORT" SERVED_MODEL_NAME="$JUDGE_SERVED_MODEL" EVAL_MODEL_NAME="$RUN_NAME" EVAL_PARSED_JSONL="$RUN/parsed.jsonl" EVAL_GT_PATHS_JSONL="$MEGA_DATA/train_val/gt_paths.jsonl" PATHWAY_OUTPUT_JSONL="$RUN/pathway.jsonl" ACCURACY_OUTPUT_JSONL="$RUN/accuracy.jsonl" QWEN_MAX_TOKENS=4096 EVAL_REQUEST_WORKERS=1 bash "$MEGA_CODE/mega_cdp/scripts/evaluate/singleturn_evaluate_qwen3_8b_sglang.sh"

python "$MEGA_CODE/mega_cdp/scripts/evaluate/summarize_eval_metrics.py" --exp-name "$RUN_NAME" --input-jsonl "$RUN/pathway.jsonl" --accuracy-input-jsonl "$RUN/accuracy.jsonl" --summary-txt "$RUN/summary.txt" --case-metrics-jsonl "$RUN/case_metrics.jsonl" --eval-method qwen --expected-results 5 --accuracy-expected-results 5
```

Qwen judge默认上下文32768、显存比例0.75、输出4096、并发1；每条路径/结论分别5次评分。固定temperature0.7、top_p0.8、top_k20、thinking=false；上游未传seed。5次judge不作为5个独立患者或方法seed。主论文用GPT-5.4 Mini judge，官方Qwen替代轨道必须单列。

路径指标：5次相似度矩阵先求均值，再将1–5分归一化为0–1，以`1−相似度`求DTW路径平均成本；主PCC成本越低越好。最终结论`Acc`实际是5次judge分数各按`(score−1)/4`归一化后的均值，**不是二元精确正确率**。连续Acc改善/退化可以报告；若另外计算二元纠正/退化，应预先单独定义正确标准并注明派生指标，不能混用。

官方汇总默认要求正好5次，已存在的失败行按PCC=1/Acc=0计入分母；但不对照冻结manifest补齐整个缺失病例。汇总前应核对生成、解析和两种评价的身份集合完整且唯一，防止缺行缩小分母。依赖numpy和POT；缺POT会使路径计算异常并被归为最差，不能误报为模型退化。逐例指标再按manifest真实文档组交给`paired_statistics.py`；跨seed保留组内相关性。

官方生成resume按身份、judge resume按行号，没有全面指纹核对。每份冻结输入使用独立新输出路径；改变数据/方法/预算/顺序后不得复用旧输出。暂存输入、原始参考和病例/指南全文不提交。

## 可复核来源

- [官方生成脚本](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/mega_cdp/scripts/inference/predict_singleturn_sglang.py)
- [官方启动器](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/mega_cdp/scripts/inference/run_singleturn_inference_sglang.sh)
- [英文提示](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/mega_cdp/template/template_en/DECISION_MAKING_PREDICTION_TEMPLATE.txt)
- [Qwen judge入口](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/mega_cdp/scripts/evaluate/singleturn_evaluate_qwen3_8b_sglang.sh)
- [官方指标汇总](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/mega_cdp/scripts/evaluate/summarize_eval_metrics.py)，blob`f67a7b8bc14809fe0ef357503b828fe273f6174d`
- [DTW实现](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/mega_cdp/scripts/evaluate/pathway_metrics.py)，blob`735728cf96ad94b78a01d53adbf9a72611a43447`
