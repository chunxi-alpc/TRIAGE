# C_V6 20例真实补充验收

## 实际完成

在GPU6与gcx261014解释器下实际执行：48个去重TrialMatchAI请求、20例BioLinkBERT证据与V6逻辑推理、20例Qwen约束选项评分。全程811.20秒（13.52分钟）；真实TrialMatchAI批次共775.24秒。不是计划结果或缓存旧病例复评。

模型一峰值allocated 13.174 GiB、reserved 14.973 GiB；整个进程含Qwen峰值allocated 15.292 GiB、reserved 15.318 GiB。三套模块一模型的实际本地权重/配置摘要已流式核验，见sources/c_full_model_manifest.json，与既有可信清单共有项全部一致。

## 真实观察

- 48条原始生成中42条解析接受，6条达到2048词元上限后截断；没有技术异常。原始输出全部保存，没有补写、改标签或截断后猜测。
- 20个NEW规则状态为12个U、8个F，无T；其中10/20状态与程序参考一致。该数字仅属于预选的2个合成家族。
- 全部20个NEW规则证据状态为disputed。60个NEW原子中47个disputed，授权C2和例外E1各20/20 disputed；数值C1为7/20 disputed。
- 因预先规定的证据门控，最终20例signed_gate全部为0。因此C_V6实际关闭全部干预，约束选项评分为12/20，与同病例A基线答案和所有选项分数逐值完全相同。
- 这只能说明现有V6在这组虚构路由输入上未提供可开启的可靠证据门控，不能支持动态控制改善。也不能凭两家族推断临床表现。
- 自由生成补充可对这20例精确复用同item、alpha=0的A生成；本子任务未创建自由生成结果。

可见的错误链：候选句检索将其它含current的观察也纳入授权/例外NLI，BioLinkBERT对一些无关句给出支持或矛盾。例如patient_routing-alder-00-min_apply中，“The current patient age is 20 years.”被判为蕴含“有路由授权”，其它条件的否定句被判为矛盾；因此同时出现支持和反驳。这里只报告模型真实关系输出，不据此推断模型的一般医学能力。

另外，冻结V6仍将排除条件Not Violated映射U；8个唯一模型请求出现此标签。该行为保留，未为了改善结果切换成inclusion。6个截断请求也原样留为未知。

## 验收与时间解释

20个输出ID唯一且与固定子集完全一致；所有alpha为0；和A的候选分数完全一致。audit/c_full_acceptance.json保存验收结果，results/c_full/new_atom_diagnostics.json保存逐原子的标签、证据句与NLI关系。

results/c_full/structures.jsonl保存完整模块一输出及最终门控，gates.json保存每item的真实状态和控制量，results/c_full_predictions.jsonl保存最终Qwen输出。主predictions.jsonl未由本子任务写入。

结构中的旧V6字段trialmatch_seconds是缓存重放调用所花时间，不能当Phi-4模型延迟。真实模型时间应读取原始生成inference_seconds/batch_elapsed_seconds、complete.json的批次总耗时，或顶层module1_trialmatch_batch_amortized_seconds。重复请求时间已按被使用次数分配；这表示本次有缓存实验的吞吐成本，不是单病例独立部署延迟。

C_V6执行了现有V6完整预测与证据链，但原CARE上传文件未找到，不能称原CARE完整复现。与C_reduced相比还增加了证据missing/disputed关门策略，不能把变化全部归因于适用性预测误差。没有重新调方向、系数或阈值，没有修正金标，没有删除病例。
