# ASTER-CR 最终实验缺口核验与最小补实验执行审计

<!-- HISTORICAL-SCOPE-START -->
> **历史批次/审计快照。** 以下完成或缺失状态对应当时范围；后续CPU补算已完成，GPU接续按统一队列执行。当前逐项状态见[实验框架](../../docs/ASTER/experiments.zh.md)及其执行状态附件。原始数值和审计证据保留。
<!-- HISTORICAL-SCOPE-END -->

审计日期：2026-11-14。服务器：`gcx@10.29.128.115`；环境：`gcx261014`。本轮输出审计材料和待批准计划，未执行新模型前向、训练、模型下载、补实验统计区间或GitHub写入。

**最小GPU范围为四个E6官方MEGA单元和E5十三行干扰路径。Gemma3、MedGemma的806条受控自然路径可由既有全头路径缓存完成CPU重建，GPU任务应从原六个E6单元减为四个。训练任务为0。**

已找到上一轮未纳入复用清单的两款27B全头路径缓存；它们补足了各自的实际访问节点。服务器仍缺对应的当前E1最终预测文件，需在批准后进行CPU重建和统计。缓存覆盖检查与实验交付完成状态分别登记。

## 1. 基准、证据与搜索范围

- 主线：`a783bc6330512028fd167868aff0f85666e747f8`。归档：`37cce4de0fc64cf1bfcb928a07ba3c60fb5c2584`。关联：[PR #69](https://github.com/chunxi-alpc/TRIAGE/pull/69)。
- 已首先通读[上一轮完整报告](https://github.com/chunxi-alpc/TRIAGE/blob/37cce4de0fc64cf1bfcb928a07ba3c60fb5c2584/reports/aster_remaining_experiments_audit_20261114/REPORT.zh.md)，其缺失判断作为待核查假设。副本保存在 `previous_audit.zh.md`。
- 服务器发布工作树位于`b67f7367…`。GitHub比较确认其至指定归档提交的文件差异为实验管理文档和上一轮审计报告；预测文件未改变。因此服务器发布树可用于核对该归档的逐例内容。

| 范围 | 本轮实际检查 |
| --- | --- |
| 文件清单 | 2,525,550个文件；包含隐藏文件、非Git文件及软链接目标记录。 |
| 目录 | `/home/gcx`、`/tmp`、`/var/tmp`、`/dev/shm`、`/data`、`/mnt`、`/media`，并补查`/srv`、`/opt`、`/var/backups`及存在的备份/工作目录。 |
| 工作树与Git历史 | 70条注册工作树记录（其中1条失效临时工作树）、22个有改动的工作树；另检查7个仓库。165个本地/远程引用、106个不同提交；6个Git bundle列出引用，未导入或改写仓库。 |
| 逐例文件 | 检查19,745个JSONL候选；识别2,100份路径结果、3,841份分数/视图、1,853份配对结果、90份整题结果。 |
| 其他格式 | 检查8,075个CSV及1,877个其他GZIP；205份CSV路径结果按SHA256归并为14份不同内容。 |
| 压缩归档 | 143个ZIP/TAR/压缩包或Git bundle，SHA256归并为42份不同文件；归档成员12,284项。检查了成员列表及其中的逐例文件。 |
| 内容检索 | 243,944份元数据、代码和日志；按内容归并93,836份。检索E1协议、模型/方法名、目标分母、revision、SHA256、任务ID及完成/停止记录。 |
| 重复识别 | 逐例内容归并为1,928组，其中744组含副本；保存物理文件SHA256及GZIP解压内容SHA256。 |
| 模型缓存 | 检查模型清单、已有权重哈希、分词器/模板、全头/选头数组、最终残差、缓存UID及调度历史；未重新递归计算基础模型权重哈希。 |

用户指定的13个目录均已检查。搜索还覆盖`aster_five_models_20261109`、`aster_cr_repair_v2_run_20261108`、各发布临时工作树、`aster_cr_final_results`、历史`aster_*`目录及`TRIAGE/results/diagnostic_backups`。

范围记录见`search_scope.json`、`worktree_status.json`、`additional_search_scope.json`、`git_ref_artifact_inventory.json`、`archive_summary.json`、`archive_members.jsonl`和`metadata_hits.jsonl`。原始完整文件清单保存在服务器`AUDIT/inventory.jsonl`。

本报告的“全面搜索未发现”限定于上述可访问实验空间。权限拒绝涉及其他用户/系统临时目录、容器目录和lost+found；坏软链接主要属于软件依赖。4份不按JSONL编码的候选属于TempoMed/CounterFact数据，已记录，未作为ASTER结果。系统软件目录、虚拟环境、依赖包和Git对象实体不递归搜索；Git内容通过引用树检查。

| 路径简称 | 服务器绝对路径 |
| --- | --- |
| `R` | `/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112` |
| `Q` | `/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111` |
| `F` | `/home/gcx/aster-cr-balanced-publish-20261112/reports/aster_cr_final_method_10006_20261112` |
| `D` | `/home/gcx/aster-cr-diagnostics-e3cv-20261111/reports/aster_cr_m3_intercept_diagnostics_20261112` |
| `E5` | `/home/gcx/aster-e5-gaps-20261112/reports/aster_e5_gaps_20261112` |
| `EXT` | `/home/gcx/aster-codex03-external-20261112/reports/aster_codex03_external_20261112` |
| `PUB` | `/home/gcx/aster-results-publish-20261114` |
| `AUDIT` | `/home/gcx/aster-final-missing-audit-20261114` |
| `OLD5` | `/home/gcx/aster_five_models_20261109/results` |
| `OLDM` | `/home/gcx/aster_cross_model_mega_20261111` |
| `OLDQ` | `/home/gcx/aster_next_20261109/primary_results/mega_official_test` |

下文“简称/相对路径”拼接后为精确服务器路径。哈希均为完整SHA256。审计附件提供精确缺失case、路径与UID清单；不含原始临床提示或患者文本。

## 2. 当前E1协议与版本核查

E1规范化训练清单实际为2,540对，A/P/C/F各635；拟合80来源、开发20来源、受控测试40来源。五IT及两Base均有冻结的M1/M2/M3。参数为各检查点独立选择的51头、16维、λ=0.01、δ=0.2923349。A/P权重合计4,887为损失归一化权重，不是旧版4,887个A/P训练对。

规范化清单`Q/data/E1_pairs.jsonl`与包含原文的`/home/gcx/aster_cr_pair_dataset_20261111/balanced_train.jsonl`采用不同表示；样本角色、类别和来源对应E1。`F/models/E1/full`的五个参数文件与`Q/models/E1/full`逐字节相同，F目录中的这个E1对照可复用。F的其他混合来源模型单独保留。

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `Q/data/E1_pairs.jsonl` | 682,659 | `1ae3fc5dd11b58c9e8a4913c79f7768516013486c99291a87b9f28be0f113291` |
| `/home/gcx/aster_cr_pair_dataset_20261111/balanced_train.jsonl` | 77,681,550 | `f0995a8978a590a5ebe8c8331e597fded5d16d25b5ccddb576a85d5f42c65c37` |
| `Q/data/path_features.jsonl` | 16,994,432 | `3e8c362692904f5146310afb88c3a4c9853c8dffe0f576829b6bb24cf7d34509` |
| `Q/data/path_interference_views.jsonl` | 53,608,608 | `ae423ba12c4ebc85e0ed59ac5137687de598d49a4cca5245a5e1f19b899ee4dd` |
| `/home/gcx/aster_next_20261109/inventory/official_all_path_cases.jsonl` | 214,838,390 | `54a349d8b9b251dd645613b0c7d685ba46e77b666dff0b1f5bc8b89c84a6bd43` |

评分保持原检查点模板：最后提示token的“是/否”单token分数，BF16主干、FP32标签投影，取是减否的logit差，判定阈值为0；Gemma2保留原生softcap。IT使用各自聊天模板，Base使用配置中记录的原生completion模板。未把旧提示、旧M3或旧头集合产生的修正分数记作当前结果。

| 检查点 | 上游不可变revision | 本地证据与追溯状态 |
| --- | --- | --- |
| Qwen3-8B-IT | 未找到 | NEEDS_VALIDATION：本地权重路径/清单/模板可定位，上游revision未确认 |
| Llama-3.1-8B-Instruct | 未找到 | NEEDS_VALIDATION：本地权重路径/清单/模板可定位，上游revision未确认 |
| Gemma-2-9B-IT | 未找到 | NEEDS_VALIDATION：本地权重路径/清单/模板可定位，上游revision未确认 |
| Gemma-3-27B-IT | 未找到 | NEEDS_VALIDATION：本地权重路径/清单/模板可定位，上游revision未确认 |
| MedGemma-27B-Text-IT | 未找到 | NEEDS_VALIDATION：本地权重路径/清单/模板可定位，上游revision未确认 |
| Qwen3-8B-Base | 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 | Base清单及分词器小文件与固定revision记录一致 |
| Gemma-2-9B-Base | 未找到 | NEEDS_VALIDATION：本地权重路径/清单/模板可定位，上游revision未确认 |

模型权重分片清单、大小和已有哈希来源、分词器及配置文件本轮SHA256均保存在`model_identity.json`。Qwen IT使用历史权重哈希记录；Gemma2 Base/IT保留上一轮已实际计算的分片哈希。其余下载清单中的revision/hash缺项如实保留，未用可变分支名推测不可变revision。所有完成项的“本地E1计算完整性”和“上游版本追溯”分别列出；追溯缺项不自动生成重训任务。

五IT参数、冻结记录、种子、来源清单、M1头集合、M2数组形状、M3配置及独立截距文件见`protocol_current.json`。本地参数哈希见下表；每行参数目录还包含`m1.json`、`m3.json`、`model.json`，完整哈希在`artifact_sha256_registry.json`。

| 模型 | 参数目录 | m2.npz SHA256 | m3.npz SHA256 |
| --- | --- | --- | --- |
| Qwen3-8B-IT | `Q/models/E1/full` | `a8d7af033fb3de99445156a195d68f4e0a29e1a6629fe90fe49aaae03c8d8f18` | `a23d5226dbf7976d85c440a09b42691c63302f52f40d24058fa761e2d0bbb571` |
| Llama-3.1-8B-Instruct | `R/models/llama31/models/E1/full` | `b30bed9422129ab68edc070f029fbec59abe20a77070f2b124b83cbfb5417807` | `2515cf4f5d9304e262e82f8720f35baf5ea51ddf1f03f5f63816929a8a7b07f5` |
| Gemma-2-9B-IT | `R/models/gemma2/models/E1/full` | `7b11bd35510857aeb8b3d200bd5fb5243e859dae6c2863cbd1b7748c40484d3c` | `5eb48a4d11132ca88220fc445e3c660ec2aab7b72e8d44f66ec2802666214b00` |
| Gemma-3-27B-IT | `R/models/gemma3/models/E1/full` | `b5cc6fcdbad3e2a2e18c78de4aeaa1ae00f28cfa82db0510d90f5bf17e47e8bd` | `c174501a4e6bf97f4acc51dc859539cc0a134e769917b7ced6d52a89ad197dcc` |
| MedGemma-27B-Text-IT | `R/models/medgemma/models/E1/full` | `6666bbb2baebd625876ef5a846df1d3f37081698913606563b74331583defac0` | `30f095fb3fef519ce50bb0a87959e1febf90ea6e09b5e0c4861ae98cdfc01250` |
| Qwen3-8B-Base | `R/models/qwen_base/models/E1/full` | `1f7cee9573d3d79405058aeec4c8bfdd20bec2a7a489ed614d8d36c379fab36b` | `67a1b9e309d790f3c59dd330de398591340c8f916b858f68dcf59c11ae243ec0` |
| Gemma-2-9B-Base | `R/models/gemma2_base/models/E1/full` | `c4280d8c36f01378871abad5ea7a2aaa579c16af00037723d51f6486b9b1f106` | `ba8c9751d233c463ad2ba6df09754c6c2e99f279af18c1980c4b250ad68a2915` |

当前源任务独立截距来自761个MEGA拟合节点、80个fit来源。Qwen的冻结b₀=3.0993948961894406，`D/configs/independent_intercepts_reused.json`的761个ID均能映射到源任务fit自然节点，`target_task_fit=false`。四个非Qwen模型现存迁移逐例的独立截距分数与raw+b₀的最大差为0。

## 3. E6逐项结论与实际预测计数

| 模型/目标 | 当前结果状态 | 已保存的raw / 截距 / E1唯一路径 | 来源 | 缺失路径 | 处理 |
| --- | --- | --- | --- | --- | --- |
| Qwen IT 官方7458 | NOT_FOUND_AFTER_EXHAUSTIVE_SEARCH | 0 / 0 / 0 | 0 | 7458 | 当前E1补充评价；历史完整结果另列HISTORICAL_ONLY |
| Gemma2 IT 官方7458 | PARTIAL_CURRENT | 2379 / 2379 / 2379 | 115 | 5079 | 保留已有分片与未完成路径的6个节点，接续 |
| Gemma3 受控806 | PARTIAL_CURRENT | 0 / 0 / 0 | 0 | 806 | 兼容缓存可遍历三方法各806；CPU重建 |
| Gemma3 官方7458 | NOT_FOUND_AFTER_EXHAUSTIVE_SEARCH | 0 / 0 / 0 | 0 | 7458 | 当前E1补充评价；历史结果HISTORICAL_ONLY |
| MedGemma 受控806 | PARTIAL_CURRENT | 0 / 0 / 0 | 0 | 806 | 兼容缓存可遍历三方法各806；CPU重建 |
| MedGemma 官方7458 | NOT_FOUND_AFTER_EXHAUSTIVE_SEARCH | 0 / 0 / 0 | 0 | 7458 | 当前E1补充评价；历史结果HISTORICAL_ONLY |

六单元均完成跨工作树、分片、归档、缓存、日志与Git引用检索。没有找到另一份完整当前E1官方预测或Gemma2剩余分片。目标分母：受控806路径/40来源，官方7,458路径/418来源。当前结果缺失与旧协议完成分开记录。版本追溯缺项见第2节。

| 已有完整评价 | raw正确数 | 独立截距正确数 | E1正确数 | 每方法路径/来源 |
| --- | --- | --- | --- | --- |
| Qwen3-8B-IT 受控 | 768 | 767 | 768 | 806/40 |
| Llama-3.1-8B-Instruct 受控 | 629 | 677 | 672 | 806/40 |
| Llama-3.1-8B-Instruct 官方 | 5689 | 5898 | 5899 | 7458/418 |
| Gemma-2-9B-IT 受控 | 761 | 765 | 763 | 806/40 |
| Qwen3-8B-Base 受控 | 52 | 114 | 111 | 806/40 |
| Qwen3-8B-Base 官方 | 521 | 1009 | 1087 | 7458/418 |
| Gemma-2-9B-Base 受控 | 242 | 406 | 435 | 806/40 |
| Gemma-2-9B-Base 官方 | 2369 | 3716 | 4128 | 7458/418 |

Qwen Base受控和官方两个单元登记为COMPLETE_CURRENT；表内其余完整计算单元因上游不可变revision缺项登记为NEEDS_VALIDATION，具体追溯状态见第2节。

以上均重新读取逐例文件计数，并将保存的路径/终点与数据清单中的gold路径/终点比较：现存完整E6记录的正确标记差异为0，重复路径为0。Qwen受控独立截距来自`D/predictions/test/paths.jsonl.gz`。Qwen Base官方结果属于Base，原`qwen_mega_parallel`目录不能用作Qwen IT完成证据。

### 3.1 两款27B的新增缓存复用证据

| 检查 | Gemma3 | MedGemma |
| --- | --- | --- |
| 原计划已映射的自然候选UID | 761 | 761 |
| 旧路径缓存新增可复用UID | 4251 | 4262 |
| 合并后的自然缓存UID | 5012 | 5023 |
| 三方法可完整遍历的自然路径 | 806 / 806 / 806 | 806 / 806 / 806 |
| 实际路径首个缺失UID | 0 | 0 |
| 旧/当前病例、指南、定义、gold结构一致 | 806 | 806 |
| 重叠缓存raw分数核对 | 755条，最大差0 | 740条，最大差0 |

新增缓存位于`OLD5/{gemma3,medgemma}/mega/paths/nodes.jsonl`及其`node_*.npz`。数组包含62×32×128全头激活和5376维最终残差；对应metadata在模型路径、分词器模板、答案token、BF16/FP32评分、激活位置和attention实现上与当前记录一致。原先3,100条全头视图缓存中，当前选中头及raw分数也已逐文件检查可读、有限。

审计只在内存中使用冻结参数检查分支能否由现有缓存走完，输出覆盖数量及缺失ID，没有生成新实验预测、正确率或区间。所有实际访问节点均有兼容缓存。E6受控单元按本次指定的806条自然路径及已有配对指标验收。

执行时需要按访问节点读取缓存的CPU包装脚本。现有`evaluate_model(test)`会先要求完整候选节点集合；直接调用会把未访问节点也当作缺失。计划复用其冻结评分和统计函数，使用`cache_maps/e6_*_additional_fullhead_nodes.jsonl`补充索引，再遍历实际分支。两模型各11,662/11,651个未缓存自然候选节点均未被这三种冻结评分访问，无需为本单元补前向。

| 关键缓存/部分结果 | 字节 | SHA256 |
| --- | ---: | --- |
| `R/models/gemma2/evaluation/mega_official/shard0/path_predictions.jsonl` | 5,368,962 | `8c11526a7d3a7dece08bcb3bbf611d467ec1512e8bb80758b3da0420676cac2a` |
| `R/models/gemma2/evaluation/mega_official/shard0/nodes.jsonl` | 4,722,474 | `d94e80845bfec2acaf064af580451bea526894cf2355b44d668031f8a94c4d89` |
| `OLD5/gemma3/mega/paths/nodes.jsonl` | 2,364,556 | `22ec85f2592d0ee3751cf3c6f0b39c5febce25c5286a8362676520177ed67b6d` |
| `OLD5/medgemma/mega/paths/nodes.jsonl` | 2,406,472 | `0aada86727d5ba46fdd903ea491541a24e7ca3253f702263a9f0d3ab3273bce3` |

### 3.2 官方评价的确切接续范围

| 模型 | 剩余路径 | 首批路径起点唯一UID | 缺失金标准节点唯一前向 | 可复用未完成路径节点 | 旧缓存缺当前头数 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B-IT | 7458 | 7458 | 45627 | 0 | 9 |
| Gemma-2-9B-IT | 5079 | 5078 | 30572 | 6 | 7 |
| Gemma-3-27B-IT | 7458 | 7458 | 45627 | 0 | 11 |
| MedGemma-27B-Text-IT | 7458 | 7458 | 45627 | 0 | 2 |

沿用现有官方评价的金标准自然节点及动态路径统计：四任务共有167,453个已知缺失金标准节点请求；路径首次访问节点去重为27,452个。分支产生的额外节点按需追加，三种方法共享同一次原模型前向及当前51头。Gemma2已有15,363个当前节点缓存，其中6个属于尚未完成的下一条路径。

已逐个检查旧官方索引引用的特征容器：Qwen 45,098、Gemma2 46,408、Gemma3 46,232、MedGemma 46,100个官方自然输入缓存均采用历史选头数组；缺当前头分别9、7、11、2个。可保留历史raw及旧协议比较，当前修正需要补取所需头。缺头列表和原参数文件哈希见`e6_official_queue_audit.json`。

缺失路径ID保存在`missing_ids/e6_{model}_{scope}_{method}.jsonl`；已完成单元的缺失表为空，重复ID列表为空。官方起点、金标准节点、完整候选节点分别保存在`e6_*_official_initial_root_nodes.jsonl`、`e6_*_official_initial_gold_nodes.jsonl`、`e6_*_official_candidate_nodes.jsonl`。完整候选数为上限，不是一次性前向清单。

## 4. E5逐项状态与节点去重

| 方法/种子 | 状态 | 自然路径正确/806 | 已保存干扰路径 | 缓存可走完/2418 | 首批缺失UID |
| --- | --- | --- | --- | --- | --- |
| lora_seed11 | PARTIAL_CURRENT | 756 | 0 | 43 | 2375 |
| lora_seed22 | PARTIAL_CURRENT | 774 | 0 | 45 | 2373 |
| lora_seed33 | PARTIAL_CURRENT | 737 | 0 | 45 | 2373 |
| lofit_seed11 | PARTIAL_CURRENT | 768 | 0 | 45 | 2373 |
| lofit_seed22 | PARTIAL_CURRENT | 769 | 0 | 45 | 2373 |
| lofit_seed33 | PARTIAL_CURRENT | 769 | 0 | 45 | 2373 |
| loreft_seed11 | PARTIAL_CURRENT | 764 | 0 | 45 | 2373 |
| loreft_seed22 | PARTIAL_CURRENT | 772 | 0 | 45 | 2373 |
| loreft_seed33 | PARTIAL_CURRENT | 650 | 0 | 44 | 2374 |
| leace_pair | PARTIAL_CURRENT | 767 | 0 | 45 | 2373 |
| iti_style | PARTIAL_CURRENT | 321 | 0 | 42 | 2376 |
| sea_adaptation | PARTIAL_CURRENT | 768 | 0 | 45 | 2373 |
| strong_evidence_prompt | PARTIAL_CURRENT | 772 | 0 | 45 | 2373 |
| CC | PARTIAL_CURRENT | 769 | 0 | 45 | 2373 |

上述14行（13个前向方法加CC）均已有完整自然路径、2,339个配对记录、A/P与C/F指标、纠正/新增错误、来源95%区间和自然路径相对raw的配对差值区间。自然路径均为806唯一case/40来源，重复、执行错误和正确标记差异均为0。配对总体保持A/P=2,293、C/F=40，F0=6另列。无需重复计算已经完备的自然路径前向。

13行所需的干扰候选UID联合去重后为50,022个，每行已保存的修正分数覆盖2,283个，缺失候选UID联合为47,739个。raw缓存覆盖50,022/50,022，新增raw评分任务为0。按当前分支推到第一个未知节点后，13行第一批请求涉及2,387个唯一UID、30,855个“方法/种子×UID”评分请求。

LEACE-pair、ITI-style和SEA的现有实现均先计算原模型提示前缀，再对最后token施加方法修改。三方法首批共7,122个评分请求涉及2,385个唯一输入；安排同一worker按UID共同处理，可共享提示前缀缓存，再分别执行方法修改后的最后token前向。Engine.run会复制前缀缓存，三方法的修正分数分别保存。LoRA、LoFiT、LoReFT及强证据提示继续使用各自适配器或提示对应的计算。源码依据和请求数见`e5_shared_prefill_plan.json`。

同一方法/种子的重复UID只计算一次。不同方法或随机种子的修正结果独立保存；47,739个输入不能当成所有方法共用的一次修正前向。全候选上限为620,607个方法/种子节点请求，实际执行沿树逐批扩展，访问结束即停止。精确清单见`missing_ids/e5_unique_candidate_nodes.jsonl`及每行的`e5_*_first_missing.jsonl`。

SEA另需最后Transformer层输出、最终norm之前的原始残差。已检查E1原始特征、INLP的原始残差和E5 truth_features；47,739个缺修正分数的候选节点中，可直接使用的最终残差为0。INLP test的3,100份残差对应已有视图，未补足这些缺失节点。已有attention heads不能替代SEA最终残差；SEA沿实际访问节点补取残差并复用冻结算子。

CC冻结offset=-1.0324354956866095；本轮复查3,100份已有输入分数，均精确等于raw+offset。全部候选raw可用，干扰路径与统计转为CPU。原`e1_prompt_controls.py:main()`会加载Engine，CPU补算应使用独立包装脚本。

### 4.1 修复后参数与11行已完成外部方法

LoRA复用学习率0.001；LoFiT/LoReFT复用0.0001；三方法均保留11、22、33种子和4,637个训练端点。LoReFT正式适配器读取结果有限且非零，记录的旋转参数更新L2为0.457101、0.463157、0.468976，旋转范数约1。旧零旋转适配器和旧失败smoke记为FAILED，不进入接续计划。PSR seed22已有4,475个训练步、参数变化L2=0.5425931811；其预测与raw相同属于已完成结果。

| 新增方法/种子 | 计算状态 | 自然正确/806 | 干扰正确/2418 | 剩余 |
| --- | --- | --- | --- | --- |
| CAA-E1-task-adapted | PARTIAL_CURRENT：前向完整 | 766 | 2332 | CPU补两总体相对raw配对差值区间 |
| DoRA-E1-SFT-seed11 | PARTIAL_CURRENT：前向完整 | 786 | 2351 | CPU补两总体相对raw配对差值区间 |
| DoRA-E1-SFT-seed22 | PARTIAL_CURRENT：前向完整 | 781 | 2334 | CPU补两总体相对raw配对差值区间 |
| DoRA-E1-SFT-seed33 | PARTIAL_CURRENT：前向完整 | 783 | 2345 | CPU补两总体相对raw配对差值区间 |
| INLP-task-adapted | PARTIAL_CURRENT：前向完整 | 769 | 2339 | CPU补两总体相对raw配对差值区间 |
| PSR-MSE-E1-task-adapted-seed11 | PARTIAL_CURRENT：前向完整 | 776 | 2343 | CPU补两总体相对raw配对差值区间 |
| PSR-MSE-E1-task-adapted-seed22 | PARTIAL_CURRENT：前向完整 | 768 | 2339 | CPU补两总体相对raw配对差值区间 |
| PSR-MSE-E1-task-adapted-seed33 | PARTIAL_CURRENT：前向完整 | 777 | 2343 | CPU补两总体相对raw配对差值区间 |
| RePS-E1-task-adapted-seed11 | PARTIAL_CURRENT：前向完整 | 474 | 1372 | CPU补两总体相对raw配对差值区间 |
| RePS-E1-task-adapted-seed22 | PARTIAL_CURRENT：前向完整 | 609 | 1793 | CPU补两总体相对raw配对差值区间 |
| RePS-E1-task-adapted-seed33 | PARTIAL_CURRENT：前向完整 | 60 | 191 | CPU补两总体相对raw配对差值区间 |

11行均有自然806、干扰2418、40来源，方法自身的指标区间已保存。新增差值区间必须按同一来源抽样索引同时重算方法和raw，再相减；不采用两套独立抽样的区间相减。对应输入为`EXT/evaluation/test/{method}/path_predictions.jsonl`和`Q/evaluation/test/path_predictions.jsonl`。

温度、E1端点截距、仿射、监督线性读出已有完整自然/干扰评价，保存在`E5/output_controls_e1/evaluation/test/`。原模型、固定缩放和E1参照均复用。完成产物本地E1协议一致；上游revision待追溯的项标记NEEDS_VALIDATION，不加入GPU队列。

## 5. E7二十个目标单元

| 模型 | 总体 | 目标分母 | raw/独立截距/E1正确数 | 视图数 | 来源簇 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B-IT | nli4ct | 500 | 378 / 待CPU / 375 | 2864 | 113 | PARTIAL_CURRENT |
| Qwen3-8B-IT | nli4ct_official5500 | 5500 | 4073 / 待CPU / 4043 | 5500 | 500 | PARTIAL_CURRENT |
| Qwen3-8B-IT | medpic | 188 | 70 / 66 / 66 | 2820 | 150 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Qwen3-8B-IT | medpic_official467 | 467 | 188 / 待CPU / 183 | 7005 | 376 | PARTIAL_CURRENT |
| Llama-3.1-8B-Instruct | nli4ct | 500 | 347 / 342 / 343 | 2864 | 113 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Llama-3.1-8B-Instruct | nli4ct_official5500 | 5500 | 3883 / 3941 / 3923 | 5500 | 500 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Llama-3.1-8B-Instruct | medpic | 188 | 59 / 63 / 63 | 2820 | 150 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Llama-3.1-8B-Instruct | medpic_official467 | 467 | 144 / 149 / 147 | 2335 | 376 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-2-9B-IT | nli4ct | 500 | 359 / 368 / 366 | 2864 | 113 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-2-9B-IT | nli4ct_official5500 | 5500 | 4101 / 4122 / 4109 | 5500 | 500 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-2-9B-IT | medpic | 188 | 24 / 23 / 25 | 2820 | 150 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-2-9B-IT | medpic_official467 | 467 | 78 / 70 / 75 | 2335 | 376 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-3-27B-IT | nli4ct | 500 | 374 / 382 / 381 | 2864 | 113 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-3-27B-IT | nli4ct_official5500 | 5500 | 4280 / 4246 / 4250 | 5500 | 500 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-3-27B-IT | medpic | 188 | 61 / 46 / 48 | 2820 | 150 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| Gemma-3-27B-IT | medpic_official467 | 467 | 157 / 120 / 124 | 2335 | 376 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| MedGemma-27B-Text-IT | nli4ct | 500 | 364 / 366 / 366 | 2864 | 113 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| MedGemma-27B-Text-IT | nli4ct_official5500 | 5500 | 4171 / 4172 / 4176 | 5500 | 500 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| MedGemma-27B-Text-IT | medpic | 188 | 74 / 77 / 74 | 2820 | 150 | NEEDS_VALIDATION（计算已完整，追溯待补） |
| MedGemma-27B-Text-IT | medpic_official467 | 467 | 199 / 201 / 197 | 2335 | 376 | NEEDS_VALIDATION（计算已完整，追溯待补） |

所有目标总体已有前向。四个非Qwen模型的16个单元均有三方法逐例及区间，逐例截距恒等式核查通过。Qwen MedPIC188的独立截距和95%区间保存在`D/predictions/medpic_test/`，只补版本说明。Qwen其他三单元从已有raw和冻结源截距补统计，不重新加载主干。

NLI500包含500原声明和共2,864视图；NLI5500含5,500声明。NLI5500旧`natural_*`部分字段的分母是500，完整总体正确数已按5,500逐例重计。MedPIC188为2,820含扰动选项视图；Qwen MedPIC467保存7,005视图，其中原始选项2,335；其他模型的MedPIC467保存原始2,335选项。

**统计口径待用户确认：** NLI500现有113个“临床试验或试验组合”簇，NLI5500现有500个“原始声明及其变体”簇。可分别沿用并明确标注，也可在确认后统一到临床试验/组合。两项新NLI区间不在确认前计算。MedPIC188/467分别为150/376来源；467含开发题，保留描述性评价标注。目标测试标签从未用于本计划的截距拟合或选参。

## 6. E8 Base/IT追溯

| 比较 | 状态 | Base/IT头数 | 交集/并集 | Jaccard | 剩余 |
| --- | --- | --- | --- | --- | --- |
| qwen | NEEDS_VALIDATION | 51/51 | 20/82 | 0.2439024390 | 模型/分词器上游revision追溯；不拟合 |
| gemma2 | NEEDS_VALIDATION | 51/51 | 36/66 | 0.5454545455 | 模型/分词器上游revision追溯；不拟合 |

两组比较的头集合、M2子空间、谱与有效秩产物已存在，比较文件与归档一致。本轮重计头交集与并集相符。Qwen Base确认固定revision；Qwen IT及Gemma2两侧不可变revision未查明，明列在`model_identity.json`。未找到同一训练链的SFT/RL中间检查点，结论保持Base/IT端点比较。阶段因果归因及新增中间检查点研究记为OPTIONAL，本轮无任务。

| 比较文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `R/base_it/qwen/comparison.json` | 187,972 | `06753eaf4fe14c296c6ca2642cdec1c34105d5418d556688908904f863b22f61` |
| `R/base_it/gemma2/comparison.json` | 67,330 | `dc28b6703e3c8b92ac764be2a70320f68e4b4d6a86f2b1d62524eb7465b88de9` |

## 7. A–D分类：复用、归档、接续与无有效结果

### A. 计算已完成，无需再运行

五IT和两Base的E1拟合/开发/冻结；Qwen/Llama/Gemma2的受控806；Llama、Qwen Base、Gemma2 Base的官方7458；E7所有模型前向及四个非Qwen模型的全量统计；E8两组比较；E5所有已训练适配器、自然路径、四个输出控制和11行新增方法完整前向。版本缺项统一保留NEEDS_VALIDATION，已完成计算不得因此重跑。

### B. 已完成但尚未上传的产物

Qwen NLI5500与MedPIC467的7份关键逐例/指标文件合计51,423,608字节，服务器存在，指定归档未包含这些完整逐例。其中raw/E1计算产物为COMPLETE_NOT_ARCHIVED，整个三方法单元因缺独立截距统计仍为PARTIAL_CURRENT。建议归档到`reports/aster_final_gap_completion_20261114/e7/qwen/{nli5500,medpic467}/`，先补齐三方法及统计再统一提交。

| 服务器文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `F/evaluation/nli4ct_official5500/predictions.jsonl` | 2,783,921 | `7dad20fec338410484aa8fba90845788eadb22ed72ab6112a72fc8f822fc3ed8` |
| `F/evaluation/nli4ct_official5500/pair_predictions.jsonl` | 24,653,771 | `06c4829da19ab146230a48eff32f16f438ae779db52b6058a446990c4f397bf5` |
| `F/evaluation/nli4ct_official5500/metrics.json` | 105,160 | `580ad399a528379e2af30c701f133ada46f2c49710ff59ebd5b829d658cb8880` |
| `F/evaluation/medpic_official467/predictions.jsonl` | 3,052,163 | `d9c6b38db2a8d5fb5217bd47877d992d2f24b983947d65130fc8b4aa25cf909d` |
| `F/evaluation/medpic_official467/pair_predictions.jsonl` | 19,204,775 | `779cee5d663c0c4d2a82e0ef4f62f1425a8b21485cc31b22ccf032dfa582baee` |
| `F/evaluation/medpic_official467/question_predictions.jsonl` | 1,478,527 | `d62782692e246eb4b6b9678ff6db423ea91ca5ada266e36a11c953dd7d710a9c` |
| `F/evaluation/medpic_official467/metrics.json` | 145,291 | `c0f966787aa6e6d3b643516f6c14783c9dd2b69e2d5ff7b591b4599d1700cc69` |

SEA算子、DoRA三个适配器、选中INLP投影及Qwen Base下载清单也有服务器原件可补充参数归档；大缓存保留服务器，并提交清单和引用。Gemma2官方7,137行部分路径预测尚未归档，应先保留，再与接续结果按ID合并。新发现的两27B全头路径缓存同样保留原位，通过本轮映射清单复用。

| 参数原件 | 字节 | SHA256 |
| --- | ---: | --- |
| `E5/representation_e1_plan/sea_adaptation/operator.pt` | 134,219,629 | `dc423b9d09b9587ad802742063e0071f46e905122312df4d801d95a6d456cb91` |
| `EXT/dora/full/seed11/adapter/adapter_model.safetensors` | 16,101,768 | `bbb12c364a25edc9649011b27ee2246826da5476ba03eab3eff7a3eb30923198` |
| `EXT/dora/full/seed22/adapter/adapter_model.safetensors` | 16,101,768 | `ea89e8a8e163e9440a59037796ba837f9c6ad73be6702d1371b8ffbe8503c392` |
| `EXT/dora/full/seed33/adapter/adapter_model.safetensors` | 16,101,768 | `f3be46a4b306236d4b5b463502412d879d6547ffebd644c284ea75067b7ed513` |
| `EXT/inlp/iterations_1/projection.npz` | 62,591,408 | `3e507afdbf930abf2516422916823f431049ced219bc1d9a400d3e12be92125e` |
| `/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/qwen_base_download_manifest_v2.json` | 4,556 | `06f4f65124120a2a2130e697807616130c9e68005033f434cd104b010fdedb63` |

### C. 部分完成，接续即可

Gemma2官方缺5,079路径；两27B受控缺当前预测导出与统计，兼容缓存已足；E5十三行缺干扰路径结果，CC缺CPU导出；11行新增方法缺相对raw差值区间；Qwen三个迁移单元缺独立截距统计。缺失UID、路径ID、可复用缓存及输出位置已写入`MINIMAL_EXECUTION_PLAN.json`和`missing_ids/`。

### D. 全面搜索后未发现当前有效结果

Qwen IT、Gemma3、MedGemma的当前E1官方7458评价：NOT_FOUND_AFTER_EXHAUSTIVE_SEARCH。三个模型各自有HISTORICAL_ONLY完整旧协议结果，旧M3/头集合不能直接充当当前修正结果。E5十三行的完整干扰路径预测未发现；其方法整体已有当前训练与自然结果，按PARTIAL_CURRENT接续。此状态表示未找到当前有效产物，不推断服务器过去从未尝试运行。

## 8. E–F最小执行计划、资源与验收

### E. CPU任务

| 编号 | 任务 | 输入 | 实施与预期输出 |
| --- | --- | --- | --- |
| C01 | Gemma3 受控806 | `R/models/gemma3/models/E1/full`；`AUDIT/cache_maps/e6_gemma3_additional_fullhead_nodes.jsonl`；`R/models/gemma3/data/reused_scores.jsonl` | 新建 cpu_rebuild_controlled_from_cache.py；复用 evaluation.py 的冻结线性评分、原树遍历及来源统计函数，按实际访问节点读取缓存。输出：`reports/aster_final_gap_completion_20261114/e6/gemma3/test/` |
| C02 | MedGemma 受控806 | `R/models/medgemma/models/E1/full`；`AUDIT/cache_maps/e6_medgemma_additional_fullhead_nodes.jsonl`；`R/models/medgemma/data/reused_scores.jsonl` | 同C01，使用MedGemma自己的头、子空间、M3和截距。输出：`reports/aster_final_gap_completion_20261114/e6/medgemma/test/` |
| C03 | CC干扰路径 | `E5/prompt_controls_e1/cc_parameters.json`；`Q/cache`；`Q/data/path_interference_views.jsonl`；`/home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl` | 新建 cpu_complete_cc.py；raw+冻结offset后按树遍历；复用 bootstrap_summary 的来源配对抽样，不调用原prompt main()。输出：`reports/aster_final_gap_completion_20261114/e5/CC/` |
| C04 | 新增方法11行配对差值 | `EXT/evaluation/test`；`Q/evaluation/test/path_predictions.jsonl` | 新建 cpu_paired_path_ci.py；按(case_id,variant,source_id)配对，再对40来源共同抽样。输出：`reports/aster_final_gap_completion_20261114/e5/external_paired_ci/` |
| C05 | Qwen NLI500 | `Q/evaluation/nli4ct/predictions.jsonl`；`D/configs/independent_intercepts_reused.json` | 新建 cpu_qwen_transfer.py；使用冻结b0=3.0993948961894406，不拟合任何参数；聚类口径待确认。输出：`reports/aster_final_gap_completion_20261114/e7/qwen/nli500/` |
| C06 | Qwen NLI5500 | `F/evaluation/nli4ct_official5500`；`D/configs/independent_intercepts_reused.json` | 同C05；全量5500单独计算，避免natural字段的500分母；聚类口径待确认。输出：`reports/aster_final_gap_completion_20261114/e7/qwen/nli5500/` |
| C07 | Qwen MedPIC467 | `F/evaluation/medpic_official467`；`D/configs/independent_intercepts_reused.json` | 同C05；使用已冻结的原整题集合解码；376来源共同抽样。输出：`reports/aster_final_gap_completion_20261114/e7/qwen/medpic467/` |
| C08 | E5十三行接续统计 | `E5`；`G05–G17新增节点分数` | 完成路径后复用现有自然/配对预测；补干扰路径、配对来源区间和纠正/新增错误。输出：`reports/aster_final_gap_completion_20261114/e5/` |
| C09 | E6官方合并统计 | `G01–G04新节点/路径`；`R/models/gemma2/evaluation/mega_official/shard0` | 复用 official_mega.py 的路径和自然节点汇总；按来源共同抽样。输出：`reports/aster_final_gap_completion_20261114/e6/` |

CPU工作预计1–3小时，按8–16线程、约8–16 GiB内存安排。使用现有2,000次来源bootstrap配置和既有种子；每个对比共享抽样来源索引。全部CPU补实验同样等待用户批准，本轮仅计数、检查兼容性及生成清单。

### F. GPU任务

| 编号 | 任务 | 模型加载/激活 | GPU/显存 | 预计GPU小时 | 结果目录 |
| --- | --- | --- | --- | --- | --- |
| G01 | Qwen3-8B-IT 官方接续 | 当前51头；同一次前向供raw/独立截距/ASTER使用 | 1张，18–28 GiB | 3–4 | `reports/aster_final_gap_completion_20261114/e6/qwen/mega_official/` |
| G02 | Gemma-2-9B-IT 官方接续 | 当前51头；同一次前向供raw/独立截距/ASTER使用 | 1张，26–40 GiB | 5.5–7 | `reports/aster_final_gap_completion_20261114/e6/gemma2/mega_official/` |
| G03 | Gemma-3-27B-IT 官方接续 | 当前51头；同一次前向供raw/独立截距/ASTER使用 | 1张，54–66 GiB | 8–11 | `reports/aster_final_gap_completion_20261114/e6/gemma3/mega_official/` |
| G04 | MedGemma-27B-Text-IT 官方接续 | 当前51头；同一次前向供raw/独立截距/ASTER使用 | 1张，54–66 GiB | 7.5–10 | `reports/aster_final_gap_completion_20261114/e6/medgemma/mega_official/` |
| G05–G17 | E5十三行干扰路径 | 复用Qwen主干驻留与冻结参数；方法/种子分数独立，SEA补最终残差 | 每任务1张，18–28 GiB | 合计8–14 | `reports/aster_final_gap_completion_20261114/e5/{method}/` |

允许资源固定为GPU **0、1、5、6**，均为A100-SXM4-80GB。建议首轮分配：Qwen官方→0，Gemma2官方→1，Gemma3官方→5，MedGemma官方→6；完成后释放卡给E5队列。只使用一个统一调度器，每卡一个任务，最多四个已协调任务。旧队列允许列表含GPU4，新增计划显式采用0/1/5/6，不继承GPU4。

总量约32–46 GPU小时；四卡协调执行预计12–18小时墙钟时间。估算来自已有记录：Qwen约0.18–0.25秒/节点、Gemma2部分官方约0.61秒/节点、Gemma3约0.60秒/节点、MedGemma约0.56–0.59秒/节点，E5原生自然路径任务约0.21–0.22秒/输入。时长包含合理的分支追加和I/O余量，不包含新训练或下载。实际访问节点数在推理过程中确定；候选全树UID不会全部预提交。

当前没有模型驻留GPU，因此首次GPU任务需加载现有本地检查点；后续同卡Qwen任务可连续复用主干。原始节点分数、当前参数、已完成分片与合法全头缓存均复用。GPU任务启动后确认稳定即停止持续监控，由队列保留日志和完成标记。

| 任务 | 完成验收 |
| --- | --- |
| C01 | raw/calibration/E1 各806个唯一自然路径、40来源，已有配对指标和来源配对95%区间；0新模型前向 |
| C02 | 同C01，使用MedGemma参数 |
| C03 | 2418干扰路径、806自然路径、40来源，纠正/新增错误、A/P、C/F与相对raw配对区间；0新模型前向 |
| C04 | 11行×自然/干扰两总体的差值点估计与95%区间；保留逐种子、方法自身既有区间及配对计数 |
| C05 | 500原声明、2864视图；三方法逐例、准确率/宏F1、原生配对指标、纠正/新增错误、95%区间 |
| C06 | 三方法各5500声明；完整统计、原生关系指标、配对差值区间及脱敏逐例归档 |
| C07 | 三方法各467原题/2335原始选项，保留7005含扰动视图；标注含开发题的描述性评价 |
| C08 | 每方法/种子自然806、干扰2418、40来源；A/P2293与C/F40，F0另列；相对raw来源配对95%区间 |
| C09 | 每模型三方法各7458路径、418来源；重复ID=0；缺失ID=0；保留45627金标准自然节点总体；失败记录计入分母 |
| G01–G04 | 逐例三方法各7458、418来源；原部分结果逐字节保留；无重复/缺失ID；原始/独立截距/E1同一分母；参数冻结与模板保持一致。 |
| G05–G17 | 每方法/种子自然806、干扰2418、40来源；同方法UID不重复前向；已修复适配器生效；追加节点与方法参数一一对应。 |
| 归档 | 压缩逐例分数/预测、配置、参数清单、来源配对95%区间、图表及报告；不公开原始临床提示、患者文本或带prompt的节点缓存；获准后再GitHub提交。 |

计划文件中所有新计算任务的准入状态均为PARTIAL_CURRENT或NOT_FOUND_AFTER_EXHAUSTIVE_SEARCH。NEEDS_VALIDATION版本事项为追溯任务，HISTORICAL_ONLY结果不重复报告为当前完成，OPTIONAL扩展不进入本队列。

## 9. 进程、队列及旧状态误判

快照时间：2026-11-14T05:17:38Z（北京时间13:17:38）。GPU0/1/5/6均无计算进程，显存占用记录为0。统一队列`completed_with_cancellations`，完成46、取消3、pending/running/failed/blocked均0；旧调度器PID 477956已不存在。

| 旧记录 | 本轮依据 | 正确处理 |
| --- | --- | --- |
| Gemma3/MedGemma run_status=stopped_by_user | 后续dev、冻结、四目标迁移逐例与指标均存在 | 这些已完成阶段不重跑；受控自然路径转CPU；官方当前评价继续列缺失。 |
| Llama官方shard status仍为running | done.json及三方法各7458完整逐例 | 逐例证据完整，按计算已完成登记。 |
| Qwen Base shard0仍为running | 五个resume分片并集与原分片完整覆盖7458，三方法无重复 | Base官方已完成；不再按旧分片状态重跑。 |
| Gemma2停止记录 | 2379完整路径及下一路径6个节点，未见其余分片 | PARTIAL_CURRENT，精确接续。 |
| LoReFT旧失败记录 | final有效性记录、三个非零旋转适配器及正式预测 | 旧适配器FAILED，修复后训练已完成。 |
| PSR seed22预测等于raw | 训练步及参数变化L2均存在 | 已完成，保留该种子。 |

旧状态文件原样保留。当前有效RUNNING实验任务为0。本轮没有恢复被取消的旧全流程，也没有创建活动GPU队列。

## 10. 各单元主要文件与SHA256

以下提供逐项结果入口；完整配置、冻结参数、模型版本、分片及缓存清单见`artifact_sha256_registry.json`，物理/解压内容对应关系见`publication_prediction_mapping.json`和`prediction_duplicate_groups.jsonl`。157份同路径文本/参数文件与发布树SHA256相同，83份已登记逐例文件与发布树解压内容相同；空文件不作为完成证据。

### E6现存路径文件

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `Q/evaluation/test/path_predictions.jsonl` | 6,539,369 | `32cfbc29e8d3f617bfd580c3923a6e5080b2638d6647b81f338d2e62f97bbf72` |
| `D/predictions/test/paths.jsonl.gz` | 236,582 | `2423c5762265422c12309e8014f4f071556a208ba9b8e4477ceacec412aa9b75` |
| `R/models/llama31/evaluation/test/path_predictions.jsonl` | 3,904,665 | `6511c2d84a7ca3b6f1b7779b2f9fa45511700e02dbd55e7c3a41947c26b8a827` |
| `R/models/llama31/evaluation/mega_official/shard0/path_predictions.jsonl` | 16,451,462 | `5fe6ddd9dce6ab8d93c85159f1c93e3ca409577b6eeb3d9fb0c913b09e8eb024` |
| `R/models/gemma2/evaluation/test/path_predictions.jsonl` | 3,947,793 | `fc08b483b6fae9d4f186a2f7f761559fe338502fba98075f4be49ee5b46450ac` |
| `R/models/gemma2/evaluation/mega_official/shard0/path_predictions.jsonl` | 5,368,962 | `8c11526a7d3a7dece08bcb3bbf611d467ec1512e8bb80758b3da0420676cac2a` |
| `R/models/qwen_base/evaluation/test/path_predictions.jsonl` | 3,378,973 | `45934b63b515b9c3156b4ce93c3d6a0e01b2651e702fcac713cae1e535e2ecef` |
| `R/models/qwen_base/evaluation/mega_official/shard0/path_predictions.jsonl` | 5,111,713 | `4c323f6e8f765149776f1ca149dbac41b90024365be2e3d1814c3bd0b1373d01` |
| `R/models/qwen_base/evaluation/mega_official/shard_resume0/path_predictions.jsonl` | 2,106,875 | `21b971a28f841228c69549a23fe462d76996a94e6ae3745c10cbd5f95577c82f` |
| `R/models/qwen_base/evaluation/mega_official/shard_resume1/path_predictions.jsonl` | 2,109,475 | `246793f3d1f93aa7e661cda8f3260a658cd7909de4027d0d3b7685cc6abf1dd0` |
| `R/models/qwen_base/evaluation/mega_official/shard_resume2/path_predictions.jsonl` | 2,106,957 | `a53244437a464f7fd2583b6821c043c3b39ab549479333790112062739cbe8d4` |
| `R/models/qwen_base/evaluation/mega_official/shard_resume3/path_predictions.jsonl` | 2,098,750 | `3c6458f498fcb7cfd7860a28970cc659048ea6d1178beacff881cf1fbaf37240` |
| `R/models/qwen_base/evaluation/mega_official/shard_resume4/path_predictions.jsonl` | 2,097,612 | `3add0e1cf5bc2e414016d7e3b6cd0e30f9836956f71afce326969c225b0c646b` |
| `R/models/gemma2_base/evaluation/test/path_predictions.jsonl` | 3,796,039 | `906404b91357cb1a479a237247ff1f963cf0b93807204e53051ca8dd78cb868c` |
| `R/models/gemma2_base/evaluation/mega_official/shard0/path_predictions.jsonl` | 16,358,481 | `c30cccb95d65cc7e47a87c5f40cb32b56b3ab857cd087cea9496946ec4ecded4` |

### E5每行自然路径与11行全路径文件

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `E5/native_supervised_e1/lora/lr0.001_seed11/test_path_predictions.jsonl` | 273,573 | `e1a004a888ca1f02466c0a72761e5f250d5864de3dec2df27e3d7d8d42a7edc5` |
| `E5/native_supervised_e1/lora/lr0.001_seed22/test_path_predictions.jsonl` | 274,251 | `b9bec423627947a787b619e9572d231c38bdc32899976502f384c1803dfa71a8` |
| `E5/native_supervised_e1/lora/lr0.001_seed33/test_path_predictions.jsonl` | 273,602 | `493f4f34a7f534311a1578c4393232a9e3bf541666706e1d546b50a75af7e191` |
| `E5/native_supervised_e1/lofit/lr0.0001_seed11/test_path_predictions.jsonl` | 273,980 | `e6d0a23326e09f209cb78e197159c5c4654c225657d61d552860c9fc63c361fe` |
| `E5/native_supervised_e1/lofit/lr0.0001_seed22/test_path_predictions.jsonl` | 274,035 | `025301b5121e7ca2da2778206d1c3385fa222e558367d211d55d7e16e8adb045` |
| `E5/native_supervised_e1/lofit/lr0.0001_seed33/test_path_predictions.jsonl` | 274,035 | `025301b5121e7ca2da2778206d1c3385fa222e558367d211d55d7e16e8adb045` |
| `E5/loreft_e1/lr0.0001_seed11/test_path_predictions.jsonl` | 273,654 | `adbdf02a051bc5237c14b3c30fe000ecc870e089c35be585b3cc2c67104e185a` |
| `E5/loreft_e1/lr0.0001_seed22/test_path_predictions.jsonl` | 274,176 | `6bfa34afe9e12c2f7b69b82ad231d01485f1daad1bfca1900e4348b935a1f63d` |
| `E5/loreft_e1/lr0.0001_seed33/test_path_predictions.jsonl` | 269,738 | `9477bba6ea13fb8652b46557929e5a74f911d6292f01dfb72813e8ab8a6f03a6` |
| `E5/representation_e1_plan/leace_pair/test_path_predictions.jsonl` | 274,036 | `1014ca9af7ae3e6a2ae4b98442e1f43b6249ece5e6eb296e6ee43b261b95193e` |
| `E5/representation_e1_plan/iti_style/test_path_predictions.jsonl` | 242,193 | `0817f79a62d33c51cc105078cb595da94bf1b0bcef0f6e2ffb721683fa5c4362` |
| `E5/representation_e1_plan/sea_adaptation/test_path_predictions.jsonl` | 273,979 | `1dc304a94520e851a48b532d61ba89fe386e53091134baa09d62c56542f77e10` |
| `E5/prompt_controls_e1/strong_evidence_prompt/test_path_predictions.jsonl` | 274,194 | `586a9cd083b7b1900d03d1816ea82bcaf730695865df7f2704a9b7d9faad7569` |
| `E5/prompt_controls_e1/CC/test_path_predictions.jsonl` | 274,007 | `e2c9fa7cb3b999e655a7f7737fcecde67ca5a841946e30b12401463a1b9e3bb4` |
| `EXT/evaluation/test/CAA-E1-task-adapted/path_predictions.jsonl` | 2,291,651 | `397871070f0068cd830eabf8dd96042c98e0cf9f6b4b27fb60f87f92fc61476e` |
| `EXT/evaluation/test/DoRA-E1-SFT-seed11/path_predictions.jsonl` | 2,288,595 | `b99cb37e13a6c770b31f6b34a37f954488b49d71ac2e3e64307610300dc6be37` |
| `EXT/evaluation/test/DoRA-E1-SFT-seed22/path_predictions.jsonl` | 2,287,540 | `3306ef27d7a2a665d73178920b61f93731baa16e9d22e062e19868a8b386f396` |
| `EXT/evaluation/test/DoRA-E1-SFT-seed33/path_predictions.jsonl` | 2,287,717 | `1e099bd3e8f54acc670712395294e84be7e70f062add6a1db937c78d44c99f00` |
| `EXT/evaluation/test/INLP-task-adapted/path_predictions.jsonl` | 2,285,260 | `be5e43fd61cfcf8e16b4d2eb68bdf6a77bacee77cfd1aef1b921d68c56c5133c` |
| `EXT/evaluation/test/PSR-MSE-E1-task-adapted-seed11/path_predictions.jsonl` | 2,327,409 | `cf8fcbc86ad3169d34f01130a612d173661f8eb5bf250f87a38a88d8fc4b8876` |
| `EXT/evaluation/test/PSR-MSE-E1-task-adapted-seed22/path_predictions.jsonl` | 2,327,118 | `9a209a8123a99412b0d00d9755e0f33243bb49f885377cf51c1e907c74e1d6ae` |
| `EXT/evaluation/test/PSR-MSE-E1-task-adapted-seed33/path_predictions.jsonl` | 2,327,484 | `927a07af7c3dadcccf1cfa4f1489df46a540141587592bd5de61b8896a96b07a` |
| `EXT/evaluation/test/RePS-E1-task-adapted-seed11/path_predictions.jsonl` | 2,257,649 | `0ce3258481b557be07dd0649ea9243f75bbdcae612ff098075b2c34f902dfb56` |
| `EXT/evaluation/test/RePS-E1-task-adapted-seed22/path_predictions.jsonl` | 2,285,223 | `cdd2e98dda0d8a847f78ee0839d41e825d4894c31fe2604453388a1f4c8ed8b5` |
| `EXT/evaluation/test/RePS-E1-task-adapted-seed33/path_predictions.jsonl` | 2,159,751 | `5ef1a44d90666a9bfff89ca70353060ba1b687169d5a3b029c01be0b7f01f5a4` |
| `E5/output_controls_e1/evaluation/test/path_predictions.jsonl` | 6,863,143 | `2b82b52537793b59761baa907e1fcf6ba754272aea88c3d580f688ff25b97a58` |

### E7每个总体的逐视图分数文件

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `Q/evaluation/nli4ct/predictions.jsonl` | 1,400,019 | `01008269843aa9354134a4a78328f85c26e9117bd6898ec3678a56991cab039c` |
| `F/evaluation/nli4ct_official5500/predictions.jsonl` | 2,783,921 | `7dad20fec338410484aa8fba90845788eadb22ed72ab6112a72fc8f822fc3ed8` |
| `Q/evaluation/medpic/predictions.jsonl` | 1,059,268 | `b12868665866b2ba5f3790ff88ca126471324e5bf900886048fe43478c8924b4` |
| `F/evaluation/medpic_official467/predictions.jsonl` | 3,052,163 | `d9c6b38db2a8d5fb5217bd47877d992d2f24b983947d65130fc8b4aa25cf909d` |
| `R/models/llama31/evaluation/nli4ct/predictions.jsonl` | 1,323,474 | `bfed44ec8915fc12265d5308133beee6f41acaaa7a81a66fbfde9162bb3cf50a` |
| `R/models/llama31/evaluation/nli4ct_official5500/predictions.jsonl` | 2,368,853 | `329b82bb646dc1bf33db4f89ccac4d209a1317a9c1e5e45e082d30a9bb71da23` |
| `R/models/llama31/evaluation/medpic/predictions.jsonl` | 988,601 | `bd605ed1ee7ee9339df1af961b0db29a18776a4f83b0389948e4a81a9c6bb4a4` |
| `R/models/llama31/evaluation/medpic_official467/predictions.jsonl` | 812,628 | `3f0d04b042a44340218b3af70d30fbf11b393c19147e7841cbac70614d1e86a3` |
| `R/models/gemma2/evaluation/nli4ct/predictions.jsonl` | 1,319,523 | `1ce31f0f2d1dac39215670aca68ffb09fd0daa4d0f5bd18e8fae9350ac040404` |
| `R/models/gemma2/evaluation/nli4ct_official5500/predictions.jsonl` | 2,361,701 | `ef8ae0233d910af4867c5404000db47ab6a625a99754ea2928ee2dea44ae3910` |
| `R/models/gemma2/evaluation/medpic/predictions.jsonl` | 980,943 | `cb2e10e22ce36062df60b0d7168513b9add5592067d0203ad724da9654da39c0` |
| `R/models/gemma2/evaluation/medpic_official467/predictions.jsonl` | 805,404 | `3af51aae24d75e69f93db840cd5d0a8861b46420a84232c7eb2aba8b00bd5211` |
| `R/models/gemma3/evaluation/nli4ct/predictions.jsonl` | 1,321,846 | `5b5cc9c1946543287eba5cc6160ef0a6f6245e7756e3b695139ab920f78e7556` |
| `R/models/gemma3/evaluation/nli4ct_official5500/predictions.jsonl` | 2,366,005 | `807f7ea701316998cf517149ad94b51ba5ad9166a225620a689ba9035b0d3823` |
| `R/models/gemma3/evaluation/medpic/predictions.jsonl` | 982,814 | `761a92fccdd27d0e7640939051a9a0272abddf2cbbdd2867209dd4a8cd0e5492` |
| `R/models/gemma3/evaluation/medpic_official467/predictions.jsonl` | 806,817 | `774168f5b087d8e5dbc8d0ff8ca220558fa7022900cf8c3b6e9ab10b6e61b79f` |
| `R/models/medgemma/evaluation/nli4ct/predictions.jsonl` | 1,320,784 | `dc2cd985146b148be2952e5a6854a104ea3600c4d32397f4d2705a1768d9fa49` |
| `R/models/medgemma/evaluation/nli4ct_official5500/predictions.jsonl` | 2,363,659 | `11d1a7232d4e6a9c04e3cc2952d637c4e0e5890861a7f45fa82e230153631908` |
| `R/models/medgemma/evaluation/medpic/predictions.jsonl` | 985,634 | `5a8ae5fb6afa6d6b33964b9c287da8dbca0f40374164242bf91c01f926a092bc` |
| `R/models/medgemma/evaluation/medpic_official467/predictions.jsonl` | 809,459 | `90713c725593ada0008f86316148eec7e32a9f2e683cc3f7f2753dc15e9ce266` |
| `D/predictions/medpic_test/questions.jsonl.gz` | 24,477 | `1bbf3eff0295ca4512e188136dc920be0d4ccd15b2ad09d29f19ea209acaa5ec` |
| `D/configs/independent_intercepts_reused.json` | 84,034 | `dfe1d172eee8da13d4dc332584fab0535341e3a279d3b4a4b9b372e242478ad1` |

历史路径文件逐片计数及SHA256在`historical_prediction_recount.json`；它们的评价范围可能同时包含420开发、806受控和7458官方，官方结果按`evaluation_group=official_test`区分。旧五模型15组、MedEinst混合来源模型和旧M3均保留原身份。

## 11. 审计结束时的结论与后续授权

| 必须回答的问题 | 结论 |
| --- | --- |
| 哪些确实未完成？ | 三个IT的当前E1官方完整评价、Gemma2剩余5079、两27B受控当前文件/统计、E5十三行干扰路径、CC导出、11行差值区间、Qwen三个迁移截距统计。 |
| 哪些只是没有上传？ | Qwen NLI5500与MedPIC467已有raw/E1逐例；部分冻结参数、下载清单及服务器缓存。整个三方法单元仍需补统计。 |
| 哪些被旧状态误判？ | 两27B后续开发/冻结/迁移、Llama官方、Qwen Base官方续片；LoReFT修复后训练与PSR seed22均已完成。 |
| 哪些只有部分结果？ | Gemma2官方2379/7458；E5十三行自然路径完整、干扰只覆盖少量可遍历缓存；Qwen三项迁移已有raw/E1。 |
| 哪些CPU即可？ | 两27B受控806、CC2418、11行相对raw区间、Qwen NLI500/NLI5500/MedPIC467独立截距统计、归档与E8追溯。 |
| 哪些需要GPU？ | 四个官方E6接续单元；E5十三行实际访问的未知方法分数，其中SEA需要最终残差。 |
| 哪些完全不应再运行？ | 全部既有M1/M2/M3与基线训练；E6完整评价；所有E7前向；两组E8拟合/比较；E5自然路径和11行完整前向；LoReFT旧失败流程、PSR seed22训练；既有E0–E4、E9–E11不扩大本轮范围。 |

**后续授权已完成。** 用户已批准开始实验，并确认保留NLI两种聚类口径。C01–C07已完成；执行进展与C08/C09状态见[实验框架](../../docs/ASTER/experiments.zh.md)。下表保留审计结束时的缺口判断，新增产物见CPU结果报告。

附：状态统一使用COMPLETE_CURRENT、COMPLETE_NOT_ARCHIVED、PARTIAL_CURRENT、HISTORICAL_ONLY、RUNNING、FAILED、NEEDS_VALIDATION、NOT_FOUND_AFTER_EXHAUSTIVE_SEARCH、OPTIONAL。COMPLETE_CURRENT用于计算和所需身份均已可确认的单元；其余已完成计算若缺上游版本，保留NEEDS_VALIDATION，不将追溯缺项等同为缺实验。

审计证据包：`AUDIT_EVIDENCE.zip`。包含完整报告、待批准计划、逐项计数、缺失UID/路径清单、缓存映射、哈希目录、搜索范围和审计脚本。体量较大的原始全盘文件清单保留在服务器；压缩包内的`EVIDENCE_INDEX.json`列出对应位置。所有附件仅含路径、ID、配置、计数和证据元数据，不含临床提示原文。
