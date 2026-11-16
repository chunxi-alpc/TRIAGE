# C_V6 补充验证实施说明

该补充预先固定20例：patient_routing-alder-00与equipment_routing-ebony-30两个家族，各保留全部10种情形。选择仅由数据划分与家族字典序决定，未读取模型输出。它是现有V6模型链路的工程/误差诊断，样本仅2家族，不支持泛化显著性结论。

## 冻结与真实执行路径

- 冻结现有 sources/care_v6_frozen.py；其相对导入的trialmatch_parser.py亦复制到sources并保存摘要。原始模块源文件未修改。
- 仅使用NEW及ADDITIONAL结构化规则，共80次逻辑原子调用；按患者全文、条件文本与criterion_kind精确去重为48个请求。
- 在用户指定gcx261014环境激活后，原先仅bitsandbytes的--no-deps下载速度过慢，随后终止本任务下载；经主代理确认，复制服务器n/.venv中既有bitsandbytes 0.50.2包及dist-info到本项目.deps，通过PYTHONPATH加载，不修改共享环境。71个源文件摘要保存在sources/bitsandbytes_local_dependency.json；在指定解释器下真实导入已通过。
- 原现有TrialMatchAIAdapter加载本地Phi-4 NF4 + 非零LoRA；judge_many batch=8、max_new_tokens=2048、greedy。既有inclusion/exclusion解析及Not Violated→U映射不变。
- 完整原始生成按8请求原子批次保存，可在同一源码/输入/设置契约下复用。每次批次成功后写原始输出，再更新状态。
- 删除Phi-4后加载BioLinkBERT，使用CAREModule1V6逐例重放已计算的真实TrialMatchAI输出并实际运行证据推断。
- NEW状态T给+1、F且旧规则有效给-1、U给0；NEW证据为missing/disputed或无冲突时强制0。旧有效性只来自提示中公开的结构化规则字段。V6自身action_gate也原样保存供审查。
- 删除模块一模型后加载Qwen3-8B Engine，使用冻结direction.pt与selected.json（父实验的alpha及layer），没有重新选参数。
- 最终只写results/c_full_predictions.jsonl，不向主predictions追加；Engine的元数据写入补充目录，避免覆盖主实验记录。

## 时间与输出

每条预测分别保留Qwen实际时间、模块一证据/逻辑实际时间、TrialMatchAI批次均摊时间、排除模型加载的端到端均摊时间。批次均摊不是孤立请求延迟；重复请求的时间进一步平均分配到其被复用的病例，避免重复计算总GPU开销。全程真实耗时与加载耗时单列。

所有原始请求、生成、结构、门控、源码摘要、版本和资源峰值都在results/c_full/。补充是否真正完成，以complete.json为准；本实施说明本身不代表实验完成。

## 必须保留的解释限制

C_V6运行现有完整V6预测/证据链，但没有访问用户缺失的原CARE上传文件，因此仍不能叫原CARE方法复现。NEW+ADDITIONAL规则为离线结构化输入，旧有效性已明确给出；没有验证自由文本规则解析、医学知识真实性或旧知识参数先验。

与C_reduced比较时，既包含模块一预测差异，也包含本次预先指定的证据门控：C_reduced在模板上直接解析，而C_V6在证据missing/disputed时关闭。不能把所有效应归因于单独的适用性准确率。

这是后加、结果无关选样的小样本补充，两个家族的差异只做描述，不给出看似稳定的临床或跨领域结论。


实际执行与验收已完成，见sources/c_full_review.md和results/c_full/complete.json。旧V6结构字段trialmatch_seconds是缓存重放延迟；真实Phi-4成本看原始生成时间与顶层批次均摊字段。
