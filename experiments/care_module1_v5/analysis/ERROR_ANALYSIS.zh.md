# CARE 模块一 V5 错误分析（192 条回归样本）

本报告只分析已完成的真实运行结果，不重算主指标、不修改推理配置。所有标签仅用于事后核对，不进入模型输入。V4 数据曾用于方法设计，因此本次仍属于探索性回归。

## 运行和计数

- 运行状态：completed；生成阶段使用 4 张 NVIDIA A100-SXM4-80GB（cuda:4/5/6/7），每卡独立加载 microsoft/phi-4 + majdabd33/trialmatchai-phi4-reasoning-lora，default adapter active。MedNLI 标签配置实际为 0=entailment、1=neutral、2=contradiction。
- 条件数：492；复用原单卡完成的 256 条，新增 236 条；总分配生成时间 6178.096 秒，4 卡生成阶段墙钟约 1313.020 秒。该时间是批次墙钟时间按 batch size=8 分摊到条件的估计，不是逐例独立延迟。
- TrialMatchAI 原始输出：接受 75/492（15.24%）；解析失败 417/492（84.76%）。其中 expected_exactly_one_target_criterion 312（63.41%）、invalid_json_response 75（15.24%）、generation_truncated 29（5.89%）、invalid_final_decision 1（0.20%）。
- 成功解析的目标条件标签：Met 44、Not Met 21、Unclear 10；没有将推理段中的关键词当成标签。

## 真实原始输出的结构问题

1. **非目标数组额外生成占位条件（主要问题）**：279 条失败输出的两个数组计数为目标数组 1 条、非目标数组 1 条；最常见非目标项是 “No specific exclusion criteria provided” + “Irrelevant”（232 条），另有 listed/mentioned 等近义占位项。模型按官方完整 trial schema 同时填充 inclusion 和 exclusion，而 CARE 的单条件请求只需要一个目标数组。
2. **模型使用 Markdown JSON 围栏**：75 条 invalid_json_response 在清理后均以 json 代码围栏开头，围栏后还带有说明文字；其后包含可识别的完整 JSON。诊断性去除围栏后，74 条存在恰好一个目标条件，1 条仍有目标数组结构问题。这属于接口解析兼容问题，不能归因于医学判断失败。
3. **真实截断**：29 条达到 2048 token 上限且没有 EOS。诊断性读取可见的目标数组时，29 条中有 20 条包含 Met、6 条包含 Not Met、3 条包含 Unclear；但因输出截断均被保守标为未知。
4. **其它数组扩张**：33 条 expected_exactly_one_target_criterion 的目标或非目标数组超过 1 条，包含模型自行复述多个规则条件；这属于模型输出格式不稳定。
5. **最终决定字段**：1 条 JSON 结构完整但 Final Decision 不在官方枚举内，归为格式错误。

因此，严格接口成功率很低，不能把 417 条解析失败直接报告为 TrialMatchAI 的医学判断错误。诊断性读取完整 JSON 后，312 条数组失败中有 284 条存在恰好一个目标条件；75 条围栏失败中有 74 条存在恰好一个目标条件。这只能说明信息可能存在，不能替代预先冻结的解析协议。

## 13 个出现纯 TrialMatchAI 已知状态的样本

“纯模型已知”指该规则的 TrialMatchAI-only 三值状态为 T/F，而不是指最终 CARE 结果正确。13 条中 12 条至少有一个条件级解析失败；这 13 条中仅 9 条的纯模型规则状态与 gold 一致。192 条规则中，188 条至少含一个条件级解析失败，只有 4 条的全部条件成功通过接口。因此，大量 U 与格式拒绝直接相关，而不是模型明确给出了 Unclear。唯一没有解析失败、但最终变为未知的代表是 C11_counterfactual：两个 Met 原始标签均保留，但数值观察缺失且存在冲突，代码按三值逻辑保守输出 U。

|样本|gold|TrialMatchAI-only|TrialMatchAI+数值|最终|代表原因|
|---|---:|---:|---:|---:|---|
|C02_contradiction|U|T|U|U|一条件解析失败；证据冲突|
|C11_original|U|F|U|U|一条件解析失败；缺失/数值冲突|
|C11_counterfactual|T|T|U|U|无解析失败；数值观察缺失、模型证据冲突|
|C12_counterfactual|F|F|U|U|一条件 JSON 解析失败|
|C17_counterfactual|F|F|F|F|另一条件数组占位导致解析失败，但数值逻辑仍完成|
|C21_contradiction|U|F|U|U|两个条件解析失败；证据冲突|
|C24_counterfactual|F|F|F|F|一条件截断；另一条件可判定|
|C26_multiple_values|F|F|F|F|一条件数组失败；另一条件保留 Not Met|
|C27_multiple_values|T|T|T|U|4/5 条条件解析失败；证据状态未知|
|C28_original|T|T|T|T|3 条条件解析失败，但证据支持其余条件|
|C28_partial_missing|T|T|T|T|2 条条件解析失败；其余证据支持|
|C28_contradiction|U|T|U|U|一条件解析失败；冲突未解决|
|C31_original|F|F|F|F|2 条条件解析失败；其余条件足以得到 F|

## 失败样本摘要

以下8条均为最终判定失败；只公开规则条件、标签、状态和简短原因。诊断性标签不进入主指标。

|样本|条件|gold|纯TrialMatchAI|加数值|最终|主要原因|
|---|---|---:|---:|---:|---:|---|
|C01_multiple_values|c1,c2|T|U|U|U|两个条件的模型数组均失败；首条件MedNLI反驳，第二条件出现数值跨阈值/多值未知，规则最终U。|
|C05_negation|c1,c2,c3|T|U|U|U|围栏、截断、数组失败并存；三个分期条件均未得到程序可用的数值观察，最终U。|
|C08_original|c1,c2,c3,c4,c5,c6,c7|T|U|U|F|术前分期数值观察缺失；未接受的用药条件由MedNLI判F，AND逻辑使最终错误拒绝F。|
|C09_original|c1,c2,c3|T|U|U|U|三条模型输出均因JSON围栏失效；MedNLI分别支持，但数值观察程序均未抽取到可用数值，最终U。|
|C11_counterfactual|c1,c2|T|T|U|U|两个目标条件均成功输出Met；数值观察缺失且证据关系未知，程序保守输出U；纯模型原本正确。|
|C20_original|c1|T|U|U|F|单条件生成达到2048无EOS而失败；MedNLI给contradiction并解未知为F，造成错误拒绝。|
|C22_original|c1,c2,c3|T|U|U|F|两个Met条件中一条与证据冲突；另一未解析条件由MedNLI判F，最终AND错误拒绝F。|
|C27_multiple_values|c1,c2,c3,c4,c5|T|T|T|U|五条件有四条解析失败；仅有的Met也遇支持/反驳冲突，OR结构最终U。|

逐条件证据（accepted raw label为正式parser接受标签；诊断标签只从完整目标JSON读取）：

- C01_multiple_values：c1 [arsenic trioxide exposure before the first relapse is negative] accepted=None, diagnostic=Met, raw=U, final=F, reason=expected_exactly_one_target_criterion; c2 [months from initial therapy completion to first relapse is less than 6 months] accepted=None, diagnostic=Not Met, raw=U, final=U, reason=expected_exactly_one_target_criterion
- C05_negation：c1 [original pathological T stage is at least 1] accepted=None, diagnostic=Met, raw=U, final=U, reason=invalid_json_response; c2 [original pathological T stage is at most 2] accepted=None, diagnostic=Met, raw=U, final=U, reason=generation_truncated; c3 [original pathological N stage is 0] accepted=None, diagnostic=Met, raw=U, final=U, reason=expected_exactly_one_target_criterion
- C08_original：c1 [preoperative clinical T stage is at least 1] accepted=None, diagnostic=Met, raw=U, final=U, reason=expected_exactly_one_target_criterion; c2 [preoperative clinical T stage is at most 3] accepted=Met, diagnostic=Met, raw=T, final=U, reason=numeric_observation_missing; c3 [preoperative clinical N stage is 0] accepted=None, diagnostic=Met, raw=U, final=U, reason=invalid_json_response; c4 [chemotherapy received before breast surgery is negative] accepted=None, diagnostic=Met, raw=U, final=F, reason=expected_exactly_one_target_criterion; c5 [positive sentinel lymph node count is at least 1] accepted=None, diagnostic=Met, raw=U, final=T, reason=expected_exactly_one_target_criterion; c6 [positive sentinel lymph node count is at most 2] accepted=None, diagnostic=Met, raw=U, final=T, reason=expected_exactly_one_target_criterion; c7 [whole breast radiotherapy planned for this patient is positive] accepted=Unclear, diagnostic=Unclear, raw=U, final=T, reason=mednli_resolved_unknown
- C09_original：c1 [positive axillary lymph node count is 0] accepted=None, diagnostic=Met, raw=U, final=U, reason=invalid_json_response; c2 [invasive tumor diameter in centimeters is at most 5 cm] accepted=None, diagnostic=Met, raw=U, final=U, reason=invalid_json_response; c3 [closest resection margin in millimeters is at least 1 mm] accepted=None, diagnostic=Met, raw=U, final=U, reason=invalid_json_response
- C11_counterfactual：c1 [preoperative clinical N stage is at least 1] accepted=Met, diagnostic=Met, raw=T, final=U, reason=numeric_observation_missing; c2 [post-neoadjuvant pathological N stage is 0] accepted=Met, diagnostic=Met, raw=T, final=U, reason=numeric_observation_missing
- C20_original：c1 [malignant cells in latest CSF cytology is negative] accepted=None, diagnostic=Met, raw=U, final=F, reason=generation_truncated
- C22_original：c1 [thyroid lobectomy margin involvement is negative] accepted=Met, diagnostic=Met, raw=T, final=T, reason=trialmatch_retained; c2 [contralateral thyroid lesion on postoperative imaging is negative] accepted=Met, diagnostic=Met, raw=T, final=U, reason=unresolved_model_evidence_conflict; c3 [suspicious cervical lymph nodes on postoperative imaging is negative] accepted=None, diagnostic=Not Met, raw=U, final=F, reason=invalid_json_response
- C27_multiple_values：c1 [gross positive thyroid lobectomy margin is positive] accepted=None, diagnostic=Met, raw=U, final=U, reason=expected_exactly_one_target_criterion; c2 [gross extrathyroidal extension is positive] accepted=None, diagnostic=Not Met, raw=U, final=U, reason=expected_exactly_one_target_criterion; c3 [confirmed cervical nodal thyroid metastasis is positive] accepted=Met, diagnostic=Met, raw=T, final=U, reason=unresolved_model_evidence_conflict; c4 [confirmed contralateral thyroid carcinoma is positive] accepted=None, diagnostic=Not Met, raw=U, final=F, reason=invalid_json_response; c5 [vascular invasion is positive] accepted=None, diagnostic=Not Met, raw=U, final=F, reason=invalid_json_response

## 与整体结果的关系

历史评价结果为：TrialMatchAI-only 60/192（31.25%，macro-F1 0.2251）；加入确定性数值比较后 63/192（32.81%，macro-F1 0.2461）；加入 BioLinkBERT-MedNLI 后 104/192（54.17%，macro-F1 0.5254）。证据模型和确定性比较能补回部分未知状态，但同时出现 273 条 condition.conflict=true 的冲突记录（涉及 135 个样本）；最终联合正确 53/192（27.60%）。

已知样本证据定位准确率为 TrialMatchAI-only 不适用、加入数值后 5/137（3.65%）、最终联合 76/137（55.47%）；BioLinkBERT 对证据定位有明显帮助，但不能消除规则条件缺失、否定、多值和冲突处理问题。联合方法反事实成对准确率为 7/32（21.88%），仍低于 Qwen3 直接判定的 22/32（68.75%）。

结论仅限于本次回归：当前端到端方案尚不能作为正式模块一。31.25% 是固定解析协议下的实际运行准确率；它不能独立估计 TrialMatchAI 在成功读取全部目标标签后的医学判断能力。当前主要瓶颈是官方完整 schema 与单原子请求之间的输出格式不一致、Markdown 围栏和长推理截断。BioLinkBERT 改善了证据定位，但在未知条件和模型/证据冲突上仍会触发保守未知。

诊断原始文件（服务器私有）：runs/v5_20261018/private/trialmatch_precomputed.jsonl、runs/v5_20261018/private/predictions_private.jsonl。本报告不包含原始患者文本和完整推理。
