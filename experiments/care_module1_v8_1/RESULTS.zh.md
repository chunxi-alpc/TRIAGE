# CARE 模块一 V8.1 实验结果

报告生成时间（Asia/Shanghai）：2026-10-20T00:22:57.601944+08:00

开发集选择已冻结为 **A1**，冻结时间（UTC）：2026-10-19T15:58:52.076333+00:00。

本报告只陈述结果；是否采用 V8.1 的最终结论单独撰写。

## 实验设置

基于 PR #12 的 Phi-4＋TrialMatchAI LoRA。A1 使用原始语义判断；B1 在单次判断中要求检查所有条件、例外、否定和未知信息；B2 复用 A1 首次判断，再进行一次语义复核。没有加入 RAG、置信度或复杂规则门控。每阶段仅保留既有的一次格式重试。

96 例开发集用于选择；历史 192 例、原新增 72 例和本次新写自然语言 72 例仅用于评价。每例含 3 条规则。选择按规则准确率、F→T 比率、U 召回率、完整调用成本与 token 成本依次排序，评价集不反向影响选择。历史 A1 结果直接复用；B2 复用对应 A1 输出作为首次判断，新增 GPU 工作为复核。自然语言 72 例的 A1、B1、B2 复核均实际执行新推理。

## 主要指标

T/F/U 是规则适用、不适用和信息不足。准确率按全部规则计算；格式错误计为错误，不记为 U。F→T 的分母为金标准 F 规则数。反事实配对要求同一对中目标规则的 T 与 F 均预测正确。

|数据|方案|正确规则|准确率|宏平均 F1|F→T|F→T 比率|U 召回|目标配对|病例全部正确|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|dev|A1|220/288|76.39%|0.70|24/176|13.64%|43.75%|12/36|62.50%|
|dev|B1|216/288|75.00%|0.65|22/176|12.50%|31.25%|10/36|59.38%|
|dev|B2|218/288|75.69%|0.67|24/176|13.64%|37.50%|12/36|61.46%|
|confirmation|A1|450/576|78.12%|0.66|38/352|10.80%|26.56%|36/72|65.62%|
|confirmation|B1|449/576|77.95%|0.63|36/352|10.23%|18.75%|39/72|66.15%|
|confirmation|B2|446/576|77.43%|0.65|34/352|9.66%|25.00%|36/72|61.98%|
|new72|A1|192/216|88.89%|0.86|3/120|2.50%|68.75%|20/24|79.17%|
|new72|B1|192/216|88.89%|0.86|3/120|2.50%|68.75%|19/24|79.17%|
|new72|B2|193/216|89.35%|0.86|3/120|2.50%|64.58%|22/24|81.94%|
|natural72|A1|189/216|87.50%|0.84|5/120|4.17%|64.58%|18/24|79.17%|
|natural72|B1|186/216|86.11%|0.82|4/120|3.33%|56.25%|19/24|77.78%|
|natural72|B2|190/216|87.96%|0.85|4/120|3.33%|64.58%|19/24|80.56%|

## 各类别表现

|数据|方案|类别|支持数|精确率|召回率|F1|
|---|---|---|---:|---:|---:|---:|
|dev|A1|T|80|62.07%|67.50%|0.65|
|dev|A1|F|176|81.28%|86.36%|0.84|
|dev|A1|U|32|100.00%|43.75%|0.61|
|dev|B1|T|80|61.18%|65.00%|0.63|
|dev|B1|F|176|79.79%|87.50%|0.83|
|dev|B1|U|32|100.00%|31.25%|0.48|
|dev|B2|T|80|61.36%|67.50%|0.64|
|dev|B2|F|176|80.85%|86.36%|0.84|
|dev|B2|U|32|100.00%|37.50%|0.55|
|confirmation|A1|T|160|66.11%|74.38%|0.70|
|confirmation|A1|F|352|82.85%|89.20%|0.86|
|confirmation|A1|U|64|100.00%|26.56%|0.42|
|confirmation|B1|T|160|67.03%|76.25%|0.71|
|confirmation|B1|F|352|82.89%|89.49%|0.86|
|confirmation|B1|U|64|85.71%|18.75%|0.31|
|confirmation|B2|T|160|65.50%|70.00%|0.68|
|confirmation|B2|F|352|81.75%|90.34%|0.86|
|confirmation|B2|U|64|100.00%|25.00%|0.40|
|new72|A1|T|48|82.69%|89.58%|0.86|
|new72|A1|F|120|89.23%|96.67%|0.93|
|new72|A1|U|48|97.06%|68.75%|0.80|
|new72|B1|T|48|82.35%|87.50%|0.85|
|new72|B1|F|120|88.64%|97.50%|0.93|
|new72|B1|U|48|100.00%|68.75%|0.81|
|new72|B2|T|48|81.82%|93.75%|0.87|
|new72|B2|F|120|90.00%|97.50%|0.94|
|new72|B2|U|48|100.00%|64.58%|0.78|
|natural72|A1|T|48|75.44%|89.58%|0.82|
|natural72|A1|F|120|89.84%|95.83%|0.93|
|natural72|A1|U|48|100.00%|64.58%|0.78|
|natural72|B1|T|48|74.14%|89.58%|0.81|
|natural72|B1|F|120|88.55%|96.67%|0.92|
|natural72|B1|U|48|100.00%|56.25%|0.72|
|natural72|B2|T|48|76.79%|89.58%|0.83|
|natural72|B2|F|120|89.92%|96.67%|0.93|
|natural72|B2|U|48|100.00%|64.58%|0.78|

## 混淆矩阵

|数据|方案|金标准|预测 T|预测 F|预测 U|格式错误|
|---|---|---|---:|---:|---:|---:|
|dev|A1|T|54|26|0|0|
|dev|A1|F|24|152|0|0|
|dev|A1|U|9|9|14|0|
|dev|B1|T|52|28|0|0|
|dev|B1|F|22|154|0|0|
|dev|B1|U|11|11|10|0|
|dev|B2|T|54|26|0|0|
|dev|B2|F|24|152|0|0|
|dev|B2|U|10|10|12|0|
|confirmation|A1|T|119|41|0|0|
|confirmation|A1|F|38|314|0|0|
|confirmation|A1|U|23|24|17|0|
|confirmation|B1|T|122|37|1|0|
|confirmation|B1|F|36|315|1|0|
|confirmation|B1|U|24|28|12|0|
|confirmation|B2|T|112|48|0|0|
|confirmation|B2|F|34|318|0|0|
|confirmation|B2|U|25|23|16|0|
|new72|A1|T|43|5|0|0|
|new72|A1|F|3|116|1|0|
|new72|A1|U|6|9|33|0|
|new72|B1|T|42|6|0|0|
|new72|B1|F|3|117|0|0|
|new72|B1|U|6|9|33|0|
|new72|B2|T|45|3|0|0|
|new72|B2|F|3|117|0|0|
|new72|B2|U|7|10|31|0|
|natural72|A1|T|43|5|0|0|
|natural72|A1|F|5|115|0|0|
|natural72|A1|U|9|8|31|0|
|natural72|B1|T|43|5|0|0|
|natural72|B1|F|4|116|0|0|
|natural72|B1|U|11|10|27|0|
|natural72|B2|T|43|5|0|0|
|natural72|B2|F|4|116|0|0|
|natural72|B2|U|9|8|31|0|

## 反事实配对

|数据|方案|目标规则配对|全部规则配对|发生 T/F 翻转的规则配对|
|---|---|---:|---:|---:|
|dev|A1|12/36|12/36|33/72|
|dev|B1|10/36|10/36|32/72|
|dev|B2|12/36|12/36|33/72|
|confirmation|A1|36/72|36/72|80/144|
|confirmation|B1|39/72|39/72|84/144|
|confirmation|B2|36/72|31/72|75/144|
|new72|A1|20/24|20/24|43/48|
|new72|B1|19/24|19/24|42/48|
|new72|B2|22/24|22/24|45/48|
|natural72|A1|18/24|18/24|38/48|
|natural72|B1|19/24|19/24|39/48|
|natural72|B2|19/24|19/24|39/48|

## 推理成本

完整成本表示从头运行该方案所需的全部判断和格式重试；新增成本表示本次实际执行的工作。token 为模型 tokenizer 计数。时间为真实 GPU 批次生成时间分摊后的合计，不含模型加载。历史与本次批次、并发和硬件运行条件可能不同，因此时间比只作描述性比较。

|数据|方案|完整/新增调用|完整输入/输出 token|新增输入/输出 token|完整/新增生成秒|调用比 A1|token 比 A1|时间比 A1|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|dev|A1|96/0|213900/15198|0/0|259.33/0.00|1.00|1.00|1.00|
|dev|B1|96/96|227340/15091|227340/15091|264.02/264.02|1.00|1.06|1.02|
|dev|B2|192/96|461046/30391|247146/15193|543.28/283.96|2.00|2.15|2.09|
|confirmation|A1|192/0|425322/30506|0/0|513.29/0.00|1.00|1.00|1.00|
|confirmation|B1|192/192|452202/30467|452202/30467|529.07/529.07|1.00|1.06|1.03|
|confirmation|B2|384/192|917246/61000|491924/30494|1075.04/561.76|2.00|2.15|2.09|
|new72|A1|72/0|154153/12193|0/0|199.30/0.00|1.00|1.00|1.00|
|new72|B1|72/72|164233/12150|164233/12150|205.98/205.98|1.00|1.06|1.03|
|new72|B2|144/72|334035/24357|179882/12164|415.82/216.52|2.00|2.15|2.09|
|natural72|A1|72/72|154563/12615|154563/12615|207.19/207.19|1.00|1.00|1.00|
|natural72|B1|72/72|164643/12685|164643/12685|215.48/215.48|1.00|1.06|1.04|
|natural72|B2|144/72|335277/25226|180714/12611|436.25/229.06|2.00|2.16|2.11|

## B2 复核改变了什么

直接比较 B2 自身首次判断与复核后的规则状态。纠正为错→对，改错为对→错，保持错包含错→另一个错。F→T 修正表示原来的 F→T 不再出现，可能改成 F 或 U；具体状态保存在 review_transitions.json。

|数据|纠正|改错|保持错|保持对|新增 F→T|消除 F→T|净增加正确|
|---|---:|---:|---:|---:|---:|---:|---:|
|dev|0|2|68|218|0|0|-2|
|confirmation|4|8|122|442|0|4|-4|
|new72|3|2|21|190|0|0|1|
|natural72|1|0|26|189|0|1|1|

## 错误案例

以下按每个数据集和方案选取至多三个不同病例的错误，优先显示 F→T，其次显示 U 误识别；不是随机样本。完整错误在 analysis/errors.jsonl。

### dev_A1：mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_A1：mechanism_04_conjunction_nested_disjunction-p02-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_A1：mechanism_04_conjunction_nested_disjunction-p03-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_B1：mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_B1：mechanism_04_conjunction_nested_disjunction-p02-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_B1：mechanism_04_conjunction_nested_disjunction-p03-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_B2：mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_B2：mechanism_04_conjunction_nested_disjunction-p02-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### dev_B2：mechanism_04_conjunction_nested_disjunction-p03-unit_equivalent-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient does not have routing authorization. The patient's current tumor diameter is 25 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient fails the age-and-authorization or tumor diameter condition.

### confirmation_A1：mechanism_02_disjunction-p01-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 216 months. The current patient does not have routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When (the patient's current age is at least 19 years or the current patient has routing authorization), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 18 years old, meeting the age requirement, and has no suspension flag.

### confirmation_A1：mechanism_02_disjunction-p02-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 216 months. The current patient does not have routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When (the patient's current age is at least 19 years or the current patient has routing authorization), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 18 years old, meeting the age requirement, and has no suspension flag.

### confirmation_A1：mechanism_05_disjunction_nested_conjunction-p01-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient has routing authorization. The patient's current tumor diameter is 31 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When ((the patient's current age is at least 22 years and the current patient has routing authorization) or the patient's current tumor diameter is at most 3 cm), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 21 years old, and the tumor diameter is 31 mm, meeting the age or tumor criteria.

### confirmation_B1：mechanism_02_disjunction-p01-scope_and_order-app / R_ALTERNATIVE

金标准 **F**，预测 **T**。

**病例：** The patient's historical age was 18 years. The patient's sibling's current age is 19 years. The current patient does not have a suspension waiver. The current patient does not have a routing suspension flag. The current patient does not have routing authorization. The patient's current age is 19 years.

**规则：** When it is not the case that ((the patient's current age is at least 19 years or the current patient has routing authorization)), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型理由：** The patient lacks authorization, fulfilling the negation of the age-or-authorization condition.

### confirmation_B1：mechanism_02_disjunction-p01-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 216 months. The current patient does not have routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When (the patient's current age is at least 19 years or the current patient has routing authorization), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 18 years old, satisfying the age condition.

### confirmation_B1：mechanism_02_disjunction-p02-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 216 months. The current patient does not have routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When (the patient's current age is at least 19 years or the current patient has routing authorization), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 18 years old, satisfying the age condition.

### confirmation_B2：mechanism_02_disjunction-p01-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 216 months. The current patient does not have routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When (the patient's current age is at least 19 years or the current patient has routing authorization), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 18 years old, meeting the age requirement, and has no suspension flag.

### confirmation_B2：mechanism_02_disjunction-p02-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 216 months. The current patient does not have routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When (the patient's current age is at least 19 years or the current patient has routing authorization), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 18 years old, meeting the age requirement, and has no suspension flag.

### confirmation_B2：mechanism_05_disjunction_nested_conjunction-p01-unit_equivalent-non / R_TARGET

金标准 **F**，预测 **T**。

**病例：** The patient's current age is 252 months. The current patient has routing authorization. The patient's current tumor diameter is 31 mm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When ((the patient's current age is at least 22 years and the current patient has routing authorization) or the patient's current tumor diameter is at most 3 cm), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型理由：** The patient is 21 years old, and the tumor diameter is 31 mm, meeting the age or tumor criteria.

### new72_A1：v8_new_09_checkin_departure_room_override_p01_T / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The coordinator is directing the patient to an appointment room. Yesterday's visit ended normally. The patient checked in for today's appointment at 09:00. The patient remains in the facility and has not left since today's check-in. The assigned room is open today. No approved replacement room is available for this appointment.

**规则：** If the assigned room is closed and no approved replacement room is available, assign Route 09 Exception; this exception takes precedence over the other two branches.

**模型理由：** The assigned room is open, but no replacement is available, triggering the exception.

### new72_A1：v8_new_12_diary_review_new_entry_p01_F / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The research coordinator is scheduling a discussion of a nonclinical study diary. The partner keeps a separate diary. The patient's own diary has entries only on Monday and Thursday of this week. The patient asked to discuss the diary. The coordinator confirms that the current diary has not yet been reviewed. No new diary entry has been added after the review.

**规则：** If the current diary was already reviewed and no new entry has been added after that review, assign Route 12 Exception; this exception takes precedence over the other two branches.

**模型理由：** The exception condition is met, assigning Route 12 Exception.

### new72_A1：v8_new_12_diary_review_new_entry_u01 / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The research coordinator is scheduling a discussion of a nonclinical study diary. The partner keeps a separate diary. Only the partner's diary dates are available; dates in the patient's diary are missing. The patient asked to discuss the diary. The coordinator confirms that the current diary has not yet been reviewed. No new diary entry has been added after the review.

**规则：** If the current diary was already reviewed and no new entry has been added after that review, assign Route 12 Exception; this exception takes precedence over the other two branches.

**模型理由：** The diary was reviewed, and no new entries were added, triggering this exception.

### new72_B1：v8_new_09_checkin_departure_room_override_p01_T / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The coordinator is directing the patient to an appointment room. Yesterday's visit ended normally. The patient checked in for today's appointment at 09:00. The patient remains in the facility and has not left since today's check-in. The assigned room is open today. No approved replacement room is available for this appointment.

**规则：** If the assigned room is closed and no approved replacement room is available, assign Route 09 Exception; this exception takes precedence over the other two branches.

**模型理由：** The exception condition is met due to the lack of an approved replacement room.

### new72_B1：v8_new_12_diary_review_new_entry_p01_F / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The research coordinator is scheduling a discussion of a nonclinical study diary. The partner keeps a separate diary. The patient's own diary has entries only on Monday and Thursday of this week. The patient asked to discuss the diary. The coordinator confirms that the current diary has not yet been reviewed. No new diary entry has been added after the review.

**规则：** If the current diary was already reviewed and no new entry has been added after that review, assign Route 12 Exception; this exception takes precedence over the other two branches.

**模型理由：** The exception condition is met, assigning Route 12 Exception.

### new72_B1：v8_new_12_diary_review_new_entry_u01 / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The research coordinator is scheduling a discussion of a nonclinical study diary. The partner keeps a separate diary. Only the partner's diary dates are available; dates in the patient's diary are missing. The patient asked to discuss the diary. The coordinator confirms that the current diary has not yet been reviewed. No new diary entry has been added after the review.

**规则：** If the current diary was already reviewed and no new entry has been added after that review, assign Route 12 Exception; this exception takes precedence over the other two branches.

**模型理由：** The exception rule is triggered by the diary review status.

### new72_B2：v8_new_09_checkin_departure_room_override_p01_T / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The coordinator is directing the patient to an appointment room. Yesterday's visit ended normally. The patient checked in for today's appointment at 09:00. The patient remains in the facility and has not left since today's check-in. The assigned room is open today. No approved replacement room is available for this appointment.

**规则：** If the assigned room is closed and no approved replacement room is available, assign Route 09 Exception; this exception takes precedence over the other two branches.

**模型理由：** The assigned room is open, but no replacement is available, triggering the exception.

### new72_B2：v8_new_12_diary_review_new_entry_p01_F / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The research coordinator is scheduling a discussion of a nonclinical study diary. The partner keeps a separate diary. The patient's own diary has entries only on Monday and Thursday of this week. The patient asked to discuss the diary. The coordinator confirms that the current diary has not yet been reviewed. No new diary entry has been added after the review.

**规则：** If the current diary was already reviewed and no new entry has been added after that review, assign Route 12 Exception; this exception takes precedence over the other two branches.

**模型理由：** The exception condition is met, assigning Route 12 Exception.

### new72_B2：v8_new_12_diary_review_new_entry_u01 / R_EXCEPTION

金标准 **F**，预测 **T**。

**病例：** The research coordinator is scheduling a discussion of a nonclinical study diary. The partner keeps a separate diary. Only the partner's diary dates are available; dates in the patient's diary are missing. The patient asked to discuss the diary. The coordinator confirms that the current diary has not yet been reviewed. No new diary entry has been added after the review.

**规则：** If the current diary was already reviewed and no new entry has been added after that review, assign Route 12 Exception; this exception takes precedence over the other two branches.

**模型理由：** The diary was reviewed, and no new entries were added, triggering this exception.

### natural72_A1：v81_natural_01_headache_daily_activity_p01_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** Mara describes a band-like ache while sorting invoices. Her sister has migraine, but this account concerns Mara. She has had three episodes since Monday. She keeps working and doing her usual activities throughout each episode without interruption. She remembers the same sort of headache on several occasions last year.

**规则：** Use the standard assessment queue if headaches interrupt ordinary activities and at least two episodes occurred this week, unless the person reports that headaches began for the first time in their life this week.

**模型理由：** Headaches do not interrupt activities and occurred at least twice this week.

### natural72_A1：v81_natural_01_headache_daily_activity_p02_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** She is clear that she had never experienced any headache before this week. Mara describes a band-like ache while sorting invoices. Her sister has migraine, but this account concerns Mara. She has had three episodes since Monday. During each episode she puts her work aside until the ache settles.

**规则：** Use the standard assessment queue if headaches interrupt ordinary activities and at least two episodes occurred this week, unless the person reports that headaches began for the first time in their life this week.

**模型理由：** Headaches began this week, so the standard assessment queue is not applicable.

### natural72_A1：v81_natural_02_breathlessness_stairs_p01_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** Joel is discussing his walk to an upstairs office. He says the breathing trouble settles after he pauses. His colleague has asthma; Joel gives no diagnosis for himself. He climbs the stairs without any breathlessness; the trouble occurs only during running. Sitting quietly, he breathes comfortably and has no breathlessness.

**规则：** Use the standard assessment queue if breathlessness occurs when climbing stairs and settles when the person stops, unless breathlessness also occurs while the person is sitting quietly.

**模型理由：** Breathlessness occurs when climbing stairs and settles when he stops, with no breathlessness while sitting quietly.

### natural72_B1：v81_natural_01_headache_daily_activity_p01_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** Mara describes a band-like ache while sorting invoices. Her sister has migraine, but this account concerns Mara. She has had three episodes since Monday. She keeps working and doing her usual activities throughout each episode without interruption. She remembers the same sort of headache on several occasions last year.

**规则：** Use the standard assessment queue if headaches interrupt ordinary activities and at least two episodes occurred this week, unless the person reports that headaches began for the first time in their life this week.

**模型理由：** Headaches do not interrupt activities and occurred at least twice this week.

### natural72_B1：v81_natural_01_headache_daily_activity_p02_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** She is clear that she had never experienced any headache before this week. Mara describes a band-like ache while sorting invoices. Her sister has migraine, but this account concerns Mara. She has had three episodes since Monday. During each episode she puts her work aside until the ache settles.

**规则：** Use the standard assessment queue if headaches interrupt ordinary activities and at least two episodes occurred this week, unless the person reports that headaches began for the first time in their life this week.

**模型理由：** Headaches interrupt activities and occurred at least twice this week.

### natural72_B1：v81_natural_02_breathlessness_stairs_p01_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** Joel is discussing his walk to an upstairs office. He says the breathing trouble settles after he pauses. His colleague has asthma; Joel gives no diagnosis for himself. He climbs the stairs without any breathlessness; the trouble occurs only during running. Sitting quietly, he breathes comfortably and has no breathlessness.

**规则：** Use the standard assessment queue if breathlessness occurs when climbing stairs and settles when the person stops, unless breathlessness also occurs while the person is sitting quietly.

**模型理由：** Breathlessness occurs when climbing stairs and settles when he stops, with no breathlessness while sitting quietly.

### natural72_B2：v81_natural_01_headache_daily_activity_p01_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** Mara describes a band-like ache while sorting invoices. Her sister has migraine, but this account concerns Mara. She has had three episodes since Monday. She keeps working and doing her usual activities throughout each episode without interruption. She remembers the same sort of headache on several occasions last year.

**规则：** Use the standard assessment queue if headaches interrupt ordinary activities and at least two episodes occurred this week, unless the person reports that headaches began for the first time in their life this week.

**模型理由：** Headaches do not interrupt activities and occurred at least twice this week.

### natural72_B2：v81_natural_02_breathlessness_stairs_p01_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** Joel is discussing his walk to an upstairs office. He says the breathing trouble settles after he pauses. His colleague has asthma; Joel gives no diagnosis for himself. He climbs the stairs without any breathlessness; the trouble occurs only during running. Sitting quietly, he breathes comfortably and has no breathlessness.

**规则：** Use the standard assessment queue if breathlessness occurs when climbing stairs and settles when the person stops, unless breathlessness also occurs while the person is sitting quietly.

**模型理由：** Breathlessness occurs when climbing stairs and settles when he stops, with no breathlessness while sitting quietly.

### natural72_B2：v81_natural_03_postural_lightheadedness_p02_F / R_TARGET

金标准 **F**，预测 **T**。

**病例：** During one episode she blacked out and later found herself on the floor. Anika describes a brief floating feeling when moving around the kitchen. Whenever it comes on, sitting back in the chair makes it fade. Her husband once fainted, but she is describing her own episodes here. The feeling begins just as she rises from her chair.

**规则：** Use the standard assessment queue if lightheadedness starts on standing and improves on sitting back down, unless the person has lost consciousness during one of these episodes.

**模型理由：** Lightheadedness starts on standing and improves on sitting, with no loss of consciousness.

## 数据与解释限制

历史 192 例及原新增 72 例已用于前序实验，属于复用对照，不能称为全新盲测。本次自然语言 72 例在推理前冻结，标签由人工设定的观察事实和既有三值逻辑生成，不由被测模型生成；它们仍是合成病例和虚构分流规则，不代表真实临床指南或临床患者分布。

病例包含反事实配对和重复主题，同一主题内的结果相关，不能把每条规则视为独立抽样来夸大统计证据。自然语言病例与原实验使用相同规则语义体系，因此结果说明叙述变化下的表现，不能单独证明跨任务泛化。

复核可能纠正错误，也可能把正确答案改错；成本和 F→T/U 表现需要与总准确率一起解释。开发集选出的方案保持冻结，不因后续某个数据集较好而重新挑选。

## 复现及结果文件

- `selection.json`：开发集选择及输入预测哈希。
- `analysis/metrics.json`、`summary.csv`、`family_metrics.json`：完整指标和病例组指标。
- `analysis/case_predictions.jsonl`、`rule_predictions.jsonl`、`errors.jsonl`：逐例、逐规则及错误明细。
- `analysis/paired_comparison.json`：B1/B2 相对 A1 的逐规则变化。
- `analysis/review_transitions.json`：B2 首次判断到复核的变化。
- `analysis/cost_ratios.json`：各数据集相对 A1 的完整成本比。
- `results/*/predictions.jsonl`：模型原始输出、提示、token 和实际 GPU 时间。

所有推理完成后在实验环境运行 `python experiments/care_module1_v8_1/analyze.py`，随后运行 `python experiments/care_module1_v8_1/build_report.py`。开发集选择命令为 `python experiments/care_module1_v8_1/analyze.py --select`，应在其他数据集推理开始前执行。
