# A1 增加针对 U 的对照示例：原96例开发集

报告时间：2026-10-20T08:43:35.026105+08:00

本轮只运行一个新增版本及原96例开发集。A1结果直接复用缓存，未运行历史192例或任何72例数据集，也未启动后续实验。

## 唯一改动

沿用原A1系统指令、目标输入、输出字段及顺序、Phi-4＋TrialMatchAI LoRA、NF4 4bit、batch8、greedy、384输出token预算。只替换训练演示：从三个演示中的四段病例，改成两个对照组中的六段新编病例。OR组覆盖T/F/U和NOT U；例外组覆盖明确不存在、明确存在和未知，并示范无关未知不影响已成立的T或F。示例在推理前冻结，不使用开发集原病例或金标作演示。

## 结果

|指标|原A1缓存|A1＋U对照示例|
|---|---:|---:|
|规则准确率|220/288（76.39%）|230/288（79.86%）|
|U正确识别|14/32（43.75%）|10/32（31.25%）|
|U精确率|100.00%|90.91%|
|F→T|24/176|18/176|
|宏平均F1|0.6976|0.6877|
|目标反事实配对全对|12/36|19/36|

相对A1纠正 33 条、改错 23 条；恢复原本漏识别的U 2 条，同时丢失原本正确U 6 条；新增F→T 10 条、消除F→T 16 条。

## 成本

|指标|原A1完整成本|本轮候选完整成本|
|---|---:|---:|
|模型请求|96|96|
|本轮实际新增请求|0|96|
|输入token|213900|245772|
|输出token|15198|16595|
|总token|229098|262367|
|批次生成秒|259.3|312.5|

输入token比A1增加 14.90%，总token增加 14.52%。历史与本轮耗时来自不同运行，不能视为严格同条件在线延迟比较。GPU0峰值已分配显存 18.07 GiB，含加载运行时间 344.5 秒。格式重试 0 次，失败 0 例。

## 结果解释与扩展建议

本轮总正确数增加10条，由T类增加8条、F类增加6条、U类减少4条组成。它改善了总体准确率和F→T，但没有实现U识别改善，宏平均F1也由0.6976下降至0.6877。建议暂不以U改进方案扩展；保留为总体准确率候选，是否扩展由用户另行决定。

丢失的6条原本正确U全部集中在3个仅提供亲属年龄的病例，涉及患者本人信息缺失与嵌套逻辑；恢复的2条U来自同一年龄矛盾病例。这说明简单OR/例外未知演示未在本开发集稳定覆盖更复杂的信息归属和组合条件，不能据此断言模型内部原因。

## 改变实例

### corrected：mechanism_10_entry_with_two_exclusions-u-contradictory_current-canonical-unknown / R_ALTERNATIVE

金标 U；A1 T → 新版 U。

**病例：** The current patient does not have routing authorization. The patient's current tumor diameter is 1.6 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver. The patient's current age is 18 years. The patient's current age is 17 years.

**规则：** When it is not the case that ((the patient's current age is at least 18 years and it is not the case that ((the current patient has routing authorization or the patient's current tumor diameter is at most 1.5 cm)))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**A1理由：** The age-and-authorization condition fails, triggering the alternative route.

**新版理由：** Age contradiction leaves the negated age condition unresolved.

### corrected：mechanism_10_entry_with_two_exclusions-u-contradictory_current-canonical-unknown / R_TARGET

金标 U；A1 F → 新版 U。

**病例：** The current patient does not have routing authorization. The patient's current tumor diameter is 1.6 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver. The patient's current age is 18 years. The patient's current age is 17 years.

**规则：** When (the patient's current age is at least 18 years and it is not the case that ((the current patient has routing authorization or the patient's current tumor diameter is at most 1.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**A1理由：** The patient lacks authorization, failing the age-and-authorization condition.

**新版理由：** Age is contradictory, making the age-related condition unresolved.

### corrected：mechanism_10_entry_with_two_exclusions-p01-canonical-app / R_ALTERNATIVE

金标 F；A1 T → 新版 F。

**病例：** The patient's current age is 18 years. The current patient does not have routing authorization. The patient's current tumor diameter is 1.6 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**规则：** When it is not the case that ((the patient's current age is at least 18 years and it is not the case that ((the current patient has routing authorization or the patient's current tumor diameter is at most 1.5 cm)))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**A1理由：** The age condition holds, but the lack of authorization and tumor diameter condition fails, making the negation true.

**新版理由：** The prerequisites for R_TARGET are met, making the negation false.

### broken：mechanism_04_conjunction_nested_disjunction-u-relative_only-canonical-unknown / R_ALTERNATIVE

金标 U；A1 U → 新版 T。

**病例：** The current patient does not have routing authorization. The patient's current tumor diameter is 2.5 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver. The patient's sibling's current age is 21 years.

**规则：** When it is not the case that ((the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**A1理由：** Patient's age is missing, so the negation of the age-and-authorization-or-tumor condition is unresolved.

**新版理由：** The OR condition is false, establishing the negation of the whole condition.

### broken：mechanism_04_conjunction_nested_disjunction-u-relative_only-canonical-unknown / R_TARGET

金标 U；A1 U → 新版 F。

**病例：** The current patient does not have routing authorization. The patient's current tumor diameter is 2.5 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver. The patient's sibling's current age is 21 years.

**规则：** When (the patient's current age is at least 21 years and (the current patient has routing authorization or the patient's current tumor diameter is at most 2.5 cm)), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**A1理由：** Patient's age is missing, so the age condition is unresolved despite tumor diameter and no suspension.

**新版理由：** The patient lacks routing authorization, making the OR condition false despite the tumor diameter.

### broken：mechanism_10_entry_with_two_exclusions-u-relative_only-canonical-unknown / R_ALTERNATIVE

金标 U；A1 U → 新版 F。

**病例：** The current patient does not have routing authorization. The patient's current tumor diameter is 1.6 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver. The patient's sibling's current age is 18 years.

**规则：** When it is not the case that ((the patient's current age is at least 18 years and it is not the case that ((the current patient has routing authorization or the patient's current tumor diameter is at most 1.5 cm)))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**A1理由：** Patient's age is missing, so the negation of the age condition cannot be evaluated.

**新版理由：** The negated condition is false due to the age condition being met.

## 复现与范围

运行入口见 README.md；逐例原始输出保存在 results/dev_A1_U/predictions.jsonl，全部变化保存在 analysis/changes.json。A1原始缓存保持不变。
这是在已反复使用的开发集上的探索结果，不能代替新数据验证。是否扩展需在查看上述收益和代价后另作决定；本轮已停在96例。
