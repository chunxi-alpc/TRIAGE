# V8.1 开发集语义错误分析

本分析仅读取已经完成的 dev96 三臂预测和开发集病例，不使用后续评价集结果，不修改提示。以下解释依据实验中的虚构分流规则及病例文字，不增加医学判断。

## 开发集变化

|方案|正确规则|F→T|U 正确识别|相对 A1 纠正/改错|
|---|---:|---:|---:|---:|
|A1|220/288|24/176|14/32（43.75%）|—|
|B1|216/288|22/176|10/32（31.25%）|9/13|
|B2|218/288|24/176|12/32（37.50%）|0/2|

B1 消除了 5 条原有 F→T，同时新增 3 条，净减少 2 条；总正确数反而减少 4 条，U 识别也减少 4 条。B2 没有纠正任何规则，仅把同一病例的两条正确 U 改错。已核对 B2 首次判断与 A1 的全部开发集标签一致，因此上述改变就是复核带来的改变。

**不存在可展示的 B2 改对案例。** 不能为了正反各举一例而编造纠正。开发集仍按既定标准选择 A1。

## 未知条件应如何参与完整规则判断

OR：已知任一分支为 T 就可为 T；仅当所有分支均为 F 才可为 F。没有 T 且至少一项 U 时，结果为 U。AND：任一必要条件已知为 F 就可为 F；只有全部为 T 才可为 T。没有 F 且至少一项 U 时，结果为 U。NOT U 仍为 U。必须先按括号判断内部表达式，再组合外层条件；不能把内部 OR 的单个 F 当作外层 AND 已经失败。以下三个案例均明确没有暂停标记，例外条件不成立。

## B2 复核把正确的 U 改错

病例：`mechanism_16_three_alternatives-u-missing-canonical-unknown`；规则：`R_TARGET`。金标准 U，A1 U → B2 F。

**原始病例**

> The current patient does not have routing authorization. The patient's current tumor diameter is 4.6 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver. The current value of age is not documented.

**原始规则**

> When (the patient's current age is at least 24 years or the current patient has routing authorization or the patient's current tumor diameter is at most 4.5 cm), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型原始理由（仅作为待分析的输出）**

- A1：Age is missing, so the age condition is unresolved, but authorization and tumor size are not met.
- B2：Authorization is not met, and tumor size exceeds 4.5 cm.

**按文本独立解释**

规则包含三个并列的“或”：年龄至少 24 岁、具有授权、肿瘤直径至多 4.5 cm。病例说明无授权，直径 4.6 cm，但年龄未记录，因此三项依次是 U、F、F。OR 中只有所有分支都为 F 才能判 F；这里年龄仍可能满足，所以 U OR F OR F = U。病例明确没有暂停标记，例外不成立，不能据此消除年龄的不确定性。

A1 判断 U 正确；B2 变成 F，理由只列出了已知不满足的两个分支，未处理可能满足的年龄分支。同例 R_ALTERNATIVE 是整个 OR 的否定，NOT U 仍是 U，但 B2 将其从 U 改为 T。这解释了两条新增错误；不是两个独立病例。

## B1 修复 F→T：嵌套否定

病例：`mechanism_10_entry_with_two_exclusions-p01-canonical-app`；规则：`R_ALTERNATIVE`。金标准 F，A1 T → B1 F。

**原始病例**

> The patient's current age is 18 years. The current patient does not have routing authorization. The patient's current tumor diameter is 1.6 cm. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**原始规则**

> When it is not the case that ((the patient's current age is at least 18 years and it is not the case that ((the current patient has routing authorization or the patient's current tumor diameter is at most 1.5 cm)))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**模型原始理由（仅作为待分析的输出）**

- A1：The age condition holds, but the lack of authorization and tumor diameter condition fails, making the negation true.
- B1：The age condition holds, so the negation of the age-and-authorization combination fails.

**按文本独立解释**

病例中年龄 18 岁满足“至少 18 岁”；无授权为 F；直径 1.6 cm 对“至多 1.5 cm”为 F。因此内部“授权 OR 直径条件”为 F，否定后为 T；再与年龄条件作 AND 得 T；最外层否定得到 F。没有暂停标记，例外不成立。完整规则的结果确实应为 F。

A1 给 T，B1 改为 F，修复了该条 F→T。不过 B1 的理由仅说年龄条件成立，所以否定失败，并未完整说明另外两项。单独知道年龄满足不足以决定这个嵌套规则；这里之所以可判 F，还依赖无授权及直径超过阈值。故该例证明最终标签改对，不能据简短理由断言模型已经稳定掌握嵌套逻辑。

## B1 丢失 U：患者与亲属信息、OR 与 AND 的组合

病例：`mechanism_22_five_condition_composition-u-relative_only-canonical-unknown`；规则：`R_TARGET`。金标准 U，A1 U → B1 F。

**原始病例**

> The current patient does not have routing authorization. The patient's current tumor diameter is 3.6 cm. The current patient does not have an audit certificate. The patient's current relapse interval is 24 months. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver. The patient's sibling's current age is 21 years.

**原始规则**

> When ((the patient's current age is at least 21 years or the current patient has routing authorization) and it is not the case that ((the patient's current tumor diameter is at most 3.5 cm and the current patient has an audit certificate)) and the patient's current relapse interval is greater than 12 months), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Cedar.

**模型原始理由（仅作为待分析的输出）**

- A1：Patient's age is missing, so the age-or-authorization condition is unresolved.
- B1：The patient lacks authorization, failing the age-or-authorization condition.

**按文本独立解释**

年龄信息属于患者的兄弟姐妹，不能当作患者本人年龄；因此患者“年龄至少 21 岁”是 U。患者无授权，所以第一个括号“年龄条件 OR 授权”为 U OR F = U，不是 F。直径 3.6 cm 使“至多 3.5 cm”为 F，无审计证明亦为 F，二者 AND 为 F，外层 NOT 后为 T；复发间隔 24 个月满足“超过 12 个月”，为 T。完整前提为 U AND T AND T = U；无暂停标记，例外不成立。

A1 给 U；B1 因患者缺少授权而判 F，但在 OR 中，一个分支不满足不代表整个括号不满足。这里没有任何已知为 F 的顶层 AND 必要条件，不能用“一个条件不满足即可判 F”把整个规则判 F。A1 的 evidence_quote 也不是病例中的逐字原句，但其 U 标签与文本语义相符；证据引用质量与适用性标签应分别评价。

## 可以支持的判断

在该开发集，额外要求完整检查或增加一次复核，并没有带来总体改善。B1 的部分嵌套否定标签得到修复，但同时出现把 OR 中未解决的条件当作失败的错误；B2 保留了绝大多数原判，并把一例原本正确保留的未知改成确定判断。以上是对实际输出的解释，不足以证明模型内部采用了哪一种推理过程，也不扩展为对后续数据集的结论。

## 补充：历史 192 例的 B2 复核

本节单独读取已完成的历史 192 例结果，不涉及尚在运行的数据集。开发集选择仍冻结为 A1，不因这段对照分析重新选择。

历史 192 例包含 576 条规则，B2 纠正 **4** 条、改错 **8** 条、保持错误 122 条、保持正确 442 条，正确数从 A1 的 450 降为 446。与开发集“无纠正、改错 2 条”不同，历史集确有纠正；4 条均消除了原有 F→T，且没有新增 F→T，因此 F→T 从 38 降至 34，但总体准确率仍下降。

### 纠正实例：标签改对，但理由没有变正确

病例：`mechanism_08_negated_conjunction-p01-unit_equivalent-app`；规则：`R_ALTERNATIVE`。金标准 F，A1 T → B2 F。

**原始病例：** The patient's current age is 288 months. The current patient has routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**原始规则：** When it is not the case that (it is not the case that ((the patient's current age is at least 25 years and the current patient has routing authorization))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**A1 原始理由：** The age-and-authorization combination holds, so its negation is false.

**B2 原始理由：** The age-and-authorization combination holds, so its negation is false.

**按文本独立解释：** 患者年龄 288 个月，即 24 岁，未达到 25 岁；有授权。内部 AND 为 F，连续两次否定仍为 F，没有暂停标记，例外不成立。因此 A1 的 T 错，B2 的 F 对。然而两者理由都声称年龄与授权的组合成立，与实际年龄不符；B2 的标签改善不能证明其已正确处理单位换算或双重否定。

### 改错实例：同一双重否定规则下把 T 改为 F

病例：`mechanism_08_negated_conjunction-p01-canonical-non`；规则：`R_ALTERNATIVE`。金标准 T，A1 T → B2 F。

**原始病例：** The patient's current age is 25 years. The current patient has routing authorization. The current patient does not have a routing suspension flag. The current patient does not have a suspension waiver.

**原始规则：** When it is not the case that (it is not the case that ((the patient's current age is at least 25 years and the current patient has routing authorization))), unless (the current patient has a routing suspension flag and it is not the case that (the current patient has a suspension waiver)), assign Route Maple.

**A1 原始理由：** The age-and-authorization combination holds, so its negation is false.

**B2 原始理由：** The age-and-authorization combination holds, so its negation is false.

**按文本独立解释：** 患者年龄 25 岁且有授权，内部 AND 为 T；连续两次否定仍为 T。没有暂停标记，例外不成立，所以 A1 的 T 正确，B2 改成 F 错误。输出理由提到“其否定为假”，只对应一次否定，不能解释这条含两次否定的完整规则。这与上一例同属一种规则，说明单独看到某些 F→T 被修复，仍不足以认定复核稳定改善了完整条件判断。
