> Source snapshot: /home/gcx/aster_closed_form_20261105/analysis_failure/behavior.zh.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# 旧轮 ASTER 与完整阅读：逐例行为分析

仅使用旧轮已完成数据做CPU分析；不修改分数、标签、主表或算子。详细逐例记录在同目录 behavior.json。

## 关键结论

按原标签，ASTER宏平均93.602%，full为94.407%。差异完全来自同一节点的两项P纠正与另一个节点的一项C计分退化；A/F和200个原输入正确性均未改变。

唯一C退化存在明确的文字与谓词不一致，保存正标签没有病历依据；不能据此认定ASTER破坏了有效C信号。两项P收益来自同一来源与节点，也不能解释成多个独立临床成功。路径差异则是一条病例最后节点从同分判否移动到正侧。

## 配对与输入变化

| 类型 | 总数 | 两者正确 | 纠正 | 退化 | 两者未全对 | 宏平均贡献（百分点） |
| --- | --- | --- | --- | --- | --- | --- |
| A | 200 | 198 | 0 | 0 | 2 | +0.0000 |
| P_clinical | 49 | 47 | 1 | 0 | 1 | +0.2551 |
| P_preference | 49 | 47 | 1 | 0 | 1 | +0.2551 |
| C | 19 | 15 | 0 | 1 | 3 | -1.3158 |
| F | 200 | 197 | 0 | 0 | 3 | +0.0000 |

| 输入池 | 总数 | 两者正确 | 纠正 | 退化 | 两者错误 |
| --- | --- | --- | --- | --- | --- |
| original | 200 | 199 | 0 | 0 | 1 |
| all_views | 615 | 603 | 2 | 1 | 9 |

P两分项各贡献+0.2551个百分点，C一对贡献−1.3158个百分点，合计−0.8056个百分点。这是既定指标权重的算术分解，不是因果机制识别。

## 所有预测标签改变的视图

### expanded-confirmation-2303-G0:P_clinical1

来源：train_val:2303；保存标签：0。
条件：In a patient with epilepsy, is the NDDI-E (Neurological Disorders Depression Inventory for Epilepsy) score > 13?

病历：An adult with established epilepsy is undergoing depression assessment. The NDDI-E score is 13. The patient has a history of generalized tonic-clonic seizures.

编辑：[]

| 方法 | 是减否分数 | 按保存标签正确 |
| --- | --- | --- |
| full | 0.5 | False |
| aster | -4.0 | True |
| direct | -18.0 | True |
| logistic_direct | -0.6618643417030236 | True |
| per_type_mean | -0.5 | True |
| no_C | -3.75 | True |
| orthogonal | -5.25 | True |
| random_13 | -0.75 | True |
| random_37 | 0.75 | False |
| random_73 | -3.5 | True |

### expanded-confirmation-2303-G0:P_preference1

来源：train_val:2303；保存标签：0。
条件：In a patient with epilepsy, is the NDDI-E (Neurological Disorders Depression Inventory for Epilepsy) score > 13?

病历：An adult with established epilepsy is undergoing depression assessment. The NDDI-E score is 13. The patient is open to medication if necessary.

编辑：[]

| 方法 | 是减否分数 | 按保存标签正确 |
| --- | --- | --- |
| full | 0.5 | False |
| aster | -3.5 | True |
| direct | -16.875 | True |
| logistic_direct | -0.621556871580687 | True |
| per_type_mean | 0.0 | True |
| no_C | -3.25 | True |
| orthogonal | -4.25 | True |
| random_13 | -0.75 | True |
| random_37 | 1.0 | False |
| random_73 | -2.5 | True |

### expanded-confirmation-2498-G0:condition

来源：train_val:2498；保存标签：1。
条件：Is chorionicity determined with reasonable accuracy on sonography OR is the pregnancy monochorionic?

病历：The patient has a twin pregnancy and is undergoing ultrasound assessment. Sonography explicitly cannot determine chorionicity with reasonable accuracy.

编辑：[{"old": "Is chorionicity determined with reasonable accuracy on sonography?", "new": "Is chorionicity determined with reasonable accuracy on sonography OR is the pregnancy monochorionic?"}]

| 方法 | 是减否分数 | 按保存标签正确 |
| --- | --- | --- |
| full | 5.5 | True |
| aster | -0.25 | False |
| direct | -13.5 | False |
| logistic_direct | 0.351330554688789 | True |
| per_type_mean | 4.0 | True |
| no_C | 0.25 | True |
| orthogonal | 1.0 | True |
| random_13 | 4.5 | True |
| random_37 | 5.5 | True |
| random_73 | 1.0 | True |

## 唯一C退化的构造错误

2498/G0原条件是绒毛膜性是否能由超声合理确定；患者明确不能确定。新条件增加“或者为单绒毛膜妊娠”。病历没有给出单绒毛膜事实；不能确定绒毛膜性不等价于已知单绒毛膜。

保存的修改谓词：

~~~json
{
  "op": "or",
  "args": [
    {
      "op": "==",
      "field": "chorionicity_determined",
      "value": true
    },
    {
      "op": "==",
      "field": "chorionicity_determined",
      "value": false
    }
  ]
}
~~~

程序实际计算A或非A，恒为真；文本却是A或B。树的处置“无法确定时按单绒毛膜双胎建议管理”也不是患者的单绒毛膜事实。标签1缺乏依据，不能简单改成0；应将这条C变体标为无效或重构有明确真值的版本。

full +5.5→ASTER −0.25的分数移动是真实观测，但“退化”依赖无效标签。no_C +0.25、orthogonal +1.0、mean +4.0及随机版本为正，只表明这些算子在该输入上响应不同，不能证明保护C的机制伤害了真实信号。

## 无效C的敏感性分析

主项排除独立文字检查发现的3个明确无效测试C：1633/G0（未迅速进展不推出非迅速进展截瘫）、2498/G0（A或B被编码成A或非A）、11/G2（重复头痛例子没有真值翻转）。扩展项再排除1918/G0（缺少明确远处转移事实）。这些不是按照模型输赢筛选；原结果完整保留。

| 敏感性范围 | full PA_C | ASTER PA_C | full宏平均 | ASTER宏平均 | 差值（百分点） | 差值95%区间 |
| --- | --- | --- | --- | --- | --- | --- |
| three_clear_invalid_C | 15/16 | 15/16 | 96.792% | 97.302% | +0.5102 | [+0.0000, +1.5625] |
| plus_1918_insufficient_evidence | 14/15 | 14/15 | 96.688% | 97.198% | +0.5102 | [+0.0000, +1.5625] |

这仅展示明显构造问题对结论的影响，不替代原主表，也不能据此宣称ASTER提升。

## 所有错误视图与对照

表内列full或ASTER至少一方错误的所有视图；完整条件、病历、编辑与分数见behavior.json/error_views。logistic_direct是固定的拟合数据训练输出方法，测试点估计较好；实际路径的开发优选方法是cad_direct（α=4）。

| 视图 | 标签 | full | aster | direct | logistic_direct | per_type_mean | no_C | orthogonal | random_13 | random_37 | random_73 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanded-confirmation-1633-G0:condition | 1 | 错(-24.875) | 错(-24.875) | 错(-17.25) | 错(-4.958715150659814) | 错(-26.875) | 错(-24.25) | 错(-24.25) | 错(-24.625) | 错(-24.875) | 错(-25.0) |
| expanded-confirmation-2303-G0:P_clinical1 | 0 | 错(0.5) | 对(-4.0) | 对(-18.0) | 对(-0.6618643417030236) | 对(-0.5) | 对(-3.75) | 对(-5.25) | 对(-0.75) | 错(0.75) | 对(-3.5) |
| expanded-confirmation-2303-G0:P_preference1 | 0 | 错(0.5) | 对(-3.5) | 对(-16.875) | 对(-0.621556871580687) | 对(0.0) | 对(-3.25) | 对(-4.25) | 对(-0.75) | 错(1.0) | 对(-2.5) |
| expanded-confirmation-2303-G4:swap1 | 1 | 错(0.0) | 错(-1.0) | 对(19.25) | 对(0.5875642785352094) | 对(1.0) | 错(-3.0) | 错(-3.5) | 对(0.25) | 对(0.25) | 对(1.0) |
| expanded-confirmation-2303-G4:P_clinical0 | 1 | 错(-3.25) | 错(-4.75) | 对(17.0) | 错(-0.046827922046066714) | 错(-1.75) | 错(-6.75) | 错(-6.25) | 错(-3.0) | 错(-2.75) | 错(-1.5) |
| expanded-confirmation-2303-G4:P_clinical1 | 1 | 错(-3.75) | 错(-5.5) | 对(16.25) | 错(-0.15889607038453796) | 错(-2.25) | 错(-7.25) | 错(-6.75) | 错(-3.75) | 错(-3.75) | 错(-2.25) |
| expanded-confirmation-2303-G4:P_preference1 | 1 | 错(-3.25) | 错(-3.75) | 对(20.75) | 对(0.08753031169505539) | 错(-1.0) | 错(-5.25) | 错(-5.25) | 错(-2.75) | 错(-3.0) | 错(-1.5) |
| expanded-confirmation-2498-G0:condition | 1 | 对(5.5) | 错(-0.25) | 错(-13.5) | 对(0.351330554688789) | 对(4.0) | 对(0.25) | 对(1.0) | 对(4.5) | 对(5.5) | 对(1.0) |
| expanded-confirmation-11-G2:condition | 0 | 错(29.0) | 错(28.5) | 错(28.5) | 错(5.860378347330947) | 错(29.25) | 错(28.25) | 错(28.5) | 错(28.75) | 错(28.5) | 错(28.75) |
| expanded-confirmation-1330-G20:base0 | 0 | 错(24.5) | 错(24.5) | 错(21.5) | 错(4.842807796701967) | 错(24.75) | 错(24.25) | 错(24.25) | 错(24.75) | 错(24.75) | 错(24.5) |
| expanded-confirmation-1330-G20:swap0 | 0 | 错(24.0) | 错(24.25) | 错(21.5) | 错(4.757611295111721) | 错(24.5) | 错(24.25) | 错(24.0) | 错(24.25) | 错(24.25) | 错(24.0) |
| expanded-confirmation-2791-G0:condition | 0 | 错(34.125) | 错(32.75) | 错(32.375) | 错(6.872479330163468) | 错(34.25) | 错(33.0) | 错(32.625) | 错(34.25) | 错(34.375) | 错(33.375) |

## 路径95与96的具体差异

| 总病例 | 两者正确 | 纠正 | 退化 | 两者错误 |
| --- | --- | --- | --- | --- |
| 100 | 95 | 0 | 1 | 4 |

### path-2432-11（train_val:2432）

病历：The patient presented with a mesenteric mass attributed to a desmoid tumor but had no mechanical or surgical complication, with no evidence of bowel occlusion, digestive perforation, or mesenteric ischemia. A histological diagnosis was obtained via percutaneous core-needle biopsy when anatomically possible, rather than relying on cytology. Molecular testing was performed as first-line evaluation to assess for a somatic pathogenic *CTNNB1* variant, and no pathogenic *CTNNB1* variant was detected. Given the negative *CTNNB1* result, constitutional testing for a pathogenic *APC* variant was performed and returned positive, consistent with APC-linked familial adenomatous polyposis. Even in the context of proven APC-linked FAP and expert-center imaging, biopsy was not considered potentially morbid in this case. There was strong clinical concern for an alternative diagnosis such as lymphoma or carcinoma in the setting of an FAP patient with a suggestive mesenteric mass, yet percutaneous biopsy was not feasible and endoscopic biopsy was not required because a percutaneous route had been achievable. The primary clinical focus was management.

唯一分歧节点G23：Is percutaneous biopsy feasible (approach to be discussed between radiologist and surgeon) or is endoscopic biopsy required because percutaneous route is not possible?
full=0.0，同分判否；ASTER=1.75，改判是。

| 方法 | 路径正确 | 终点 | 最后节点分数 |
| --- | --- | --- | --- |
| full | True | L25 | 0.0 |
| aster | False | L24 | 1.75 |
| direct | False | L24 | 2.75 |
| semantic | False | L24 | 11.5 |
| best_output | True | L25 | -11.0 |

该来源2432不同于P收益2303和C无效配对2498。病例同时提到当前经皮活检不可行、此前途径曾可行，时间关系未清楚分开。保存N与当前“不可行、无需内镜”的直接文字一致；保留金标，仅标注措辞需核验。不能从一个靠近决策边界的样本推出普遍损害。

## 数据难度与不确定性

原输入full已达199/200，最多还可提高0.5个百分点；全部615视图full错11个，ASTER错10个。A/F与原输入已接近本样本池的正确率上限。原计划的定义子集实际为零，开发C只有3对，数量更少的C仍承担宏平均25%权重，选参和排序容易受少数构造标签影响。

原主指标差值95%来源重采样区间[-4.1881,+1.3514]个百分点，包含0。两项P收益属于同一来源，不能按独立病例解释。

full绝对分数中位数：原输入25.250，全部视图25.750；全部视图|分数|≤1的有5个。这只是当前输入的分数间隔，不能代表医学任务整体难度。

平均分、局部干预响应、构造标签有效性与真实临床判断收益需要分别解释。当前证据不足以归结为单一的“擦除失败”或“保护机制失败”。


## 尚未纠正的有效错误

2303/G4问抑郁症状是否与抗癫痫药物有关，患者明确有专科评估将症状归因于该药。full在swap1、临床背景两侧及偏好1共4个视图错误；ASTER的分数分别由0、−3.25、−3.75、−3.25变为−1、−4.75、−5.5、−3.75，均未纠正。直接遮挡在这4个视图均正确，分数为+16.25至+20.75。这是本轮编辑没有解决、且遮挡可改善的具体案例；单凭它不能确定未覆盖方向、参考均值或跨层变化中哪一项是原因。

另外两类错误出现在严格不等号边界。1330/G20患者曲张静脉直径为2.0 cm，条件要求>2 cm，另两项风险条件也均不成立；full与ASTER原输入分数均为+24.5，直接遮挡为+21.5。2791/G0的有效C条件要求血红蛋白<55 g/L，患者恰为55；full、ASTER、直接遮挡分别为+34.125、+32.75、+32.375，全部错误。信号探针也分别给出约+16.8、+22.54。现有证据不支持把这两例归因于处置或背景干扰；规则比较或词义先验等解释需要另做针对性实验。

逐类型均值差和随机种子13额外纠正2303/G4的swap1，但它们相对ASTER的宏平均优势还依赖无效2498/G0的标签。单看这些方法的测试宏平均相近，不能否定选定头对干扰效应的恢复证据。

H3的4/11是按旧保存标签计算的探针准确率，11个full错误视图中含1633/G0与11/G2这两个明确无效C。该数值不能直接解释为有效样本上的信号可读性，更不能据有限线性探针未能纠正就断言真实判断信息不存在。
