> Source snapshot: /home/gcx/aster_bottleneck_20261031/structure_followup_20261101/regression_analysis_20261101.zh.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# 只抑制编辑额外弄错的9例：已有输出逐例分析

日期：2026-11-01。只读取开发100例、模块二判断和已完成的8条件输出；没有启动模型、重新生成或修改原评分。

比较：suppress_structured 对 zero_structured。两者采用完全相同的合法路径生成，主要指标为路径和最终处置同时正确。零编辑82/100，只抑制77/100；编辑纠正4例、退化9例。9例来自6个来源，不是9个独立来源。

## 结论

按现有金标，4例是已接受门控分支与金标不符，编辑在同一节点跟随了该错误门控；3例是模块二完整路径正确，但编辑反而走入被施加负增益的兄弟规则；2例是正确前缀门控后，未直接编辑的下一组判断退化。

其中4例“门控错误”里，365:3和948:2存在时间或文本冲突，不能把基准标签差异直接解释为临床判断错误。3例完整正确门控的失败是不能全部归咎模块二的直接证据；2例未编辑组失败表明保留子树KV并不保证该子树的后续判断不受影响。

## 九例索引

| 病例（train_val） | 第一次分歧 | 模块二 | 只抑制输出 | 分类 |
|---|---|---|---|---|
| 1170:5 | G0：应R0N | R0Y | R0Y | 错误门控 |
| 396:17 | G1：应R1N | R1Y | R1Y | 错误门控 |
| 396:38 | G70：应R70Y | R70Y | R70N | 完整正确门控后的编辑退化 |
| 348:23 | G39：应R39Y | R39Y | R39N | 完整正确门控后的编辑退化 |
| 365:10 | G17：应R17Y | R17Y | R17N | 完整正确门控后的编辑退化 |
| 365:3 | G2：应R2N | R2Y | R2Y | 错误门控 |
| 604:9 | G6：应R6N | U | R6Y | 正确前缀之后的间接退化 |
| 604:12 | G6：应R6N | U | R6Y | 正确前缀之后的间接退化 |
| 948:2 | G3：应R3Y | R3N | R3N | 错误门控 |

## mega:train_val:1170:5

病历先写入院时怀疑髋部骨折，随后描述阴性X线后没有支持持续怀疑的表现。当前条件问的是阴性X线后是否仍怀疑。模块二给R0Y，而零编辑选择金标R0N。可能混淆初始怀疑与当前条件；该原因由文字对照推断，并非注意力机制证据。

- 模块二已接受路径：R0Y。
- 首次分歧组的模块二输出：R0Y，受限标签概率0.999998230，间隔0.999996470；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：True；实际错误分支是否施加负增益：False。
- 全指南34条规则，负增益覆盖31条；非路径词元3743，总输入词元4107。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0N,R4Y,R5Y,R6N,R8N,R10Y | L11 | True |
| suppress_structured | R0Y,R1N | L3 | False |
| mean_structured | R0Y,R1N | L3 | False |
| clip_structured | R0Y,R1Y | L2 | False |
| zero_free | R0N,R4Y,R5Y,R6N,R8N,R10Y | L11 | True |
| suppress_free | R0Y,R1N | L3 | False |
| mean_free | R0Y,R1N | L3 | False |
| clip_free | R0Y,R1Y | L2 | False |

患者原文：

> The patient presented to hospital with a suspected hip fracture. Standard hip radiographs of adequate quality did not show findings that raised ongoing concern for a hip fracture despite the negative X-rays. On initial assessment, the patient was in pain. Preoperatively, paracetamol was not contraindicated and was administered, but paracetamol alone did not provide sufficient analgesia. Opioid analgesia was then added to paracetamol, and this combination provided adequate preoperative pain relief. The primary clinical focus was management.

当前条件：Is hip fracture suspected despite negative X-rays of the hip of an adequate standard?（判断为是）

## mega:train_val:396:17

病历明确写复发风险不符合高危、归入中危，随后又写存在高危提示因素。当前条件是风险类别是否高危。模块二把R1N改判R1Y，置信度0.562、间隔0.124；编辑和裁剪都沿错误分支。

- 模块二已接受路径：R0Y,R1Y。
- 首次分歧组的模块二输出：R1Y，受限标签概率0.562040401，间隔0.124322897；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：True；实际错误分支是否施加负增益：False。
- 全指南78条规则，负增益覆盖48条；非路径词元6463，总输入词元11192。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0Y,R1N,R31Y,R32Y,R33N | L35 | True |
| suppress_structured | R0Y,R1Y,R2N,R10Y,R11N,R17N | L19 | False |
| mean_structured | R0Y,R1Y,R2N,R10Y,R11N,R17N | L19 | False |
| clip_structured | R0Y,R1Y,R2N,R10Y,R11N,R17N | L19 | False |
| zero_free | R0Y,R1N,R31Y,R32Y,R33N | L35 | True |
| suppress_free | R0Y,R1Y,R10Y,R11N,R17N | L19 | False |
| mean_free | R0Y,R1Y,R10Y,R11N,R17N | L19 | False |
| clip_free | R0Y,R1Y,R10Y,R11N,R17N | L19 | False |

患者原文：

> 患者为分化型甲状腺癌（DTC，属于乳头状或滤泡状甲状腺癌谱系）术后病例，已完成初次手术治疗。结合术后病理及临床资料评估，复发风险不符合高危标准，但被归入中危人群。进一步梳理中危相关特征时，患者合并至少一项高危提示因素，表现为颈部淋巴结转移灶较多，清扫标本中转移淋巴结为 8 枚（>5枚），提示需在中危框架下重点关注后续治疗策略的适配性。
> 
> 在拟定放射性碘治疗方案的可行性评估中，患者不符合接受¹³¹I治疗的适用条件：当时处于妊娠状态，因此不适合行¹³¹I治疗（亦不满足“非妊娠、非哺乳且近期无妊娠计划”的要求）。本病例的主要临床关注点为管理。

当前条件：根据术后病理及临床资料，患者DTC复发风险是否属于高危？（判断为是）

## mega:train_val:396:38

病历明确提示ATC或其他高度侵袭性非分化型肿瘤。模块二完整路径正确，零编辑在G70选Y；只抑制和双向编辑都选N，裁剪仍选Y。被选中的R70N恰是负增益覆盖的分支。病历下游对转移与分期的描述另有张力，但不影响这个首次分歧处的明确诊断表述。

- 模块二已接受路径：R0N,R52N,R70Y,R71N,R73N,R75Y。
- 首次分歧组的模块二输出：R70Y，受限标签概率0.999999999，间隔0.999999999；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：False；实际错误分支是否施加负增益：True。
- 全指南78条规则，负增益覆盖72条；非路径词元10109，总输入词元11213。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0N,R52N,R70Y,R71N,R73N,R75Y | L76 | True |
| suppress_structured | R0N,R52N,R70N | L78 | False |
| mean_structured | R0N,R52N,R70N | L78 | False |
| clip_structured | R0N,R52N,R70Y,R71N,R73N,R75Y | L76 | True |
| zero_free | R0N,R52N,R70Y,R71N,R73N,R75Y | L76 | True |
| suppress_free | R0N,R70N,R71N,R73N,R75Y | L76 | False |
| mean_free | R0N,R70N,R71N,R73N,R75Y | L76 | False |
| clip_free | R0N,R52N,R70Y,R71N,R73N,R75Y | L76 | True |

患者原文：

> 患者既往评估并不符合分化型甲状腺癌（如乳头状或滤泡状甲状腺癌）且已完成初次手术治疗的情形，也不符合甲状腺髓样癌并已接受或计划接受根治性手术治疗的情形。就诊时临床与病理特征提示为甲状腺未分化癌或其他高度侵袭性非分化型肿瘤。进一步颈部影像学评估显示局部肿瘤体积较大并呈局部进展，未见“仍较小且存在明确手术切除机会”的表现，同时存在远处转移灶，肺部可见多发结节，最大约12 mm，因此不符合“无广泛远处转移”的情况。患者当时未出现明显气管受压所致呼吸困难，也未提示迫近的气道梗阻风险。综合临床与影像学分期表现，病程符合IVA或IVB期（局部进展但尚无广泛远处转移）的管理情境。最后，本病例的主要临床关注点为管理。

当前条件：患者是否为甲状腺未分化癌（ATC）或其他高度侵袭性非分化型肿瘤？（判断为是）

## mega:train_val:348:23

病历明确新生儿／出生后超声已经完成并确认CSK。模块二在G39选Y且全路径正确；只抑制结构生成改选N，双向编辑及裁剪仍正确。只抑制自由生成也正确，其分隔形式是连字符，而结构生成使用逗号空格。

- 模块二已接受路径：R0Y,R1N,R39Y,R40Y,R41N,R47Y。
- 首次分歧组的模块二输出：R39Y，受限标签概率0.999999998，间隔0.999999997；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：False；实际错误分支是否施加负增益：True。
- 全指南64条规则，负增益覆盖58条；非路径词元12625，总输入词元13963。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0Y,R1N,R39Y,R40Y,R41N,R47Y | L48 | True |
| suppress_structured | R0Y,R1N,R39N | L51 | False |
| mean_structured | R0Y,R1N,R39Y,R40Y,R41N,R47Y | L48 | True |
| clip_structured | R0Y,R1N,R39Y,R40Y,R41N,R47Y | L48 | True |
| zero_free | R0Y,R1N,R2Y,R3N,R5N,R27Y,R28Y,R29N,R31N,R33N | L35 | False |
| suppress_free | R0Y,R1N,R39Y,R40Y,R41N,R47Y | L48 | True |
| mean_free | R0Y,R1N,R39Y,R40Y,R41N,R47Y | L48 | True |
| clip_free | R0Y,R1N,R39Y,R40Y,R41N,R47Y | L48 | True |

患者原文：

> Prenatal ultrasound raised concern for a congenital solitary kidney after abnormal fetal kidney/urinary tract morphology was detected. On subsequent specialized fetal ultrasonography, no extrarenal malformations were identified. Given the apparently isolated prenatal finding, a neonatal/postnatal renal and urinary tract ultrasound was performed and confirmed congenital solitary kidney, which was anatomically classified on imaging. However, postnatal ultrasound did not demonstrate a normal solitary kidney and urinary tract as expected: compensatory enlargement was absent by 2 years of age, with a measured kidney length around the 40th percentile for age, consistent with an abnormal solitary kidney/urinary tract on ultrasonography. The primary clinical focus was management.

当前条件：After isolated prenatal CSK diagnosis without extrarenal malformations, has neonatal/postnatal US been performed to confirm CSK or abnormal kidney/urinary tract morphology?（判断为是）

## mega:train_val:365:10

病历明确合并冠心病，G17的候选CVD列表包含冠心病。模块二全路径正确；只抑制结构生成改选R17N，双向与裁剪正确。只抑制自由生成在该处插入G17后选R17Y，而结构生成不包含组编号。

- 模块二已接受路径：R0Y,R1N,R17Y,R18Y,R19Y,R20N。
- 首次分歧组的模块二输出：R17Y，受限标签概率0.999976295，间隔0.999954850；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：False；实际错误分支是否施加负增益：True。
- 全指南32条规则，负增益覆盖26条；非路径词元3975，总输入词元5228。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0Y,R1N,R17Y,R18Y,R19Y,R20N | L22 | True |
| suppress_structured | R0Y,R1N,R17N,R27N | L31 | False |
| mean_structured | R0Y,R1N,R17Y,R18Y,R19Y,R20N | L22 | True |
| clip_structured | R0Y,R1N,R17Y,R18Y,R19Y,R20N | L22 | True |
| zero_free | R0Y,R1N,R2Y,R3N | L5 | False |
| suppress_free | R0Y,R1N,R17Y,R18Y,R19Y,R20N | L22 | True |
| mean_free | R0Y,R1N,R17Y,R18Y,R19Y,R20N | L22 | True |
| clip_free | R0Y,R1N,R17Y,R18Y,R19Y,R20N | L22 | True |

患者原文：

> 患者既往明确为心血管疾病（CVD）患者，临床合并冠心病。病程中未表现为难治性高血压，亦无已诊断的心房颤动或肺动脉高压。近来就诊时伴有阻塞性睡眠呼吸暂停（OSA）相关症状/体征，且OSA筛查量表提示阳性，STOP-Bang 评分为 4 分。随后完成了客观睡眠呼吸监测（连续记录时间≥4小时），结果显示呼吸暂停低通气指数（AHI）为 10 次/小时，阻塞性事件占比为 70%，并且伴有至少一种指南所列的临床表现；但未达到 AHI ≥15 次/小时的程度。以上信息用于管理。

当前条件：患者是否存在以下任一 CVD：心力衰竭、冠心病、病态窦房结综合征、室性心动过速或心原性猝死幸存？（判断为是）

## mega:train_val:365:3

前文说尚未完成连续至少4小时的完整客观证据链，后文说随后已完成PM，REI18、阻塞性事件70%。后文未明确监测时长及完整合取条件，时间更新也不清楚。模块二R2Y与金标R2N不符，但不能仅凭该不符把问题全部归为模型能力。

- 模块二已接受路径：R0Y,R1Y,R2Y,R3Y。
- 首次分歧组的模块二输出：R2Y，受限标签概率0.880790957，间隔0.761588864；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：True；实际错误分支是否施加负增益：False。
- 全指南32条规则，负增益覆盖28条；非路径词元4470，总输入词元5286。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0Y,R1Y,R2N,R6Y,R7Y,R8Y | L9 | True |
| suppress_structured | R0Y,R1Y,R2Y,R3Y | L4 | False |
| mean_structured | R0Y,R1Y,R2Y,R3Y | L4 | False |
| clip_structured | R0Y,R1Y,R2Y,R3Y | L4 | False |
| zero_free | R0Y,R1Y,R2N,R6Y,R7Y,R8Y | L9 | True |
| suppress_free | R0Y,R1Y,R2Y,R3Y | L4 | False |
| mean_free | R0Y,R1Y,R2Y,R3Y | L4 | False |
| clip_free | R0Y,R1Y,R2Y,R3Y | L4 | False |

患者原文：

> 患者既往已明确为心血管疾病（CVD）患者，并合并存在心房颤动。在随访评估中，患者虽尚未接受连续≥4小时的客观睡眠呼吸监测（PSG或PM）来证实是否满足“AHI/REI≥5次/h且阻塞性事件占≥50%并伴指南所列临床表现”等完整客观证据链，但已完成阻塞性睡眠呼吸暂停（OSA）的初步筛查（结合病史、体格检查及STOP-Bang或其他量表评估），临床高度怀疑合并OSA。随后已安排并完成便携式睡眠监测（PM，优先采用Ⅱ型或Ⅲ型设备）进一步评估，监测结果显示REI为18次/h，阻塞性事件占比为70%，符合“AHI/REI≥15次/h且阻塞性事件占≥50%”的标准，从而提示存在达到管理评估阈值的睡眠呼吸相关异常负荷。该病例的主要临床关注点为管理。

当前条件：是否已进行客观睡眠呼吸监测（PSG 或 PM，连续≥4小时）并证实 AHI 或 REI ≥5 次/h，且阻塞性事件占≥50%，且患者存在至少一种指南列出的临床表现？（判断为是）

## mega:train_val:604:9

模块二正确接受R0Y,R1Y,R2N，在G6输出U而停止。病历写未达到主症加次症等证候标准。零编辑选R6N，只抑制结构生成选R6Y；R6Y和R6N都未被直接编辑。只抑制自由生成正确，双向及结构裁剪错误。病历还有已存在脂肪性肝病与未病阶段的冲突，但不能解释为模块二把R6Y选入门控。

- 模块二已接受路径：R0Y,R1Y,R2N。
- 首次分歧组的模块二输出：U，受限标签概率0.999992111，间隔0.999984221；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：False；实际错误分支是否施加负增益：False。
- 全指南30条规则，负增益覆盖5条；非路径词元550，总输入词元6285。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0Y,R1Y,R2N,R6N,R16Y,R17Y,R18Y,R19Y,R20Y,R21N | L23 | True |
| suppress_structured | R0Y,R1Y,R2N,R6Y,R7N,R9N,R11N,R13Y | L14 | False |
| mean_structured | R0Y,R1Y,R2N,R6Y,R7N,R9N,R11N,R13Y | L14 | False |
| clip_structured | R0Y,R1Y,R2N,R6Y,R7N,R9N,R11N,R13N | L15 | False |
| zero_free | R0Y,R1Y,R2N,R6N,R16Y,R17Y,R18Y,R19Y,R20Y,R21N | L23 | True |
| suppress_free | R0Y,R1Y,R2N,R6N,R16Y,R17Y,R18Y,R19Y,R20Y,R21N | L23 | True |
| mean_free | R0Y,R1Y,R2N,R6Y,R7N,R9N,R11N,R13Y | L14 | False |
| clip_free | R0Y,R1Y,R2N,R6Y,R7N,R9Y | L10 | False |

患者原文：

> 患者年龄≥16周岁。既往评估提示存在非酒精性单纯性脂肪性肝病，肝脂肪变累及约10%肝细胞，并已排除大量饮酒史、长期促脂肪形成药物使用及单基因遗传紊乱等其他导致肝脂肪变的原因。此次就诊时并未处于治疗后已痊愈阶段，亦未形成符合中医证候诊断要求的明确证候表现（未达到“主症+次症≥2项并结合舌脉特点”的标准）。
> 
> 当前处于未病/高危阶段，虽尚未获得明确影像学脂肪肝诊断，但已存在至少一项高危因素：腰围测量为92 cm，符合腹型肥胖。患者已接受健康教育与管理，包括对相关危害与预防措施的宣教，并已共同制定饮食与运动干预方案。随访中患者能够按建议执行科学饮食，进行总能量控制并调整膳食结构；同时坚持规律运动，每周运动4次、每次45分钟，以有氧运动为主，周累计约180分钟，并配合轻中度阻力性训练。然而，合并的糖尿病/高血压/高脂血症等共病并未做到积极且规律按医嘱药物治疗与指标监测。
> 
> 本病例的主要临床关注点为管理。

当前条件：患者是否具有中医证候诊断所需的“主症+次症≥2项，参考舌脉特点”的明确中医证候（肝郁气滞、肝郁脾虚、痰湿内阻或湿热蕴结）？（判断为是）

## mega:train_val:604:12

模块二同样在G6给U，只有前三个正确分支参与门控。病历写未形成明确证候归属。零编辑及裁剪选R6N，只抑制和双向编辑选R6Y；两个G6规则都未被直接编辑。病历另有既往影像／病理脂肪变与当前无明确影像诊断的张力。

- 模块二已接受路径：R0Y,R1Y,R2N。
- 首次分歧组的模块二输出：U，受限标签概率0.999999931，间隔0.999999863；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：False；实际错误分支是否施加负增益：False。
- 全指南30条规则，负增益覆盖5条；非路径词元550，总输入词元6225。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0Y,R1Y,R2N,R6N,R16Y,R17Y,R18N | L26 | True |
| suppress_structured | R0Y,R1Y,R2N,R6Y,R7N,R9N,R11N,R13Y | L14 | False |
| mean_structured | R0Y,R1Y,R2N,R6Y,R7N,R9N,R11N,R13Y | L14 | False |
| clip_structured | R0Y,R1Y,R2N,R6N,R16Y,R17Y,R18N | L26 | True |
| zero_free | R0Y,R1Y,R2N,R6N,R16Y,R17Y,R18N | L26 | True |
| suppress_free | R0Y,R1Y,R2N,R6Y,R11Y | L12 | False |
| mean_free | R0Y,R1Y,R2N,R6Y,R11Y | L12 | False |
| clip_free | R0Y,R1Y,R2N,R6N,R16Y,R17Y,R18N | L26 | True |

患者原文：

> 患者年龄为24岁。既往评估提示存在非酒精性单纯性脂肪性肝病，影像/病理所见肝细胞脂肪变约8%，并已排除大量饮酒史、长期使用促脂肪形成药物及单基因遗传紊乱等其他肝脂肪变原因。患者并非治疗后痊愈进入防复阶段。就中医证候方面，病程中未能形成符合“主症+次症≥2项并结合舌脉特点”的明确证候归属，未明确为肝郁气滞、肝郁脾虚、痰湿内阻或湿热蕴结。当前临床处于未病/高危阶段，即虽已存在相关危险因素但尚未获得明确影像学脂肪肝诊断；危险因素方面合并腹型肥胖，腰围测量为92 cm，属于高危因素之一。既往尚未系统接受针对该病危害与预防措施的健康宣教，也未制定并落实饮食与运动等生活方式干预方案。以上病例重点用于管理。

当前条件：患者是否具有中医证候诊断所需的“主症+次症≥2项，参考舌脉特点”的明确中医证候（肝郁气滞、肝郁脾虚、痰湿内阻或湿热蕴结）？（判断为是）

## mega:train_val:948:2

同一病历先写生命体征总体平稳，后写血氧93%、心率108并明确称生命体征异常。金标要求R3Y，模块二给R3N。两种表述与当前稳定性条件存在明显张力，不应把标签一致性直接当作医学判断正确性。

- 模块二已接受路径：R0Y,R1N,R3N。
- 首次分歧组的模块二输出：R3N，受限标签概率0.999541931，间隔0.999111386；这些不是校准后的错误风险。
- 首次分歧处，金标分支是否施加负增益：True；实际错误分支是否施加负增益：False。
- 全指南30条规则，负增益覆盖27条；非路径词元5222，总输入词元5825。

| 条件 | 生成路径 | 处置 | 两者正确 |
|---|---|---|---|
| zero_structured | R0Y,R1N,R3Y,R4Y | L5 | True |
| suppress_structured | R0Y,R1N,R3N | L29 | False |
| mean_structured | R0Y,R1N,R3N | L29 | False |
| clip_structured | R0Y,R1N,R3N | L29 | False |
| zero_free | R0Y,R1N,R3Y,R4Y | L5 | True |
| suppress_free | R0Y,R1N,R3N | L29 | False |
| mean_free | R0Y,R1N,R3N | L29 | False |
| clip_free | R0Y,R1N,R3N | L29 | False |

患者原文：

> 患者既往新型冠状病毒感染后，仍遗留多种相关不适，表现为头痛头昏、鼻塞、咽痛、嗅觉味觉减退及咳嗽，并伴乏力与畏寒等症状。就诊时已不处于发热阶段，体温测得为 36.8℃，提示体温已恢复正常。查体时生命体征总体平稳，体温≤37.2℃。但同时出现重症相关表现：主诉呼吸困难，血氧饱和度测得为 93%（＜95%），并有心率持续增快至 108 次/分（＞100 次/分），提示存在合并生命体征异常的情况。最后需围绕该患者新冠感染后持续症状并伴重症表现的情况进行管理。

当前条件：患者生命体征是否稳定，且体温正常（≤37.2℃）？（判断为是）

## 全100例分层，避免只看退化样例

| 模块二已接受路径 | 例数 | 零编辑 | 只抑制 | 双向编辑 | 裁剪 | 只抑制纠正/退化 |
|---|---:|---:|---:|---:|---:|---:|
| 完整正确门控 | 46 | 44 | 42 | 43 | 46 | 1/3 |
| 正确前缀 | 39 | 29 | 30 | 29 | 28 | 3/2 |
| 门控与金标不符 | 15 | 9 | 5 | 3 | 2 | 0/4 |

以上全为结构生成且路径与处置同时正确数。完整门控正确的46例中，零编辑44、只抑制42、双向43、裁剪46；正确门控仍不能保证键编辑产生正确行为。门控与金标不符的15例中，只抑制由零编辑9下降到5，说明错误接受确实会带来额外损失。

## 三个不能忽略的细节

1. **负增益不等于注意力一定减少。** 396:38、348:23、365:10都生成了负增益直接覆盖的错误兄弟。键位移经查询内积才成为注意力打分变化，最终生成行为还经过多层传播。现有结构生成日志保存允许词元及所选词元，没有保存这些节点的逐头查询、注意力位移或完整分支logit，无法据此断言具体是符号翻转、强度过大或患者信息读取变差。
2. **只抑制不是普遍优于双向。** 348:23和365:10的双向编辑正确，只抑制错误。三例348:23、365:10、604:9在只抑制自由生成时正确，在结构生成时错误。自由输出分别使用连字符路径、插入G17组编号、连字符路径；结构生成固定逗号空格。两种生成的查询内容和位置不同，存在编辑与生成格式的交互，但没有同一词元前缀下的对照，不能仅凭这三个样例把原因锁定为标点或RoPE。
3. **未直接编辑的子树仍可受影响。** 两个604病例的G6两侧都没有被编辑；模块二在此U停止，只编辑上层被排除的5条规则。当前实现先编辑指南缓存，再用它读取患者和生成路径，上层干预可能通过患者／路径表示以及后续查询间接影响G6。现有证据说明存在行为影响，不能分离这些传播路径。

## 用已有输出检查一个简单回退方案

考虑只在模块二非空路径是零编辑生成路径的严格前缀时采用编辑结果；若两者冲突或已经完整一致，则保留零编辑结果。此选择只使用线上可得的两条路径，不用金标。对全100例已有输出离线计算：26例采用编辑，联合正确82/100，相对零编辑纠正2例、退化2例，净变化0。

这不是新生成实验，也不是新方法的独立验证；它只说明“遇到分歧就回退”并没有从现有结果中带来净收益。它同时丢掉了1389:6和604:2两例原本可被编辑纠正的错误。不能只根据9例损失设计保护规则而忽略4例收益。

## 建议如何改

### 1. 保留零编辑＋合法路径生成为强基线，停止默认全局编辑

当前最强开发结果仍是82/100。只抑制77/100不能替代它作为最终方案。保留均值方向与三模块研究接口，但先不新增方向、不扩大头搜索，也不继续提高增益。

### 2. 模块二优先修正当前条件的证据使用，不拆复杂特征

1170:5要区分“曾经怀疑”与“阴性X线后仍怀疑”；396:17要区分“风险类别为高危”与“中危中存在提示因素”。可考虑让同一次大模型判断先引用一句针对当前条件的证据，再给T/F/U；显式区分未满足、未记录及前后冲突。证据仅来自病历和必要祖先条件，仍对候选处置盲。

365:3和948:2应先标记时间／文本歧义，保留原评分并给出敏感性说明，不能为了让模型答案变对而自动改标签，也不需要对所有样本另设审核流程。单纯提高概率阈值不能解决全部错误：1170:5的错误标签概率约0.999998，948:2约0.999542。

### 3. 模块三优先把全局持久干预改成当前决策的局部干预

建议下一项候选是：患者信息先由未编辑的共享指南缓存编码；在某个需要干预的决策点，只修改当前兄弟组的读取，避免一次修改整份指南中所有后代的键并持续影响所有生成位置。模块二为U的节点保持不编辑。

重要实现限制：不能仅把指南K恢复，就声称后续状态已无编辑影响。患者KV和已生成路径KV可能已受前面干预影响；若要严格比较局部效果，应从同一份未编辑指南／患者缓存及实际已选路径前缀构建对照，必要的重算成本需要计入。这里是待验证设计，当前没有新增结果。

### 4. 换算子与换方向要有区分

在这些“模块二已正确而键编辑选错”的位置，先检验实际打分变化是否符合负向目标。若不符合，再考虑当前组上的有限负打分偏置作为对照；不要把直接偏置的收益写成学到方向的收益。现有全局正负bias结果不能代替当前组负偏置，但本次不启动任何新实验。

## 结论范围

已有输出足以把9例分为4例门控不符、3例正确完整门控后退化、2例正确前缀后间接退化，并发现文字歧义和生成方式交互。它不足以确定各头的因果机制或证明某个新改法有效。最直接的下一步是先改干预发生的位置和持续范围，再决定是否需要改算子；不能把所有损失都推给模块二。
