# 可控读取：机制示范样本

这些解释性示范由运行前固定的规则从本轮结果中选出；总体分母和全部候选保存在 `case_selection_census.csv`。示范的成功标准只针对规则读取，不等于回答纠正，也不把精选子集作为总体收益估计。

共有 12 个指南/处置映射/顺序配置，每个配置包含两名金标相反的模拟患者。最高预设增益为 4；其中 12 个配置的两名患者均提高了正确/兄弟规则注意力质量比。

## 典型成功：接近成功配置的中位数

- 配置：`38d1513234f0f3f2060ce27e9c8e3d2c:node:23:swap0:order1`；来源：guideline:636。
- 处置置换：0；规则顺序：A → B。
- 决定性问题：Is the woman within the postpartum period (up to 6 weeks after delivery), with de novo or known hypertension?
- 两名患者均保留相同祖先条件，决定性事实分别为 YES 和 NO。
- 规则 A 处置：Treat as severe hypertension in a non-postpartum pregnant woman or non-pregnant context per local hypertension and obstetric protocols, including urgent blood pressure control and evaluation for hypertensive disorders of pregnancy if pregnant.
- 规则 B 处置：Treat as postpartum severe hypertension: urgently administer antihypertensive therapy to reduce blood pressure to <160/110 mm Hg and ideally <140/90 mm Hg; ensure systems for recognition and treatment of postpartum hypertension are in place; consider use of labetalol, nifedipine, enalapril, or captopril compatible with lactation; instruct on home blood pressure self-monitoring over the first 2 weeks postpartum.
- 两患者最小 Δlog(正确/兄弟规则质量)：1.56623。

- 金标 A：Δlog 质量比 1.58134；正确编号 logit 间隔变化 0.00231361；编号答案未改变。
- 金标 B：Δlog 质量比 1.56623；正确编号 logit 间隔变化 0.00774765；编号答案未改变。

## 清晰成功：两名患者中的较小效应最大

- 配置：`38d1513234f0f3f2060ce27e9c8e3d2c:node:23:swap1:order1`；来源：guideline:636。
- 处置置换：1；规则顺序：A → B。
- 决定性问题：Is the woman within the postpartum period (up to 6 weeks after delivery), with de novo or known hypertension?
- 两名患者均保留相同祖先条件，决定性事实分别为 YES 和 NO。
- 规则 A 处置：Treat as postpartum severe hypertension: urgently administer antihypertensive therapy to reduce blood pressure to <160/110 mm Hg and ideally <140/90 mm Hg; ensure systems for recognition and treatment of postpartum hypertension are in place; consider use of labetalol, nifedipine, enalapril, or captopril compatible with lactation; instruct on home blood pressure self-monitoring over the first 2 weeks postpartum.
- 规则 B 处置：Treat as severe hypertension in a non-postpartum pregnant woman or non-pregnant context per local hypertension and obstetric protocols, including urgent blood pressure control and evaluation for hypertensive disorders of pregnancy if pregnant.
- 两患者最小 Δlog(正确/兄弟规则质量)：1.61625。

- 金标 A：Δlog 质量比 1.61768；正确编号 logit 间隔变化 -0.0034523；编号答案未改变。
- 金标 B：Δlog 质量比 1.61625；正确编号 logit 间隔变化 0.00558662；编号答案未改变。

## 边界示范：其余配置中较小患者效应最弱

- 配置：`8be236bd48498ccf2bfed6161ec4baa6:node:3:swap0:order0`；来源：guideline:1517。
- 处置置换：0；规则顺序：B → A。
- 决定性问题：Is the patient in persistent pulmonary oedema or cardiogenic shock despite medical therapy?
- 两名患者均保留相同祖先条件，决定性事实分别为 YES 和 NO。
- 规则 A 处置：Perform emergent/urgent surgery for infective endocarditis with severe regurgitation/obstruction/fistula causing AHF.
- 规则 B 处置：Perform emergency surgery for infective endocarditis, irrespective of infection status.
- 两患者最小 Δlog(正确/兄弟规则质量)：1.47605。

- 金标 A：Δlog 质量比 1.47605；正确编号 logit 间隔变化 -0.00525093；编号答案未改变。
- 金标 B：Δlog 质量比 1.54159；正确编号 logit 间隔变化 -0.00270844；编号答案未改变。
