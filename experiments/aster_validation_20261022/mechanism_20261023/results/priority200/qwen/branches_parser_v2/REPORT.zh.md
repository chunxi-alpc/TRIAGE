# 分支与动作机制实验

病例数：200；K：76；U：124。K仅表示原闭卷动作偏好。

## 1. 分支失败与动作竞争

分支和动作是独立调用；以下为配对诊断，不等于同一次推理的内部轨迹。
|版本/条件|题数|分支正确|分支正确且动作错误|已判断正确时动作错误率|
|---|---:|---:|---:|---:|
|A/hidden/all|200|133|40|30.1%|
|A/hidden/unique_action_branch|200|133|40|30.1%|
|A/visible/all|200|140|41|29.3%|
|A/visible/unique_action_branch|200|140|41|29.3%|
|A/hide_inapp/all|200|143|44|30.8%|
|A/hide_inapp/unique_action_branch|200|143|44|30.8%|
|B/hidden/all|200|133|71|53.4%|
|B/hidden/unique_action_branch|200|133|71|53.4%|
|B/visible/all|200|104|40|38.5%|
|B/visible/unique_action_branch|200|104|40|38.5%|
|B/hide_inapp/all|200|138|74|53.6%|
|B/hide_inapp/unique_action_branch|200|138|74|53.6%|
|C/hidden/all|200|133|40|30.1%|
|C/hidden/unique_action_branch|200|133|40|30.1%|
|C/visible/all|200|143|45|31.5%|
|C/visible/unique_action_branch|200|143|45|31.5%|
|C/hide_inapp/all|200|143|44|30.8%|
|C/hide_inapp/unique_action_branch|200|143|44|30.8%|
## 2. 动作归属是否改变分支判断
主要比较A动作可见与D动作可见；有害翻转为原先正确、交换后选中兄弟分支。
|分组|有效配对|翻转率|有害/反向|净有害率|疾病簇95%区间|Holm p|
|---|---:|---:|---:|---:|---|---:|
|branch_visible/all/all|200|35.5%|22/6|8.0%|[0.047117622052177024, 0.12354195166438663]|0.1302|
|branch_visible/all/K|76|40.8%|12/2|13.2%|[0.04912269467213115, 0.21052631578947367]|0.427|
|branch_visible/all/U|124|32.3%|10/4|4.8%|[0.01052088985349973, 0.09244086523498286]|1|
|branch_visible/judgment/all|60|46.7%|5/1|6.7%|[0.038461538461538464, 0.14285714285714285]|1|
|branch_visible/judgment/K|18|50.0%|3/0|16.7%|[0.0625, 0.3076923076923077]|1|
|branch_visible/judgment/U|42|45.2%|2/1|2.4%|[0.0, 0.08108108108108109]|1|
|branch_visible/objective_candidate/all|59|20.3%|6/1|8.5%|[0.02564102564102564, 0.1860828488372091]|1|
|branch_visible/objective_candidate/K|27|25.9%|5/0|18.5%|[0.06896551724137931, 0.34782608695652173]|1|
|branch_visible/objective_candidate/U|32|15.6%|1/1|0.0%|[0.0, 0.0]|1|
|branch_visible/ambiguous/all|76|36.8%|9/4|6.6%|[-0.016964020220041627, 0.18184589800443443]|1|
|branch_visible/ambiguous/K|31|48.4%|4/2|6.5%|[-0.10354064039408867, 0.24324324324324326]|1|
|branch_visible/ambiguous/U|45|28.9%|5/2|6.7%|[0.0, 0.16666666666666666]|1|
|branch_visible/treatment_judgment/all|5|60.0%|2/0|40.0%|None|1|
|branch_visible/treatment_judgment/U|5|60.0%|2/0|40.0%|None|1|
未达到预设系统性偏移门槛，不执行注意力阻断；这不等于证明不存在效应。
## 3. 隐藏动作与规则隔离的收益
|版本/比较|配对数|准确率差|疾病簇95%区间|Holm p|
|---|---:|---:|---|---:|
正差值表示隐藏动作裁剪较高；区间跨0则尚不能确认收益。A不下降须检查A差值区间，不能由不显著推出。
## K组调和平均
CS仅2题且同一疾病；BCDCS主指标仅作描述，不作总体排名。
|方法/指标|共同K题数|值|疾病簇95%区间|
|---|---:|---:|---|
|existing_app/primary_BCDCS|0|NA|None|
|existing_app/secondary_BCD|0|NA|None|
|existing_baseline/primary_BCDCS|0|NA|None|
|existing_baseline/secondary_BCD|0|NA|None|
|existing_prune/primary_BCDCS|0|NA|None|
|existing_prune/secondary_BCD|0|NA|None|
|existing_signed/primary_BCDCS|0|NA|None|
|existing_signed/secondary_BCD|0|NA|None|
完整各版本准确率、原动作/误用/诱饵率、解析失败率及配对检验见summary.json；逐题数值见scores.jsonl。注意力阶段另见attention目录，CAD原结果只读审计另见cad_finished_audit.json。
