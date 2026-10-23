# 分支与动作机制实验

病例数：200；K：76；U：124。K仅表示原闭卷动作偏好。

## 1. 分支失败与动作竞争

分支和动作是独立调用；以下为配对诊断，不等于同一次推理的内部轨迹。
|版本/条件|题数|分支正确|分支正确且动作错误|已判断正确时动作错误率|
|---|---:|---:|---:|---:|
|A/hidden/all|200|49|8|16.3%|
|A/hidden/unique_action_branch|200|49|8|16.3%|
|A/visible/all|200|28|4|14.3%|
|A/visible/unique_action_branch|200|28|4|14.3%|
|A/hide_inapp/all|200|46|11|23.9%|
|A/hide_inapp/unique_action_branch|200|46|11|23.9%|
|B/hidden/all|200|49|22|44.9%|
|B/hidden/unique_action_branch|200|49|22|44.9%|
|B/visible/all|200|22|11|50.0%|
|B/visible/unique_action_branch|200|22|11|50.0%|
|B/hide_inapp/all|200|43|21|48.8%|
|B/hide_inapp/unique_action_branch|200|43|21|48.8%|
|C/hidden/all|200|49|10|20.4%|
|C/hidden/unique_action_branch|200|49|10|20.4%|
|C/visible/all|200|30|3|10.0%|
|C/visible/unique_action_branch|200|30|3|10.0%|
|C/hide_inapp/all|200|46|12|26.1%|
|C/hide_inapp/unique_action_branch|200|46|12|26.1%|
## 2. 动作归属是否改变分支判断
主要比较A动作可见与D动作可见；有害翻转为原先正确、交换后选中兄弟分支。
|分组|有效配对|翻转率|有害/反向|净有害率|疾病簇95%区间|Holm p|
|---|---:|---:|---:|---:|---|---:|
|branch_visible/all/all|19|26.3%|1/1|0.0%|[-0.1766042780748663, 0.11764705882352941]|1|
|branch_visible/all/K|13|30.8%|1/1|0.0%|[-0.2857142857142857, 0.15789473684210525]|1|
|branch_visible/all/U|6|16.7%|0/0|0.0%|[0.0, 0.0]|1|
|branch_visible/judgment/all|2|0.0%|0/0|0.0%|None|1|
|branch_visible/judgment/K|1|0.0%|0/0|0.0%|None|1|
|branch_visible/judgment/U|1|0.0%|0/0|0.0%|None|1|
|branch_visible/objective_candidate/all|10|10.0%|1/0|10.0%|[0.0, 0.1875]|1|
|branch_visible/objective_candidate/K|8|12.5%|1/0|12.5%|[0.0, 0.25]|1|
|branch_visible/objective_candidate/U|2|0.0%|0/0|0.0%|[0.0, 0.0]|1|
|branch_visible/ambiguous/all|7|57.1%|0/1|-14.3%|[-0.42857142857142855, 0.0]|1|
|branch_visible/ambiguous/K|4|75.0%|0/1|-25.0%|[-1.0, 0.0]|1|
|branch_visible/ambiguous/U|3|33.3%|0/0|0.0%|[0.0, 0.0]|1|
|branch_visible/treatment_judgment/all|0|NA|0/0|NA|None|1|
|branch_visible/treatment_judgment/U|0|NA|0/0|NA|None|1|
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
