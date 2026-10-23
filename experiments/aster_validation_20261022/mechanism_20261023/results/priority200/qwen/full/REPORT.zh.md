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
|D/hidden/all|200|49|39|79.6%|
|D/hidden/unique_action_branch|200|49|39|79.6%|
|D/visible/all|200|16|11|68.8%|
|D/visible/unique_action_branch|200|16|11|68.8%|
|D/hide_inapp/all|200|41|33|80.5%|
|D/hide_inapp/unique_action_branch|200|41|33|80.5%|
|CS/hidden/all|2|1|0|0.0%|
|CS/hidden/unique_action_branch|2|1|0|0.0%|
|CS/visible/all|2|1|0|0.0%|
|CS/visible/unique_action_branch|2|1|0|0.0%|
|CS/hide_inapp/all|2|1|0|0.0%|
|CS/hide_inapp/unique_action_branch|2|1|0|0.0%|
## 2. 动作归属是否改变分支判断
主要比较A动作可见与D动作可见；有害翻转为原先正确、交换后选中兄弟分支。
|分组|有效配对|翻转率|有害/反向|净有害率|疾病簇95%区间|Holm p|
|---|---:|---:|---:|---:|---|---:|
|branch_visible/all/all|19|26.3%|1/1|0.0%|[-0.17647058823529413, 0.11538461538461539]|1|
|branch_visible/all/K|13|30.8%|1/1|0.0%|[-0.3, 0.15394736842105208]|1|
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
|A/existing_baseline->hidden_prune|200|-40.5%|[-0.5063318302819579, -0.270830455801105]|3.279e-16|
|A/existing_prune->hidden_prune|200|-74.0%|[-0.8591562007423602, -0.5647232214228621]|1.3e-42|
|A/gold_branch_prune->hidden_prune|200|-73.0%|[-0.8520066752945985, -0.5462294929157349]|5.157e-42|
|A/existing_signed->hidden_prune|200|-50.0%|[-0.6066438668018792, -0.3575749111223972]|1.776e-22|
|A/visible_prune->hidden_prune|200|11.5%|[0.03073891625615764, 0.23045516304347816]|0.004074|
|B/existing_baseline->hidden_prune|200|-23.5%|[-0.35830284266816775, -0.11600616234594144]|4.988e-06|
|B/existing_prune->hidden_prune|200|-65.0%|[-0.7760644168215614, -0.48240423098913726]|3.056e-37|
|B/gold_branch_prune->hidden_prune|200|-58.5%|[-0.7360042553191489, -0.38814144736842116]|2.431e-33|
|B/existing_signed->hidden_prune|200|-30.5%|[-0.4343989769820972, -0.1739084377302874]|2.519e-10|
|B/visible_prune->hidden_prune|200|10.5%|[0.02923340961098398, 0.20349594952055544]|0.02191|
|C/existing_baseline->hidden_prune|200|-41.5%|[-0.5304948405253284, -0.24441068139963185]|8.37e-16|
|C/existing_prune->hidden_prune|200|-73.5%|[-0.855389891696751, -0.5527156132947708]|2.59e-42|
|C/gold_branch_prune->hidden_prune|200|-72.5%|[-0.849772243763511, -0.5345475829775516]|1.022e-41|
|C/existing_signed->hidden_prune|200|-49.0%|[-0.6111183074265977, -0.3058556149732622]|6.296e-22|
|C/visible_prune->hidden_prune|200|10.0%|[0.01545420400973942, 0.20765225310841848]|0.05779|
|D/existing_baseline->hidden_prune|200|8.0%|[-0.04815831754634181, 0.26565104166666653]|1|
|D/existing_prune->hidden_prune|200|-65.5%|[-0.7837992809323084, -0.48683771929824565]|1.543e-37|
|D/gold_branch_prune->hidden_prune|200|-65.5%|[-0.7837992809323084, -0.48683771929824565]|1.543e-37|
|D/existing_signed->hidden_prune|200|3.0%|[-0.10033654720756777, 0.21547080396265955]|1|
|D/visible_prune->hidden_prune|200|16.5%|[0.06735751295336788, 0.3051009392205548]|1.334e-05|
|CS/existing_baseline->hidden_prune|2|0.0%|None|1|
|CS/existing_prune->hidden_prune|2|-50.0%|None|1|
|CS/gold_branch_prune->hidden_prune|2|-50.0%|None|1|
|CS/existing_signed->hidden_prune|2|0.0%|None|1|
|CS/visible_prune->hidden_prune|2|0.0%|None|1|
正差值表示隐藏动作裁剪较高；区间跨0则尚不能确认收益。A不下降须检查A差值区间，不能由不显著推出。
## K组调和平均
CS仅2题且同一疾病；BCDCS主指标仅作描述，不作总体排名。
|方法/指标|共同K题数|值|疾病簇95%区间|
|---|---:|---:|---|
|existing_app/primary_BCDCS|2|0.0%|None|
|existing_app/secondary_BCD|76|31.5%|[0.19463849153026985, 0.44032014385062807]|
|existing_baseline/primary_BCDCS|2|0.0%|None|
|existing_baseline/secondary_BCD|76|21.7%|[0.11186617893762184, 0.3061076592561691]|
|existing_prune/primary_BCDCS|2|100.0%|None|
|existing_prune/secondary_BCD|76|86.8%|[0.7621451377384739, 0.9482407790152119]|
|existing_signed/primary_BCDCS|2|0.0%|None|
|existing_signed/secondary_BCD|76|28.4%|[0.16207160797723338, 0.40709653437392185]|
|gold_branch_prune/primary_BCDCS|2|100.0%|None|
|gold_branch_prune/secondary_BCD|76|85.3%|[0.749761590606827, 0.9329666667351353]|
|hidden_prune/primary_BCDCS|2|50.0%|None|
|hidden_prune/secondary_BCD|76|28.6%|[0.14389218387391173, 0.47198084219137415]|
|hidden_signed/primary_BCDCS|2|0.0%|None|
|hidden_signed/secondary_BCD|76|14.6%|[0.059235857716911845, 0.2544555709341092]|
|hidden_top2/primary_BCDCS|2|50.0%|None|
|hidden_top2/secondary_BCD|76|22.8%|[0.10912590994528341, 0.37966596511957046]|
|hidden_unsigned/primary_BCDCS|2|0.0%|None|
|hidden_unsigned/secondary_BCD|76|11.1%|[0.030767543859649124, 0.2058221948620879]|
|visible_prune/primary_BCDCS|2|0.0%|None|
|visible_prune/secondary_BCD|76|13.6%|[0.04936876297635791, 0.241063309865348]|
完整各版本准确率、原动作/误用/诱饵率、解析失败率及配对检验见summary.json；逐题数值见scores.jsonl。注意力阶段另见attention目录，CAD原结果只读审计另见cad_finished_audit.json。
