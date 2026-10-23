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
|D/hidden/all|200|133|101|75.9%|
|D/hidden/unique_action_branch|200|133|101|75.9%|
|D/visible/all|200|93|67|72.0%|
|D/visible/unique_action_branch|200|93|67|72.0%|
|D/hide_inapp/all|200|130|100|76.9%|
|D/hide_inapp/unique_action_branch|200|130|100|76.9%|
|CS/hidden/all|2|2|1|50.0%|
|CS/hidden/unique_action_branch|2|2|1|50.0%|
|CS/visible/all|2|2|1|50.0%|
|CS/visible/unique_action_branch|2|2|1|50.0%|
|CS/hide_inapp/all|2|2|1|50.0%|
|CS/hide_inapp/unique_action_branch|2|2|1|50.0%|
## 2. 动作归属是否改变分支判断
主要比较A动作可见与D动作可见；有害翻转为原先正确、交换后选中兄弟分支。
|分组|有效配对|翻转率|有害/反向|净有害率|疾病簇95%区间|Holm p|
|---|---:|---:|---:|---:|---|---:|
|branch_visible/all/all|200|35.5%|22/6|8.0%|[0.04544812532100668, 0.1282079646017699]|0.4984|
|branch_visible/all/K|76|40.8%|12/2|13.2%|[0.047619047619047616, 0.20954887218045098]|1|
|branch_visible/all/U|124|32.3%|10/4|4.8%|[0.010101010101010102, 0.089289343786295]|1|
|branch_visible/judgment/all|60|46.7%|5/1|6.7%|[0.038461538461538464, 0.14705882352941177]|1|
|branch_visible/judgment/K|18|50.0%|3/0|16.7%|[0.0625, 0.3076923076923077]|1|
|branch_visible/judgment/U|42|45.2%|2/1|2.4%|[0.0, 0.08695652173913043]|1|
|branch_visible/objective_candidate/all|59|20.3%|6/1|8.5%|[0.027768264840182647, 0.1836734693877551]|1|
|branch_visible/objective_candidate/K|27|25.9%|5/0|18.5%|[0.06666666666666667, 0.34782608695652173]|1|
|branch_visible/objective_candidate/U|32|15.6%|1/1|0.0%|[0.0, 0.0]|1|
|branch_visible/ambiguous/all|76|36.8%|9/4|6.6%|[-0.02531645569620253, 0.18040300546448046]|1|
|branch_visible/ambiguous/K|31|48.4%|4/2|6.5%|[-0.10349364791288566, 0.24324324324324326]|1|
|branch_visible/ambiguous/U|45|28.9%|5/2|6.7%|[0.0, 0.16666666666666666]|1|
|branch_visible/treatment_judgment/all|5|60.0%|2/0|40.0%|None|1|
|branch_visible/treatment_judgment/U|5|60.0%|2/0|40.0%|None|1|
未达到预设系统性偏移门槛，不执行注意力阻断；这不等于证明不存在效应。
## 3. 隐藏动作与规则隔离的收益
|版本/比较|配对数|准确率差|疾病簇95%区间|Holm p|
|---|---:|---:|---|---:|
|A/existing_baseline->hidden_prune|200|12.0%|[0.037488425925925925, 0.18840579710144928]|0.4981|
|A/existing_prune->hidden_prune|200|-21.5%|[-0.29257908190435616, -0.13413492891613416]|4.161e-11|
|A/gold_branch_prune->hidden_prune|200|-21.5%|[-0.29257908190435616, -0.13413492891613416]|4.161e-11|
|A/existing_signed->hidden_prune|200|2.5%|[-0.0656978014773373, 0.09649122807017543]|1|
|A/visible_prune->hidden_prune|200|-2.5%|[-0.05797335203366059, 0.012196992368696683]|1|
|B/existing_baseline->hidden_prune|200|16.0%|[0.06479872251931075, 0.23348333574059363]|0.01923|
|B/existing_prune->hidden_prune|200|-25.5%|[-0.35761858611963604, -0.13743880527055885]|1.803e-13|
|B/gold_branch_prune->hidden_prune|200|-25.5%|[-0.35761858611963604, -0.13743880527055885]|1.803e-13|
|B/existing_signed->hidden_prune|200|9.0%|[0.0, 0.16956866506193258]|1|
|B/visible_prune->hidden_prune|200|8.5%|[0.025153301886792455, 0.13889157014157014]|1|
|C/existing_baseline->hidden_prune|200|10.5%|[0.015004119888763004, 0.19731089214066555]|1|
|C/existing_prune->hidden_prune|200|-21.5%|[-0.30131919201776, -0.12833490935176736]|4.161e-11|
|C/gold_branch_prune->hidden_prune|200|-21.5%|[-0.30131919201776, -0.12833490935176736]|4.161e-11|
|C/existing_signed->hidden_prune|200|3.0%|[-0.07596504732580682, 0.13334224598930478]|1|
|C/visible_prune->hidden_prune|200|-4.5%|[-0.08755760368663594, -0.0053475935828877]|1|
|D/existing_baseline->hidden_prune|200|43.0%|[0.3244614058097736, 0.5467751172196571]|1.923e-16|
|D/existing_prune->hidden_prune|200|-25.0%|[-0.346852146263911, -0.1538359788359789]|3.57e-13|
|D/gold_branch_prune->hidden_prune|200|-25.0%|[-0.346852146263911, -0.1538359788359789]|3.57e-13|
|D/existing_signed->hidden_prune|200|36.5%|[0.26141711229946524, 0.49411935704514365]|4.349e-12|
|D/visible_prune->hidden_prune|200|16.5%|[0.10587757054041129, 0.226856884057971]|0.0008659|
|CS/existing_baseline->hidden_prune|2|50.0%|None|1|
|CS/existing_prune->hidden_prune|2|0.0%|None|1|
|CS/gold_branch_prune->hidden_prune|2|0.0%|None|1|
|CS/existing_signed->hidden_prune|2|50.0%|None|1|
|CS/visible_prune->hidden_prune|2|0.0%|None|1|
正差值表示隐藏动作裁剪较高；区间跨0则尚不能确认收益。A不下降须检查A差值区间，不能由不显著推出。
## K组调和平均
CS仅2题且同一疾病；BCDCS主指标仅作描述，不作总体排名。
|方法/指标|共同K题数|值|疾病簇95%区间|
|---|---:|---:|---|
|existing_app/primary_BCDCS|2|0.0%|None|
|existing_app/secondary_BCD|76|34.3%|[0.2365304534162773, 0.4483782748049052]|
|existing_baseline/primary_BCDCS|2|0.0%|None|
|existing_baseline/secondary_BCD|76|23.4%|[0.13497468866358814, 0.32575613885313265]|
|existing_prune/primary_BCDCS|2|100.0%|None|
|existing_prune/secondary_BCD|76|86.8%|[0.7621451377384739, 0.9482407790152119]|
|existing_signed/primary_BCDCS|2|0.0%|None|
|existing_signed/secondary_BCD|76|33.0%|[0.20872156468571149, 0.4490975921814678]|
|gold_branch_prune/primary_BCDCS|2|100.0%|None|
|gold_branch_prune/secondary_BCD|76|86.8%|[0.7621451377384739, 0.9482407790152119]|
|hidden_prune/primary_BCDCS|2|100.0%|None|
|hidden_prune/secondary_BCD|76|61.4%|[0.5154861662509479, 0.721022378334406]|
|hidden_signed/primary_BCDCS|2|0.0%|None|
|hidden_signed/secondary_BCD|76|32.3%|[0.21935482514017246, 0.43262437877968885]|
|hidden_top2/primary_BCDCS|2|100.0%|None|
|hidden_top2/secondary_BCD|76|56.0%|[0.4421313027027733, 0.6778720355506225]|
|hidden_unsigned/primary_BCDCS|2|0.0%|None|
|hidden_unsigned/secondary_BCD|76|32.3%|[0.21156450732131063, 0.4350041006014214]|
|visible_prune/primary_BCDCS|2|66.7%|None|
|visible_prune/secondary_BCD|76|50.8%|[0.35674454876987455, 0.6516873458574534]|
完整各版本准确率、原动作/误用/诱饵率、解析失败率及配对检验见summary.json；逐题数值见scores.jsonl。注意力阶段另见attention目录，CAD原结果只读审计另见cad_finished_audit.json。
