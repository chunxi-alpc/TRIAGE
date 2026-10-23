"""Readable numerical report, without turning uncertain associations into claims."""
def render(r):
 def fmt(x):return 'NA' if x is None else f'{100*x:.1f}%'
 lines=['# 分支与动作机制实验',f"病例数：{r['n_cases']}；K：{r['K']}；U：{r['U']}。K仅表示原闭卷动作偏好。",'## 1. 分支失败与动作竞争','分支和动作是独立调用；以下为配对诊断，不等于同一次推理的内部轨迹。','|版本/条件|题数|分支正确|分支正确且动作错误|已判断正确时动作错误率|','|---|---:|---:|---:|---:|']
 for k,x in r['decomposition'].items():
  if x['n']:lines.append(f"|{k}|{x['n']}|{x['branch_correct']}|{x['branch_correct_action_wrong']}|{fmt(x['action_error_given_branch_correct'])}|")
 lines+=['## 2. 动作归属是否改变分支判断','主要比较A动作可见与D动作可见；有害翻转为原先正确、交换后选中兄弟分支。','|分组|有效配对|翻转率|有害/反向|净有害率|疾病簇95%区间|Holm p|','|---|---:|---:|---:|---:|---|---:|']
 for k,x in r['action_induced_branch_shift'].items():
  if k.startswith('branch_visible/') and x['n']:lines.append(f"|{k}|{x['valid_pairs']}|{fmt(x['flip_rate'])}|{x['harmful']}/{x['reverse']}|{fmt(x['net_harmful_rate'])}|{x['net_disease_ci']}|{x['p_holm']:.4g}|")
 lines += ['满足预设系统性偏移门槛；注意力因果结论仍需阻断结果。' if r['attention_gate']['triggered'] else '未达到预设系统性偏移门槛，不执行注意力阻断；这不等于证明不存在效应。','## 3. 隐藏动作与规则隔离的收益','|版本/比较|配对数|准确率差|疾病簇95%区间|Holm p|','|---|---:|---:|---|---:|']
 for k,x in r['comparisons'].items():
  if k.endswith('->hidden_prune'):lines.append(f"|{k}|{x['n']}|{fmt(x['delta'])}|{x['disease_bootstrap_ci']}|{x['p_holm']:.4g}|")
 lines+=['正差值表示隐藏动作裁剪较高；区间跨0则尚不能确认收益。A不下降须检查A差值区间，不能由不显著推出。','## K组调和平均','CS仅2题且同一疾病；BCDCS主指标仅作描述，不作总体排名。','|方法/指标|共同K题数|值|疾病簇95%区间|','|---|---:|---:|---|']
 for k,x in r['harmonic'].items():lines.append(f"|{k}|{x['n_common_K']}|{fmt(x['value'])}|{x['disease_bootstrap_ci']}|")
 lines+=['完整各版本准确率、原动作/误用/诱饵率、解析失败率及配对检验见summary.json；逐题数值见scores.jsonl。注意力阶段另见attention目录，CAD原结果只读审计另见cad_finished_audit.json。']
 return '\n\n'.join(lines[:4])+'\n'+'\n'.join(lines[4:])+'\n'
