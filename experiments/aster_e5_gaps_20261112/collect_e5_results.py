"""Collect completed E5 cells only; no training, model calls, or history overwrite."""
from pathlib import Path
import csv,json,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/aster_e5_gaps_20261112'
def load(p):return json.loads(p.read_text())
def main():
 completed=[];indices=[]
 # Source CSV retains all common metrics, extra path scopes and its own CI index.
 p=OUT/'output_controls_e1/main_table.csv'
 if (p.parent/'completion.json').exists():
  with p.open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
  for r in rr:
   if r['scope']=='test' and r['method']!='raw':
    completed.append(dict(method=r['method'],training_protocol='E1 endpoint weighted native NLL',AP_n=r['ap_pairs'],AP_response_rms=r['ap_score_change_rms'],CF_n=r['cf_pairs'],CF_correction_rms=r['cf_correction_rms'],CF_paired_accuracy=r['cf_paired_accuracy'],path_n=r['natural_path_count'],path_correct=r['natural_path_correct'],path_accuracy=r['natural_path_accuracy'],interference_path_n=r['interference_path_count'],interference_path_correct=r['interference_path_correct'],brier=r['natural_brier'],log_loss=r['natural_log_loss'],source=str(p),ci_source=str(p.parent/'all_source_bootstrap.csv')))
  indices.append(str(p.parent/'completion.json'))
 candidates=[]
 for family in ['lora','lofit']:
  d=OUT/'native_supervised_e1'/family
  if (d/'completion.json').exists():
   done=load(d/'completion.json');indices.append(str(d/'completion.json'))
   candidates.extend((Path(x['metrics']),family+'_SFT_E1_seed'+str(x['seed'])) for x in done['results'])
 d=OUT/'loreft_e1'
 if (d/'completion.json').exists():
  done=load(d/'completion.json');indices.append(str(d/'completion.json'))
  candidates.extend((Path(x['metrics']),'LoReFT_binary_task_adaptation_E1_seed'+str(x['seed'])) for x in done['results'])
 for family in ['leace_pair','iti_style','sea_adaptation']:
  p=OUT/'representation_e1_plan'/family/'test_metrics.json'
  if p.exists():candidates.append((p,family+'_E1'))
 for family in ['CC','strong_evidence_prompt']:
  p=OUT/'prompt_controls_e1'/family/'test_metrics.json'
  if p.exists():candidates.append((p,family))
 for p,name in candidates:
  m=load(p);a=m['groups']['AP'];c=m['groups']['CF'];z=m['path']
  r=dict(method=name,training_protocol=m.get('training',{}).get('objective','See linked method configuration; no implicit claim of unchanged official setup'),AP_n=a['n'],AP_response_rms=a['difference_rms'],AP_rms_ci_low=a['difference_rms_ci'][0],AP_rms_ci_high=a['difference_rms_ci'][1],CF_n=c['n'],CF_correction_rms=c['correction_rms'],CF_correction_rms_ci_low=c['correction_rms_ci'][0],CF_correction_rms_ci_high=c['correction_rms_ci'][1],CF_paired_correct=c['paired_correct'],CF_paired_accuracy=c['paired_accuracy'],path_n=z['n'],path_correct=z['correct'],path_accuracy=z['accuracy'],path_accuracy_ci_low=z['accuracy_ci'][0],path_accuracy_ci_high=z['accuracy_ci'][1],path_difference_vs_raw=z['difference_vs_raw'],path_difference_ci_low=z['difference_vs_raw_ci'][0],path_difference_ci_high=z['difference_vs_raw_ci'][1],source=str(p),ci_source=str(p))
  completed.append(r)
 if completed:
  keys=list(dict.fromkeys(k for r in completed for k in r))
  with (OUT/'section_4_5_new_e1.csv').open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(completed)
 status=load(OUT/'extension_queue/status.json') if (OUT/'extension_queue/status.json').exists() else {}
 (OUT/'new_results_index.json').write_text(json.dumps(dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_numeric_rows=len(completed),completed_markers=indices,queue=status,history='40 sources / 806 natural paths reused historically; not independent testing. Official 7458-path results are not inferred.',target_task_reruns=0),ensure_ascii=False,indent=2)+'\n')
 text=['# Codex02 E5 实际完成结果','','本报告只收录已有完成记录与真实预测支持的单元。正式训练固定E1/Dataset A 2540对，Qwen3-8B；源训练端点4637。未重训ASTER，未重评NLI500或MedPIC188。','','[新增E1结果统一表](section_4_5_new_e1.csv)保存逐方法结果和95%区间来源；[运行状态与完成文件](new_results_index.json)单列。原有历史13行结果继续保存在section_4_5_data.csv及对应原始目录；不把历史旧训练行冒充E1。','','LoReFT沿用二元监督交叉熵的单层任务适配；LoRA采用单层rank1完整词表SFT；LoFiT采用Qwen架构适配的两阶段SFT，部署和选择阶段参数分开报告。LEACE-pair共享E1选头监督、使用配对概念编码，明确作为任务适配控制；ITI和SEA也明确沿用各自任务适配。它们不冒称未改动的官方原配置。','','所有40来源、806自然路径结果均有历史使用记录；部分CPU控制另覆盖2418干扰路径，分母分别保存。没有将806替代官方7458。外部新候选由Codex03单列，此处未运行方法没有数字。','','|方法|A/P RMS|C/F修正RMS|自然路径全对|','|---|---:|---:|---:|']
 for r in completed:text.append('|{}|{:.6f}|{:.6f}|{}/{}|'.format(r['method'],float(r['AP_response_rms']),float(r['CF_correction_rms']),r['path_correct'],r['path_n']))
 (OUT/'FINAL_E5_CODEX02.zh.md').write_text('\n'.join(text)+'\n')
 doc=ROOT/'docs/ASTER/experiments.zh.md';s=doc.read_text()
 s+='\n\n### Codex02 E5新增结果汇总（2026-11-12）\n\n本任务实际完成的E1输出控制、表示适配及监督微调结果，以[结果清单](../../reports/aster_e5_gaps_20261112/FINAL_E5_CODEX02.zh.md)和[完成状态](../../reports/aster_e5_gaps_20261112/new_results_index.json)为准。未完成或失败单元不填数值，历史40来源806路径不冒充全新独立测试或官方7458路径。E5整体仍需结合Codex03新增候选状态，未修改论文正文。\n'
 doc.write_text(s)
 print(json.dumps({'completed_numeric_rows':len(completed),'output':str(OUT/'FINAL_E5_CODEX02.zh.md')}))
if __name__=='__main__':main()
