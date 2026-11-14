"""Publish only completed LoReFT runs into the E5 working report; preserve historical reports."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/aster_e5_gaps_20261112'
RUN=OUT/'loreft_e1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 done=read(RUN/'completion.json')
 if done['state']!='complete':raise RuntimeError('LoReFT completion required')
 with (OUT/'section_4_5_data.csv').open(encoding='utf-8-sig') as f:table=list(csv.DictReader(f))
 table=[r for r in table if not r['method'].startswith('LoReFT E1 seed')]
 results=[]
 for item in done['results']:
  m=read(Path(item['metrics']));ap=m['groups']['AP'];cf=m['groups']['CF'];pa=m['path']
  r=dict(method='LoReFT E1 seed'+str(item['seed']),status='completed_new_E1_training_historical_evaluation',E1_compatibility='E1 Dataset A native supervised BCE',AP_n=ap['n'],AP_response_rms=ap['difference_rms'],AP_rms_ci_low=ap['difference_rms_ci'][0],AP_rms_ci_high=ap['difference_rms_ci'][1],CF_n=cf['n'],CF_correction_rms=cf['correction_rms'],CF_correction_rms_ci_low=cf['correction_rms_ci'][0],CF_correction_rms_ci_high=cf['correction_rms_ci'][1],CF_paired_correct=cf['paired_correct'],CF_paired_accuracy=cf['paired_accuracy'],CF_paired_accuracy_ci_low=cf['paired_accuracy_ci'][0],CF_paired_accuracy_ci_high=cf['paired_accuracy_ci'][1],path_n=pa['n'],path_correct=pa['correct'],path_accuracy=pa['accuracy'],path_accuracy_ci_low=pa['accuracy_ci'][0],path_accuracy_ci_high=pa['accuracy_ci'][1],path_difference_vs_raw=pa['difference_vs_raw'],path_difference_ci_low=pa['difference_vs_raw_ci'][0],path_difference_ci_high=pa['difference_vs_raw_ci'][1],training_protocol='E1 Dataset A 2540 pairs; 4637 distinct labeled endpoints; native supervised BCE; no ASTER pair loss',provenance_file=item['metrics'],history='Historical 40-source controlled evaluation; not new independent test',seed=item['seed'])
  table.append(r);results.append(r)
 fields=list(dict.fromkeys(k for r in table for k in r))
 with (OUT/'section_4_5_data.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
 manifest=read(OUT/'new_runs.json');manifest['loreft_e1']={'status':'complete','completion':str(RUN/'completion.json'),'training_jobs':5,'selected_test_seeds':[11,22,33],'learning_rate':done['lr'],'ASTER_retraining':0,'NLI4CT_runs':0,'MedPIC_runs':0};write(OUT/'new_runs.json',manifest)
 scope=read(OUT/'scope.json');scope.update(status='partial_loreft_completed_other_baselines_queued',new_baseline_training_runs=5);write(OUT/'scope.json',scope)
 p=OUT/'compatibility_matrix.csv'
 with p.open(encoding='utf-8') as f:matrix=list(csv.DictReader(f))
 for row in matrix:
  if row['method']=='LoReFT':
   row['training_protocol']='E1 Dataset A 2540 pairs; native supervised BCE; seeds 11/22/33'
   row['current_action']='Completed repaired adapter E1 fitting and historical controlled evaluation; see LOREFT_RESULTS.zh.md'
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
 p=OUT/'COMPATIBILITY_MATRIX.zh.md';p.write_text(p.read_text().replace('正式补跑已启动，开发集冻结配置，完成前不填数值','E1三种子补跑完成；学习率仅由开发集冻结；结果见LOREFT_RESULTS.zh.md'))

 text=['# LoReFT E1 修复与正式结果','','旧适配器因Householder旋转参数被AdamW权重衰减归零而失效。只取消旋转参数衰减，保留监督二元交叉熵，修复通过真实GPU激活、梯度、输出和保存加载测试。','','正式训练使用Dataset A全部2540对的4637个不同有标签端点，一轮；rank=1、层18。三个学习率先用种子11在20来源开发集选择，再冻结学习率训练种子22、33；无A/P或C/F差分损失，不训练ASTER。','','测试为有历史使用记录的40来源、2293 A/P、40 C/F、806自然路径。以下是逐种子结果及2000次来源簇重采样95%区间；三种子不是三个独立测试集。NLI500、MedPIC188未重评。','','|种子|A/P RMS|C/F修正RMS|C/F全对|自然路径全对|路径95%区间|','|---|---:|---:|---:|---:|---:|']
 for r in results:text.append('|{}|{:.6f}|{:.6f}|{}/{}|{}/{}|[{:.2f}%, {:.2f}%]|'.format(r['seed'],r['AP_response_rms'],r['CF_correction_rms'],r['CF_paired_correct'],r['CF_n'],r['path_correct'],r['path_n'],100*r['path_accuracy_ci_low'],100*r['path_accuracy_ci_high']))
 text+=['','逐例输入、配对、完整动态路径和指标位于loreft_e1对应种子目录。完整指标和区间已并入[第4.5节数据](section_4_5_data.csv)。','Codex01审计已到达，其他E1补跑由extension_queue接续，E5整体保持部分完成。']
 (OUT/'LOREFT_RESULTS.zh.md').write_text('\n'.join(text)+'\n')
 report=(OUT/'REPORT.zh.md').read_text()
 report+='\n\n## LoReFT正式运行完成\n\n[修复、逐种子结果与95%区间](LOREFT_RESULTS.zh.md)。5次训练和3个选定配置测试已完成；上方启动记录保留执行顺序，当前以completion.json为准。其他基线缺口由extension_queue执行，E5整体未宣称完成。\n'
 (OUT/'REPORT.zh.md').write_text(report)
 doc=ROOT/'docs/ASTER/experiments.zh.md';s=doc.read_text()
 s+='\n\n### E5 LoReFT专项结果（2026-11-12）\n\n已确认旧LoReFT适配器失效，修复后通过有效性测试，并完成E1/Dataset A 2540对原生监督训练：开发集冻结学习率，种子11/22/33分别报告。逐例预测、共同指标、95%区间与第4.5节数据见[LoReFT结果](../../reports/aster_e5_gaps_20261112/LOREFT_RESULTS.zh.md)。评价仍是历史40来源/806自然路径；没有重训ASTER-CR或重评已完成Qwen NLI500/MedPIC188。其他基线补跑依审计结论由extension_queue接续，E5整体保持部分完成。\n'
 s=s.replace('LoReFT修复通过，E1补跑运行中','LoReFT E1三种子补跑已完成；其他基线队列接续').replace('LoReFT修复通过，E1补跑准备就绪','LoReFT E1三种子补跑已完成；其他基线队列接续')
 doc.write_text(s)
 write(OUT/'delivery_status.json',dict(status='partial_loreft_completed_other_baselines_queued',table_rows=len(table),new_training_runs=5,selected_test_seeds=[11,22,33],manuscript_changed=False,historical_reports_changed=False))
 print('Completed LoReFT rows and E5 partial status published.')
if __name__=='__main__':main()
