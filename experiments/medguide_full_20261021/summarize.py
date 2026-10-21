"""Join predictions to gold only after inference; preserve the full denominator."""
import json,collections,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
ARMS=['baseline','a1_attention']
def load(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def main():
 gold=load(P/'private/gold.jsonl');assert len(gold)==7747
 a1={};qwen={};manifests=[]
 for k in range(4):
  a1.update({x['case_id']:x for x in load(P/f'private/a1_{k}/predictions.jsonl')})
  qwen.update({x['case_id']:x for x in load(P/f'private/qwen_{k}.jsonl')})
  for name in [f'a1_{k}/manifest.json',f'qwen_{k}_manifest.json']:
   f=P/'private'/name
   if f.exists():manifests.append(json.loads(f.read_text()))
 rows=[]
 for g in gold:
  rid=g['case_id'];a=a1.get(rid,{});q=qwen.get(rid,{})
  counts=dict(collections.Counter(str(x['state']) for x in a.get('rules',[])))
  row={**g,'a1_status':a.get('status','not_run'),'rule_state_counts':counts,'signal_available':any(x.get('state') in ['T','F'] for x in a.get('rules',[])),'prompt_sha256':q.get('prompt_sha256'),'target_token_count':q.get('target_token_count',0)}
  for arm in ARMS:
   item=q.get('arms',{}).get(arm,{})
   row[arm]={'status':item.get('status','not_run'),'prediction':item.get('prediction'),'correct':item.get('prediction')==g['answer'],'candidate_logits':item.get('candidate_logits')}
  rows.append(row)
 def metrics(items):
  n=len(items);out={'n':n,'a1_completed':sum(r['a1_status'] in ['ok','parse_failure'] for r in items),'a1_valid':sum(r['a1_status']=='ok' for r in items),'signal_cases':sum(r['signal_available'] for r in items)}
  for arm in ARMS:
   correct=sum(r[arm]['correct'] for r in items);completed=sum(r[arm]['status']=='completed' for r in items)
   out[arm]={'correct':correct,'completed':completed,'accuracy_full_denominator':correct/n,'failures_or_pending':n-completed}
  out['corrected']=sum(not r['baseline']['correct'] and r['a1_attention']['correct'] for r in items)
  out['damaged']=sum(r['baseline']['correct'] and not r['a1_attention']['correct'] for r in items)
  out['prediction_changes']=sum(r['baseline']['prediction']!=r['a1_attention']['prediction'] for r in items)
  return out
 complete=len(a1)==7747 and len(qwen)==7747 and all(m.get('status')=='completed' for m in manifests) and len(manifests)==8
 summary={'status':'completed' if complete else 'incomplete','overall':metrics(rows),'per_rule_accuracy':None,'per_rule_accuracy_reason':'No independent per-rule T/F/U labels; MCQA gold paths are never treated as T/F/U ground truth.','by_cancer':{},'by_disease':{},'by_path_hash':{},'gpu_manifests':manifests,'gpu_hours_sum_process_walltime':sum(m.get('elapsed_seconds',0) for m in manifests)/3600}
 for key in ['cancer','disease','path_hash']:
  groups=collections.defaultdict(list)
  for r in rows:groups[r[key]].append(r)
  summary['by_'+key]={k:metrics(v) for k,v in groups.items()}
 (P/'results').mkdir(exist_ok=True)
 (P/'results/summary.json').write_text(json.dumps(summary,indent=2))
 with (P/'results/predictions.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps({k:v for k,v in r.items() if k!='answer'})+'\n')
 failures=[r for r in rows if not r['baseline']['correct'] or not r['a1_attention']['correct'] or r['a1_status']!='ok']
 with (P/'results/failures.jsonl').open('w') as f:
  for r in failures:f.write(json.dumps({k:v for k,v in r.items() if k!='answer'})+'\n')
 (P/'results/REPORT.zh.txt').write_text('运行状态：'+summary['status']+'\n完整分母：7747\n'+json.dumps(summary['overall'],ensure_ascii=False,indent=2)+'\n逐规则准确率不可计算：没有独立逐规则标签。\nGPU小时（各进程实际墙钟时间之和）：'+str(summary['gpu_hours_sum_process_walltime'])+'\n原始文本和推理输出仅保存在服务器private目录，未重新发布。\n')
 print(json.dumps({'status':summary['status'],**summary['overall']},ensure_ascii=False))
if __name__=='__main__':main()
