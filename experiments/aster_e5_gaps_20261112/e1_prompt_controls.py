"""Frozen CC and strong evidence prompt; no backbone or ASTER training."""
import os,json,time,csv
from pathlib import Path
import numpy as np
os.environ.setdefault('ASTER_CR_ROOT','/home/gcx/aster_cr_repair_v2_run_20261108')
from experiments.aster_cr.engine import Engine,SYSTEM
from experiments.aster_cr.common import TASK_KINDS
from experiments.aster_closed_form.data import canonical_guide,prompt
from experiments.aster_cr.paths import execute
from experiments.aster_e5_gaps_20261112.loreft_native import bootstrap_summary
ROOT=Path(__file__).resolve().parents[2]
OLD=Path(os.environ['ASTER_CR_ROOT'])
OUT=ROOT/'reports/aster_e5_gaps_20261112/prompt_controls_e1'
INSTRUCTION='仅依据当前条件、指南中的定义和患者中与该条件有关的事实作出判断。后续处置的内容以及与当前条件无关的背景不作为当前条件成立的证据。保留并阅读完整输入，不得把后续处置反推为当前条件结论。'
REFS=['N/A','','[MASK]']
def rows(p):
 if not p.exists():return []
 with p.open() as f:return [json.loads(s) for s in f if s.strip()]
def write(p,x):
 p.parent.mkdir(exist_ok=True,parents=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def append(p,x):
 with p.open('a') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
class PromptEngine(Engine):
 strong=False
 def render(self,prompt,task='binary'):
  system=SYSTEM+('\n'+INSTRUCTION if self.strong else '')
  return self.tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def content_free(s):
 return f'完整决策树：\n{s}\n\n指南原文定义与说明：\n{s}\n\n病历：\n{s}\n\n当前节点：{s}\n当前条件：{s}\n请判断当前条件。只输出“是”或“否”。\n答案：'
def main():
 import torch
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
 if (OUT/'completion.json').exists():return
 started=time.time();eng=PromptEngine('qwen')
 torch.cuda.reset_peak_memory_stats()
 config=dict(model='Qwen3-8B',methods=['CC','strong_evidence_prompt'],CC_references=REFS,CC_rule='Mean binary answer probability on three content-free prompts; subtract prior log odds from unchanged raw margin.',strong_prompt=INSTRUCTION,training='none; fully frozen prompts/references before development or historical-test evaluation',development_selection='one prespecified configuration each; no test-based or dev-based prompt search',label_scoring='same Chinese yes/no single-token FP32 projection',no_gold_evidence=True,history='Previously used 40-source controlled evaluation; no new independent test.',NLI4CT_evaluations=0,MedPIC_evaluations=0)
 write(OUT/'configuration.json',config)
 if (OUT/'cc_parameters.json').exists():cc=json.loads((OUT/'cc_parameters.json').read_text())
 else:
  eng.strong=False;ell=[eng.score(content_free(s),capture=False)['ell'] for s in REFS]
  prob=[float(1/(1+np.exp(-s))) for s in ell];mean=float(np.mean(prob));mean=float(np.clip(mean,1e-12,1-1e-12))
  cc=dict(reference_logits=ell,reference_yes_probabilities=prob,mean_yes_probability=mean,offset=-float(np.log(mean/(1-mean))),fit_labels_used=False,reference_forwards=3)
  write(OUT/'cc_parameters.json',cc)
 allviews=rows(OLD/'data/views.jsonl');allpairs=rows(OLD/'data/pairs.jsonl');allcases=rows(OLD/'data/path_cases.jsonl')
 summary=[]
 for method in ['CC','strong_evidence_prompt']:
  eng.strong=method=='strong_evidence_prompt';target=OUT/method;target.mkdir(exist_ok=True)
  for split in ['dev','test']:
   if (target/(split+'_metrics.json')).exists():summary.append(json.loads((target/(split+'_metrics.json')).read_text()));continue
   t=time.time()
   vv=[v for v in allviews if v['split']==split and v['kind'] in TASK_KINDS];pp=[p for p in allpairs if p['split']==split];cases=[c for c in allcases if c['split']==split]
   raw={r['view_id']:r['ell'] for p in (OLD/'qwen/scores').glob(split+'_*.jsonl') for r in rows(p)}
   raw_text={v['prompt']:raw[v['view_id']] for v in vv}
   predfile=target/(split+'_input_predictions.jsonl');nodefile=target/(split+'_path_node_scores.jsonl');pathfile=target/(split+'_path_predictions.jsonl')
   saved={r['view_id']:r['score'] for r in rows(predfile)};cache={v['prompt']:saved[v['view_id']] for v in vv if v['view_id'] in saved}
   for r in rows(nodefile):cache[r['prompt']]=r['score']
   def score(text):
    if text not in cache:
     if method=='CC':
      base=raw_text.get(text)
      if base is None:base=eng.score(text,capture=False)['ell']
      cache[text]=float(base+cc['offset'])
     else:cache[text]=float(eng.score(text,capture=False)['ell'])
    return cache[text]
   for i,v in enumerate(vv,1):
    if v['view_id'] not in saved:
     s=score(v['prompt']);saved[v['view_id']]=s
     append(predfile,dict(view_id=v['view_id'],source_id=v['source_id'],label=v['label'],score=s,prediction=int(s>0),raw_score=raw[v['view_id']]))
    if i==10 and method=='strong_evidence_prompt' and split=='dev':
     early=[v['view_id'] for v in vv[:10]]
     write(target/'smoke.json',dict(examples=10,finite=all(np.isfinite(saved[k]) for k in early),changed_scores=sum(saved[k]!=raw[k] for k in early),method='fixed stronger system instruction; no parameter training'))
    if i%100==0:write(OUT/'status.json',dict(method=method,split=split,stage='inputs',done=i,total=len(vv),seconds=time.time()-t))
   done={r['case_id'] for r in rows(pathfile)}
   for i,c in enumerate(cases,1):
    if c['case_id'] in done:continue
    tree=canonical_guide(c['guideline']);nodes={n['id']:n for n in tree['nodes']};gid=tree['nodes'][0]['id'];ss={}
    while gid in nodes and gid not in ss:
     text=prompt(c['guideline'],c['definitions'],c['patient'],gid);s=score(text);ss[gid]=s
     append(nodefile,dict(case_id=c['case_id'],source_id=c['source_id'],gid=gid,prompt=text,score=s))
     gid=nodes[gid]['yes' if s>0 else 'no']
    append(pathfile,dict(case_id=c['case_id'],source_id=c['source_id'],**execute(c,ss)))
   paths=rows(pathfile);raw_paths=rows(OLD/'qwen/path_results'/('full_'+split+'.jsonl'))
   metrics,predictions=bootstrap_summary(vv,pp,saved,raw,paths,raw_paths)
   with (target/(split+'_pair_predictions.jsonl')).open('w') as f:
    for r in predictions:f.write(json.dumps(r)+'\n')
   nat=[v for v in vv if v.get('original') and v.get('variant')=='full']
   z=np.array([saved[v['view_id']] for v in nat]);y=np.array([v['label'] for v in nat]);prob=1/(1+np.exp(-np.clip(z,-700,700)))
   metrics.update(method=method,split=split,configuration='../configuration.json',training_runs=0,parameters=cc if method=='CC' else {'fixed_instruction':INSTRUCTION},probability_metrics={'brier':float(np.mean((prob-y)**2)),'log_loss':float(np.mean(np.logaddexp(0,z)-y*z))},seconds=time.time()-t,peak_gib=torch.cuda.max_memory_allocated()/2**30,history='Historical controlled evaluation, not independent confirmation')
   write(target/(split+'_metrics.json'),metrics);summary.append(metrics)
 table=[]
 for m in summary:
  if m['split']!='test':continue
  a=m['groups']['AP'];c=m['groups']['CF'];p=m['path']
  table.append(dict(method=m['method'],AP_n=a['n'],AP_response_rms=a['difference_rms'],AP_rms_ci_low=a['difference_rms_ci'][0],AP_rms_ci_high=a['difference_rms_ci'][1],CF_n=c['n'],CF_correction_rms=c['correction_rms'],CF_correction_rms_ci_low=c['correction_rms_ci'][0],CF_correction_rms_ci_high=c['correction_rms_ci'][1],CF_paired_correct=c['paired_correct'],path_n=p['n'],path_correct=p['correct'],path_accuracy=p['accuracy'],path_accuracy_ci_low=p['accuracy_ci'][0],path_accuracy_ci_high=p['accuracy_ci'][1],brier=m['probability_metrics']['brier'],log_loss=m['probability_metrics']['log_loss']))
 with (OUT/'section_4_5_data.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 write(OUT/'completion.json',dict(state='complete',methods=['CC','strong_evidence_prompt'],seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30,model_forward_counts=eng.counts,training_runs=0,history='historical controlled evaluation'))
 write(OUT/'status.json',dict(state='complete'))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback
  write(OUT/'error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
