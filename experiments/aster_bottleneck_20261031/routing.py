"""Direct action-blind LLM applicability judgments; no fact/atom extraction."""
import argparse,json,math,re,time,traceback
import numpy as np
from .prepare import OUT
from .run import BASE,read,write

def softmax(x):
 a=np.asarray(x,dtype=float);a=np.exp(a-a.max());return (a/a.sum()).tolist()

def route(case,decisions,threshold=0.):
 groups={g['group_id']:g for g in case['guideline']['groups']};gid='G0';chosen=[];steps=[]
 while gid in groups:
  d=decisions.get(gid)
  if not d:break
  prediction=d['prediction'];p=d.get('probability');margin=d.get('margin',0)
  if prediction in ['NONE','U'] or margin<threshold:
   steps.append({'group':gid,'reason':prediction if prediction in ['NONE','U'] else 'low_margin'});break
  chosen.append(prediction);steps.append({'group':gid,'prediction':prediction,'probability':p,'margin':margin})
  child=next((g['group_id'] for g in groups.values() if g.get('parent_rule_id')==prediction),None)
  if child is None:break
  gid=child
 return chosen,steps

def decision(logits,temperature=1.,none_bias=0.):
 keys=list(logits);z=[(logits[k]-(none_bias if k=='NONE' else 0))/temperature for k in keys]
 p=softmax(z);order=sorted(range(len(p)),key=lambda i:p[i],reverse=True)
 return {'prediction':keys[order[0]],'probability':p[order[0]],'margin':p[order[0]]-p[order[1]],'probs':dict(zip(keys,p))}

def calibrate():
 records=[]
 for shard in [0,1]:
  for r in read(BASE/f'classification{shard}/records.jsonl'):
   case=r['case']
   if case.get('phase')!='calibration' or r.get('status')!='complete':continue
   for rid in case['gold_rules']:
    raw=r['raw_decisions'].get('G'+rid[1:-1])
    if raw:records.append((raw['logits'],rid))
 best=None
 for bias in [0,1,2,4,8]:
  for temp in [1,2,4,5.74809379053039,8]:
   loss=-sum(math.log(max(decision(z,temp,bias)['probs'][gold],1e-30)) for z,gold in records)/len(records)
   item=(loss,bias,temp)
   if best is None or item<best:best=item
 obj={'none_bias':best[1],'temperature':best[2],'nll':best[0],'groups':len(records),
  'scope':'old calibration18 visited reference groups; no unknown calibration guarantee'}
 write(OUT/'routing_calibration.json',obj);return obj

def score_route(case,path,steps):
 gold=case['gold_rules'];accepted=len(path);wrong=sum(i>=len(gold) or x!=gold[i] for i,x in enumerate(path))
 return {'exact_path':path==gold,'accepted':accepted,'wrong_accepted':wrong,
  'prefix_length':next((i for i,(x,y) in enumerate(zip(path,gold)) if x!=y),min(len(path),len(gold))),
  'first_stop':steps[-1].get('reason') if steps else 'no_decision'}

def direct_prompt(case,group):
 yes=next(r for r in group['rules'] if r['rule_id'].endswith('Y'))
 condition=re.sub(r'（判断为是）$','',yes['condition'])
 ancestors='\n'.join(group.get('ancestor_conditions',[])) or '(root node)'
 # Intentionally access no actions, original question, options, or gold path.
 return (
 'Judge the CURRENT CONDITION directly from the patient record. Do not infer missing normal results, '
 'negative findings, or treatment choices. The ancestor conditions describe how this node is reached; '
 'use them only to resolve references, not as new patient evidence. Do not repeat their applicability checks. '
 'Output exactly one letter.\nPATIENT RECORD:\n'+case['patient']+
 '\nANCESTOR CONDITIONS:\n'+ancestors+'\nCURRENT CONDITION:\n'+condition+
 '\nA. True: the record supports the condition.'
 '\nB. False: the record contradicts the condition.'
 '\nC. Unknown: the record does not provide enough information, or is uncertain or conflicting.'
 )

class DirectClassifier:
 def __init__(self,model,tok,backend):
  self.backend=backend
  from src.aster.action_blind import HFLabelScorer
  self.scorer=HFLabelScorer(model,tok,max_input_tokens=40960)
 def classify(self,case,group):
  import torch
  prompt=direct_prompt(case,group);ids,labels=self.scorer.prepare(prompt,['A','B','C'])
  with self.backend.session():
   logits=self.backend._forward(torch.tensor([ids])).logits[0,-1,labels].float().cpu().tolist()
  p=softmax(logits);states=['T','F','U'];i=int(np.argmax(p));order=sorted(p,reverse=True)
  yes=next(r['rule_id'] for r in group['rules'] if r['rule_id'].endswith('Y'))
  no=next(r['rule_id'] for r in group['rules'] if r['rule_id'].endswith('N'))
  return {'prediction':[yes,no,'U'][i],'state':states[i],'probability':p[i],'margin':order[0]-order[1],
   'probs':dict(zip([yes,no,'U'],p)),'logits':dict(zip(states,logits)),
   'confidence_kind':'restricted_label_probability_not_calibrated_risk'}

def evaluate(case,legacy,direct,cal):
 variants={
  'legacy':{gid:decision(v['logits'],5.74809379053039) for gid,v in legacy.items() if v},
  'calibrated':{gid:decision(v['logits'],cal['temperature'],cal['none_bias']) for gid,v in legacy.items() if v},
  'direct':{g['group_id']:direct.classify(case,g) for g in case['guideline']['groups']}}
 outcomes={}
 for name,ds in variants.items():
  for threshold in ([0.,.1,.2] if name=='direct' else [0.]):
   path,steps=route(case,ds,threshold)
   outcomes[name+':'+str(threshold)]={'path':path,'steps':steps,'score':score_route(case,path,steps)}
 first_errors={}
 for name,outcome in outcomes.items():
  path=outcome['path'];first=next((i for i,r in enumerate(case['gold_rules']) if i>=len(path) or path[i]!=r),None)
  if first is not None:
   rid=case['gold_rules'][first];gid='G'+rid[1:-1]
   first_errors[name]={'depth':first,'group':gid,'gold_rule':rid,
    'prediction':variants[name.split(':')[0]].get(gid,{}).get('prediction'),
    'kind':'stopped' if first>=len(path) else 'wrong_branch'}
 return {'decisions':variants,'outcomes':outcomes,'first_errors':first_errors,
  'method':'direct whole-condition LLM judgment; no structured fact extraction or atomic decomposition'}

def worker(shard):
 import torch
 from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier,classify_case
 from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
 from experiments.aster_manuscript_clinical_20261030.engine import ClinicalExecutor
 from src.aster.manuscript_core import load_expert
 torch.set_num_threads(4);torch.manual_seed(20261031)
 tok,model,classifier=model_and_classifier();freeze=json.loads((BASE/'frozen.json').read_text())
 engine=ClinicalExecutor(model,tok,load_expert(BASE/'analysis/expert.json'),freeze);direct=DirectClassifier(model,tok,engine.backend)
 cal=json.loads((OUT/'routing_calibration.json').read_text())
 from .module_one_data import M1
 development100=read(M1/'development100.jsonl')
 cases=read(OUT/'development43.jsonl')+development100+read(OUT/'validation120.jsonl')
 stages={r['case_id']:label for label,rs in [('development',read(OUT/'development43.jsonl')),('development100',development100),('validation',read(OUT/'validation120.jsonl'))] for r in rs}
 output=OUT/f'routing_{shard}.jsonl'
 done={r['case_id'] for r in read(output) if r['status']=='complete'} if output.exists() else set()
 start=time.time();count=0
 with query_chunked_attention(model,256),output.open('a') as f:
  for i,case in enumerate(cases):
   if i%4!=shard or case['case_id'] in done:continue
   try:
    legacy=classify_case(classifier,case)
    result=evaluate(case,legacy['raw_decisions'],direct,cal)
    row={'case_id':case['case_id'],'source_id':case['source_id'],'stage':stages[case['case_id']],
     'status':'complete','result':result,'legacy':legacy}
   except Exception as e:
    traceback.print_exc();row={'case_id':case['case_id'],'source_id':case['source_id'],'status':'failed','error':str(e)}
   f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();count+=1
   write(OUT/f'routing_{shard}_status.json',{'status':'running','cases':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
   if row['status']=='failed':raise RuntimeError(row['error'])
 write(OUT/f'routing_{shard}_status.json',{'status':'complete','cases':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})

def summarize():
 rows=[]
 for i in range(4):rows+=read(OUT/f'routing_{i}.jsonl')
 rows=list({r['case_id']:r for r in rows}.values());out={}
 for stage in ['development','development100','validation']:
  subset=[r for r in rows if r.get('stage')==stage and r['status']=='complete'];arms={}
  for name in subset[0]['result']['outcomes']:
   scores=[r['result']['outcomes'][name]['score'] for r in subset];accepted=sum(s['accepted'] for s in scores)
   arms[name]={'cases':len(scores),'path':sum(s['exact_path'] for s in scores),'accepted':accepted,
    'wrong_accepted':sum(s['wrong_accepted'] for s in scores),
    'risk':sum(s['wrong_accepted'] for s in scores)/accepted if accepted else None,
    'mean_prefix':sum(s['prefix_length'] for s in scores)/len(scores)}
  out[stage]=arms
 baseline=out['development100']['legacy:0.0']
 candidates={k:v for k,v in out['development100'].items() if v['risk'] is not None and v['risk']<=baseline['risk']}
 selected=max(candidates,key=lambda k:(candidates[k]['path'],-candidates[k]['risk'],candidates[k]['mean_prefix'],-float(k.split(':')[1])))
 direct_candidates={k:v for k,v in candidates.items() if k.startswith('direct:')}
 if not direct_candidates:direct_candidates={k:v for k,v in out['development100'].items() if k.startswith('direct:')}
 out['selected_direct']=max(direct_candidates,key=lambda k:(direct_candidates[k]['path'],-(direct_candidates[k]['risk'] if direct_candidates[k]['risk'] is not None else 1.),direct_candidates[k]['mean_prefix'],-float(k.split(':')[1])))
 out['selected']=selected;out['selection_scope']='development100 only; old43 historical; validation untouched'
 write(OUT/'routing_summary.json',out);return out

def beam_routes(case,decisions,k=3):
 groups={g['group_id']:g for g in case['guideline']['groups']};beam=[([],0.,'G0')]
 for _ in range(len(groups)+1):
  expanded=[];active=False
  for path,score,gid in beam:
   if gid is None:expanded.append((path,score,None));continue
   active=True;d=decisions.get(gid)
   if not d:expanded.append((path,score,None));continue
   for rid,p in d['probs'].items():
    if p<=0:continue
    if rid in ['NONE','U']:expanded.append((path,score+math.log(p),None));continue
    child=next((g['group_id'] for g in groups.values() if g.get('parent_rule_id')==rid),None)
    expanded.append((path+[rid],score+math.log(p),child))
  beam=sorted(expanded,key=lambda x:x[1],reverse=True)[:k]
  if not active or all(x[2] is None for x in beam):break
 weights=softmax([s for _,s,_ in beam])
 return [{'path':p,'log_score':s,'weight':w} for (p,s,_),w in zip(beam,weights)]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--calibrate',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
 if a.calibrate:calibrate()
 elif a.summarize:summarize()
 else:worker(a.shard)
