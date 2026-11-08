"""Native external tasks; no MEGA macro-average or training records."""
import argparse,itertools,difflib
from collections import defaultdict,Counter
from .core import *
DATA=Path('/home/gcx/TRIAGE/data')
def norm(s):return re.sub(r'\s+',' ',s.strip().lower())
def nli4ct():
 d=DATA/'NLI4CT-2024';test=read(d/'test.json');gold=read(d/'gold_test.json');items=[];pairs=[]
 if set(test)!=set(gold):raise ValueError('NLI test/gold IDs differ')
 for ident,r in gold.items():
  original=test[ident]
  if any(original.get(k)!=r.get(k) for k in ('Statement','Primary_id','Secondary_id','Section_id')):raise ValueError('NLI input/gold disagreement '+ident)
  premise=[]
  for role,k in [('Primary','Primary_id'),('Secondary','Secondary_id')]:
   if k not in r:continue
   trial=read(d/'original/CT json'/(r[k]+'.json'));section=trial[r['Section_id']]
   premise.append(dict(role=role,trial_id=r[k],section=r['Section_id'],lines=section))
  text='\n\n'.join(p['role']+' trial ('+p['trial_id']+'), '+p['section']+':\n'+'\n'.join(p['lines']) for p in premise)
  prompt_text='Determine whether the statement is entailed by or contradicts the clinical trial information. Output only Entailment or Contradiction.\n\n'+text+'\n\nStatement: '+r['Statement']+'\nLabel:'
  c=r.get('Causal_type');base=c[1] if c else ident
  items.append(dict(id=ident,group_id=base,split='external_test',dataset='NLI4CT-2024',statement=r['Statement'],premise=premise,prompt=prompt_text,label=r['Label'],labels=['Entailment','Contradiction'],original=not bool(c),intervention=r.get('Intervention'),published_causal_type=c[0] if c else None))
  if c:
   if base not in gold or gold[base].get('Causal_type'):raise ValueError('NLI original relation invalid')
   same=gold[base]['Label']==r['Label']
   pairs.append(dict(pair_id=base+':'+ident,group_id=base,left=base,right=ident,left_label=gold[base]['Label'],right_label=r['Label'],relation='preserving' if same else 'changing',intervention=r.get('Intervention'),published_causal_type=c[0]))
 out=ROOT/'external/nli4ct';lines(out/'items.jsonl',items);lines(out/'pairs.jsonl',pairs)
 summary=dict(items=len(items),originals=sum(x['original'] for x in items),pairs=len(pairs),relations=dict(Counter(p['relation'] for p in pairs)),by_intervention={k:dict(Counter(p['relation'] for p in pairs if p['intervention']==k)) for k in sorted({p['intervention'] for p in pairs})},role='independent validation only',source='https://arxiv.org/html/2404.04963v1',label_policy='released labels and original IDs; never infer flip from intervention name')
 write(out/'manifest.json',summary);return summary

def candidate_group(ident):
 if ident.startswith('PG-'):
  p=ident.split('-');return '-'.join([p[0],p[1],p[-1]])
 return ident.rsplit('-',1)[0]
def patient_diff(a,b):
 return [dict(left=a[i:j],right=b[k:l],left_start=i,right_start=k) for op,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if op!='equal']
def compatible(a,b):
 opts_a={norm(v):k for k,v in a['options'].items()};opts_b={norm(v):k for k,v in b['options'].items()}
 if len(opts_a)!=len(a['options']) or set(opts_a)!=set(opts_b):return False,'option content differs',{}
 if norm(a['question'])!=norm(b['question']):return False,'question differs',{}
 # Record actual shared patient context and every changed span; IDs alone never suffice.
 sm=difflib.SequenceMatcher(None,a['patient_vignette'],b['patient_vignette'],autojunk=False)
 shared=[a['patient_vignette'][m.a:m.a+m.size] for m in sm.get_matching_blocks() if m.size>=12]
 if not shared or a['patient_vignette']==b['patient_vignette']:return False,'no shared changed patient context',{}
 return True,'question and option contents match; shared patient context recorded',dict(option_map={k:opts_b[norm(v)] for k,v in a['options'].items()},shared_context=shared,patient_changes=patient_diff(a['patient_vignette'],b['patient_vignette']))
def medpic():
 raw=read(DATA/'MedPIC-Bench/questions.json');items=[];candidates=defaultdict(list);pairs=[];groups=[];unpaired=[]
 for r in raw:
  if not r['answer'] or not set(r['answer'])<=set(r['options']):raise ValueError('invalid MedPIC answer')
  text=r['patient_vignette']+'\n\n'+r['question']+'\n'+'\n'.join(k+'. '+v for k,v in r['options'].items())+'\nSelect ALL correct options. Output a JSON array of option letters only.\nAnswer:'
  items.append(dict(r,id=r['instance_id'],dataset='MedPIC-Bench',split='external_test',prompt=text,answer_texts=sorted(norm(r['options'][x]) for x in r['answer'])))
  if r['benchmark_task_family']=='counterfactual':candidates[candidate_group(r['instance_id'])].append(r)
 for ident,rr in sorted(candidates.items()):
  valid_members=set();edges=[]
  for a,b in itertools.combinations(rr,2):
   ok,why,evidence=compatible(a,b)
   if not ok:
    unpaired.append(dict(group_id=ident,left=a['instance_id'],right=b['instance_id'],reason=why));continue
   aa=sorted(norm(a['options'][x]) for x in a['answer']);bb=sorted(norm(b['options'][x]) for x in b['answer'])
   p=dict(pair_id=a['instance_id']+':'+b['instance_id'],group_id=ident,left=a['instance_id'],right=b['instance_id'],left_answer=a['answer'],right_answer=b['answer'],left_answer_texts=aa,right_answer_texts=bb,relation='preserving' if aa==bb else 'changing',verification=why,**evidence)
   pairs.append(p);edges.append(p['pair_id']);valid_members.update([a['instance_id'],b['instance_id']])
  if valid_members:groups.append(dict(group_id=ident,members=sorted(valid_members),pair_ids=edges,member_count=len(valid_members)))
  for r in rr:
   if r['instance_id'] not in valid_members:unpaired.append(dict(id=r['instance_id'],group_id=ident,reason='no verified same-case counterpart; retain native question'))
 out=ROOT/'external/medpic';lines(out/'items.jsonl',items);lines(out/'pairs.jsonl',pairs);lines(out/'groups.jsonl',groups);lines(out/'unpaired.jsonl',unpaired)
 summary=dict(items=len(items),counterfactual_questions=sum(x['benchmark_task_family']=='counterfactual' for x in items),verified_groups=len(groups),verified_members=len({x for g in groups for x in g['members']}),pairs=len(pairs),group_sizes=dict(Counter(g['member_count'] for g in groups)),relations=dict(Counter(p['relation'] for p in pairs)),source='https://huggingface.co/datasets/TIM0927/MedPIC-Bench',role='independent validation only',answer_policy='retain published multi-select labels; compare option texts across permutations',pair_policy='ID proposes candidates; question, option content, patient context and differences verify relationship; multi-member groups retained')
 write(out/'manifest.json',summary);return summary

def evaluate_external(dataset,prediction_file):
 folder=ROOT/'external'/dataset;items=rows(folder/'items.jsonl');pairs=rows(folder/'pairs.jsonl');prs=rows(prediction_file)
 if len({p['id'] for p in prs})!=len(prs):raise ValueError('duplicate prediction IDs')
 pred={p['id']:p.get('prediction') for p in prs};correct={};canon={};invalid=[]
 for r in items:
  v=pred.get(r['id'])
  if dataset=='nli4ct':
   if v not in r['labels']:invalid.append(r['id']);canon[r['id']]=None
   else:canon[r['id']]=v
   correct[r['id']]=v==r['label']
  else:
   good=isinstance(v,list) and bool(v) and all(isinstance(k,str) for k in v) and len(set(v))==len(v) and set(v)<=set(r['options'])
   if not good:invalid.append(r['id']);canon[r['id']]=None
   else:canon[r['id']]=sorted(norm(r['options'][k]) for k in v)
   correct[r['id']]=good and set(v)==set(r['answer'])
 def metrics(pp):
  both=sum(correct[p['left']] and correct[p['right']] for p in pp)
  relation=sum(canon[p['left']] is not None and canon[p['right']] is not None and ((canon[p['left']]==canon[p['right']])==(p['relation']=='preserving')) for p in pp)
  return dict(pairs=len(pp),both_correct=both/len(pp) if pp else None,relation_accuracy=relation/len(pp) if pp else None)
 result=dict(dataset=dataset,items=len(items),predictions_present=sum(r['id'] in pred for r in items),missing_or_invalid=len(invalid),exact_accuracy=sum(correct.values())/len(items),pair_metrics={rel:metrics([p for p in pairs if p['relation']==rel]) for rel in ('preserving','changing')},denominator='all released items; missing/invalid count as wrong; pair results separate from MEGA')
 group_members=defaultdict(set)
 for p in pairs:group_members[p['group_id']].update([p['left'],p['right']])
 result['groups_all_correct']=sum(all(correct[i] for i in ids) for ids in group_members.values())/len(group_members) if group_members else None
 if dataset=='nli4ct':result['original_accuracy']=sum(correct[r['id']] for r in items if r['original'])/sum(r['original'] for r in items)
 else:result['by_family']={f:sum(correct[r['id']] for r in items if r['benchmark_task_family']==f)/sum(r['benchmark_task_family']==f for r in items) for f in {r['benchmark_task_family'] for r in items}}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evaluate',choices=['nli4ct','medpic']);p.add_argument('--predictions');p.add_argument('--output');a=p.parse_args()
 if a.evaluate:
  result=evaluate_external(a.evaluate,a.predictions)
  if a.output:write(a.output,result)
 else:result=dict(nli4ct=nli4ct(),medpic=medpic());write(ROOT/'external/summary.json',result)
 print(json.dumps(result,ensure_ascii=False))
