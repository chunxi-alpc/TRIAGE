"""Source-cluster audit and explicit-logic minimal pairs, never clinical gold."""
import argparse, copy, json, random, re, unicodedata
from pathlib import Path
from collections import Counter

def norm(x): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(x))).strip().casefold()
def read(p): return [json.loads(s) for s in Path(p).open() if s.strip()]
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def signature(n):
 return (norm(n.get('description','')),tuple((k,signature(n[k])) for k in ('yes','no') if isinstance(n.get(k),dict)))
def verify(row):
 # Parse numeric factual assignments independently of classifier and generated text.
 assert row['kind']=='synthetic_logical_patient'
 vals={s:int(re.search(r'\nC=([012])\n',p).group(1)) for s,p in row['states'].items()}
 truth=lambda v: None if v==2 else ('R1' if v==1 else 'R2')
 assert truth(vals['positive'])==row['target_rule_id']
 assert truth(vals['negative'])==row['verification']['negative_rule_id']
 assert truth(vals['unknown']) is None
 for field in ['guideline','question']: assert row[field]
 return vals

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--original-actions',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
 root=Path('/home/gcx/TRIAGE/data/MEGA-CDP'); data={s:read(root/s/'decision_trees.jsonl') for s in ['train_val','test']}
 parents={}; parsed=[]; errors=[]
 def find(x):
  parents.setdefault(x,x)
  while parents[x]!=x: x=parents[x]
  return x
 def union(a,b): parents[find(b)]=find(a)
 seen={}
 for split,rows in data.items():
  for r in rows:
   key=split+':'+str(r['guideline_id']);find(key)
   try: tree=json.loads(r['text'])['decision_tree']; sig=signature(tree)
   except Exception as e: errors.append({'key':key,'reason':str(e)});continue
   for identity in [('filename',norm(r['filename'])),('tree',sig)]:
    if identity in seen: union(key,seen[identity])
    else:seen[identity]=key
   parsed.append((key,split,r,tree))
 locked={find(key) for key,split,r,t in parsed if split=='test'}
 eligible=[x for x in parsed if x[1]=='train_val' and find(x[0]) not in locked]
 rng=random.Random(20261030);rng.shuffle(eligible)
 selected=[];used=set();excluded=Counter()
 for key,split,r,t in eligible:
  if find(key) in used: continue
  if not all(isinstance(t.get(b),dict) and str(t[b].get('description','')).strip() for b in ['yes','no']):excluded['nonbinary_or_missing_action']+=1;continue
  if not str(t.get('description','')).strip():excluded['empty_condition']+=1;continue
  used.add(find(key));selected.append((key,r,t))
  if len(selected)==30:break
 assert len(selected)==30
 splits=['fit']*12+['calibration']*4+['selection']*6+['development']*4+['heldout_exploratory']*4
 rows=[]
 for (key,r,t),split in zip(selected,splits):
  source='mega-source:'+find(key);condition=t['description'].strip()
  guide={'groups':[{'group_id':'g','mutually_exclusive':True,'ancestor_conditions':[], 'rules':[
   {'rule_id':'R1','condition':'条件C成立：'+condition,'action':'Alpha'},
   {'rule_id':'R2','condition':'条件C不成立：'+condition,'action':'Beta'}]}]}
  for target in ['R1','R2']:
   positive=1 if target=='R1' else 0; negative=1-positive
   def patient(v,phrase='以下记录条件真值',fact=0):
    return f'合成逻辑患者。{phrase}。编码：0=不成立，1=成立，2=未记录（未知）。\n条件C：{condition}\nC={v}\n非决定性事实：记录序号={fact}。'
   row={'kind':'synthetic_logical_patient','pair_id':key+':'+target,'source_id':source,'split':split,
     'guideline':guide,'target_rule_id':target,'question':'请只回答适用的规则编号；信息不足时回答NONE。',
     'states':dict(zip(['positive','negative','unknown'],[patient(positive),patient(negative),patient(2)])),
     'verification':{'verifier':'explicit-condition-value-parser-v1','reference':f'MEGA-CDP/{r["guideline_id"]}/root; logical assertions only',
      'positive_rule_id':target,'negative_rule_id':'R2' if target=='R1' else 'R1','unknown_label':'NONE','ancestors_reachable':True,'independent_of_classifier':True},
     'source_actions':{'R1':t['yes']['description'],'R2':t['no']['description']},'action_adaptation':'fixed synthetic Alpha/Beta labels; original source actions are provenance only','source_snapshot':str(root.resolve()),'source_guideline_id':r['guideline_id']}
   verify(row);rows.append(row)
   if split=='heldout_exploratory' and target=='R1':
    for control in ['nondetermining_fact','paraphrase','action_mapping']:
     controlrow=copy.deepcopy(row);controlrow['pair_id']+=':'+control;controlrow['control']=control
     if control=='nondetermining_fact':controlrow['states']={s:patient(v,fact=1) for s,v in zip(['positive','negative','unknown'],[positive,negative,2])}
     elif control=='paraphrase':controlrow['states']={s:patient(v,phrase='以下说明条件状态') for s,v in zip(['positive','negative','unknown'],[positive,negative,2])}
     else:
      rr=controlrow['guideline']['groups'][0]['rules'];rr[0]['action'],rr[1]['action']=rr[1]['action'],rr[0]['action']
     verify(controlrow);rows.append(controlrow)
 if a.original_actions:
  for row in rows:
   actions=row['source_actions']
   if row.get('control')=='action_mapping':actions={'R1':actions['R2'],'R2':actions['R1']}
   for rule in row['guideline']['groups'][0]['rules']:rule['action']=actions[rule['rule_id']].strip()
   row['action_adaptation']='original source child descriptions; synthetic logical patient, no natural clinical gold'
 with (out/'pairs.jsonl').open('x') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 import pyarrow.parquet as pq
 med=pq.read_table('/home/gcx/TRIAGE/data/MedGUIDE/data/train-00000-of-00001.parquet',columns=['disease'])
 audit={'raw_counts':{s:{n:sum(1 for _ in (root/s/(n+'.jsonl')).open()) for n in ['decision_trees','vignette_cases','case_paths','gt_paths']} for s in data},
  'medguide_rows':med.num_rows,'medguide_diseases':len(set(med['disease'].to_pylist())),
  'source_cluster_method':'normalized filename OR complete recursive description/branch identity; no hashes',
  'train_rows_overlapping_official_test':sum(find(x[0]) in locked for x in parsed if x[1]=='train_val'),
  'selected_sources':dict(Counter(splits)),'submitted_pairs':dict(Counter(r['split'] for r in rows)),
  'source_map':[{'source_id':'mega-source:'+find(k),'guideline_id':r['guideline_id'],'split':s} for (k,r,t),s in zip(selected,splits)],
  'excluded':dict(excluded),'parse_errors':errors,'clinical_gold_minimal_pairs':0,'clinical_semantic_verification':'not implemented; no natural-patient minimal pairs claimed',
  'cross_corpus_identity':'unresolved; no independent external validation claim','prior_PR40_exposure':'not resolved; heldout is within-run exploratory only',
  'official_test_answers_loaded':False,'official_test_run':False}
 write(out/'data_audit.json',audit)
 protocol={'seed':20261030,'kind':'synthetic logic mechanism; no clinical task efficacy claim','rank_candidates':[1,2,4,8],
  'sample_size_candidates':[4,8,16,24],'repeats':3,'top_k_candidates':[1,4],
  'rank_selection':'maximize mean heldout all-difference energy retention on selection set minus rank/128; tie smaller rank',
  'head_selection':'manuscript positive separation, read+ and read-, score descending; allow empty',
  'sample_size_selection':'use all 24 fit pairs; sizes/repeats only sensitivity, not outcome selection',
  'gain_candidates':[0.25,1.0,4.0],'gain_selection':'development gold-route first-token accuracy; tie smaller gain',
  'threshold_grid':{'probability':[0.0,0.7,0.9],'margin':[0.0,0.1]},'threshold_selection':'development accepted accuracy>=0.9 then maximum coverage; otherwise max accepted accuracy then coverage',
  'random_control':'one fixed unit direction per selected KV head; seed=20261030; not a distributional null',
  'bias_calibration':'RMS of direct fixed-query score shift per KV head on development, per unit gain',
  'inference_max_new_tokens':12,'bootstrap_repeats':1000,'statistics_unit':'source guideline cluster',
  'code_specification':'60027c426a3155bf55c26273f4c3f49fd563ce1f','model_revision':'local-modelscope-snapshot-20261014-upstream-revision-unavailable'}
 write(out/'protocol_predevelopment.json',protocol)
 print(json.dumps({'pairs':len(rows),'sources':len(selected),'clinical_gold_pairs':0}))
if __name__=='__main__':main()
