"""Audit existing P1 data and predictions without model inference or test data."""
from pathlib import Path
import collections,csv,hashlib,json,re,sys
from run_p1 import render
from prepare import split_question
D=Path(__file__).resolve().parent
OUT=D/'results/p1_source_audit';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
rows=[json.loads(s) for s in (D/'private/cases.jsonl').read_text().splitlines()]
source=[json.loads(s) for s in Path('/home/gcx/aster_dynamic_review_20261025/dev.jsonl').read_text().splitlines()]
src={(r['id'],r['variant']):r for r in source};groups=collections.defaultdict(dict)
for r in rows:groups[r['id']][r['variant']]=r
assert all(r['split']=='dev' for r in rows+source)
checks=collections.Counter();ambiguous=[];removed_facts=[];constraints=[]
for cid,vs in groups.items():
 n=vs['NC'];g=int(n['gold'])-1;other=1-g
 assert len({render(vs[v],'blind') for v in ['NC','IC','AC','SW']})==1
 checks['identical_blind_across_four_volumes']+=1
 for v in ['NC','IC','AC','SW']:
  r=vs[v];s=src[cid,v]
  assert r['rules'][g]['source_path_index']==s['gold_rule'] and r['gold']==n['gold']
  assert r['rules'][other]['source_path_index']==s['decoy_rule']
  assert all(q['condition']==s['rules'][q['source_path_index']]['condition'] and q['action']==s['rules'][q['source_path_index']]['action'] for q in r['rules'])
  assert r['patient']==n['patient'] and r['options']==n['options'] and r['source_question']==n['source_question']
  checks['source_rule_and_label_mapping']+=1
 assert vs['AC']['rules'][g]['action']!=n['rules'][g]['action'] and vs['AC']['rules'][other]['action']==n['rules'][other]['action']
 assert vs['IC']['rules'][g]['action']==n['rules'][g]['action'] and vs['IC']['rules'][other]['action']!=n['rules'][other]['action']
 assert vs['SW']['rules'][g]['action']==n['rules'][other]['action'] and vs['SW']['rules'][other]['action']==n['rules'][g]['action']
 checks['correct_AC_IC_SW_action_edits']+=1
 # Semantic ambiguity visible in the extracted conditions; no outcome-based selection.
 if {q['condition'] for q in n['rules']}=={'Negative margins','Negative margins for invasive disease'}:ambiguous.append(cid)
 sentences=re.split(r'(?<=[.!?])\s+',re.sub(r'\b(Mrs|Mr|Ms|Dr)\.',r'\1',n['source_question']))
 factual=[s for s in sentences if '?' not in s and len(s.split())>6 and re.match(r'^(?:Given|Considering|Based on)\b',s,re.I)]
 if factual:removed_facts.append(cid)
 for q in n['rules']:
  if re.search(r'\b(if|criteria|when|unless)\b',q['action'],re.I):constraints.append(cid);break
summary=list(csv.DictReader((D/'results/summary.csv').open()))
metrics=[];sensitivity=[];hashes={}
for model in ['qwen','llama']:
 ps=[json.loads(s) for s in (D/'private'/model/'predictions.jsonl').read_text().splitlines()];assert len(ps)==2*len(rows)
 by={(r['id'],r['variant'],r['condition']):r for r in ps};assert len(by)==len(ps)
 manifest=json.loads((D/'private'/model/'manifest.json').read_text())
 assert manifest['cases_sha256']==sha(D/'private/cases.jsonl') and manifest['script_sha256']==sha(D/'run_p1.py')
 prompts={r['sha256']:r['prompt'] for r in map(json.loads,(D/'private'/model/'prompts.jsonl').read_text().splitlines())}
 for r in ps:
  match=re.fullmatch(r'\s*(?:Branch\s*)?([12])[.。]?\s*',r['raw_output'],re.I);pred=match[1] if match else None
  assert pred==r['prediction'] and r['correct']==(pred==r['gold'])
  assert r['gold']==groups[r['id']][r['variant']]['gold']
  text=prompts[r['prompt_sha256']];assert hashlib.sha256(text.encode()).hexdigest()==r['prompt_sha256']
  assert render(groups[r['id']][r['variant']],r['condition']) in text
 checks['raw_outputs_labels_prompt_hashes_verified']+=len(ps)
 for v in ['NC','IC','AC','SW','CS']:
  ids=[cid for cid,vs in groups.items() if v in vs]
  for tree in ['ALL']+sorted({groups[cid][v]['tree'] for cid in ids}):
   ii=[cid for cid in ids if tree=='ALL' or groups[cid][v]['tree']==tree];pairs=[(by[cid,v,'blind'],by[cid,v,'visible']) for cid in ii]
   b=sum(x['correct'] for x,y in pairs);a=sum(y['correct'] for x,y in pairs)
   z={'model':model,'variant':v,'tree':tree,'n':len(ii),'blind_correct':b,'visible_correct':a,'wrong_to_right':sum(not x['correct'] and y['correct'] for x,y in pairs),'right_to_wrong':sum(x['correct'] and not y['correct'] for x,y in pairs),'delta_pp':100*(a-b)/len(ii),'parse_failures':sum(x['parse_failure']+y['parse_failure'] for x,y in pairs)};metrics.append(z)
   if tree=='ALL':
    old=next(s for s in summary if s['model']==model and s['volume']==v)
    assert int(old['blind_correct'])==b and int(old['visible_correct'])==a and abs(float(old['visible_minus_blind_pp'])-z['delta_pp'])<1e-9
  if v!='CS':
   for label,ii in [('ambiguous_condition_pair',[x for x in ids if x in ambiguous]),('remaining_NOT_REVALIDATED',[x for x in ids if x not in ambiguous])]:
    pairs=[(by[cid,v,'blind'],by[cid,v,'visible']) for cid in ii]
    sensitivity.append({'model':model,'variant':v,'subset':label,'n':len(ii),'blind_accuracy_pct':100*sum(b['correct'] for b,a in pairs)/len(ii),'visible_accuracy_pct':100*sum(a['correct'] for b,a in pairs)/len(ii),'delta_pp':100*sum(int(a['correct'])-int(b['correct']) for b,a in pairs)/len(ii)})
 hashes[model]={'manifest_sha256':sha(D/'private'/model/'manifest.json'),'predictions_sha256':sha(D/'private'/model/'predictions.jsonl')}
result={'status':'ARITHMETIC_AND_MAPPING_PASS_DATA_VALIDITY_REQUIRES_REPAIR','checks':dict(checks),'ambiguous_pair_n':len(ambiguous),'ambiguous_pair_ids':ambiguous,'removed_nonquestion_Given_Considering_Based_sentences_n':len(removed_facts),'removed_nonquestion_ids':removed_facts,'action_fields_contain_if_criteria_when_unless_n':len(constraints),'action_constraint_ids':constraints,'models':hashes,'test_cases_read':0,'new_model_inference':0,'sensitivity_is_posthoc_not_revalidated_dataset':True}
dump(OUT/'audit.json',result)
for name,rs in [('by_tree.csv',metrics),('sensitivity.csv',sensitivity)]:
 with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
dump(D/'results/data_validity_hold.json',{'active':True,'blocked_stages':['P5','P6'],'reason':'P1 audit found 31 extracted branch pairs with overlapping conditions; renderer also removes entire source question. Revalidate development cases and prompts before further method claims. Existing outputs remain descriptive; do not relabel or silently replace them.','audit':'p1_source_audit/audit.json'})
print(json.dumps(result,ensure_ascii=False))
print(json.dumps(sensitivity,ensure_ascii=False))
