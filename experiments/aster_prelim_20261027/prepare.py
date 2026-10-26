"""Prepare P1 from the frozen, dev-only seed-0 cohort. Never load test cases."""
from pathlib import Path
import argparse,collections,copy,hashlib,json,random,re
BASE=Path(__file__).resolve().parent
SOURCE=Path('/home/gcx/aster_dynamic_review_20261025/dev.jsonl')
SPLIT=Path('/home/gcx/TRIAGE-worktrees/aster-e0-e8-20261024/experiments/aster_e0_e8_20261024/results/split.json')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def split_question(s):
 # The source patient column sometimes embeds the question. Preserve all factual sentences.
 parts=re.split(r'(?<=[.!?])\s+',s.strip())
 q=[]
 while parts and ('?' in parts[-1] or re.match(r'^(?:Given|Considering|Based on|What|Which|How)\b',parts[-1],re.I)):
  q.insert(0,parts.pop())
 return ' '.join(parts),' '.join(q)
def explicit_months(s):
 patterns=[r'(\d+(?:\.\d+)?)\s*months?\s+(?:after|following)\s+(?:completing|completion of|finishing|the completion of)\b',r'(?:completed|completing|finished|finishing)\s+(?:[^.!?;]{0,100}?)\b(\d+(?:\.\d+)?)\s*months?\s+(?:ago|prior|earlier)',r'(\d+(?:\.\d+)?)\s*months?\s+(?:ago|earlier)[^.!?;]{0,70}\b(?:completed|finished)\b']
 hits=[]
 for pat in patterns:
  for m in re.finditer(pat,s,re.I):
   if re.search(r'(?:more than|less than|at least|at most|over|under)\s*$',s[max(0,m.start(1)-20):m.start(1)],re.I):continue
   hits.append({'value':float(m[1]),'span':list(m.span()),'text':m[0]})
 return hits if len({h['value'] for h in hits})==1 else []
def main():
 out=BASE/'private';out.mkdir(exist_ok=True);res=BASE/'results';res.mkdir(exist_ok=True)
 split=json.loads(SPLIT.read_text());dev=set(split['dev_trees']);test=set(split['test_trees']);assert not dev&test
 raw=[json.loads(s) for s in SOURCE.read_text().splitlines()];assert all(r['split']=='dev' and r['tree'] in dev and r['tree'] not in test for r in raw)
 assert len({(r['id'],r['variant']) for r in raw})==len(raw)
 selected=[];exclusions=[];base_by_id={r['id']:r for r in raw if r['variant']=='NC'}
 for r in raw:
  n=base_by_id[r['id']];a=n['gold_rule'];b=n['decoy_rule'];assert n['rules'][a]['condition']!=n['rules'][b]['condition']
  assert n['rules'][a]['condition'].split(' > ')[:-1]==n['rules'][b]['condition'].split(' > ')[:-1]
  order=[a,b];random.Random('p1-order-20261027:'+r['id']).shuffle(order)
  patient,question=split_question(r['patient'])
  selected.append({'id':r['id'],'tree':r['tree'],'split':'dev','variant':r['variant'],'patient':patient,'source_question':question,'options':r['options'],'rules':[dict(r['rules'][j],bid=str(k+1)) for k,j in enumerate(order)],'gold':str(order.index(a)+1),'source_gold_rule':a,'source_decoy_rule':b,'source':'frozen_dev_200'})
 # CS is generated only from these dev cases: two complementary 6-month branches.
 for r in [r for r in selected if r['variant']=='NC']:
  if r['tree']!='pancreatic_11':
   exclusions.append({'id':r['id'],'variant':'CS','reason':'no_supported_complementary_numeric_pair'});continue
  specs=[re.search(r'([<≥])6 mo from completion of primary therapy',q['condition']) for q in r['rules']]
  hits=explicit_months(r['patient'])
  if not all(specs) or {s[1] for s in specs}!={'<','≥'} or not hits:
   exclusions.append({'id':r['id'],'variant':'CS','reason':'no_unique_explicit_completion_interval'});continue
  x=hits[0]['value'];g=int(r['gold'])-1
  if (x<6)!=(specs[g][1]=='<'):
   exclusions.append({'id':r['id'],'variant':'CS','reason':'source_branch_disagrees_with_explicit_interval'});continue
  threshold=x+max(.1,abs(x)*.1) if x>=6 else x-max(.1,abs(x)*.1)
  if threshold<=0:
   exclusions.append({'id':r['id'],'variant':'CS','reason':'nonpositive_counterfactual_threshold'});continue
  c=copy.deepcopy(r);c['variant']='CS';c['gold']=str(2-g)
  for q in c['rules']:q['condition']=re.sub(r'([<≥])6 mo from completion of primary therapy',lambda m:m[1]+f'{threshold:g} mo from completion of primary therapy',q['condition'])
  c['threshold_proof']={'patient_value':x,'unit':'month','old_threshold':6,'new_threshold':threshold,'evidence':hits,'old_gold':r['gold'],'new_gold':c['gold'],'scope':'selected complementary sibling conditions; source-aligned remaining facts'}
  selected.append(c)
 p=out/'cases.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 dump(res/'split.json',split);dump(res/'exclusions.json',exclusions)
 pub=[{k:r[k] for k in ['id','tree','split','variant','gold','source_gold_rule','source_decoy_rule','source']} for r in selected]
 (res/'selection.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pub))
 audit={'status':'prepared','base_commit':'781ee863784c60f1f6710ba5eede22bc8ee6b93c','counts':dict(collections.Counter(r['variant'] for r in selected)),'trees':sorted({r['tree'] for r in selected}),'dev_tree_count':len(dev),'test_tree_intersection':sorted({r['tree'] for r in selected}&test),'test_case_files_opened':[],'source':str(SOURCE),'source_sha256':sha(SOURCE),'split_source':str(SPLIT),'split_sha256':sha(SPLIT),'cases_sha256':sha(p),'labels':'published source path for NC/AC/IC/SW; scalar execution for CS; not independent clinical adjudication','branch_unit':'two same-parent siblings with distinct conditions; random neutral 1/2 IDs fixed across variants','CS_requested':200,'CS_actual':sum(r['variant']=='CS' for r in selected),'CS_exclusions':dict(collections.Counter(x['reason'] for x in exclusions)),'patient_handling':'remove embedded trailing question from blind input; historical treatment facts retained; target disposition hidden','no_model_output_based_selection':True}
 dump(res/'data_audit.json',audit);print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
