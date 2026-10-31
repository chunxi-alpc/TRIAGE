"""New source-separated development cohort for module one (no model outputs)."""
import json,random
from collections import Counter,defaultdict
from pathlib import Path
from .prepare import ROOT,BASE,OUT,read,tree_guideline,category,build
from .run import write
M1=OUT/'module_one'
def prepare():
 M1.mkdir(exist_ok=True)
 if (M1/'manifest.json').exists():return json.loads((M1/'manifest.json').read_text())
 clusters=json.loads((BASE/'source_clusters.json').read_text())
 excluded=set(clusters['official_test_clusters'])|set(clusters['pr40_clusters'])|set(clusters['previous_mechanism_clusters'])
 excluded.update(r['source_id'] for r in read(BASE/'data02/candidate_pairs.jsonl'))
 for folder in [BASE/'cases']:
  for f in folder.glob('*.jsonl'):
   for r in read(f):
    if r.get('source_id'):excluded.add(r['source_id'])
 excluded.update(r['source_id'] for r in read(OUT/'validation120.jsonl'))
 counts=Counter(int(r['guideline_id']) for r in read(ROOT/'train_val/vignette_cases.jsonl'))
 candidates=defaultdict(list)
 for t in read(ROOT/'train_val/decision_trees.jsonl'):
  gid=int(t['guideline_id']);source=clusters['records']['train_val:'+str(gid)]['cluster']
  if source in excluded or counts[gid]<10:continue
  try:tree=json.loads(t['text'])['decision_tree'];tree_guideline(tree)
  except (ValueError,KeyError,TypeError):continue
  candidates[category(str(tree.get('description','')))].append((gid,source))
 rng=random.Random(20261031);chosen=[];used=set()
 for cat in ['age','laboratory','stage_diagnosis','time_treatment','symptom_other']:
  xs=sorted(candidates[cat]);rng.shuffle(xs);n=0
  for gid,source in xs:
   if source in used:continue
   chosen.append({'category':cat,'guideline_id':gid,'source_id':source});used.add(source);n+=1
   if n==2:break
  if n!=2:raise ValueError('Insufficient sources for '+cat)
 full=build('train_val',{x['guideline_id']:'module_one_development' for x in chosen},clusters['records'])
 bysource=defaultdict(list)
 for r in full:bysource[r['source_id']].append(r)
 picked=[]
 for source,rows in sorted(bysource.items()):
  bins=defaultdict(list)
  for r in rows:
   depth=len(r['gold_rules']);band=0 if depth<=3 else 1 if depth<=7 else 2
   bins[(r['gold_rules'][0][-1],band)].append(r)
  for rs in bins.values():rng.shuffle(rs)
  take=[]
  while len(take)<10:
   for k in sorted(bins):
    if bins[k] and len(take)<10:take.append(bins[k].pop())
  picked+=take
 if len(picked)!=100 or len(used)!=10 or used&excluded:raise ValueError('Development allocation mismatch')
 (M1/'development100.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in picked))
 manifest={'seed':20261031,'cases':100,'sources':chosen,'source_counts':dict(Counter(r['source_id'] for r in picked)),
  'root_branch':dict(Counter(r['gold_rules'][0][-1] for r in picked)),
  'depth':dict(sorted(Counter(len(r['gold_rules']) for r in picked).items())),
  'excluded_sources':sorted(excluded),'source_overlap':[],
  'selection':'2 distinct sources per coarse root category, 10 cases per source, root/depth round robin; no model outcomes',
  'roles':{'fit':'original 16 pairs / 8 sources, unchanged','old43':'historical comparison and candidate proposal',
   'development100':'module-one heads and gain selection','validation120':'frozen validation','confirm4':'exposed diagnostic pairs only'},
  'validation_case_ids':[r['case_id'] for r in read(OUT/'validation120.jsonl')]}
 write(M1/'manifest.json',manifest);return manifest
if __name__=='__main__':
 m=prepare();print(json.dumps({'cases':m['cases'],'sources':len(m['sources']),'root_branch':m['root_branch']},ensure_ascii=False))
