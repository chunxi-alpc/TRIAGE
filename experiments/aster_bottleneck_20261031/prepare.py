"""Register development and new-source validation before model runs."""
import json,random,re
from pathlib import Path
from collections import Counter,defaultdict
from experiments.aster_manuscript_clinical_20261030.data import ROOT,BASE,read,tree_guideline
from experiments.aster_manuscript_clinical_20261030.cases import build
OUT=Path('/home/gcx/aster_bottleneck_20261031')
SEED=20261031
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def category(text):
 for name,pat in [('time_treatment',r'treat|therap|previous|month|week|治疗|术后|既往'),('laboratory',r'level|count|test|positive|negative|检测|阳性|浓度'),('stage_diagnosis',r'diagnos|stage|cancer|disease|症|分期|诊断'),('age',r'age|years|年龄|岁')]:
  if re.search(pat,text,re.I):return name
 return 'symptom_other'
def main():
 OUT.mkdir(exist_ok=False)
 clusters=json.loads((BASE/'source_clusters.json').read_text())
 excluded=set(clusters['official_test_clusters'])|set(clusters['pr40_clusters'])|set(clusters['previous_mechanism_clusters'])
 for x in read(BASE/'data02/candidate_pairs.jsonl'):excluded.add(x['source_id'])
 grouped=defaultdict(list)
 for x in read(ROOT/'train_val/vignette_cases.jsonl'):grouped[int(x['guideline_id'])].append(x)
 candidates=[];failures=[]
 for t in read(ROOT/'train_val/decision_trees.jsonl'):
  gid=int(t['guideline_id']);key='train_val:'+str(gid);source=clusters['records'][key]['cluster']
  if source in excluded or len(grouped[gid])<10:continue
  try:
   tree=json.loads(t['text'])['decision_tree'];guide,mapping=tree_guideline(tree)
   candidates.append((category(str(tree.get('description',''))),gid,source,len(guide['groups'])))
  except Exception as e:failures.append({'guideline_id':gid,'reason':str(e)})
 rng=random.Random(SEED);rng.shuffle(candidates);buckets=defaultdict(list)
 for c in candidates:buckets[c[0]].append(c)
 chosen=[];seen=set()
 while len(chosen)<12 and any(buckets.values()):
  for cat in sorted(buckets):
   while buckets[cat] and buckets[cat][-1][2] in seen:buckets[cat].pop()
   if buckets[cat] and len(chosen)<12:
    c=buckets[cat].pop();chosen.append(c);seen.add(c[2])
 if len(chosen)!=12:raise RuntimeError('Fewer than 12 available new source clusters')
 full=build('train_val',{x[1]:'new_validation' for x in chosen},clusters['records'])
 bysource=defaultdict(list)
 for c in full:bysource[c['source_id']].append(c)
 validation=[]
 for source,rows in sorted(bysource.items()):
  # Round-robin by root branch and path-depth band; never inspect predictions.
  bins=defaultdict(list)
  for c in rows:
   depth=len(c.get('gold_rules',[]));branch=c.get('gold_rules',['UNKNOWN'])[0][-1:]
   bins[(branch,min(depth//4,3))].append(c)
  for xs in bins.values():rng.shuffle(xs)
  picked=[]
  while len(picked)<10:
   for k in sorted(bins):
    if bins[k] and len(picked)<10:picked.append(bins[k].pop())
  validation.extend(picked)
 dev=read(BASE/'cases/confirm.jsonl')
 for name,rows in [('development43',dev),('validation120',validation)]:
  (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  write(OUT/(name+'_coverage.json'),{'cases':len(rows),'sources':dict(Counter(x['source_id'] for x in rows)),
   'root_branch':dict(Counter(x.get('gold_rules',['UNKNOWN'])[0][-1:] for x in rows)),
   'depth':dict(sorted(Counter(len(x.get('gold_rules',[])) for x in rows).items()))})
 manifest={'date':'2026-10-31','seed':SEED,'model':'Qwen3-8B','physical_gpus':[0,1,5,6],
  'development_cases':43,'new_validation_cases':120,'new_validation_sources':12,
  'new_sources':[{'category':x[0],'guideline_id':x[1],'source':x[2],'groups':x[3]} for x in chosen],
  'selection':'source-stratified random selection, then root/depth round-robin; no performance filtering',
  'excluded_sources':'all prior direction/calibration/development/confirmation sources, prior mechanism/PR40 and official-test source clusters',
  'source_population':'MEGA-CDP train_val benchmark vignettes; not real prospective patient records',
  'calibration':'old calibration18; development43 is exposed and used only for development',
  'initial_arms':['zero','spectral025','mean025','clip','neutral','hard','hard_siblings','bias1','bias2','bias4','bias8'],
  'validation_arms':['zero','clip','hard','mean025','selected_soft'],
  'decoding':{'greedy':True,'max_new_tokens':512,'thinking':False},
  'metrics':['strict_exact_path','leaf_correct','root_decision_log_probability_margin','paired_corrections_regressions'],
  'selection_rule':'development exact path, then decision margin, then smaller beta; validation never selects settings',
  'resources_estimate':{'memory_gb_per_gpu':[25,40],'initial_minutes':[40,100]},
  'no_hash_checks':True,'construction_failures':failures}
 write(OUT/'manifest.json',manifest)
 print(json.dumps({'cases':163,'sources_new':12,'root_branch':dict(Counter(x['gold_rules'][0][-1] for x in validation))}))
if __name__=='__main__':main()
