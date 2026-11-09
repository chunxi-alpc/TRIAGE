
import os,sys,json,time
from pathlib import Path
from collections import Counter,defaultdict
OUT=Path('/home/gcx/aster_next_20261109/inventory')
OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/home/gcx/TRIAGE/data/MEGA-CDP')
OLD=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_representation.tree import tree_guideline
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 start=time.time()
 clusters=json.loads(Path('/home/gcx/aster_manuscript_20261030/clinical01/source_clusters.json').read_text());rec=clusters['records']
 current={r['source_id']:r['split'] for r in rows(OLD/'data/sources.jsonl')}
 hist=json.loads((OLD/'data/history_usage.json').read_text())
 used=set(hist['excluded'])|set(current)
 actualhistory={s:[x for x in values if ':official_test_clusters' not in x and ':pr40_clusters' not in x and ':previous_mechanism_clusters' not in x] for s,values in hist['excluded'].items()}
 actualhistory={s:v for s,v in actualhistory.items() if v}
 trees={int(r['guideline_id']):r for r in rows(DATA/'test/decision_trees.jsonl')}
 paths={int(r['guideline_id']):r['paths'] for r in rows(DATA/'test/case_paths.jsonl')}
 cases=rows(DATA/'test/vignette_cases.jsonl');cache={};errors=[];sourcestat={};groups=Counter();goldnodes=allnodes=0
 output=OUT/'path_cases.jsonl'
 with output.open('w',encoding='utf-8') as f:
  for c in cases:
   gid=int(c['guideline_id']);key='test:'+str(gid);source=rec[key]['cluster'];role=current.get(source,'none')
   if gid not in cache:
    try:cache[gid]=tree_guideline(json.loads(trees[gid]['text'])['decision_tree'])
    except Exception as e:cache[gid]=None;errors.append(dict(guideline_id=gid,error=str(e)))
   if cache[gid] is None:continue
   guide,mapping=cache[gid]
   path=paths[gid][int(c['path_id'])-1]
   gold=[dict(gid='G'+str(mapping[str(x['index'])]['node']),label=int(x['decision_outcome']=='yes')) for x in path if 'decision_outcome' in x]
   leaf='L'+str(mapping[str(path[-1]['index'])]['node'])
   history_group='current_'+role if role!='none' else ('earlier_recorded_source' if source in actualhistory else 'no_recorded_input_use')
   r=dict(case_id='mega:test:'+str(gid)+':'+str(c['path_id']),source_id=source,source_key=key,split='test',
    dataset='MEGA-CDP',dataset_split='official_test',guideline_id=gid,source_filename=trees[gid]['filename'],
    patient=c['text'],guideline=guide,definitions=[],gold_path=gold,gold_leaf=leaf,language=trees[gid].get('language'),
    history_group=history_group,current_source_role=role,known_earlier_input_use=source in actualhistory)
   f.write(json.dumps(r,ensure_ascii=False)+'\n');groups[history_group]+=1
   goldnodes+=len(gold);allnodes+=len(guide['groups'])
   ss=sourcestat.setdefault(source,dict(source_id=source,source_keys=set(),guideline_ids=set(),cases=0,current_role=role,history_group=history_group,known_earlier_input_files=actualhistory.get(source,[])))
   ss['source_keys'].add(key);ss['guideline_ids'].add(gid);ss['cases']+=1
 with (OUT/'sources.jsonl').open('w',encoding='utf-8') as f:
  for source,ss in sorted(sourcestat.items()):
   ss['source_keys']=sorted(ss['source_keys']);ss['guideline_ids']=sorted(ss['guideline_ids'])
   f.write(json.dumps(ss,ensure_ascii=False)+'\n')
 totals={};remainingcases=remainingtrees=0;remainingsources=set();totalsources=set()
 for part in ['train_val','test']:
  tr=rows(DATA/part/'decision_trees.jsonl');pa=rows(DATA/part/'case_paths.jsonl');ca=rows(DATA/part/'vignette_cases.jsonl')
  totals[part]=dict(trees=len(tr),cases=len(ca),reference_paths=sum(len(x['paths']) for x in pa),
   source_clusters=len({rec[part+':'+str(int(x['guideline_id']))]['cluster'] for x in tr}))
  totalsources.update(rec[part+':'+str(int(x['guideline_id']))]['cluster'] for x in tr)
  if part=='train_val':
   remainingtrees=sum(rec['train_val:'+str(int(x['guideline_id']))]['cluster'] not in used for x in tr)
   for x in ca:
    source=rec['train_val:'+str(int(x['guideline_id']))]['cluster']
    if source not in used:remainingcases+=1;remainingsources.add(source)
 summary=dict(dataset='MEGA-CDP',raw_dataset_root=str(DATA),description='Generated case vignettes paired with guideline reference paths, not independently sampled real patients',
  totals=totals,all_trees=sum(x['trees'] for x in totals.values()),all_cases=sum(x['cases'] for x in totals.values()),
  all_reference_paths=sum(x['reference_paths'] for x in totals.values()),all_source_clusters=len(totalsources),
  current_paths=dict(Counter(r['split'] for r in rows(OLD/'data/path_cases.jsonl'))),
  official_manifest=dict(path=str(output),cases=sum(groups.values()),trees=len(cache)-len(errors),source_clusters=len(sourcestat),
    current_role_sources=dict(Counter(s['current_role'] for s in sourcestat.values())),history_cases=dict(groups),
    history_source_clusters=dict(Counter(s['history_group'] for s in sourcestat.values())),gold_decision_nodes=goldnodes,
    all_reachable_case_node_contexts=allnodes,conversion_errors=errors),
  extra_trainval_without_recorded_prior_or_current_source_use=dict(source_clusters=len(remainingsources),trees=remainingtrees,cases=remainingcases),
  historical_notes=['Earlier source history uses existing history_usage.json; official_test_clusters declaration alone is not counted as actual earlier input use.',
   'No current fit/dev/old-test source overlaps are hidden: current_source_role is recorded per case and source.',
   'No new review of every historical directory; no claim that no_recorded_input_use means never seen anywhere.',
   'The total counts describe benchmark vignettes and paths; no independent real-patient sample count is provided.'],
  generator=str(OUT/'prepare_official.py'),seconds=time.time()-start)
 dump(OUT/'inventory.json',summary)
 dump(OUT/'status.json',dict(state='complete',cases=sum(groups.values()),seconds=time.time()-start))
 print(json.dumps(summary,ensure_ascii=True))
if __name__=='__main__':main()

