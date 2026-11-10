"""Validate source separation and saved task/pair joins without model calls."""
import sys,json
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE))
from data import OUT,BIG,BASE,MAP,rows,dump,match_source
selection=json.loads((OUT/'source_selection.json').read_text());records=json.loads(MAP.read_text())['records']
history=json.loads((OUT/'history_scan.json').read_text())
used=set(json.loads(Path('/home/gcx/aster_cr_20261108/data/history_usage.json').read_text())['excluded'])
for p in [BASE/'data/sources.jsonl',Path('/home/gcx/aster_cr_repair_20261108/data/sources.jsonl')]:
 for r in rows(p):used.update(match_source(r,records))
for name in ['mega_train_val','mega_official_test']:
 for p in (Path('/home/gcx/aster_next_20261109/primary_results')/name).glob('shard*/results.jsonl'):
  for r in rows(p):used.update(match_source(r,records))
if len(used)==history['used_source_count']:
 history['excluded_source_ids']=sorted(used);dump(OUT/'history_scan.json',history)
source_sets={s:set() for s in ['fit','dev','final']};counts={};bad=[];meta={}
for split in ['fit','dev','final']:
 n=0;seen=set();variants=Counter();case_counts=Counter()
 for c in rows(OUT/f'case_index_{split}.jsonl'):
  key=(c['case_id'],c['variant'])
  if key in seen:bad.append(['duplicate_case_variant',split,list(key)])
  seen.add(key);source_sets[split].add(c['source_id']);variants[c['variant']]+=1
  if c['variant']=='original':case_counts[c['source_id']]+=1
 for t in rows(BIG/f'tasks_{split}.jsonl'):
  if t['id'] in meta:bad.append(['duplicate_main_task_id',t['id']])
  meta[t['id']]={k:t.get(k) for k in ['source_id','split','label','case_id','gid','variant']};n+=1
 if split!='fit' and max(case_counts.values())>15:bad.append(['source_case_cap',split])
 counts[split]=dict(tasks=n,cases_original=variants['original'],sources=len(source_sets[split]),maximum_cases_per_source=max(case_counts.values()),variants=dict(variants))
for split in ['dev','final']:
 for t in rows(BIG/f'tasks_diagnostic_{split}.jsonl'):
  m={k:t.get(k) for k in ['source_id','split','label','case_id','gid','variant']}
  if t['id'] in meta and meta[t['id']]!=m:bad.append(['diagnostic_main_id_metadata_conflict',t['id']])
  meta[t['id']]=m
pair_counts=Counter()
for p in rows(BIG/'pairs.jsonl'):
 pair_counts[p['kind']]+=1
 if p['left'] not in meta or p['right'] not in meta:bad.append(['pair_task_missing',p['pair_id']]);continue
 a,b=meta[p['left']],meta[p['right']]
 if a['source_id']!=b['source_id'] or a['source_id']!=p['source_id'] or a['split']!=b['split'] or a['split']!=p['split']:bad.append(['pair_source_split_mismatch',p['pair_id']])
 if a['label']!=p['left_label'] or b['label']!=p['right_label']:bad.append(['pair_label_mismatch',p['pair_id']])
overlaps={a+'_'+b:sorted(source_sets[a]&source_sets[b]) for a,b in [('fit','dev'),('fit','final'),('dev','final')]}
new=source_sets['dev']|source_sets['final'];recorded=sorted(new&set(history.get('excluded_source_ids',used)))
result=dict(source_counts=counts,source_overlaps=overlaps,new_sources_in_recorded_history=recorded,pair_counts=dict(pair_counts),inconsistencies=bad,passed=not bad and not any(overlaps.values()) and not recorded,final_scoring_status='not performed by data preparation; final method evaluation belongs after frozen selection',seed=selection['seed'],source_cluster_definition=str(MAP),new_source_claim='No recorded project use in scanned experimental records; not evidence of model-pretraining exclusion or new clinical patients.')
dump(OUT/'data_checks.json',result)
print(json.dumps(dict(passed=result['passed'],source_counts=counts,pair_counts=dict(pair_counts),inconsistencies=len(bad))))
if not result['passed']:raise SystemExit(1)
