
"""One fixed error-pool retest of the frozen NLI supervised residual model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,csv,sys,time,datetime
from collections import defaultdict,Counter
import numpy as np
BASE=Path('/home/gcx/aster_next_20261109')
OUT=Path('/home/gcx/aster_hard_optimization_20261109/nli_adaptation')
DEST=OUT/'error_retest'
NAME='nli_supervised_residual_adaptation'
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def csvrows(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(path,rr):
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 started=time.time();DEST.mkdir(parents=True,exist_ok=True)
 selected=json.loads((OUT/'selected.json').read_text())
 if not selected['frozen'] or selected['family']!='trace16' or selected['lambda_']!=.01:raise RuntimeError('Unexpected frozen selected model')
 sys.path.insert(0,str(OUT));from predict import load_model,predict_file,SUPPORT
 model=load_model(selected['model']);support=dict(np.load(SUPPORT))
 sys.path.insert(0,str(BASE/'analysis'));from summarize_error_pairs import stats
 cases={c['case_id']:c for c in rows(BASE/'inventory/nli4ct_all_cases.jsonl')}
 # Use the exact previously frozen old-error case and pair IDs.
 poolrecords=[r for r in rows(BASE/'error_retest/retest_results.jsonl') if r['stage']=='nli4ct_all' and r['method']=='old_linear']
 case_ids={r['case_id'] for r in poolrecords}
 if len(case_ids)!=2554:raise RuntimeError('Old-error case pool count changed')
 error_pair_rows=[r for r in csvrows(BASE/'error_retest/selected_error_pair_scores.csv') if r['dataset']=='NLI4CT-2024' and r['method']=='old_linear']
 pairids={r['pair_id'] for r in error_pair_rows}
 if len(pairids)!=2677:raise RuntimeError('Old-error pair pool count changed')
 native={p['pair_id']:p for p in rows(BASE/'inventory/nli4ct_native_pairs.jsonl')}
 required=set(case_ids)
 for p in pairids:required.update([native[p]['left'],native[p]['right']])
 cached={}
 for path in sorted((BASE/'primary_results/nli4ct_all').glob('shard*/nodes.jsonl')):
  for r in rows(path):
   if r.get('case_id') in required and r.get('gid','G0')=='G0' and r.get('status')=='ok':cached[r['case_id']]=r
 scores={};failures=[]
 for cid in sorted(required):
  r=cached.get(cid)
  if r is None or not r.get('feature') or not Path(r['feature']).is_file():
   failures.append(cid);continue
  scores[cid]=predict_file(r,model,support)
 existing=np.load(BASE/'primary_results/nli4ct_all/pair_summary/case_scores.npz')
 ix={v:i for i,v in enumerate(existing['ids'].tolist())}
 rows_case=[];grouped=defaultdict(list)
 dev_native={c['native_id'] for c in cases.values() if c['dataset_split']=='dev'}
 for cid in sorted(case_ids):
  c=cases[cid];s=scores.get(cid);label=int(c['label'])
  row=dict(case_id=cid,dataset_split=c['dataset_split'],native_id=c['native_id'],base_id=c['base_id'],
   intervention=c.get('native_intervention') or 'original',label=label,source_id=c['source_id'],
   old_linear_score=float(existing['old_linear'][ix[cid]]),raw_score=float(existing['raw'][ix[cid]]),
   previous_balanced_score=float(existing['optimized_balanced_fit_calibration'][ix[cid]]),
   new_score=s,new_prediction=int(s>0) if s is not None else None,
   repaired=(int(s>0)==label) if s is not None else None,still_wrong=(int(s>0)!=label) if s is not None else None,
   status='ok' if s is not None else 'cache_missing',
   training_role='fit' if c['dataset_split']=='train' else 'selection' if c['dataset_split']=='dev' else 'evaluation',
   duplicates_dev_native_id=c['native_id'] in dev_native,
   shares_dev_base_family=c['base_id'] in dev_native)
  rows_case.append(row);grouped[c['dataset_split']].append(row)
 csvwrite(DEST/'case_results.csv',rows_case)
 split_metrics={}
 for split in ['test','practice','dev','train']:
  rr=grouped[split];valid=[r for r in rr if r['status']=='ok']
  repaired=sum(r['repaired'] for r in valid)
  split_metrics[split]=dict(old_error_cases=len(rr),evaluated=len(valid),missing=len(rr)-len(valid),
   repaired=repaired,still_wrong=len(valid)-repaired,conditional_repair_rate=repaired/len(valid) if valid else None,
   duplicate_dev_native_ids=sum(r['duplicates_dev_native_id'] for r in rr),
   shared_dev_families=sum(r['shares_dev_base_family'] for r in rr),
   interpretation='Official old-error subset; not full-population accuracy or a new independent test' if split=='test' else
    'Contains the200 dev originals and their native families; not independent of dev selection' if split=='practice' else
    'Used for model selection' if split=='dev' else 'In-sample1700-record supervised fit')
 # References retain their previously saved endpoints; only the one new model is recomputed.
 scoremap={}
 sourcepair=BASE/'primary_results/nli4ct_all/pair_summary/pair_results.csv'
 reference_names={'raw','old_linear','optimized_balanced_fit_calibration'}
 for r in csvrows(sourcepair):
  if r['pair_id'] in pairids and r['method'] in reference_names:scoremap[r['method'],r['pair_id']]=r
 pair_rows=[]
 for p in sorted(pairids):
  meta=native[p];left=meta['left'];right=meta['right']
  if left in scores and right in scores:
   sl=scores[left];sr=scores[right]
   row=dict(method=NAME,pair_id=p,split=meta['dataset_split'],family=meta['family'],intervention=meta['intervention'],
    relation=meta['relation'],left=left,right=right,left_label=int(meta['left_label']),right_label=int(meta['right_label']),
    left_score=sl,right_score=sr,
    paired_correct=int(sl>0)==int(meta['left_label']) and int(sr>0)==int(meta['right_label']))
   scoremap[NAME,p]=row;pair_rows.append(row)
 csvwrite(DEST/'pair_results.csv',pair_rows)
 groups={}
 methods=['raw','old_linear','optimized_balanced_fit_calibration',NAME]
 for split in ['test','practice']:
  splitids=[p for p in pairids if native[p]['dataset_split']==split]
  interventions=['all']+sorted({native[p]['intervention'] for p in splitids})
  for intervention in interventions:
   for relation in ['preserving','changing']:
    pp=sorted(p for p in splitids if native[p]['relation']==relation and (intervention=='all' or native[p]['intervention']==intervention))
    if not pp:continue
    entry=dict(dataset_split=split,intervention=intervention,relation=relation,old_error_pairs=len(pp),methods={})
    for method in methods:
     point=stats(pp,native,scoremap,method,relation)
     if relation=='changing':point['mean_signed_response_retention']=point.get('signed_response_retention')
     entry['methods'][method]=point
    groups[split+':'+intervention+':'+relation]=entry
 pair_metrics=dict(method_name=selected['method_name'],selected_model=str(OUT/'selected_model.npz'),
  selection='Exact2677 native pairs previously recorded as old_linear paired_correct=False; no new-model outcomes used for pool selection',
  groups=groups,expected_error_pairs=len(pairids),new_model_pairs_scored=len(pair_rows),
  endpoint_case_records_required=len(required),old_error_case_records=len(case_ids),necessary_reference_endpoint_records=len(required-case_ids),
  split_counts=dict(Counter(native[p]['dataset_split'] for p in pairids)),
  interpretation='Native same-label groups are not automatically semantic invariance targets. Signed response retention is relative to raw responses on this conditional error subset. Alpha_raw1 is not a behavior-protection guarantee.',
  whole_population_new_regressions=None,
  regression_note='No full originally-correct population was scored. Necessary pair reference endpoints are not a whole-population regression audit.',
  no_new_backbone_forward=True,no_GPU=True,no_parameter_changes=True)
 dump(DEST/'pair_metrics.json',pair_metrics)
 metrics=dict(method_name=selected['method_name'],model=str(OUT/'selected_model.npz'),model_frozen_at=selected['saved_at'],retest_completed_at=now(),
  pool_definition='Previously saved old16-dimensional old_linear incorrect cases only',
  case_pool_definition_file=str(BASE/'error_retest/retest_results.jsonl'),pair_pool_definition_file=str(BASE/'error_retest/selected_error_pair_scores.csv'),
  case_metrics_by_split=split_metrics,total_old_error_cases=len(case_ids),new_model_case_records_scored=sum(r['status']=='ok' for r in rows_case),
  case_accuracy_on_full_NLI=None,full_population_new_errors=None,
  explicit_limit='Conditional repair counts do not determine net full-population benefit. The new model was not evaluated on the whole old-correct population; prior balanced-model regressions must not be reused as this model regression counts.',
  fit_dev_protocol='1700 NLI train records fit the supervised residual;200 NLI dev records select the sole frozen trace16 lambda.01 model. Practice shares dev originals/families. Official test is shown separately; its old-error pool was already diagnosed, so this is a fixed post-hoc error retest.',
  coverage=dict(required_cached_endpoints=len(required),cached_endpoints=len(scores),missing=failures),
  no_refitting_or_new_search=True,no_new_backbone_forward=True,device='CPU',seconds=time.time()-started)
 dump(DEST/'metrics.json',metrics)
 dump(DEST/'done.json',dict(state='complete',cases=len(case_ids),pairs=len(pairids),seconds=time.time()-started))
 print(json.dumps(dict(case_metrics=split_metrics,required_endpoints=len(required),new_error_pairs_scored=len(pair_rows),seconds=time.time()-started)))
if __name__=='__main__':main()

