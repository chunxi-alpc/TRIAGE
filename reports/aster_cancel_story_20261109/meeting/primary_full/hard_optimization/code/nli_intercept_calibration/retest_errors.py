
"""Retest two pre-frozen scalar controls on the exact prior NLI error pools."""
from pathlib import Path
import json,csv,sys,time,datetime
from collections import defaultdict,Counter
import numpy as np
BASE=Path('/home/gcx/aster_next_20261109')
OUT=Path('/home/gcx/aster_hard_optimization_20261109/nli_intercept_calibration')
DEST=OUT/'error_retest'
CONFIGS={
 'balanced_NLI_intercept':dict(original='optimized_balanced_fit_calibration',selected=OUT/'selected.json'),
 'old_linear_NLI_intercept':dict(original='old_linear',selected=OUT/'old_linear/selected.json')}
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def csvrows(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(path,rr):
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def main():
 started=time.time();DEST.mkdir(parents=True,exist_ok=True)
 configs={}
 for name,cfg in CONFIGS.items():
  s=json.loads(cfg['selected'].read_text())
  if not s['frozen'] or not s['dev_criterion_met']:raise RuntimeError('Scalar control did not pass its predetermined dev rule')
  configs[name]=dict(original=cfg['original'],selected=str(cfg['selected']),shift=float(s['intercept_shift']),
   b=float(s['b']),saved_at=s['saved_at'],fit_dev_metrics=s['fit_dev_metrics'])
 cases={c['case_id']:c for c in rows(BASE/'inventory/nli4ct_all_cases.jsonl')}
 old_errors={r['case_id'] for r in rows(BASE/'error_retest/retest_results.jsonl') if r['stage']=='nli4ct_all' and r['method']=='old_linear'}
 selected_rows=[r for r in csvrows(BASE/'error_retest/selected_error_pair_scores.csv') if r['dataset']=='NLI4CT-2024' and r['method']=='old_linear']
 pairids={r['pair_id'] for r in selected_rows};native={p['pair_id']:p for p in rows(BASE/'inventory/nli4ct_native_pairs.jsonl')}
 assert len(old_errors)==2554 and len(pairids)==2677
 required=set(old_errors)
 for p in pairids:required.update([native[p]['left'],native[p]['right']])
 a=np.load(BASE/'primary_results/nli4ct_all/pair_summary/case_scores.npz')
 ix={v:i for i,v in enumerate(a['ids'].tolist())}
 new_scores={name:{cid:float(a[cfg['original']][ix[cid]]+cfg['shift']) for cid in required} for name,cfg in configs.items()}
 case_rows=[];case_metrics={};repair_sets={}
 dev_ids={c['native_id'] for c in cases.values() if c['dataset_split']=='dev'}
 for name,values in new_scores.items():
  repair_sets[name]=set();by_split=defaultdict(list)
  for cid in sorted(old_errors):
   case=cases[cid];score=values[cid];correct=int(score>0)==int(case['label'])
   row=dict(method=name,case_id=cid,dataset_split=case['dataset_split'],native_id=case['native_id'],base_id=case['base_id'],
    label=int(case['label']),original_method=configs[name]['original'],
    original_score=float(a[configs[name]['original']][ix[cid]]),intercept_shift=configs[name]['shift'],
    calibrated_score=score,repaired=correct,still_wrong=not correct,
    duplicates_dev_native_id=case['native_id'] in dev_ids,shares_dev_base_family=case['base_id'] in dev_ids)
   case_rows.append(row);by_split[case['dataset_split']].append(row)
   if correct:repair_sets[name].add(cid)
  case_metrics[name]={sp:dict(old_error_cases=len(rr),repaired=sum(r['repaired'] for r in rr),still_wrong=sum(r['still_wrong'] for r in rr),
   conditional_repair_rate=sum(r['repaired'] for r in rr)/len(rr),
   duplicate_dev_native_ids=sum(r['duplicates_dev_native_id'] for r in rr),shared_dev_families=sum(r['shares_dev_base_family'] for r in rr)) for sp,rr in by_split.items()}
 csvwrite(DEST/'case_results.csv',case_rows)
 sys.path.insert(0,str(BASE/'analysis'));from summarize_error_pairs import stats
 scoremap={}
 for r in csvrows(BASE/'primary_results/nli4ct_all/pair_summary/pair_results.csv'):
  if r['pair_id'] in pairids and r['method'] in ('raw','old_linear','optimized_balanced_fit_calibration'):scoremap[r['method'],r['pair_id']]=r
 pair_rows=[]
 for name,values in new_scores.items():
  for p in sorted(pairids):
   meta=native[p];sl=values[meta['left']];sr=values[meta['right']]
   row=dict(method=name,pair_id=p,split=meta['dataset_split'],intervention=meta['intervention'],relation=meta['relation'],
    left_label=meta['left_label'],right_label=meta['right_label'],left_score=sl,right_score=sr,
    paired_correct=int(sl>0)==meta['left_label'] and int(sr>0)==meta['right_label'])
   pair_rows.append(row);scoremap[name,p]=row
 csvwrite(DEST/'pair_results.csv',pair_rows)
 groups={};checks={}
 diffkeys=('difference_rms','raw_difference_rms','correction_rms','difference_rms_ratio','variance_reduction',
  'signed_response_retention','response_direction_degraded','response_direction_improved')
 methods=['raw','old_linear','optimized_balanced_fit_calibration']+list(configs)
 for sp in ('test','practice'):
  spp=[p for p in pairids if native[p]['dataset_split']==sp]
  interventions=['all']+sorted({native[p]['intervention'] for p in spp})
  for intervention in interventions:
   for relation in ('preserving','changing'):
    pp=sorted(p for p in spp if native[p]['relation']==relation and (intervention=='all' or native[p]['intervention']==intervention))
    if not pp:continue
    key=sp+':'+intervention+':'+relation;g=dict(split=sp,intervention=intervention,relation=relation,old_error_pairs=len(pp),methods={})
    for method in methods:
     point=stats(pp,native,scoremap,method,relation)
     if relation=='changing':point['mean_signed_response_retention']=point.get('signed_response_retention')
     g['methods'][method]=point
    for name,cfg in configs.items():
     before=g['methods'][cfg['original']];after=g['methods'][name];maximum=0.
     for k in diffkeys:
      if k in before and before[k] is not None:maximum=max(maximum,abs(after[k]-before[k]))
     if maximum>1e-10:raise RuntimeError('Constant intercept changed pair differences beyond roundoff: '+key+' '+name)
     checks[key+':'+name]=maximum
    groups[key]=g
 pair_metrics=dict(protocol='Two scalar in-domain calibrations with unchanged frozen correction functions',groups=groups,
  original_error_pairs=len(pairids),required_endpoints=len(required),max_difference_metric_roundoff=max(checks.values(),default=0.),
  difference_invariance_verified=True,faithfulness_improvement_claim=False,
  meaning='Classification may change via one train-fitted threshold; paired score differences/VR/ER/direction do not change.')
 dump(DEST/'pair_metrics.json',pair_metrics)
 previous_case=Path('/home/gcx/aster_hard_optimization_20261109/nli_adaptation/error_retest/case_results.csv')
 comparison={}
 if previous_case.exists():
  previous={r['case_id'] for r in csvrows(previous_case) if r['dataset_split']=='test' and r['repaired'].lower()=='true'}
  for name,values in repair_sets.items():
   nowset={cid for cid in values if cases[cid]['dataset_split']=='test'}
   comparison[name]=dict(new_supervised_weight_repairs=len(previous),scalar_control_repairs=len(nowset),
    repairs_shared=len(previous&nowset),supervised_only_repairs=len(previous-nowset),scalar_only_repairs=len(nowset-previous))
 metrics=dict(protocol='NLI in-domain intercept calibration; train1700 fits one scalar for each of two fixed functions; dev checks only; no threshold sweep',
  frozen_controls=configs,case_metrics_by_method=case_metrics,pair_metrics_file=str(DEST/'pair_metrics.json'),
  comparison_with_supervised_residual= comparison,
  old_error_pool_cases=len(old_errors),old_error_pool_pairs=len(pairids),new_scores_required_endpoints=len(required),
  full_population_new_errors=None,net_full_population_benefit=None,
  limitations='The entire old-correct population was not rescored. Conditional fixes are not net benefit. Practice includes dev records and all its base families. No improvement in faithfulness is attributed to an intercept.',
  no_new_forward=True,no_GPU=True,no_additional_fits=True,no_parameter_reselection=True,
  finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.time()-started)
 dump(DEST/'metrics.json',metrics)
 dump(DEST/'done.json',dict(state='complete',methods=list(configs),seconds=time.time()-started))
 print(json.dumps(dict(case_metrics=case_metrics,supervised_overlap=comparison,difference_metrics_invariant=True)))
if __name__=='__main__':main()

