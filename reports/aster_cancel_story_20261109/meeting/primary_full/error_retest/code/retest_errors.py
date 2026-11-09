
"""CPU-only replay of frozen methods on the fixed old_linear error pool."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,csv,time,argparse
from pathlib import Path
from collections import defaultdict,Counter
BASE=Path('/home/gcx/aster_next_20261109')
sys.path.insert(0,str(BASE/'inference'))
from frozen import FrozenMethods
from run_primary import iter_rows,tree_nodes,evaluation_group
DEFAULT_METHODS=['old_linear','optimized_balanced_fit_calibration','optimized_strict_fit_calibration','trace_only_balanced_fit_calibration','raw','fit_bias_only']
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lightweight_cases(path,wanted):
 found={}
 for c in iter_rows(path):
  if c['case_id'] not in wanted:continue
  nodes,root=tree_nodes(c)
  keep={k:c.get(k) for k in ['case_id','source_id','source_key','dataset','dataset_split','split','evaluation_group','history_group','task_type','gold_path','gold_leaf']}
  keep['evaluation_group']=evaluation_group(c);keep['root']=root
  keep['nodes']={gid:{k:n[k] for k in ['id','yes','no']} for gid,n in nodes.items()}
  found[c['case_id']]=keep
 return found
def first_error(path,gold):
 return next((p['gid'] for i,p in enumerate(path) if i>=len(gold) or p!=gold[i]),None)
def group_key(record):return record.get('evaluation_group') or record.get('split_group') or record.get('dataset_split') or 'unspecified'
def main(args):
 started=time.time();out=Path(args.out_root);out.mkdir(parents=True,exist_ok=True)
 plan=json.loads(Path(args.plan).read_text());definitions={s['name']:s for s in plan['stages']}
 stages=args.stage or ['mega_official_test','nli4ct_all'];methods=DEFAULT_METHODS
 adapters=FrozenMethods(args.frozen_manifest)
 all_rows=[];stage_metrics={};mismatches=[];compared=0;missing_all=[];cached_node_calculations=0
 for stage_name in stages:
  stage=definitions[stage_name];stage_root=Path(plan['results_root'])/stage_name
  existing=defaultdict(dict);result_files=sorted(stage_root.glob('shard*/results.jsonl'))
  for path in result_files:
   for row in iter_rows(path):
    name=row['method']
    if name not in methods:continue
    # Retain decisions for consistency and background counts, not prompt material.
    existing[row['case_id']][name]={k:row.get(k) for k in ['case_id','method','correct','status','error','predicted_path','predicted_leaf','first_error','evaluation_group','split_group','dataset_split']}
  wrong={cid for cid,by_method in existing.items() if 'old_linear' in by_method and by_method['old_linear'].get('status')=='ok' and by_method['old_linear'].get('correct') is False}
  cases=lightweight_cases(stage['cases'],wrong)
  cached={};cache_origins={}
  node_files=sorted(stage_root.glob('shard*/nodes.jsonl'))+sorted((BASE/'results').glob('shard*/nodes.jsonl'))
  for path in node_files:
   for r in iter_rows(path):
    if r.get('case_id') not in wrong or r.get('input_variant','canonical')!='canonical' or '|' in r.get('id',''):continue
    if r.get('status')=='ok' and r.get('feature') and Path(r['feature']).exists():
     if r['id'] not in cached:cached[r['id']]=r;cache_origins[r['id']]=str(path)
  # Additional immutable canonical path features, if a stage imported only its results.
  for path in sorted(Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen/paths').glob('features_*.jsonl')):
   for r in iter_rows(path):
    if r.get('case_id') in wrong and r.get('status')=='ok' and r.get('feature') and Path(r['feature']).exists() and r['id'] not in cached:
     cached[r['id']]=r;cache_origins[r['id']]=str(path)
  memo={};node_errors={}
  def scores(cid,gid):
   nonlocal cached_node_calculations
   key=cid+':'+gid
   if key in memo:return memo[key]
   if key not in cached:node_errors[key]='canonical_node_cache_absent';memo[key]=None;return None
   try:memo[key]=adapters.predict_file(cached[key]);cached_node_calculations+=1
   except (OSError,ValueError,KeyError) as exc:node_errors[key]='cached_feature_unavailable: '+str(exc);memo[key]=None
   return memo[key]
  def replay(c,name):
   gid=c['root'];nodes=c['nodes'];path=[];error=None;missing=[]
   while gid in nodes:
    if len(path)>len(nodes):error='cycle';break
    values=scores(c['case_id'],gid)
    if values is None:error='cache_missing';missing.append(c['case_id']+':'+gid);break
    label=int(values[name]>0);path.append(dict(gid=gid,label=label));gid=nodes[gid]['yes' if label else 'no']
   correct=not error and path==c['gold_path'] and gid==c['gold_leaf']
   return dict(correct=bool(correct),status='ok' if error is None else 'unavailable',error=error,
    predicted_path=path,predicted_leaf=gid,new_first_error=first_error(path,c['gold_path']),missing_nodes=missing)
  rows=[]
  for cid in sorted(wrong):
   old=existing[cid]['old_linear'];case=cases.get(cid)
   for name in methods:
    if case is None:
     result=dict(correct=False,status='unavailable',error='case_manifest_missing',predicted_path=[],predicted_leaf=None,new_first_error=None,missing_nodes=[])
     group=group_key(old);old_first=old.get('first_error')
    else:
     result=replay(case,name);group=case['evaluation_group']
     old_first=old.get('first_error') or first_error(old.get('predicted_path') or [],case['gold_path'])
    evaluated=result['status']=='ok';reference=existing[cid].get(name);match=None
    if evaluated and reference and reference.get('status')=='ok':
     match=bool(result['correct']==reference['correct'] and result['predicted_path']==reference.get('predicted_path') and result['predicted_leaf']==reference.get('predicted_leaf'))
     compared+=1
     if not match:mismatches.append(dict(stage=stage_name,case_id=cid,method=name))
    row=dict(stage=stage_name,dataset=stage.get('dataset'),case_id=cid,source_id=case.get('source_id') if case else None,
     dataset_split=case.get('dataset_split') if case else old.get('dataset_split'),evaluation_group=group,method=name,
     original_old_linear_correct=False,original_first_error=old_first,
     first_error_changed=bool(old_first!=result['new_first_error']),repaired=bool(evaluated and result['correct']),
     still_wrong=bool(evaluated and not result['correct']),reference_match=match,reference_correct=reference.get('correct') if reference else None,
     gold_path=case.get('gold_path') if case else None,gold_leaf=case.get('gold_leaf') if case else None,**result)
    rows.append(row)
    if not evaluated:missing_all.append(dict(stage=stage_name,case_id=cid,method=name,error=result['error'],missing_nodes=result['missing_nodes']))
  groups={}
  for group in ['all']+sorted({r['evaluation_group'] for r in rows}):
   selected=rows if group=='all' else [r for r in rows if r['evaluation_group']==group]
   pool=len({r['case_id'] for r in selected});stats={}
   for name in methods:
    rr=[r for r in selected if r['method']==name];available=[r for r in rr if r['status']=='ok'];repair=[r for r in available if r['repaired']]
    stats[name]=dict(expected=pool,evaluated=len(available),missing=pool-len(available),repaired=len(repair),still_wrong=sum(r['still_wrong'] for r in available),
     repair_rate=len(repair)/pool if pool else None,repair_rate_available=len(repair)/len(available) if available else None,
     repaired_cases=[r['case_id'] for r in repair],first_error_resolved=len(repair),
     first_error_moved=sum(not r['repaired'] and r['first_error_changed'] for r in available),
     first_error_unchanged=sum(not r['first_error_changed'] for r in available))
   groups[group]=dict(pool_size=pool,methods=stats)
  # Context only: these outcomes are copied from the earlier completed full evaluation.
  background_groups={}
  for group in ['all']+sorted({group_key(v['old_linear']) for v in existing.values() if 'old_linear' in v}):
   correct_pool={cid:v for cid,v in existing.items() if v.get('old_linear',{}).get('status')=='ok' and v['old_linear'].get('correct') is True and (group=='all' or group_key(v['old_linear'])==group)}
   bm={}
   for name in methods:
    known={cid:v[name] for cid,v in correct_pool.items() if name in v and v[name].get('status')=='ok'}
    errors=[cid for cid,r in known.items() if r.get('correct') is False]
    bm[name]=dict(original_correct_pool=len(correct_pool),cached_available=len(known),cached_new_errors=len(errors),cached_new_error_cases=errors)
   background_groups[group]=dict(methods=bm)
  common_available=len({cid for cid in wrong if all(r['status']=='ok' for r in rows if r['case_id']==cid)})
  stage_metrics[stage_name]=dict(dataset=stage.get('dataset'),source_manifest=stage['cases'],full_record_count=stage['expected'],
   pool=dict(criterion='Existing old_linear status=ok and correct=false',expected=len(wrong),available=common_available,
    missing_cases=sorted({r['case_id'] for r in rows if r['status']!='ok'})),
   groups=groups,cached_background=dict(description='Previously completed full-evaluation cache only; no new full-data test performed',methods=background_groups['all']['methods'],groups=background_groups),
   canonical_cached_nodes=len(cached),recalculated_nodes=len(memo)-sum(v is None for v in memo.values()),
   node_cache_files=[str(p) for p in node_files],result_files=[str(p) for p in result_files])
  all_rows.extend(rows)
  dump(out/(stage_name+'.summary.json'),stage_metrics[stage_name])
  dump(out/'status.json',dict(state='running',last_stage_completed=stage_name,seconds=time.time()-started,case_method_rows=len(all_rows)))
 with (out/'retest_results.jsonl').open('w') as f:
  for row in all_rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
 fields=['stage','dataset','dataset_split','evaluation_group','case_id','source_id','method','original_old_linear_correct','repaired','still_wrong','correct','status','error','original_first_error','new_first_error','first_error_changed','reference_match','reference_correct','predicted_path','predicted_leaf','gold_path','gold_leaf','missing_nodes']
 with (out/'retest_results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for row in all_rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items() if k in fields})
 validation=dict(checked_case_method_results=compared,prediction_mismatches=len(mismatches),mismatches=mismatches,
  same_frozen_configuration=True,new_training=False,new_gpu_forwards=0,new_full_evaluation=False,missing_case_method_results=len(missing_all))
 dump(out/'consistency.json',validation)
 metrics=dict(protocol='Recompute frozen numerical readouts on fixed old_linear error cases; MEGA dynamic paths and NLI single-node replay',
  baseline='old_linear',methods=methods,frozen_manifest=str(args.frozen_manifest),stages=stage_metrics,consistency=validation,
  scope='Error-pool retest only. Earlier full-cache newly wrong counts are context, not a new full-data accuracy test.',
  cpu_only=True,new_gpu_forwards=0,new_training=False,cached_node_calculations=cached_node_calculations,
  files=dict(case_jsonl=str(out/'retest_results.jsonl'),case_csv=str(out/'retest_results.csv')),seconds=time.time()-started)
 dump(out/'metrics.json',metrics);dump(out/'missing.json',missing_all)
 dump(out/'done.json',dict(state='complete_with_missing' if missing_all else 'complete',seconds=time.time()-started,case_method_rows=len(all_rows),prediction_mismatches=len(mismatches)))
 print(json.dumps(dict(state='complete_with_missing' if missing_all else 'complete',out=str(out),stages={s:dict(pool=v['pool'],repairs={k:x['repaired'] for k,x in v['groups']['all']['methods'].items()}) for s,v in stage_metrics.items()},consistency=validation,seconds=time.time()-started)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',default=str(BASE/'primary_plan.json'));p.add_argument('--frozen-manifest',default=str(BASE/'frozen_methods.json'))
 p.add_argument('--stage',action='append');p.add_argument('--out-root',default=str(BASE/'error_retest'));main(p.parse_args())

