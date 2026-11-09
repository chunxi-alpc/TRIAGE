
"""CPU-only fidelity on the previously fixed old_linear error-pair IDs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,csv,time,importlib.util
from collections import defaultdict
import numpy as np
ROOT=Path('/home/gcx/aster_hard_optimization_20261109')
BASE=Path('/home/gcx/aster_next_20261109')
OLD=BASE/'error_retest'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BASE/'inference'))
from predict import Predictor
from frozen import FrozenMethods
from run_primary import iter_rows
def dump(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read_csv(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def finite(x):return float(x) if np.isfinite(x) else None
def main():
 started=time.time();out=ROOT/'evaluation';out.mkdir(exist_ok=True)
 old=json.loads((OLD/'pair_metrics.json').read_text())
 reference_rows=read_csv(OLD/'selected_error_pair_scores.csv')
 fixed=defaultdict(set);scores={};methods=list(old['methods'])
 for r in reference_rows:
  dataset=r['dataset'];pid=r['pair_id'];method=r['method'];scores[dataset,method,pid]=r
  if method=='old_linear':
   if str(r['paired_correct']).lower() not in ('false','0'):raise ValueError('Pool file contains an original correct pair')
   fixed[dataset].add(pid)
 metadata={}
 for p in iter_rows('/home/gcx/aster_cr_repair_v2_run_20261108/data/pairs.jsonl'):
  if p.get('split')=='test' and p['pair_id'] in fixed['MEGA_previous_controlled']:metadata['MEGA_previous_controlled',p['pair_id']]=p
 for p in iter_rows(BASE/'inventory/nli4ct_native_pairs.jsonl'):
  if p['pair_id'] in fixed['NLI4CT-2024']:metadata['NLI4CT-2024',p['pair_id']]=p
 predictor=Predictor(ROOT/'protected_frozen_methods.json');new_methods=list(predictor.entries);methods+=new_methods
 endpoint_scores={};coverage={};missing_endpoints=[]
 for dataset in ['MEGA_previous_controlled','NLI4CT-2024']:
  needed={metadata[dataset,p][side] for p in fixed[dataset] for side in ('left','right')}
  vectors={};feature_sources={}
  if dataset=='MEGA_previous_controlled':
   # Shared frozen coordinates include the old controlled test; the new training data do not.
   sh=np.load('/home/gcx/aster_optimization_20261109/shared/features.npz')
   ids=sh['ids'].tolist();sx={v:i for i,v in enumerate(ids)}
   ell=sh['ell'];r32=sh['r32'];res64=sh['residual64']
   for vid in sorted(needed):
    if vid not in sx:missing_endpoints.append(dict(dataset=dataset,id=vid,reason='shared_coordinates_absent'));continue
    i=sx[vid];s=np.float32(ell[i]);q=np.array([s/10,abs(s)/10,np.tanh(s/4)],dtype='float32')
    vectors[vid]=dict(ell=float(ell[i]),r32=r32[i],residual64=res64[i],q=q)
    feature_sources[vid]='frozen shared coordinates'
  else:
   records={}
   for path in sorted((BASE/'primary_results/nli4ct_all').glob('shard*/nodes.jsonl')):
    for r in iter_rows(path):
     if r.get('case_id') in needed and r.get('gid')=='G0' and r.get('input_variant','canonical')=='canonical' and r.get('status')=='ok':
      records[r['case_id']]=r
   adapter=FrozenMethods(BASE/'frozen_methods.json')
   for vid in sorted(needed):
    r=records.get(vid)
    if not r or not r.get('feature') or not Path(r['feature']).exists():
     missing_endpoints.append(dict(dataset=dataset,id=vid,reason='canonical_feature_absent'));continue
    with np.load(r['feature']) as f:coords=adapter.coordinates(r['ell'],adapter.raw_selected(f),f['residual'])
    vectors[vid]=dict(coords,ell=r['ell']);feature_sources[vid]=r['feature']
  available=sorted(vectors)
  if available:
   a={k:np.stack([vectors[v][k] for v in available]) for k in ['r32','residual64','q']}
   a['ell']=np.array([vectors[v]['ell'] for v in available])
   pred=predictor.predict_from_coordinates(a)
   for name in new_methods:
    for i,vid in enumerate(available):endpoint_scores[dataset,name,vid]=float(pred[name][i])
  for pid in sorted(fixed[dataset]):
   meta=metadata[dataset,pid]
   for method in new_methods:
    keys=[(dataset,method,meta[s]) for s in ('left','right')]
    if not all(k in endpoint_scores for k in keys):continue
    left,right=[endpoint_scores[k] for k in keys]
    scores[dataset,method,pid]=dict(left_score=left,right_score=right)
  coverage[dataset]=dict(selected_old_error_pairs=len(fixed[dataset]),necessary_endpoint_count=len(needed),cached_endpoint_count=len(available),
   missing_endpoints=len(needed)-len(available),new_gpu_forwards=0,
   evaluated_pair_counts={m:sum((dataset,m,p) in scores for p in fixed[dataset]) for m in methods})
 def point(dataset,pids,method,relation,normalization):
  valid=[p for p in pids if (dataset,method,p) in scores and (dataset,'raw',p) in scores]
  if not valid:return dict(n=0,expected=len(pids),missing=len(pids),status='no_old_error_pairs' if not pids else 'missing_scores')
  yl=np.array([metadata[dataset,p]['left_label'] for p in valid]);yr=np.array([metadata[dataset,p]['right_label'] for p in valid])
  l=np.array([float(scores[dataset,method,p]['left_score']) for p in valid]);r=np.array([float(scores[dataset,method,p]['right_score']) for p in valid])
  rl=np.array([float(scores[dataset,'raw',p]['left_score']) for p in valid]);rr=np.array([float(scores[dataset,'raw',p]['right_score']) for p in valid])
  raw=rr-rl;dy=r-l;dc=dy-raw;co=((l>0)==yl)&((r>0)==yr);raw_rms=float(np.sqrt(np.mean(raw**2)))
  z=dict(n=len(valid),expected=len(pids),missing=len(pids)-len(valid),status='complete' if len(valid)==len(pids) else 'missing_scores',
   paired_correct=int(co.sum()),paired_accuracy=float(co.mean()),repair_rate=float(co.mean()),still_wrong=int(np.sum(~co)),
   raw_difference_rms=raw_rms,difference_rms=float(np.sqrt(np.mean(dy**2))),correction_rms=float(np.sqrt(np.mean(dc**2))),
   correction_abs_p95=float(np.quantile(np.abs(dc),.95)),correction_abs_max=float(np.max(np.abs(dc))),
   correction_rms_over_error_pool_raw_rms=float(np.sqrt(np.mean(dc**2))/raw_rms) if raw_rms else None,
   raw_difference_mean=float(np.mean(raw)),difference_mean=float(np.mean(dy)),
   raw_difference_sign_changed_or_zero=int(np.sum((raw!=0)&(np.sign(raw)*dy<=0))))
  if normalization:z['correction_rms_over_full_raw_AP_rms']=z['correction_rms']/normalization
  if relation=='changing':
   sign=yr-yl;before=sign*raw;after=sign*dy;den=float(np.mean(before))
   z.update(raw_signed_response_mean=den,signed_response_mean=float(np.mean(after)),
    signed_response_retention=float(np.mean(after)/den) if den else None,
    response_direction_degraded=int(np.sum((before>0)&(after<=0))),response_direction_improved=int(np.sum((before<=0)&(after>0))))
   if dataset=='MEGA_previous_controlled':z['mean_evidence_retention']=z['signed_response_retention']
   else:z['mean_signed_response_retention']=z['signed_response_retention']
  else:z['variance_reduction']=finite(1-np.var(dy)/np.var(raw)) if np.var(raw) else None
  return z
 groups={};flat=[];selected_rows=[]
 for group_key,prior in old['groups'].items():
  dataset=prior['dataset'];relation=prior['relation'];split=prior['split'];category=prior['category']
  if dataset=='MEGA_previous_controlled':
   kinds={'AP':('A','P'),'CF':('C','F')}.get(category,(category,))
   pids=sorted(p for p in fixed[dataset] if metadata[dataset,p]['kind'] in kinds)
   norm=old['datasets']['MEGA_previous_controlled']['full_raw_AP_rms']
  else:
   pids=sorted(p for p in fixed[dataset] if metadata[dataset,p]['dataset_split']==split and metadata[dataset,p]['relation']==relation and (category=='all' or metadata[dataset,p]['intervention']==category))
   norm=None
  group=dict(dataset=dataset,split=split,category=category,relation=relation,full_pair_count=prior['full_pair_count'],error_pair_count=len(pids),methods={},comparisons={})
  for method in methods:
   stat=point(dataset,pids,method,relation,norm);group['methods'][method]=stat
   flat.append(dict(group=group_key,dataset=dataset,split=split,category=category,relation=relation,method=method,error_pair_count=len(pids),**stat))
  for method in new_methods:
   z=group['methods'][method];comparison={}
   for reference in ['old_linear','optimized_balanced_fit_calibration']:
    r=group['methods'][reference]
    comparison[reference]={k:float(z[k]-r[k]) if z.get(k) is not None and r.get(k) is not None else None for k in ['paired_correct','paired_accuracy','difference_rms','correction_rms','variance_reduction','signed_response_retention','response_direction_degraded']}
   group['comparisons'][method]=comparison
  groups[group_key]=group
 for dataset,pids in fixed.items():
  for pid in sorted(pids):
   meta=metadata[dataset,pid]
   for method in methods:
    r=scores.get((dataset,method,pid))
    if not r:continue
    l=float(r['left_score']);rr=float(r['right_score'])
    selected_rows.append(dict(dataset=dataset,pair_id=pid,method=method,split=meta.get('dataset_split','previous_test'),
      category=meta.get('kind',meta.get('intervention')),relation=meta.get('relation'),source_id=meta.get('source_id'),
      left=meta['left'],right=meta['right'],left_label=int(meta['left_label']),right_label=int(meta['right_label']),
      left_score=l,right_score=rr,raw_left_score=float(scores[dataset,'raw',pid]['left_score']),raw_right_score=float(scores[dataset,'raw',pid]['right_score']),
      correction_difference=(rr-l)-(float(scores[dataset,'raw',pid]['right_score'])-float(scores[dataset,'raw',pid]['left_score'])),
      paired_correct=bool((l>0)==bool(meta['left_label']) and (rr>0)==bool(meta['right_label']))))
 result=dict(selection=old['selection'],fixed_pool_source=str(OLD/'selected_error_pair_scores.csv'),methods=methods,new_methods=new_methods,
  frozen_manifest=str(ROOT/'protected_frozen_methods.json'),datasets=old['datasets'],groups=groups,coverage=coverage,missing_endpoints=missing_endpoints,
  interpretation=dict(old['interpretation'],regression='No new originally-correct whole-population scores computed; selected-pair counterpart endpoints are necessary references only.'),
  scope='Fixed old_linear error pairs only; MEGA previous controlled 48, NLI native 2677. Not official7458 path-pair coverage.',
  new_training=False,new_parameter_selection=False,new_gpu_forwards=0,new_full_evaluation=False,seconds=time.time()-started)
 dump(out/'error_pair_metrics.json',result)
 for filename,rs in [('error_pair_summary.csv',flat),('error_pair_scores.csv',selected_rows)]:
  fields=list(dict.fromkeys(k for r in rs for k in r))
  with (out/filename).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rs)
 dump(out/'error_pair_done.json',dict(state='complete_with_missing' if missing_endpoints else 'complete',seconds=time.time()-started,
  selected_MEGA=len(fixed['MEGA_previous_controlled']),selected_NLI=len(fixed['NLI4CT-2024']),new_methods=new_methods,new_gpu_forwards=0))
 print(json.dumps(dict(coverage=coverage,groups=len(groups),new_methods=new_methods,seconds=time.time()-started)))
if __name__=='__main__':main()

