
"""CPU-only fidelity metrics from existing immutable paired scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,csv,json,time
from collections import defaultdict
import numpy as np

BASE=Path('/home/gcx/aster_next_20261109')
MEGA=Path('/home/gcx/aster_optimization_20261109/final/results_calibration_ablation/metrics.json')
NLI=BASE/'primary_results/nli4ct_all/pair_summary'
METHODS=['raw','linear_output','optimized_strict_fit_calibration','optimized_balanced_fit_calibration','trace_only_balanced_fit_calibration']
NLI_METHODS=['raw','old_linear']+METHODS[2:]
GROUPS=['A','P','AP','C','F','CF','F0']
def load(path):return json.loads(Path(path).read_text())
def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def ratio(x,y):return float(x/y) if abs(y)>1e-12 else None
def ci(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return np.quantile(x,[.025,.975]).tolist() if len(x) else None
def truth(x):return str(x).strip().lower() in ['true','1','yes']
def mega_metrics(path,pair_csv=None):
 src=load(path);result={};checks=[];rows=[]
 f0=defaultdict(list)
 pair_csv=pair_csv or path.with_name('test_pair_results.csv')
 if pair_csv.exists():
  with pair_csv.open(encoding='utf-8-sig',newline='') as handle:
   for row in csv.DictReader(handle):
    if row.get('kind')=='F0' and row.get('method') in METHODS:f0[row['method']].append(row)
 families=src.get('source_ids',[])
 family_index={name:j for j,name in enumerate(families)}
 bdraws=np.random.default_rng(11).multinomial(len(families),np.full(len(families),1/len(families)),size=2000) if families else None
 for name in METHODS:
  if name not in src['methods']:raise ValueError('MEGA method missing: '+name)
  source=src['methods'][name]['groups'];out={}
  for group in GROUPS:
   m=dict(source[group]);raw=src['methods']['raw']['groups'][group]
   mu0=float(raw['difference_mean']);mu=float(m['difference_mean'])
   var0=max(0.,float(raw['difference_rms'])**2-mu0**2)
   var=max(0.,float(m['difference_rms'])**2-mu**2)
   m.update(raw_difference_mean=mu0,raw_difference_variance=var0,difference_variance=var,
    difference_variance_ratio=ratio(var,var0),
    mean_and_variance_both_reduced=bool(abs(mu)<abs(mu0) and var<var0))
   if group in ('A','P','AP'):
    m['prediction_r2']=None if var0<=1e-12 else 1-float(m['difference_rms'])**2/var0
   if group=='F0':
    ff=f0.get(name,[])
    m['prediction_nonflip_coverage']=dict(available=len(ff),expected=m['n'],missing=m['n']-len(ff))
    if ff:
     agreement=np.array([(float(r['left_score'])>0)==(float(r['right_score'])>0) for r in ff],dtype=float)
     m['prediction_nonflip_rate']=float(agreement.mean())
     if bdraws is not None:
      agg=np.zeros((len(families),2))
      for j,row in enumerate(ff):agg[family_index[row['source_id']]]+=[1,agreement[j]]
      zz=bdraws@agg
      with np.errstate(divide='ignore',invalid='ignore'):rates=zz[:,1]/zz[:,0]
      m['prediction_nonflip_rate_ci']=ci(rates)
    else:m['prediction_nonflip_rate']=None
   if group in ('A','P','AP','F0'):
    recomputed=None if var0<=1e-12 else 1-var/var0
    if recomputed is not None and abs(recomputed-m['variance_reduction'])>1e-7:raise ValueError('MEGA variance/RMS mismatch '+name+'/'+group)
   out[group]=m
   rows.append(dict(dataset='MEGA-CDP',scope='previously_used_held_out_test',group=group,method=name,
    n=m['n'],expected=m.get('expected'),missing=m.get('missing'),sources=m.get('sources'),
    variance_reduction=m.get('variance_reduction'),prediction_r2=m.get('prediction_r2'),prediction_nonflip_rate=m.get('prediction_nonflip_rate'),difference_rms=m['difference_rms'],
    raw_difference_rms=m['raw_difference_rms'],mean_evidence_retention=m.get('mean_evidence_retention'),
    correction_rms=m['correction_rms'],correction_rms_over_ap_rms=m['correction_rms_over_ap_rms'],
    direction_degraded=m.get('response_direction_degraded'),paired_accuracy=m['paired_accuracy'],
    mean_and_variance_both_reduced=m['mean_and_variance_both_reduced']))
  if out['A']['n']+out['P']['n']!=out['AP']['n'] or out['C']['n']+out['F']['n']!=out['CF']['n']:raise ValueError('MEGA pair group coverage mismatch')
  result[name]={'groups':out}
 return dict(status='complete',dataset='MEGA-CDP',input=str(path),
  sample_identity='Existing train_val source-held-out subset previously used for legacy and optimization evaluations; not the newly expanded official test',
  test_views=src.get('test_views'),test_sources=src.get('test_sources'),path_cases=src.get('test_path_cases'),
  uncertainty=src.get('uncertainty'),methods=result,
  coverage={g:{k:src['methods']['raw']['groups'][g].get(k) for k in ['n','expected','missing','sources']} for g in GROUPS},
  numeric_checks=dict(group_count_sums=True,centered_variance_matches_existing_VR=True)),rows

def nli_metrics(folder,iterations=2000,seed=11):
 mp=folder/'metrics.json';cp=folder/'pair_results.csv'
 if not mp.exists() or not cp.exists():
  return dict(status='pending',dataset='NLI4CT-2024',missing_inputs=[str(p) for p in [mp,cp] if not p.exists()],
   scope='Native robustness/response under frozen MEGA-to-NLI zero-shot transfer; not MEGA A/P/C/F'),[]
 native=load(mp)
 with cp.open(encoding='utf-8-sig',newline='') as handle:
  rawrows=list(csv.DictReader(handle))
 records=[];splitfamilies=defaultdict(set)
 for r in rawrows:
  if r['method'] not in NLI_METHODS:continue
  x=float(r['raw_difference']);y=float(r['difference']);dc=float(r['correction_difference'])
  if not np.all(np.isfinite([x,y,dc])):continue
  if abs((y-x)-dc)>1e-7:raise ValueError('NLI CSV correction difference mismatch')
  label0=int(r['left_label']);label1=int(r['right_label'])
  relation='preserving' if label0==label1 else 'changing'
  if relation!=r['relation']:raise ValueError('NLI relation does not match released pair labels')
  record=dict(r,x=x,y=y,dc=dc,direction=label1-label0,paired=truth(r['paired_correct']))
  records.append(record);splitfamilies[r['split']].add(r['family'])
 draws={};family_index={}
 for split,fs in splitfamilies.items():
  fs=sorted(fs);family_index[split]={f:i for i,f in enumerate(fs)}
  draws[split]=np.random.default_rng(seed).multinomial(len(fs),np.full(len(fs),1/len(fs)),size=iterations).astype('float64')
 grouped=defaultdict(list)
 for r in records:
  for intervention in ['all',r['intervention']]:
   grouped[(r['split'],intervention,r['relation'],r['method'])].append(r)
 outgroups={};flat=[];warnings=[]
 for (split,intervention,relation,method),rr in grouped.items():
  key=split+':'+intervention+':'+relation
  expected=native.get('groups',{}).get(key,{}).get('expected_pairs',len(rr))
  group=outgroups.setdefault(key,dict(split=split,intervention=intervention,relation=relation,expected_pairs=expected,methods={}))
  x=np.array([r['x'] for r in rr]);y=np.array([r['y'] for r in rr]);dc=y-x
  direction=np.array([r['direction'] for r in rr]);correct=np.array([r['paired'] for r in rr],dtype=float)
  sx=direction*x;sy=direction*y;degrade=(sx>0)&(sy<=0);improve=(sx<=0)&(sy>0)
  mu0=float(x.mean());mu=float(y.mean());v0=float(np.var(x));v=float(np.var(y))
  rms0=float(np.sqrt(np.mean(x*x)));rms=float(np.sqrt(np.mean(y*y)));crms=float(np.sqrt(np.mean(dc*dc)))
  m=dict(n=len(rr),expected=expected,missing=expected-len(rr),families=len({r['family'] for r in rr}),
   raw_difference_mean=mu0,difference_mean=mu,raw_difference_variance=v0,difference_variance=v,
   variance_reduction=None if v0<=1e-12 else 1-v/v0,raw_difference_rms=rms0,difference_rms=rms,
   difference_rms_ratio=ratio(rms,rms0),correction_rms=crms,correction_rms_over_raw_difference_rms=ratio(crms,rms0),
   paired_correct=int(correct.sum()),paired_accuracy=float(correct.mean()),
   mean_and_variance_both_reduced=bool(abs(mu)<abs(mu0) and v<v0),
   raw_difference_sign_changed_or_zero_count=int(np.sum((x!=0)&(np.sign(x)*y<=0))))
  if relation=='changing':
   m.update(raw_signed_response_mean=float(sx.mean()),signed_response_mean=float(sy.mean()),
    mean_signed_response_retention=ratio(float(sy.mean()),float(sx.mean())),
    raw_signed_response_mean_nonpositive=bool(sx.mean()<=0),raw_gold_direction_positive=int(np.sum(sx>0)),gold_direction_positive=int(np.sum(sy>0)),gold_direction_degraded=int(degrade.sum()),
    gold_direction_improved=int(improve.sum()),gold_direction_degraded_fraction=float(degrade.mean()),
    signed_response_and_variance_both_reduced=bool(sx.mean()>0 and sy.mean()<sx.mean() and v<v0))
  fi=np.array([family_index[split][r['family']] for r in rr])
  values=np.column_stack([np.ones(len(rr)),x,x*x,y,y*y,dc*dc,correct,sx,sy,degrade.astype(float)])
  agg=np.zeros((len(family_index[split]),values.shape[1]));np.add.at(agg,fi,values)
  z=draws[split]@agg
  with np.errstate(divide='ignore',invalid='ignore'):
   mx=z[:,1]/z[:,0];my=z[:,3]/z[:,0]
   vx=z[:,2]/z[:,0]-mx*mx;vy=z[:,4]/z[:,0]-my*my
   vr=1-vy/vx;vr[vx<=1e-12]=np.nan
   reps=dict(variance_reduction=vr,raw_difference_mean=mx,difference_mean=my,
    raw_difference_variance=vx,difference_variance=vy,
    difference_rms=np.sqrt(np.maximum(z[:,4]/z[:,0],0)),
    correction_rms=np.sqrt(np.maximum(z[:,5]/z[:,0],0)),
    paired_accuracy=z[:,6]/z[:,0],
    correction_rms_over_raw_difference_rms=np.sqrt(np.maximum(z[:,5]/z[:,2],0)))
   if relation=='changing':
    retention=z[:,8]/z[:,7];retention[np.abs(z[:,7])<=1e-12]=np.nan
    reps.update(mean_signed_response_retention=retention,gold_direction_degraded_fraction=z[:,9]/z[:,0])
  for metric,rep in reps.items():m[metric+'_ci']=ci(rep)
  m['valid_bootstrap_replicates']={metric:int(np.isfinite(rep).sum()) for metric,rep in reps.items()}
  group['methods'][method]=m
  flat.append(dict(dataset='NLI4CT-2024',scope=split,group=key,method=method,**{k:v for k,v in m.items() if isinstance(v,(int,float,bool,type(None)))}))
  if m['mean_and_variance_both_reduced'] or m.get('signed_response_and_variance_both_reduced'):
   warnings.append(dict(group=key,method=method,absolute_difference_mean_reduced=abs(mu)<abs(mu0),
    difference_variance_reduced=v<v0,signed_response_mean_reduced=m.get('signed_response_and_variance_both_reduced',False),
    interpretation='Concurrent shrinkage is an observed statistic, not proof of selective nuisance removal; inspect changing-label response and paired correctness.'))
 checks=dict(csv_rows=len(rawrows),selected_rows=len(records),label_relations_match=True,score_differences_match=True)
 for key,g in outgroups.items():
  base=g['methods'].get('raw')
  if base and base['variance_reduction'] is not None and abs(base['variance_reduction'])>1e-10:raise ValueError('Raw native VR must be zero')
 return dict(status='complete',dataset='NLI4CT-2024',inputs=[str(mp),str(cp)],
  scope='Native robustness/response under frozen MEGA-to-NLI zero-shot transfer; these groups are not MEGA A/P/C/F',
  case_subsets=native.get('case_subsets'),coverage=native.get('coverage'),native_label_inventory=native.get('native_label_inventory'),
  uncertainty=dict(replicates=iterations,seed=seed,grouping='native base family within each original split, using families present in the pair CSV',
   qualification='Variants share their base; different families may share trial material. These are not independent patients.'),
  groups=outgroups,concurrent_shrinkage_observations=warnings,numeric_checks=checks),flat


def official_mega_metrics(path):
 plan_path=BASE/'faithfulness_pipeline_plan.json'
 plan=load(plan_path) if plan_path.exists() else {}
 inventory_path=Path(plan.get('inventory',str(BASE/'fidelity/mega_official/inventory.json')))
 inventory=load(inventory_path) if inventory_path.exists() else {}
 selected=inventory.get('selected',{})
 queue_status_path=BASE/'faithfulness_pipeline_status.json'
 queue_status=load(queue_status_path) if queue_status_path.exists() else {}
 cancelled=queue_status.get('phase') in ('cancelled_before_start','cancelled','stopped_for_error_retesting')
 planned=dict(plan=str(plan_path),queue_state=queue_status.get('phase','queued_or_running_until_summary'),selected=selected,
  gpus=plan.get('gpus'),expected_views=plan.get('expected_views'),
  expected_pairs=plan.get('expected_pairs'),new_forward_upper_bound=plan.get('new_forward_upper_bound'))
 if not path.exists():
  return dict(status='cancelled' if cancelled else 'scheduled',input=str(path),dataset='MEGA-CDP',dataset_split='official_test',
   scope='Planned first gold decision node of each of 7458 official reference paths; controlled node A/P evaluation only',planned_collection=planned,
   planned_original_endpoints=7458,planned_groups=['A','P','AP'],missing_evidence_groups=['C','F','CF','F0'],
   reason=('User stopped expanded paired testing before start; official C/F/F0 remain unmeasured.' if cancelled else 'Official first-node A/P forward collection is queued; no new metric is filled before results exist. C/F/F0 pairs are not prepared.')),[]
 src=load(path);groups={};flat=[];aliases={'old_linear':'linear_output'}
 for key,item in src.get('groups',{}).items():
  target=dict(expected_pairs=item.get('expected_pairs'),unavailable_reason=item.get('unavailable_reason'),methods={})
  for name,value in item.get('methods',{}).items():
   canonical=aliases.get(name,name)
   if canonical not in METHODS:continue
   m=dict(value)
   if 'paired_prediction_r2' in m:m['prediction_r2']=m['paired_prediction_r2']
   if 'paired_prediction_r2_ci' in m:m['prediction_r2_ci']=m['paired_prediction_r2_ci']
   if 'prediction_unchanged_fraction' in m:m['prediction_nonflip_rate']=m['prediction_unchanged_fraction']
   target['methods'][canonical]=m
   flat.append(dict(dataset='MEGA-CDP',scope='official_test_first_gold_node',group=key,method=canonical,**{k:v for k,v in m.items() if isinstance(v,(int,float,bool,type(None)))}))
  groups[key]=target
 missing=[g for g in ['C','F','CF','F0'] if not groups.get(g,{}).get('expected_pairs')]
 return dict(status='available_partial_scope',input=str(path),dataset='MEGA-CDP',dataset_split='official_test',
  scope=src.get('scope'),planned_collection=planned,planned_original_endpoints=7458,groups=groups,coverage=src.get('coverage'),
  uncertainty=src.get('uncertainty'),missing_evidence_groups=missing,
  interpretation='Measured A/P first-node behavior does not complete full-path counterfactual evaluation or C/F evidence-response protection.'),flat


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mega-metrics',type=Path,default=MEGA)
 p.add_argument('--mega-pairs',type=Path)
 p.add_argument('--nli-pair-dir',type=Path,default=NLI)
 p.add_argument('--official-mega-metrics',type=Path,default=BASE/'primary_results/mega_official_pairs/summary/metrics.json')
 p.add_argument('--out',type=Path,default=BASE/'faithfulness')
 a=p.parse_args();start=time.time();a.out.mkdir(parents=True,exist_ok=True)
 mega,mrows=mega_metrics(a.mega_metrics,a.mega_pairs);nli,nrows=nli_metrics(a.nli_pair_dir)
 official,orows=official_mega_metrics(a.official_mega_metrics)
 result=dict(definitions=dict(VR='1 - Var(corrected pair difference) / Var(raw pair difference); centered variance',
  R2='1 - mean(corrected pair difference squared) / Var(raw pair difference); distinct from centered VR',
  F0_nonflip='Fraction of F0 pairs with equal left/right prediction signs; distinct from both-correct',
  ER='Mean truth-direction response after correction / before correction; undefined at zero denominator',
  correction='Corrected pair difference minus raw pair difference',
  MEGA_interference_unit='Correction RMS divided by raw MEGA A/P RMS',
  scope='Behavioral pair dependence and evidence response; no assertion about internal reasoning or chain-of-thought faithfulness'),
  methods=METHODS,mega=mega,nli=nli,official_mega_pair_evaluation=official,
  no_training=True,no_new_backbone_forward=True,no_selection=True,seconds=time.time()-start)
 dump(a.out/'metrics.json',result)
 rows=mrows+nrows+orows;fields=list(dict.fromkeys(k for r in rows for k in r))
 with (a.out/'summary.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 print(json.dumps(dict(state='complete',MEGA_methods=len(mega['methods']),MEGA_groups=len(mega['coverage']),NLI_status=nli['status'],NLI_groups=len(nli.get('groups',{})),out=str(a.out),seconds=time.time()-start)))
if __name__=='__main__':main()

