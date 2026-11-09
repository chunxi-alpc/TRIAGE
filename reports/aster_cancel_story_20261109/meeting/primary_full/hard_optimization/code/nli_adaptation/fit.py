
"""Six predeclared CPU-only NLI train/dev supervised residual fits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,time,datetime,sys,shutil
from collections import Counter
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
BASE=Path('/home/gcx/aster_next_20261109')
OUT=Path('/home/gcx/aster_hard_optimization_20261109/nli_adaptation')
LAMBDAS=(0.001,0.01,0.1)
FAMILIES=(('trace16',16),('trace_signal_residual96',96))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def evaluate(score,label):
 signed=(2*label-1)*score
 return dict(n=len(label),correct=int(np.sum((score>0)==label)),accuracy=float(np.mean((score>0)==label)),
  binary_cross_entropy=float(np.mean(np.logaddexp(0,score)-label*score)),mean_signed_margin=float(np.mean(signed)))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 plan=dict(created_at=now(),method_name='NLI training-set supervised residual adaptation',
  base_protocol='MEGA-fitted features remain frozen; this is task-supervised readout adaptation and is not the original MEGA nuisance-constrained method or its zero-shot setting.',
  train_manifest=str(BASE/'inventory/nli4ct_train_cases.jsonl'),dev_manifest=str(BASE/'inventory/nli4ct_dev_cases.jsonl'),
  train_records=1700,dev_records=200,feature_families=[dict(name=n,dimension=d) for n,d in FAMILIES],
  raw_coefficient=1.,lambdas=list(LAMBDAS),fits=6,
  feature_standardization='mean/std on the1700 NLI train records only; coordinates from the unchanged old FrozenMethods',
  objective='mean BCE(ell+z@w+b,y) + lambda/2*(sum(w^2)+mean((z@w)^2)); b unpenalized; raw coefficient fixed1',
  initialization='w=0,b=0',optimizer=dict(name='L-BFGS-B',maxiter=800,ftol=1e-12,gtol=1e-8),
  selection='Highest dev classification accuracy, then lowest dev BCE, then fewer coefficients, then larger regularizer. Select exactly one saved model; no threshold sweep.',
  historical_official_scores_previously_viewed=True,official_or_practice_labels_used_in_this_fit_or_selection=False,
  no_pseudo_AP_CF=True,no_original_fit_budget_guarantee=True,
  raw_coefficient_note='Fixing the raw coefficient does not guarantee behavior preservation; a residual can still weaken evidence response. Pair/error diagnostics must be measured after freeze.',
  compute=dict(device='CPU',gpu_memory_gib=0,estimated_seconds=60))
 dump(OUT/'plan.json',plan)
 started=time.time()
 train=rows(BASE/'inventory/nli4ct_train_cases.jsonl');dev=rows(BASE/'inventory/nli4ct_dev_cases.jsonl')
 if len(train)!=1700 or len(dev)!=200:raise ValueError('Unexpected native split counts')
 if any(c['dataset_split']!='train' for c in train) or any(c['dataset_split']!='dev' for c in dev):raise ValueError('Unexpected split labels')
 cases=train+dev;allowed={c['case_id'] for c in cases};node={}
 for path in sorted((BASE/'primary_results/nli4ct_all').glob('shard*/nodes.jsonl')):
  for r in rows(path):
   if r.get('case_id') in allowed and r.get('gid','G0')=='G0' and r.get('status')=='ok':node[r['case_id']]=r
 missing=[c['case_id'] for c in cases if c['case_id'] not in node or not Path(node[c['case_id']]['feature']).is_file()]
 if missing:raise RuntimeError('Missing cached train/dev nodes: '+str(missing[:5]))
 sys.path.insert(0,str(BASE/'inference'));from frozen import FrozenMethods
 adapter=FrozenMethods(BASE/'frozen_methods.json')
 features={n:[] for n,d in FAMILIES};ell=[]
 for case in cases:
  record=node[case['case_id']]
  with np.load(record['feature']) as a:
   selected=adapter.raw_selected(a);residual=a['residual'].astype('float32')
  c=adapter.coordinates(record['ell'],selected,residual)
  features['trace16'].append(c['r16'])
  features['trace_signal_residual96'].append(np.concatenate([c['r32'],c['residual64']]))
  ell.append(float(record['ell']))
 features={k:np.stack(v).astype('float64') for k,v in features.items()}
 ell=np.array(ell);label=np.array([c['label'] for c in cases],dtype='float64')
 train_mask=np.arange(len(cases))<len(train);dev_mask=~train_mask
 data_time=time.time()-started
 predictions=dict(ids=np.array([c['case_id'] for c in cases]),split=np.array([c['dataset_split'] for c in cases]),labels=label.astype('int8'),ell=ell,raw=ell)
 records=[]
 raw_metrics=dict(train=evaluate(ell[train_mask],label[train_mask]),dev=evaluate(ell[dev_mask],label[dev_mask]))
 for family,dimension in FAMILIES:
  x=features[family];mu=x[train_mask].mean(0);scale=x[train_mask].std(0);scale=np.where(scale<1e-6,1.,scale)
  z=(x-mu)/scale;zt=z[train_mask];et=ell[train_mask];yt=label[train_mask]
  covariance=zt.T@zt/len(yt)
  for reg in LAMBDAS:
   fit_start=time.time();name='nli_residual_'+family+'_l'+str(reg).replace('.','p')
   def objective(parameters):
    w=parameters[:-1];b=parameters[-1];correction=zt@w;score=et+correction+b
    delta=expit(score)-yt
    ce=np.mean(np.logaddexp(0,score)-yt*score)
    loss=ce+.5*reg*(w@w+np.mean(correction**2))
    gradient=np.concatenate([zt.T@delta/len(yt)+reg*(w+covariance@w),[np.mean(delta)]])
    return float(loss),gradient
   result=minimize(objective,np.zeros(dimension+1),jac=True,method='L-BFGS-B',
    options=dict(maxiter=800,ftol=1e-12,gtol=1e-8))
   w=result.x[:-1];bias=float(result.x[-1]);score=ell+z@w+bias
   train_metric=evaluate(score[train_mask],label[train_mask]);dev_metric=evaluate(score[dev_mask],label[dev_mask])
   parameter_path=OUT/(name+'.npz')
   np.savez_compressed(parameter_path,w=w,b=np.array(bias),mu=mu,scale=scale,family=np.array(family),
    raw_coefficient=np.array(1.),regularizer=np.array(reg))
   predictions[name]=score
   record=dict(name=name,family=family,dimension=dimension,lambda_=reg,weights=str(parameter_path),
    optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),
    objective=float(result.fun),b=bias,train=train_metric,dev=dev_metric,
    train_correction_rms=float(np.sqrt(np.mean((zt@w)**2))),dev_correction_rms=float(np.sqrt(np.mean((z[dev_mask]@w)**2))),
    seconds=time.time()-fit_start)
   dump(OUT/(name+'.json'),record);records.append(record)
   dump(OUT/'status.json',dict(state='fitting',completed=len(records),total=6,last_name=name))
 # Do not use any field besides the predeclared dev criteria for selection.
 selected=sorted(records,key=lambda r:(-r['dev']['accuracy'],r['dev']['binary_cross_entropy'],r['dimension'],-r['lambda_']))[0]
 shutil.copy2(selected['weights'],OUT/'selected_model.npz')
 np.savez_compressed(OUT/'fit_dev_scores.npz',**predictions)
 summary=dict(protocol=plan['method_name'],plan=str(OUT/'plan.json'),raw=raw_metrics,models=records,
  selected_name=selected['name'],data_preparation_seconds=data_time,seconds=time.time()-started)
 dump(OUT/'fit_dev_summary.json',summary)
 frozen=dict(frozen=True,saved_at=now(),method_name='NLI training-set supervised residual adaptation',
  selection_split='dev',selected_name=selected['name'],family=selected['family'],dimension=selected['dimension'],
  lambda_=selected['lambda_'],raw_coefficient=1.,b=selected['b'],model=str(OUT/'selected_model.npz'),
  original_model=selected['weights'],plan=str(OUT/'plan.json'),
  training=dict(split='NLI train',n=1700,supervision='native binary labels',objective=plan['objective']),
  selection_criteria=plan['selection'],raw_metrics=raw_metrics,selected_fit_dev_metrics=dict(train=selected['train'],dev=selected['dev']),
  official_practice_used_for_fit_or_selection=False,test_or_error_pool_evaluation_started=False,
  scope='Different supervised task adaptation; no original MEGA per-fit-C/F hard-bound guarantee; alpha_raw1 is not behavior protection.')
 dump(OUT/'selected.json',frozen)
 dump(OUT/'status.json',dict(state='fit_dev_frozen',fits=len(records),selected=selected['name'],seconds=time.time()-started))
 print(json.dumps(dict(raw=raw_metrics,models=[dict(name=r['name'],train_correct=r['train']['correct'],dev_correct=r['dev']['correct'],dev_CE=r['dev']['binary_cross_entropy']) for r in records],selected=selected['name'],weights=str(OUT/'selected_model.npz'),seconds=time.time()-started)))
if __name__=='__main__':main()

