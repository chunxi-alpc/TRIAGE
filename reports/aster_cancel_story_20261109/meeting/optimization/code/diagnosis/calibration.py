
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,numpy as np
from scipy.optimize import brentq,minimize
from scipy.special import expit
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT=Path('/home/gcx/aster_optimization_20261109/diagnosis')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
a=np.load(OLD/'coordinates.npz');ix={s:i for i,s in enumerate(a['ids'].tolist())};meta=json.loads((OLD/'metadata.json').read_text())
cases=rows(ROOT/'data/path_cases.jsonl')
data={}
for sp in ['fit','dev']:
 rec={}
 for p in (ROOT/'qwen/paths').glob('features_'+sp+'_*.jsonl'):
  for r in rows(p):
   if r.get('status')=='ok':rec[r['id']]=r
 cs=[c for c in cases if c['split']==sp] if rec else []
 lower=[];upper=[];missing=[]
 for c in cs:
  vals=[]
  for n in c['gold_path']:
   key=c['case_id']+':'+n['gid']
   if key not in rec:missing.append(key);continue
   vals.append((float(rec[key]['ell']),int(n['label'])))
  lower.append(max([-e for e,y in vals if y==1],default=-np.inf))
  upper.append(min([-e for e,y in vals if y==0],default=np.inf))
 vv=[v for v in meta['views'] if v['split']==sp and v['variant']=='full' and v['original']]
 ii=np.array([ix[v['view_id']] for v in vv]);e=a['ell'][ii];y=np.array([v['label'] for v in vv])
 data[sp]=dict(cases=cs,lower=np.array(lower),upper=np.array(upper),e=e,y=y,missing=missing,records=len(rec))
def count(d,b):return int(np.sum((b>d['lower'])&(b<=d['upper'])))
def nc(d,b):return int(np.sum((d['e']+b>0)==d['y']))
def metrics(b):
 return {sp:dict(path_correct=count(d,b),path_n=len(d['cases']),natural_correct=nc(d,b),natural_n=len(d['e'])) for sp,d in data.items()}
def candidates(bounds):
 bounds=np.unique(bounds);bounds=bounds[np.isfinite(bounds)]
 return np.unique(np.concatenate([[0.],bounds,(bounds[1:]+bounds[:-1])/2,[bounds[0]-1,bounds[-1]+1]]))
f=data['fit'];d=data['dev']
grid=candidates(np.concatenate([d['lower'],d['upper']]))
devbest=min(grid,key=lambda b:(-count(d,b),-nc(d,b),-nc(f,b),abs(b)))
fitgrid=candidates(-f['e'])
fitbest=min(fitgrid,key=lambda b:(-nc(f,b),abs(b)))
def ce(x):
 scale=np.exp(x[0]);z=scale*f['e']+x[1]
 return float(np.mean(np.logaddexp(0,z)-f['y']*z)),np.array([np.mean((expit(z)-f['y'])*f['e'])*scale,np.mean(expit(z)-f['y'])])
r=minimize(ce,[0.,0.],jac=True,method='BFGS')
pos_temp=float(np.exp(r.x[0]));intercept=float(r.x[1]);equiv_b=intercept/pos_temp
biasce=float(brentq(lambda b:np.mean(expit(f['e']+b)-f['y']),-100,100))
variants={'raw':0.,'fit_natural_CE':biasce,'fit_logistic_positive_scale':equiv_b,'fit_natural_accuracy':float(fitbest),'dev_path_optimal':float(devbest)}
result={'protocol':'fit/dev only; all test rows excluded from this diagnostic',
 'positive_scale':pos_temp,'logistic_intercept':intercept,
 'monotonic_fact':'Any increasing scalar calibration with a single zero crossing is a threshold change; positive temperature cannot add classification capacity beyond a bias',
 'variants':{name:dict(b=b,**metrics(b)) for name,b in variants.items()},
 'coverage':{sp:dict(cases=len(dd['cases']),node_records=dd['records'],missing_gold_nodes=len(dd['missing']),
  threshold_impossible_cases=int(np.sum(dd['lower']>=dd['upper']))) for sp,dd in data.items()}}
(OUT/'calibration.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=True))

