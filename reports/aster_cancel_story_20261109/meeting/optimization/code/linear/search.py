
import os
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
import json,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from scipy.linalg import cho_factor,cho_solve
from sklearn.utils.extmath import randomized_svd

OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT=Path('/home/gcx/aster_optimization_20261109/linear')
OUT.mkdir(parents=True,exist_ok=True);(OUT/'candidates').mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rms(x):return float(np.sqrt(np.mean(x*x))) if len(x) else None
def main():
 start=time.time()
 a=np.load(OLD/'coordinates.npz');ids=a['ids'].tolist();ell=a['ell'];ix={v:i for i,v in enumerate(ids)}
 np.savez_compressed(OUT/'ids.npz',ids=np.array(ids))
 meta=json.loads((OLD/'metadata.json').read_text());views=meta['views'];ps=meta['pairs']
 fitap=[p for p in ps if p['split']=='fit' and p['kind'] in ('A','P')]
 fitcf=[p for p in ps if p['split']=='fit' and p['kind'] in ('C','F')]
 al=np.array([ix[p['left']] for p in fitap]);ar=np.array([ix[p['right']] for p in fitap])
 cl=np.array([ix[p['left']] for p in fitcf]);cr=np.array([ix[p['right']] for p in fitcf])
 y=ell[ar]-ell[al]
 fiv=[v for v in views if v['split']=='fit'];fidx=np.array([ix[v['view_id']] for v in fiv])
 fv=[v for v in views if v['split']=='fit' and v['variant']=='full' and v['original']]
 fi=np.array([ix[v['view_id']] for v in fv]);fy=np.array([v['label'] for v in fv])
 pairsets={}
 for sp in ('fit','dev'):
  for group,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
   pp=[p for p in ps if p['split']==sp and p['kind'] in kinds]
   pairsets[sp,group]=(np.array([ix[p['left']] for p in pp]),np.array([ix[p['right']] for p in pp]),np.array([p['left_label'] for p in pp]),np.array([p['right_label'] for p in pp]))
 natsets={}
 for sp in ('fit','dev'):
  vv=[v for v in views if v['split']==sp and v['variant']=='full' and v['original']]
  natsets[sp]=np.array([ix[v['view_id']] for v in vv]),np.array([v['label'] for v in vv])
 apscale={sp:rms(ell[pairsets[sp,'AP'][1]]-ell[pairsets[sp,'AP'][0]]) for sp in ('fit','dev')}
 def measure(delta,b):
  ss=ell+delta+b;out={}
  for sp in ('fit','dev'):
   g={}
   for group in ('A','P','AP','C','F','CF','F0'):
    li,ri,yl,yr=pairsets[sp,group];raw=ell[ri]-ell[li];dc=delta[ri]-delta[li];dy=raw+dc
    pcor=((ss[li]>0)==yl)&((ss[ri]>0)==yr)
    v=dict(n=len(li),difference_rms=rms(dy),correction_rms=rms(dc),correction_rms_over_ap_rms=rms(dc)/apscale[sp],
      paired_accuracy=float(np.mean(pcor)),paired_correct=int(pcor.sum()))
    if group in ('A','P','AP','F0'):
     v['variance_reduction']=1-float(np.var(dy)/np.var(raw))
    else:
     signed=(yr-yl)*raw;new=(yr-yl)*dy
     v['mean_evidence_retention']=float(np.mean(new)/np.mean(signed))
     v['response_direction_degraded']=int(np.sum((signed>0)&(new<=0)))
    g[group]=v
   ni,ny=natsets[sp];co=(ss[ni]>0)==ny
   g['natural']=dict(n=len(ni),correct=int(co.sum()),accuracy=float(np.mean(co)))
   out[sp]=g
  return out
 def calibrate(delta):
  base=ell[fi]+delta[fi]
  f=lambda b:float(np.mean(expit(base+b)-fy))
  lo,hi=-1.,1.
  while f(lo)>0:lo*=2
  while f(hi)<0:hi*=2
  return float(brentq(f,lo,hi))
 # Bases derive exclusively from fit features and fit pairs.
 support=dict(np.load('/home/gcx/aster_cr_repair_v2_run_20261108/qwen/supports_r16.npz'))
 raw=np.load(OLD/'raw_selected.npy',mmap_mode='r')
 dx=((raw[ar]-raw[al])/support['scale']).astype(np.float32)
 ed=((raw[cr]-raw[cl])/support['scale']).astype(np.float32)
 dump(OUT/'status.json',dict(stage='fit_only_bases',seconds=time.time()-start))
 V=randomized_svd(dx,n_components=256,random_state=11,n_iter=3)[2].T
 # Remove all observed fit evidence directions, rather than only the first sixteen.
 Es=np.linalg.svd(ed.astype('float64'),full_matrices=False)[2].T
 dx-=((dx@Es)@Es.T).astype('float32')
 Ve=randomized_svd(dx,n_components=128,random_state=11,n_iter=3)[2].T
 families={'r8':a['r8'],'r16':a['r16'],'r32':a['r32'],'U':a['U']}
 featfull=np.empty((len(ids),256),dtype='float32');feate=np.empty((len(ids),128),dtype='float32')
 for lo in range(0,len(ids),512):
  zz=(raw[lo:lo+512]-support['mu'])/support['scale']
  featfull[lo:lo+512]=zz@V;feate[lo:lo+512]=zz@Ve
 for dim in (64,128,256):families['AP'+str(dim)]=featfull[:,:dim]
 for dim in (64,128):families['evidence_null'+str(dim)]=feate[:,:dim]
 t32=a['r32'];mu32=t32[fidx].mean(0);sd32=np.maximum(t32[fidx].std(0),1e-5)
 t32=(t32-mu32)/sd32
 rng=np.random.default_rng(11);W=rng.normal(size=(32,256));phase=rng.uniform(0,2*np.pi,256)
 for sigma in (1.,2.):families['RFF256_s'+str(sigma)]=np.sqrt(2/256)*np.cos(t32@W/sigma+phase)
 np.savez_compressed(OUT/'feature_bases.npz',AP=V,evidence_null=Ve,evidence_directions=Es,rff_W=W,rff_phase=phase,rff_input_mean=mu32,rff_input_scale=sd32)
 np.savez_compressed(OUT/'feature_coordinates.npz',**families)
 dump(OUT/'status.json',dict(stage='candidate_fits',families=len(families),seconds=time.time()-start))
 counts=Counter(p['source_id'] for p in fitap)
 sourceweight=np.array([1/counts[p['source_id']] for p in fitap]);sourceweight/=sourceweight.mean()
 # Mild P upweighting remains below equal-type weighting; both A/P counts saved.
 typeweight=np.array([10 if p['kind']=='P' else 1 for p in fitap]);typeweight=typeweight/typeweight.mean()
 weightsets={'plain':np.ones(len(y)),'source':sourceweight,'P10':typeweight}
 results=[];number=0
 for fam,zz in families.items():
  mu=zz[fidx].mean(0);sd=np.maximum(zz[fidx].std(0),1e-6);Z=((zz-mu)/sd).astype('float64')
  X=Z[ar]-Z[al];E=Z[cr]-Z[cl];E2=E.T@E/len(E)
  dim=Z.shape[1]
  # Source weighting on high-dimensional families and P weighting on r16/r32 only.
  balances=('plain','source') if fam not in ('r16','r32') else ('plain','source','P10')
  for balance in balances:
   sw=weightsets[balance];Q=X.T@(sw[:,None]*X)/len(X);q=X.T@(sw*y)/len(X)
   for alpha in (.001,.1,10.):
    for beta in (0.,1.,10.,100.,1000.):
     M=Q+beta*E2+alpha*np.eye(dim)
     try:w=cho_solve(cho_factor(M,lower=True,check_finite=False),-q,check_finite=False)
     except Exception:w=np.linalg.lstsq(M,-q,rcond=1e-10)[0]
     delta=Z@w;b=calibrate(delta)
     name='lin_'+str(number).zfill(4);number+=1
     metrics=measure(delta,b)
     cfg=dict(name=name,family=fam,dimension=dim,balance=balance,alpha=alpha,evidence_weight=beta,b=b,
      objective='mean(weight_AP*(raw_AP+delta_AP)^2)+beta*mean(delta_CF^2)+alpha*sum(w^2)',
      score_file=str(OUT/'candidates'/(name+'.npz')),model_file=str(OUT/'candidates'/(name+'_model.npz')))
     np.savez_compressed(OUT/'candidates'/(name+'.npz'),delta=delta.astype('float32'),b=b)
     np.savez_compressed(OUT/'candidates'/(name+'_model.npz'),w=w,b=b,mu=mu,scale=sd)
     results.append(dict(config=cfg,metrics=metrics))
     if number%20==0:
      dump(OUT/'candidates.json',results)
      dump(OUT/'status.json',dict(stage='candidate_fits',completed=number,last=fam,seconds=time.time()-start))
  del Z
 dump(OUT/'candidates.json',results)
 # Matched correction tiers; natural dev not lower than raw. Main natural old is 196/199.
 rawdev=float(np.mean((ell[natsets['dev'][0]]>0)==natsets['dev'][1]))
 tiers={}
 for tier in (.05,.1,.2,.4):
  eligible=[r for r in results if r['metrics']['dev']['CF']['correction_rms_over_ap_rms']<=tier and r['metrics']['dev']['CF']['mean_evidence_retention']>=.98 and r['metrics']['dev']['natural']['accuracy']>=rawdev]
  best=sorted(eligible,key=lambda r:r['metrics']['dev']['AP']['variance_reduction'],reverse=True)[:10]
  tiers[str(tier)]=[r['config']['name'] for r in best]
 # Pareto for dev AP VR and CF correction, requiring ER and natural correctness.
 pool=[r for r in results if r['metrics']['dev']['CF']['mean_evidence_retention']>=.98 and r['metrics']['dev']['natural']['accuracy']>=rawdev]
 frontier=[];best=-float('inf')
 for r in sorted(pool,key=lambda r:r['metrics']['dev']['CF']['correction_rms']):
  vr=r['metrics']['dev']['AP']['variance_reduction']
  if vr>best:frontier.append(r['config']['name']);best=vr
 dump(OUT/'selection.json',dict(tiers=tiers,pareto=frontier,selection_split='dev',test_metrics_computed=False,raw_dev_natural_accuracy=rawdev))
 dump(OUT/'done.json',dict(state='complete',candidates=number,seconds=time.time()-start))
 print(json.dumps(dict(candidates=number,tiers=tiers,pareto=frontier,seconds=time.time()-start)))
if __name__=='__main__':main()

