
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

 # Refine promising higher-dimensional features; all bases already fit-only.
 families=dict(np.load(OUT/'feature_coordinates.npz'))
 results=json.loads((OUT/'candidates.json').read_text())
 counts=Counter(p['source_id'] for p in fitap)
 sourceweight=np.array([1/counts[p['source_id']] for p in fitap]);sourceweight/=sourceweight.mean()
 weightsets={'plain':np.ones(len(y)),'source':sourceweight}
 number=0
 for fam in ('AP128','AP256'):
  zz=families[fam];mu=zz[fidx].mean(0);sd=np.maximum(zz[fidx].std(0),1e-6);Z=(zz-mu)/sd
  X=Z[ar]-Z[al];E=Z[cr]-Z[cl];En=E.T@E/len(E)
  A=Z[fi];An=A.T@A/len(A);dim=Z.shape[1]
  for balance in ('plain','source'):
   sw=weightsets[balance];Q=X.T@(sw[:,None]*X)/len(X);q=X.T@(sw*y)/len(X)
   for alpha in (.01,.1,1.):
    for beta in (.3,1.,3.,10.,30.):
     for anchor in (0.,.1,1.):
      M=Q+beta*En+anchor*An+alpha*np.eye(dim)
      w=cho_solve(cho_factor(M,lower=True,check_finite=False),-q,check_finite=False)
      delta=Z@w;b=calibrate(delta);name='ref_'+str(number).zfill(4);number+=1
      cfg=dict(name=name,family=fam,dimension=dim,balance=balance,alpha=alpha,evidence_weight=beta,natural_anchor=anchor,b=b,
       objective='mean(weight_AP*(raw_AP+delta_AP)^2)+beta*mean(delta_CF^2)+alpha*sum(w^2)+anchor*mean(delta_natural_fit^2)',
       score_file=str(OUT/'candidates'/(name+'.npz')),model_file=str(OUT/'candidates'/(name+'_model.npz')))
      np.savez_compressed(OUT/'candidates'/(name+'.npz'),delta=delta.astype('float32'),b=b)
      np.savez_compressed(OUT/'candidates'/(name+'_model.npz'),w=w,b=b,mu=mu,scale=sd)
      results.append(dict(config=cfg,metrics=measure(delta,b)))
 dump(OUT/'candidates.json',results)
 rawdev=float(np.mean((ell[natsets['dev'][0]]>0)==natsets['dev'][1]))
 scaled=[];tierpools={str(k):[] for k in (.05,.1,.2,.4)}
 dal,dar,_,_=pairsets['dev','AP'];dcl,dcr,_,_=pairsets['dev','CF'];rawap=ell[dar]-ell[dal]
 for r in results:
  base=np.load(r['config']['score_file']);delta=base['delta'].astype('float64')
  dc=delta[dar]-delta[dal];variance=float(np.var(dc))
  opt=max(0.,-float(np.mean((rawap-rawap.mean())*(dc-dc.mean())))/max(variance,1e-15))
  er=r['metrics']['dev']['CF']['mean_evidence_retention'];cost=r['metrics']['dev']['CF']['correction_rms_over_ap_rms']
  for tier in (.05,.1,.2,.4):
   gain=min(1.,opt,tier/max(cost,1e-15),(.02/(1-er) if er<1 else 1.))
   if gain<1e-9:continue
   ds=delta*gain;b=calibrate(ds)
   cfg=dict(r['config'],parent=r['config']['name'],gain=gain,b=b,gain_selection='dev AP variance optimum clipped by dev evidence RMS tier and ER>=.98',dev_tier=tier)
   met=measure(ds,b)
   rec=dict(config=cfg,metrics=met);scaled.append(rec)
   if met['dev']['CF']['mean_evidence_retention']>=.98-1e-8 and met['dev']['natural']['accuracy']>=rawdev:
    tierpools[str(tier)].append(rec)
 dump(OUT/'scaled_candidates.json',scaled)
 tiers={};toprecords=[]
 for tier,rr in tierpools.items():
  top=sorted(rr,key=lambda r:r['metrics']['dev']['AP']['variance_reduction'],reverse=True)[:5]
  tiers[tier]=[]
  for j,r in enumerate(top):
   name='cal_'+tier.replace('.','p')+'_'+str(j)
   cfg=r['config'];base=np.load(cfg['score_file']);delta=base['delta']*cfg['gain']
   cfg['name']=name;cfg['parent_score_file']=cfg['score_file'];cfg['score_file']=str(OUT/'candidates'/(name+'.npz'))
   np.savez_compressed(cfg['score_file'],delta=delta,b=cfg['b'])
   tiers[tier].append(name);toprecords.append(r)
 dump(OUT/'calibrated_top.json',toprecords)
 dump(OUT/'selection.json',dict(tiers=tiers,selection_split='dev',test_metrics_computed=False,raw_dev_natural_accuracy=rawdev,
  note='Full parent models/configurations in candidates.json; amplitude selected from dev AP and CF, b fitted on fit natural nodes'))
 dump(OUT/'refine_done.json',dict(state='complete',original_plus_refine=len(results),dev_amplitude_candidates=len(scaled),seconds=time.time()-start))
 print(json.dumps(dict(original_plus_refine=len(results),dev_amplitude_candidates=len(scaled),tiers=tiers,seconds=time.time()-start)))
if __name__=='__main__':main()
