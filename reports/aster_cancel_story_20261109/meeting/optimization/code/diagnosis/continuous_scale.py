
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.optimize import brentq
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT=Path('/home/gcx/aster_optimization_20261109/diagnosis')
torch.set_num_threads(2)
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
a=np.load(OLD/'coordinates.npz');idsall=a['ids'].tolist();indexall={s:i for i,s in enumerate(idsall)}
meta=json.loads((OLD/'metadata.json').read_text())
views=[v for v in meta['views'] if v['split'] in ('fit','dev')]
ids=[v['view_id'] for v in views];ix={v:i for i,v in enumerate(ids)};gi=np.array([indexall[v] for v in ids])
ell=a['ell'][gi];trace=a['r16'][gi].astype('float32')
pairs=[p for p in meta['pairs'] if p['split'] in ('fit','dev')]
fitcf=[p for p in pairs if p['split']=='fit' and p['kind'] in ('C','F')]
fitap=[p for p in pairs if p['split']=='fit' and p['kind'] in ('A','P')]
cl=np.array([ix[p['left']] for p in fitcf]);cr=np.array([ix[p['right']] for p in fitcf])
al=np.array([ix[p['left']] for p in fitap]);ar=np.array([ix[p['right']] for p in fitap])
budget=.1*np.sqrt(np.mean((ell[ar]-ell[al])**2))
nv=[v for v in views if v['split']=='fit' and v['variant']=='full' and v['original']]
ni=np.array([ix[v['view_id']] for v in nv]);ny=np.array([v['label'] for v in nv])
head=np.load(OLD/'output_head.npz')
norm=torch.tensor(head['norm']);weights=torch.tensor(head['weights']);eps=float(head['eps'])
V=torch.tensor(np.load(OLD/'bases.npz')['V'])
h=np.load(OLD/'residual.npy',mmap_mode='r')
def G(hh):
 z=hh*torch.rsqrt(hh.square().mean(-1,keepdim=True)+eps)*norm
 logits=z@weights.T
 return logits[:,0]-logits[:,1]
def rms(v):return float(np.sqrt(np.mean(v*v)))
def metrics(score):
 result={}
 for sp in ('fit','dev'):
  out={}
  aps=[p for p in pairs if p['split']==sp and p['kind'] in ('A','P')]
  dl=np.array([ix[p['left']] for p in aps]);dr=np.array([ix[p['right']] for p in aps]);den=rms(ell[dr]-ell[dl])
  for name,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
   pp=[p for p in pairs if p['split']==sp and p['kind'] in kinds]
   li=np.array([ix[p['left']] for p in pp]);ri=np.array([ix[p['right']] for p in pp])
   raw=ell[ri]-ell[li];dy=score[ri]-score[li];dc=dy-raw
   yl=np.array([p['left_label'] for p in pp]);yr=np.array([p['right_label'] for p in pp])
   r=dict(n=len(pp),raw_rms=rms(raw),difference_rms=rms(dy),correction_rms=rms(dc),correction_over_AP=rms(dc)/den,
    paired_correct=int(np.sum(((score[li]>0)==yl)&((score[ri]>0)==yr))))
   if name in ('A','P','AP','F0'):r['VR']=1-float(np.var(dy)/np.var(raw))
   else:
    direction=yr-yl;r['ER']=float(np.mean(direction*dy)/np.mean(direction*raw))
    r['direction_degraded']=int(np.sum((direction*raw>0)&(direction*dy<=0)))
   out[name]=r
  vv=[v for v in views if v['split']==sp and v['variant']=='full' and v['original']]
  vi=np.array([ix[v['view_id']] for v in vv]);yl=np.array([v['label'] for v in vv])
  out['natural']=dict(n=len(vv),correct=int(np.sum((score[vi]>0)==yl)))
  result[sp]=out
 return result
summary=[]
for kind in ('mlp_output','linear_representation','mlp_representation'):
 for seed in (11,22,33):
  start=time.time();base=kind+'_seed'+str(seed);name='continuous_'+base
  state=torch.load(OLD/(base+'.pt'),map_location='cpu',weights_only=False)
  original_scale=float(state['scale'])
  if kind=='mlp_output':
   old=np.load(OLD/(base+'.npz'));rawdelta=old['delta'][gi]/original_scale
   maximum=float(np.max(np.abs(rawdelta[cr]-rawdelta[cl])))
   scale=min(1.,budget/max(maximum,1e-12));scale*=1-1e-9
   delta=rawdelta*scale
  else:
   model=nn.Linear(16,16,bias=False) if kind=='linear_representation' else nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16,bias=False))
   model.load_state_dict(state['state_dict']);model.eval()
   fi=np.unique(np.concatenate([cl,cr]));local={int(v):i for i,v in enumerate(fi)}
   ll=torch.tensor([local[int(i)] for i in cl]);rr=torch.tensor([local[int(i)] for i in cr])
   hh=torch.tensor(np.array(h[gi[fi]]))
   with torch.no_grad():
    change=model(torch.tensor(trace[fi]))@V.T;baseh=G(hh)
    def maxcorr(factor):
     delta_=G(hh+factor*change)-baseh
     return float((delta_[rr]-delta_[ll]).abs().max())
    if maxcorr(1.)<=budget:scale=1.
    else:
     lo,hi=0.,1.
     for _ in range(28):
      mid=(lo+hi)/2
      if maxcorr(mid)<=budget:lo=mid
      else:hi=mid
     scale=lo*(1-1e-7)
    delta=np.empty(len(ids),dtype='float64')
    for lo in range(0,len(ids),256):
     hh1=torch.tensor(np.array(h[gi[lo:lo+256]]))
     delta[lo:lo+256]=(G(hh1+scale*(model(torch.tensor(trace[lo:lo+256]))@V.T))-G(hh1)).numpy()
  maximum=float(np.max(np.abs(delta[cr]-delta[cl])))
  # FP32 native G can round at the boundary: retreat one small step when necessary.
  if maximum>budget+1e-6:
   raise RuntimeError(name+' fit budget violated '+str(maximum))
  b=float(brentq(lambda z:np.mean(expit(ell[ni]+delta[ni]+z)-ny),-200,200))
  score=ell+delta+b
  state.update(scale=float(scale),b=b,name=name,feature='r16',normalization='none',selection='continuous exact fit-CF rescaling of existing model; no test selection')
  torch.save(state,OUT/(name+'.pt'))
  np.savez_compressed(OUT/(name+'.npz'),ids=np.array(ids),delta=delta,b=b,scores=score)
  info=dict(name=name,kind=kind,seed=seed,old_scale=original_scale,new_scale=float(scale),scale_gain=float(scale/original_scale),
   budget=budget,max_fit_cf_correction=maximum,b=b,seconds=time.time()-start,metrics=metrics(score))
  dump(OUT/(name+'.json'),info);summary.append(info)
dump(OUT/'continuous_summary.json',dict(test_used=False,candidates=summary))
for r in summary:
 d=r['metrics']['dev']
 print(r['name'],'scale',round(r['new_scale'],4),'gain',round(r['scale_gain'],3),'VR',round(d['AP']['VR']*100,3),'ER',round(d['CF']['ER']*100,3),'CF_RMS',round(d['CF']['correction_rms'],4),'natural',d['natural']['correct'])

