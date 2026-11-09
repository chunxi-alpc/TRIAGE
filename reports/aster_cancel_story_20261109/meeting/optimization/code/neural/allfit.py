
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json,time,itertools,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq
from scipy.special import expit
OUT=Path('/home/gcx/aster_optimization_20261109/neural/allfit')
CACHE=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OLD=CACHE.parent/'evidence_preserving'
OUT.mkdir(parents=True,exist_ok=True)
def dump(p,v):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
class Editor(nn.Module):
 def __init__(self,dim,width,depth=1):
  super().__init__()
  layers=[nn.Linear(dim,width),nn.GELU()]
  if depth==2:layers += [nn.Linear(width,width),nn.GELU()]
  layers += [nn.Linear(width,1,bias=False)]
  self.net=nn.Sequential(*layers)
  nn.init.zeros_(self.net[-1].weight)
  self.register_buffer('base_w',torch.zeros(dim))
 def forward(self,z):return self.net(z)[:,0]+z@self.base_w
def main():
 torch.set_num_threads(2);torch.cuda.set_device(1);dev=torch.device('cuda:1')
 started=time.time();aa=np.load(CACHE/'coordinates.npz')
 ids=aa['ids'].tolist();ix={v:i for i,v in enumerate(ids)};ell=aa['ell']
 meta=json.loads((CACHE/'metadata.json').read_text())
 vs=[v for v in meta['views'] if v['split'] in ('fit','dev')]
 train=[v for v in vs if v['split']=='fit'];valid=[v for v in vs if v['split']=='dev']
 include=np.array([ix[v['view_id']] for v in vs]);local={v['view_id']:i for i,v in enumerate(vs)}
 e=ell[include];yt=np.array([v['label'] for v in vs])
 pairs=[p for p in meta['pairs'] if p['split'] in ('fit','dev')]
 def pairidx(split,kinds):
  pp=[p for p in pairs if p['split']==split and p['kind'] in kinds]
  return pp,np.array([local[p['left']] for p in pp]),np.array([local[p['right']] for p in pp])
 AP,al,ar=pairidx('fit',('A','P'));CF,cl,cr=pairidx('fit',('C','F'))
 natural={sp:np.array([local[v['view_id']] for v in vs if v['split']==sp and v['variant']=='full' and v['original']]) for sp in ('fit','dev')}
 fullidx=np.array([local[v['view_id']] for v in vs if v['split']=='fit' and v['variant']=='full'])
 apscale=float(np.sqrt(np.mean((e[ar]-e[al])**2)));budget=.1*apscale
 def calibrate(c):
  ni=natural['fit'];f=lambda b:float(np.mean(expit(e[ni]+c[ni]+b)-yt[ni]))
  lo,hi=-1.,1.
  while f(lo)>0:lo*=2
  while f(hi)<0:hi*=2
  return float(brentq(f,lo,hi))
 braw=calibrate(np.zeros_like(e))
 old=np.load(OLD/'weights.npz');prior_w=old['bounded_0.1']
 ee=torch.tensor(e,dtype=torch.float32,device=dev);yy=torch.tensor(yt,dtype=torch.float32,device=dev)
 al0=torch.tensor(al,device=dev);ar0=torch.tensor(ar,device=dev);cl0=torch.tensor(cl,device=dev);cr0=torch.tensor(cr,device=dev)
 ni=torch.tensor([local[v['view_id']] for v in vs if v['split']=='fit'],device=dev)
 # Shared-input noise on CF endpoints regularizes protection away from observed exact endpoints.
 rows_by_kind={k:torch.tensor([i for i,p in enumerate(AP) if p['kind']==k],device=dev) for k in ('A','P')}
 metricspec={}
 for sp in ('fit','dev'):
  for k,ks in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
   pp,li,ri=pairidx(sp,ks);metricspec[sp,k]=(pp,li,ri)
 def measures(delta,b):
  ss=e+delta+b;out={}
  for sp in ('fit','dev'):
   _,apl,apr=metricspec[sp,'AP'];apraw=e[apr]-e[apl];den=np.sqrt(np.mean(apraw**2))
   groups={}
   for k in ('A','P','AP','C','F','CF','F0'):
    pp,li,ri=metricspec[sp,k];raw=e[ri]-e[li];dy=ss[ri]-ss[li];dc=delta[ri]-delta[li]
    yl=np.array([p['left_label'] for p in pp]);yr=np.array([p['right_label'] for p in pp]);direction=yr-yl
    cor=((ss[li]>0)==yl)&((ss[ri]>0)==yr)
    r=dict(n=len(pp),difference_rms=float(np.sqrt(np.mean(dy*dy))),correction_rms=float(np.sqrt(np.mean(dc*dc))),
     correction_ratio=float(np.sqrt(np.mean(dc*dc))/den),correction_max=float(np.max(np.abs(dc))),paired_accuracy=float(np.mean(cor)),
     paired_correct=int(np.sum(cor)),sources=len(set(p['source_id'] for p in pp)))
    if k in ('A','P','AP','F0'):r['VR']=float(1-np.var(dy)/np.var(raw))
    else:r.update(ER=float(np.mean(direction*dy)/np.mean(direction*raw)),degraded=int(np.sum((direction*raw>0)&(direction*dy<=0))))
    groups[k]=r
   nn=natural[sp];groups['natural']=dict(n=len(nn),correct=int(np.sum((ss[nn]>0)==yt[nn])),accuracy=float(np.mean((ss[nn]>0)==yt[nn])))
   out[sp]=groups
  return out
 # 96 core runs + 48 targeted capacity/CE/robust variants, two checkpoints each.
 configs=[]
 for feature,width,cfweight,ce in itertools.product(('r16','r32'),(64,128),(30.,100.,1000.),(1.,10.,100.)):
  configs.append(dict(feature=feature,width=width,depth=1,penalty='mse',cfweight=cfweight,ce=ce,jitter=.05,balance=False,prior=False,lr=.001,ce_pool='all_fit'))
 dump(OUT/'search_plan.json',dict(runs=len(configs),checkpoints=[600,1600],seed=11,gpu=1,fit_AP=len(AP),fit_CF=len(CF),
   configurations=configs,selection_inputs=['fit','dev'],fit_AP_raw_rms=apscale,fit_hinge_budget=budget))
 done=[];feature_cache={}
 for fi in ('r16','r32'):
  x=aa[fi][include].astype('float32');mu=x[fullidx].mean(0);scale=x[fullidx].std(0).clip(.01)
  z=(x-mu)/scale
  feature_cache[fi]=(torch.tensor(z,device=dev),mu,scale)
 for job,cfg in enumerate(configs):
  torch.manual_seed(11);np.random.seed(11)
  z,mu,scale=feature_cache[cfg['feature']];dim=z.shape[1]
  model=Editor(dim,cfg['width'],cfg['depth']).to(dev)
  if cfg['prior']:
   pw=np.zeros(dim,dtype='float32');pw[:16]=prior_w.astype('float32');model.base_w.copy_(torch.tensor(pw*scale,device=dev))
  train_b=torch.tensor(braw,dtype=torch.float32,device=dev,requires_grad=True)
  opt=torch.optim.Adam(list(model.parameters())+[train_b],lr=cfg['lr'])
  for step in range(1,1601):
   if cfg['balance']:
    pa=rows_by_kind['A'][torch.randint(len(rows_by_kind['A']),(224,),device=dev)]
    pp=rows_by_kind['P'][torch.randint(len(rows_by_kind['P']),(32,),device=dev)]
    pick=torch.cat([pa,pp])
   else:pick=torch.randint(len(al0),(256,),device=dev)
   li=al0[pick];ri=ar0[pick]
   ci=ni[torch.randint(len(ni),(512,),device=dev)]
   zz=torch.cat([z[li],z[ri],z[cl0],z[cr0],z[ci]])
   cc=model(zz);dl,dr,dc1,dc2,dn=cc.split([len(li),len(ri),len(cl0),len(cr0),len(ci)])
   ap=(ee[ri]-ee[li]+dr-dl).square().mean()
   cfdiff=dc2-dc1
   protected=cfdiff.square().mean() if cfg['penalty']=='mse' else torch.relu(cfdiff.abs()-budget).square().mean()
   if cfg['jitter']:
    noise=torch.randn_like(z[cl0])*cfg['jitter']
    aug=model(torch.cat([z[cl0]+noise,z[cr0]+noise]));aa0,bb0=aug.chunk(2)
    protected=.5*(protected+(bb0-aa0).square().mean())
   ce=nn.functional.binary_cross_entropy_with_logits(ee[ci]+dn+train_b,yy[ci])
   loss=ap+cfg['cfweight']*protected+cfg['ce']*ce+.001*sum(p.square().sum() for p in model.parameters())/len(AP)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),100.);opt.step()
   if step in (600,1600):
    with torch.no_grad():
     delta=np.concatenate([model(z[lo:lo+1024]).cpu().numpy() for lo in range(0,len(z),1024)]).astype('float64')
    if not np.isfinite(delta).all():continue
    b=calibrate(delta);name=f'a{job:03d}_s{step}'
    metrics=measures(delta,b)
    record=dict(name=name,config=cfg,steps=step,seed=11,b=b,metrics=metrics,parameters=sum(p.numel() for p in model.parameters()),
      seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(dev)/2**30,fitdev_only=True)
    dump(OUT/(name+'.json'),record)
    np.savez_compressed(OUT/(name+'.npz'),ids=np.array([v['view_id'] for v in vs]),delta=delta.astype('float32'),b=b)
    torch.save(dict(config=cfg,steps=step,seed=11,state_dict=model.state_dict(),mu=mu,scale=scale,b=b),OUT/(name+'.pt'))
    done.append(record)
   if step==1600:
    dump(OUT/'status.json',dict(state='running',runs_completed=job+1,total_runs=len(configs),candidates=len(done),seconds=time.time()-started,last_name=name))
 dump(OUT/'candidates.json',done)
 # Pareto choices are fit/dev only, no protected test scores consumed.
 chosen={}
 for tier in (.05,.1,.2,.4):
  eligible=[r for r in done if r['metrics']['dev']['CF']['correction_ratio']<=tier and r['metrics']['dev']['CF']['ER']>=.98]
  if eligible:chosen[str(tier)]=max(eligible,key=lambda r:(r['metrics']['dev']['AP']['VR'],r['metrics']['dev']['natural']['accuracy']))['name']
 dump(OUT/'selection.json',dict(criteria='dev correction_ratio tiers and ER>=.98, then AP VR; natural accuracy tie break',chosen=chosen))
 dump(OUT/'done.json',dict(state='complete',runs=len(configs),candidates=len(done),seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(dev)/2**30))
if __name__=='__main__':main()

