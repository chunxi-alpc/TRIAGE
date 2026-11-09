
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import numpy as np,json,time,argparse,itertools
import torch
from torch import nn
from scipy.special import expit
from scipy.optimize import brentq
OUT=Path('/home/gcx/aster_optimization_20261109/score_conditioned')
SH=Path('/home/gcx/aster_optimization_20261109/shared')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT.mkdir(exist_ok=True);(OUT/'candidates').mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
class Editor(nn.Module):
 def __init__(self,d,width=64):
  super().__init__();self.net=nn.Sequential(nn.Linear(d,width),nn.GELU(),nn.Linear(width,1,bias=False))
  nn.init.zeros_(self.net[-1].weight)
 def forward(self,z):return self.net(z)[:,0]
def features(family,a,lin=None):
 score=a['ell'].astype('float32')
 q=np.column_stack([score/10,np.abs(score)/10,np.tanh(score/4)]).astype('float32')
 if family=='score_only':return q
 if family=='r16_score':x=a['r16']
 elif family=='r32_score':x=a['r32']
 elif family=='res64_score':x=a['residual64']
 elif family=='combined_score':x=np.concatenate([a['r32'],a['residual64']],axis=1)
 elif family.startswith('AP'):
  dim=int(family.split('_')[0][2:]);x=lin['AP256'][:len(score),:dim]
 else:raise ValueError(family)
 return np.concatenate([x,q],axis=1).astype('float32')
def main(seed=11,confirm=None,tag='score'):
 torch.set_num_threads(2);torch.cuda.set_device(0);device='cuda:0';start=time.time()
 a=dict(np.load(SH/'features.npz'));ids=a['ids'].tolist();nr=len(np.load(OLD/'coordinates.npz')['ids'])
 # Candidate search predicts only fit/dev view inputs. Paths and test are evaluated after selection.
 a={k:v[:nr] for k,v in a.items()};ids=ids[:nr];ix={v:i for i,v in enumerate(ids)}
 lin=np.load('/home/gcx/aster_optimization_20261109/linear/feature_coordinates.npz')
 meta=json.loads((OLD/'metadata.json').read_text());vs=meta['views'];ps=meta['pairs'];ell=a['ell']
 vv=[v for v in vs if v['split'] in ('fit','dev')];vi=np.array([ix[v['view_id']] for v in vv]);outids=np.array([v['view_id'] for v in vv])
 def pairs(sp,kk):
  pp=[p for p in ps if p['split']==sp and p['kind'] in kk]
  return pp,np.array([ix[p['left']] for p in pp]),np.array([ix[p['right']] for p in pp])
 AP,al,ar=pairs('fit',('A','P'));CF,cl,cr=pairs('fit',('C','F'))
 nat=[v for v in vs if v['split']=='fit' and v['variant']=='full' and v['original']]
 ni=np.array([ix[v['view_id']] for v in nat]);ny=np.array([v['label'] for v in nat])
 unique=np.unique(np.concatenate([al,ar,cl,cr,ni]));local=np.full(nr,-1,dtype=int);local[unique]=np.arange(len(unique))
 al0=torch.tensor(local[al],device=device);ar0=torch.tensor(local[ar],device=device)
 cl0=torch.tensor(local[cl],device=device);cr0=torch.tensor(local[cr],device=device);ni0=torch.tensor(local[ni],device=device)
 ee=torch.tensor(ell[unique],dtype=torch.float32,device=device);yy=torch.tensor(ny,dtype=torch.float32,device=device)
 Pidx=np.array([i for i,p in enumerate(AP) if p['kind']=='P']);Ptorch=torch.tensor(Pidx,device=device)
 b0=float(brentq(lambda b:np.mean(expit(ell[ni]+b)-ny),-50,50))
 budget=.1*np.sqrt(np.mean((ell[ar]-ell[al])**2))
 cfgs=[]
 if confirm:
  cfgs=json.loads(Path(confirm).read_text())
 else:
  for family,beta,ce,mode in itertools.product(['r16_score','r32_score','res64_score','combined_score','AP64_score','AP128_score'],[1.,4.,16.,64.],[0.,5.,30.],['mse','hinge']):
   cfgs.append(dict(family=family,beta=beta,ce=ce,mode=mode,width=64,steps=1200,lr=.002,p_fraction=.1))
 output=[]
 for j,cfg in enumerate(cfgs):
  torch.manual_seed(seed)
  x=features(cfg['family'],a,lin)
  mu=x[ni].mean(0);scale=np.maximum(x[ni].std(0),1e-4)
  z=(x-mu)/scale;zz=torch.tensor(z[unique],device=device)
  model=Editor(z.shape[1],cfg['width']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-3)
  for step in range(cfg['steps']):
   pick=torch.randint(len(AP),(116,),device=device)
   psel=Ptorch[torch.randint(len(Ptorch),(12,),device=device)]
   pick=torch.cat([pick,psel])
   left=al0[pick];right=ar0[pick]
   selected=torch.cat([left,right,cl0,cr0,ni0])
   c=model(zz[selected])
   dl,dr,dc1,dc2,cn=c.split([128,128,len(CF),len(CF),len(ni)])
   loss=(ee[right]-ee[left]+dr-dl).square().mean()
   cfd=dc2-dc1
   if cfg.get('jitter',0):
    noise=torch.randn((len(CF),z.shape[1]),device=device)*cfg['jitter'];cfd=model(zz[cr0]+noise)-model(zz[cl0]+noise)
   evidence=cfd.square().mean() if cfg['mode']=='mse' else torch.relu(cfd.abs()-budget).square().mean()
   loss=loss+cfg['beta']*evidence
   if cfg['ce']:loss=loss+cfg['ce']*nn.functional.binary_cross_entropy_with_logits(ee[ni0]+cn+b0,yy)
   loss=loss+1e-5*cn.square().mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
  with torch.no_grad():
   delta=np.empty(nr)
   for lo in range(0,nr,2048):delta[lo:lo+2048]=model(torch.tensor(z[lo:lo+2048],device=device)).cpu().numpy()
  bf=lambda b:float(np.mean(expit(ell[ni]+delta[ni]+b)-ny))
  b=float(brentq(bf,-100,100))
  g={}
  for kind,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
   pp,l,r=pairs('dev',kinds);dy=ell[r]-ell[l];dc=delta[r]-delta[l];new=dy+dc;direction=np.array([p['right_label']-p['left_label'] for p in pp])
   vals=dict(n=len(pp),correction_rms=float(np.sqrt(np.mean(dc**2))),difference_rms=float(np.sqrt(np.mean(new**2))))
   if kind in ('A','P','AP','F0'):vals['variance_reduction']=1-float(np.var(new)/np.var(dy))
   else:
    vals['mean_evidence_retention']=float(np.mean(direction*new)/np.mean(direction*dy))
    vals['response_direction_degraded']=int(np.sum((direction*dy>0)&(direction*new<=0)))
   g[kind]=vals
  pp,l,r=pairs('dev',('A','P'));apscale=np.sqrt(np.mean((ell[r]-ell[l])**2));g['CF']['correction_rms_over_ap_rms']=g['CF']['correction_rms']/apscale
  devnat=[v for v in vs if v['split']=='dev' and v['variant']=='full' and v['original']]
  di=np.array([ix[v['view_id']] for v in devnat]);dy=np.array([v['label'] for v in devnat])
  g['natural']=dict(correct=int(np.sum((ell[di]+delta[di]+b>0)==dy)),n=len(di))
  name=tag+'_'+str(j).zfill(4)+'_seed'+str(seed)
  torch.save(dict(state_dict=model.state_dict(),mu=mu,scale=scale,cfg=cfg,seed=seed,b=b),OUT/'candidates'/(name+'.pt'))
  np.savez_compressed(OUT/'candidates'/(name+'.npz'),ids=outids,delta=delta[vi],b=b)
  output.append(dict(name=name,cfg=cfg,seed=seed,b=b,dev=g))
  dump(OUT/(tag+'_candidates_seed'+str(seed)+'.json'),output)
  if j%10==0:dump(OUT/'status.json',dict(stage='search',completed=j+1,total=len(cfgs),seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
 dump(OUT/(tag+'_done_seed'+str(seed)+'.json'),dict(state='complete',count=len(cfgs),seconds=time.time()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=11);p.add_argument('--confirm');p.add_argument('--tag',default='score');args=p.parse_args();main(args.seed,args.confirm,args.tag)

