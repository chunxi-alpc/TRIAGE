
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import sys,json,time,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.optimize import brentq
OUT=Path(__file__).resolve().parent
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main(gpu):
 torch.set_num_threads(2);torch.cuda.set_device(gpu);device=torch.device('cuda',gpu)
 a=np.load(OUT/'coordinates.npz');ids=a['ids'].tolist();ix={v:i for i,v in enumerate(ids)}
 ell=a['ell'];t=a['r16'].astype('float32')
 meta=json.loads((OUT/'metadata.json').read_text())
 ps=meta['pairs']
 def inds(kinds):
  pp=[p for p in ps if p['split']=='fit' and p['kind'] in kinds]
  return np.array([ix[p['left']] for p in pp]),np.array([ix[p['right']] for p in pp])
 al,ar=inds(('A','P'));cl,cr=inds(('C','F'))
 natural=[v for v in meta['views'] if v['split']=='fit' and v['variant']=='full' and v['original']]
 ni=np.array([ix[v['view_id']] for v in natural]);ny=np.array([v['label'] for v in natural])
 budget=.1*np.sqrt(np.mean((ell[ar]-ell[al])**2))
 h=np.load(OUT/'residual.npy',mmap_mode='r')
 V=torch.tensor(np.load(OUT/'bases.npz')['V'],device=device)
 head=np.load(OUT/'output_head.npz')
 norm=torch.tensor(head['norm'],device=device);weights=torch.tensor(head['weights'],device=device)
 eps=float(head['eps'])
 def G(x):
  z=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*norm
  logits=z@weights.T
  return logits[:,0]-logits[:,1]
 unique=np.unique(np.concatenate([al,ar,cl,cr]))
 local=np.full(len(ids),-1,dtype=int);local[unique]=np.arange(len(unique))
 tt=torch.tensor(t[unique],device=device);hh=torch.tensor(np.array(h[unique]),device=device)
 ee=torch.tensor(ell[unique],dtype=torch.float32,device=device)
 al0=torch.tensor(local[al],device=device);ar0=torch.tensor(local[ar],device=device)
 cl0=torch.tensor(local[cl],device=device);cr0=torch.tensor(local[cr],device=device)
 original=G(hh)
 zero_error=float((ee+G(hh+torch.zeros_like(hh))-original-ee).abs().max())
 # Evaluation uses ell+(G(h+dh)-G(h)); grouping keeps zero editing bit-exact.
 zero_error_grouped=float(((G(hh+torch.zeros_like(hh))-original)).abs().max())
 types=['mlp_output','linear_representation'] if gpu==0 else ['mlp_representation']
 for kind in types:
  for seed in (11,22,33):
   start=time.time();torch.manual_seed(seed);np.random.seed(seed);torch.cuda.reset_peak_memory_stats(device)
   outdim=1 if kind=='mlp_output' else 16
   if kind=='linear_representation':model=nn.Linear(16,16,bias=False)
   else:model=nn.Sequential(nn.Linear(16,32,bias=True),nn.GELU(),nn.Linear(32,outdim,bias=False))
   model=model.to(device);last=model if isinstance(model,nn.Linear) else model[-1]
   nn.init.zeros_(last.weight)
   opt=torch.optim.Adam(model.parameters(),lr=.001)
   def correction(z,hidden,base,factor=1.):
    change=model(z)*factor
    return change[:,0] if kind=='mlp_output' else G(hidden+change@V.T)-base
   for step in range(600):
    pick=torch.randint(len(al0),(64,),device=device)
    left=al0[pick];right=ar0[pick]
    wanted=torch.cat([left,right,cl0,cr0])
    delta=correction(tt[wanted],hh[wanted],original[wanted])
    dl,dr,dc1,dc2=delta.split([64,64,len(cl0),len(cr0)])
    loss=(ee[right]-ee[left]+dr-dl).square().mean()
    loss=loss+.01/len(al)*sum(p.square().sum() for p in model.parameters())
    loss=loss+100*torch.relu((dc2-dc1).abs()-budget).square().mean()
    opt.zero_grad();loss.backward();opt.step()
   model.eval()
   with torch.no_grad():
    scale=1.;halvings=0
    for trial in range(21):
     c=correction(tt,hh,original,scale);maximum=float((c[cr0]-c[cl0]).abs().max())
     if maximum<=budget+1e-7:break
     if trial==20:scale=0.;halvings=20
     else:scale*=.5;halvings+=1
    fitmax=float((correction(tt,hh,original,scale)[cr0]-correction(tt,hh,original,scale)[cl0]).abs().max())
    all_delta=np.empty(len(ids),dtype='float64')
    for lo in range(0,len(ids),256):
     z=torch.tensor(t[lo:lo+256],device=device)
     hidden=torch.tensor(np.array(h[lo:lo+256]),device=device)
     all_delta[lo:lo+256]=correction(z,hidden,G(hidden),scale).cpu().numpy()
   base=ell[ni]+all_delta[ni]
   f=lambda b:float(np.mean(expit(base+b)-ny))
   lo,hi=-1.,1.
   while f(lo)>0:lo*=2
   while f(hi)<0:hi*=2
   b=float(brentq(f,lo,hi))
   name=kind+'_seed'+str(seed)
   np.savez_compressed(OUT/(name+'.npz'),delta=all_delta,b=b)
   torch.save(dict(kind=kind,seed=seed,state_dict=model.state_dict(),scale=scale,b=b),OUT/(name+'.pt'))
   info=dict(kind=kind,seed=seed,parameters=sum(p.numel() for p in model.parameters()),steps=600,
    learning_rate=.001,ap_batch_pairs=64,cf_batch_pairs=len(cl),regularizer='.01 * sum(parameter^2) / n_fit_AP',
    evidence_penalty='100 * mean(relu(abs(delta_CF)-budget)^2)',budget=budget,
    fit_max_evidence_correction=fitmax,scale=scale,halvings=halvings,
    zero_correction_max_abs_error=zero_error_grouped,ungrouped_roundoff=zero_error,
    bias=b,gpu=gpu,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,seconds=time.time()-start)
   dump(OUT/(name+'.json'),info)
   dump(OUT/('gpu'+str(gpu)+'_status.json'),dict(last_completed=name,state='running',**info))
 dump(OUT/('gpu'+str(gpu)+'_done.json'),dict(state='complete'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);main(p.parse_args().gpu)

