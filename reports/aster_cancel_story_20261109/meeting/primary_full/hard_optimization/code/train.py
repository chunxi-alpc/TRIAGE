
"""Two fixed tail-loss schemes; no official examples or labels are loaded."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,json,time,math
import numpy as np,torch
from torch import nn
from scipy.special import expit
from scipy.optimize import brentq
BASE=Path('/home/gcx/aster_hard_optimization_20261109')
class Editor(nn.Module):
 def __init__(self,d,width=128):
  super().__init__();self.net=nn.Sequential(nn.Linear(d,width),nn.GELU(),nn.Linear(width,1,bias=False))
 def forward(self,x):return self.net(x)[:,0]
def dump(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def calibrate(ell,delta,y,indices):
 f=lambda b:float(np.mean(expit(ell[indices]+delta[indices]+b)-y[indices]))
 return float(brentq(f,-200,200))
def tail(x,fraction=.2):return torch.topk(x,k=max(1,int(math.ceil(len(x)*fraction))),sorted=False).values.mean()
def main(gpu,scheme):
 torch.set_num_threads(2);torch.cuda.set_device(gpu);device=torch.device('cuda',gpu)
 a=dict(np.load(BASE/'data.npz'));meta=json.loads((BASE/'data.json').read_text());plan=json.loads((BASE/'plan.json').read_text())
 ids=a['ids'].tolist();ix={v:i for i,v in enumerate(ids)};ell=a['ell'];labels=a['labels']
 ps=meta['pairs']
 def pair_indices(kinds):
  pp=[p for p in ps if p['split']=='fit' and p['kind'] in kinds]
  return np.array([ix[p['left']] for p in pp]),np.array([ix[p['right']] for p in pp])
 al,ar=pair_indices(('A','P'));cl,cr=pair_indices(('C','F'));budget=.1*np.sqrt(np.mean((ell[ar]-ell[al])**2))
 natural=a['natural_fit'];ci=a['ce_indices'];cw=a['ce_weights']
 selected=np.unique(np.concatenate([al,ar,cl,cr,ci]));local=np.full(len(ids),-1,dtype=int);local[selected]=np.arange(len(selected))
 aa=torch.tensor(local[al],device=device);ab=torch.tensor(local[ar],device=device);ca=torch.tensor(local[cl],device=device);cb=torch.tensor(local[cr],device=device)
 ceidx=torch.tensor(local[ci],device=device);nidx=torch.tensor(local[natural],device=device)
 ee=torch.tensor(ell[selected],dtype=torch.float32,device=device);cy=torch.tensor(labels[ci],dtype=torch.float32,device=device);weights=torch.tensor(cw,device=device)
 ydiff=ee[ab]-ee[aa]
 out=BASE/scheme;out.mkdir(exist_ok=True);records=[];started=time.time()
 for seed in (11,22,33):
  torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats(device);before=time.time()
  old=next(m for m in meta['initial_balanced']['members'] if int(m['seed'])==seed)
  state=torch.load(old['model_file'],map_location='cpu',weights_only=False)
  mu=state['mu'];scale=state['scale'];model=Editor(len(mu),128).to(device)
  model.load_state_dict(state['state_dict'])
  with torch.no_grad():model.net[-1].weight.mul_(float(meta['initial_balanced']['gain']))
  z=((a['x']-mu)/scale).astype('float32');zz=torch.tensor(z[selected],device=device)
  with torch.no_grad():initial=model(zz).detach()
  initial_all=np.zeros(len(ids));initial_all[selected]=initial.cpu().numpy()
  b0=calibrate(ell,initial_all,labels,natural)
  bias=torch.tensor(b0,dtype=torch.float32,device=device,requires_grad=True)
  optimizer=torch.optim.AdamW(list(model.parameters())+[bias],lr=.0003,weight_decay=.001)
  for step in range(1200):
   correction=model(zz);residual=ydiff+correction[ab]-correction[aa]
   ap_loss=residual.square().mean()+tail(residual.square())
   cd=correction[cb]-correction[ca]
   if scheme=='tail_pair':
    cf_loss=1000*cd.square().mean()+1000*torch.relu(cd.abs()-budget).square().mean()
   elif scheme=='tail_robust':
    noise=torch.randn((len(ca),zz.shape[1]),device=device)*.1
    local_delta=model(zz[cb]+noise)-model(zz[ca]+noise)
    cf_loss=500*(tail(cd.square())+tail(local_delta.square()))+1000*torch.relu(cd.abs()-budget).square().max()
   else:raise ValueError(scheme)
   ce_loss=(nn.functional.binary_cross_entropy_with_logits(ee[ceidx]+correction[ceidx]+bias,cy,reduction='none')*weights).sum()
   anchor=.01*(correction[nidx]-initial[nidx]).square().mean()
   loss=ap_loss+cf_loss+ce_loss+anchor
   optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
  model.eval()
  with torch.no_grad():
   unscaled=model(zz);maximum=float((unscaled[cb]-unscaled[ca]).abs().max())
   factor=min(1.,float(budget)/max(maximum,1e-12))
   if factor<1:model.net[-1].weight.mul_(factor)
   delta=np.concatenate([model(torch.tensor(z[i:i+1024],device=device)).cpu().numpy() for i in range(0,len(ids),1024)]).astype('float64')
  b=calibrate(ell,delta,labels,natural);name=scheme+'_seed'+str(seed)
  maximum_after=float(np.max(np.abs(delta[cr]-delta[cl])))
  if maximum_after>budget+1e-5:raise RuntimeError('Fit CF budget scaling failed')
  cfg=dict(family='combined_score',width=128,scheme=scheme,steps=1200,lr=.0003,weight_decay=.001,ap_loss='mean squared difference + worst-20-percent squared difference',
   cf_loss='mean square + per-pair hinge' if scheme=='tail_pair' else 'worst-20-percent anchor+jitter square + worst hinge',
   ce='0.5*source-balanced original761 + 0.5*source-balanced extra1720',anchor_weight=.01)
  torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mu=mu,scale=scale,cfg=cfg,seed=seed,b=b,fit_scale=factor),out/(name+'.pt'))
  np.savez_compressed(out/(name+'.npz'),ids=a['ids'],delta=delta,b=b)
  info=dict(name=name,scheme=scheme,seed=seed,model_file=str(out/(name+'.pt')),prediction_file=str(out/(name+'.npz')),
   cfg=cfg,steps=1200,b=b,fit_scale=factor,fit_cf_max_before_scale=maximum,fit_cf_max_after_scale=maximum_after,fit_budget=float(budget),
   initialization=old['model_file'],initial_balanced_gain=meta['initial_balanced']['gain'],natural_fit_bce_calibration_count=len(natural),
   extra_fit_count=len(a['extra_fit']),train_sources=80,extra_fit_sources=12,seconds=time.time()-before,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
   official_data_loaded=False,dev_used_for_training=False)
  dump(out/(name+'.json'),info);records.append(info);dump(out/'status.json',dict(state='running',completed_seeds=len(records),last=name,seconds=time.time()-started))
 dump(out/'models.json',records);dump(out/'done.json',dict(state='complete',scheme=scheme,seeds=[11,22,33],seconds=time.time()-started))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);p.add_argument('--scheme',choices=['tail_pair','tail_robust'],required=True);a=p.parse_args();main(a.gpu,a.scheme)

