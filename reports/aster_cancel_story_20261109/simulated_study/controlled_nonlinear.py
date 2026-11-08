"""Actual CPU training on a controlled synthetic nonlinear nuisance model.

This is a mechanism experiment, not a language-model/MEGA benchmark run.
Only the CSV is written. No model checkpoint, external dataset or GPU is used.
Run in the server environment:
  CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python controlled_nonlinear.py

e is independent of nuisance nu. t_1=nu+0.1*z, and the other 15 trace
coordinates are independent noise. Hidden state h=(u=e+a(nu),t).
The frozen downstream readout is G(h)=u+0.05*t_1**2.

Output cancellation learns c(t), score'=G(h)+c(t).
Hidden erasure learns the same size function g(t), T(h)=(u+g(t),0_16),
and is trained through G(T(h)). The latter really zeros all trace coordinates.
Evidence pairs change e only: ER=100% follows from this construction.
"""
import csv,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.linalg import solve

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
OUT=Path(__file__).with_name('controlled_nonlinear_results.csv')
SEEDS=(11,22,33)
SCENARIOS=('linear_nuisance','nonlinear_nuisance')
COUNTS={'fit':1000,'dev':300,'test':1000}
STEPS=600
BATCH=128
LR=.01
ALPHA=.01

def nuisance(nu,scenario):
 return nu if scenario=='linear_nuisance' else .5*nu+1.1*(nu**2-1.)
def build(rng,n,scenario):
 nu=rng.normal(size=(n,2))
 t=rng.normal(size=(n,2,16))
 t[:,:,0]=nu+.1*rng.normal(size=(n,2))
 e=rng.normal(size=n)
 u=e[:,None]+nuisance(nu,scenario)
 h=np.concatenate((u[:,:,None],t),axis=2)
 # Separate evidence pairs, with identical nuisance and trace on both sides.
 nu_e=rng.normal(size=n)
 te=rng.normal(size=(n,16))
 te[:,0]=nu_e+.1*rng.normal(size=n)
 sign=rng.choice([-1.,1.],size=n)
 mag=.5+np.abs(rng.normal(size=n))
 ee=np.column_stack((-sign*mag,sign*mag))
 ue=ee+nuisance(nu_e,scenario)[:,None]
 he=np.concatenate((ue[:,:,None],np.repeat(te[:,None,:],2,axis=1)),axis=2)
 return {'h':h,'he':he,'evidence_sign':sign}
def frozen_G_np(h):
 return h[...,0]+.05*h[...,1]**2
def frozen_G(h):
 return h[...,0]+.05*h[...,1].square()
class Corrector(nn.Module):
 def __init__(self):
  super().__init__()
  self.network=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,1,bias=False))
 def forward(self,t):
  return self.network(t).squeeze(-1)
def score_tensor(h,net,hidden):
 if hidden:
  corrected=torch.cat(((h[...,0]+net(h[...,1:])).unsqueeze(-1),torch.zeros_like(h[...,1:])),dim=-1)
  return frozen_G(corrected)
 return frozen_G(h)+net(h[...,1:])
def fit_mlp(train,hidden,seed):
 # Same initialization, optimizer budget and sampled batches for both MLP methods.
 torch.manual_seed(seed+20261109)
 net=Corrector()
 optimizer=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-4)
 h=torch.tensor(train['h'],dtype=torch.float32)
 index_rng=np.random.default_rng(seed+1024)
 started=time.perf_counter()
 net.train()
 for _ in range(STEPS):
  idx=index_rng.integers(0,len(h),size=BATCH)
  s=score_tensor(h[idx],net,hidden)
  loss=(s[:,1]-s[:,0]).square().mean()
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
 net.eval()
 return net,time.perf_counter()-started
def apply_np(h,method,model):
 if method=='raw':return frozen_G_np(h)
 if method=='linear_output_ridge':return frozen_G_np(h)+h[...,1:]@model
 with torch.no_grad():
  correction=model(torch.tensor(h[...,1:],dtype=torch.float32)).numpy().astype(np.float64)
 if method=='nonlinear_hidden_erase_mlp':
  # Explicit evaluation of G after replacing u and zeroing all 16 trace coordinates.
  edited=np.concatenate(((h[...,0]+correction)[...,None],np.zeros_like(h[...,1:])),axis=-1)
  return frozen_G_np(edited)
 return frozen_G_np(h)+correction
def rms(x):return float(np.sqrt(np.mean(x*x)))
def evaluate(data,method,model):
 raw=frozen_G_np(data['h'])
 s=apply_np(data['h'],method,model)
 d=s[:,1]-s[:,0];d0=raw[:,1]-raw[:,0]
 var=float(np.var(d0))
 he=data['he']
 e0=frozen_G_np(he);es=apply_np(he,method,model)
 de0=e0[:,1]-e0[:,0];de=es[:,1]-es[:,0]
 sign=data['evidence_sign']
 return dict(
  invariant_rms=rms(d),raw_invariant_rms=rms(d0),
  variance_reduction=1-float(np.var(d))/var,
  r2=1-float(np.mean(d*d))/var,
  evidence_retention=float(sign@de/(sign@de0)),
  cf_correction_over_ap_rms=rms(de-de0)/rms(d0),
  trace_coordinate_rms_after=0. if method=='nonlinear_hidden_erase_mlp' else rms(data['h'][...,1:]),
  invariant_mean=float(np.mean(d)))
results=[]
started=time.perf_counter()
for scenario_index,scenario in enumerate(SCENARIOS):
 for seed in SEEDS:
  rng=np.random.default_rng(20261109+scenario_index*10000+seed)
  data={split:build(rng,n,scenario) for split,n in COUNTS.items()}
  fit=data['fit'];raw=frozen_G_np(fit['h'])
  x=fit['h'][:,1,1:]-fit['h'][:,0,1:]
  y=raw[:,1]-raw[:,0]
  w=-solve(x.T@x+ALPHA*np.eye(16),x.T@y,assume_a='pos')
  output_net,output_seconds=fit_mlp(fit,False,seed)
  hidden_net,hidden_seconds=fit_mlp(fit,True,seed)
  methods=[
   ('raw',None,0,0,0.),
   ('linear_output_ridge',w,16,0,0.),
   ('nonlinear_output_mlp',output_net,sum(p.numel() for p in output_net.parameters()),STEPS,output_seconds),
   ('nonlinear_hidden_erase_mlp',hidden_net,sum(p.numel() for p in hidden_net.parameters()),STEPS,hidden_seconds)]
  for method,model,params,steps,seconds in methods:
   for split,dd in data.items():
    results.append(dict(status='controlled_synthetic',scenario=scenario,seed=seed,method=method,split=split,
     n_invariant_pairs=len(dd['h']),n_evidence_pairs=len(dd['he']),parameters=params,
     training_steps=steps,training_seconds=seconds,
     paired_intercept=False,learning_rate=LR if steps else None,ridge_alpha=ALPHA if method=='linear_output_ridge' else None,
     gpu_used=False,test_used_for_training_or_selection=False,dev_used_for_selection=False,
     er_is_guaranteed_by_pure_trace_construction=True,**evaluate(dd,method,model)))
with OUT.open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
summary=[]
for scenario in SCENARIOS:
 for method in ('raw','linear_output_ridge','nonlinear_output_mlp','nonlinear_hidden_erase_mlp'):
  rr=[r for r in results if r['scenario']==scenario and r['method']==method and r['split']=='test']
  summary.append(dict(scenario=scenario,method=method,parameters=rr[0]['parameters'],
   VR_mean=float(np.mean([r['variance_reduction'] for r in rr])),
   VR_std=float(np.std([r['variance_reduction'] for r in rr],ddof=1)),
   R2_mean=float(np.mean([r['r2'] for r in rr])),
   ER_mean=float(np.mean([r['evidence_retention'] for r in rr]))))
print({'status':'controlled_synthetic','seconds':time.perf_counter()-started,'csv_rows':len(results),'test_summary':summary},flush=True)

