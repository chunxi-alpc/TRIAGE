
"""Frozen six-method score adapter. Does not fit or select anything."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path('/home/gcx/aster_next_20261109')
SUPPORT=Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen/supports_r16.npz')
RESIDUAL_BASIS=Path('/home/gcx/aster_optimization_20261109/shared/residual_basis.npz')
class SmallEditor(nn.Module):
 def __init__(self,dim,width,depth=1,trace=False):
  super().__init__();layers=[nn.Linear(dim,width),nn.GELU()]
  if depth==2:layers.extend([nn.Linear(width,width),nn.GELU()])
  layers.append(nn.Linear(width,1,bias=False));self.net=nn.Sequential(*layers)
  if trace:self.register_buffer('base_w',torch.zeros(dim))
 def forward(self,x):
  value=self.net(x)[:,0]
  return value+x@self.base_w if hasattr(self,'base_w') else value
class FrozenMethods:
 def __init__(self,manifest=ROOT/'frozen_methods.json'):
  torch.set_num_threads(2)
  self.manifest=json.loads(Path(manifest).read_text())
  assert self.manifest.get('frozen_before_expanded_evaluation')
  self.support=dict(np.load(SUPPORT));self.heads=self.support['heads'].astype(int)
  self.residual_basis=dict(np.load(RESIDUAL_BASIS))
  w=np.load(self.manifest['original_linear_weights']);key=self.manifest['original_linear_key']
  self.w=w[key];self.linear_b=float(w[key+'_b']);self.bias=float(w['bias_only_b'])
  self.models={}
  for entry in self.manifest['methods'].values():
   for member in entry.get('members',[entry]):
    path=member['model_file']
    if path in self.models:continue
    state=torch.load(path,map_location='cpu',weights_only=False)
    cfg=state.get('config',state.get('cfg'));kind=member['kind']
    model=SmallEditor(len(state['mu']),cfg['width'],cfg.get('depth',1),trace=kind=='neural')
    model.load_state_dict(state['state_dict']);model.eval()
    self.models[path]=dict(model=model,mu=state['mu'],scale=state['scale'],cfg=cfg,kind=kind)
  self.names=['raw','fit_bias_only','old_linear']+list(self.manifest['methods'])
 def raw_selected(self,arrays):
  if 'selected_heads' in arrays:return arrays['selected_heads'].reshape(-1).astype('float32')
  a=arrays['heads'];return a.reshape(-1,a.shape[-1])[self.heads].reshape(-1).astype('float32')
 def coordinates(self,ell,selected_heads,residual):
  raw=(np.asarray(selected_heads,dtype='float32').reshape(-1)-self.support['mu'])/self.support['scale']
  t=raw@self.support['N'];r32=np.concatenate([t,raw@self.support['S']]).astype('float32')
  b=self.residual_basis;res64=((np.asarray(residual,dtype='float32').reshape(-1)-b['mu'])/b['scale'])@b['R']
  score=np.float32(ell);q=np.array([score/10,abs(score)/10,np.tanh(score/4)],dtype='float32')
  return dict(r16=t.astype('float32'),r32=r32,residual64=res64.astype('float32'),q=q)
 def predict(self,ell,selected_heads,residual):
  a=self.coordinates(ell,selected_heads,residual)
  predictions={}
  for path,obj in self.models.items():
   cfg=obj['cfg']
   if obj['kind']=='neural':x=a[cfg['feature']]
   else:
    family=cfg['family']
    if family=='combined_score':x=np.concatenate([a['r32'],a['residual64'],a['q']])
    elif family=='r16_score':x=np.concatenate([a['r16'],a['q']])
    elif family=='r32_score':x=np.concatenate([a['r32'],a['q']])
    elif family=='res64_score':x=np.concatenate([a['residual64'],a['q']])
    elif family=='score_only':x=a['q']
    else:raise ValueError('Unsupported frozen feature family: '+family)
   z=((x-obj['mu'])/obj['scale']).astype('float32')
   with torch.no_grad():predictions[path]=float(obj['model'](torch.from_numpy(z[None,:]))[0])
  result=dict(raw=float(ell),fit_bias_only=float(ell+self.bias),old_linear=float(ell+a['r16']@self.w+self.linear_b))
  for name,entry in self.manifest['methods'].items():
   if entry['kind']=='ensemble':
    delta=sum(float(member['coefficient'])*predictions[member['model_file']] for member in entry['members'])
   else:delta=predictions[entry['model_file']]
   result[name]=float(ell+entry.get('gain',1.)*delta+entry['b'])
  return result
 def predict_file(self,record):
  with np.load(record['feature']) as arrays:
   return self.predict(record['ell'],self.raw_selected(arrays),arrays['residual'])

