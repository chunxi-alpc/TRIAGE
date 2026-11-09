
"""CPU inference for two newly fitted frozen tail-loss ensembles."""
from pathlib import Path
import json,numpy as np,torch
from train import Editor
BASE=Path('/home/gcx/aster_hard_optimization_20261109')
def raw_ell(a):
 if 'ell' in a:return np.asarray(a['ell'],dtype='float64')
 if 'rawell' in a:return np.asarray(a['rawell'],dtype='float64')
 if 'q' in a:return np.asarray(a['q'],dtype='float64')[...,0]*10.
 raise KeyError('Supply ell/rawell or q with first coordinate ell/10')
class Predictor:
 def __init__(self,manifest=BASE/'frozen_methods.json'):
  torch.set_num_threads(2);self.manifest=json.loads(Path(manifest).read_text())
  self.entries={n:e for n,e in self.manifest['methods'].items() if n.startswith('hard_tail_')}
  self.models={}
  for e in self.entries.values():
   for member in e['members']:
    p=member['model_file']
    if p in self.models:continue
    s=torch.load(p,map_location='cpu',weights_only=False);m=Editor(len(s['mu']),s['cfg']['width'])
    m.load_state_dict(s['state_dict']);m.eval();self.models[p]=(m,s['mu'],s['scale'])
 def corrections_from_coordinates(self,a):
  ell=raw_ell(a);single=ell.ndim==0;ee=ell.reshape(-1).astype('float32')
  q=np.asarray(a['q'],dtype='float32').reshape(len(ee),3) if 'q' in a else np.column_stack([ee/10,np.abs(ee)/10,np.tanh(ee/4)]).astype('float32')
  x=np.concatenate([np.asarray(a['r32']).reshape(len(ee),32),np.asarray(a['residual64']).reshape(len(ee),64),q],axis=1).astype('float32')
  values={}
  for p,(m,mu,scale) in self.models.items():
   z=((x-mu)/scale).astype('float32')
   with torch.no_grad():values[p]=np.concatenate([m(torch.from_numpy(z[i:i+1024])).numpy() for i in range(0,len(z),1024)]).astype('float64')
  result={}
  for name,e in self.entries.items():
   c=sum(float(k['coefficient'])*values[k['model_file']] for k in e['members'])*float(e.get('gain',1.))
   result[name]=float(c[0]) if single else c
  return result
 def predict_from_coordinates(self,a):
  ell=raw_ell(a)
  return {name:ell+correction+self.entries[name]['b'] for name,correction in self.corrections_from_coordinates(a).items()}
_predictor=None
def predict_from_coordinates(a,manifest=BASE/'frozen_methods.json'):
 global _predictor
 if _predictor is None:_predictor=Predictor(manifest)
 return _predictor.predict_from_coordinates(a)

