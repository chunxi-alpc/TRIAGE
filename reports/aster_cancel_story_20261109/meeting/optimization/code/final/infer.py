
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,importlib.util
import numpy as np,torch
BASE=Path('/home/gcx/aster_optimization_20261109')
OUT=BASE/'final'
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def main():
 torch.set_num_threads(2)
 frozen=json.loads((OUT/'selected.json').read_text())
 if not frozen.get('frozen') or frozen.get('selection_split')!='dev':raise RuntimeError('Selection not frozen')
 a=dict(np.load(BASE/'shared/features.npz'));old=np.load(OLD/'coordinates.npz');ids=old['ids'];n=len(ids)
 a={k:v[:n] for k,v in a.items()};ell=a['ell']
 lin=dict(np.load(BASE/'linear/feature_coordinates.npz'))
 neural=module('neural_model',BASE/'neural/search.py')
 score=module('score_model',BASE/'score_conditioned/search.py')
 memo={}
 def predict(member):
  key=(member['kind'],member['model_file'])
  if key in memo:return memo[key]
  if member['kind']=='linear':
   cfg=member['config'];p=np.load(member['model_file']);x=lin[cfg['family']]
   y=((x-p['mu'])/p['scale'])@p['w']
  else:
   s=torch.load(member['model_file'],map_location='cpu',weights_only=False)
   if member['kind']=='neural':
    cfg=s['config'];model=neural.Editor(len(s['mu']),cfg['width'],cfg['depth']);x=a[cfg['feature']]
   elif member['kind']=='scorecond':
    cfg=s['cfg'];model=score.Editor(len(s['mu']),cfg['width']);x=score.features(cfg['family'],a,lin)
   else:raise ValueError(member['kind'])
   model.load_state_dict(s['state_dict']);model.eval();z=((x-s['mu'])/s['scale']).astype('float32')
   with torch.no_grad():y=np.concatenate([model(torch.from_numpy(z[i:i+2048])).numpy() for i in range(0,n,2048)]).astype('float64')
  memo[key]=y
  return y
 scores={'ids':ids,'dev_path_calibration':ell+frozen['calibration_control']['b']}
 for name,entry in frozen['methods'].items():
  if entry['kind']=='ensemble':
   delta=sum(float(member['coefficient'])*predict(member) for member in entry['members'])
  else:delta=predict(entry)
  scores[name]=ell+entry.get('gain',1.)*delta+entry['b']
 np.savez_compressed(OUT/'scores.npz',**scores)
 print('Frozen methods scored: '+str(len(scores)-1))
if __name__=='__main__':main()

