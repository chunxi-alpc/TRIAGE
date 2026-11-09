
"""Apply the fit-only hard-CF residual to an already scored frozen balanced model."""
from pathlib import Path
import numpy as np
DEFAULT_MODEL=Path(__file__).resolve().parent/'frozen_model.npz'
_CACHE={}
def predict_from_coordinates(a,ell,base_balanced_score,model_path=DEFAULT_MODEL):
 """a['r32'] uses original cached coordinates; base score includes its old b."""
 key=str(model_path)
 if key not in _CACHE:
  with np.load(model_path) as f:_CACHE[key]={k:f[k] for k in f.files}
 model=_CACHE[key]
 coordinates=np.asarray(a['r32'],dtype='float64')
 residual=((coordinates-model['mu'])/model['scale'])@model['w']
 # The caller's base includes the original .436452 gain already; do not rescale.
 base_delta=np.asarray(base_balanced_score,dtype='float64')-np.asarray(ell,dtype='float64')-float(model['base_b'])
 score=np.asarray(ell,dtype='float64')+base_delta+float(model['amplitude'])*residual+float(model['b'])
 return float(score) if score.ndim==0 else score

