
"""Pure CPU inference for the selected task-supervised NLI residual readout."""
from pathlib import Path
import numpy as np
DEFAULT_MODEL=Path('/home/gcx/aster_hard_optimization_20261109/nli_adaptation/selected_model.npz')
SUPPORT=Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen/supports_r16.npz')
def load_model(path=DEFAULT_MODEL):
 with np.load(path,allow_pickle=False) as arrays:return {k:arrays[k] for k in arrays.files}
def predict_from_r16(ell,r16,model):
 """score = ell + ((r16-mu)/scale) @ w + b; no fitting."""
 if str(model['family'])!='trace16':raise ValueError('The frozen selected model must use trace16')
 return np.asarray(ell,dtype=float)+((np.asarray(r16,dtype=float)-model['mu'])/model['scale'])@model['w']+float(model['b'])
def predict_file(record,model,support=None):
 support=dict(np.load(SUPPORT)) if support is None else support
 with np.load(record['feature']) as arrays:
  if 'selected_heads' in arrays:raw=arrays['selected_heads'].reshape(-1).astype('float32')
  else:
   a=arrays['heads'];raw=a.reshape(-1,a.shape[-1])[support['heads']].reshape(-1).astype('float32')
 r16=((raw-support['mu'])/support['scale'])@support['N']
 return float(predict_from_r16(record['ell'],r16,model))

