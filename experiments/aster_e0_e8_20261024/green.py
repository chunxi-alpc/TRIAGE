"""Attach the inspected official SEKA hook to an already loaded model.
No identity/random projection fallback. Requires matching per-model projection.
"""
import contextlib,hashlib,importlib.util,sys
from pathlib import Path
import torch
@contextlib.contextmanager
def official_green(engine,roles,selected,projection,layers='last10',gain=1.,root='/home/gcx/TRIAGE/third_party/SEKA'):
 path=Path(projection)
 if not path.is_file():raise FileNotFoundError('Matching official SEKA projection unavailable')
 # Use the upstream implementation, including k_norm vs k_proj and pre-RoPE placement.
 sys.path.insert(0,root)
 try:
  from src.model.seka_llm import SEKALLM
  obj=SEKALLM.__new__(SEKALLM);obj.model=engine.model;obj._hooks=[];obj.pos_pt=str(path);obj.neg_pt=None;obj.layers=layers;obj.amplify_pos=gain;obj.amplify_neg=0.;obj.feature_function=None
  mask=torch.tensor([engine.role(rs)[0] in ('action','option') and selected in engine.role(rs)[1] for rs in roles],device=engine.model.device,dtype=torch.bool)
  obj.attach_projection(steer_mask_tensor=mask,silence=True)
  try:yield
  finally:obj.remove_projection()
 finally:sys.path.pop(0)
