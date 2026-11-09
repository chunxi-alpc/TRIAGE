"""Frozen new tail-loss ensembles plus the unchanged original-method controls."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path('/home/gcx/aster_hard_optimization_20261109')
OLD=Path('/home/gcx/aster_next_20261109')
sys.path.insert(0,str(OLD/'inference'))
from frozen import FrozenMethods
class HardMethods:
 def __init__(self,manifest):
  config=json.loads(Path(manifest).read_text())
  if not config.get('frozen_before_error_retest'):raise ValueError('New evaluation configuration must be frozen before use')
  self.base=FrozenMethods(config['original_manifest']);self.heads=self.base.heads
  # The new predictor imports its training architecture from this new directory.
  sys.path.insert(0,str(ROOT))
  from predict import Predictor
  self.new=Predictor(config['new_mlp_manifest'])
  self.names=list(config['methods'])
 def predict(self,ell,selected_heads,residual):
  scores=self.base.predict(ell,selected_heads,residual)
  a=self.base.coordinates(ell,selected_heads,residual);a['ell']=float(ell)
  scores.update({k:float(v) for k,v in self.new.predict_from_coordinates(a).items()})
  return scores
 def predict_file(self,record):
  with np.load(record['feature']) as arrays:
   return self.predict(record['ell'],self.base.raw_selected(arrays),arrays['residual'])
