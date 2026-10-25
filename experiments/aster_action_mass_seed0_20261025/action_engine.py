import contextlib,sys
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'aster_e0_e8_20261024'))
from engine import Engine
from mass_operator import attention,action_scores,group_roles

class ActionEngine(Engine):
 def render(self,c,form='full',selected=None,**kw):
  if form=='action_crop':
   # All conditions and original option set remain. Keep every rule sharing winner action.
   cc={**c,'rules':[{**r,'action':r['action'] if r['action']==selected else '[action omitted]'} for r in c['rules']]}
   return super().render(cc,'full',**kw)
  return super().render(c,form,selected,**kw)
 def answer(self,c,*args,**kw):
  self.current_case=c
  return super().answer(c,*args,**kw)
 @contextlib.contextmanager
 def intervene(self,roles,phase_start,mode,p=None,lam=1.,layers='all',capture=None,**controls):
  if mode not in ('mass','action_bias'):
   with super().intervene(roles,phase_start,mode,p,lam,layers,capture,**controls):yield
   return
  old=self.mod.eager_attention_forward
  groups=group_roles(self.current_case,roles);_,s=action_scores(self.current_case,p)
  allowed=set(range(self.nl)) if layers=='all' else set(range(self.nl//2,self.nl))
  def wrapped(module,q,k,v,attention_mask,scaling,dropout=0.,**kw):
   if module.layer_idx not in allowed:
    repeat=q.shape[1]//k.shape[1]
    output=F.scaled_dot_product_attention(q,k.repeat_interleave(repeat,1),v.repeat_interleave(repeat,1),attn_mask=attention_mask,dropout_p=0.,scale=scaling)
    return output.transpose(1,2).contiguous(),None
   output=attention(q,k,v,attention_mask,scaling,groups,s,alpha=lam if mode=='mass' else 0.,bias=lam if mode=='action_bias' else 0.)
   return output.transpose(1,2).contiguous(),None
  self.mod.eager_attention_forward=wrapped
  try:yield
  finally:self.mod.eager_attention_forward=old
