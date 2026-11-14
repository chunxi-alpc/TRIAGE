"""Native FlexAttention controls for full, unpadded ASTER sequences."""
import contextlib,math
import torch
from torch.nn.attention.flex_attention import flex_attention,create_block_mask
from engine import Engine
# Different sequence shapes and control policies need distinct compiled graphs.
torch._dynamo.config.cache_size_limit=128
compiled_flex=torch.compile(flex_attention,dynamic=False)
def controls(roles,phase,mode,p=None,lam=1.,blind=False,blind_phase='judge',gate_phase='answer',gate_scope='action',gate_options=False,device='cuda'):
 n=len(roles);size=(n+127)//128*128
 action=torch.zeros(size,dtype=torch.bool,device=device);bad=action.clone();bias=torch.zeros(size,device=device)
 gate=mode in ('hard','gate','strict');hard=mode in ('hard','strict')
 if gate and p is None:raise ValueError('Probabilities required')
 winner=max(range(len(p)),key=lambda j:p[j]) if gate else None
 for i,r in enumerate(roles):
  kind,rs=Engine.role(r);action[i]=kind in ('action','option')
  target=bool(rs) and (kind=='action' or (kind=='condition' and gate_scope=='rule') or (kind=='option' and gate_options))
  if gate and target:
   bad[i]=winner not in rs;bias[i]=lam*(math.log(max(max(p[j] for j in rs),1e-12))-math.log(max(max(p),1e-12)))
 blind=blind or mode=='blind';allgate=gate_phase=='all' or mode=='strict'
 def mask_mod(b,h,q,k):
  allowed=(q>=k)&(q<n)&(k<n)
  if blind:
   judge=(q<n) if blind_phase=='all' else q<phase
   allowed=allowed & ~(judge & ~action[q] & action[k])
  if gate and hard:
   active=(q<n) if allgate else q>=phase
   allowed=allowed & ~(active & bad[k])
  return allowed
 def score_mod(score,b,h,q,k):
  if gate and not hard:return score+torch.where((q<n) if allgate else q>=phase,bias[k],0.)
  return score
 return score_mod,create_block_mask(mask_mod,None,None,n,n,device=device)
class FlexEngine(Engine):
 @contextlib.contextmanager
 def intervene(self,roles,phase_start,mode,p=None,lam=1.,layers='all',capture=None,*,blind=False,blind_layers='all',blind_phase='judge',gate_phase='answer',gate_scope='action',gate_options=False):
  old=self.mod.eager_attention_forward;cache={}
  def active(i,x):return x=='all' or (x=='last_half' and i>=self.nl//2) or (not isinstance(x,str) and i in x)
  def wrapped(module,q,k,v,attention_mask,scaling,dropout=0.,**kw):
   if q.shape[0]!=1 or q.shape[-2]!=len(roles) or k.shape[-2]!=len(roles):raise ValueError('Only single unpadded full sequences are supported')
   if capture is not None:capture[module.layer_idx]=k.detach().float().cpu()
   g=active(module.layer_idx,layers) and mode in ('gate','hard','strict');b=active(module.layer_idx,blind_layers) and (blind or mode=='blind')
   if (g,b) not in cache:cache[g,b]=controls(roles,phase_start,mode if g else 'none',p,lam,blind=b,blind_phase=blind_phase,gate_phase=gate_phase,gate_scope=gate_scope,gate_options=gate_options,device=q.device)
   score,block=cache[g,b]
   # Engine.forward uses attention_mask=ones: only causal structure is needed.
   out=compiled_flex(q,k,v,score_mod=score,block_mask=block,scale=scaling,enable_gqa=q.shape[1]!=k.shape[1])
   return out.transpose(1,2).contiguous(),None
  self.mod.eager_attention_forward=wrapped
  try:yield
  finally:self.mod.eager_attention_forward=old
