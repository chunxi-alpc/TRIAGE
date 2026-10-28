"""Generation-safe additive attention bias, applied after RoPE and cache assembly."""
import math
import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
class GenerationGate:
 def __init__(self,model):
  self.model=model;self.original=ALL_ATTENTION_FUNCTIONS['sdpa'];self.bias=None;self.calls=0;self.nonzero_calls=0;self.key_hits=0
  ALL_ATTENTION_FUNCTIONS.register('sdpa',self.attention)
 def reset(self,bias=None):
  self.bias=bias;self.calls=self.nonzero_calls=self.key_hits=0
 def attention(self,module,q,k,v,mask,**kw):
  if self.bias is None:return self.original(module,q,k,v,mask,**kw)
  d=q.shape[-1];scale=kw.get('scaling') or d**-.5
  # q'=concat(q,1/scale,0...), k'=concat(k,bias,0...) gives q'k'*scale=qk*scale+bias.
  # Pad to a multiple of eight so fused SDPA remains available; cache tensors stay original-sized.
  extra=8;q2=F.pad(q,(0,extra));k2=F.pad(k,(0,extra));v2=F.pad(v,(0,extra))
  q2[...,d]=1./scale
  n=min(k.shape[-2],len(self.bias));values=self.bias[:n].to(k.device,k.dtype)
  k2[...,:n,d]=values
  self.calls+=1
  nz=int(torch.count_nonzero(values).item())
  if nz:self.nonzero_calls+=1;self.key_hits+=nz
  args=dict(kw);args['scaling']=scale
  out,weights=self.original(module,q2,k2,v2,mask,**args)
  return out[...,:d],weights
 def close(self):ALL_ATTENTION_FUNCTIONS.register('sdpa',self.original)
def check_equivalence(device='cpu'):
 torch.manual_seed(91);results={}
 for qn,kn,causal in [(9,9,True),(1,13,False)]:
  q=torch.randn(1,4,qn,128,device=device);k=torch.randn(1,4,kn,128,device=device);v=torch.randn_like(k);bias=torch.linspace(-2,0,kn,device=device)
  mask=bias[None,None,None,:].expand(1,1,qn,kn).clone()
  if causal:mask.masked_fill_(torch.ones(qn,kn,device=device,dtype=torch.bool).triu(1),float('-inf'))
  direct=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
  q2=F.pad(q,(0,8));k2=F.pad(k,(0,8));v2=F.pad(v,(0,8));q2[...,128]=128**.5;k2[...,128]=bias
  padded=F.scaled_dot_product_attention(q2,k2,v2,is_causal=causal,scale=128**-.5)[...,:128]
  err=(direct-padded).abs().max().item();assert err<2e-5,(qn,kn,err)
  results[f'q{qn}_k{kn}']=err
 return results
if __name__=='__main__':print(check_equivalence())
