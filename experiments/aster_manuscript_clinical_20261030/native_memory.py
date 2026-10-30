"""Exact full-attention equations with query-row chunks, for native baseline memory."""
from contextlib import contextmanager
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv,eager_attention_forward
@contextmanager
def query_chunked_attention(model,chunk_size=256):
 originals={}
 for layer,module in enumerate(model.model.layers):
  attn=module.self_attn;originals[layer]=attn.forward
  def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,cache_position=None,**kwargs):
   shape=hidden_states.shape[:-1];headshape=(*shape,-1,self.head_dim)
   q=self.q_norm(self.q_proj(hidden_states).view(headshape)).transpose(1,2)
   k=self.k_norm(self.k_proj(hidden_states).view(headshape)).transpose(1,2)
   v=self.v_proj(hidden_states).view(headshape).transpose(1,2)
   cos,sin=position_embeddings;q,k=apply_rotary_pos_emb(q,k,cos,sin)
   if past_key_values is not None:k,v=past_key_values.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
   if q.shape[-2]<=chunk_size:
    out,weights=eager_attention_forward(self,q,k,v,attention_mask,scaling=self.scaling,dropout=0.)
   else:
    kk=repeat_kv(k,self.num_key_value_groups);vv=repeat_kv(v,self.num_key_value_groups);parts=[]
    for start in range(0,q.shape[-2],chunk_size):
     end=min(start+chunk_size,q.shape[-2])
     scores=torch.matmul(q[:,:,start:end],kk.transpose(2,3))*self.scaling
     if attention_mask is not None:scores=scores+attention_mask[:,:,start:end,:kk.shape[-2]]
     w=torch.nn.functional.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
     parts.append(torch.matmul(w,vv).transpose(1,2).contiguous())
    out=torch.cat(parts,dim=1);weights=None
   return self.o_proj(out.reshape(*shape,-1).contiguous()),weights
  attn.forward=MethodType(forward,attn)
 try:yield
 finally:
  for layer,forward in originals.items():model.model.layers[layer].self_attn.forward=forward

