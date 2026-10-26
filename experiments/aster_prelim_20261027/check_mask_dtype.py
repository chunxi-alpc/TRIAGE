"""Regression check for real Engine.mask using SDPA, without loading a model."""
import json
from pathlib import Path
import torch
from attention_engine import Engine

torch.manual_seed(20261027)
engine=Engine.__new__(Engine)
engine.state={'roles':torch.tensor([False,True,False,False,False,False]),'branches':torch.tensor([-1,0,-1,-1,-1,-1]),'segments':torch.tensor([-1,0,-1,-1,-1,-1]),'phase':'blind','selected':None}
q=torch.randn(1,2,6,4);k=torch.randn(1,2,6,4);v=torch.randn(1,2,6,4);changed=v.clone();changed[:,:,1]+=10
allow=torch.ones(6,6,dtype=torch.bool).tril()[None,None]
checks=[]
for kind in ['bool','additive']:
 incoming=allow.clone() if kind=='bool' else torch.zeros_like(allow,dtype=torch.float32).masked_fill_(~allow,float('-inf'))
 _,kw=engine.mask(None,(),{'hidden_states':torch.zeros(1,6,4),'attention_mask':incoming})
 mask=kw['attention_mask'];assert mask.dtype.is_floating_point and mask[0,0,-1,1]<-1e4
 y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
 z=torch.nn.functional.scaled_dot_product_attention(q,k,changed,attn_mask=mask)
 diff=float((y[:,:,-1]-z[:,:,-1]).abs().max());assert diff==0
 plain=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=allow)
 plain2=torch.nn.functional.scaled_dot_product_attention(q,k,changed,attn_mask=allow)
 positive=float((plain[:,:,-1]-plain2[:,:,-1]).abs().max());assert positive>0
 checks.append({'input_mask_kind':kind,'masked_value_perturbation_last_query_difference':diff,'unmasked_positive_control_difference':positive})
result={'status':'PASS','scope':'real Engine.mask + PyTorch SDPA boolean/additive semantics; synthetic CPU regression, not P4 model performance','checks':checks}
(Path(__file__).parent/'results/mask_dtype_regression.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
