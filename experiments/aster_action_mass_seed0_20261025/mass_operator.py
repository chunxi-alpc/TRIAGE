"""Action-group attention redistribution; exact masks, bounded query chunks."""
import torch
import torch.nn.functional as F

def action_scores(c,p):
 actions=list(dict.fromkeys(r['action'] for r in c['rules']))
 scores=[sum(p[r['rid']] for r in c['rules'] if r['action']==a) for a in actions]
 return actions,scores

def group_roles(c,roles):
 actions,_=action_scores(c,[0.]*len(c['rules']))
 lookup={r['rid']:actions.index(r['action']) for r in c['rules']}
 return [lookup[r['rules'][0]] if r and r['kind'] in ('action','option') and r['rules'] else -1 for r in roles]

def redistribute(prob,groups,s,alpha,allowed):
 """Preserve nonaction entries and total union mass, per head/query.
 Orphan options have group -1. Renormalize S only over causally visible groups.
 """
 if alpha==0:return prob
 if len(s)==0:return prob
 is_action=groups>=0
 index=groups.clamp_min(0).expand_as(prob)
 shape=(*prob.shape[:-1],len(s))
 mass=torch.zeros(shape,device=prob.device,dtype=prob.dtype)
 mass.scatter_add_(-1,index,prob*is_action)
 count=torch.zeros_like(mass)
 count.scatter_add_(-1,index,(allowed&is_action).to(prob.dtype))
 target=(count>0).to(prob.dtype)*s
 denom=target.sum(-1,keepdim=True)
 target=target/denom.clamp_min(torch.finfo(prob.dtype).tiny)
 desired=(1-alpha)*mass+alpha*mass.sum(-1,keepdim=True)*target
 desired=torch.where(denom>0,desired,mass)
 m=mass.gather(-1,index)
 conditional=torch.where(m>0,prob/m.clamp_min(torch.finfo(prob.dtype).tiny),allowed/count.gather(-1,index).clamp_min(1))
 return torch.where(is_action,conditional*desired.gather(-1,index),prob)

def attention(q,k,v,mask,scale,groups,s,alpha=0.,bias=0.,chunk=256):
 if q.shape[1]!=k.shape[1]:
  repeat=q.shape[1]//k.shape[1];k=k.repeat_interleave(repeat,1);v=v.repeat_interleave(repeat,1)
 outputs=[];n=q.shape[-2];nk=k.shape[-2]
 groups=torch.as_tensor(groups,device=q.device);s=torch.as_tensor(s,device=q.device,dtype=torch.float32)
 for lo in range(0,n,chunk):
  hi=min(n,lo+chunk);scores=torch.matmul(q[...,lo:hi,:].float(),k.float().transpose(-1,-2))*scale
  if mask is None:
   allowed=torch.arange(nk,device=q.device)[None,:]<=torch.arange(lo,hi,device=q.device)[:,None]
  else:
   m=mask[...,lo:hi,:nk] if mask.shape[-2]!=1 else mask[...,:nk]
   if m.dtype==torch.bool:allowed=m
   else:
    allowed=torch.isfinite(m)&(m>-1e4);scores=scores+m.float()
  allowed=allowed.expand_as(scores)
  scores=scores.masked_fill(~allowed,float('-inf'))
  if bias:
   weights=torch.zeros(nk,device=q.device)
   for a in range(len(s)):weights[groups==a]=bias*(s[a].clamp_min(1e-12).log()-s.max().clamp_min(1e-12).log())
   scores=scores+weights
  prob=scores.softmax(-1).nan_to_num(0.)
  if alpha:prob=redistribute(prob,groups,s,alpha,allowed)
  outputs.append(torch.matmul(prob,v.float()).to(v.dtype))
 return torch.cat(outputs,-2)
