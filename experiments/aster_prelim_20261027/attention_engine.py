"""Explicit role-separated token serialization and two-phase attention controls.

Development implementation for P4/P5, not a validated generation system.
"""
import math
import torch

class Engine:
 def __init__(self,model,tokenizer):
  self.model=model;self.tok=tokenizer;self.state={};self.handles=[]
  for layer in model.model.layers:self.handles.append(layer.self_attn.register_forward_pre_hook(self.mask,with_kwargs=True))
 def pack(self,row,task='branch',extra_patient=''):
  chunks=[]
  def add(text,role=0,branch=-1):chunks.append((text,role,branch))
  add('Experimental branch rules:\n')
  for q in row['rules']:
   b=int(q['bid'])-1;add(f"Branch {q['bid']}: IF {q['condition']}\nAction: ");add(q['action'],1,b);add('\n')
  guide_chunks=len(chunks)
  add('\nPatient:\n'+row['patient']+extra_patient+'\n')
  if task=='action':
   add('\nTreatment options:\n')
   for i,value in enumerate(row['options']):
    matches=[int(q['bid'])-1 for q in row['rules'] if q['action']==value]
    add(chr(65+i)+': '+value+'\n',1,-1)
   add('\nAccording to the experimental rules and patient conditions, which treatment option applies? Output only the option letter.')
  else:add('\nWhich branch condition applies to the patient? Output only 1 or 2.')
  marker='__ASTER_TYPED_INPUT_20261027__'
  full=self.tok.apply_chat_template([{'role':'system','content':'Follow the supplied experimental rules. Return only the requested identifier.'},{'role':'user','content':marker}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  prefix,suffix=full.split(marker);ids=[];roles=[];branches=[];segments=[];segment_counter=[0]
  def extend(t,r,b):
   x=self.tok.encode(t,add_special_tokens=False);ids.extend(x);roles.extend([r]*len(x));branches.extend([b]*len(x));segments.extend([segment_counter[0] if r else -1]*len(x));segment_counter[0]+=int(bool(r))
  extend(prefix,0,-1)
  guide_end=None
  for i,(t,r,b) in enumerate(chunks):
   extend(t,r,b)
   if i+1==guide_chunks:guide_end=len(ids)
  extend(suffix,0,-1)
  return {'ids':ids,'roles':roles,'branches':branches,'guide_end':guide_end,'positions':list(range(len(ids))),'segments':segments}
 def mask(self,module,args,kw):
  st=self.state
  if not st or st.get('plain'):return
  h=kw['hidden_states'] if 'hidden_states' in kw else args[0];l=h.shape[1]
  mask=kw.get('attention_mask')
  if mask is None:
   mask=torch.zeros((1,1,l,l),device=h.device,dtype=h.dtype)
   mask.masked_fill_(torch.ones((l,l),device=h.device,dtype=torch.bool).triu(1),torch.finfo(h.dtype).min)
  elif mask.dtype==torch.bool:mask=torch.zeros_like(mask,dtype=h.dtype).masked_fill_(~mask,torch.finfo(h.dtype).min)
  else:mask=mask.clone()
  act=st['roles'];branch=st['branches'];segment=st['segments'];minimum=torch.finfo(h.dtype).min
  # Blind judgment isolates actions. Execution is a fresh causal pass:
  # gate competitor action keys at every query to prevent indirect rereading.
  if st['phase']=='blind' or st.get('isolated'):
   blocked=(~act)[:,None]&act[None,:]
   blocked|=act[:,None]&act[None,:]&(segment[:,None]!=segment[None,:])
   if st['phase']!='blind':blocked[-1,:]=False
  else:blocked=torch.zeros((l,l),device=h.device,dtype=torch.bool)
  if st['phase']!='blind' and st.get('selected') is not None:
   competitor=act&(branch>=0)&(branch!=st['selected'])
   if st.get('hard'):blocked[:,competitor]=True
   else:mask[0,0,:,competitor]+=st.get('lam',0)*math.log(st.get('floor',1e-4))
  mask[0,0].masked_fill_(blocked,minimum);kw['attention_mask']=mask;return args,kw
 def score(self,packed,phase='blind',selected=None,hard=False,prune=False,preserve_positions=False,lam=0,plain=False,isolated=False,compact_hard_positions=False):
  ids=packed['ids'];roles=packed['roles'];branches=packed['branches'];positions=packed['positions'];keep=list(range(len(ids)))
  if prune:
   keep=[i for i in keep if not (roles[i] and (phase=='blind' or branches[i]>=0 and branches[i]!=selected))]
  x=torch.tensor([[ids[i] for i in keep]],device='cuda');ar=torch.tensor([bool(roles[i]) for i in keep],device='cuda');br=torch.tensor([branches[i] for i in keep],device='cuda')
  if phase=='blind':pos=(~ar).long().cumsum(0).sub(1).clamp_min(0)[None,:]
  elif hard and compact_hard_positions and selected is not None:
   retained=~(ar&(br>=0)&(br!=selected))
   pos=retained.long().cumsum(0).sub(1).clamp_min(0)[None,:]
  elif preserve_positions:pos=torch.tensor([[positions[i] for i in keep]],device='cuda')
  else:pos=torch.arange(len(keep),device='cuda')[None,:]
  self.state={'roles':ar,'branches':br,'segments':torch.tensor([packed['segments'][i] for i in keep],device='cuda'),'phase':phase,'selected':selected,'hard':hard,'lam':lam,'plain':plain,'isolated':isolated}
  with torch.inference_mode():out=self.model(input_ids=x,attention_mask=torch.ones_like(x),position_ids=pos,use_cache=False,logits_to_keep=1)
  return out.logits[0,-1].float().detach().cpu()
 def choose(self,logits,labels):
  ids=[self.tok.encode(s,add_special_tokens=False) for s in labels];assert all(len(x)==1 for x in ids)
  scores=logits[[x[0] for x in ids]];probs=scores.softmax(0).tolist();return labels[int(scores.argmax())],probs
