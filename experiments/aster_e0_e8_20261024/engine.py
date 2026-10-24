"""Independent blind judging and attention gating; no implicit oracle access."""
import contextlib, importlib, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class Engine:
 def __init__(self,path):
  torch.set_num_threads(4); torch.manual_seed(0)
  self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
  self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval()
  self.mod=importlib.import_module('transformers.models.'+self.model.config.model_type+'.modeling_'+self.model.config.model_type)
  self.nl=self.model.config.num_hidden_layers
 def render(self,c,form='full',selected=None,order='rules_first',repeat=1):
  if form in ('crop','crop_action') and selected is None: raise ValueError('Cropping requires an explicit selected rule')
  text='The supplied rules are the sole authority. Select the applicable rule using patient conditions.\n'; spans=[]
  patient='Patient: '+c['patient']+'\n'
  if order=='patient_first': text+=patient
  for r in c['rules']:
   if form=='crop' and r['rid']!=selected: continue
   a=len(text); text+=f"R{r['rid']+1} Condition: {r['condition']}\n"; spans.append((a,len(text),'condition',[r['rid']]))
   if form=='hidden' or (form=='crop_action' and r['rid']!=selected): continue
   a=len(text); text+='Action: '+r['action']+'\n'; spans.append((a,len(text),'action',[r['rid']]))
  if order!='patient_first': text+=patient
  text+='Question: Which action follows from the supplied rules?\n'
  if form!='hidden':
   for j,opt in enumerate(c['options']):
    a=len(text);text+=f'{chr(65+j)}. {opt}\n';spans.append((a,len(text),'option',[r['rid'] for r in c['rules'] if r['action']==opt]))
  text+='Select the applicable rule using patient conditions.\n'*repeat
  rendered=self.tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  start=rendered.index(text.rstrip()); enc=self.tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
  roles=[]
  for a,b in enc['offset_mapping']:
   matches=[{'kind':kind,'rules':rs} for x,y,kind,rs in spans if b>x+start and a<y+start]
   roles.append(matches[0] if matches else None)
  return enc['input_ids'],roles
 @staticmethod
 def role(r):
  if isinstance(r,dict): return r['kind'],r['rules']
  return ('action',r) if r is not None else (None,[])
 @contextlib.contextmanager
 def intervene(self,roles,phase_start,mode,p=None,lam=1.,layers='all',capture=None,*,blind=False,blind_layers='all',blind_phase='judge',gate_phase='answer',gate_scope='action',gate_options=False):
  old=self.mod.eager_attention_forward
  def layer_set(value):
   if value=='all':return set(range(self.nl))
   if value=='last_half':return set(range(self.nl//2,self.nl))
   return set(value)
  gate_allowed=layer_set(layers); blind_allowed=layer_set(blind_layers)
  kinds=[self.role(r)[0] for r in roles]; rids=[self.role(r)[1] for r in roles]
  def wrapped(module,q,k,v,attention_mask,scaling,dropout=0.,**kw):
   n=q.shape[-2];device=q.device
   if capture is not None:capture[module.layer_idx]=k.detach().float().cpu()
   mask=attention_mask[...,:k.shape[-2]].clone() if attention_mask is not None else torch.zeros((1,1,n,n),dtype=q.dtype,device=device).masked_fill(torch.ones(n,n,device=device,dtype=torch.bool).triu(1),float('-inf'))
   judge=torch.arange(n,device=device)<phase_start
   if (blind or mode=='blind') and module.layer_idx in blind_allowed:
    act=torch.tensor([kind in ('action','option') for kind in kinds],device=device)
    queries=torch.ones(n,device=device,dtype=torch.bool) if blind_phase=='all' else judge
    blocked=(queries & ~act)[:,None]&act[None,:]
    mask.masked_fill_(blocked[None,None],-1e4)
   if mode in ('gate','hard','strict') and module.layer_idx in gate_allowed:
    if p is None:raise ValueError('Gating requires branch probabilities')
    weights=torch.zeros(n,device=device,dtype=q.dtype);best=max(p);winner=max(range(len(p)),key=lambda j:p[j])
    for j,(kind,rs) in enumerate(zip(kinds,rids)):
     target=(kind=='action' or (kind=='condition' and gate_scope=='rule') or (kind=='option' and gate_options))
     if target and rs:
      weights[j]=(-1e4 if winner not in rs else 0.) if mode in ('hard','strict') else lam*(math.log(max(max(p[r] for r in rs),1e-12))-math.log(max(best,1e-12)))
    if gate_phase=='all' or mode=='strict': mask+=weights[None,None,None,:]
    else:mask+=(~judge)[None,None,:,None]*weights[None,None,None,:]
   groups=q.shape[1]//k.shape[1];k=k.repeat_interleave(groups,dim=1);v=v.repeat_interleave(groups,dim=1)
   output=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=scaling)
   return output.transpose(1,2).contiguous(),None
  self.mod.eager_attention_forward=wrapped
  try:yield
  finally:self.mod.eager_attention_forward=old
 def forward(self,ids,roles,phase,mode='none',p=None,lam=1.,layers='all',capture=None,keep=1,positions=None,**controls):
  x=torch.tensor([ids],device='cuda');kwargs={}
  if positions is not None:kwargs['position_ids']=torch.tensor([positions],device='cuda')
  with self.intervene(roles,phase,mode,p,lam,layers,capture,**controls),torch.inference_mode():
   return self.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=keep,**kwargs).logits[0].float()
 def branch(self,c,hidden=False,blind=False,order='rules_first',repeat=1):
  ids,roles=self.render(c,'hidden' if hidden else 'full',order=order,repeat=repeat)
  prefix=self.tok.encode('Applicable rule:',add_special_tokens=False);ids+=prefix;roles += [None]*len(prefix)
  cache={};scores=[]
  for r in c['rules']:
   suffix=self.tok.encode(f" R{r['rid']+1}",add_special_tokens=False);score=0.
   for j,token in enumerate(suffix):
    key=tuple(suffix[:j])
    if key not in cache:
     cache[key]=self.forward(ids+list(key),roles+[None]*len(key),len(ids)+len(key),'blind' if blind else 'none')[-1].log_softmax(-1)
    score+=float(cache[key][token])
   scores.append(score)
  return torch.tensor(scores).softmax(0).tolist()
 def answer(self,c,form='full',mode='none',p=None,lam=1.,layers='all',*,selected=None,write_rule=False,**controls):
  # gold_rule and gold_option are never read here. Oracle callers pass selected explicitly.
  ids,roles=self.render(c,form,selected)
  if write_rule:
   if selected is None:raise ValueError('A written rule requires an explicit selection')
   prefix=self.tok.encode(f'Applicable rule: R{selected+1}',add_special_tokens=False);ids+=prefix;roles += [None]*len(prefix)
  phase=len(ids);suffix=self.tok.encode('\nAnswer:',add_special_tokens=False);ids+=suffix;roles += [None]*len(suffix)
  labels=[self.tok.encode(' '+chr(65+i),add_special_tokens=False) for i in range(len(c['options']))]
  if any(len(x)!=1 for x in labels):raise ValueError('Multi-token option label requires sequence scoring')
  logits=self.forward(ids,roles,phase,mode,p,lam,layers,**controls)[-1]
  scores=[float(logits[x[0]]) for x in labels];return max(range(len(scores)),key=lambda j:scores[j]),scores
 @staticmethod
 def direct_map(c,p):
  scores=[sum(p[r['rid']] for r in c['rules'] if r['action']==opt) for opt in c['options']]
  if sum(scores)<=0:raise ValueError('No branch action maps to an option')
  return max(range(len(scores)),key=lambda j:scores[j]),scores
