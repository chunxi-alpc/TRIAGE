"""Full-query staged masks. No cross-intervention KV reuse."""
import contextlib,importlib,math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
class Engine:
 def __init__(self,path):
  torch.set_num_threads(4);torch.manual_seed(0)
  self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
  self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval()
  self.mod=importlib.import_module('transformers.models.'+self.model.config.model_type+'.modeling_'+self.model.config.model_type)
  self.nl=self.model.config.num_hidden_layers
 def render(self,c,form='full',selected=None):
  text='The supplied rules are the sole authority. Select the applicable rule using patient conditions.\n';spans=[]
  for r in c['rules']:
   if form=='crop' and r['rid']!=selected:continue
   text+=f"R{r['rid']+1} Condition: {r['condition']}\n"
   if form=='hidden' or (form=='crop_action' and r['rid']!=selected):continue
   a=len(text);text+='Action: '+r['action']+'\n';spans.append((a,len(text),[r['rid']]))
  text+='Patient: '+c['patient']+'\nQuestion: Which action follows from the supplied rules?\n'
  if form!='hidden':
   for j,opt in enumerate(c['options']):
    a=len(text);text+=f'{chr(65+j)}. {opt}\n';spans.append((a,len(text),[r['rid'] for r in c['rules'] if r['action']==opt]))
  text+='Select the applicable rule.\n'
  rendered=self.tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  start=rendered.index(text.rstrip());spans=[(a+start,b+start,rs) for a,b,rs in spans]
  enc=self.tok(rendered,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];roles=[]
  for a,b in enc['offset_mapping']:
   matches=[rs for x,y,rs in spans if b>x and a<y];roles.append(matches[0] if matches else None)
  # Every token overlapping an action/option line is protected, including labels.
  return ids,roles
 @contextlib.contextmanager
 def intervene(self,roles,phase_start,mode,p=None,lam=1.,layers='all',capture=None):
  old=self.mod.eager_attention_forward;allowed=set(range(self.nl)) if layers=='all' else set(range(self.nl//2,self.nl))
  def wrapped(module,q,k,v,attention_mask,scaling,dropout=0.,**kw):
   n=q.shape[-2];device=q.device
   if capture is not None:capture[module.layer_idx]=k.detach().float().cpu()
   mask=attention_mask[...,:k.shape[-2]].clone() if attention_mask is not None else torch.zeros((1,1,n,n),dtype=q.dtype,device=device).masked_fill(torch.ones(n,n,device=device,dtype=torch.bool).triu(1),float('-inf'))
   if module.layer_idx in allowed and mode!='none':
    act=torch.tensor([r is not None for r in roles],device=device);judge=torch.arange(n,device=device)<phase_start
    blocked=(judge & ~act)[:,None]&act[None,:]
    if mode in ['blind','gate','hard','strict']:mask.masked_fill_(blocked[None,None],-1e4)
    if mode in ['gate','hard','strict']:
     assert p is not None
     weights=torch.zeros(n,device=device,dtype=q.dtype);best=max(p)
     winner=max(range(len(p)),key=lambda j:p[j])
     for j,rs in enumerate(roles):
      if rs:
       weights[j]=(-1e4 if winner not in rs else 0.) if mode in ['hard','strict'] else lam*(math.log(max(p[r] for r in rs))-math.log(best))
     if mode=='strict':mask+=weights[None,None,None,:]
     else:mask+=(~judge)[None,None,:,None]*weights[None,None,None,:]
   groups=q.shape[1]//k.shape[1];k=k.repeat_interleave(groups,dim=1);v=v.repeat_interleave(groups,dim=1)
   output=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=scaling)
   return output.transpose(1,2).contiguous(),None
  self.mod.eager_attention_forward=wrapped
  try:yield
  finally:self.mod.eager_attention_forward=old
 def forward(self,ids,roles,phase,mode='none',p=None,lam=1.,layers='all',capture=None,keep=1,positions=None):
  x=torch.tensor([ids],device='cuda');kwargs={}
  if positions is not None:kwargs['position_ids']=torch.tensor([positions],device='cuda')
  with self.intervene(roles,phase,mode,p,lam,layers,capture),torch.inference_mode():
   return self.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=keep,**kwargs).logits[0].float()
 def branch(self,c,hidden=False,blind=False):
  ids,roles=self.render(c,'hidden' if hidden else 'full');prefix=self.tok.encode('Applicable rule:',add_special_tokens=False);ids+=prefix;roles += [None]*len(prefix);scores=[]
  for r in c['rules']:
   suffix=self.tok.encode(f" R{r['rid']+1}",add_special_tokens=False)
   logits=self.forward(ids+suffix,roles+[None]*len(suffix),len(ids)+len(suffix),'blind' if blind else 'none',keep=len(suffix)+1)
   lp=logits[:-1].log_softmax(-1);scores.append(float(lp[range(len(suffix)),suffix].sum()))
  return torch.tensor(scores).softmax(0).tolist()
 def answer(self,c,form='full',mode='none',p=None,lam=1.,layers='all'):
  selected=c['gold_rule'];ids,roles=self.render(c,form,selected)
  prefix=self.tok.encode(f'Applicable rule: R{selected+1}',add_special_tokens=False);ids+=prefix;roles += [None]*len(prefix);phase=len(ids)
  suffix=self.tok.encode('\nAnswer:',add_special_tokens=False);ids+=suffix;roles += [None]*len(suffix)
  logits=self.forward(ids,roles,phase,mode,p,lam,layers)[-1]
  labels=[self.tok.encode(' '+chr(65+i),add_special_tokens=False) for i in range(len(c['options']))]
  if any(len(x)!=1 for x in labels):raise ValueError('Multi-token option label requires sequence scoring')
  scores=[float(logits[x[0]]) for x in labels];return max(range(len(scores)),key=lambda j:scores[j]),scores
