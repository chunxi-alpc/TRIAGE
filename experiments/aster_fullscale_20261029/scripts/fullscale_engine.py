"""Qwen action-blind applicability and signed key projection interventions."""
import json,re
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS,repeat_kv
class Engine:
 def __init__(self,model):
  self.tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
  self.model=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
  self.model.generation_config.temperature=None;self.model.generation_config.top_p=None;self.model.generation_config.top_k=None
  self.original=ALL_ATTENTION_FUNCTIONS['sdpa'];self.active=None;self.capture=None;self.captured={}
  ALL_ATTENTION_FUNCTIONS.register('sdpa',self.attention)
 def prompt(self,user):
  return self.tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def encode(self,text,offset=False):
  x=self.tok(text,return_tensors='pt',return_offsets_mapping=offset,add_special_tokens=False)
  offsets=x.pop('offset_mapping')[0].tolist() if offset else None
  if x['input_ids'].shape[1]>self.model.config.max_position_embeddings:raise ValueError('context_overflow')
  return x.to(self.model.device),offsets
 def attention(self,module,q,k,v,mask,scaling,dropout=0.,**kw):
  layer=module.layer_idx
  if self.capture is not None and layer in self.capture:self.captured[layer]=q[0,:,-1,:].detach().float().cpu()
  out,w=self.original(module,q,k,v,mask,scaling=scaling,dropout=dropout,**kw);a=self.active
  if a is None or a['strength']==0 or layer!=a['layer']:return out,w
  k=repeat_kv(k,module.num_key_value_groups);v=repeat_kv(v,module.num_key_value_groups)
  h=a['head'];query=q[:,h:h+1,-1:,:].float();key=k[:,h:h+1].float()
  signs=torch.zeros(key.shape[-2],device=q.device);n=min(len(a['signs']),len(signs));signs[:n]=a['signs'][:n].to(q.device)
  if not bool(signs.any()):return out,w
  score=query@key.transpose(-2,-1)*scaling
  if mask is not None:score+=mask[:,:1,-1:,:key.shape[-2]].float()
  if a['method']=='direct':score+=a['strength']*signs[None,None,None,:]
  else:
   d=a['direction'].to(q.device).float();delta=(key@d)[...,None]*d*signs[None,None,:,None]*a['strength']
   score+=(query@delta.transpose(-2,-1))*scaling
  changed=(score.softmax(-1).to(v.dtype)@v[:,h:h+1]).transpose(1,2)
  out=out.clone();out[:,-1:,h:h+1,:]=changed
  return out,w
 @torch.inference_mode()
 def generate(self,text,max_tokens=512):
  inp,_=self.encode(text);y=self.model.generate(**inp,max_new_tokens=max_tokens,do_sample=False,use_cache=True,pad_token_id=self.tok.eos_token_id)
  return self.tok.decode(y[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
 @torch.inference_mode()
 def query(self,text,layers):
  self.capture=layers;self.captured={}
  try:
   inp,_=self.encode(text);self.model(**inp,use_cache=False);return dict(self.captured)
  finally:self.capture=None
 def applicability(self,case):
  rules=[{'id':r['id'],'condition':r['condition']} for r in case['rules']]
  text=self.prompt('Evaluate each condition against the patient. T=explicitly supported, F=explicitly contradicted, U=unknown. Missing information is U. Return JSON object mapping every id to T/F/U only.\nPatient:\n'+case['patient']+'\nConditions:\n'+json.dumps(rules,ensure_ascii=False))
  raw=self.generate(text,min(4096,max(128,len(rules)*12)))
  try:
   obj=json.loads(re.search(r'\{.*\}',raw,re.S).group())
   return {r['id']:{'T':1,'F':-1,'U':0}.get(str(obj.get(r['id'],'U')).upper(),0) for r in rules},raw,None
  except Exception:return {r['id']:0 for r in rules},raw,'applicability_parse_failure'
 def task(self,c):
  chunks=['Clinical guideline:\n'+c.get('guideline','')+'\nDecision rules:\n'];spans={}
  for r in c['rules']:
   start=sum(map(len,chunks));line='R'+r['id']+': '+r['text']+'\n';chunks.append(line);spans[r['id']]=(start,start+len(line))
  chunks.append('\nPatient:\n'+c['patient'])
  if c['dataset']=='mega':chunks.append('')
  else:chunks.append('\nOptions:\n'+'\n'.join(f'{chr(65+i)}: {x}' for i,x in enumerate(c['options']))+'\nReturn only the best option letter. Answer:')
  user=''.join(chunks)
  if c['dataset']=='mega':
   lang=c.get('language','en');name='ZH_DECISION_MAKING_PREDICTION_TEMPLATE.txt' if lang=='zh' else 'DECISION_MAKING_PREDICTION_TEMPLATE.txt'
   template=(Path('/home/gcx/aster_fullscale_20261030/official_mega/mega_cdp/template')/('template_'+lang)/name).read_text()
   guideline=''.join(chunks[:-1]);user=template.replace('{Guideline}',guideline).replace('{Case}',c['patient']);shift=user.index(guideline)
   spans={k:(a+shift,b+shift) for k,(a,b) in spans.items()}
  text=self.prompt(user);base=text.index(user);inp,offset=self.encode(text,True)
  masks={rid:torch.tensor([y>base+a and x<base+b and y>x for x,y in offset]) for rid,(a,b) in spans.items()}
  return text,masks
 def activate(self,selection,method,masks,route,directions):
  signs=torch.zeros(len(next(iter(masks.values()))))
  for rid,m in masks.items():signs[m]=float(route.get(rid,0))
  self.active=dict(layer=selection['layer'],head=selection['head'],strength=0. if method in ('baseline','zero') else selection['strength'],signs=signs,method=method,direction=directions.get(method,directions['learned']))
