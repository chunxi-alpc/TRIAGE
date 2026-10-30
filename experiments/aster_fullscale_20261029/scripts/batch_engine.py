"""Left-padded batched Qwen generation with independent intervention state per row."""
import json,re,time
from transformers.cache_utils import DynamicCache
from collections import defaultdict
import torch
from fullscale_engine import Engine
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
class BatchEngine(Engine):
 def __init__(self,model,config_overrides=None):
  super().__init__(model,config_overrides=config_overrides);self.tok.padding_side='left';self.tok.pad_token=self.tok.eos_token
  self.prefix_cache={};self.reuse_prefixes=True
  self.batch_stats={'calls':0,'rows':0,'oom_splits':0,'seconds':0.,'prefix_builds':0,'prefix_reuses':0,'prefix_rows':0,'token_budget_splits':0}
 def attention(self,module,q,k,v,mask,scaling,dropout=0.,**kw):
  layer=module.layer_idx
  if self.capture is not None and layer in self.capture:self.captured[layer]=q[0,:,-1,:].detach().float().cpu()
  out,w=self.original(module,q,k,v,mask,scaling=scaling,dropout=dropout,**kw)
  active=self.active
  if active is None:return out,w
  configs=active if isinstance(active,list) else [active]
  eligible=[]
  for i,a in enumerate(configs):
   if not a or a['strength']==0 or a['layer']!=layer:continue
   if '_enabled' not in a:a['_enabled']=bool(torch.as_tensor(a['signs']).any())
   if a['_enabled']:eligible.append(i)
  if not eligible:return out,w
  heads={configs[i]['head'] for i in eligible}
  if len(heads)!=1:raise ValueError('batch requires common frozen head')
  h=next(iter(heads));k=repeat_kv(k,module.num_key_value_groups);v=repeat_kv(v,module.num_key_value_groups)
  query=q[:,h:h+1,-1:,:].float();key=k[:,h:h+1].float();length=key.shape[-2]
  score=query@key.transpose(-2,-1)*scaling
  if mask is not None:
   bias=mask[:,:1,-1:,:length]
   score+=torch.where(bias,0.,float('-inf')) if bias.dtype==torch.bool else bias.float()
  for i in eligible:
   a=configs[i]
   if '_signs_gpu' not in a:a['_signs_gpu']=torch.as_tensor(a['signs'],device=q.device).float()
   n=min(len(a['_signs_gpu']),length);signs=a['_signs_gpu'][:n]
   if a['method']=='direct':score[i,...,:n]+=a['strength']*signs[None,None,:]
   else:
    if '_direction_gpu' not in a:a['_direction_gpu']=a['direction'].to(q.device).float()
    d=a['_direction_gpu']
    if '_key_projection' not in a:a['_key_projection']=(key[i,...,:n,:]@d).detach()
    correction=(query[i]@d)[...,None]*a['_key_projection'][...,None,:n]
    score[i,...,:n]+=correction*signs[None,None,:]*a['strength']*scaling
  changed=(score.softmax(-1).to(v.dtype)@v[:,h:h+1]).transpose(1,2)
  out=out.clone()
  for i in eligible:out[i,-1:,h:h+1,:]=changed[i]
  return out,w
 def prefill(self,prefix,position):
  return self.model(**prefix,position_ids=position,use_cache=True,logits_to_keep=1)
 @torch.inference_mode()
 def generate_many(self,texts,max_tokens,configs=None,sample=False,batch_size=4,on_result=None):
  share_prefix=(configs is not None or sample) and self.reuse_prefixes
  results=[None]*len(texts);configs=configs or [None]*len(texts)
  def run(indices):
   if not indices:return
   started=time.time();self.batch_stats['calls']+=1
   try:
    tokenized=[self.tok(t,add_special_tokens=False)['input_ids'] for t in (texts[i] for i in indices)]
    for ids in tokenized:
     if len(ids)>self.model.config.max_position_embeddings:raise ValueError('context_overflow')
    width=max(map(len,tokenized));adjusted=[]
    if width*len(indices)>65536 and len(indices)>1:
     self.batch_stats['token_budget_splits']+=1;half=len(indices)//2;run(indices[:half]);run(indices[half:]);return
    for i,ids in zip(indices,tokenized):
     a=configs[i]
     if a:
      a=dict(a);a['signs']=torch.cat([torch.zeros(width-len(ids)),torch.as_tensor(a['signs']).cpu()]);adjusted.append(a)
     else:adjusted.append(None)
    inp=self.tok.pad({'input_ids':tokenized},padding=True,return_tensors='pt').to(self.model.device)
    active=adjusted if any(adjusted) else None
    cached=None
    if share_prefix and width>1:
     cache_key=tuple(texts[i] for i in indices)
     if cache_key not in self.prefix_cache:
      self.active=None
      if len(self.prefix_cache)>=2:self.prefix_cache.pop(next(iter(self.prefix_cache)))
      unique={};representatives=[];inverse=[]
      for j,i in enumerate(indices):
       if texts[i] not in unique:unique[texts[i]]=len(representatives);representatives.append(j)
       inverse.append(unique[texts[i]])
      pick=torch.tensor(representatives,device=self.model.device);inverse=torch.tensor(inverse,device=self.model.device)
      prefix={k:v.index_select(0,pick)[:,:-1] for k,v in inp.items()};position=prefix['attention_mask'].long().cumsum(-1)-1;position.masked_fill_(prefix['attention_mask']==0,1)
      prefix_output=self.prefill(prefix,position)
      self.prefix_cache[cache_key]=(prefix_output.past_key_values.to_legacy_cache(),inverse);del prefix_output
      self.batch_stats['prefix_rows']+=len(representatives)
      self.batch_stats['prefix_builds']+=1
     else:self.batch_stats['prefix_reuses']+=1
     original,inverse=self.prefix_cache[cache_key]
     legacy=tuple((k.index_select(0,inverse),v.index_select(0,inverse)) for k,v in original)
     cached=DynamicCache.from_legacy_cache(legacy)
    self.active=active
    settings=dict(max_new_tokens=max_tokens,do_sample=sample,use_cache=True,pad_token_id=self.tok.eos_token_id)
    if sample:settings.update(temperature=.7,top_p=.8,top_k=20)
    if cached is not None:settings['past_key_values']=cached
    y=self.model.generate(**inp,**settings)
    for i,row in zip(indices,y):
     results[i]=self.tok.decode(row[width:],skip_special_tokens=True)
     if on_result:on_result(i,results[i])
    self.batch_stats['rows']+=len(indices)
   except Exception as ex:
    message=str(ex);is_oom=isinstance(ex,torch.OutOfMemoryError) or 'out of memory' in message.lower();ex.__traceback__=None
    self.active=None;self.prefix_cache.clear();cached=None;legacy=None;original=None;inverse=None;inp=None;prefix=None;prefix_output=None;y=None;adjusted=None;active=None;settings=None;torch.cuda.empty_cache()
    if len(indices)>1:
     if is_oom:self.batch_stats['oom_splits']+=1
     half=len(indices)//2;run(indices[:half]);run(indices[half:])
    else:
     results[indices[0]]=RuntimeError(message)
     if on_result:on_result(indices[0],results[indices[0]])
   finally:
    self.active=None;self.batch_stats['seconds']+=time.time()-started
  # Length ordering changes execution order only, never task admission or truncation.
  order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
  for start in range(0,len(order),batch_size):run(order[start:start+batch_size])
  return results
 def applicability_many(self,cases,batch_size=4):
  result=[None]*len(cases);groups=defaultdict(list);prompts=[]
  for i,c in enumerate(cases):
   rules=[{'id':r['id'],'condition':r['condition']} for r in c['rules']]
   prompts.append(self.prompt('Evaluate each condition against the patient. T=explicitly supported, F=explicitly contradicted, U=unknown. Missing information is U. Return JSON object mapping every id to T/F/U only.\nPatient:\n'+c['patient']+'\nConditions:\n'+json.dumps(rules,ensure_ascii=False)))
   groups[min(4096,max(128,len(rules)*12))].append(i)
  for cap,indices in groups.items():
   outputs=self.generate_many([prompts[i] for i in indices],cap,batch_size=batch_size)
   for i,raw in zip(indices,outputs):
    route={r['id']:0 for r in cases[i]['rules']};failure=None
    if isinstance(raw,Exception):failure=str(raw);raw=''
    else:
     try:
      obj=json.loads(re.search(r'\{.*\}',raw,re.S).group());route={rid:{'T':1,'F':-1,'U':0}.get(str(obj.get(rid,'U')).upper(),0) for rid in route}
     except Exception:failure='applicability_parse_failure'
    result[i]=(route,raw,failure)
  return result
 def config(self,selection,method,masks,route,directions):
  self.activate(selection,method,masks,route,directions);a=self.active;self.active=None;return a
