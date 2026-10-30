"""Same-backbone, same-routing execution controls; instance-local Qwen3 adapter."""
from contextlib import contextmanager
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
from src.aster.manuscript_core import edit_kv_layers,parse_head
from src.aster.action_blind import summarize_logits
from src.aster.manuscript_core import plan_gains

METHODS=['baseline','zero','spectral','mean','random','reverse','score_bias','enhance_only','suppress_only','direct_map','text_append','text_clip']

def decision(record,freeze,state,route):
 if route=='gold':
  label=record['row']['target_rule_id'] if state=='positive' else record['row']['verification']['negative_rule_id'] if state=='negative' else 'NONE'
  return {'prediction':label,'abstain':False,'diagnostic_only':True}
 c=record['classification'].get(state,{})
 if 'logits' not in c:return None
 f=freeze['thresholds']
 return summarize_logits(c['logits'],temperature=freeze['temperature']['temperature'],probability_threshold=f['probability'],margin_threshold=f['margin'])

@contextmanager
def attention_probe(backend,heads,positions,gains,bias_unit,capture_positions=None):
 """Original eager math, with optional score bias; no global registry changes."""
 original={};stats=[]
 for layer in sorted({h[0] for h in heads}):
  module=backend.blocks[layer].self_attn;original[layer]=module.forward
  localheads=[h[1] for h in heads if h[0]==layer]
  def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,cache_position=None,_layer=layer,_heads=localheads,**kwargs):
   input_shape=hidden_states.shape[:-1];shape=(*input_shape,-1,self.head_dim)
   q=self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1,2)
   k=self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1,2)
   v=self.v_proj(hidden_states).view(shape).transpose(1,2)
   cos,sin=position_embeddings;q,k=apply_rotary_pos_emb(q,k,cos,sin)
   if past_key_values is not None:k,v=past_key_values.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
   kk=repeat_kv(k,self.num_key_value_groups);vv=repeat_kv(v,self.num_key_value_groups)
   scores=torch.matmul(q,kk.transpose(2,3))*self.scaling
   if attention_mask is not None:scores=scores+attention_mask[:,:,:,:kk.shape[-2]]
   for kv in _heads:
    name=f'{_layer}:{kv}'
    if name in bias_unit:
     for rid,indices in positions.items():
      if gains[rid]==0:continue
      scores[:,kv*backend.group_size:(kv+1)*backend.group_size,:,list(indices)]+=gains[rid]*bias_unit[name]
   weights=torch.nn.functional.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
   for kv in (_heads if capture_positions is None or int(cache_position[-1]) in capture_positions else []):
    masses={rid:weights[0,kv*backend.group_size:(kv+1)*backend.group_size,-1,list(indices)].float().sum(-1).detach().cpu().tolist() for rid,indices in positions.items()}
    stats.append({'layer':_layer,'kv_head':kv,'query_absolute_position':int(cache_position[-1]),'mass':masses})
   result=torch.matmul(weights,vv).transpose(1,2).contiguous().reshape(*input_shape,-1)
   return self.o_proj(result),weights
  module.forward=MethodType(forward,module)
 try:yield stats
 finally:
  for layer,f in original.items():backend.blocks[layer].self_attn.forward=f

def directions_for(expert,method):
 gen=torch.Generator().manual_seed(20261030);out={}
 for name in expert['selected_heads']:
  c=expert['candidates'][name];spectral=torch.tensor(c['direction'],dtype=torch.float64);norm=spectral.norm()
  if method=='mean':v=torch.tensor(c['mean'],dtype=torch.float64);v=v/v.norm()*norm
  elif method=='random':v=torch.randn(len(spectral),generator=gen,dtype=torch.float64);v=v/v.norm()*norm
  elif method=='reverse':v=-spectral
  else:v=spectral
  out[parse_head(name)]=v
 return out

def execute(backend,builder,expert,freeze,record,state,route,method,gain,*,capture=True,max_new_tokens=None):
 row=record['row'];d=decision(record,freeze,state,route)
 plan=plan_gains(row['guideline'],{'g':d},g_plus=gain,g_minus=gain);gains=plan['gains']
 if method=='direct_map':
  value='NONE' if d is None or d['abstain'] else d['prediction']
  return {'text':value,'token_ids':[],'decision':d,'plan':plan,'edit':{'skipped':True},'attention':[]}
 question=row['question'];guide=row['guideline']
 if method=='text_append':
  lines=[rid+': '+('适用' if g>0 else '不适用') for rid,g in gains.items() if g]
  question+='\n适用性分类结果：'+('；'.join(lines) if lines else '无法确认。')
 if method=='text_clip' and any(gains.values()):
  import copy
  guide=copy.deepcopy(guide)
  guide['groups'][0]['rules']=[r for r in guide['groups'][0]['rules'] if gains[r['rule_id']]>=0]
 prompt=builder.prepare(guide,row['states'][state],question,online=True)
 if expert['identity']!=backend.identity or expert['provenance']['template_id']!=prompt.template_id:
  raise ValueError('Frozen expert and execution backbone/template differ')
 prefix=backend.prefill(prompt);heads=list(directions_for(expert,'spectral'))
 direct={};directions={}
 if method in ['spectral','mean','random','reverse','enhance_only','suppress_only']:directions=directions_for(expert,method)
 if method=='enhance_only':gains={r:max(0,g) for r,g in gains.items()}
 if method=='suppress_only':gains={r:min(0,g) for r,g in gains.items()}
 if method in ['baseline','zero','text_append','text_clip']:gains={r:0. for r in prompt.positions}
 if method=='score_bias':direct=freeze['bias_unit_rms']
 n=freeze['max_new_tokens'] if max_new_tokens is None else max_new_tokens;backend._budget(prompt.input_ids.shape[1],n)
 eos=backend.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);tokens=[];first=None
 with backend.session():
  layers,audit=edit_kv_layers(prefix.cache.to_legacy_cache(),directions,prompt.positions,gains,prefix.cos,prefix.sin)
  cache=type(prefix.cache).from_legacy_cache(tuple(layers))
  with attention_probe(backend,heads if capture or direct else [],prompt.positions,gains,direct) as stats:
   result=backend._forward(prompt.input_ids[:,prompt.prefix_length:],cache=cache,start=prompt.prefix_length,use_cache=True)
   first=result.logits[0,-1].detach().float().cpu()
   for i in range(n):
    token=int(result.logits[0,-1].argmax());tokens.append(token)
    if token in eos or i+1==n:break
    result=backend._forward(torch.tensor([[token]],dtype=torch.long),cache=result.past_key_values,start=prompt.input_ids.shape[1]+i,use_cache=True)
 return {'text':builder.tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'decision':d,'plan':plan,'edit':audit,
  'attention':stats,'first_logits':first,'stopped_on_eos':tokens[-1] in eos,'output_limit':len(tokens)==n and tokens[-1] not in eos}

