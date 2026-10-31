"""Full-rule masks, shared-prefix cache execution and decision-point diagnostics."""
import copy,math,re,time
from contextlib import contextmanager
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
from src.aster.manuscript_core import edit_kv_layers,rotate
from experiments.aster_manuscript_20261030.controls import directions_for

def rule_masks(builder,prompt,case,route):
 text=prompt.text if hasattr(prompt,'text') else prompt.rendered
 # Rendered field differs between historical data classes; positions are recomputed
 # from the exact existing rendering, with the original tokenization.
 guide,_=builder.guideline(case['guideline'])
 offset=text.index(guide)
 enc=builder.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 positions={};groupspans={}
 starts=[m for m in re.finditer(r'规则编号：',guide)]
 for i,m in enumerate(starts):
  rid=guide[m.end():].split('\n',1)[0];end=starts[i+1].start() if i+1<len(starts) else len(guide)
  # End at a group header if it precedes the next rule.
  nxt=guide.find('规则组：',m.end(),end)
  if nxt>=0:end=nxt
  lo,hi=offset+m.start(),offset+end
  positions[rid]=[j for j,(a,b) in enumerate(enc['offset_mapping']) if b>a and b>lo and a<hi]
 chosen=set(route);rejected=set();desc=set();protected=set()
 for g in case['guideline']['groups']:
  ids={r['rule_id'] for r in g['rules']}
  if ids&chosen:rejected|=ids-chosen
 # Propagate only rejected branches, leaving the unconfirmed selected subtree intact.
 blocked=set(rejected)
 changed=True
 while changed:
  changed=False
  for g in case['guideline']['groups']:
   if g.get('parent_rule_id') in blocked:
    for r in g['rules']:
     if r['rule_id'] not in blocked:blocked.add(r['rule_id']);desc.add(r['rule_id']);changed=True
 def union(rids):return sorted({j for rid in rids for j in positions.get(rid,[]) if j>=4})
 path=union(chosen);siblings=union(rejected);subtree=union(desc);nonpath=union(blocked)
 # Include dropped group headers / inherited ancestor text: this information is
 # absent from text clipping as well, and must not remain as a readable loophole.
 groupmarkers=list(re.finditer(r'规则组：(G\d+)\n',guide))
 for i,m in enumerate(groupmarkers):
  group=next(g for g in case['guideline']['groups'] if g['group_id']==m.group(1))
  if group.get('parent_rule_id') in blocked:
   lo=offset+m.start();hi=offset+(groupmarkers[i+1].start() if i+1<len(groupmarkers) else len(guide))
   extra=[j for j,(a,b) in enumerate(enc['offset_mapping']) if b>a and b>lo and a<hi and j>=4]
   nonpath.extend(extra);subtree.extend(extra)
 return {'path':path,'siblings':siblings,'subtree':sorted(set(subtree)),'nonpath':sorted(set(nonpath)),
  'patient':list(range(prompt.prefix_length,prompt.input_ids.shape[1])),
  'sink':list(range(min(4,prompt.prefix_length))),'rules':positions,'chosen':sorted(chosen),'blocked':sorted(blocked)}

def clip_guide(case,mask):
 guide=copy.deepcopy(case['guideline']);blocked=set(mask['blocked'])
 guide['groups']=[g for g in guide['groups'] if g.get('parent_rule_id') not in blocked]
 for g in guide['groups']:g['rules']=[r for r in g['rules'] if r['rule_id'] not in blocked]
 return guide

def cache_copy(cache):
 return type(cache).from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in cache.to_legacy_cache()))

@contextmanager
def attention_control(backend,mask=None,beta=0.,hard=False,siblings_only=False,heads=None,capture_at=None,
                      base_layers=None,edited_layers=None,gains=None,directions=None,prefix=None):
 originals={};trace=[]
 chosen_heads=None if heads is None else set(map(tuple,heads))
 for layer,block in enumerate(backend.blocks):
  module=block.self_attn;originals[layer]=module.forward
  local=list(range(backend.n_kv)) if chosen_heads is None else [h for l,h in chosen_heads if l==layer]
  def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,cache_position=None,
              _layer=layer,_local=local,**kwargs):
   shape=hidden_states.shape[:-1];hs=(*shape,-1,self.head_dim)
   q=self.q_norm(self.q_proj(hidden_states).view(hs)).transpose(1,2)
   k=self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1,2)
   v=self.v_proj(hidden_states).view(hs).transpose(1,2)
   cos,sin=position_embeddings;q,k=apply_rotary_pos_emb(q,k,cos,sin)
   if past_key_values is not None:k,v=past_key_values.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
   kk,vv=repeat_kv(k,self.num_key_value_groups),repeat_kv(v,self.num_key_value_groups)
   n=q.shape[-2];parts=[];start_abs=int(cache_position[0]) if cache_position is not None else kk.shape[-2]-n
   for st in range(0,n,256):
    en=min(st+256,n);scores=torch.matmul(q[:,:,st:en],kk.transpose(2,3))*self.scaling
    if attention_mask is not None:scores=scores+attention_mask[:,:,st:en,:kk.shape[-2]]
    raw_last=scores[0,:,-1].float().clone() if capture_at is not None and start_abs+en-1==capture_at else None
    if mask and _local and (hard or beta):
     bad=mask['siblings'] if siblings_only else mask['nonpath'];good=mask['path']
     # All hooks during suffix/decode; no intervention during original prefill.
     for kv in _local:
      a,b=kv*backend.group_size,(kv+1)*backend.group_size
      if bad:scores[:,a:b,:,bad]=float('-inf') if hard else scores[:,a:b,:,bad]-beta
      if not hard and beta and good:scores[:,a:b,:,good]+=beta
    weights=torch.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
    if raw_last is not None and mask:
     for kv in range(backend.n_kv):
      a,b=kv*backend.group_size,(kv+1)*backend.group_size
      query=q[0,a:b,en-1].float()
      row={'layer':_layer,'kv_head':kv,'query_position':capture_at}
      for group in ['path','siblings','subtree','patient','sink']:
       ids=[j for j in mask[group] if j<weights.shape[-1]]
       row['mass_'+group]=float(weights[0,a:b,-1,ids].float().sum(-1).mean()) if ids else 0.
      valid=raw_last[a:b,:capture_at+1]
      row['score_std']=float(valid.std(-1).mean())
      row['query_norm_over_sqrt_dim']=float(query.norm(dim=-1).mean()/math.sqrt(backend.dim))
      if base_layers is not None and directions and (_layer,kv) in directions:
       actual=[];intended=[];theory=[];unit_aligned=[]
       for rid,ids in mask['edit_positions'].items():
        gain=gains.get(rid,0.)
        if gain==0:continue
        ids=list(ids)
        delta=edited_layers[_layer][0][0,kv,ids].float()-base_layers[_layer][0][0,kv,ids].float()
        ds=query@delta.T*self.scaling;actual.append(ds.flatten());intended.append((ds*gain>0).flatten())
        vec=directions[_layer,kv]
        rot=rotate(vec[None],prefix.cos[ids],prefix.sin[ids]).to(query.device).float()
        theory.append((query@rot.T*self.scaling*gain).flatten());unit_aligned.append((query@rot.T*self.scaling).flatten())
       if actual:
        ds=torch.cat(actual);row.update(delta_mean=float(ds.mean()),delta_abs_mean=float(ds.abs().mean()),
          delta_sign_fraction=float(torch.cat(intended).float().mean()),
          upper_bound_over_std=abs(max(gains.values(),key=abs))*row['query_norm_over_sqrt_dim']/max(row['score_std'],1e-9),
          quantization_delta_mae=float((ds-torch.cat(theory)).abs().mean()),
          theory_unit_aligned_mean=float(torch.cat(unit_aligned).mean()),theory_unit_square_mean=float(torch.cat(unit_aligned).square().mean()))
      trace.append(row)
    parts.append(torch.matmul(weights,vv).transpose(1,2).contiguous())
   out=torch.cat(parts,dim=1).reshape(*shape,-1)
   return self.o_proj(out),None
  module.forward=MethodType(forward,module)
 try:yield trace
 finally:
  for layer,old in originals.items():backend.blocks[layer].self_attn.forward=old

def execute(engine,case,spec,route=None,capture=False):
 start=time.time();route=case['gold_rules'] if route is None else route
 builder=engine.builder;back=engine.backend
 prompt=builder.prepare(case['guideline'],case['patient'],case['question'],online=spec.get('order')!='patient_first')
 if spec.get('order')=='patient_first':prompt.prefix_length=prompt.input_ids.shape[1]-1
 mask=rule_masks(builder,prompt,case,route);mask['edit_positions']=prompt.positions
 p=prompt.prefix_length
 if spec['kind']=='clip':
  prompt=builder.prepare(clip_guide(case,mask),case['patient'],case['question'],online=True)
  mask=rule_masks(builder,prompt,{**case,'guideline':clip_guide(case,mask)},route);mask['edit_positions']=prompt.positions;p=prompt.prefix_length
 elif spec['kind']=='neutral':
  prompt=copy.copy(prompt);prompt.input_ids=prompt.input_ids.clone()
  neutral=builder.tokenizer.encode(' ',add_special_tokens=False)[0]
  prompt.input_ids[:,mask['nonpath']]=neutral
 key=tuple(prompt.input_ids[0,:p].tolist())
 if getattr(engine,'new_prefix_key',None)!=key:
  engine.new_prefix=back.prefill(prompt);engine.new_prefix_key=key
 prefix=engine.new_prefix;recompute_fraction=0.
 if spec['kind']=='repair':prefix,recompute_fraction=selective_recompute(back,prefix,mask,spec['fraction'])
 base=prefix.cache.to_legacy_cache();directions={};gains={}
 if spec['kind'] in ['spectral','mean','reverse','random','lowfreq']:
  expert=copy.deepcopy(engine.expert) if spec.get('direction_heads') is not None else engine.expert
  if spec.get('direction_heads') is not None:expert['selected_heads']=[str(l)+':'+str(h) for l,h in spec['direction_heads']]
  directions=directions_for(expert,'mean' if spec['kind']=='lowfreq' else spec['kind'])
  if spec['kind']=='lowfreq':
   for h,vec in directions.items():
    vec=vec.clone();half=vec.shape[-1]//2;vec[:half//2]=0;vec[half:half+half//2]=0;directions[h]=vec/(vec.norm()+1e-12)
  if spec.get('direction_mix') is not None:
   aa,bb,cc=spec['direction_mix']
   for (ll,hh),vec in directions.items():
    low=vec.clone();half=vec.shape[-1]//2;low[:half//2]=0;low[half:half+half//2]=0;low=low/(low.norm()+1e-12)
    gen=torch.Generator().manual_seed(13+ll*8+hh);random=torch.randn(vec.shape,generator=gen,dtype=vec.dtype);random=random/(random.norm()+1e-12)
    mixed=aa*vec+bb*low+cc*random;directions[ll,hh]=mixed/(mixed.norm()+1e-12)
  # Original key operator controls reached siblings only; descendants are a separate arm.
  groups=case['guideline']['groups']
  for g in groups:
   ids=[r['rule_id'] for r in g['rules']]
   selected=next((x for x in ids if x in route),None)
   for rid in ids:gains[rid]=(spec.get('gain',.25)*(1 if rid==selected else -1)) if selected else 0.
  edit_positions=prompt.positions
  if spec.get('full_rule_keys'):
   edit_positions={'PATH':mask['path'],'NONPATH':mask['nonpath']}
   edit_positions={r:ids for r,ids in edit_positions.items() if ids}
   gains={r:spec.get('gain',1.)*(1 if r=='PATH' else -1) for r in edit_positions}
   mask['edit_positions']=edit_positions
  if spec.get('calibrate_only'):layers=tuple((k.clone(),v.clone()) for k,v in base)
  else:layers,_=edit_kv_layers(base,directions,edit_positions,gains,prefix.cos,prefix.sin)
  cache=type(prefix.cache).from_legacy_cache(tuple(layers))
 else:
  cache=cache_copy(prefix.cache)
  if spec['kind']=='erase':cache=erase_cached_keys(engine,cache,prompt,prefix,{**spec,**({'erase_positions':sorted(set(mask['path']+mask['nonpath']))} if spec.get('full_rule_keys') else {})})
  layers=cache.to_legacy_cache()
 kind=spec['kind'];beta=spec.get('beta',0.)
 kwargs=dict(mask=mask,beta=beta,hard=kind in ['hard','repair'],siblings_only=spec.get('siblings_only',False),heads=spec.get('heads'))
 # Root decision scores use the full contextualized identifier; shared prefix is
 # teacher-forced and candidate suffix log probabilities are summed.
 depth=min(spec.get('decision_depth',0),len(case['gold_rules'])-1)
 gold=case['gold_rules'][depth];gid='G'+gold[1:-1]
 group=next(g for g in case['guideline']['groups'] if g['group_id']==gid)
 candidates=[r['rule_id'] for r in group['rules']]
 decision_prefix='PATH: '+(','.join(case['gold_rules'][:depth])+',' if depth else '')
 sequences=[builder.tokenizer.encode(decision_prefix+rid,add_special_tokens=False) for rid in candidates]
 common=0
 while all(len(s)>common for s in sequences) and len({s[common] for s in sequences})==1:common+=1
 margin=None;traces=[]
 with back.session():
  with attention_control(back,**kwargs):
   res=back._forward(prompt.input_ids[:,p:],cache=cache,start=p,use_cache=True)
  engine.last_first_log_probs=torch.log_softmax(res.logits[0,-1].float(),-1).detach().cpu()
  reference=getattr(engine,"behavior_reference",None)
  format_kl=float((reference.exp()*(reference-engine.last_first_log_probs)).sum()) if reference is not None else 0.
  suffix_cache=res.past_key_values
  if common:
   with attention_control(back,**kwargs,capture_at=prompt.input_ids.shape[1]+common-1 if capture else None,
       base_layers=base,edited_layers=layers,gains=gains,directions=directions,prefix=prefix) as traces:
    probe=back._forward(torch.tensor([sequences[0][:common]]),cache=cache_copy(suffix_cache),
       start=prompt.input_ids.shape[1],use_cache=True)
   scores=[]
   for seq in sequences:
    ccache=cache_copy(probe.past_key_values);logits=probe.logits[0,-1].float();total=0.
    with attention_control(back,**kwargs):
     for j,token in enumerate(seq[common:]):
      total+=float(torch.log_softmax(logits,-1)[token])
      if j+common+1<len(seq):
       pred=back._forward(torch.tensor([[token]]),cache=ccache,start=prompt.input_ids.shape[1]+common+j,use_cache=True)
       logits=pred.logits[0,-1].float();ccache=pred.past_key_values
    scores.append(total)
   if gold in candidates:margin=scores[candidates.index(gold)]-max(s for i,s in enumerate(scores) if candidates[i]!=gold)
  if spec.get('probe_only'):
   return {'root_margin':margin if depth==0 else None,'decision_margin':margin,'decision_depth':depth,'trace':traces,'format_kl':format_kl,
    'seconds':time.time()-start,'input_tokens':prompt.input_ids.shape[1],'nonpath_tokens':len(mask['nonpath'])}
  tokens=[];eos=engine.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
  with attention_control(back,**kwargs):
   for step in range(engine.freeze['max_new_tokens']):
    token=int(res.logits[0,-1].argmax());tokens.append(token)
    if token in eos or step+1==engine.freeze['max_new_tokens']:break
    res=back._forward(torch.tensor([[token]]),cache=res.past_key_values,start=prompt.input_ids.shape[1]+step,use_cache=True)
 result={'text':builder.tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,
  'input_tokens':prompt.input_ids.shape[1],'nonpath_tokens':len(mask['nonpath']),
  'root_margin':margin if depth==0 else None,'decision_margin':margin,'decision_depth':depth,'seconds':time.time()-start,'output_limit':len(tokens)==engine.freeze['max_new_tokens'] and tokens[-1] not in eos,
  'trace':traces,'recomputed_fraction':recompute_fraction}
 return result

def selective_recompute(backend,prefix,mask,fraction):
 """Recompute selected retained tokens layer by layer against reused other KV."""
 from dataclasses import replace
 pool=mask['path'];budget=min(len(pool),max(1,round(prefix.input_ids.shape[1]*fraction)))
 selected=sorted({pool[min(len(pool)-1,int(i*len(pool)/budget))] for i in range(budget)})
 ids=torch.tensor(selected,device=backend.device)
 with backend.session():
  hidden=backend.model.model.embed_tokens(prefix.input_ids.to(backend.device)[:,ids])
  layers=[];cos=prefix.cos[selected].to(backend.device,dtype=backend.dtype)[None]
  sin=prefix.sin[selected].to(backend.device,dtype=backend.dtype)[None]
  for layer,block in enumerate(backend.blocks):
   residual=hidden;norm=block.input_layernorm(hidden);attn=block.self_attn
   shape=(*norm.shape[:-1],-1,attn.head_dim)
   q=attn.q_norm(attn.q_proj(norm).view(shape)).transpose(1,2)
   k=attn.k_norm(attn.k_proj(norm).view(shape)).transpose(1,2)
   v=attn.v_proj(norm).view(shape).transpose(1,2)
   q,k=apply_rotary_pos_emb(q,k,cos,sin)
   kk,vv=(x.clone() for x in prefix.cache.to_legacy_cache()[layer])
   kk[:,:,ids]=k;vv[:,:,ids]=v;layers.append((kk,vv))
   keys,values=repeat_kv(kk,attn.num_key_value_groups),repeat_kv(vv,attn.num_key_value_groups)
   parts=[]
   for start in range(0,len(selected),128):
    stop=min(start+128,len(selected));scores=q[:,:,start:stop]@keys.transpose(2,3)*attn.scaling
    positions=torch.arange(keys.shape[-2],device=backend.device)
    causal=positions[None,:]>ids[start:stop,None]
    scores.masked_fill_(causal[None,None],float('-inf'))
    if mask['nonpath']:scores[:,:,:,mask['nonpath']]=float('-inf')
    weights=torch.softmax(scores,-1,dtype=torch.float32).to(q.dtype)
    parts.append((weights@values).transpose(1,2).reshape(1,stop-start,-1))
   hidden=residual+attn.o_proj(torch.cat(parts,dim=1))
   hidden=hidden+block.mlp(block.post_attention_layernorm(hidden))
  result=replace(prefix,cache=type(prefix.cache).from_legacy_cache(tuple(layers)))
 return result,len(selected)/prefix.input_ids.shape[1]

def marginal_execute(engine,case,routes,beta):
 """Exact token posterior mixture over a truncated, fixed set of route experts."""
 start=time.time();back=engine.backend;builder=engine.builder
 prompt=builder.prepare(case['guideline'],case['patient'],case['question'],online=True);p=prompt.prefix_length
 key=tuple(prompt.input_ids[0,:p].tolist())
 if getattr(engine,'new_prefix_key',None)!=key:engine.new_prefix=back.prefill(prompt);engine.new_prefix_key=key
 masks=[rule_masks(builder,prompt,case,r['path']) for r in routes]
 with back.session():
  results=[]
  for mask in masks:
   with attention_control(back,mask=mask,beta=beta):
    results.append(back._forward(prompt.input_ids[:,p:],cache=cache_copy(engine.new_prefix.cache),start=p,use_cache=True))
  logw=torch.tensor([math.log(max(r['weight'],1e-30)) for r in routes],device=back.device)
  eos=engine.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);tokens=[]
  for step in range(engine.freeze['max_new_tokens']):
   lp=torch.stack([torch.log_softmax(r.logits[0,-1].float(),-1) for r in results])
   mix=torch.logsumexp(lp+logw[:,None],dim=0);token=int(mix.argmax());tokens.append(token)
   logw=logw+lp[:,token];logw-=torch.logsumexp(logw,0)
   if token in eos or step+1==engine.freeze['max_new_tokens']:break
   new=[]
   for mask,res in zip(masks,results):
    with attention_control(back,mask=mask,beta=beta):
     new.append(back._forward(torch.tensor([[token]]),cache=res.past_key_values,start=prompt.input_ids.shape[1]+step,use_cache=True))
   results=new
 return {'text':builder.tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'routes':routes,
  'seconds':time.time()-start,'input_tokens':prompt.input_ids.shape[1],'root_margin':None,'decision_margin':None,
  'mixture':'posterior-updated sequence mixture over top3 routes','trace':[]}

def erase_cached_keys(engine,cache,prompt,prefix,spec):
 if not hasattr(engine,'leace_maps'):
  from pathlib import Path
  engine.leace_maps=torch.load(Path('/home/gcx/aster_bottleneck_20261031/leace.pt'),map_location='cpu',weights_only=False)
 positions=spec.get('erase_positions',sorted({j for ids in prompt.positions.values() for j in ids}))
 for layer,head in map(tuple,spec['erase_heads']):
  params=engine.leace_maps.get((layer,head))
  if params is None:continue
  k=cache.layers[layer].keys
  original=k[0,head,positions].double()
  cos=prefix.cos[positions].to(k.device);sin=prefix.sin[positions].to(k.device)
  pre=rotate(original,cos,-sin)
  a,b=(params['a'],params['b']) if 'erase_random_seed' not in spec else params['random'][str(spec['erase_random_seed'])]
  mu=params['mean'].to(k.device);a=a.to(k.device);b=b.to(k.device)
  updated=pre-((pre-mu)@b)[:,None]*a[None,:]
  k[0,head,positions]=rotate(updated,cos,sin).to(k.dtype)
 return cache
