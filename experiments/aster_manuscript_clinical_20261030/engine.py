"""Clinical cache translation executors; gold fields are absent from routing."""
import copy,json,re,time
from contextlib import nullcontext
from .native_memory import query_chunked_attention
from pathlib import Path
import torch
from src.aster.manuscript_runtime import PromptBuilder,ManuscriptBackbone
from src.aster.manuscript_core import plan_gains,edit_kv_layers,guideline_groups
from src.aster.action_blind import summarize_logits
from experiments.aster_manuscript_20261030.controls import directions_for,attention_probe
METHODS=['baseline','zero','spectral','mean','random','reverse','score_bias','enhance_only','suppress_only','direct_map','text_append','text_clip','native_official']
def decisions_from(raw,freeze):
 out={}
 for gid,d in raw.items():
  out[gid]=None if d is None else summarize_logits(d['logits'],temperature=freeze['temperature']['temperature'],
   probability_threshold=freeze['thresholds']['probability'],margin_threshold=freeze['thresholds']['margin'])
 return out
def direct_text(case,plan):
 group='G0';path=[];steps=[];leaf=None;conclusion='The available information does not support a complete guideline decision.'
 groups=guideline_groups(case['guideline'])
 while group in groups:
  state=plan['groups'][group]
  if not state['enabled']:break
  rid=state['prediction'];rule=next(r for r in groups[group]['rules'] if r['rule_id']==rid)
  path.append(rid);steps.append(rule['condition'])
  action=rule['action'];match=re.match(r'L(\d+)[：:]',action)
  if match:leaf='L'+match.group(1);conclusion=action.split('：',1)[-1];break
  match=re.search(r'G\d+',action)
  if not match:break
  group=match.group()
 zh=case.get('language')=='zh'
 header='### 临床决策过程：' if zh else '### Clinical Decision Process:'
 outcome='### 临床决策结论：' if zh else '### Clinical Decision Outcome:'
 return header+'\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(steps or ['Applicability could not be confirmed.']))+'\n'+outcome+' '+conclusion+'\nPATH: '+','.join(path)+'\nFINAL: '+(leaf or 'NONE')
def parse_prediction(text):
 fm=re.findall(r'FINAL\s*[:：]\s*(L\d+|NONE)',text,re.I);leaf=fm[-1].upper() if fm else None
 if leaf is None:
  found=set(re.findall(r'\bL\d+\b',text))
  if len(found)==1:leaf=next(iter(found))
 match=re.findall(r'PATH\s*[:：]\s*([^\n]+)',text,re.I);path=re.findall(r'R\d+[YN]',match[-1] if match else text)
 return leaf,list(dict.fromkeys(path)),leaf is None
def grade(case,text):
 leaf,path,invalid=parse_prediction(text);gold=case['gold_rules'];inter=len(set(path)&set(gold))
 return {'final_leaf_correct':leaf==case['gold_leaf'],'path_f1':2*inter/(len(set(path))+len(set(gold))) if path or gold else 1.,
  'exact_path':path==gold,'predicted_leaf':leaf,'predicted_path':path,'invalid_generation':invalid,'scoring_failure':False}
class ClinicalExecutor:
 def __init__(self,model,tok,expert,freeze):
  self.model,self.tok,self.expert,self.freeze=model,tok,expert,freeze
  self.builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
  self.backend=ManuscriptBackbone(model,model_id='Qwen/Qwen3-8B',model_revision=freeze['model_revision'])
  self.prefix=None
 def native(self,case):
  lang='zh' if case.get('language')=='zh' else 'en'
  root=Path('/home/gcx/aster_manuscript_20261030/clinical01/official_mega/mega_cdp/template')
  name='ZH_DECISION_MAKING_PREDICTION_TEMPLATE.txt' if lang=='zh' else 'DECISION_MAKING_PREDICTION_TEMPLATE.txt'
  template=(root/('template_'+lang)/name).read_text()
  prompt=template.replace('{Guideline}',Path(case['original_guideline_path']).read_text()).replace('{Case}',case['patient'])
  rendered=self.tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=self.tok(rendered,return_tensors='pt',add_special_tokens=False).to(self.backend.device);self.backend._budget(ids['input_ids'].shape[1],self.freeze['max_new_tokens'])
  memory=query_chunked_attention(self.model,self.freeze['native_attention_query_chunk_size'],balanced=self.freeze.get('native_attention_balanced',False)) if self.freeze.get('native_attention_query_chunk_size') else nullcontext()
  with torch.inference_mode(),memory:
   out=self.model.generate(**ids,max_new_tokens=self.freeze['max_new_tokens'],do_sample=False,pad_token_id=self.tok.eos_token_id,use_cache=True)
  generated=out[0,ids['input_ids'].shape[1]:].tolist()
  return {'text':self.tok.decode(generated,skip_special_tokens=True),'token_ids':generated,'input_tokens':ids['input_ids'].shape[1],
   'official_native_prompt':True,'native_attention_query_chunk_size':self.freeze.get('native_attention_query_chunk_size'),'structural_score_applicable':False,'edit':{'skipped':True},'attention':[]}
 def execute(self,case,raw,method,gain=None,capture=True,decisions_override=None):
  start=time.time();freeze=self.freeze;gain=freeze['g_plus'] if gain is None else gain
  decisions=decisions_from(raw,freeze) if decisions_override is None else decisions_override;plan=plan_gains(case['guideline'],decisions,g_plus=gain,g_minus=gain)
  if method=='native_official':
   result=self.native(case);result.update(plan=plan,decisions=decisions,seconds=time.time()-start);return result
  if method=='direct_map':
   return {'text':direct_text(case,plan),'token_ids':[],'input_tokens':0,'decisions':decisions,'plan':plan,'attention':[],'edit':{'skipped':True},'seconds':time.time()-start}
  guide=case['guideline'];question=case['question']
  if method=='text_append':
   accepted=[rid+('：适用' if g>0 else '：不适用') for rid,g in plan['gains'].items() if g]
   question+='\n处置盲分类结果：'+('；'.join(accepted) if accepted else '无法确认适用分支。')
  if method=='text_clip' and any(plan['gains'].values()):
   guide=copy.deepcopy(guide);dropped=set()
   for g in guide['groups']:
    if g.get('parent_group_id') in dropped or plan['gains'].get(g.get('parent_rule_id'),0)<0:dropped.add(g['group_id'])
   guide['groups']=[g for g in guide['groups'] if g['group_id'] not in dropped]
   for g in guide['groups']:g['rules']=[r for r in g['rules'] if plan['gains'][r['rule_id']]>=0]
  prompt=self.builder.prepare(guide,case['patient'],question,online=True)
  if self.expert['identity']!=self.backend.identity or self.expert['provenance']['template_id']!=prompt.template_id:raise ValueError('expert/model/template mismatch')
  self.backend._budget(prompt.input_ids.shape[1],freeze['max_new_tokens'])
  p=prompt.prefix_length
  if self.prefix is None or self.prefix.template_id!=prompt.template_id or not torch.equal(self.prefix.input_ids,prompt.input_ids[:,:p]):
   self.prefix=self.backend.prefill(prompt)
  prefix=self.prefix;gains=plan['gains'].copy();directions={};bias={}
  if method in ['spectral','mean','random','reverse','enhance_only','suppress_only']:directions=directions_for(self.expert,method)
  if method=='enhance_only':gains={r:max(0,v) for r,v in gains.items()}
  if method=='suppress_only':gains={r:min(0,v) for r,v in gains.items()}
  if method in ['baseline','zero','text_append','text_clip']:gains={r:0. for r in prompt.positions}
  if method=='score_bias':bias=freeze['bias_unit_rms']
  observed={r:idx for r,idx in prompt.positions.items() if r in ['R0Y','R0N'] or plan['gains'].get(r,0)!=0}
  heads=list(directions_for(self.expert,'spectral'));tokens=[];first=None;eos=self.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);last=prompt.input_ids.shape[1]-1
  with self.backend.session():
   layers,audit=edit_kv_layers(prefix.cache.to_legacy_cache(),directions,prompt.positions,gains,prefix.cos,prefix.sin)
   cache=type(prefix.cache).from_legacy_cache(tuple(layers))
   # Score-bias applies all rule targets; measurement records root and routed targets.
   probe_positions=prompt.positions if bias else observed
   with attention_probe(self.backend,heads if capture or bias else [],probe_positions,gains,bias,capture_positions={last+n for n in [0,1,2,4,8,16]}) as stats:
    result=self.backend._forward(prompt.input_ids[:,p:],cache=cache,start=p,use_cache=True);first=result.logits[0,-1].detach().float().cpu()
    for step in range(freeze['max_new_tokens']):
     token=int(result.logits[0,-1].argmax());tokens.append(token)
     if token in eos or step+1==freeze['max_new_tokens']:break
     result=self.backend._forward(torch.tensor([[token]],dtype=torch.long),cache=result.past_key_values,start=prompt.input_ids.shape[1]+step,use_cache=True)
  stats=[x for x in stats if x['query_absolute_position']-last in [0,1,2,4,8,16]]
  return {'text':self.tok.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'input_tokens':prompt.input_ids.shape[1],'decisions':decisions,
   'plan':plan,'edit':audit,'attention':stats,'seconds':time.time()-start,'first_logits':first,'output_limit':len(tokens)==freeze['max_new_tokens'] and tokens[-1] not in eos}

