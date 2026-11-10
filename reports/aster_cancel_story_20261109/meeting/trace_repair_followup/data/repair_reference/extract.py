"""Separate source-only rule compilation, evidence extraction and edit planning."""
import argparse,time,os
from .core import *
RULE='''The output MUST follow this JSON structure:
{"available": BOOLEAN, "predicate": EXPRESSION, "reason": STRING}
EXPRESSION is either a leaf {"op": OPERATOR, "name": EXACT_SOURCE_VARIABLE, "source": EXACT_ATOMIC_CONDITION, "value": JSON_SCALAR_OR_LIST, "unit": STRING},
or {"op":"and", "args":[EXPRESSION,...]}, or {"op":"or", "args":[EXPRESSION,...]}, or {"op":"not", "arg":EXPRESSION}.
A comparison leaf NEVER has args. Boolean presence uses == and value true; absence uses == and value false. in requires a LIST of categories; NEVER use in or not_in for a yes/no observation.
Compile ONLY the supplied current condition into JSON. No patient or answer is supplied.
Return available (boolean), predicate, reason. Predicate is an expression with op and/or and args, not and arg, or a leaf with op (< <= > >= == != in not_in), name, source, value, unit.
Every leaf's source is an EXACT contiguous quote of its entire atomic condition. name is an EXACT substring of source naming the variable. Use the same name as field; do not invent English aliases. Include ALL current-condition clauses, NO other conditions. Preserve whether parentheses define a term or add requirements. Do not turn numeric conditions into opaque boolean facts. Preserve conjunction/disjunction, endpoints and units. A yes/no question is a condition to encode, not a reason for unavailability. Boolean facts are allowed for explicitly stated nonnumeric conditions; the name must include the actual property, not just the organ or subject. Lists use in/not_in. If an expression cannot be represented faithfully, available=false. Do not invent definitions or missing cutoffs. No examples, no prose outside JSON.'''
FACT='''Output shape: observations is an OBJECT keyed by the exact variable names. Each entry has value (number, boolean, category string or null), state (observed/explicit_negative/unknown), sentence_ids (array of integer IDs). No predicate or args in the output.
Extract patient observations ONLY for the requested variables. Return JSON with observations mapping each EXACT variable name to {value, state, sentence_ids}. state is observed, explicit_negative, or unknown. Unmentioned is UNKNOWN with value null; explicit denial is not unknown. sentence_ids are integers from the numbered original sentences. Include every sentence mentioning that variable, including repeated numeric and categorical descriptions. A sentence saying A OR B does not establish A or B individually. Do not treat this as both true. Numeric values must be actual measurements, not thresholds or hypothetical examples; preserve units, time and population. Do not infer a point value from a bound. For an in/not_in variable return the observed category, not a boolean. No answers or edits. No invented facts.'''
EDIT='''Plan independent, minimal edits of the numbered patient text. Return JSON with edits (list) and background (object or null). Never invent a new patient.
For at most ONE current-rule variable that is explicitly observed, propose an edit with field (exact variable name), spans [{old,template}], step (positive numeric, native measurement precision), unit, minimum, integer_only, affected_ancestors (list of ancestor gid), ancestor_field_map (mapping ancestor gid to corresponding exact ancestor variable name if known). Numeric templates must contain {value}. Include ALL occurrences, repeats, inequalities and categorical paraphrases of this variable. Remove obsolete categorical threshold statements inside edited spans rather than leaving contradictory claims. Each old must be an exact substring and each replacement must retain other facts. If a single numeric template cannot represent the change without changing other facts, omit it. For boolean or category edits use legal_values [{value,replacements}] with one replacement per span, instead of numeric templates; preserve other facts. Return empty edits if unavailable. Do not default missing facts to false.
Identify EVERY ancestor condition potentially dependent on this edited variable (including synonyms and derived conditions). Do not list unrelated ancestors.
Independently background may contain old,new,field,before,after,reason,unrelated_to_all_conditions=true. It must replace a real, explicitly located clinical background fact unrelated to current AND every ancestor condition. Do NOT use generic last-sentence task/focus text, insert new background, or presume age/sex irrelevant. If no such fact exists return null. Do not alter any decisive fact. Return concise JSON only.'''
class Generator:
 def __init__(self,batch=6):
  import torch
  from transformers import AutoTokenizer,AutoModelForCausalLM
  torch.set_num_threads(4);self.torch=torch;self.batch=batch
  model='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B'
  self.tok=AutoTokenizer.from_pretrained(model,local_files_only=True,padding_side='left');self.tok.pad_token=self.tok.eos_token
  self.model=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
 def many(self,jobs,instruction,max_tokens):
  for start in range(0,len(jobs),self.batch):
   batch=jobs[start:start+self.batch]
   texts=[self.tok.apply_chat_template([dict(role='user',content=instruction+'\nINPUT:\n'+json.dumps(payload,ensure_ascii=False))],tokenize=False,add_generation_prompt=True,enable_thinking=False) for _,payload in batch]
   inp=self.tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to(self.model.device)
   with self.torch.inference_mode():out=self.model.generate(**inp,max_new_tokens=max_tokens,do_sample=False,pad_token_id=self.tok.pad_token_id)
   for (ident,_),raw in zip(batch,self.tok.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)):
    try:
     v,_=json.JSONDecoder().raw_decode(raw[raw.index('{'):]);yield ident,dict(value=v,raw=raw)
    except (ValueError,TypeError) as exc:yield ident,dict(error='JSON: '+str(exc),raw=raw)

def affected_ancestor_ids(g,record):
 # Invalid planner output is local to this edit and cannot abort another category.
 er=record.get('value',{})
 if not isinstance(er,dict):return set()
 edits=er.get('edits',[])
 if not isinstance(edits,list):return set()
 known={a['gid'] for a in g['ancestors']};affected=set()
 for ed in edits:
  if not isinstance(ed,dict):continue
  ids=ed.get('affected_ancestors',[])
  if isinstance(ids,list):affected.update(a for a in ids if isinstance(a,str) and a in known)
  field=ed.get('field')
  if isinstance(field,str) and field:
   affected.update(a['gid'] for a in g['ancestors'] if field in a['condition'])
 return affected

def run(shard,scope='pilot',batch=6):
 folder=ROOT/scope;folder.mkdir(parents=True,exist_ok=True)
 groups=rows(folder/'nodes.jsonl') if scope=='pilot' else rows(ROOT/'data/nodes.jsonl')
 assignment=read(folder/'assignment.json') if scope=='bulk' else {}
 if assignment:
  shards=assignment['shards'];assigned=[ident for ids in shards for ident in ids]
  if not shards or len(assigned)!=len(set(assigned)) or set(assigned)!={g['group_id'] for g in groups}:raise ValueError('invalid case assignment')
  owned=set(shards[shard]);groups=[g for g in groups if g['group_id'] in owned]
 else:
  gids=sorted({g['guideline_id'] for g in groups});owned=set(gids[shard::3]);groups=[g for g in groups if g['guideline_id'] in owned]
 rules={}
 # Cache by tree node. Each rule is extracted once, independent of any patient.
 for f in sorted((ROOT/'pilot').glob('rules_*.jsonl'))+sorted(folder.glob('rules_*.jsonl')):
  rules.update({r['id']:r for r in rows(f)})
 overrides=read(ROOT/'pilot/review_overrides.json')
 rules.update(overrides.get('rules',{}))
 rule_inputs={key(g):dict(current_condition=condition(g),definitions=g['definitions']) for g in groups}
 for g in groups:
  for a in g['ancestors']:rule_inputs[key(g,a['gid'])]=dict(current_condition=a['condition'],definitions=g['definitions'])
 gen=None;started=time.time()
 def generator():
  nonlocal gen
  if gen is None:gen=Generator(batch)
  return gen
 def progress(stage,n,total):
  write(folder/f'status_{shard}.json',dict(stage=stage,done=n,total=total,pid=os.getpid(),seconds=round(time.time()-started),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_gib=round(gen.torch.cuda.max_memory_allocated()/2**30,2) if gen else 0))
 for ident,payload in rule_inputs.items():
  if ident in rules:continue
  direct=direct_rule(payload['current_condition'])
  if direct:
   r=dict(id=ident,condition=payload['current_condition'],value=direct,state='complete',method='program')
   rules[ident]=r;append(folder/f'rules_{shard}.jsonl',r)
 jobs=[(k,p) for k,p in rule_inputs.items() if k not in rules]
 if jobs:
  for i,(ident,r) in enumerate(generator().many(jobs,RULE,1100)):
   r.update(id=ident,condition=rule_inputs[ident]['current_condition'])
   try:
    if 'error' in r:raise ValueError(r['error'])
    r['value']=validate_rule(r['condition'],r['value']);r['state']='complete'
   except (ValueError,KeyError,TypeError,IndexError) as exc:r.update(state='extraction_problem',error=str(exc))
   rules[ident]=r;append(folder/f'rules_{shard}.jsonl',r);progress('rules',i+1,len(jobs))
 facts={}
 for f in sorted((ROOT/'pilot').glob('facts_*.jsonl'))+sorted(folder.glob('facts_*.jsonl')):facts.update({r['id']:r for r in rows(f)})
 facts.update(overrides.get('facts',{}))
 jobs=[];gs={g['group_id']:g for g in groups}
 for g in groups:
  r=rules.get(key(g),{})
  if r.get('state')!='complete' or g['group_id'] in facts:continue
  jobs.append((g['group_id'],dict(variables=list(leaves(r['value']['predicate'])),sentences=sentences(g['patient']))))
 if jobs:
  for i,(ident,r) in enumerate(generator().many(jobs,FACT,1300)):
   r['id']=ident;g=gs[ident]
   try:
    if 'error' in r:raise ValueError(r['error'])
    r['value']=validate_facts(g,r['value'],rules[key(g)]['value']);r['state']='complete'
   except (ValueError,KeyError,TypeError,IndexError) as exc:r.update(state='extraction_problem',error=str(exc))
   facts[ident]=r;append(folder/f'facts_{shard}.jsonl',r);progress('facts',i+1,len(jobs))
 edits={}
 for f in sorted((ROOT/'pilot').glob('edits_*.jsonl'))+sorted(folder.glob('edits_*.jsonl')):edits.update({r['id']:r for r in rows(f)})
 jobs=[]
 from .program_edits import numeric_edits
 for g in groups:
  if g['group_id'] in edits:continue
  r=rules.get(key(g),{});f=facts.get(g['group_id'],{})
  if r.get('state')!='complete' or f.get('state')!='complete':
   rec=dict(id=g['group_id'],state='complete',value=dict(edits=[],background=None),reason='No observed current variables; independent program A/P still enabled')
   edits[g['group_id']]=rec;append(folder/f'edits_{shard}.jsonl',rec);continue
  try:
   pe=numeric_edits(g,r['value'],f['value'],dict(rules=rules))
  except (ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
   pe=[]
   append(folder/f'edit_preparation_failures_{shard}.jsonl',dict(id=g['group_id'],stage='numeric_edit_templates',state='unavailable',error=type(exc).__name__+': '+str(exc)))
  numeric=any(l['op'] in ('<','<=','>','>=') for l in leaves(r['value']['predicate']))
  if pe or numeric:
   rec=dict(id=g['group_id'],state='complete',value=dict(edits=pe,background=None),method='program-measurement-templates')
   edits[g['group_id']]=rec;append(folder/f'edits_{shard}.jsonl',rec);continue
  ancestors_payload=[dict(a,predicate=rules.get(key(g,a['gid']),{}).get('value',{}).get('predicate')) for a in g['ancestors']]
  jobs.append((g['group_id'],dict(current_condition=condition(g),predicate=r.get('value',{}).get('predicate'),observations=f.get('value',{}).get('observations',{}),ancestors=ancestors_payload,sentences=sentences(g['patient']))))
 if jobs:
  progress('edits',0,len(jobs))
  for i,(ident,r) in enumerate(generator().many(jobs,EDIT,1600)):
   r.update(id=ident,state='complete' if 'value' in r else 'extraction_problem');edits[ident]=r;append(folder/f'edits_{shard}.jsonl',r);progress('edits',i+1,len(jobs))
 # Only ancestors affected by an actual proposed fact edit need case-specific observations.
 af={}
 for f in sorted((ROOT/'pilot').glob('ancestor_facts_*.jsonl'))+sorted(folder.glob('ancestor_facts_*.jsonl')):af.update({r['id']:r for r in rows(f)})
 jobs=[];lookup={}
 for g in groups:
  affected=affected_ancestor_ids(g,edits.get(g['group_id'],{}))
  for gid in sorted(affected):
   ident=g['group_id']+'@'+gid;r=rules.get(key(g,gid),{})
   if r.get('state')!='complete' or ident in af:continue
   lookup[ident]=(g,r['value']);jobs.append((ident,dict(variables=list(leaves(r['value']['predicate'])),sentences=sentences(g['patient']))))
 if jobs:
  for i,(ident,r) in enumerate(generator().many(jobs,FACT,1300)):
   r['id']=ident;g,rule=lookup[ident]
   try:
    if 'error' in r:raise ValueError(r['error'])
    r['value']=validate_facts(g,r['value'],rule);r['state']='complete'
   except (ValueError,KeyError,TypeError,IndexError) as exc:r.update(state='extraction_problem',error=str(exc))
   append(folder/f'ancestor_facts_{shard}.jsonl',r);progress('ancestor_facts',i+1,len(jobs))
 progress('complete',len(groups),len(groups))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--scope',default='pilot');p.add_argument('--batch',type=int,default=6);a=p.parse_args();run(a.shard,a.scope,a.batch)
