"""Branch/action interventions. Uses fixed existing gain and original closed-book strata."""
import argparse,contextlib,hashlib,importlib,json,re,sys,time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]/'aster_medguide_20261022'))
from run_pilot import encode_prompt,signed_attention_patch

def parse(text,labels):
 clean=text.strip().replace('**','').replace('`','').splitlines()[0].strip() if text.strip() else '';m=re.match(r'^(?:(?:the\s+)?(?:correct\s+)?(?:answer|branch)\s*(?:is\s*)?:?\s*)?\(?([A-Z])\)?(?=\s*$|[.):])',clean,re.I)
 return m[1].upper() if m and m[1].upper() in labels else None

def branch_prompt(case,v,visibility):
 rules=case['variant_rules'][v];parts=[case['patient'],'\nChoose the branch whose CONDITIONS apply to this patient. Judge conditions, not which action you prefer.']
 for b,branch in enumerate(case['branches']):
  c=rules[branch['rule_indices'][0]]['condition'];text=f'{chr(65+b)}: IF {c}'
  if visibility=='visible' or (visibility=='hide_inapp' and b==case['gold_branch'][v]):text+='\nActions: '+' | '.join(rules[i]['action'] for i in branch['rule_indices'])
  parts.append(text)
 return '\n\n'.join(parts)+'\nReturn only one branch letter.'

def final_prompt(case,v):
 if v in case['source_variants']:return case['source_variants'][v]['prompt_text']
 a=case['source_variants']['A'];prefix=a['prompt_text'].split('IF the clinical sequence is: ',1)[0]
 return prefix+'\n\n'.join('IF the clinical sequence is: '+r['condition']+'\nTHEN the next step is: '+r['action']+'.' for r in case['variant_rules'][v])+'\n\n'+case['closed_book_prompt']

def prune_prompt(case,v,bs=None,single=None):
 indices=[single] if single is not None else [i for b in bs for i in case['branches'][b]['rule_indices']]
 rules=case['variant_rules'][v];texts=['IF the clinical sequence is: '+rules[i]['condition']+'\nTHEN the next step is: '+rules[i]['action']+'.' for i in indices]
 return 'Follow this supplied rule even if it conflicts with medical memory:\n'+'\n\n'.join(texts)+'\n\n'+case['closed_book_prompt']

def greedy_with_logits(model,enc,pad_token_id):
 # generate already computes the decision logits: reuse its RAW logits, not processed scores.
 # No extra prefill and no change to decoding, cache settings, gain, or maximum length.
 result=model.generate(**enc,do_sample=False,max_new_tokens=8,pad_token_id=pad_token_id,return_dict_in_generate=True,output_logits=True)
 return result.sequences,result.logits[0][0].float()

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--base',required=True);p.add_argument('--gain-file',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--phase',choices=['all','branch','action'],default='all');p.add_argument('--branches',type=Path);p.add_argument('--reuse-actions',type=Path);a=p.parse_args()
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(4);torch.manual_seed(20261023);start=time.time();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(str(out))
 cases=list(map(json.loads,Path(a.input).read_text().splitlines()));cases=cases[:a.limit] if a.limit else cases;existing={};strata={}
 for line in Path(a.base).read_text().splitlines():
  r=json.loads(line);existing[r['id'],r['variant'][0],r['method'],r.get('assigned_wrong',False)]=r
  if r['method']=='closed_book':strata[r['id']]=r['stratum']
 action_saved={}
 if a.reuse_actions:
  reuse_manifest=json.loads(a.reuse_actions.with_suffix('.manifest.json').read_text());assert reuse_manifest['complete'] and reuse_manifest['input_sha256']==hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
  for r in map(json.loads,a.reuse_actions.read_text().splitlines()):
   if r['kind']=='action':action_saved[r['id'],r['variant'],r['method']]=r
 branch_saved={}
 if a.branches:
  for r in map(json.loads,a.branches.read_text().splitlines()):
   if r['kind']=='branch':branch_saved[r['id'],r['variant'],r['method']]=r
 gain=json.loads(Path(a.gain_file).read_text())['selected_gain'];tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval();modeling=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type)
 cache={};n=0
 def score(prompt,labels,intervention=None):
  key=json.dumps([prompt,list(labels),intervention]);
  if key in cache:return dict(cache[key])
  spans=[] if intervention is None else intervention['spans'];enc,masks=encode_prompt(tok,prompt,spans);enc={k:v.to(model.device) for k,v in enc.items()};ids={k:tok.encode(k,add_special_tokens=False) for k in labels};assert all(len(x)==1 for x in ids.values())
  patch=contextlib.nullcontext()
  if intervention:
   ma=sum((masks[i] for i in intervention['positive']),torch.zeros_like(masks[0]));mb=sum((masks[i] for i in intervention['other']),torch.zeros_like(masks[0]));patch=signed_attention_patch(modeling,gain,intervention['other_gain'],ma,mb)
  with patch,torch.inference_mode():
   gen,z=greedy_with_logits(model,enc,tok.eos_token_id);vals={k:float(z[ids[k][0]]) for k in labels}
   text=tok.decode(gen[0,enc['input_ids'].shape[1]:],skip_special_tokens=True)
  r={'prediction':parse(text,labels),'completion':text,'option_logits':vals,'ranked':sorted(vals,key=vals.get,reverse=True),'reused':False};cache[key]=r;return dict(r)
 with out.open('w') as f:
  for ci,c in enumerate(cases):
   cache.clear();id=c['id'];stratum=strata.get(id)
   if stratum is None:raise ValueError('Missing original closed-book stratum '+id)
   def record(v,method,result,kind,**extra):
    nonlocal n
    gold=chr(65+c['gold_branch'][v]) if kind=='branch' else c['gold_answer'][v];r=dict(id=id,disease=c['disease'],variant=v,method=method,kind=kind,stratum=stratum,type=c['type'],gold=gold,**result,**extra);r['correct']=r['prediction']==gold;r['parse_failure']=r['prediction'] is None
    if kind=='action':
     r['original_action_selected']=r['prediction']==c['original_answer'];r['branch_permitted_action']=r['prediction'] in [k for k,text in c['options'].items() if text in [c['variant_rules'][v][i]['action'] for i in c['branches'][c['gold_branch'][v]]['rule_indices']]]
     r['sibling_action_selected']=r['prediction'] in [k for k,text in c['options'].items() if text in [c['variant_rules'][v][i]['action'] for i in c['branches'][c['sib_branch']]['rule_indices']]]
    f.write(json.dumps(r)+'\n');f.flush();n+=1
   def action(v,method,prompt,intervention=None,predicted_branch=None,runner_up=None):
    old=action_saved.get((id,v,method))
    if old and not old.get('upstream_branch_parse_failure') and (predicted_branch is None or (old.get('predicted_branch')==predicted_branch and old.get('runner_up')==runner_up)):
     return {**{k:old[k] for k in ['prediction','completion','option_logits','ranked'] if k in old},'reused':True,'inference_reused_after_parser_repair':True}
    return score(prompt,c['options'],intervention)
   for v in c['variant_rules']:
    labels=[chr(65+i) for i in range(len(c['branches']))];branches={}
    for visibility in ['hidden','visible','hide_inapp']:
     saved=branch_saved.get((id,v,'branch_'+visibility))
     if a.branches and saved is None:raise ValueError('Missing paired branch prediction')
     s={k:saved[k] for k in ['prediction','completion','option_logits','ranked']} if saved else score(branch_prompt(c,v,visibility),labels)
     if saved:s['reused']=True
     branches[visibility]=s;record(v,'branch_'+visibility,s,'branch',oracle=visibility=='hide_inapp')
    for method in ['baseline','prune','app','signed']:
     old=existing.get((id,v,method,False))
     if old:
      s={'prediction':old['prediction'],'completion':old.get('completion',''),'option_logits':old['option_logits'],'reused':True};record(v,'existing_'+method,s,'action',oracle=method!='baseline')
    if a.phase=='branch':continue
    # D and CS have no existing final answer; direct and gold-single are generated now.
    if v not in 'ABC':
     record(v,'existing_baseline',action(v,'existing_baseline',final_prompt(c,v)),'action',oracle=False)
     single=c['app_rule'] if v!='CS' else c['sib_rule'];record(v,'existing_prune',action(v,'existing_prune',prune_prompt(c,v,single=single)),'action',oracle=True)
    if v not in 'ABC':
     prompt=final_prompt(c,v);end=prompt.index('Patient:\n');starts=[m.start() for m in re.finditer('IF the clinical sequence is: ',prompt[:end])];spans=[[x,(starts[i+1] if i+1<len(starts) else end)-2] for i,x in enumerate(starts)]
     positive=c['app_rule'] if v!='CS' else c['sib_rule'];other=c['sib_rule'] if v!='CS' else c['app_rule']
     for oracle_method,og in [('app',0.),('signed',-gain)]:
      intervention={'spans':spans,'positive':[positive],'other':[other],'other_gain':og};record(v,'existing_'+oracle_method,action(v,'existing_'+oracle_method,prompt,intervention),'action',oracle=True)
    record(v,'gold_branch_prune',action(v,'gold_branch_prune',prune_prompt(c,v,bs=[c['gold_branch'][v]])),'action',oracle=True)
    for visibility in ['visible','hidden']:
     s=branches[visibility];pred=s['prediction'];first=labels.index(pred) if pred in labels else None;ranking=[labels.index(x) for x in s['ranked']];other=next((x for x in ranking if x!=first),None)
     for mode in (['prune'] if visibility=='visible' else ['prune','top2','signed','unsigned']):
      method=visibility+'_'+mode
      if first is None:
       record(v,method,{'prediction':None,'completion':'','option_logits':{},'upstream_branch_parse_failure':True},'action',oracle=False);continue
      if mode in ['prune','top2']:
       bs=[first] if mode=='prune' or other is None else [first,other];s2=action(v,method,prune_prompt(c,v,bs=bs),predicted_branch=first,runner_up=other)
      else:
       prompt=final_prompt(c,v);starts=[m.start() for m in re.finditer('IF the clinical sequence is: ',prompt[:prompt.index('Patient:\n')])];end=prompt.index('Patient:\n');spans=[[x,(starts[i+1] if i+1<len(starts) else end)-2] for i,x in enumerate(starts)];assert len(spans)==len(c['variant_rules'][v]);intervention={'spans':spans,'positive':c['branches'][first]['rule_indices'],'other':c['branches'][other]['rule_indices'] if other is not None else [],'other_gain':-gain if mode=='signed' else gain};s2=action(v,method,prompt,intervention,predicted_branch=first,runner_up=other)
      record(v,method,s2,'action',oracle=False,predicted_branch=first,runner_up=other)
   print(json.dumps({'cases_done':ci+1,'rows':n,'seconds':time.time()-start}),flush=True)
 out.with_suffix('.manifest.json').write_text(json.dumps({'complete':True,'cases':len(cases),'rows':n,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'gain':gain,'decoding':'greedy 8 tokens; top2 ranking from decision logits; no oracle fallback','seconds':time.time()-start,'model':a.model},indent=2))
if __name__=='__main__':main()
