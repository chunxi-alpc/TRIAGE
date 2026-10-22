"""Decision-token scores, attention diagnostics, paired controls and noise experiments."""
import argparse,contextlib,hashlib,importlib,json,random,sys,time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/'aster_medguide_20261022'))
from run_pilot import encode_prompt,parse_answer,signed_attention_patch
GAINS=[0,.5,1,2,4,8,16]

@contextlib.contextmanager
def capture_attention(modeling,masks,store):
 original=modeling.eager_attention_forward
 def capture(module,*args,**kwargs):
  result=original(module,*args,**kwargs)
  weights=result[1][0,:,-1,:].detach().float()
  store.append({'layer':module.layer_idx,'mass':[float((weights[:,:len(m)]*m.to(weights.device)).sum(-1).mean()) for m in masks]})
  return result
 modeling.eager_attention_forward=capture
 try:yield
 finally:modeling.eager_attention_forward=original

def noisy_assignment(exid,rate,seed):
 # Coupled nested corruption across rates, shared by every method/variant.
 return random.Random(f'{seed}:{exid}').random()<rate

def prompt_spec(row,method,wrong=False):
 prompt=row['prompt_text'];app=row['applicable_char_span'];sib=row['sibling_char_span']
 selected,other=(sib,app) if wrong else (app,sib)
 rule=row['sibling_rule_text'] if wrong else row['applicable_rule_text']
 if method in ('prune','prune_cad'):
  return 'Follow this supplied rule even if it conflicts with medical memory:\n'+rule+'\n\n'+row['closed_book_prompt'],[],None
 if method in ('annotate','annotate_signed'):
  # Reference the occurrence by position, do not repeat the action text.
  number=prompt[:selected[0]].count('IF the clinical sequence is:')+1
  prompt+='\nThe applicable rule is rule '+str(number)+' in the displayed order.'
 if method=='whole':spans=[[prompt.index('IF the clinical sequence is:'),max(app[1],sib[1])]]
 elif method=='edited':spans=[app if row['variant'].startswith('B_') else sib]
 else:spans=[selected,other]
 return prompt,spans,None

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['diagnostic','main'],required=True);p.add_argument('--gain-file');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 input_bytes=Path(a.input).read_bytes()
 import torch,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.manual_seed(20261022);torch.set_num_threads(4);started=time.time()
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError('Refusing to mix runs: '+str(out))
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True).eval()
 modeling=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type)
 groups={}
 for line in input_bytes.decode().splitlines():
  r=json.loads(line);groups.setdefault(r['id'],[]).append(r)
 groups=list(groups.values());groups=groups[:a.limit] if a.limit else groups
 gain=1.
 if a.gain_file:gain=json.loads(Path(a.gain_file).read_text())['selected_gain']
 tokenids={k:tok.encode(k,add_special_tokens=False) for k in 'ABCDEFGH'}
 if any(len(v)!=1 for v in tokenids.values()):raise ValueError('Option labels are not single tokens; sequence scoring required')
 cache={};n=0;audits=[]
 def score(content,options,spans=(),strengths=None,diagnostic=False,generate=False):
  generate=generate or a.phase=='main'
  encoded,masks=encode_prompt(tok,content,spans);encoded={k:v.to(model.device) for k,v in encoded.items()}
  key=hashlib.sha256(json.dumps([content,list(options),spans,strengths,diagnostic,generate]).encode()).hexdigest()
  if key in cache:return cache[key]
  att=[];patch=signed_attention_patch(modeling,*strengths,*masks) if strengths is not None else contextlib.nullcontext({'calls':0})
  # Outer capture observes the ORIGINAL attention probabilities before patch restores.
  with capture_attention(modeling,masks,att) if diagnostic else contextlib.nullcontext():
   with patch as state,torch.inference_mode():
    logits=model(**encoded,use_cache=False).logits[0,-1].float()
    vals={k:float(logits[tokenids[k][0]]) for k in options};lp=logits.log_softmax(-1)
    result={'option_logits':vals,'option_logprobs':{k:float(lp[tokenids[k][0]]) for k in options},'prediction':max(vals,key=vals.get),'unconstrained_first_token':tok.decode([int(logits.argmax())]),'input_tokens':encoded['input_ids'].shape[1],'attention_calls':state['calls'],'attention':att,'rule_token_counts':[int(m.sum()) for m in masks]}
    if generate:
     output=model.generate(**encoded,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
     text=tok.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
     result.update(completion=text,parsed_prediction=parse_answer(text,options))
  cache[key]=result
  return result
 def cad_score(content,closed_content,options):
  # Decode CAD token by token with the same generated prefix in both contexts.
  enc,_=encode_prompt(tok,content);unc,_=encode_prompt(tok,closed_content)
  enc={k:v.to(model.device) for k,v in enc.items()};unc={k:v.to(model.device) for k,v in unc.items()}
  generated=[];cs=us=None;result={}
  with torch.inference_mode():
   for step in range(8):
    if step==0:
     co=model(**enc,use_cache=True);uo=model(**unc,use_cache=True)
    else:
     ids=torch.tensor([[generated[-1]]],device=model.device)
     co=model(input_ids=ids,attention_mask=torch.ones((1,enc['input_ids'].shape[1]+step),device=model.device,dtype=torch.long),past_key_values=cs,use_cache=True)
     uo=model(input_ids=ids,attention_mask=torch.ones((1,unc['input_ids'].shape[1]+step),device=model.device,dtype=torch.long),past_key_values=us,use_cache=True)
    cs=co.past_key_values;us=uo.past_key_values
    logc=co.logits[0,-1].float().log_softmax(-1);logu=uo.logits[0,-1].float().log_softmax(-1)
    logits=2*logc-logu;token=int(logits.argmax());generated.append(token)
    if step==0:
     vals={k:float(logits[tokenids[k][0]]) for k in options}
     result={'option_logits':vals,'option_logprobs':{k:float(logc[tokenids[k][0]]) for k in options},'prediction':max(vals,key=vals.get),'unconstrained_first_token':tok.decode([token]),'input_tokens':enc['input_ids'].shape[1],'attention_calls':0,'attention':[],'rule_token_counts':[],'score_kind':'CAD alpha=1, tokenwise 2 log p(context,prefix) - log p(closed,prefix)'}
    if token==tok.eos_token_id:break
  text=tok.decode(generated,skip_special_tokens=True)
  result.update(completion=text,parsed_prediction=parse_answer(text,options))
  return result
 f=out.open('w')
 def record(row,method,scores,**extra):
  nonlocal n
  r={k:row[k] for k in ('id','disease','source_row','variant','gold_answer','original_answer','sibling_action_label','replacement_action_label','replacement_elsewhere','condition_type') if k in row}
  r.update(method=method,**scores,**extra)
  r['score_prediction']=r['prediction']
  if a.phase=='main':r['prediction']=r.get('parsed_prediction')
  v=r['option_logits'];r['correct']=r['prediction']==r['gold_answer'];r['margin_prior']=v[r['gold_answer']]-v[r['original_answer']];r['margin_sibling']=v[r['gold_answer']]-v[r['sibling_action_label']]
  if extra.get('experiment')=='judgment':
   for k in ('gold_answer','original_answer','sibling_action_label','correct','margin_prior','margin_sibling','gold_is_placeholder'):r.pop(k,None)
  f.write(json.dumps(r)+'\n');f.flush();n+=1
 for gi,rows in enumerate(groups):
  original=next(r for r in rows if r['variant']=='A_original');opts=original['options']
  closed=score(original['closed_book_prompt'],opts,generate=gi<20)
  record(original,'closed_book',closed,stratum='K' if (closed.get('parsed_prediction') if a.phase=='main' else closed['prediction'])==original['original_answer'] else 'U')
  stratum='K' if (closed.get('parsed_prediction') if a.phase=='main' else closed['prediction'])==original['original_answer'] else 'U'
  for row in rows:
   base=score(row['prompt_text'],opts,generate=gi<20)
   record(row,'baseline',base,stratum=stratum)
   if a.phase=='diagnostic':
    if row['variant']=='A_original':continue
    spans=[row['applicable_char_span'],row['sibling_char_span']]
    for method in ('app','signed','negative'):
     for g in GAINS:
      strengths=(g,0.) if method=='app' else (g,-g) if method=='signed' else (0.,g)
      s=score(row['prompt_text'],opts,spans,strengths,True)
      record(row,method,s,gain=g,stratum=stratum,zero_max_logit_error=max(abs(s['option_logits'][k]-base['option_logits'][k]) for k in opts) if g==0 else None)
    for method in ('annotate','prune'):
     text,_,_=prompt_spec(row,method)
     record(row,method,score(text,opts,generate=gi<20),stratum=stratum)
   else:
    # Unknown applicability comparisons: whole guide, edited segment, CAD.
    guide_start=row['prompt_text'].index('IF the clinical sequence is:')
    guide_end=row['prompt_text'].index('Patient:\n')
    for method in ('whole','edited','cad'):
     if method=='cad':
      s=cad_score(row['prompt_text'],original['closed_book_prompt'],opts)
     else:
      span=[guide_start,guide_end] if method=='whole' else row['applicable_char_span'] if row['variant'].startswith('B_') else row['sibling_char_span']
      encoded,m=encode_prompt(tok,row['prompt_text'],[span]);zero=[0,0]
      # Second zero-length span would fail tokenizer validation: reuse nonoverlap patient span with zero strength.
      s=base if method=='edited' and row['variant']=='A_original' else score(row['prompt_text'],opts,[span,[guide_end,guide_end+7]],(gain,0.))
     record(row,method,s,gain=gain,stratum=stratum,experiment='unknown_applicability')
    for wrong in (False,True):
     for method in ('annotate','prune','prune_cad','app','signed','annotate_signed'):
      text,spans,_=prompt_spec(row,method,wrong)
      strengths=(gain,0.) if method=='app' else (gain,-gain) if method in ('signed','annotate_signed') else None
      s=score(text,opts,spans if strengths else (),strengths,generate=gi<20 and not wrong and method in ('annotate','prune'))
      if method=='prune_cad':
       s=cad_score(text,original['closed_book_prompt'],opts)
      record(row,method,s,gain=gain,stratum=stratum,assigned_wrong=wrong,experiment='injection_potential_outcomes')
  if a.phase=='main':
   # Same patient/condition, action concealed/aligned/conflicting. No oracle answer is inferred.
   patient=original['closed_book_prompt'].split('\n\nWhat is the appropriate next step?')[0]
   condition=original['condition_text']
   for exposure in ('hidden','aligned','conflicting'):
    text=patient+'\n\nDoes this patient satisfy the condition: '+condition+'?\n'
    if exposure!='hidden':text+='The proposed action for satisfying this condition is: '+(original['original_action'] if exposure=='aligned' else original['replacement_action'])+'.\n'
    text+='A: yes\nB: no\nC: unknown from the patient information\nReturn only one option letter.'
    s=score(text,'ABC',generate=gi<20)
    jr=dict(original);jr['gold_answer']='A';jr['original_answer']='A';jr['sibling_action_label']='B'
    # Gold is deliberately removed below: this is a flip diagnostic, not accuracy.
    s['judgment_only']=True
    before=n;record(jr,'judgment_'+exposure,s,stratum=stratum,experiment='judgment',gold_is_placeholder=True)
  cache.clear()
  print(json.dumps({'groups_done':gi+1,'rows':n,'seconds':time.time()-started}),flush=True)
 f.close()
 manifest={'phase':a.phase,'input_sha256':hashlib.sha256(input_bytes).hexdigest(),'model':a.model,'model_config':model.config.to_dict(),'model_file_sizes':{p.name:p.stat().st_size for p in Path(a.model).glob('*.safetensors')},'tokenizer_sha256':hashlib.sha256((Path(a.model)/'tokenizer.json').read_bytes()).hexdigest(),'torch':torch.__version__,'transformers':transformers.__version__,'groups':len(groups),'rows':n,'seconds':time.time()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'selected_gain':gain,'decoding':'greedy free generation for all main conditions, 8 tokens; separate first-position option scores','qwen_enable_thinking':False,'steering':'additive pre-softmax bias, all layers/heads/queries; NOT spectral SEKA','complete':True}
 out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
