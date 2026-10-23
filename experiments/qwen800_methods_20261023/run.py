"""PR23 inference methods on frozen 800 cases, with resumable complete-key accounting."""
import argparse,contextlib,hashlib,importlib,importlib.util,json,math,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/aster_medguide_20261022'))
from run_pilot import encode_prompt,signed_attention_patch,parse_answer
spec=importlib.util.spec_from_file_location('mechanism_pr23',ROOT/'experiments/aster_validation_20261022/mechanism_20261023/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
METHODS=['existing_baseline','existing_prune','existing_app','existing_signed','gold_branch_prune','visible_prune','hidden_prune','hidden_top2','hidden_signed','hidden_unsigned','CAD8','CAD128','CAD_filtered128']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--model',default='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(4);torch.manual_seed(20261023);start=time.time();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);cases=list(map(json.loads,Path(a.input).read_text().splitlines()));cases=cases[:a.limit] if a.limit else cases
 config={'input_sha256':sha(a.input),'limit':a.limit,'model':a.model,'runner_sha256':sha(__file__),'gain':0.5,'attention':'eager','tokens':8,'CAD_tokens':128,'CAD_beta':0.1,'torch':torch.__version__,'transformers':transformers.__version__}
 runpath=out.with_suffix('.run.json')
 if runpath.exists():assert json.loads(runpath.read_text())==config,'resume config differs'
 else:runpath.write_text(json.dumps(config,indent=2))
 done={}
 if out.exists():
  for s in out.read_text().splitlines():
   r=json.loads(s);key=(r['id'],r['variant'],r['method']);assert key not in done;done[key]=r
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval();modeling=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type);gain=.5
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos or tok.eos_token_id]);cache={};calls=0
 def score(prompt,labels,intervention=None):
  nonlocal calls
  key=json.dumps([prompt,list(labels),intervention]);
  if key in cache:return dict(cache[key],cache_reused=True)
  enc,masks=encode_prompt(tok,prompt,[] if intervention is None else intervention['spans']);enc={k:v.to(model.device) for k,v in enc.items()};ids={k:tok.encode(k,add_special_tokens=False) for k in labels};assert all(len(x)==1 for x in ids.values());patch=contextlib.nullcontext()
  if intervention:
   ma=sum((masks[i] for i in intervention['positive']),torch.zeros_like(masks[0]));mb=sum((masks[i] for i in intervention['other']),torch.zeros_like(masks[0]));patch=signed_attention_patch(modeling,gain,intervention['other_gain'],ma,mb)
  with patch,torch.inference_mode():gen,z=old.greedy_with_logits(model,enc,tok.eos_token_id)
  vals={k:float(z[ids[k][0]]) for k in labels};text=tok.decode(gen[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);calls+=1
  r={'prediction':old.parse(text,labels),'completion':text,'option_logits':vals,'ranked':sorted(vals,key=vals.get,reverse=True),'input_tokens':enc['input_ids'].shape[1]};cache[key]=r;return dict(r)
 def cad(prompt,closed,labels,beta):
  nonlocal calls
  key=('cad',prompt,closed,beta)
  if key in cache:return cache[key]
  enc,_=encode_prompt(tok,prompt);unc,_=encode_prompt(tok,closed);enc={k:v.to(model.device) for k,v in enc.items()};unc={k:v.to(model.device) for k,v in unc.items()};generated=[];cs=us=None;vals={}
  with torch.inference_mode():
   for step in range(128):
    if step==0:co=model(**enc,use_cache=True);uo=model(**unc,use_cache=True)
    else:
     ids=torch.tensor([[generated[-1]]],device=model.device);co=model(input_ids=ids,attention_mask=torch.ones((1,enc['input_ids'].shape[1]+step),device=model.device,dtype=torch.long),past_key_values=cs,use_cache=True);uo=model(input_ids=ids,attention_mask=torch.ones((1,unc['input_ids'].shape[1]+step),device=model.device,dtype=torch.long),past_key_values=us,use_cache=True)
    cs=co.past_key_values;us=uo.past_key_values;logc=co.logits[0,-1].float().log_softmax(-1);logu=uo.logits[0,-1].float().log_softmax(-1);z=2*logc-logu
    if step==0:vals={k:float(z[tok.encode(k,add_special_tokens=False)[0]]) for k in labels}
    if beta:z=z.masked_fill(logc<logc.max()+math.log(beta),float('-inf'))
    token=int(z.argmax());generated.append(token)
    if token in eos:break
  calls+=1;result={}
  for cap in [8,128]:
   ids=generated[:cap];text=tok.decode(ids,skip_special_tokens=True);result[cap]={'prediction':parse_answer(text,labels),'completion':text,'option_logits':vals,'generated_tokens':len(ids),'max_new_tokens':cap,'stop_reason':'eos' if ids[-1] in eos else 'length','input_tokens':enc['input_ids'].shape[1],'derived_from_same_decode':True}
  cache[key]=result;return result
 with out.open('a') as f:
  for ci,c in enumerate(cases):
   cache.clear()
   def record(v,m,kind,fn,**extra):
    key=(c['id'],v,m)
    if key in done:return done[key]
    s=fn();gold=chr(65+c['gold_branch'][v]) if kind=='branch' else c['gold_answer'][v];r={'id':c['id'],'disease':c['disease'],'variant':v,'method':m,'kind':kind,'stratum':c['stratum'],'gold':gold,'original_answer':c['original_answer'],'pair_protocol':c['pair_protocol'],'single_action_gold_branch':c['single_action_gold_branch'],**s,**extra};r['correct']=r['prediction']==gold;r['parse_failure']=r['prediction'] is None
    if kind=='action':
     r['original_action_selected']=r['prediction']==c['original_answer'];actions={c['variant_rules'][v][i]['action'] for i in c['branches'][c['gold_branch'][v]]['rule_indices']};r['branch_permitted_action']=r['prediction'] in [k for k,val in c['options'].items() if val in actions]
    f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();done[key]=r;return r
   for v in c['variant_rules']:
    labels=[chr(65+j) for j in range(len(c['branches']))];br={}
    for mode in ['hidden','visible','hide_inapp']:br[mode]=record(v,'branch_'+mode,'branch',lambda mode=mode:score(old.branch_prompt(c,v,mode),labels),oracle=mode=='hide_inapp')
    prompt=old.final_prompt(c,v);end=prompt.index('Patient:\n');starts=[x.start() for x in re.finditer('IF the clinical sequence is: ',prompt[:end])];spans=[[x,(starts[j+1] if j+1<len(starts) else end)-2] for j,x in enumerate(starts)];assert len(spans)==len(c['variant_rules'][v]);positive=c['gold_rule'][v];other=c['app_rule'] if v=='CS' else c['sib_rule'];others=[] if other is None else [other]
    for m,fn in [('existing_baseline',lambda:score(prompt,c['options'])),('existing_prune',lambda:score(old.prune_prompt(c,v,single=positive),c['options'])),('gold_branch_prune',lambda:score(old.prune_prompt(c,v,bs=[c['gold_branch'][v]]),c['options']))]:record(v,m,'action',fn,oracle=m!='existing_baseline')
    for m,og in [('existing_app',0.),('existing_signed',-gain)]:record(v,m,'action',lambda og=og:score(prompt,c['options'],{'spans':spans,'positive':[positive],'other':others,'other_gain':og}),oracle=True)
    for visibility in ['visible','hidden']:
     r=br[visibility];first=labels.index(r['prediction']) if r['prediction'] in labels else None;second=next((labels.index(k) for k in r['ranked'] if k!=r['prediction']),None)
     for mode in (['prune'] if visibility=='visible' else ['prune','top2','signed','unsigned']):
      m=visibility+'_'+mode
      if first is None:fn=lambda:{'prediction':None,'completion':'','upstream_branch_parse_failure':True}
      elif mode in ['prune','top2']:
       bs=[first]+([second] if mode=='top2' and second is not None else []);fn=lambda bs=bs:score(old.prune_prompt(c,v,bs=bs),c['options'])
      else:
       intervention={'spans':spans,'positive':c['branches'][first]['rule_indices'],'other':c['branches'][second]['rule_indices'] if second is not None else [],'other_gain':-gain if mode=='signed' else gain};fn=lambda intervention=intervention:score(prompt,c['options'],intervention)
      record(v,m,'action',fn,oracle=False,predicted_branch=first,runner_up=second)
    for m,beta,cap in [('CAD8',None,8),('CAD128',None,128),('CAD_filtered128',.1,128)]:record(v,m,'action',lambda beta=beta,cap=cap:cad(prompt,c['closed_book_prompt'],c['options'],beta)[cap],oracle=False)
   print(json.dumps({'cases_done':ci+1,'rows':len(done),'calls':calls,'seconds':time.time()-start}),flush=True)
 expected={(c['id'],v,m) for c in cases for v in c['variant_rules'] for m in METHODS+['branch_hidden','branch_visible','branch_hide_inapp']};assert set(done)==expected
 out.with_suffix('.manifest.json').write_text(json.dumps({**config,'complete':True,'cases':len(cases),'rows':len(done),'calls':calls,'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'output_sha256':sha(out)},indent=2))
if __name__=='__main__':main()
