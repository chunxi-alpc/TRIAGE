"""Development gain selection, then frozen same-input control matrix."""
import argparse,json,re,subprocess,time,traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.aster.manuscript_runtime import PromptBuilder,ManuscriptBackbone
from src.aster.manuscript_core import load_expert
from .analyze import load,gold
from .prepare import write
from .controls import execute,METHODS

def label(text):
 matches=set(re.findall(r'\b(?:R1|R2|NONE)\b',text.strip()))
 return next(iter(matches)) if len(matches)==1 else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--mode',choices=['gain','evaluate'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--resume',action='store_true');a=p.parse_args()
 run=Path(a.run);out=run/(('gain'+str(a.shard)) if a.mode=='gain' else 'evaluate'+str(a.shard));out.mkdir(exist_ok=a.resume)
 torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 rows=load(run);expert=load_expert(run/'analysis/expert.json');freeze=json.loads((run/'analysis/freeze_pre_gain.json').read_text())
 path='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B';tok=AutoTokenizer.from_pretrained(path,local_files_only=True);builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
 model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
 backend=ManuscriptBackbone(model,model_id='Qwen/Qwen3-8B',model_revision='local-modelscope-snapshot-20261014-upstream-revision-unavailable')
 # Numerical reference for the instance-local attention adapter, genuine nonempty expert at zero gain.
 ref=next(r for r in rows if r['row']['split']=='development');row=ref['row'];prompt=builder.prepare(row['guideline'],row['states']['positive'],row['question'],online=True)
 prefix=backend.prefill(prompt)
 with backend.session():
  cache=type(prefix.cache).from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in prefix.cache.to_legacy_cache()))
  native=backend._forward(prompt.input_ids[:,prompt.prefix_length:],cache=cache,start=prompt.prefix_length,use_cache=True).logits[0,-1].float().cpu()
 zero=execute(backend,builder,expert,freeze,ref,'positive','gold','zero',0.)
 biaszero=execute(backend,builder,expert,freeze,ref,'positive','gold','score_bias',0.)
 reference={'zero_wrapper_max_logit_error':float((native-zero['first_logits']).abs().max()),'zero_bias_wrapper_max_logit_error':float((native-biaszero['first_logits']).abs().max())}
 assert torch.equal(native,zero['first_logits']) and torch.equal(native,biaszero['first_logits'])
 native_generation=backend.generate(prompt,expert,{rid:0. for rid in prompt.positions},max_new_tokens=freeze['max_new_tokens'],prefix=prefix)
 assert native_generation['token_ids']==zero['token_ids'];reference['nonempty_expert_zero_generation_equal']=True
 write(out/'reference.json',reference)
 if a.mode=='gain':
  development=[r for r in rows if r['row']['split']=='development'];development=[r for i,r in enumerate(development) if i%2==a.shard];results=[]
  for gain in freeze['gain_candidates']:
   first=[];exact=[];failures=[]
   for r in development:
    for state in ['positive','negative']:
     try:
      result=execute(backend,builder,expert,freeze,r,state,'gold','spectral',gain,capture=False)
      target=gold(r,state);first.append(result['token_ids'][0]==tok(target,add_special_tokens=False)['input_ids'][0]);exact.append(label(result['text'])==target)
     except Exception as e:failures.append(str(e));first.append(False);exact.append(False)
   results.append({'gain':gain,'first_token_accuracy':sum(first)/len(first),'rule_accuracy':sum(exact)/len(exact),'denominator':len(first),'failures':failures})
  chosen=max(results,key=lambda r:(r['first_token_accuracy'],-r['gain']))['gain']
  freeze.update(g_plus=chosen,g_minus=chosen,gain_selection_results=results,frozen_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
  write(out/'frozen.json',freeze);write(out/'status.json',{'status':'complete','elapsed_seconds':time.time()-start,'gain':chosen,'results':results})
  return
 freeze=json.loads((run/'gain/frozen.json').read_text());cases=[]
 for r in rows:
  if r['row']['split']=='heldout_exploratory':
   for state in ['positive','negative','unknown']:cases.append((r,state))
 existing=[]
 results_path=out/'results.jsonl'
 if a.resume and results_path.exists():
  existing=[json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
 seen={(r['pair_id'],r['state'],r['route'],r['method']) for r in existing}
 if len(seen)!=len(existing):raise ValueError('Duplicate completed units in resume file')
 failures=sum(r['status']=='failed' for r in existing);done=len(existing)
 with results_path.open('a' if a.resume else 'x') as f:
  for i,(r,state) in enumerate(cases):
   if i%2!=a.shard:continue
   for route in ['gold','predicted']:
    for method in METHODS:
     if (r['row']['pair_id'],state,route,method) in seen:continue
     unit={'pair_id':r['row']['pair_id'],'source_id':r['row']['source_id'],'control':r['row'].get('control'),'state':state,'route':route,'method':method,'gold':gold(r,state)}
     try:
      result=execute(backend,builder,expert,freeze,r,state,route,method,freeze['g_plus'])
      result.pop('first_logits',None);prediction=label(result['text'])
      unit.update(status='complete',prediction=prediction,correct=prediction==unit['gold'],invalid_output=prediction is None,result=result)
     except Exception as e:
      failures+=1;unit.update(status='failed',correct=False,invalid_output=False,error_type=type(e).__name__,error=str(e));traceback.print_exc()
     f.write(json.dumps(unit,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done+=1
   write(out/'status.json',{'status':'running','completed_units':done,'failures':failures})
 write(out/'status.json',{'status':'complete','completed_units':done,'failures':failures,'elapsed_seconds':time.time()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9})
if __name__=='__main__':main()

