"""Check patient evidence before pruning; workers receive no evaluation labels."""
import argparse,hashlib,json,re,sys,time
from pathlib import Path
E=Path(__file__).parent;ROOT=E.parents[1];OUT=E/'results';P=Path('/home/gcx/cs44_explicit_check_20261024/private');Q=Path('/home/gcx/qwen800_methods_20261023/private');C=Path('/home/gcx/cs44_threshold_order_20261024/private')
sys.path.insert(0,str(ROOT/'experiments/qwen800_methods_20261023'))
from reuse import key as action_key
CHECK_INSTRUCTION='''Check the patient facts against the two supplied branch CONDITIONS before any treatment decision. Identify the relevant nodule type, number and measured size from the patient text. Explicitly compare that value with each supplied numeric threshold and consider any other conditions. Use the thresholds as written, even if they differ from familiar guidelines. Give a short factual check, then finish with exactly "Branch: A" or "Branch: B" for the condition that applies. If the patient facts do not resolve the choice, finish with "Branch: uncertain". Do not choose a treatment here.'''
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def jsonl(p,rs):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def parse_check(text):
 text=text.replace('**','').replace('`','')
 matches=re.findall(r'(?i)\bBranch\s*:\s*(A|B|uncertain)\b',text)
 return matches[-1].upper() if matches and matches[-1].upper() in ['A','B'] else None

def prepare():
 import run as prev
 old=ROOT/'experiments/cs44_threshold_order_20261024/results';plan=read(old/'request_manifest.jsonl');cases={c['id']:c for c in read(Q/'cases.jsonl')};patterns=read(old/'case_patterns.jsonl');focus={r['id'] for r in patterns if r['stratum']=='K' and r['fixed_closed_book_all4']};assert len(focus)==18
 cache={};sources={}
 for path in [Q/'reuse_cache.jsonl',Q/'predictions0.inference.jsonl',Q/'predictions1.inference.jsonl']:
  sources[str(path)]=sha(path)
  for r in read(path):cache[r['key']]={**{k:r['result'][k] for k in ['completion','option_logits'] if k in r['result']},'source':str(path)}
 for path in [C/'reused.jsonl',C/'new0.jsonl',C/'new1.jsonl']:
  sources[str(path)]=sha(path)
  for r in read(path):cache[r['request_key']]={**{k:r[k] for k in ['completion','option_logits'] if k in r},'source':str(path)}
 # Evaluation information and runnable requests deliberately live in separate files.
 evaluation=[];requests=[];route_keys=set();manifest=[]
 for r in plan:
  c=cases[r['id']];v='A' if r['threshold_state']=='original' else 'CS';bs=r['branch_order'];conditions=[c['variant_rules'][v][c['branches'][b]['rule_indices'][0]]['condition'] for b in bs]
  prompt=CHECK_INSTRUCTION+'\n\n'+c['patient']+'\n\nCandidate conditions:\nA: '+conditions[0]+'\nB: '+conditions[1]+'\n\nFactual check:'
  ck=hashlib.sha256(('Qwen3-8B|BF16|eager|greedy160|explicit-check-v1\n'+prompt).encode()).hexdigest();routes={}
  for tag,selected in [('A',[bs[0]]),('B',[bs[1]]),('uncertain',bs)]:
   final=prev.old.prune_prompt(c,v,bs=selected);k=action_key(final,c['options']);routes[tag]={'prompt':final,'key':k};route_keys.add(k)
  requests.append({'cell_key':r['request_key'],'check_key':ck,'check_prompt':prompt,'routes':routes,'options':c['options']})
  gold_slot='AB'[bs.index(c['gold_branch'][v])];evaluation.append({**r,'check_key':ck,'gold_slot':gold_slot,'focus18':c['id'] in focus})
  manifest.append({'cell_key':r['request_key'],'check_key':ck,'check_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'routes':{tag:{'key':z['key'],'cached':z['key'] in cache} for tag,z in routes.items()}})
 assert len(requests)==176 and len({r['check_key'] for r in requests})==176
 needed={k:cache[k] for k in route_keys if k in cache};jsonl(P/'action_cache.jsonl',[{'key':k,'result':v} for k,v in needed.items()]);jsonl(P/'evaluation.jsonl',evaluation)
 # A cell appears on only one worker; final action caches are local to each worker.
 assert all(len({x['id'] for x in plan[j:j+4]})==1 for j in range(0,176,4))
 for k in range(2):jsonl(P/f'requests{k}.jsonl',[r for j,r in enumerate(requests) if (j//4)%2==k])
 jsonl(OUT/'request_manifest.jsonl',manifest);jsonl(OUT/'evaluation_manifest.jsonl',evaluation)
 for path in [old/'request_manifest.jsonl',old/'scores.jsonl',old/'case_patterns.jsonl',Q/'cases.jsonl']:sources[str(path)]=sha(path)
 dump(OUT/'provenance.json',{'source_head':'3a3e9ff4ad5ce5559984b966a5798eb8855ae719','cases':44,'cells':176,'new_check_requests':176,'focus_K_cases':18,'possible_unique_final_requests':len(route_keys),'cached_final_requests':len(needed),'missing_possible_final_requests':len(route_keys)-len(needed),'sources':sources,'checker_inputs':['patient text','two conditions in displayed order'],'checker_excludes':['gold labels','K/U','focus18','old/new threshold flag','closed-book predictions','source paths','patient_value annotations','actions','answer options'],'uncertain_policy':'keep original two candidates; use existing final prompt/cache','check_instruction':CHECK_INSTRUCTION,'checker_max_tokens':160,'final_max_tokens':8});print(json.dumps({'checks':176,'route_keys':len(route_keys),'cached_routes':len(needed),'missing_possible_routes':len(route_keys)-len(needed)}),flush=True)

def worker(shard):
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 import run as prev
 reqpath=P/f'requests{shard}.jsonl';requests=read(reqpath);out=P/f'outputs{shard}.jsonl';done={r['cell_key']:r for r in read(out)} if out.exists() else {};journal=P/f'checks{shard}.jsonl';checks={r['key']:r['text'] for r in read(journal)} if journal.exists() else {};actioncache={r['key']:r['result'] for r in read(P/'action_cache.jsonl')};aj=P/f'actions{shard}.jsonl'
 if aj.exists():actioncache.update({r['key']:r['result'] for r in read(aj)})
 if set(done)=={r['cell_key'] for r in requests}:
  completed=json.loads((OUT/f'worker{shard}.json').read_text());assert completed['complete'] and completed['output_sha256']==sha(out);return
 config={'input_sha256':sha(reqpath),'cache_sha256':sha(P/'action_cache.jsonl'),'script_sha256':sha(__file__),'torch':torch.__version__,'transformers':transformers.__version__,'model':'/home/gcx/TRIAGE/models/modelscope/Qwen3-8B','dtype':'BF16','attention':'eager','checker_max_tokens':160,'final_max_tokens':8,'do_sample':False};cp=P/f'worker{shard}.config.json'
 if cp.exists():assert json.loads(cp.read_text())==config
 else:dump(cp,config)
 if len(done)==len(requests):return
 torch.set_num_threads(4);torch.manual_seed(20261023);start=time.time();tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval();checkcalls=finalcalls=0
 with out.open('a') as f,journal.open('a') as cj,aj.open('a') as af:
  for r in requests:
   if r['cell_key'] in done:continue
   ck=r['check_key']
   if ck not in checks:
    enc,_=prev.encode_prompt(tok,r['check_prompt']);enc={k:v.to(model.device) for k,v in enc.items()}
    with torch.inference_mode():gen=model.generate(**enc,do_sample=False,max_new_tokens=160,pad_token_id=tok.eos_token_id)
    text=tok.decode(gen[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);checks[ck]=text;cj.write(json.dumps({'key':ck,'text':text})+'\n');cj.flush();checkcalls+=1
   text=checks[ck];choice=parse_check(text);route=r['routes'][choice or 'uncertain'];key=route['key'];reused=key in actioncache
   if not reused:
    enc,_=prev.encode_prompt(tok,route['prompt']);enc={k:v.to(model.device) for k,v in enc.items()}
    with torch.inference_mode():gen,z=prev.old.greedy_with_logits(model,enc,tok.eos_token_id)
    actioncache[key]={'completion':tok.decode(gen[0,enc['input_ids'].shape[1]:],skip_special_tokens=True),'option_logits':{label:float(z[tok.encode(label,add_special_tokens=False)[0]]) for label in r['options']},'source':'new_final_inference'};af.write(json.dumps({'key':key,'result':actioncache[key]})+'\n');af.flush();finalcalls+=1
   res=actioncache[key];record={'cell_key':r['cell_key'],'check_key':ck,'check_completion':text,'selected_slot':choice,'fallback_two_rules':choice is None,'final_request_key':key,'final_reused':reused,'final_source':res.get('source','cache'),'prediction':prev.old.parse(res['completion'],r['options']),'completion':res['completion']};f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();done[r['cell_key']]=record
 print(json.dumps({'shard':shard,'check_calls':checkcalls,'new_final_calls':finalcalls,'rows':len(done),'seconds':time.time()-start}),flush=True);dump(OUT/f'worker{shard}.json',{**config,'complete':True,'rows':len(done),'check_calls':checkcalls,'new_final_calls':finalcalls,'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'output_sha256':sha(out)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','worker']);p.add_argument('--shard',type=int,choices=[0,1]);a=p.parse_args()
 if a.phase=='prepare':prepare()
 else:worker(a.shard)
