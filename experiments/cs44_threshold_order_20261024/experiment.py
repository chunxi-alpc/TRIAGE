"""Frozen 44-case threshold x order experiment; only 109 uncached requests run."""
import argparse,collections,datetime,hashlib,json,sys,time
from pathlib import Path
EXP=Path(__file__).parent;ROOT=EXP.parents[1];OUT=EXP/'results';PRIVATE=Path('/home/gcx/cs44_threshold_order_20261024/private');OLDPRIVATE=Path('/home/gcx/qwen800_methods_20261023/private')
sys.path.insert(0,str(ROOT/'experiments/qwen800_methods_20261023'))
from reuse import key as inference_key

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def jsonl(p,rs):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def prepare():
 import run as previous
 planpath=ROOT/'experiments/qwen800_methods_20261023/results/proposed_cs_2x2_manifest.jsonl';plan=read(planpath);cases={c['id']:c for c in read(OLDPRIVATE/'cases.jsonl')};cbpath=ROOT/'experiments/medguide_expand_cs_20261023/results/strata.jsonl';cb={r['id']:r for r in read(cbpath)};cache={};sources={str(planpath):sha(planpath),str(OLDPRIVATE/'cases.jsonl'):sha(OLDPRIVATE/'cases.jsonl'),str(cbpath):sha(cbpath)}
 for p in [OLDPRIVATE/'reuse_cache.jsonl',OLDPRIVATE/'predictions0.inference.jsonl',OLDPRIVATE/'predictions1.inference.jsonl']:
  sources[str(p)]=sha(p)
  for r in read(p):cache[r['key']]={**r['result'],'cache_source':str(p)}
 allrows=[];saved=[];missing=[]
 for r in plan:
  c=cases[r['id']];v='A' if r['threshold_state']=='original' else 'CS';prompt=previous.old.prune_prompt(c,v,bs=r['branch_order']);k=inference_key(prompt,c['options']);assert k==r['request_key'];assert (k in cache)==r['already_cached'];assert r['gold_answer']==c['gold_answer'][v]
  actions=[{c['variant_rules'][v][i]['action'] for i in c['branches'][b]['rule_indices']} for b in r['branch_order']];assert all(len(a)==1 for a in actions) and actions[0]!=actions[1]
  first=[label for label,text in c['options'].items() if text in actions[0]];last=[label for label,text in c['options'].items() if text in actions[1]];assert len(first)==len(last)==1
  row={**r,'disease':c['disease'],'old_gold':c['gold_answer']['A'],'new_gold':c['gold_answer']['CS'],'closed_book_prediction':cb[c['id']]['prediction'],'first_action_label':first[0],'last_action_label':last[0],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()};allrows.append(row)
  if k in cache:saved.append({**row,**cache[k],'prediction':previous.old.parse(cache[k]['completion'],c['options']),'reused':True})
  else:missing.append({**row,'prompt':prompt,'options':c['options']})
 assert len(allrows)==176 and len(saved)==67 and len(missing)==109 and len({r['request_key'] for r in allrows})==176
 # A/CS differ only in conditions; order differs only by swapping the same two rules.
 for i in {r['id'] for r in allrows}:
  c=cases[i];assert c['gold_answer']['A']!=c['gold_answer']['CS'];assert all(a['action']==b['action'] for a,b in zip(c['variant_rules']['A'],c['variant_rules']['CS']))
 jsonl(OUT/'request_manifest.jsonl',allrows);jsonl(PRIVATE/'reused.jsonl',saved)
 for k in range(2):jsonl(PRIVATE/f'missing{k}.jsonl',missing[k::2])
 write(OUT/'provenance.json',{'source_head':'59b99b8083453ac2f3bc0de192c44d59a75091b2','cases':44,'K':20,'U':24,'cells':176,'reused':67,'new_requests':109,'shards':[55,54],'sources':sources,'protocol':'Qwen3-8B BF16 eager greedy 8 tokens; same encode_prompt, greedy_with_logits and parse as PR23/25; candidates fixed from CS; no candidate reranking.'});print('prepared: 44 cases, 67 reused, 109 missing, shard 55/54',flush=True)

def worker(shard):
 import torch,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 import run as previous
 inp=PRIVATE/f'missing{shard}.jsonl';out=PRIVATE/f'new{shard}.jsonl';requests=read(inp);done={r['request_key']:r for r in read(out)} if out.exists() else {};assert set(done)<={r['request_key'] for r in requests}
 modelpath='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B';config={'input_sha256':sha(inp),'script_sha256':sha(__file__),'model':modelpath,'torch':torch.__version__,'transformers':transformers.__version__,'attention':'eager','dtype':'bfloat16','max_new_tokens':8,'do_sample':False,'shard':shard};mp=PRIVATE/f'new{shard}.config.json'
 if mp.exists():assert json.loads(mp.read_text())==config
 else:write(mp,config)
 if len(done)==len(requests):return
 torch.set_num_threads(4);torch.manual_seed(20261023);start=time.time();tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval();calls=0
 with out.open('a') as f:
  for r in requests:
   if r['request_key'] in done:continue
   assert not r['already_cached'];enc,_=previous.encode_prompt(tok,r['prompt']);enc={k:v.to(model.device) for k,v in enc.items()}
   with torch.inference_mode():gen,z=previous.old.greedy_with_logits(model,enc,tok.eos_token_id)
   text=tok.decode(gen[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);vals={label:float(z[tok.encode(label,add_special_tokens=False)[0]]) for label in r['options']};result={k:v for k,v in r.items() if k not in ['prompt','options']};result.update({'completion':text,'prediction':previous.old.parse(text,r['options']),'option_logits':vals,'reused':False});f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();done[r['request_key']]=result;calls+=1
 print(json.dumps({'shard':shard,'new_calls':calls,'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True);write(OUT/f'worker{shard}.json',{**config,'complete':True,'rows':len(done),'new_calls_this_run':calls,'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'output_sha256':sha(out)})

def analyze():
 rows=read(PRIVATE/'reused.jsonl')+read(PRIVATE/'new0.jsonl')+read(PRIVATE/'new1.jsonl');plan=read(OUT/'request_manifest.jsonl');assert len(rows)==176 and len({r['request_key'] for r in rows})==176;assert {r['request_key'] for r in rows}=={r['request_key'] for r in plan};assert sum(not r['reused'] for r in rows)==109
 for k in range(2):m=json.loads((OUT/f'worker{k}.json').read_text());assert m['complete'] and m['output_sha256']==sha(PRIVATE/f'new{k}.jsonl')
 for r in rows:r.update({'correct':r['prediction']==r['gold_answer'],'matches_closed_book':r['prediction'] is not None and r['prediction']==r['closed_book_prediction'],'parse_failure':r['prediction'] is None})
 idx={(r['id'],r['threshold_state'],r['rule_order']):r for r in rows};case_rows=[]
 for i in sorted({r['id'] for r in rows}):
  rr=[idx[i,t,o] for t in ['original','changed'] for o in ['current','reversed']];ps=[r['prediction'] for r in rr];valid=all(p is not None for p in ps);fixed=valid and len(set(ps))==1;allcorrect=all(r['correct'] for r in rr);first=valid and all(r['prediction']==r['first_action_label'] for r in rr);last=valid and all(r['prediction']==r['last_action_label'] for r in rr)
  category='threshold_following_all4' if allcorrect else 'fixed_action_all4' if fixed else 'first_position_all4' if first else 'last_position_all4' if last else 'parse_failure' if not valid else 'mixed'
  rec={'id':i,'stratum':rr[0]['stratum'],'predictions':ps,'old_gold':rr[0]['old_gold'],'new_gold':rr[0]['new_gold'],'closed_book':rr[0]['closed_book_prediction'],'category':category,'fixed_closed_book_all4':fixed and ps[0]==rr[0]['closed_book_prediction'],'fixed_old_gold_all4':fixed and ps[0]==rr[0]['old_gold'],'fixed_new_gold_all4':fixed and ps[0]==rr[0]['new_gold'],'threshold_pairs':[],'order_pairs':[]}
  for o in ['current','reversed']:
   a,b=idx[i,'original',o],idx[i,'changed',o];rec['threshold_pairs'].append({'order':o,'correct_flip':a['correct'] and b['correct'],'answer_changed':a['prediction']!=b['prediction'],'both_valid':a['prediction'] is not None and b['prediction'] is not None})
  for t in ['original','changed']:
   a,b=idx[i,t,'current'],idx[i,t,'reversed'];rec['order_pairs'].append({'threshold':t,'answer_changed':a['prediction']!=b['prediction'],'both_valid':a['prediction'] is not None and b['prediction'] is not None,'current_correct':a['correct'],'reversed_correct':b['correct']})
  case_rows.append(rec)
 summary={'complete':True,'cases':44,'cells':176,'reused':67,'new_requests':109,'groups':{}}
 for st in ['all','K','U']:
  rr=[r for r in rows if st=='all' or r['stratum']==st];cc=[r for r in case_rows if st=='all' or r['stratum']==st];tp=[x for r in cc for x in r['threshold_pairs']];op=[x for r in cc for x in r['order_pairs']];cells={}
  for t in ['original','changed']:
   for o in ['current','reversed']:
    ss=[r for r in rr if r['threshold_state']==t and r['rule_order']==o];cells[t+'/'+o]={'n':len(ss),'correct':sum(r['correct'] for r in ss),'closed_book_matches':sum(r['matches_closed_book'] for r in ss),'old_gold_matches':sum(r['prediction']==r['old_gold'] for r in ss),'new_gold_matches':sum(r['prediction']==r['new_gold'] for r in ss),'parse_failures':sum(r['parse_failure'] for r in ss)}
  summary['groups'][st]={'n':len(cc),'cells':cells,'categories':dict(collections.Counter(r['category'] for r in cc)),'fixed_closed_book_all4':sum(r['fixed_closed_book_all4'] for r in cc),'fixed_old_gold_all4':sum(r['fixed_old_gold_all4'] for r in cc),'fixed_new_gold_all4':sum(r['fixed_new_gold_all4'] for r in cc),'threshold_pairs':len(tp),'correct_threshold_flips':sum(r['correct_flip'] for r in tp),'any_threshold_answer_changes':sum(r['answer_changed'] for r in tp),'order_pairs':len(op),'order_answer_changes':sum(r['answer_changed'] for r in op),'order_wrong_to_right':sum(not r['current_correct'] and r['reversed_correct'] for r in op),'order_right_to_wrong':sum(r['current_correct'] and not r['reversed_correct'] for r in op)}
 jsonl(OUT/'scores.jsonl',[{k:v for k,v in r.items() if k not in ['completion','option_logits']} for r in rows]);jsonl(OUT/'case_patterns.jsonl',case_rows);write(OUT/'summary.json',summary);write(OUT/'STATUS.json',{'status':'complete','cases':44,'cells':176,'reused':67,'new_requests':109,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','worker','analyze']);p.add_argument('--shard',type=int,choices=[0,1]);a=p.parse_args()
 if a.phase=='prepare':prepare()
 elif a.phase=='worker':worker(a.shard)
 else:analyze()
