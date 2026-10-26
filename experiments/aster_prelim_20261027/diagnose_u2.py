"""Separate hidden-action content leakage from sequence-shape and precision effects."""
import argparse,collections,copy,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from attention_engine import Engine
from run_p1 import dump,sha
BASE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--model',required=True);a=p.parse_args();out=BASE/'private'/('u2_diagnosis_'+a.name);out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(20261027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 old=[json.loads(s) for s in (BASE/'results/p4'/(a.name+'_records.jsonl')).read_text().splitlines()]
 bad={r['id'] for r in old if not r['U2_all_four_agree']};controls=[]
 for tree in sorted({r['tree'] for r in old}):controls.append(next(r['id'] for r in old if r['tree']==tree and r['id'] not in bad))
 selected=sorted(bad|set(controls));data=collections.defaultdict(dict)
 for r in map(json.loads,(BASE/'private/cases.jsonl').read_text().splitlines()):
  if r['id'] in selected and r['variant'] in ['NC','IC','AC','SW']:data[r['id']][r['variant']]=r
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);records=[];start=time.time()
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval();eng=Engine(model,tok)
 for dtype in ['bfloat16','float32']:
  if dtype=='float32':model.float()
  for cid in selected:
   ps={v:eng.pack(data[cid][v]) for v in ['NC','IC','AC','SW']};canonical=[];variants={}
   for v,packed in ps.items():
    visible=[x for x,r in zip(packed['ids'],packed['roles']) if not r];canonical.append(visible)
    logits=eng.score(packed,'blind');pred,prob=eng.choose(logits,['1','2']);ix=[tok.encode(t,add_special_tokens=False)[0] for t in ['1','2']]
    variants[v]={'prediction':pred,'probabilities':prob,'logits':logits[ix].tolist(),'length':len(packed['ids'])}
   assert all(x==canonical[0] for x in canonical),'Non-action input content differs'
   pack=copy.deepcopy(ps['NC']);replacement=tok.encode('Z',add_special_tokens=False)[0]
   for i,r in enumerate(pack['roles']):
    if r:pack['ids'][i]=replacement
   logits=eng.score(pack,'blind');ix=[tok.encode(t,add_special_tokens=False)[0] for t in ['1','2']];altered=logits[ix].tolist()
   base=variants['NC']['logits'];maxfixed=max(abs(x-y) for x,y in zip(base,altered))
   maxvary=max(abs(z['logits'][i]-base[i]) for z in variants.values() for i in [0,1])
   z={'model':a.name,'dtype':dtype,'id':cid,'previously_inconsistent':cid in bad,'non_action_token_ids_identical':True,'variants':variants,'all_four_argmax_agree':len({x['prediction'] for x in variants.values()})==1,'fixed_shape_hidden_content_logit_max_difference':maxfixed,'variable_shape_logit_max_difference':maxvary};records.append(z)
 with (out/'records.jsonl').open('w') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 summary={'model':a.name,'previous_bad_count':len(bad),'controls':len(controls),'dtype_results':{},'elapsed_seconds':time.time()-start,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'script_sha256':sha(__file__),'engine_sha256':sha(BASE/'attention_engine.py')}
 for dtype in ['bfloat16','float32']:
  rr=[r for r in records if r['dtype']==dtype];summary['dtype_results'][dtype]={'n':len(rr),'all_four_agree_count':sum(r['all_four_argmax_agree'] for r in rr),'max_fixed_shape_content_logit_difference':max(r['fixed_shape_hidden_content_logit_max_difference'] for r in rr),'max_variable_shape_logit_difference':max(r['variable_shape_logit_max_difference'] for r in rr)}
 dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
