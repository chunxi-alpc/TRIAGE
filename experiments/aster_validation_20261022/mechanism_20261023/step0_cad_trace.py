"""Frozen 18-case CAD diagnostic. Only run on an available authorized GPU.

No tuning; preserve old files. Full token trajectories stay outside Git.
Instrument the original CAD function; both choices are evaluated at each prefix.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def step_evidence(logc,logu,gold_id,eos_ids):
 raw=2*logc-logu;threshold=float(logc.max()+math.log(.1))
 mask=logc < logc.max()+math.log(.1)
 filtered=raw.masked_fill(mask,float('-inf'))
 original=int(raw.argmax());filtered_token=int(filtered.argmax())
 observed=sorted({original,filtered_token,gold_id,*eos_ids})
 return {'unfiltered_choice':original,'filtered_choice':filtered_token,
  'conditional_logp_max':float(logc.max()),'threshold':threshold,
  'candidate_ids':(~mask).nonzero(as_tuple=False).flatten().tolist(),
  'tracked_tokens':{str(t):{'logc':float(logc[t]),'logu':float(logu[t]),
   'cad_score':float(raw[t]),'masked':bool(mask[t])} for t in observed}}

def instrument(source):
 tree=ast.parse(source);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='cad_score')
 code=ast.unparse(fn)
 replacements={
  'range(8)':'range(128)',
  'token = int(logits.argmax())':
   "trace.append(dict(step=step, prefix_token_ids=list(generated), **step_evidence(logc, logu, gold_id, eos_ids)))\n            token = trace[-1]['filtered_choice' if filtered_mode else 'unfiltered_choice']",
  'token == tok.eos_token_id':'token in eos_ids',
  'result.update(completion=text, parsed_prediction=parse_answer(text, options))':
   "result.update(completion=text, parsed_prediction=parse_answer(text, options), token_ids=generated, stop_reason='eos' if generated[-1] in eos_ids else 'length')"}
 for before,after in replacements.items():
  if code.count(before)!=1:raise ValueError('original CAD code changed: '+before)
  code=code.replace(before,after)
 compile(code,'<instrumented-original-cad>','exec')
 return code

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
 p.add_argument('--model',default='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B')
 p.add_argument('--input',type=Path,default=Path('/home/gcx/aster_validation_20261022/private/run500/all.jsonl'))
 a=p.parse_args()
 if HERE.parents[2] in a.output.resolve().parents:raise ValueError('full traces must stay outside repository')
 # Reserve a fresh path before loading the model; never append to completed diagnostics.
 a.output.mkdir(parents=True,exist_ok=False)
 import sys
 sys.path.insert(0,str(HERE.parent))
 from run import encode_prompt,parse_answer
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 frozen=HERE/'results/step0_review/private_structure_summary.json';audit=json.loads(frozen.read_text())
 ids=audit['cad']['frozen_case_ids'];expected={r['id']:r for r in audit['cad']['cases']}
 if len(ids)!=18 or len(set(ids))!=18:raise ValueError('frozen CAD list changed')
 inputs={}
 for line in a.input.read_text().splitlines():
  r=json.loads(line)
  if r['id'] in ids and r['variant']=='B_applicable_rule_changed':
   if r['id'] in inputs:raise ValueError('duplicate B input')
   inputs[r['id']]=r
 if set(inputs)!=set(ids):raise ValueError('incomplete frozen B inputs')
 # Verify input provenance from the completed original CAD manifest.
 oldpath=Path(audit['cad']['raw_inventory'][0]['path'])
 oldmanifest=json.loads(oldpath.with_name('qwen_full_8token_original.manifest.json').read_text())
 if oldmanifest['input_sha256']!=sha(a.input):raise ValueError('original CAD input fingerprint mismatch')
 filtered_input=oldpath.parent/'cad200/all.jsonl'
 filtered_manifest=json.loads(oldpath.parent.joinpath('cad200/original_filtered.manifest.json').read_text())
 if sha(filtered_input)!=filtered_manifest['input_sha256']:raise ValueError('filtered CAD input fingerprint mismatch')
 filtered_rows={r['id']:r for r in map(json.loads,filtered_input.read_text().splitlines()) if r['id'] in ids and r['variant']=='B_applicable_rule_changed'}
 if set(filtered_rows)!=set(ids):raise ValueError('filtered B inputs incomplete')
 for id_ in ids:
  for field in ('prompt_text','closed_book_prompt','options','gold_answer'):
   if inputs[id_][field]!=filtered_rows[id_][field]:raise ValueError('old/filtered input differs: '+field)
 for r in audit['cad']['raw_inventory']:
  if sha(r['path'])!=r['sha256']:raise ValueError('historical CAD file changed')
 torch.set_num_threads(4);torch.manual_seed(20261022)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True).eval()
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos if eos is not None else tok.eos_token_id])
 source=HERE.parent/'run.py';code=instrument(source.read_text());summaries=[]
 for id_ in ids:
  row=inputs[id_];tokenids={k:tok.encode(k,add_special_tokens=False) for k in row['options']}
  if any(len(v)!=1 for v in tokenids.values()):raise ValueError('option labels must be single tokens')
  for filtered in (False,True):
   trace=[];ns=dict(torch=torch,tok=tok,model=model,encode_prompt=encode_prompt,
    parse_answer=parse_answer,tokenids=tokenids,gold_id=tokenids[row['gold_answer']][0],
    eos_ids=eos,filtered_mode=filtered,trace=trace,step_evidence=step_evidence)
   exec(compile(code,'<instrumented-original-cad>','exec'),ns)
   result=ns['cad_score'](row['prompt_text'],row['closed_book_prompt'],row['options'])
   mode='filtered' if filtered else 'old';digest=hashlib.sha256(json.dumps(result['completion'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
   summary=dict(id=id_,mode=mode,prediction=result['parsed_prediction'],gold=row['gold_answer'],
    generated_tokens=len(result['token_ids']),stop_reason=result['stop_reason'],completion_sha256=digest,
    matches_historical_completion=digest==expected[id_][mode+'_completion_sha256'])
   (a.output/f'{id_}_{mode}.json').write_text(json.dumps(dict(summary=summary,result=result,steps=trace)))
   summaries.append(summary)
 manifest=dict(complete=True,cases=18,decodes=36,alpha=1,beta=.1,max_new_tokens=128,
  input_sha256=sha(a.input),frozen_audit_sha256=sha(frozen),source_sha256=sha(source),script_sha256=sha(__file__),
  torch=torch.__version__,transformers=__import__('transformers').__version__,
  peak_gib=torch.cuda.max_memory_allocated()/2**30,trajectories=summaries,
  historical_reproduction_complete=all(r['matches_historical_completion'] for r in summaries))
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'complete':True,'historical_reproduction_complete':manifest['historical_reproduction_complete']}))
if __name__=='__main__':main()
