"""Separate CAD128 variant with a relative conditional-probability filter; preserve all originals."""
from delegate_pending import wait_if_delegated
if __name__=='__main__':wait_if_delegated()
import argparse,ast,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run import encode_prompt,parse_answer,prompt_spec

def key(r):return (r['id'],r['variant'],r['method'],r.get('assigned_wrong',False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--base',required=True);p.add_argument('--input',required=True);p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--output',required=True);p.add_argument('--plausibility-beta',type=float,default=.1);a=p.parse_args();assert 0<a.plausibility_beta<=1
 base=Path(a.base);archive=base.with_name(base.stem+'_8token_original.jsonl');source=archive if archive.exists() else base
 rows=[json.loads(x) for x in source.read_text().splitlines()];targets=[r for r in rows if r['method'] in ('cad','prune_cad')]
 dataset={(r['id'],r['variant']):r for r in map(json.loads,Path(a.input).read_text().splitlines())}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);done={}
 if out.exists():
  for line in out.read_text().splitlines():
   r=json.loads(line);done[key(r)]=r
 torch.set_num_threads(4);torch.manual_seed(20261022)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True).eval()
 eos=model.generation_config.eos_token_id
 eos=set(eos if isinstance(eos,list) else [eos if eos is not None else tok.eos_token_id])
 # Reuse exactly the original CAD scoring math; change only cap/EOS and record stop reason.
 tree=ast.parse(Path(__file__).with_name('run.py').read_text())
 fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='cad_score')
 code=ast.unparse(fn).replace('range(8)','range(max_new_tokens)').replace('token == tok.eos_token_id','token in eos_ids')
 code=code.replace('result.update(completion=text, parsed_prediction=parse_answer(text, options))','result.update(completion=text, parsed_prediction=parse_answer(text, options), generated_tokens=len(generated), max_new_tokens=max_new_tokens, stop_reason="eos" if generated[-1] in eos_ids else "length")')
 code=code.replace('logits = 2 * logc - logu', 'logits = 2 * logc - logu\n            candidate_logits = logits.masked_fill(logc < logc.max() + math.log(beta), float("-inf"))')
 code=code.replace('token = int(logits.argmax())','token = int(candidate_logits.argmax())')
 ns=dict(math=__import__('math'),beta=a.plausibility_beta,torch=torch,tok=tok,model=model,encode_prompt=encode_prompt,parse_answer=parse_answer,tokenids={k:tok.encode(k,add_special_tokens=False) for k in 'ABCDEFGH'},max_new_tokens=a.max_new_tokens,eos_ids=eos)
 exec(compile(code,'<original_cad_with_longer_output>','exec'),ns);score=ns['cad_score'];start=time.time()
 with out.open('a') as f:
  for old in targets:
   if key(old) in done:continue
   row=dataset[old['id'],old['variant']];content=row['prompt_text'] if old['method']=='cad' else prompt_spec(row,'prune_cad',old.get('assigned_wrong',False))[0]
   s=score(content,row['closed_book_prompt'],row['options']);r=dict(old);r.update(s);r['score_prediction']=s['prediction'];r['prediction']=s['parsed_prediction'];r['correct']=r['prediction']==r['gold_answer'];r['decoding_revision']='cad128_plausibility_filter';r['plausibility_beta']=a.plausibility_beta;r['margin_prior']=s['option_logits'][r['gold_answer']]-s['option_logits'][r['original_answer']];r['margin_sibling']=s['option_logits'][r['gold_answer']]-s['option_logits'][r['sibling_action_label']]
   f.write(json.dumps(r)+'\n');f.flush();done[key(r)]=r
   if len(done)%50==0:print(json.dumps({'done':len(done),'total':len(targets),'seconds':time.time()-start}),flush=True)
 assert len(done)==len(targets)
 out.with_suffix('.manifest.json').write_text(json.dumps({'complete':True,'rows':len(done),'max_new_tokens':a.max_new_tokens,'plausibility_beta':a.plausibility_beta,'eos_ids':sorted(eos),'seconds':time.time()-start,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'source':str(source),'decoding':'free greedy CAD with conditional-relative probability mask; raw option scores retained'},indent=2))
 print('Filtered CAD complete',flush=True)
if __name__=='__main__':main()
