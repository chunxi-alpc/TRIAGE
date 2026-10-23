"""Two deterministic, resumable Qwen closed-book shards; same parser as PR23."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'aster_medguide_20261022'))
from run_pilot import parse_answer

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--model',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.manual_seed(20261023);torch.set_num_threads(4);start=time.time()
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
 out=Path(a.output);done={json.loads(s)['id'] for s in out.read_text().splitlines()} if out.exists() else set();rows=[json.loads(s) for s in Path(a.input).read_text().splitlines()][a.shard::2]
 with out.open('a') as f:
  for r in rows:
   if r['id'] in done:continue
   text=tok.apply_chat_template([{'role':'user','content':r['closed_book_prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   enc=tok(text,return_tensors='pt',add_special_tokens=False).to(model.device)
   with torch.inference_mode():result=model.generate(**enc,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
   completion=tok.decode(result[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);pred=parse_answer(completion,r['options'])
   f.write(json.dumps({'id':r['id'],'tree':r['tree'],'prediction':pred,'completion':completion,'gold':r['original_answer'],'stratum':'K' if pred==r['original_answer'] else 'U'})+'\n');f.flush()
 m={'complete':True,'shard':a.shard,'cases':len(rows),'model':a.model,'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'torch':torch.__version__,'transformers':transformers.__version__,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256((Path(a.model)/'tokenizer.json').read_bytes()).hexdigest(),'decoding':'greedy 8 tokens, enable_thinking=False, PR23 parse_answer','attention':'sdpa (PR23 eager; numerical differences possible)'}
 out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m))
if __name__=='__main__':main()
