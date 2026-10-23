"""Inspect unfiltered CAD first-token probabilities without changing existing runs."""
import argparse,json,math
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from run import encode_prompt,prompt_spec
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval()
rows=list(map(json.loads,Path(a.input).read_text().splitlines()))[:9];results=[]
for row in rows:
 for method in ['cad','prune_cad']:
  content=row['prompt_text'] if method=='cad' else prompt_spec(row,'prune_cad')[0]
  enc,_=encode_prompt(tok,content);unc,_=encode_prompt(tok,row['closed_book_prompt'])
  with torch.inference_mode():
   lc=model(**{k:v.to(model.device) for k,v in enc.items()},use_cache=False).logits[0,-1].float().log_softmax(-1)
   lu=model(**{k:v.to(model.device) for k,v in unc.items()},use_cache=False).logits[0,-1].float().log_softmax(-1)
   s=2*lc-lu;chosen=int(s.argmax());base=int(lc.argmax());mask=lc>=lc.max()+math.log(.1);filtered=int(s.masked_fill(~mask,float('-inf')).argmax())
   tokens={k:tok.encode(k,add_special_tokens=False)[0] for k in row['options']}
   results.append({'id':row['id'],'variant':row['variant'],'method':method,'conditional_top':tok.decode([base]),'cad_top':tok.decode([chosen]),'filtered_top':tok.decode([filtered]),'cad_top_logp_context':float(lc[chosen]),'cad_top_logp_closed':float(lu[chosen]),'conditional_top_logp_context':float(lc[base]),'conditional_top_logp_closed':float(lu[base]),'cad_top_relative_context_probability':float((lc[chosen]-lc[base]).exp()),'cad_option':max(tokens,key=lambda k:float(s[tokens[k]]))})
Path(a.output).write_text(json.dumps(results,indent=2));print('completed',len(results))
