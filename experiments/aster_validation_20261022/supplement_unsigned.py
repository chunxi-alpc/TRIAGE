"""Full unsigned steering ablation; kept separate from the running 52-arm protocol."""
import argparse,hashlib,importlib,json,sys,time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/'aster_medguide_20261022'))
from run_pilot import encode_prompt,parse_answer,signed_attention_patch

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--gain-file',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(4);torch.manual_seed(20261022);start=time.time()
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True).eval()
 modeling=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type)
 gain=json.loads(Path(a.gain_file).read_text())['selected_gain'];strata={}
 for line in Path(a.base).read_text().splitlines():
  r=json.loads(line)
  if r['method']=='closed_book':strata[r['id']]=r['stratum']
 out=Path(a.output)
 if out.exists():raise FileExistsError(str(out))
 n=0
 with out.open('w') as f:
  for line in Path(a.input).read_text().splitlines():
   row=json.loads(line);opts=row['options'];ids={k:tok.encode(k,add_special_tokens=False) for k in opts}
   if any(len(v)!=1 for v in ids.values()):raise ValueError('multitoken option labels')
   encoded,masks=encode_prompt(tok,row['prompt_text'],[row['applicable_char_span'],row['sibling_char_span']]);encoded={k:v.to(model.device) for k,v in encoded.items()}
   with signed_attention_patch(modeling,gain,gain,*masks) as state,torch.inference_mode():
    logits=model(**encoded,use_cache=False).logits[0,-1].float();lp=logits.log_softmax(-1)
    vals={k:float(logits[ids[k][0]]) for k in opts};probs={k:float(lp[ids[k][0]]) for k in opts}
    gen=model.generate(**encoded,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
    text=tok.decode(gen[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
   r={k:row[k] for k in ('id','disease','source_row','variant','gold_answer','original_answer','sibling_action_label','replacement_action_label','replacement_elsewhere','condition_type')}
   r.update(method='unsigned',prediction=parse_answer(text,opts),parsed_prediction=parse_answer(text,opts),completion=text,score_prediction=max(vals,key=vals.get),option_logits=vals,option_logprobs=probs,margin_prior=vals[row['gold_answer']]-vals[row['original_answer']],margin_sibling=vals[row['gold_answer']]-vals[row['sibling_action_label']],stratum=strata[row['id']],gain=gain,experiment='injection_potential_outcomes',input_tokens=encoded['input_ids'].shape[1],attention_calls=state['calls'],attention=[],rule_token_counts=[int(m.sum()) for m in masks]);r['correct']=r['prediction']==r['gold_answer']
   # Swapping the rule assignments leaves +g,+g identical: reuse exact deterministic output.
   for wrong in (False,True):f.write(json.dumps(dict(r,assigned_wrong=wrong))+'\n');n+=1
   f.flush()
   if n%600==0:print(json.dumps({'rows':n,'seconds':time.time()-start}),flush=True)
 out.with_suffix('.manifest.json').write_text(json.dumps({'complete':True,'rows':n,'model':a.model,'gain':gain,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'seconds':time.time()-start,'torch':torch.__version__,'transformers':transformers.__version__,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30},indent=2))
if __name__=='__main__':main()
