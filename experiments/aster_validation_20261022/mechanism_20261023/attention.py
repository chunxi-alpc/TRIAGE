"""Conditional causal blocking: gate first, tune only on 40 development cases."""
import argparse,contextlib,importlib,json,random,re,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run import branch_prompt,parse,encode_prompt
from analyze import paired

@contextlib.contextmanager
def block(modeling,layer,heads,mask):
 original=modeling.eager_attention_forward
 def wrapper(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
  if module.layer_idx==layer:
   b,h,q,_=query.shape;k=key.shape[-2]
   if attention_mask is None:raise ValueError('causal mask required')
   adjusted=attention_mask[...,:k].expand(b,h,q,k).clone();indices=torch.where(mask[:min(k,len(mask))]>0)[0].to(query.device)
   for head in heads:adjusted[:,head,-1,indices]=torch.finfo(query.dtype).min
   attention_mask=adjusted
  return original(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
 modeling.eager_attention_forward=wrapper
 try:yield
 finally:modeling.eager_attention_forward=original

def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--dev',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 gate=json.loads(a.summary.read_text())['attention_gate']
 if not gate['triggered']:(a.out/'status.json').write_text(json.dumps({'executed':False,'reason':gate},indent=2));return
 if (a.out/'test.jsonl').exists():raise FileExistsError('Refuse overwrite attention experiment')
 torch.set_num_threads(4);torch.manual_seed(20261023);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval();modeling=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type);nh=model.config.num_attention_heads;nl=model.config.num_hidden_layers
 dev=list(map(json.loads,a.dev.read_text().splitlines()));test=list(map(json.loads,a.test.read_text().splitlines()));assert not {r['id'] for r in dev}&{r['id'] for r in test}
 def evaluate(c,layer=None,heads=None,control='action'):
  prompt=branch_prompt(c,'D','visible');spans=[[m.start(1),m.end(1)] for m in re.finditer(r'\nActions: (.*?)(?=\n\n[A-Z]: IF|\nReturn only|\Z)',prompt,re.S)];enc,masks=encode_prompt(tok,prompt,spans);mask=sum(masks,torch.zeros_like(masks[0]));mask=(mask>0).float();count=int(mask.sum())
  if control=='random_tokens':
   indices=random.Random('tokens:'+c['id']).sample(range(len(mask)),count);mask=torch.zeros_like(mask);mask[indices]=1
  use_heads=[(heads[0]+1)%nh] if control=='random_head' else heads
  ctx=block(modeling,layer,use_heads,mask) if layer is not None else contextlib.nullcontext();enc={k:v.to(model.device) for k,v in enc.items()}
  with ctx,torch.inference_mode():gen=model.generate(**enc,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
  text=tok.decode(gen[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);prediction=parse(text,[chr(65+i) for i in range(len(c['branches']))]);gold=chr(65+c['gold_branch']['D'])
  return {'id':c['id'],'disease':c['disease'],'prediction':prediction,'correct':prediction==gold,'parse_failure':prediction is None,'layer':layer,'heads':use_heads,'control':control,'blocked_tokens':count,'completion':text}
 trials=[]
 for layer in sorted(set([nl//4,nl//2,3*nl//4,nl-1])):
  rr=[evaluate(c,layer,list(range(nh))) for c in dev];trials.append({'layer':layer,'head':None,'correct':sum(r['correct'] for r in rr)});print('dev layer',layer,flush=True)
 bestlayer=max(trials,key=lambda r:(r['correct'],-r['layer']))['layer'];headscores=[]
 for head in range(nh):
  rr=[evaluate(c,bestlayer,[head]) for c in dev];headscores.append({'layer':bestlayer,'head':head,'correct':sum(r['correct'] for r in rr)})
 best=max(headscores,key=lambda r:(r['correct'],-r['head']));(a.out/'selection.json').write_text(json.dumps({'gate':gate,'dev_cases':len(dev),'layer_trials':trials,'head_trials':headscores,'selected':best,'criterion':'dev D branch accuracy, ties lowest index; no test tuning'},indent=2))
 rows=[]
 with (a.out/'test.jsonl').open('w') as f:
  for c in test:
   for control in ['baseline','action','random_tokens','random_head']:
    r=evaluate(c,None,None) if control=='baseline' else evaluate(c,bestlayer,[best['head']],control);r['control']=control;rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
 base=[r for r in rows if r['control']=='baseline'];comparisons={k:paired(base,[r for r in rows if r['control']==k]) for k in ['action','random_tokens','random_head']};(a.out/'summary.json').write_text(json.dumps({'test_cases':len(test),'selected':best,'comparisons':comparisons,'note':'Exploratory gated causal test, selected on dev only; report negative outcomes and damage to originally correct branches'},indent=2));(a.out/'status.json').write_text(json.dumps({'executed':True,'complete':True}))
if __name__=='__main__':main()
