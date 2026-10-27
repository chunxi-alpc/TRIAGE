from model_input import encode_prompts
"""P2 minimal action substitutions and controls, restricted to P1 dev SW cases."""
import argparse,collections,copy,hashlib,json,os,random,re,time
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_p1 import render,sha,dump
BASE=Path(__file__).resolve().parent

def messages(text):return [{'role':'system','content':'Select a branch by matching its clinical conditions to the patient. Follow the supplied experimental rules. Output only 1 or 2.'},{'role':'user','content':text}]
def main():
 from frozen_guard import check
 check()
 p=argparse.ArgumentParser();p.add_argument('--name',choices=['qwen','llama'],required=True);p.add_argument('--model',required=True);a=p.parse_args()
 assert json.loads((BASE/'results/gate_status.json').read_text())['P1']=='PASS_SW_GATE'
 out=BASE/'private'/('p2_'+a.name);out.mkdir(exist_ok=False);start=time.time();torch.manual_seed(20261027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
 rows=[json.loads(s) for s in (BASE/'private/cases.jsonl').read_text().splitlines()];sw=[r for r in rows if r['variant']=='SW'];nc={r['id']:r for r in rows if r['variant']=='NC'}
 assert len(sw)==json.loads((BASE/'results/data_audit.json').read_text())['base_count'] and all(r['split']=='dev' for r in sw)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left')
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 def chat(s):return tok.apply_chat_template(messages(s),tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def length(s):return len(tok.encode(s,add_special_tokens=False))
 pool=sorted({(r['tree'],q['action']) for r in rows if r['variant']=='NC' for q in r['rules']})
 jobs={};links=[];changes=[]
 for r in sw:
  g=int(r['gold'])-1;s=1-g;familiar=r['rules'][s]['action'];target_len=length(familiar)
  candidates=[v for tree,v in pool if tree!=r['tree'] and v not in r['options']]
  unfamiliar=min(candidates,key=lambda v:(abs(length(v)-target_len),hashlib.sha256((r['id']+v).encode()).hexdigest()))
  # Match full rendered visible prompt token count, including action boundary tokenization.
  neutral=None;base_len=length(chat(render(r,'visible')))
  for n in range(1,target_len+15):
   code='Q'+''.join(' '+chr(65+((i*7+int(hashlib.sha256(r['id'].encode()).hexdigest()[:4],16))%26)) for i in range(n-1))
   z=copy.deepcopy(r);z['rules'][s]['action']=code
   if length(chat(render(z,'visible')))==base_len:neutral=code;break
  if neutral is None:raise RuntimeError('Unable to match neutral prompt token count for '+r['id'])
  changes.append({'id':r['id'],'familiar':familiar,'unfamiliar_proxy':unfamiliar,'neutral':neutral,'familiar_tokens':target_len,'neutral_tokens':length(neutral),'full_visible_tokens':base_len})
  for kind,action in [('familiar',familiar),('unfamiliar',unfamiliar),('neutral',neutral)]:
   for control in ['standard','no_options','reversed']:
    z=copy.deepcopy(r);z['rules'][s]['action']=action
    if control=='reversed':z['rules'].reverse()
    for condition in ['blind','visible']:
     text=render(z,condition)
     if control=='no_options' and condition=='visible':text=text.split('\nTreatment options:')[0]+'\n\nWhich branch condition applies to this patient? Return only the branch number (1 or 2), without explanation.'
     prompt=chat(text);key=hashlib.sha256(prompt.encode()).hexdigest();jobs.setdefault(key,prompt)
     links.append({'model':a.name,'id':r['id'],'tree':r['tree'],'kind':kind,'control':control,'condition':condition,'gold':r['gold'],'prompt_sha256':key})
 (out/'substitutions.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in changes))
 manifest={'stage':'P2','status':'loading','name':a.name,'model':a.model,'script_sha256':sha(__file__),'protocol_sha256':sha(BASE/'p2_protocol.json'),'cases_sha256':sha(BASE/'private/cases.jsonl'),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'torch':torch.__version__,'transformers':transformers.__version__,'seed':20261027,'batch':4,'dtype':'float32','max_new_tokens':8,'do_sample':False,'thinking':False,'case_count':len(sw),'unique_branch_prompts':len(jobs),'logical_branch_evaluations':len(links),'closed_book_count':len(sw),'unfamiliar_definition':'different-development-tree source action outside original options; operational proxy, not verified model unfamiliarity','neutral_matching':'exact full visible prompt token count matched per model; only sibling action changed','weight_identity_manifest':a.name+'_manifest.json'}
 dump(out/'manifest.json',manifest)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
 closed=[]
 for r in sw:
  n=nc[r['id']];options=n['options'];original=n['rules'][int(n['gold'])-1]['action'];gold=options.index(original)
  text='Patient:\n'+n['patient']+'\n\n'+n['source_question']+'\nOptions:\n'+'\n'.join(chr(65+i)+': '+v for i,v in enumerate(options))+'\nReturn only the option letter.'
  prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=[tok.encode(chr(65+i),add_special_tokens=False) for i in range(len(options))];assert all(len(x)==1 for x in ids)
  x=encode_prompts(tok,prompt,return_tensors='pt').to('cuda')
  with torch.inference_mode():logits=model(**x).logits[0,-1].float();probs=logits[[v[0] for v in ids]].softmax(0).cpu().tolist()
  closed.append({'model':a.name,'id':r['id'],'tree':r['tree'],'original_option_index':gold,'option_probabilities':probs,'original_action_probability':probs[gold],'argmax_index':max(range(len(probs)),key=probs.__getitem__),'argmax_matches_original':max(range(len(probs)),key=probs.__getitem__)==gold,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
 for i,r in enumerate(sorted(closed,key=lambda r:(r['original_action_probability'],r['id']))):r['stratum']=['low','middle','high'][min(2,3*i//len(closed))]
 (out/'closed_book.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in closed))
 (out/'prompts.jsonl').write_text(''.join(json.dumps({'sha256':k,'text':v},ensure_ascii=False)+'\n' for k,v in jobs.items()))
 results={};keys=list(jobs)
 with (out/'generations.jsonl').open('w',buffering=1) as f:
  for offset in range(0,len(keys),4):
   kk=keys[offset:offset+4];x=encode_prompts(tok,[jobs[k] for k in kk],return_tensors='pt',padding=True,truncation=False).to('cuda');assert x.input_ids.shape[1]<=8192
   torch.cuda.synchronize();t=time.time()
   with torch.inference_mode():y=model.generate(**x,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
   torch.cuda.synchronize();elapsed=time.time()-t
   for i,k in enumerate(kk):
    ids=y[i,x.input_ids.shape[1]:].tolist();text=tok.decode(ids,skip_special_tokens=True).strip();m=re.fullmatch(r'\s*(?:Branch\s*)?([12])[.。]?\s*',text,re.I)
    z={'prompt_sha256':k,'raw_output':text,'output_token_ids':ids,'prediction':m[1] if m else None,'parse_failure':m is None,'input_tokens':int(x.attention_mask[i].sum()),'batch_seconds':elapsed,'batch_size':len(kk)};results[k]=z;f.write(json.dumps(z)+'\n')
   dump(out/'status.json',{'unique_completed':len(results),'unique_total':len(keys),'seconds':time.time()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
   print(json.dumps({'completed':len(results),'total':len(keys)}),flush=True)
 with (out/'predictions.jsonl').open('w') as f:
  for link in links:
   z={**link,**results[link['prompt_sha256']]};z['correct']=z['prediction']==z['gold'];f.write(json.dumps(z)+'\n')
 manifest.update(status='completed',wall_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
 dump(out/'manifest.json',manifest);dump(out/'DONE.json',{'status':'completed'})
if __name__=='__main__':main()
