"""Fresh paired generation of branch IDs for P1 only; no subsequent stage launch."""
import argparse,collections,hashlib,json,os,platform,re,subprocess,time
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
BASE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def render(r,condition):
 lines=[]
 for q in r['rules']:
  s=f"Branch {q['bid']}: IF {q['condition']}"
  if condition=='visible':s+=f" THEN {q['action']}"
  quals=q.get('execution_qualifications',[])
  if quals:s+='\nExecution qualifications: '+'; '.join(quals)
  lines.append(s)
 s='Branch rules:\n'+'\n'.join(lines)+'\n\nPatient:\n'+(r.get('blind_patient',r['patient']) if condition=='blind' else r['patient'])
 if condition=='visible':
  s+='\nTreatment options:\n'+'\n'.join(chr(65+i)+': '+v for i,v in enumerate(r['options']))
 return s+'\n\nWhich branch condition applies to this patient? Return only the branch number (1 or 2), without explanation.'
def main():
 from frozen_guard import check
 check()
 p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['qwen','llama']);p.add_argument('--model',required=True);p.add_argument('--batch',type=int,default=4);a=p.parse_args()
 out=BASE/'private'/a.name;out.mkdir(parents=True,exist_ok=False)
 start=time.time();torch.manual_seed(20261027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
 rows=[json.loads(s) for s in (BASE/'private/cases.jsonl').read_text().splitlines()]
 assert all(r['split']=='dev' for r in rows)
 modelpath=Path(a.model)
 manifest={'stage':'P1','status':'loading','name':a.name,'base_commit':'781ee863784c60f1f6710ba5eede22bc8ee6b93c','model':a.model,'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name(0),'seed':20261027,'dtype':'float32','attention':'sdpa','batch':a.batch,'max_new_tokens':8,'do_sample':False,'qwen_enable_thinking':False,'cases_sha256':sha(BASE/'private/cases.jsonl'),'script_sha256':sha(__file__),'weights':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(modelpath.glob('*.safetensors'))},'config_sha256':sha(modelpath/'config.json')}
 dump(out/'manifest.json',manifest)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left')
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
 jobs={};links=[]
 for r in rows:
  for condition in ['blind','visible']:
   messages=[{'role':'system','content':'Select a branch by matching its clinical conditions to the patient. Follow the supplied experimental rules. Output only 1 or 2.'},{'role':'user','content':render(r,condition)}]
   prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   key=hashlib.sha256(prompt.encode()).hexdigest()
   jobs.setdefault(key,{'prompt':prompt,'condition':condition})
   links.append({'id':r['id'],'tree':r['tree'],'variant':r['variant'],'condition':condition,'gold':r['gold'],'prompt_sha256':key})
 keys=list(jobs);results={};manifest.update(status='running',paired_rows=len(rows),evaluations=len(links),unique_prompts=len(keys),load_seconds=time.time()-start)
 dump(out/'manifest.json',manifest)
 with (out/'prompts.jsonl').open('w') as f:
  for key,j in jobs.items():f.write(json.dumps({'sha256':key,**j},ensure_ascii=False)+'\n')
 with (out/'generations.jsonl').open('w',buffering=1) as f:
  for offset in range(0,len(keys),a.batch):
   kk=keys[offset:offset+a.batch];x=tok([jobs[k]['prompt'] for k in kk],return_tensors='pt',padding=True,truncation=False).to('cuda')
   assert x.input_ids.shape[1]<=8192,'P1 prompt unexpectedly exceeds 8192 tokens; no truncation allowed'
   torch.cuda.synchronize();t=time.time()
   with torch.inference_mode():y=model.generate(**x,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
   torch.cuda.synchronize();elapsed=time.time()-t
   for i,k in enumerate(kk):
    ids=y[i,x.input_ids.shape[1]:].tolist();text=tok.decode(ids,skip_special_tokens=True).strip()
    match=re.fullmatch(r'\s*(?:Branch\s*)?([12])[.。]?\s*',text,re.I)
    pred=match[1] if match else None
    z={'prompt_sha256':k,'raw_output':text,'output_token_ids':ids,'prediction':pred,'parse_failure':pred is None,'input_tokens':int(x.attention_mask[i].sum()),'batch_seconds':elapsed,'batch_size':len(kk),'batch_index':offset//a.batch}
    results[k]=z;f.write(json.dumps(z,ensure_ascii=False)+'\n')
   dump(out/'status.json',{'status':'running','unique_completed':len(results),'unique_total':len(keys),'seconds':time.time()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
   print(json.dumps({'completed':len(results),'total':len(keys),'seconds':round(time.time()-start,2)}),flush=True)
 with (out/'predictions.jsonl').open('w') as f:
  for link in links:
   z={**link,**results[link['prompt_sha256']]};z['correct']=z['prediction']==z['gold'];f.write(json.dumps(z,ensure_ascii=False)+'\n')
 manifest.update(status='completed',wall_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
 dump(out/'manifest.json',manifest);dump(out/'DONE.json',{'status':'completed','evaluations':len(links)})
if __name__=='__main__':main()
