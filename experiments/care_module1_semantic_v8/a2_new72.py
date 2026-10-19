#!/usr/bin/env python3
"""New-data A2: PR11 compact atomic Phi-4/TrialMatchAI predictions and fixed logic.
Input: authored rules+plain patient narrative; never load the gold file.
"""
import argparse,collections,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];sys.path.insert(0,str(REPO))
from src.module1.care_v6 import eval_logic,tri_and,tri_not
SYSTEM='Classify the truth of one atomic proposition using only the current patient observations. T means explicitly true; F means explicitly false; U means missing or conflicting current patient information. Historical or family-member observations do not establish the current patient state. The criterion is a proposition, not an eligibility decision. Give only the requested compact JSON; no explanation.'
def key(patient,text):return hashlib.sha256((patient+'\n'+text).encode()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def jsonl(p,rows):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,default=HERE/'data/new72_cases.jsonl');ap.add_argument('--output',type=Path,default=HERE/'results/new72_A2');ap.add_argument('--batch-size',type=int,default=8);args=ap.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'predictions.jsonl').exists():raise SystemExit('Completed results exist; refusing overwrite.')
 cases=[json.loads(l) for l in open(args.cases)];requests={}
 for case in cases:
  for rule in case['rules']:
   for a in rule['conditions']+rule['exceptions']:
    k=key(case['patient'],a['text'])
    if k not in requests:requests[k]={'request_key':k,'patient':case['patient'],'condition_id':a['condition_id'],'atom_text':a['text'],'uses':[]}
    requests[k]['uses'].append({'case_id':case['case_id'],'rule_id':rule['rule_id'],'condition_id':a['condition_id']})
 jsonl(out/'requests.jsonl',list(requests.values()))
 import torch
 from src.module1.care_v6 import TrialMatchAIAdapter
 device='cuda:0';torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
 base='/home/gcx/models/care_module1_v5/phi-4';adapter='/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora'
 start=time.perf_counter();model=TrialMatchAIAdapter.from_pretrained(device=device,base_model=base,adapter_model=adapter)
 tokenizer=model.tokenizer;tokenizer.padding_side='left'
 if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
 label_ids={s:tokenizer.encode(s,add_special_tokens=False) for s in ('T','F','U')};assert all(len(v)==1 for v in label_ids.values())
 allowed=[v[0] for v in label_ids.values()];inverse={v[0]:s for s,v in label_ids.items()}
 model_load_seconds=time.perf_counter()-start
 manifest={'status':'running','method':'A2_PR11_compact_atomic_plus_fixed_three_value_aggregation','model':base,'adapter':adapter,'device':device,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name(0),'python_executable':sys.executable,'cases':len(cases),'unique_atomic_requests':len(requests),'request_uses':sum(len(r['uses']) for r in requests.values()),'generation':{'batch_size':args.batch_size,'max_new_tokens':1,'do_sample':False,'allowed_token_ids':label_ids},'reference_labels_in_model_input':False,'structured_patient_facts_in_model_input':False,'system_prompt':SYSTEM,'input_advantage':'A2 receives author-supplied atomic rule decomposition and Boolean logic. A0/A1 receive the same complete rule meanings as natural text, without this decomposition.','relation_to_PR11':'Same Phi-4/TrialMatchAI LoRA adapter, compact constrained atomic-label interface and fixed eval_logic/tri_and/tri_not aggregation. Historical A2 reused old verbose caches, deterministic numeric processing and 17 compact repairs; all new atoms here use actual compact model generation. There are no numeric deterministic overrides.','evidence_note':'Compact atomic interface does not generate evidence quotes. Do not interpret empty evidence as a failed quote-location diagnostic or a semantic model comparison of evidence quality.','model_load_seconds':model_load_seconds}
 save(out/'manifest.json',manifest)
 predictions={};reqs=list(requests.values());inference_start=time.perf_counter()
 with (out/'raw_atomic_predictions.jsonl').open('w',buffering=1) as stream:
  for i in range(0,len(reqs),args.batch_size):
   batch=reqs[i:i+args.batch_size];prompts=[]
   for r in batch:
    cid=r['condition_id'];messages=[{'role':'system','content':SYSTEM},{'role':'user','content':'Patient: '+r['patient']+'\nAtomic proposition '+cid+': '+r['atom_text']+'\nReturn {"condition_id":"'+cid+'","state":"T|F|U"}.'}]
    prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+'{"condition_id":"'+cid+'","state":"')
   enc=tokenizer(prompts,padding=True,return_tensors='pt').to(device);torch.cuda.synchronize();batch_start=time.perf_counter()
   with torch.inference_mode():generated=model.model.generate(**enc,max_new_tokens=1,do_sample=False,prefix_allowed_tokens_fn=lambda batch_id,input_ids:allowed,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
   torch.cuda.synchronize();elapsed=time.perf_counter()-batch_start
   token_rows=generated[:,enc['input_ids'].shape[1]:].tolist()
   for r,prompt,tokens in zip(batch,prompts,token_rows):
    state=inverse.get(tokens[0]) if tokens else None
    row=dict(request_key=r['request_key'],condition_id=r['condition_id'],state=state,parse_status='accepted' if state else 'failed',raw_output=tokenizer.decode(tokens),compact_json={'condition_id':r['condition_id'],'state':state},prompt=prompt,generated_token_ids=tokens,is_model_prediction=True,input_tokens=len(tokenizer.encode(prompt,add_special_tokens=False)),batch_id=i//args.batch_size,batch_seconds=elapsed,amortized_generation_seconds=elapsed/len(batch))
    predictions[r['request_key']]=row;stream.write(json.dumps(row,ensure_ascii=False)+'\n')
   print(json.dumps({'completed':min(i+len(batch),len(reqs)),'total':len(reqs),'inference_elapsed_seconds':time.perf_counter()-inference_start}),flush=True)
 inference_seconds=time.perf_counter()-inference_start
 rows=[]
 for case in cases:
  rules=[];case_keys=set()
  for rule in case['rules']:
   states={};atom_records=[]
   for atom in rule['conditions']+rule['exceptions']:
    k=key(case['patient'],atom['text']);pred=predictions[k];case_keys.add(k);states[atom['condition_id']]=pred['state'];atom_records.append(dict(condition_id=atom['condition_id'],state=pred['state'],request_key=k))
   valid=all(v in ('T','F','U') for v in states.values())
   state=tri_and([eval_logic(rule['logic'],states),tri_not(eval_logic(rule['exception_logic'],states))]) if valid else None
   rules.append(dict(rule_id=rule['rule_id'],state=state,action=rule['action'],action_char_span=case['action_spans'][rule['rule_id']],evidence_quote='',reason_short='PR11 fixed three-valued aggregation of model atomic labels.',evidence_available=False,atoms=atom_records))
  rows.append(dict(case_id=case['case_id'],source='A2',status='ok' if all(r['state'] in ('T','F','U') for r in rules) else 'parse_failure',rules=rules,elapsed_seconds=sum(predictions[k]['amortized_generation_seconds'] for k in case_keys),model_call_count=len(case_keys),actual_unique_calls_note='A single physical atomic request can be reused by identical patient+proposition inputs; see raw_atomic_predictions.jsonl.'))
 jsonl(out/'predictions.jsonl',rows)
 import transformers,peft,bitsandbytes
 manifest.update(status='completed',inference_seconds=inference_seconds,elapsed_seconds=time.perf_counter()-start,actual_atomic_calls=len(predictions),actual_generation_batches=(len(reqs)+args.batch_size-1)//args.batch_size,parse_failures=sum(r['parse_status']!='accepted' for r in predictions.values()),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,versions={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'bitsandbytes':bitsandbytes.__version__})
 save(out/'manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
