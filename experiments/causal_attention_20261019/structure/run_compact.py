#!/usr/bin/env python3
"""Run only missing atoms through original Phi-4 + TrialMatchAI LoRA on GPU.
The model generates one constrained T/F/U token; condition_id is supplied by
input metadata and serialized into compact JSON. No gold or route label input.
"""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];sys.path.insert(0,str(REPO))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--requests',type=Path,default=HERE/'missing_requests.jsonl');ap.add_argument('--output',type=Path,default=HERE/'compact_generations.jsonl');ap.add_argument('--batch-size',type=int,default=8);args=ap.parse_args()
    requests=[json.loads(l) for l in open(args.requests)];done=set()
    if args.output.exists():
        done={json.loads(l)['request_key'] for l in open(args.output)}
    requests=[r for r in requests if r['request_key'] not in done]
    if not requests:return
    import torch
    from src.module1.care_v6 import TrialMatchAIAdapter
    start=time.perf_counter();device='cuda:0';torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    base='/home/gcx/models/care_module1_v5/phi-4';adapter='/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora'
    model=TrialMatchAIAdapter.from_pretrained(device=device,base_model=base,adapter_model=adapter)
    tokenizer=model.tokenizer;tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    label_ids={s:tokenizer.encode(s,add_special_tokens=False) for s in ('T','F','U')}
    assert all(len(v)==1 for v in label_ids.values()),label_ids
    allowed=[v[0] for v in label_ids.values()];inverse={v[0]:s for s,v in label_ids.items()}
    manifest={'status':'running','model':base,'adapter':adapter,'device':device,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name':torch.cuda.get_device_name(0),'python_executable':sys.executable,'requests':len(requests),
        'output_contract':'condition_id is input metadata; Phi generates exactly one constrained T/F/U token. JSON is serialized without copying criterion text.',
        'generation':{'batch_size':args.batch_size,'max_new_tokens':1,'do_sample':False,'allowed_token_ids':label_ids},
        'reference_labels_in_model_input':False}
    (HERE/'compact_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with args.output.open('a',buffering=1) as stream:
        for i in range(0,len(requests),args.batch_size):
            batch=requests[i:i+args.batch_size];prompts=[];ids=[]
            for r in batch:
                cid=r['uses'][0]['condition_id'];ids.append(cid)
                messages=[{'role':'system','content':'Classify the truth of one atomic proposition using only the current patient observations. T means explicitly true; F means explicitly false; U means missing or conflicting current patient information. Historical or family-member observations do not establish the current patient state. The criterion is a proposition, not an eligibility decision. Give only the requested compact JSON; no explanation.'},
                    {'role':'user','content':'Patient: '+r['patient']+'\nAtomic proposition '+cid+': '+r['condition']['text']+'\nReturn {"condition_id":"'+cid+'","state":"T|F|U"}.'}]
                prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+'{"condition_id":"'+cid+'","state":"'
                prompts.append(prompt)
            encoded=tokenizer(prompts,padding=True,return_tensors='pt').to(device)
            with torch.inference_mode():
                generated=model.model.generate(**encoded,max_new_tokens=1,do_sample=False,
                    prefix_allowed_tokens_fn=lambda batch_id,input_ids:allowed,pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            tokens=generated[:,encoded['input_ids'].shape[1]:].tolist()
            for r,cid,prompt,token in zip(batch,ids,prompts,tokens):
                state=inverse[token[0]]
                out={'request_key':r['request_key'],'condition_id':cid,'state':state,'parse_status':'accepted',
                    'raw_output':tokenizer.decode(token),'compact_json':{'condition_id':cid,'state':state},
                    'prompt':prompt,'criterion_kind_in_old_cache':r['criterion_kind'],'generated_token_ids':token,
                    'is_model_prediction':True,'max_new_tokens':1,'do_sample':False,'input_tokens':len(tokenizer.encode(prompt,add_special_tokens=False))}
                stream.write(json.dumps(out,ensure_ascii=False)+'\n')
            print(json.dumps({'completed':min(i+len(batch),len(requests)),'total':len(requests),'elapsed_seconds':time.perf_counter()-start}),flush=True)
    manifest.update(status='completed',elapsed_seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    (HERE/'compact_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
