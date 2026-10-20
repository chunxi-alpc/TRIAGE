#!/usr/bin/env python3
"""Actual Phi-4+TrialMatchAI GPU inference for all rules in each narrative."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
from prompts import prediction_input, messages_for, parse_output

BASE='/home/gcx/models/care_module1_v5/phi-4'
ADAPTER='/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora'
PREFIX='{"rules":['

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--arm',choices=['A0','A1'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split');ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--max-new-tokens',type=int,default=384);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    examples=json.load(open(HERE/'data/fewshot_examples.json'))
    source=[json.loads(l) for l in open(args.cases)]
    if args.split:source=[r for r in source if r.get('split')==args.split]
    # Labels may be present in --cases (new72). They are never stored in the
    # model-facing object. Original data are only used for case IDs here.
    requests=[{'case_id':c['case_id'],'input':prediction_input(c)} for c in source]
    completed=set(); output=args.output/'predictions.jsonl'
    if output.exists():completed={json.loads(l)['case_id'] for l in open(output)}
    requests=[r for r in requests if r['case_id'] not in completed]
    if not requests:
        print(json.dumps({'status':'already_completed','cases':len(completed)}),flush=True);return
    import torch,transformers,peft,bitsandbytes
    from src.module1.care_v6 import TrialMatchAIAdapter
    start=time.perf_counter();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    manifest={'status':'loading','arm':args.arm,'base_model':BASE,'adapter_model':ADAPTER,
      'base_model_source':'microsoft/phi-4','adapter_source':'majdabd33/trialmatchai-phi4-reasoning-lora',
      'quantization':'same PR11 TrialMatchAIAdapter: NF4 4-bit, double quant, bfloat16 compute',
      'device':'cuda:0','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name(0),
      'python_executable':sys.executable,'input_path':str(args.cases.resolve()),'split':args.split,
      'total_cases':len(source),'cases_this_invocation':len(requests),'existing_cases':len(completed),
      'generation':{'batch_size':args.batch_size,'max_new_tokens':args.max_new_tokens,'do_sample':False,'num_beams':1,'assistant_prefix':PREFIX},
      'call_budget':{'primary_case_requests':len(requests),'max_retry_case_requests':len(requests),'max_retries_per_case':1,
      'max_generated_tokens':2*len(requests)*args.max_new_tokens,'primary_generate_batches':(len(requests)+args.batch_size-1)//args.batch_size},
      'prediction_input_fields':['question','patient','rules.rule_id','rules.text (whole natural-language rule; all adjacent branches)'],
      'no_patient_structured_facts_or_gold_in_prediction_input':True,'evidence_policy':'exact quote location is diagnostic only; unmatched quotes never change predicted applicability',
      'failure_policy':'one short format retry; persistent failures are null, never medical U',
      'prompt_sha256':hashlib.sha256((HERE/'prompts.py').read_bytes()).hexdigest(),
      'fewshot_sha256':hashlib.sha256((HERE/'data/fewshot_examples.json').read_bytes()).hexdigest() if args.arm=='A1' else None,
      'fewshot_demonstrations':3 if args.arm=='A1' else 0,'fewshot_patient_narratives':4 if args.arm=='A1' else 0,
      'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'bitsandbytes':bitsandbytes.__version__}}
    save(args.output/'manifest.json',manifest)
    model=TrialMatchAIAdapter.from_pretrained(device='cuda:0',base_model=BASE,adapter_model=ADAPTER)
    tok=model.tokenizer;tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    manifest.update(status='running',model_load_seconds=time.perf_counter()-start);save(args.output/'manifest.json',manifest)
    counts={'accepted_first':0,'accepted_retry':0,'parse_failure':0,'runtime_error':0,'case_requests':0,'generate_batches':0}
    total_generate_seconds=0.
    def generate(items, retry=False):
        nonlocal total_generate_seconds
        prompts=[]
        for item in items:
            messages=messages_for(item['input'],args.arm,examples)
            if retry:
                # Keep original context and output; retry only requests a short
                # valid answer. No reference, synthetic verdict, or critique.
                messages += [{'role':'assistant','content':item['attempts'][0]['raw_output']},
                 {'role':'user','content':'The answer did not match the JSON output format. Return the complete JSON object with exactly the supplied rule IDs, T/F/U states, short evidence_quote and reason_short strings. Keep quotes and reasons short. No extra text.'}]
            prompts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+PREFIX)
        enc=tok(prompts,padding=True,return_tensors='pt').to('cuda:0');torch.cuda.synchronize();t0=time.perf_counter()
        with torch.inference_mode():
            out=model.model.generate(**enc,max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1,
                pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t0;total_generate_seconds+=elapsed
        counts['case_requests']+=len(items);counts['generate_batches']+=1
        for item,prompt,tokens,mask in zip(items,prompts,out[:,enc['input_ids'].shape[1]:],enc['attention_mask']):
            ids=tokens.tolist();n=ids.index(tok.eos_token_id)+1 if tok.eos_token_id in ids else len(ids)
            raw=PREFIX+tok.decode(ids[:n],skip_special_tokens=True)
            parsed,error=parse_output(raw,[r['rule_id'] for r in item['input']['rules']])
            item.setdefault('attempts',[]).append({'prompt':prompt,'raw_output':raw,'parse_status':'accepted' if parsed is not None else 'failed',
              'error':error,'generation_seconds':elapsed/len(items),'batch_generation_seconds':elapsed,'batch_size':len(items),
              'input_tokens':int(mask.sum()),'output_tokens':n,'truncated':n==args.max_new_tokens and ids[n-1]!=tok.eos_token_id,
              'attempt':2 if retry else 1})
            item['parsed']=parsed
    try:
        with output.open('a',buffering=1) as stream:
            for i in range(0,len(requests),args.batch_size):
                batch=requests[i:i+args.batch_size]
                generate(batch)
                failed=[r for r in batch if r['parsed'] is None]
                if failed:generate(failed,retry=True)
                for item in batch:
                    parsed=item['parsed'];status='ok' if parsed is not None else 'parse_failure'
                    if parsed is None:
                        parsed=[{'rule_id':r['rule_id'],'state':None,'evidence_quote':'','reason_short':''} for r in item['input']['rules']]
                    for row in parsed:
                        quote=row['evidence_quote'];offset=item['input']['patient'].find(quote) if quote else -1
                        row['evidence_located']=offset>=0
                        row['optional_evidence_char_span']=[offset,offset+len(quote)] if offset>=0 else None
                    key='accepted_retry' if status=='ok' and len(item['attempts'])==2 else 'accepted_first' if status=='ok' else status
                    counts[key]+=1
                    record={'case_id':item['case_id'],'arm':args.arm,'status':status,'rules':parsed,
                      'model_call_count':len(item['attempts']),'inference_seconds':sum(a['generation_seconds'] for a in item['attempts']),
                      'time_definition':'amortized actual GPU batch generation wall time; excludes model load',
                      'attempts':item['attempts'],'is_model_prediction':True,'mode':'all_rules_per_case'}
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                print(json.dumps({'arm':args.arm,'completed':i+len(batch),'total':len(requests),'elapsed_seconds':time.perf_counter()-start,'counts':counts}),flush=True)
        manifest.update(status='completed',elapsed_seconds=time.perf_counter()-start,generation_seconds=total_generate_seconds,counts=counts,
           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),elapsed_seconds=time.perf_counter()-start,counts=counts)
        save(args.output/'manifest.json',manifest);traceback.print_exc();raise
    save(args.output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','arm','elapsed_seconds','generation_seconds','counts','peak_allocated_gib','peak_reserved_gib']}),flush=True)
if __name__=='__main__':main()
