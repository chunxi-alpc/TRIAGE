"""Supplement only missing MedPAIR option scores using the original scoring function."""
import argparse,os,sys,time,traceback
from common import *
sys.path.insert(0,str(REPO/'src'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','full'],required=True);p.add_argument('--shard',type=int,default=0);args=p.parse_args()
 import torch
 from represent_faithful.runtime import load_causal_lm,score_answer_choices,answer_choice_token_ids
 torch.set_num_threads(4);torch.manual_seed(42)
 data=rows(ROOT/'data/medpair_inputs.jsonl')
 done={}
 for path in (ROOT/'results').glob('medpair_scores_*.jsonl'):
  for r in rows(path):
   if r.get('status')=='complete':done[r['score_id']]=r
 todo=[r for i,r in enumerate(data) if (r['pilot'] if args.stage=='pilot' else i%2==args.shard) and r['score_id'] not in done]
 status=ROOT/f'results/scoring_{args.stage}_{args.shard}_status.json';out=ROOT/f'results/medpair_scores_{args.stage}_{args.shard}.jsonl'
 start=time.time();meta=dict(stage=args.stage,shard=args.shard,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),total=len(todo),reused_existing=len(done),model=str(MODEL),dtype='bfloat16',score_function='represent_faithful.runtime.score_answer_choices',new_annotations=0)
 write_json(status,dict(meta,status='loading_model',done=0))
 if not todo:write_json(status,dict(meta,status='complete',done=0));return
 loaded=load_causal_lm(str(MODEL),dtype='bfloat16',device_map={'':'cuda:0'})
 load_seconds=time.time()-start;torch.cuda.reset_peak_memory_stats();capture={}
 def save_logits(module,inputs,output):capture['last']=output.logits[0,-1].detach().float().clone()
 hook=loaded.model.register_forward_hook(save_logits)
 def update(state,n):
  write_json(status,dict(meta,status=state,done=n,model_load_seconds=load_seconds,elapsed_seconds=time.time()-start,
   scoring_seconds=time.time()-start-load_seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,attention=loaded.model.config._attn_implementation))
 with out.open('a') as handle:
  for i,r in enumerate(todo):
   started=time.time()
   try:
    result=score_answer_choices(loaded,r['prompt'],r['labels'],r['gold'],max_length=4096)
    token_ids=answer_choice_token_ids(loaded.tokenizer,r['labels']);z=capture.pop('last')[token_ids].cpu().tolist()
    logp=[result.normalized_log_probabilities[k] for k in r['labels']]
    assert abs(sum(np.exp(logp))-1)<1e-5
    pred=r['labels'][int(np.argmax(logp))]
    record={k:r[k] for k in ('score_id','id','split','condition','labels','gold','input_tokens')}
    record.update(status='complete',raw_logits=z,log_probabilities=logp,vocabulary_log_probabilities=[result.choice_log_likelihoods[k] for k in r['labels']],
     prediction=pred,correct=pred==r['gold'],option_token_ids=token_ids,seconds=time.time()-started,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush()
   except Exception:
    update('failed',i);raise
   if (i+1)%10==0:update('running',i+1)
 hook.remove();update('complete',len(todo))
if __name__=='__main__':main()
