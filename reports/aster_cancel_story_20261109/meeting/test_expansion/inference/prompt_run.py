
"""Score a single fixed prompt candidate; no fitting, selection, or test tuning."""
import os,sys,time,json,argparse,traceback
from pathlib import Path
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.engine import Engine
from run import read_rows,dump,append
ROOT=Path('/home/gcx/aster_next_20261109')
def main(manifest,gpu,tag,reuse=None,candidate=None,outdir=None):
 import torch
 torch.set_num_threads(2);torch.cuda.set_device(gpu)
 out=Path(outdir) if outdir else ROOT/'prompt_results';out.mkdir(parents=True,exist_ok=True)
 target=out/(tag+'.jsonl');donefile=out/(tag+'.done.json')
 rs=read_rows(manifest);previous=read_rows(target,repair=True)
 def rid(r):return r.get('id',r.get('view_id'))
 done={rid(r) for r in previous if r.get('status') in ('ok','unavailable')}
 pending=[r for r in rs if rid(r) not in done]
 if not pending:dump(donefile,dict(state='complete',already_complete=True,rows=len(rs),tag=tag));return
 reuse_cache={}
 for item in reuse or []:
  source_manifest,source_scores=item.split('::',1)
  originals={rid(r):r for r in read_rows(source_manifest)}
  scorefile=Path(source_scores)
  if not scorefile.exists():continue
  lines=scorefile.read_text().splitlines()
  for line in lines:
   try:record=json.loads(line)
   except json.JSONDecodeError:continue
   key=rid(record)
   if key in originals and record.get('status')=='ok':
    reuse_cache[originals[key]['prompt']]=record
 started=time.time();engine=Engine('qwen',device='cuda:'+str(gpu));loaded=time.time()-started
 reused=0;forwarded=0
 for j,row in enumerate(pending):
  record={k:v for k,v in row.items() if k not in ('prompt','original_prompt')}
  before=time.perf_counter()
  try:
   if row['prompt'] in reuse_cache:
    prior=reuse_cache[row['prompt']]
    record.update({k:prior[k] for k in ('ell','prediction','tokens')})
    record.update(status='ok',inference_seconds=0.,reused_from=rid(prior));reused+=1
   else:
    record.update(engine.score(row['prompt'],capture=False),status='ok');forwarded+=1;reuse_cache[row['prompt']]=dict(record)
  except ValueError as exc:record.update(status='unavailable',error=str(exc))
  record.update(seconds=time.perf_counter()-before,gpu=gpu,prompt_candidate=candidate or row.get('method','fixed_prompt_candidate'))
  append(target,record)
  if j%10==0:dump(out/(tag+'.status.json'),dict(state='running',gpu=gpu,tag=tag,completed=len(done)+j+1,total=len(rs),
    seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(gpu)/2**30))
 dump(donefile,dict(state='complete',gpu=gpu,tag=tag,rows=len(rs),new_rows=len(pending),seconds=time.time()-started,
  model_load_seconds=loaded,reused_rows=reused,new_forwards=forwarded,counts=engine.counts,peak_gib=torch.cuda.max_memory_allocated(gpu)/2**30))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--gpu',required=True,type=int);p.add_argument('--tag',required=True);p.add_argument('--reuse',action='append',default=[]);p.add_argument('--candidate');p.add_argument('--outdir')
 a=p.parse_args()
 try:main(a.manifest,a.gpu,a.tag,a.reuse,a.candidate,a.outdir)
 except Exception as exc:
  out=Path(a.outdir) if a.outdir else ROOT/'prompt_results';out.mkdir(parents=True,exist_ok=True)
  dump(out/(a.tag+'.failed.json'),dict(state='failed',error=str(exc),traceback=traceback.format_exc()));raise

