"""Bounded data-only driver; never imports or starts experiment controllers."""
import subprocess,os,time,sys,traceback,shutil
from .core import *
def finalize():
 from .build import build
 summary=build('bulk')
 # Original prompts are compared directly, not hashed; previous results are only reusable for identical inputs.
 new={v['view_id']:v for v in rows(ROOT/'bulk/built/views.jsonl')}
 old={v['view_id']:v for v in rows(OLD/'data/views.jsonl') if v.get('original')}
 matched=[k for k,v in old.items() if k in new and new[k]['prompt']==v['prompt'] and new[k]['label']==v['label']]
 if len(matched)!=len(old):raise ValueError('original baseline input changed')
 write(ROOT/'cache_reuse.json',dict(original_full_views=len(old),identical_full_views=len(matched),reusable_view_ids=matched,activation_score_condition='same model checkpoint, scoring protocol and template',recompute=['all changed input scores and activations','affected localization','training on repaired pairs','development selection'],downstream='paused'))
 for name in ('views.jsonl','pairs.jsonl','rules.jsonl','unavailable.jsonl','extraction_review.jsonl'):
  shutil.copyfile(ROOT/'bulk/built'/name,ROOT/'data'/name)
 write(ROOT/'data/variants.json',summary)
 write(ROOT/'data/frozen.json',dict(version='repair-v2',fixed_at=time.strftime('%Y-%m-%d %H:%M:%S'),construction_complete=True,test_scored=False,source_splits=read(OLD/'data/frozen.json').get('sources'),original_nodes=read(ROOT/'manifest.json')['original_nodes'],original_paths=read(ROOT/'manifest.json')['paths'],external_in_training=False,downstream='paused'))
 write(ROOT/'status.json',dict(state='data_complete_downstream_paused',completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),summary=summary))
 from .report import report
 report()
 return summary

def run():
 root=ROOT;workers=[];start=time.time();(root/'logs').mkdir(exist_ok=True)
 for shard,gpu in enumerate([0,1,6]):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false')
  log=open(root/'logs'/f'bulk_{shard}.log','a')
  p=subprocess.Popen([sys.executable,'-u','-m','experiments.aster_cr.repair.extract','--scope','bulk','--shard',str(shard),'--batch','8'],env=env,stdout=log,stderr=subprocess.STDOUT)
  workers.append((p,gpu));log.close()
 write(root/'bulk_processes.json',[dict(pid=p.pid,gpu=gpu) for p,gpu in workers]);write(root/'status.json',dict(state='bulk_construction_running',pid=os.getpid(),started_at=time.strftime('%Y-%m-%d %H:%M:%S'),downstream='paused'))
 codes=[dict(pid=p.pid,gpu=gpu,exit_code=p.wait()) for p,gpu in workers]
 if any(x['exit_code'] for x in codes):
  write(root/'status.json',dict(state='data_worker_failed',workers=codes,downstream='paused'));return 1
 try:finalize()
 except Exception as exc:
  traceback.print_exc();write(root/'status.json',dict(state='data_finalization_failed',error=str(exc),downstream='paused'));return 1
 write(root/'runtime.json',dict(seconds=time.time()-start,workers=codes));return 0
if __name__=='__main__':raise SystemExit(run())
