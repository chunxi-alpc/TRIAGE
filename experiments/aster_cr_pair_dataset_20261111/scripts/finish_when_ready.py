"""Wait for recorded completion flags, then run the deterministic finalizer once."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import collections,fcntl,json,pathlib,subprocess,sys,time
from parallel_generation import ROOT,BASE,save,now,status
def main():
 with (BASE/'finalizer_watch.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  started=now()
  while True:
   s=status()
   for w in range(3):
    jobs=['w%d_%s'%(w,k) for k in ['P','C','F']]
    if all((BASE/j/'review_completed.json').exists() and (BASE/j/'author_completed.json').exists() for j in jobs):
     p=BASE/('window_%d_complete.json'%w)
     if not p.exists():save(p,dict(window=w,jobs={j:dict(author=json.loads((BASE/j/'author_completed.json').read_text()),review=json.loads((BASE/j/'review_completed.json').read_text())) for j in jobs},recorded_at=now()))
   ready=all(j['review_complete'] and j['author_complete'] for j in s['jobs'].values())
   save(BASE/'finalizer_watch_status.json',dict(state='finalizing' if ready else 'waiting_for_reviews',started_at=started,updated_at=now(),authored=s['authored'],reviewed=s['reviewed'],total_tasks=s['planned_tasks'],model_calls=0,gpus=[],vram_mib=0))
   if ready:
    with (ROOT/'logs/automatic_finalization.log').open('w') as out:
     result=subprocess.run([sys.executable,str(SCRIPTS/'merge_generated_parallel.py'),'--finalize'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    save(BASE/'finalizer_watch_status.json',dict(state='complete' if result.returncode==0 else 'error',exit_code=result.returncode,started_at=started,finished_at=now(),log='logs/automatic_finalization.log',model_calls=0,gpus=[],vram_mib=0))
    return
   time.sleep(30)
if __name__=='__main__':main()
