"""Run the official-pair evaluation after the existing primary GPU job exits."""
import argparse,json,os,select,subprocess,sys,time,traceback,gzip,shutil
from pathlib import Path
BASE=Path('/home/gcx/aster_next_20261109');PY=Path('/home/gcx/gcx261014/bin/python')
PUB=Path('/home/gcx/aster-real-completion-publish-20261109');REPORT='reports/aster_cancel_story_20261109'
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def command(args,cwd=None):subprocess.run([str(x) for x in args],cwd=cwd,check=True)
def main(plan_path):
 plan=read(plan_path);out=Path(plan['results_root']);out.mkdir(parents=True,exist_ok=True)
 def status(phase,**kw):
  save(BASE/'faithfulness_pipeline_status.json',dict(phase=phase,utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),gpus=plan['gpus'],**kw))
 try:
  pid=int(plan['wait_for_pid'])
  status('waiting_for_primary',primary_pid=pid,expected_views=plan['expected_views'],expected_pairs=plan['expected_pairs'])
  # A kernel process-exit notification waits for our own job, without GPU polling.
  try:
   fd=os.pidfd_open(pid)
  except ProcessLookupError:fd=None
  if fd is not None:
   try:select.select([fd],[],[])
   finally:os.close(fd)
  primary=read(BASE/'primary_pipeline_status.json')
  if primary.get('phase') not in ('complete','complete_with_unavailable','computed_with_publication_errors'):
   status('dependency_incomplete',primary_status=primary);return
  status('running',expected_views=plan['expected_views'],expected_pairs=plan['expected_pairs'])
  processes=[];logs=[]
  for rank,gpu in enumerate(plan['gpus']):
   log=(out/f'worker{rank}.log').open('a',buffering=1);logs.append(log)
   args=[PY,BASE/'inference/run_primary.py','--cases',plan['cases'],'--out-root',out,'--manifest',plan['frozen_manifest'],'--methods',*plan['methods'],'--gpu',gpu,'--shard',rank,'--shards',len(plan['gpus']),'--reuse-root',plan['reuse_root']]
   processes.append(subprocess.Popen([str(x) for x in args],stdout=log,stderr=subprocess.STDOUT,cwd=BASE))
  save(out/'worker_pids.json',{str(i):dict(pid=p.pid,gpu=plan['gpus'][i]) for i,p in enumerate(processes)})
  codes=[p.wait() for p in processes]
  for log in logs:log.close()
  if any(codes):raise RuntimeError('Official paired worker exit codes '+str(codes)+'; existing records are resumable.')
  status('summarizing',expected_views=plan['expected_views'],expected_pairs=plan['expected_pairs'])
  command([PY,BASE/'analysis/summarize_mega_pairs.py','--cases',plan['cases'],'--pairs',plan['pairs'],'--nodes',*[out/f'shard{i}/nodes.jsonl' for i in range(len(plan['gpus']))],'--manifest',plan['frozen_manifest'],'--out',out/'summary','--methods',*plan['methods']])
  command([PY,BASE/'analysis/summarize_faithfulness.py'])
  publication_error=None;commit=None
  try:
   status('publishing')
   command(['git','fetch','origin','aster-cr-20261108'],cwd=PUB)
   command(['git','merge','--ff-only','origin/aster-cr-20261108'],cwd=PUB)
   command([PY,BASE/'analysis/update_faithfulness_documents.py'])
   packet=PUB/REPORT/'meeting/primary_full/faithfulness';packet.mkdir(parents=True,exist_ok=True)
   shutil.copy2(plan_path,packet/'official_pair_plan.json')
   code=packet/'code';code.mkdir(exist_ok=True)
   for rel in ['analysis/run_faithfulness_pipeline.py','analysis/summarize_mega_pairs.py','analysis/summarize_nli_pairs.py','inference/prepare_official_fidelity.py']:
    shutil.copy2(BASE/rel,code/Path(rel).name)
   target=packet/'mega_official_pairs';target.mkdir(exist_ok=True)
   for file in (out/'summary').iterdir():
    if file.is_file() and file.suffix in ('.json','.csv'):shutil.copy2(file,target/file.name)
   for rank in range(len(plan['gpus'])):
    with (out/f'shard{rank}/results.jsonl').open('rb') as src,gzip.open(target/f'results_shard{rank}.jsonl.gz','wb') as dst:shutil.copyfileobj(src,dst)
   if Path(plan['inventory']).exists():shutil.copy2(plan['inventory'],target/'inventory.json')
   command(['git','add','--',REPORT+'/paper_story_latest.zh.md',REPORT+'/meeting/meeting_brief.zh.md',REPORT+'/README.md',REPORT+'/meeting/primary_full/faithfulness'],cwd=PUB)
   diff=subprocess.run(['git','diff','--cached','--quiet'],cwd=PUB)
   if diff.returncode==1:command(['git','commit','-m','Evaluate controlled faithfulness on official MEGA nodes'],cwd=PUB)
   elif diff.returncode:raise RuntimeError('Unable to inspect prepared result commit')
   command(['git','push','origin','HEAD:aster-cr-20261108'],cwd=PUB)
   commit=subprocess.run(['git','rev-parse','--short','HEAD'],cwd=PUB,capture_output=True,text=True,check=True).stdout.strip()
  except Exception as exc:publication_error=str(exc);traceback.print_exc()
  status('computed_with_publication_errors' if publication_error else 'complete',commit=commit,publication_error=publication_error,summary=str(out/'summary/metrics.json'))
 except Exception as exc:status('error',error=str(exc));traceback.print_exc();raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();main(a.plan)
