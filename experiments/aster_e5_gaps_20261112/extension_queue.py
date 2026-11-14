"""Single E5 owner for GPUs 0/1; waits for prior LoReFT, runs finite jobs, then releases."""
import json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/aster_e5_gaps_20261112';Q=OUT/'extension_queue'
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def alive(pid):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return False
 try:return p.read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def main():
 Q.mkdir(exist_ok=True,parents=True)
 if (OUT/'gpu_release.json').exists():raise RuntimeError('Resources already released; do not restart queue implicitly.')
 jobs=[];started=time.time()
 launch=read(OUT/'loreft_e1/launch.json');pid=launch['pid']
 write(Q/'status.json',dict(state='waiting_for_existing_loreft',existing_pid=pid,owned_gpu_ids=[0,1],other_task_gpus=[5,6],started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
 while alive(pid):time.sleep(60)
 base_ok=(OUT/'loreft_e1/completion.json').exists()
 jobs.append(dict(name='existing_loreft',success=base_ok,rerun=False))
 def run(name,args,gpus):
  cmd=[sys.executable]+args;env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),ASTER_CR_ROOT='/home/gcx/aster_cr_repair_v2_run_20261108',OMP_NUM_THREADS='4')
  before=time.time()
  with (Q/(name+'.log')).open('a') as log:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  result=dict(name=name,success=p.returncode==0,returncode=p.returncode,gpus=gpus,seconds=time.time()-before,log=str(Q/(name+'.log')))
  write(Q/(name+'.json'),result);return result
 write(Q/'status.json',dict(state='prompt_and_representation_running',owned_gpu_ids=[0,1],jobs=jobs))
 with ThreadPoolExecutor(max_workers=2) as pool:
  a=pool.submit(run,'prompt_controls',['-m','experiments.aster_e5_gaps_20261112.e1_prompt_controls'],[0])
  b=pool.submit(run,'representation_controls',['experiments/aster_e5_gaps_20261112/e1_representation_baselines.py','--mode','controller','--device','cuda:0'],[1])
  jobs.extend([a.result(),b.result()])
 for kind in ['lora','lofit']:
  write(Q/'status.json',dict(state=kind+'_native_SFT_running',owned_gpu_ids=[0,1],jobs=jobs))
  # Child controller uses physical IDs 0 and 1; no concurrent owner in this stage.
  jobs.append(run(kind+'_native_SFT',['experiments/aster_e5_gaps_20261112/native_supervised_baselines.py','--action','controller','--kind',kind,'--gpus','0,1'],[0,1]))
 write(Q/'status.json',dict(state='gpu_jobs_ended_collecting_results',owned_gpu_ids=[0,1],jobs=jobs))
 collected=run('collect_completed_results',['experiments/aster_e5_gaps_20261112/collect_e5_results.py'],[])
 jobs.append(collected)
 result=dict(state='complete' if all(x['success'] for x in jobs) else 'completed_with_failures',all_jobs_succeeded=all(x['success'] for x in jobs),jobs=jobs,seconds=time.time()-started,completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
 write(Q/'status.json',result);write(Q/'completion.json',result)
 index=OUT/'new_results_index.json'
 if index.exists():
  current=read(index);current['queue']=result;write(index,current)
 # Only this point is a GPU handoff: every GPU subprocess has exited and no work remains queued.
 write(OUT/'gpu_release.json',dict(complete=True,released_gpus=[0,1],no_gpu_jobs_inflight=True,all_jobs_succeeded=result['all_jobs_succeeded'],released_at=result['completed_at'],queue_completion=str(Q/'completion.json')))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback
  write(Q/'controller_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
  # No release on an unexpected orchestration error: child ownership must be resolved first.
  raise
