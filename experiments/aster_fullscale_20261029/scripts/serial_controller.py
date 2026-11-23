"""Server-resident sequential scheduler with process monitoring and checkpoint recovery."""
import argparse,json,os,time,subprocess,sys,fcntl
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);repo=Path(__file__).resolve().parents[3];scripts=Path(__file__).resolve().parent
lock=(root/'serial_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
meta=root/'scheduler';meta.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def write(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def event(kind,**kw):
 with (root/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps({'time':now(),'event':kind,**kw},ensure_ascii=False)+'\n')
def state(phase,status,**kw):write(root/'serial_status.json',{'time':now(),'controller_pid':os.getpid(),'phase':phase,'status':status,**kw})
def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def alive(pid):
 try:
  status=Path('/proc')/str(pid)/'stat'
  return status.exists() and status.read_text().split()[2]!='Z'
 except Exception:return False
children={};handles=[];services={}
def spawn(name,cmd,gpu,status,attempt=0):
 if gpu not in read(root/'resource_policy.json').get('allowed_gpus',[0,1,5,6]):raise RuntimeError('GPU not authorized by current resource policy: '+str(gpu))
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
 log=(meta/(name+'_attempt_'+str(attempt)+'.log')).open('a');handles.append(log)
 proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children[name]=proc
 info={'pid':proc.pid,'cmd':cmd,'gpu':gpu,'status_file':str(status),'attempt':attempt,'name':name};write(meta/(name+'.json'),info);event('worker_started',name=name,pid=proc.pid,gpu=gpu,attempt=attempt);return info

def ensure_background():
 # Lock ownership is the identity check; never create a second active publisher/queue.
 for script,relative,arguments in [
  ('context_replay.py','context_replay/controller.lock',['controller','--run',str(root)]),
  ('status_publisher.py','status_publisher.lock',['--run',str(root)])]:
  if script in services and services[script].poll() is None:continue
  path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('a') as test:
   try:fcntl.flock(test,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
  log=(root/(script.replace('.py','')+'_supervised.log')).open('a');handles.append(log)
  proc=subprocess.Popen([sys.executable,str(scripts/script),*arguments],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  services[script]=proc;event('background_service_started',script=script,pid=proc.pid)

def wait_jobs(phase,jobs):
 # jobs contain existing worker identity or new commands; no duplicate worker is spawned.
 pending={job['name']:job for job in jobs}
 while pending:
  ensure_background()
  for name,job in list(pending.items()):
   info=read(meta/(name+'.json'));status=Path(info.get('status_file',job['status_file']));s=read(status)
   if s.get('status')=='complete':pending.pop(name);continue
   proc=children.get(name)
   running=(proc.poll() is None) if proc else alive(info.get('pid',0))
   if running:continue
   attempt=info.get('attempt',-1)+1
   if attempt>2:raise RuntimeError(name+' failed after recovery attempts; '+str(status))
   cmd=list(job['cmd'])
   if phase=='select':
    cmd+=['--attempt',str(attempt)];status=root/'serial/select'/('attempt_'+str(attempt))/'status.json'
   if info:event('worker_recovery',name=name,previous_pid=info.get('pid'),reason=s.get('reason','process_exited'))
   spawn(name,cmd,job['gpu'],status,attempt)
  state(phase,'running',active_workers=list(pending),active_pids={n:read(meta/(n+'.json')).get('pid') for n in pending})
  if pending:time.sleep(30)
 state(phase,'base_predictions_complete' if phase in ('confirm','official_test','med') else 'complete');event('base_stage_complete' if phase in ('confirm','official_test','med') else 'stage_complete',phase=phase)

def wait_priority_context(phase):
 policy=read(root/'resource_policy.json')
 if policy.get('priority_phase')!=phase or policy.get('priority_context_complete'):return
 from context_replay import completed
 cohort=set(policy.get('priority_context_cases',[]))
 while True:
  ensure_background();done=completed(root,phase);pending=cohort-done
  active=[]
  for path in (root/'context_replay/scheduler').glob(phase+'_*/job.json'):
   info=read(path)
   if alive(info.get('pid',0)):active.append(info['pid'])
  if not pending and not active:break
  status=read(root/'context_replay/status.json')
  if status.get('status')=='needs_attention':raise RuntimeError('priority context replay needs engineering recovery')
  state(phase,'priority_context_replay',remaining_context_cases=len(pending),allowed_gpus=policy.get('allowed_gpus'));time.sleep(30)
 policy['priority_context_complete']=True;write(root/'resource_policy.json',policy);event('priority_context_cohort_complete',phase=phase,cases=len(cohort))

def clinical_workers(phase):
 config=root/'scheduler'/(phase+'_allocation.json')
 allocation=read(config)
 allowed=set(read(root/'resource_policy.json').get('allowed_gpus',[0,1,5,6]))
 if allocation and not set(allocation['cards'])<=allowed:raise RuntimeError('saved allocation conflicts with allowed GPUs; explicit reshard required')
 if not allocation:
  cards=[]
  output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  for line in output.splitlines():
   gpu,memory,utilization=[int(x.strip()) for x in line.split(',')]
   if gpu in allowed and memory<1024 and utilization<5:cards.append(gpu)
  if not cards:raise RuntimeError('no free GPU available for '+phase)
  allocation={'cards':cards,'world':len(cards),'created':now()};write(config,allocation);event('clinical_allocation',phase=phase,**allocation)
 return [(gpu,rank) for rank,gpu in enumerate(allocation['cards'])]

def stage_jobs(phase,workers):
 jobs=[]
 for gpu,worker in workers:
  name=phase+'_'+str(worker);status=root/'serial'/phase/(('attempt_0') if phase=='select' else ('worker_'+str(worker)))/'status.json'
  cmd=[sys.executable,str(scripts/'serial_stage.py'),'--run',str(root),'--phase',phase,'--worker',str(worker),'--world',str(len(workers))]
  if phase!='select':cmd+=['--batch-mode','--generation-batch-size','4','--judge-batch-size','8']
  jobs.append({'name':name,'gpu':gpu,'cmd':cmd,'status_file':str(status)})
 return jobs
try:
 event('controller_started',pid=os.getpid(),ordered_phases=['fit','merge','select','confirm','official_test','med','report'])
 jobs=[]
 for rank,gpu in enumerate([0,1,4,5,6]):
  name='fit_'+str(rank);status=root/'parallel_fit'/('shard_'+str(rank))/'status.json'
  cmd=[sys.executable,str(scripts/'parallel_directions.py'),'worker','--run',str(root),'--rank',str(rank),'--world','5']
  jobs.append({'name':name,'gpu':gpu,'cmd':cmd,'status_file':str(status)})
 wait_jobs('fit',jobs)
 if not (root/'parallel_fit/directions.pt').exists():
  state('merge','running');subprocess.run([sys.executable,str(scripts/'parallel_directions.py'),'merge','--run',str(root),'--world','5'],cwd=repo,check=True);event('stage_complete',phase='merge')
 if not (root/'serial/frozen.pt').exists():wait_jobs('select',stage_jobs('select',[(0,0)]))
 for phase in ['confirm','official_test','med']:
  wait_priority_context(phase)
  workers=clinical_workers(phase)
  wait_jobs(phase,stage_jobs(phase,workers))
  from context_replay import barrier
  barrier(root,phase)
  state(phase,'complete');event('stage_complete',phase=phase,context_replays_included=True)
  from fullscale_run import report
  report(root/'analysis/serial');event('report_updated',phase=phase)
 state('report','running')
 from fullscale_run import report
 report(root/'analysis/serial');state('complete','complete');event('pipeline_complete')
except Exception as ex:
 state('failed','needs_attention',reason=str(ex));event('pipeline_failed',reason=str(ex));raise
