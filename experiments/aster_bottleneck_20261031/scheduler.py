"""Event-driven GPU scheduling across independent experiment stages.

One subprocess per authorized GPU. Completed work is reused, running children
are adopted, and independent ready jobs fill every released slot. Global
selection steps still wait for the results they actually consume.
"""
import argparse,concurrent.futures,heapq,itertools,json,os,queue,select,subprocess,sys,threading,time,traceback
from pathlib import Path
from .run import OUT,GPUS,read,write
from .module_one_data import M1,prepare
from . import advanced as adv
from . import module_one as m1

class GPUPool:
 def __init__(self,adopt=(),gpus=GPUS):
  self.gpus=list(gpus);self.pending=queue.PriorityQueue();self.sequence=itertools.count()
  self.lock=threading.RLock();self.jobs={};self.futures={};self.active={};self.workers=[]
  self.adopt={};self.phase='running'
  for item in adopt:
   key=self.key(item['family'],item['stage'],item['shard'])
   if item['gpu'] in self.adopt:raise ValueError('Multiple existing jobs on one GPU')
   self.adopt[item['gpu']]=item
   self.futures[key]=concurrent.futures.Future()
   self.jobs[key]={**item,'key':key,'state':'adopted','single_shard':False}
 def key(self,family,stage,shard):return f'{family}/{stage}/{shard}'
 def folder(self,family):return M1 if family=='module_one' else OUT
 def status(self,job):return self.folder(job['family'])/f'{job["stage"]}_{job["shard"]}_status.json'
 def snapshot(self,event=None):
  with self.lock:
   write(OUT/'scheduler_status.json',{'status':self.phase,'gpus':self.gpus,'active':self.active,
    'queued':sum(j['state']=='queued' for j in self.jobs.values()),
    'complete':sum(j['state']=='complete' for j in self.jobs.values()),
    'failed':{k:j.get('error') for k,j in self.jobs.items() if j['state']=='failed'},
    'jobs':{k:{kk:vv for kk,vv in j.items() if kk!='key'} for k,j in self.jobs.items()}})
   if event:
    with (OUT/'scheduler_events.jsonl').open('a') as f:f.write(json.dumps({'unix_time':time.time(),**event})+'\n')
 def submit(self,family,stage,shard,priority=20,single_shard=False):
  key=self.key(family,stage,shard)
  with self.lock:
   if key in self.futures:return self.futures[key]
   future=concurrent.futures.Future();self.futures[key]=future
   job={'key':key,'family':family,'stage':stage,'shard':shard,'state':'queued','single_shard':single_shard}
   self.jobs[key]=job;p=self.status(job)
   if p.exists() and json.loads(p.read_text()).get('status')=='complete':
    job['state']='complete';job['reused']=True;future.set_result({'reused':True})
   else:self.pending.put((priority,next(self.sequence),job))
   self.snapshot()
   return future
 def finish(self,job,gpu,error=None):
  with self.lock:
   key=job['key'];future=self.futures[key]
   job['state']='failed' if error else 'complete'
   if error:job['error']=str(error)
   self.active.pop(gpu,None)
   self.snapshot({'event':job['state'],'job':key,'gpu':gpu,'error':str(error) if error else None})
   if error:future.set_exception(error)
   else:future.set_result({'gpu':gpu})
 def existing(self,item):
  gpu=item['gpu'];job=self.jobs[self.key(item['family'],item['stage'],item['shard'])]
  with self.lock:
   self.active[gpu]={'job':job['key'],'pid':item['pid']};self.snapshot({'event':'adopt','job':job['key'],'gpu':gpu})
  try:
   try:
    fd=os.pidfd_open(item['pid']);select.select([fd],[],[]);os.close(fd)
   except ProcessLookupError:pass
   p=self.status(job)
   if not p.exists() or json.loads(p.read_text()).get('status')!='complete':raise RuntimeError('Adopted worker did not complete: '+job['key'])
   self.finish(job,gpu)
  except Exception as e:self.finish(job,gpu,e)
 def gpu_loop(self,gpu):
  if gpu in self.adopt:self.existing(self.adopt[gpu])
  while True:
   _,_,job=self.pending.get()
   if job is None:self.pending.task_done();return
   try:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    if job['single_shard']:env['ASTER_WORKER_SHARDS']='1'
    else:env.pop('ASTER_WORKER_SHARDS',None)
    logfile=self.folder(job['family'])/f'{job["stage"]}_{job["shard"]}.log'
    module=__package__+'.'+job['family']
    with logfile.open('a') as f:
     p=subprocess.Popen([sys.executable,'-m',module,'--worker',job['stage'],'--shard',str(job['shard'])],
      env=env,stdout=f,stderr=subprocess.STDOUT)
    with self.lock:
     job.update(state='running',gpu=gpu,pid=p.pid);self.active[gpu]={'job':job['key'],'pid':p.pid}
     self.snapshot({'event':'start','job':job['key'],'gpu':gpu,'pid':p.pid})
    code=p.wait()
    if code:raise RuntimeError(f'{job["key"]} exited {code}; see {logfile}')
    status=self.status(job)
    if not status.exists() or json.loads(status.read_text()).get('status')!='complete':raise RuntimeError('Missing completion status: '+job['key'])
    self.finish(job,gpu)
   except Exception as e:self.finish(job,gpu,e)
   finally:self.pending.task_done()
 def start(self):
  for gpu in self.gpus:
   t=threading.Thread(target=self.gpu_loop,args=(gpu,),name='gpu-'+str(gpu));t.start();self.workers.append(t)
 def close(self):
  for _ in self.workers:self.pending.put((10**9,next(self.sequence),None))
  for t in self.workers:t.join()
  self.phase='failed' if any(j['state']=='failed' for j in self.jobs.values()) else 'complete';self.snapshot()

def priorities(family,stage):
 if family=='module_one':return {'baseline':0,'calibration':1,'screen':5,'finalists':4,'diagnostic':4,'oracle_validation':15,'predicted_validation':15}.get(stage,20)
 return {'windows':10,'headscan':5,'subsets':5,'key_calibration':12,'key_controls':12,'routing':25,'predicted_dev':45,'predicted_validation':45}.get(stage,90 if stage.startswith('errors') else 30)

def attach_launchers(pool):
 def launch_advanced(stage):
  jobs=[]
  if stage.startswith('errors'):
   cfg=json.loads((OUT/f'{stage}_config.json').read_text())
   # Two cases per job keep long perturbation grids from holding a GPU for hours.
   for start in range(0,len(cfg['cases']),2):
    name=stage+'_part'+str(start//2).zfill(3)
    with pool.lock:adv.register(name,cfg['cases'][start:start+2],cfg['specs'])
    jobs.append(pool.submit('advanced',name,0,priorities('advanced',stage),single_shard=True))
  else:
   jobs=[pool.submit('advanced',stage,i,priorities('advanced',stage)) for i in range(4)]
  for f in jobs:f.result()
  with pool.lock:return adv.summary(stage) if stage not in ['routing','proof','behavior'] else None
 def launch_module(stage):
  write(M1/'status.json',{'status':'running','stage':stage,'physical_gpus':GPUS,'scheduler':'dynamic'})
  fs=[pool.submit('module_one',stage,i,priorities('module_one',stage)) for i in range(4)]
  for f in fs:f.result()
  with pool.lock:return m1.summaries(stage) if stage!='diagnostic' else None
 adv.launch=launch_advanced;m1.launch=launch_module

def prepare_independent_configs():
 prepare()
 cases=read(M1/'development100.jsonl');ids=[r['case_id'] for r in cases]
 m1.register('baseline',ids,[{'name':'zero','kind':'zero'}])
 unit={'name':'unit','kind':'mean','gain':1.,'direction_heads':[[l,h] for l in range(1,36) for h in range(8)],
  'unit_directions':True,'full_rule_keys':True,'calibrate_only':True,'probe_only':True,'deeper':True}
 m1.register('calibration',ids,[unit])
 beta=json.loads((OUT/'selected_soft.json').read_text())['beta'];curve=[]
 for eps in [.1,.2,.3,.5]:
  for error in ['wrong','stop']:
   for seed in [13,37,73]:
    for op in [{'kind':'clip'},{'kind':'hard'},{'kind':'bias','beta':beta}]:
     curve.append({**op,'name':f'eps{eps}_{error}_{seed}_{op["kind"]}','epsilon':eps,'error_type':error,'seed':seed})
 adv.register('errors_dev',[c['case_id'] for c in read(OUT/'development43.jsonl')],curve)
 adv.register('errors_validation',[c['case_id'] for c in read(OUT/'validation120.jsonl')],
  [s for s in curve if s['epsilon'] in [.2,.5] and s['seed']==13])

def run():
 if json.loads((OUT/'execution_status.json').read_text())['status']!='complete':raise RuntimeError('Initial operator run is not complete')
 prepare_independent_configs()
 adopt_path=OUT/'scheduler_adopt.json';adopt=json.loads(adopt_path.read_text()) if adopt_path.exists() else []
 # Ignore exited historical pids; completed files supply their results.
 live=[]
 for x in adopt:
  try:
   args=Path(f'/proc/{x["pid"]}/cmdline').read_bytes().decode()
   if f'--worker\0{x["stage"]}\0' in args and f'--shard\0{x["shard"]}\0' in args:live.append(x)
  except FileNotFoundError:pass
 pool=GPUPool(live);attach_launchers(pool);errors=[];controllers=[]
 def guarded(name,fn):
  try:fn()
  except Exception as e:
   errors.append({'controller':name,'error':str(e)});traceback.print_exc()
   write(OUT/('scheduler_'+name+'_failure.json'),errors[-1])
 def early_lane(shard):
  pool.submit('module_one','baseline',shard,0).result()
  # This shard only consumes its own completed baseline cases.
  pool.submit('module_one','calibration',shard,1).result()
 def early_data():
  with concurrent.futures.ThreadPoolExecutor(4) as executor:
   fs=[executor.submit(early_lane,i) for i in range(4)]
   for f in fs:f.result()
  write(M1/'early_status.json',{'status':'complete','stages':['baseline','calibration'],'scheduler':'dynamic','gpus':GPUS})
 jobs=[('early_data',early_data),('routing',lambda:adv.launch('routing')),
  ('errors_dev',lambda:adv.launch('errors_dev')),('errors_validation',lambda:adv.launch('errors_validation')),
  ('main',adv.run_pipeline)]
 for name,fn in jobs:
  t=threading.Thread(target=guarded,args=(name,fn),name='controller-'+name);t.start();controllers.append(t)
 pool.start()
 for t in controllers:t.join()
 pool.close()
 if errors:
  write(OUT/'advanced_status.json',{'status':'failed','errors':errors})
  pool.phase='failed';pool.snapshot()
  raise RuntimeError('One or more experiment controllers failed')
if __name__=='__main__':run()
