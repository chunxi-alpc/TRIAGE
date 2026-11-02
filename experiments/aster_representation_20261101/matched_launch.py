"""Give each arm its own GPU and restore that GPU's previous worker on exit."""
import fcntl,os,signal,subprocess,sys,time
from .matched_data import DEST
from .data import write
from .semantic import pause_generation_workers,resume_generation_workers
def main():
 lock=(DEST/'controller.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def stop(signum,frame):raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 workers={};paused={};codes={}
 info=dict(status='starting',pid=os.getpid(),started=time.time(),tasks={0:'train_soft',1:'internal'},workers={})
 try:
  pause_generation_workers(paused)
  info['paused_prior_workers']=dict(paused)
  for gpu,arm in info['tasks'].items():
   with (DEST/(arm+'.log')).open('a') as log:
    proc=subprocess.Popen([sys.executable,'-m','experiments.aster_representation_20261101.matched',arm],
         env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT)
   workers[gpu]=proc;info['workers'][gpu]=proc.pid
  info['status']='running';write(DEST/'launch.json',info)
  while len(codes)<len(workers):
   for gpu,proc in workers.items():
    if gpu not in codes and proc.poll() is not None:
     codes[gpu]=proc.returncode
     resume_generation_workers(paused,gpu)
     info['exit_codes']=dict(codes);write(DEST/'launch.json',info)
   if len(codes)<len(workers):time.sleep(3)
 finally:
  for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,signal.SIG_IGN)
  try:
   for p in workers.values():
    if p.poll() is None:p.terminate()
   for p in workers.values():
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  finally:resume_generation_workers(paused)
  info.update(status='complete' if len(codes)==2 and not any(codes.values()) and (DEST/'paired_results.json').exists() else 'incomplete',seconds=time.time()-info['started'])
  write(DEST/'launch.json',info)
if __name__=='__main__':main()
