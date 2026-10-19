#!/usr/bin/env python3
"""Run four authorized workers and wait for exits; no continuous GPU monitor."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
def save(x):(OUT/'launch.json').write_text(json.dumps(x,indent=2))
def main():
    OUT.mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks=[]
    for gpu in range(4):
        f=open(f'/tmp/care-module2-gpu{gpu}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    snap=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in snap.splitlines():
        gpu,mem,util=map(int,line.split(','))
        if gpu in range(4):assert mem<1000 and util<=10,(gpu,mem,util)
    plan=dict(status='launching',gpus=[0,1,2,3],estimated_memory_GiB_per_gpu=[10,16],estimated_minutes=[10,20],
        cases=384,pairs=192,families=12,methods=5,score_rms_grid=[0,.01,.05,.1],no_continuous_resource_monitoring=True)
    save(plan);jobs=[]
    for gpu in range(4):
        name=f'fold{gpu}' if gpu<3 else 'validation'
        command=[sys.executable,'-u',str(ROOT/'diagnostic.py'),'--output',str(OUT/name)]
        command+=['--fold',str(gpu)] if gpu<3 else ['--validate']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        log=(OUT/'logs'/f'{name}.log').open('w')
        proc=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,proc,log))
    plan.update(status='running',jobs={n:p.pid for n,p,_ in jobs});save(plan)
    time.sleep(30)
    plan['startup_alive']={n:p.poll() is None for n,p,_ in jobs};save(plan)
    codes={n:p.wait() for n,p,_ in jobs}
    for n,p,l in jobs:l.close()
    plan.update(status='completed_real_inference' if all(c==0 for c in codes.values()) else 'failed_or_partial',exit_codes=codes);save(plan)
    if any(c!=0 for c in codes.values()):sys.exit(1)
if __name__=='__main__':main()

