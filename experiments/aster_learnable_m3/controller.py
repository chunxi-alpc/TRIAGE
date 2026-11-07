"""Detached server controller. Records progress without active chat monitoring."""
import argparse,fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
from .data import save_json
from .run import ROOT

def launch(stage,gpu,extra):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
    log=(ROOT/'logs'/f'{stage}_{gpu}.log').open('a')
    process=subprocess.Popen([sys.executable,'-m','experiments.aster_learnable_m3.run','--stage',stage]+extra,
        cwd='/home/gcx/TRIAGE',env=env,stdout=log,stderr=subprocess.STDOUT)
    return process,log

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--worklist');parser.add_argument('--dynamic-queue');args=parser.parse_args()
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    start=time.time()
    try:
        if not args.resume:
            workers=[launch('train',0,['--mode','reference']),launch('train',1,['--mode','self'])]
            save_json(ROOT/'pipeline_status.json',dict(stage='training',pids=[p.pid for p,l in workers],started=start))
            codes=[p.wait() for p,l in workers]
            for p,l in workers:l.close()
            if any(codes):raise RuntimeError('Training worker failure: '+str(codes))
        workers=[launch('evaluate',gpu,['--shard',str(gpu),'--shards','2']+(['--worklist',args.worklist] if args.worklist else [])+(['--dynamic-queue',args.dynamic_queue] if args.dynamic_queue else [])) for gpu in (0,1)]
        save_json(ROOT/'pipeline_status.json',dict(stage='test',pids=[p.pid for p,l in workers],started=start))
        codes=[p.wait() for p,l in workers]
        for p,l in workers:l.close()
        if any(codes):raise RuntimeError('Evaluation worker failure: '+str(codes))
        from .report import report
        if not report():raise RuntimeError('Incomplete evaluation output')
        save_json(ROOT/'pipeline_status.json',dict(stage='complete',elapsed_seconds=time.time()-start))
    except Exception as e:
        save_json(ROOT/'pipeline_status.json',dict(stage='failed',error=str(e),traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()
