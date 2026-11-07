import argparse,fcntl,traceback,time,gc
import torch
from .common import *
def claim(phase):
    path=ROOT/"queues"/f"{phase}.json";lock=path.with_suffix(".lock")
    with lock.open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX);q=read(path)
        if q["next"]>=len(q["jobs"]):return None
        job=q["jobs"][q["next"]];q["next"]+=1;write(path,q);return job
def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",required=True);p.add_argument("--gpu",type=int,required=True);args=p.parse_args()
    from .engine import Engine
    from .evaluation import p0,calibration,evaluate
    from .training import train
    torch.set_num_threads(4);engine=Engine();status(f"worker_{args.gpu}",state="ready",phase=args.phase)
    while (job:=claim(args.phase)) is not None:
        name=job["id"];done=ROOT/"jobs"/f"{name}.json"
        if done.exists() and read(done).get("state")=="complete":continue
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();write(done,dict(state="running",gpu=args.gpu,job=job))
        status(f"worker_{args.gpu}",state="running",phase=args.phase,job=name)
        try:
            if job["kind"]=="p0":p0(engine,job["shard"])
            elif job["kind"]=="calibration":calibration(engine)
            elif job["kind"]=="train":
                train(engine,job["name"]);evaluate(engine,job["name"],"medguide_pilot")
            else:evaluate(engine,job["name"],job["dataset"])
            write(done,dict(state="complete",gpu=args.gpu,job=job,elapsed_seconds=time.perf_counter()-start))
        except Exception as exc:
            write(done,dict(state="failed",gpu=args.gpu,job=job,error=str(exc),traceback=traceback.format_exc()))
            raise
        gc.collect();torch.cuda.empty_cache()
    status(f"worker_{args.gpu}",state="complete",phase=args.phase)
if __name__=="__main__":main()
