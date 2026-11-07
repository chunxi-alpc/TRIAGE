import argparse,os,time,subprocess,sys,traceback
from pathlib import Path
from .common import *

def worker(model,stage):
    from .engine import Engine
    from . import patching,baseline
    engine=Engine(model)
    if stage in ["worker","patch"]:patching.run(model,engine)
    if stage in ["worker","baseline"]:baseline.run(model,engine)

def scheduler():
    from .report import run as report
    from .data import prepare
    from .cpu import run as cpu
    prepare();cpu();report()
    status("scheduler",state="queued",gpus=[0,1])
    # Existing controller must finish before launching; a gap between its workers is not an idle reservation.
    wait_pids=read(ROOT/"launch.json").get("wait_pids",[]) if (ROOT/"launch.json").exists() else []
    while any(Path("/proc",str(pid)).exists() for pid in wait_pids):
        time.sleep(60)
    while True:
        raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
        mem={int(x.split(",")[0]):int(x.split(",")[1]) for x in raw.strip().splitlines()}
        if all(mem[i]<2000 for i in [0,1]):break
        time.sleep(60)
    procs=[];logs=[]
    for gpu,model in [(0,"qwen"),(1,"medgemma")]:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
        log=(ROOT/"logs"/(model+".log")).open("a");logs.append(log)
        p=subprocess.Popen([sys.executable,"-m","experiments.m1_diagnostics","--stage","worker","--model",model],
            cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append((model,gpu,p))
    status("scheduler",state="running",workers=[dict(model=m,gpu=g,pid=p.pid) for m,g,p in procs],started=time.time())
    # Children checkpoint autonomously. This controller only waits for their exit and writes the report.
    codes={m:p.wait() for m,g,p in procs}
    for f in logs:f.close()
    status("scheduler",state="complete" if all(v==0 for v in codes.values()) else "failed",exit_codes=codes)
    report()
    if any(codes.values()):raise RuntimeError("A worker failed; checkpoint retained; see logs and status.")
    priority=ROOT/"priority_change.json"
    if priority.exists() and read(priority).get("resume_previous_after_complete"):
        with (ROOT/"logs/previous_resume.log").open("a") as log:
            prior=subprocess.Popen([sys.executable,"-m","experiments.m1_diagnostics.resume_previous"],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/"previous_resume_process.json",dict(pid=prior.pid,launched_at=time.time()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","cpu","patch","baseline","report","all","worker"],default="all")
    parser.add_argument("--model",choices=list(MODELS))
    args=parser.parse_args()
    try:
        if args.stage=="prepare":
            from .data import prepare
            prepare()
        elif args.stage=="cpu":
            from .data import prepare
            from .cpu import run
            prepare();run()
        elif args.stage=="report":
            from .report import run
            run()
        elif args.stage=="all":scheduler()
        else:
            if not args.model:parser.error("--model required for this stage")
            worker(args.model,args.stage)
    except Exception as exc:
        status("error_"+(args.model or args.stage),state="failed",error=str(exc),traceback=traceback.format_exc())
        raise
if __name__=="__main__":main()
