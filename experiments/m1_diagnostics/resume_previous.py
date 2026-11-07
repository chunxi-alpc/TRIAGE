"""Resume the interrupted M3 evaluation only after this conversation's run completes."""
import os,sys,subprocess,time,traceback
from pathlib import Path
from .common import ROOT,write,read
PREVIOUS=Path("/home/gcx/TRIAGE/results/aster_learnable_m3_20261106")
def main():
    if (ROOT/"previous_resume_status.json").exists() and read(ROOT/"previous_resume_status.json").get("state")=="complete":return
    procs=[];logs=[]
    try:
        for gpu in [0,1]:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED="1")
            log=(ROOT/"logs"/f"previous_evaluate_{gpu}.log").open("a");logs.append(log)
            proc=subprocess.Popen([sys.executable,"-m","experiments.aster_learnable_m3.run","--stage","evaluate",
                "--shard",str(gpu),"--shards","2"],cwd="/home/gcx/TRIAGE",env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            procs.append(proc)
        write(ROOT/"previous_resume_status.json",dict(state="running",pids=[p.pid for p in procs],started=time.time()))
        codes=[p.wait() for p in procs]
        for f in logs:f.close()
        if any(codes):raise RuntimeError("Previous evaluation resume failed: "+str(codes))
        from experiments.aster_learnable_m3.report import report
        if not report():raise RuntimeError("Previous evaluation output is incomplete")
        write(PREVIOUS/"pipeline_status.json",dict(stage="complete",resumed_after="m1_diagnostics_20261106",completed_at=time.time()))
        write(ROOT/"previous_resume_status.json",dict(state="complete",completed_at=time.time()))
    except Exception as e:
        write(ROOT/"previous_resume_status.json",dict(state="failed",error=str(e),traceback=traceback.format_exc()))
        raise
if __name__=="__main__":main()
