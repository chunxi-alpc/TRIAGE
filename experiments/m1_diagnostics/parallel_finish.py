"""Continue MedGemma patching and original baselines on separate GPUs."""
import os,sys,time,subprocess
from .common import ROOT,REPO,status,write,read
def main():
    jobs=[];logs=[]
    for gpu,stage in [(1,"patch"),(0,"baseline")]:
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
        f=(ROOT/"logs"/("medgemma_"+stage+"_parallel.log")).open("a");logs.append(f)
        p=subprocess.Popen([sys.executable,"-m","experiments.m1_diagnostics","--stage",stage,"--model","medgemma"],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append((gpu,stage,p))
    status("scheduler",state="running",workers=[dict(model="medgemma",gpu=g,stage=s,pid=p.pid) for g,s,p in jobs],started=time.time())
    codes={s:p.wait() for g,s,p in jobs}
    for f in logs:f.close()
    status("scheduler",state="complete" if not any(codes.values()) else "failed",exit_codes=codes)
    from .report import run
    run()
    if any(codes.values()):raise RuntimeError("MedGemma worker failed; checkpoints retained.")
    if read(ROOT/"priority_change.json").get("resume_previous_after_complete"):
        with (ROOT/"logs/previous_resume.log").open("a") as log:
            p=subprocess.Popen([sys.executable,"-m","experiments.m1_diagnostics.resume_previous"],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/"previous_resume_process.json",dict(pid=p.pid,launched_at=time.time()))
if __name__=="__main__":main()
