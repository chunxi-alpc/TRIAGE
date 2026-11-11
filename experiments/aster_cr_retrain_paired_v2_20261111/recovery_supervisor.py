"""Continue evaluation after transient controller failures, retaining completed work."""
import json,os,subprocess,sys,time
from .common import OUT,REPO,read,write,append
PACKAGE="experiments.aster_cr_retrain_paired_v2_20261111"

def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
    command=[sys.executable,"-u","-m",PACKAGE+".optimized_pipeline","--mode","resume-evaluation"]
    for attempt in range(1,4):
        if read(OUT/"completion.json").get("complete"):
            write(OUT/"supervisor.json",dict(state="complete",pid=os.getpid(),attempts=attempt-1))
            return
        log=(OUT/"logs/evaluation_recovery_controller.log").open("a")
        controller=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(OUT/"supervisor.json",dict(state="running",pid=os.getpid(),controller_pid=controller.pid,attempt=attempt,policy="adopt existing GPU workers; reuse completed models, caches and evaluation scopes"))
        ret=controller.wait()
        log.close()
        if read(OUT/"completion.json").get("complete"):
            write(OUT/"supervisor.json",dict(state="complete",pid=os.getpid(),controller_pid=controller.pid,attempts=attempt))
            return
        error=read(OUT/"optimized_error.json")
        append(OUT/"recovery_history.jsonl",dict(supervisor_attempt=attempt,returncode=ret,time=time.strftime("%Y-%m-%d %H:%M:%S"),error=error))
        if attempt<3:
            time.sleep(5)
    write(OUT/"supervisor.json",dict(state="error",pid=os.getpid(),attempts=3,error=read(OUT/"optimized_error.json")))
    raise RuntimeError("Evaluation controller did not complete after recovery attempts")

if __name__=="__main__":
    main()
