"""Launch only after Codex00 assigns GPUs; one checkpoint worker per device."""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write
from .prepare import OUT, REPO
ORDER = ("gemma2", "gemma2_base", "llama31", "gemma3", "medgemma", "qwen_base")
def main():
    if (OUT/"standalone_dispatch_disabled.json").exists():
        raise SystemExit("Standalone dispatch disabled; jobs belong to /home/gcx/aster-gpu-queue-20261113.")
    allocation=read(OUT/"allocation.json")
    if not (allocation.get("assigned_by_codex00") or allocation.get("scheduling_confirmed")):
        raise SystemExit("Prepared only: waiting for Codex00 GPU assignment; no experiment started.")
    gpus=allocation.get("gpus",[])
    if not gpus or len(gpus)!=len(set(gpus)) or any(g not in (0,1,5,6) for g in gpus):
        raise SystemExit("Expected nonduplicated assigned GPUs within 0,1,5,6.")
    locks=[]
    for gpu in gpus:
        path=Path("/home/gcx/.aster_gpu_locks")/("gpu"+str(gpu)+".lock")
        handle=path.open("a+")
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(["nvidia-smi","-i",str(gpu),
            "--query-compute-apps=pid,used_memory","--format=csv,noheader"],text=True).strip()
        if active:
            raise SystemExit("GPU "+str(gpu)+" already has a process; waiting for scheduler.")
        locks.append(handle)
    queue=list(allocation.get("models",ORDER))
    if any(key not in ORDER for key in queue):
        raise SystemExit("Unknown checkpoint or Qwen IT rerun requested.")
    import threading
    mutex=threading.Lock()
    results={}
    def worker(gpu):
        while True:
            with mutex:
                if not queue:return
                key=queue.pop(0)
            root=OUT/"models"/key
            if read(root/"completion.json").get("complete"):
                results[key]={"state":"already_complete"};continue
            (root/"logs").mkdir(parents=True,exist_ok=True)
            with (root/"logs/run.log").open("a") as log:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
                proc=subprocess.Popen([sys.executable,"-u","-m","experiments.aster_e678.run",
                    "--model",key,"--gpu",str(gpu)],cwd=REPO,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write(root/"process.json",dict(pid=proc.pid,gpu=gpu,started=time.time()))
                code=proc.wait()
            with mutex:
                results[key]=dict(state="complete" if code==0 else "failed",returncode=code,gpu=gpu)
                write(OUT/"queue_status.json",dict(results=results,pending=queue))
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker,gpus))
    from .report import generate
    generate()
    for handle in locks:handle.close()
if __name__=="__main__":
    main()
