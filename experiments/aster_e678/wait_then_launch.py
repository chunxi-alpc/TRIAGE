"""Start per-device E678 workers only after the agreed predecessor release."""
import fcntl, json, os, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write
from .prepare import OUT, REPO
from .launch import ORDER

def permits_gpu(event, gpu):
    return (event.get("complete") is True and event.get("no_gpu_jobs_inflight") is True
            and gpu in event.get("released_gpus", []))

def main():
    if (OUT/"standalone_dispatch_disabled.json").exists():
        raise SystemExit("Standalone dispatch disabled; jobs belong to /home/gcx/aster-gpu-queue-20261113.")
    guard=(OUT/".handoff.lock").open("a")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(OUT/"handoff_plan.json")
    devices=plan["gpus"]
    if set(devices)-{0,1,5,6} or len(devices)!=len(set(devices)):
        raise ValueError("Unexpected GPU request")
    queue=[key for key in plan.get("models",ORDER)
           if not read(OUT/"models"/key/"completion.json").get("complete")]
    mutex=threading.RLock()
    gpu_states={str(g):{"state":"waiting_predecessor_release"} for g in devices}
    results={}
    def save():
        write(OUT/"handoff_status.json",dict(pid=os.getpid(),
            stage="queued_or_running",updated=time.strftime("%Y-%m-%d %H:%M:%S"),
            requested_gpus=devices,pending=list(queue),gpus=dict(gpu_states),results=dict(results),
            continuous_model_monitoring=False))
    save()
    def device_worker(gpu):
        handle=None
        try:
            while True:
                with mutex:
                    if not queue:
                        gpu_states[str(gpu)]={"state":"no_pending_models"};save();return
                current=read(OUT/"handoff_plan.json")
                group=next((r for r in current["groups"] if gpu in r["gpus"]),None)
                release_path=group.get("release_file") if group else None
                event=read(release_path) if release_path else {}
                if not permits_gpu(event,gpu):
                    time.sleep(60);continue
                lock_path=Path("/home/gcx/.aster_gpu_locks")/("gpu"+str(gpu)+".lock")
                candidate=lock_path.open("a")
                try:
                    fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close();time.sleep(60);continue
                active=subprocess.check_output(["nvidia-smi","-i",str(gpu),
                    "--query-compute-apps=pid,used_memory","--format=csv,noheader"],text=True).strip()
                if active:
                    candidate.close()
                    with mutex:
                        gpu_states[str(gpu)]={"state":"released_but_device_busy"};save()
                    time.sleep(60);continue
                handle=candidate
                with mutex:
                    allocation=read(OUT/"allocation.json")
                    accepted=sorted(set(allocation.get("gpus",[]))|{gpu})
                    records=allocation.get("release_records",{})
                    records[str(gpu)]=dict(owner=group["owner"],release_file=release_path,event=event)
                    write(OUT/"allocation.json",dict(scheduling_confirmed=True,assigned_by_codex00=False,
                        scheduling_basis="Codex00 parallel rules and explicit per-owner resource handoff",
                        requested_gpus=devices,gpus=accepted,release_records=records,
                        models=list(ORDER),preempt_existing_jobs=False))
                    gpu_states[str(gpu)]={"state":"released_to_codex05","predecessor":group["owner"]};save()
                break
            while True:
                with mutex:
                    if not queue:
                        gpu_states[str(gpu)]={"state":"finished"};save();return
                    key=queue.pop(0)
                root=OUT/"models"/key
                (root/"logs").mkdir(parents=True,exist_ok=True)
                with (root/"logs/run.log").open("a") as log:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",
                        OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
                    proc=subprocess.Popen([sys.executable,"-u","-m","experiments.aster_e678.run",
                        "--model",key,"--gpu",str(gpu)],cwd=REPO,env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    write(root/"process.json",dict(pid=proc.pid,gpu=gpu,started=time.time()))
                    with mutex:
                        gpu_states[str(gpu)]={"state":"running","model":key,"pid":proc.pid};save()
                    code=proc.wait()
                with mutex:
                    results[key]=dict(state="complete" if code==0 else "failed",returncode=code,gpu=gpu)
                    save()
                    from .report import generate
                    generate()
        except Exception as exc:
            with mutex:
                gpu_states[str(gpu)]={"state":"controller_error","error":str(exc),"traceback":traceback.format_exc()};save()
        finally:
            if handle is not None:
                handle.close()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(device_worker,devices))
    from .report import generate
    write(OUT/"handoff_completion.json",dict(queue_finished=True,results=results,gpus=gpu_states,
        all_jobs_succeeded=all(read(OUT/"models"/key/"completion.json").get("complete") for key in plan.get("models",ORDER)),
        finished=time.strftime("%Y-%m-%d %H:%M:%S")))
    generate()
if __name__=="__main__":
    main()
