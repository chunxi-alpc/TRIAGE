"""One checkpoint, one GPU assigned by the shared queue; no local dispatch."""
import argparse, contextlib, fcntl, os, time, traceback
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write
from .prepare import OUT, KEYS
from .run import run_model

def assigned_device():
    value=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
    if value not in {"0","1","4","5","6"}:
        raise RuntimeError("Shared queue must assign exactly one physical GPU through CUDA_VISIBLE_DEVICES (0/1/4/5/6).")
    return int(value)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=KEYS,required=True)
    args=parser.parse_args()
    gpu=assigned_device()
    root=OUT/"models"/args.model
    root.mkdir(parents=True,exist_ok=True)
    (root/"logs").mkdir(exist_ok=True)
    handle=(root/".run.lock").open("a")
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/"process.json",dict(pid=os.getpid(),gpu=gpu,started=time.time(),
        scheduler="/home/gcx/aster-gpu-queue-20261113",standalone_dispatch=False))
    error=None
    with (root/"logs/unified_run.log").open("a",buffering=1) as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                run_model(args.model,gpu,unified_scheduler=True)
            except Exception as exc:
                error=exc
                write(root/"run_status.json",dict(state="error",error=str(exc),
                    traceback=traceback.format_exc(),pid=os.getpid(),physical_gpu=gpu))
                traceback.print_exc()
            finally:
                try:
                    with (OUT/".report.lock").open("a") as report_lock:
                        fcntl.flock(report_lock,fcntl.LOCK_EX)
                        from .report import generate
                        generate()
                except Exception:
                    traceback.print_exc()
                    if error is None:
                        raise
    handle.close()
    if error is not None:
        raise RuntimeError("Checkpoint job failed; see "+str(root/"logs/unified_run.log")) from error
if __name__=="__main__":
    main()
