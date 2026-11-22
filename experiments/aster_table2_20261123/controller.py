"""Run four workers and finish Table 2 after their processes exit."""
import argparse,json,os,subprocess,sys,traceback
from .common import OUT,REPO,read,write

def run(publish=False):
    workers=[]
    write(OUT/"status.json",dict(state="running_exact_patches",gpus=[0,1,5,6]))
    for i,gpu in enumerate([0,1,5,6]):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",PYTHONUNBUFFERED="1")
        log=(OUT/("worker_"+str(i)+".log")).open("a")
        proc=subprocess.Popen([sys.executable,"-m","experiments.aster_table2_20261123.worker","--worker",str(i)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();workers.append(proc)
    write(OUT/"processes.json",dict(controller_pid=os.getpid(),workers=[dict(pid=p.pid,gpu=g) for p,g in zip(workers,[0,1,5,6])]))
    codes=[p.wait() for p in workers]
    if any(codes):raise RuntimeError("Exact-patch worker exit codes: "+str(codes))
    write(OUT/"status.json",dict(state="fitting_and_writing_table2",gpus_released=True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8")
    subprocess.run([sys.executable,"-m","experiments.aster_table2_20261123.finish"],cwd=REPO,env=env,check=True)
    if publish:
        subprocess.run(["git","add","--","experiments/aster_table2_20261123","reports/aster_table2_20261123","docs/ASTER/manuscript.zh.md","docs/ASTER/manuscript.appendix.zh.md"],cwd=REPO,check=True)
        subprocess.run(["git","commit","-m","results(ASTER): complete Table 2 with adaptive heads and hierarchical weights"],cwd=REPO,check=True)
        subprocess.run(["git","push","-u","origin","HEAD"],cwd=REPO,check=True)
        write(OUT/"publication_runtime.json",dict(state="pushed",branch="codex/aster-table2-20261123"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--publish",action="store_true");a=p.parse_args()
    try:run(a.publish)
    except Exception as e:
        write(OUT/"status.json",dict(state="error",error=str(e),traceback=traceback.format_exc()))
        raise
