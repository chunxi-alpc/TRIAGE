"""Run three independent canonical-score workers and record complete coverage."""
import os,sys,json,time,subprocess,argparse
from pathlib import Path
H=Path(__file__).resolve().parent;C=Path("/home/gcx/aster_trace_repair_20261110")
GPUS=[0,1,6]
def main():
 p=argparse.ArgumentParser();p.add_argument("--split",required=True);p.add_argument("--diagnostic",action="store_true");a=p.parse_args()
 out=C/"scoring"/a.split;out.mkdir(parents=True,exist_ok=True)
 jobs=[];started=time.time()
 for shard,gpu in enumerate(GPUS):
  log=(out/("diagnostic_" if a.diagnostic else "collect_")).with_name(("diagnostic_" if a.diagnostic else "collect_")+str(shard)+".log")
  h=log.open("a")
  cmd=[sys.executable,"-u",str(H/"score_worker.py"),"--split",a.split,"--gpu",str(gpu),"--shard",str(shard)]
  if a.diagnostic:cmd.append("--diagnostic")
  env=dict(os.environ,OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")
  proc=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,env=env);h.close();jobs.append(proc)
 result=[p.wait() for p in jobs]
 if any(result):raise RuntimeError("Scoring worker failure: "+str(result))
 done=[json.loads((out/("shard"+str(i))/("diagnostic_done.json" if a.diagnostic else "done.json")).read_text()) for i in range(3)]
 if any(x["expected"]!=x["complete"] for x in done):raise RuntimeError("Incomplete score coverage")
 dest=out/("diagnostic_done.json" if a.diagnostic else "done.json")
 dest.write_text(json.dumps(dict(state="complete",gpus=GPUS,expected=sum(x["expected"] for x in done),complete=sum(x["complete"] for x in done),seconds=time.time()-started,workers=done),indent=2)+"\n")
if __name__=="__main__":main()

