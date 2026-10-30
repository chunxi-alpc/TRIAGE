"""Resume only missing official units with the original frozen worker implementation.

The supervisor waits for process exit; it does not poll GPU/resource/progress.
It never publishes commits or alters frozen parameters.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 old=Path("/home/gcx/TRIAGE-worktrees/aster-manuscript-20261030")
 run=Path("/home/gcx/aster_manuscript_20261030/clinical01")
 python="/home/gcx/aster_manuscript_20261030/venv-overlay/bin/python"
 gpus=[0,1,5,6];started=time.time()
 observed=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free","--format=csv,noheader,nounits"],text=True)
 memory={int(parts[0]):[int(parts[1]),int(parts[2])] for line in observed.strip().splitlines() if (parts:=line.split(","))}
 if any(memory[g][0]>1024 or memory[g][1]<70000 for g in gpus):raise RuntimeError("Original four GPUs are not available; no worker started")
 initial_done=len(list((run/"official_test_queue").glob("*.done.json")))
 children=[];handles=[];records=[]
 for shard,gpu in enumerate(gpus):
  cmd=[python,"-u","-m","experiments.aster_manuscript_clinical_20261030.worker",
       "--run",str(run),"--phase","official_test","--shard",str(shard),"--worker-tag","abc_resume_"+str(shard),
       "--dynamic","--official-judge","--native-query-chunks","2048","--native-balanced","--exact-execution","--resume","--decode-graphs"]
  handle=(a.output/("worker_"+str(shard)+".log")).open("x");handles.append(handle)
  child=subprocess.Popen(cmd,cwd=old,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1"),stdout=handle,stderr=subprocess.STDOUT)
  children.append(child);records.append({"pid":child.pid,"physical_gpu":gpu,"shard":shard,"command":cmd})
  (a.output/("worker_"+str(shard)+".pid")).write_text(str(child.pid)+"\n")
 status={"status":"running","gpus":gpus,"workers":records,"complete_cases_at_resume":initial_done,
    "planned_cases":7458,"planned_methods":13,"old_run":str(run),"worker_cwd":str(old),
    "existing_success_failure_units_preserved":True,"frozen_parameters_changed":False,"polls_for_progress":False,"automatic_publication":False}
 def write(): (a.output/"status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
 write()
 print(json.dumps({"status":"running","workers":[r["pid"] for r in records],"complete_cases_at_resume":initial_done}),flush=True)
 codes=[p.wait() for p in children]
 for f in handles:f.close()
 status.update(worker_exit_codes=codes,elapsed_seconds=time.time()-started)
 if any(codes):status["status"]="interrupted_missing_units_may_be_resumed"
 else:
  with (a.output/"final_report.log").open("x") as f:
   rc=subprocess.run([python,"-m","experiments.aster_manuscript_clinical_20261030.report","--run",str(run),"--phase","official_test"],cwd=old,stdout=f,stderr=subprocess.STDOUT).returncode
  status.update(report_exit_code=rc,status="complete" if rc==0 else "workers_finished_report_incomplete")
 write()
if __name__=="__main__":main()
