"""Reproduce the frozen stage-1/2 configurations into a NEW output directory."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);p.add_argument("--gpus",nargs=4,type=int,default=[0,1,2,3]);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 if not set(a.gpus)<={0,1,2,3,6,7}:raise ValueError("Use authorized GPUs 0,1,2,3,6,7")
 def run(job):
  rank,name,split,base=job
  cmd=[sys.executable,"-u",str(HERE/"run_attention.py"),"--split",split,"--configs",str(HERE/"configs"/(name+".json")),"--output",str(a.output/name)]
  if base:cmd+=["--baseline",str(a.output/base/"baseline.jsonl")]
  with (a.output/(name+".log")).open("x") as log:
   subprocess.run(cmd,cwd=ROOT,env=os.environ|{"CUDA_VISIBLE_DEVICES":str(a.gpus[rank]),"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"},stdout=log,stderr=subprocess.STDOUT,check=True)
 phases=[
  [(i,f"group1_gpu{i}","dev",None) for i in range(4)],
  [(0,"group1_confirmation_main","confirmation",None),(1,"group1_confirmation_representative","confirmation",None),
   (2,"group2_dev_gpu2","dev","group1_gpu0"),(3,"group2_dev_gpu3","dev","group1_gpu0")],
  [(2,"group2_confirmation_effect","confirmation","group1_confirmation_main"),(3,"group2_controls_dev","dev","group1_gpu0")],
  [(3,"group2_controls_confirmation","confirmation","group1_confirmation_main")]]
 for jobs in phases:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   list(pool.map(run,jobs))
 (a.output/"reproduction_complete.json").write_text(json.dumps({"status":"completed","frozen_selections":"selection_group1.json, selection_group2.json","output":str(a.output)},indent=2))
if __name__=="__main__":main()

