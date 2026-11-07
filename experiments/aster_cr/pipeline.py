"""Complete phase ordering. Development gates precede every test forward."""
import argparse,fcntl,time,os,pickle,sys
import numpy as np
from .common import *
from .controller import phase
def choose_template(model):
 from .statistics import pair_metrics
 root=ROOT/model/"comparators";cfg=read(root/"settings.json")
 if not cfg:return None
 rr=[r for p in root.glob("dev_*.jsonl") for r in rows(p)]
 pr=[r for p in root.glob("paths_dev_*.jsonl") for r in rows(p)]
 pairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="dev"]
 full=np.mean([r["methods"]["full"]["correct"] for r in pr]);candidates=[]
 for form in ("full","order_no_yes","map_ab","map_ba","equivalent"):
  name="template_"+form;ss={r["view_id"]:r["scores"][name] for r in rr if name in r["scores"]}
  metric=pair_metrics([],pairs,ss);path=float(np.mean([r["methods"][name]["correct"] for r in pr]))
  candidates.append(dict(form=form,macro=metric["macro"],path_accuracy=path,eligible=path>=full-.005))
 allowed=[r for r in candidates if r["eligible"]]
 best=max(allowed,key=lambda r:(r["macro"] if r["macro"] is not None else -1,r["form"]=="full"))["form"]
 cfg.update(best_template=best,template_dev_candidates=candidates,template_selection="path gate then PA macro; no test outputs")
 write(root/"settings.json",cfg)
 return best
def choose(model,final=False):
 root=ROOT/model
 candidates=read(root/"readout_candidates.json")
 paths=read(root/"path_results/dev.json");full=paths["full"]["accuracy"]
 eligible=[r for r in candidates if paths[r["name"]]["accuracy"]>=full-.005]
 def key(r):
  macro=r["dev"]["macro"]
  return (macro if macro is not None else -1,-r["parameters"])
 def best(rs):return max(rs,key=key)["name"] if rs else "full"
 cr=[r for r in eligible if r["name"].startswith("cr_")]
 simple=[r for r in eligible if r["name"] in ("full","bias","temperature","logistic")]
 ordinary=[r for r in eligible if r["name"].startswith("ordinary")]
 chosen=dict(aster_cr=best(cr),calibration=best(simple),ordinary=best(ordinary),
   path_full_accuracy=full,dev_path_exclusion_margin=.005,
   excluded_path=[r["name"] for r in candidates if paths[r["name"]]["accuracy"]<full-.005],
   selection="path accuracy drop <=0.5pp, then four-class PA macro, then fewer trainable parameters")
 selected=next(r for r in candidates if r["name"]==chosen["aster_cr"])
 chosen["ordinary_matched"]=("ordinary_r"+str(selected["rank"])) if model=="qwen" and selected.get("rank") is not None else "ordinary_confirm" if model=="medgemma" else chosen["ordinary"]
 if any(r["name"]=="ordinary_matched" for r in candidates):chosen["ordinary_matched"]="ordinary_matched"
 write(root/"development_selection.json",chosen)
 if final:
  template=choose_template(model)
  baselines=[]
  for p in (root/"baselines").glob("*_dev_metrics.json"):
   r=read(p)
   r["eligible"]=r["paths"]["accuracy"]>=full-.005
   baselines.append(r)
  families={}
  for r in baselines:families.setdefault(r["name"].split("_lr")[0],[]).append(r)
  def bkey(r):return (r["pairs"]["macro"] if r["pairs"]["macro"] is not None else -1,-r["parameters"])
  winners={}
  for kind,rs in families.items():
   allowed=[r for r in rs if r["eligible"]]
   # A family can still be reported when it fails the path gate, but cannot be the promoted baseline.
   winners[kind]=max(allowed or rs,key=bkey)["name"]
  allowed=[r for r in baselines if r["eligible"]]
  strongest=max(allowed or baselines,key=bkey)["name"] if baselines else None
  core=["full","bias","temperature","logistic",chosen["aster_cr"],chosen["ordinary"],chosen["ordinary_matched"]]
  if model=="qwen":core+=["nuisance_only","signal_only","complete_exploratory","ce_only","no_AP","no_CF","random_heads","random_support"]
  available={r["name"] for r in candidates};core=sorted(set(core)&available)
  write(root/"frozen.json",dict(frozen=True,time=time.strftime("%Y-%m-%d %H:%M:%S"),development=chosen,
    readouts=core,baseline_families=winners,strongest_baseline=strongest,best_template=template,
    baseline_path_gate={r["name"]:r["eligible"] for r in baselines},test_tuning=False))
 return chosen
def cpu(module,*args):
 import subprocess,re
 key=re.sub("[^a-zA-Z0-9_.-]","_",module+"_"+"_".join(args))
 marker=ROOT/"jobs"/("cpu_"+key+".json")
 if read(marker).get("state")=="complete":return
 log=ROOT/"logs"/(module.rsplit(".",1)[-1]+"_cpu.log")
 with log.open("a") as f:
  subprocess.run([sys.executable,"-u","-m",module,*args],cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4"),
   stdout=f,stderr=subprocess.STDOUT,check=True)
 write(marker,dict(state="complete",module=module,args=args))
def downstream(phase_function=phase):
 from .paths import evaluate as readout_paths
 from .report import report
 from .evaluation import evaluate as evaluate_readouts
 # New localization and all-node unedited dev path features are independent.
 phase_function("qwen_localize_devpaths",[
  (0,["experiments.aster_cr.mechanisms","--stage","localize"]),
  (1,["experiments.aster_cr.paths","--split","dev","--shard","0","--shards","2"]),
  (6,["experiments.aster_cr.paths","--split","dev","--shard","1","--shards","2"])])
 phase_function("qwen_selectivity_and_comparator_fit",[
  (0,["experiments.aster_cr.mechanisms","--stage","h2"]),
  (1,["experiments.aster_cr.comparators","--fit"]),
  (6,["experiments.aster_cr.historical_h3"])])
 cpu("experiments.aster_cr.mechanisms","--stage","h3")
 cpu("experiments.aster_cr.diagnostics")
 cpu("experiments.aster_cr.readout")
 readout_paths("qwen","dev");choose("qwen")
 cpu("experiments.aster_cr.ablate")
 readout_paths("qwen","dev")
 jobs=[["experiments.aster_cr.baselines","--kind",kind,"--lr",str(lr)] for kind in ("loreft","lofit","lora","last_layer") for lr in (.0001,.001,.01)]
 for i in range(0,len(jobs),3):phase_function("qwen_training_"+str(i//3),list(zip(GPUS,jobs[i:i+3])))
 phase_function("qwen_comparators_dev",[(gpu,["experiments.aster_cr.comparators","--split","dev","--shard",str(i)]) for i,gpu in enumerate(GPUS)])
 choose("qwen",final=True);report(final=False)
 # All selections and baseline winners are now frozen. Collect each test input once.
 phase_function("qwen_test_features",[(gpu,["experiments.aster_cr.score","--split","test","--shard",str(i)]) for i,gpu in enumerate(GPUS)])
 phase_function("qwen_test_paths",[(gpu,["experiments.aster_cr.paths","--split","test","--shard",str(i)]) for i,gpu in enumerate(GPUS)])
 evaluate_readouts("qwen");readout_paths("qwen","test",read(ROOT/"qwen/frozen.json")["readouts"])
 cpu("experiments.aster_cr.diagnostics","--test")
 frozen=read(ROOT/"qwen/frozen.json")
 jobs=[]
 for name in frozen["baseline_families"].values():
  kind,lr=name.split("_lr");jobs.append(["experiments.aster_cr.baselines","--kind",kind,"--lr",lr,"--split","test","--evaluate"])
 jobs.extend([["experiments.aster_cr.comparators","--split","test","--shard",str(i)] for i in range(3)])
 for i in range(0,len(jobs),3):phase_function("qwen_final_comparisons_"+str(i//3),list(zip(GPUS,jobs[i:i+3])))
 status("qwen",state="complete");report(final=False)
 # MedGemma uses the same sources and only the transferred recipe plus its own fit estimates.
 for split in ("fit","dev"):
  phase_function("medgemma_"+split,[(6,["experiments.aster_cr.score","--model","medgemma","--split",split,"--shards","1"])])
 phase_function("medgemma_localize",[(6,["experiments.aster_cr.mechanisms","--model","medgemma","--stage","localize"])])
 phase_function("medgemma_h2",[(6,["experiments.aster_cr.mechanisms","--model","medgemma","--stage","h2"])])
 phase_function("medgemma_historical_h3",[(6,["experiments.aster_cr.historical_h3","--model","medgemma"])])
 cpu("experiments.aster_cr.mechanisms","--model","medgemma","--stage","h3")
 cpu("experiments.aster_cr.diagnostics","--model","medgemma")
 cpu("experiments.aster_cr.readout","--model","medgemma")
 phase_function("medgemma_devpaths",[(6,["experiments.aster_cr.paths","--model","medgemma","--split","dev","--shards","1"])])
 readout_paths("medgemma","dev");choose("medgemma")
 strongest=frozen["strongest_baseline"]
 if strongest:
  kind,lr=strongest.split("_lr")
  phase_function("medgemma_training_confirmation",[(6,["experiments.aster_cr.baselines","--model","medgemma","--kind",kind,"--lr",lr])])
 choose("medgemma",final=True)
 phase_function("medgemma_test",[(6,["experiments.aster_cr.score","--model","medgemma","--split","test","--shards","1"])])
 phase_function("medgemma_testpaths",[(6,["experiments.aster_cr.paths","--model","medgemma","--split","test","--shards","1"])])
 evaluate_readouts("medgemma");readout_paths("medgemma","test",read(ROOT/"medgemma/frozen.json")["readouts"])
 cpu("experiments.aster_cr.diagnostics","--model","medgemma","--test")
 if strongest:
  phase_function("medgemma_baseline_test",[(6,["experiments.aster_cr.baselines","--model","medgemma","--kind",kind,"--lr",lr,"--split","test","--evaluate"])])
 status("medgemma",state="complete")
 report(final=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--after-preparation",action="store_true");a=p.parse_args()
 if a.after_preparation:
  status("downstream",state="waiting_for_preparation")
  while read(ROOT/"status/preparation.json").get("state")!="complete":
   if read(ROOT/"status/pipeline.json").get("state")=="failed":raise RuntimeError("Preparation failed; downstream was not launched")
   time.sleep(30)
 with (ROOT/"controller.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  try:downstream();status("pipeline",state="complete")
  except BaseException as exc:
   import traceback
   status("pipeline",state="failed",error=str(exc),traceback=traceback.format_exc())
   from .report import report
   report(final=False);raise
if __name__=="__main__":main()
