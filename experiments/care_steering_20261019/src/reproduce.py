"""Re-run the frozen small experiment in a new server directory; never append to originals."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path
source=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument("--output-dir",required=True)
ap.add_argument("--gpu",default="6",choices=["0","1","2","3","6","7"])
ap.add_argument("--include-free-generation",action="store_true")
ap.add_argument("--analysis-only",action="store_true")
args=ap.parse_args();target=Path(args.output_dir).resolve()
if target.exists():raise SystemExit("Output directory already exists; choose a fresh path.")
target.mkdir(parents=True)
for folder in ["src","data","audit"]:
 shutil.copytree(source/folder,target/folder,ignore=shutil.ignore_patterns("__pycache__"))
(target/"sources").mkdir()
for name in ["care_v6_frozen.py","requirements_frozen.txt","facts_provenance.json","paper_method_audit.md","model_environment_manifest.json"]:
 if (source/"sources"/name).exists():shutil.copy2(source/"sources"/name,target/"sources"/name)
(target/"results").mkdir();(target/"logs").mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
def run(script,*extra):
 with (target/"logs"/(Path(script).stem+".log")).open("w") as log:
  subprocess.run([sys.executable,script,*extra],cwd=target,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if args.analysis_only:
 shutil.copytree(source/"results",target/"results",dirs_exist_ok=True)
else:
 run("src/run_facts.py")
 run("src/run_rules.py")
run("src/analyze.py","--predictions","results/predictions.jsonl","--data","data/cases.jsonl","--output","results/analysis.json","--markdown","results/analysis.md")
if args.include_free_generation:
 if not args.analysis_only:run("src/run_free.py")
 run("src/analyze.py","--predictions","results/free_predictions.jsonl","--data","data/cases.jsonl","--output","results/free_analysis.json","--markdown","results/free_analysis.md","--exploratory")
print(str(target))
