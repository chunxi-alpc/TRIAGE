"""Execute data collection, fit-only selection, frozen new-source validation and reporting."""
import os,sys,json,time,argparse,subprocess,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent;REPO=H.parents[3]
C=Path("/home/gcx/aster_trace_repair_20261110")
LOG=C/"logs";LOG.mkdir(parents=True,exist_ok=True)
ENV=dict(os.environ,OMP_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",MKL_NUM_THREADS="8")
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def status(phase,**kw):dump(H/"pipeline_status.json",dict(phase=phase,time=time.strftime("%Y-%m-%d %H:%M:%S"),pid=os.getpid(),gpus=[0,1,6],**kw))
def run(name,script,*args):
 with (LOG/(name+".log")).open("a") as out:
  subprocess.run([sys.executable,"-u",str(H/script),*args],cwd=REPO,env=ENV,stdout=out,stderr=subprocess.STDOUT,check=True)
def waitfile(p):
 while not Path(p).exists():
  failures=list((C/"scoring").glob("*/shard*/failed.json"))
  if failures:raise RuntimeError("Scoring failure: "+str(failures))
  time.sleep(15)
def scores(split):
 if not (C/"scoring"/split/"done.json").exists():run("collect_"+split,"collect.py","--split",split)
 # The data preparation owner freezes these tasks before this stage.
 waitfile(H/"data"/("diagnostic_"+split+"_ready.json"))
 if not (C/"scoring"/split/"diagnostic_done.json").exists():run("diagnostic_"+split,"collect.py","--split",split,"--diagnostic")
def fit():
 if not (H/"training/cv_done.json").exists():run("cv","train.py","--stage","cv")
 if not (H/"training/fit_done.json").exists() or not (H/"training/models.json").exists():run("fit","train.py","--stage","fit")
 run("replay_fit","training/replay_parameters.py")
def main():
 p=argparse.ArgumentParser();p.add_argument("--wait-current-fit",action="store_true");a=p.parse_args()
 started=time.time()
 try:
  status("collecting_fit")
  if a.wait_current_fit:waitfile(C/"scoring/fit/done.json")
  else:run("collect_fit","collect.py","--split","fit")
  status("training_and_new_development")
  with ThreadPoolExecutor(max_workers=2) as pool:
   training=pool.submit(fit);development=pool.submit(scores,"dev")
   training.result();development.result()
  status("development_selection")
  if not (H/"evaluation/frozen.json").exists():
   run("predict_dev","train.py","--stage","predict","--split","dev")
   run("evaluate_dev","evaluate.py","--stage","dev")
  frozen=json.loads((H/"evaluation/frozen.json").read_text())
  if not frozen.get("frozen") or frozen.get("final_scores_used"):raise RuntimeError("Invalid pre-final freeze")
  status("frozen_final_collection",selection=frozen["selection"])
  scores("final")
  status("final_evaluation")
  run("predict_final","train.py","--stage","predict","--split","final")
  run("evaluate_final","evaluate.py","--stage","final")
  run("report","report.py")
  if (H/"verify_saved.py").exists():run("replay","verify_saved.py")
  status("complete",seconds=time.time()-started)
 except Exception as e:
  status("failed",error=str(e),traceback=traceback.format_exc(),seconds=time.time()-started)
  raise
if __name__=="__main__":main()

