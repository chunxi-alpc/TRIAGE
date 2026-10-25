"""Finalize only completed real GPU runs; never manufactures missing outputs."""
import gzip,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from reproduce import ROOT,prepare,metric,predict
from balanced_analysis import paired
from render_report import render
RUN=Path("/home/gcx/aster_dynamic_review_20261025")
def main():
    execution={"state":"complete_verified","models":{},"scope":"development only; no independent new-tree evaluation"}
    frozen=json.loads((ROOT/"frozen_candidate.json").read_text())
    archive=ROOT/"fresh_numeric";archive.mkdir(exist_ok=True)
    fixed={}
    for model in ("qwen","llama"):
        out=RUN/model
        if not (out/"DONE.json").exists(): raise RuntimeError(f"{model} has no completion marker; inspect private log")
        status=json.loads((out/"status.json").read_text())
        if status["completed"]!=800 or not status["complete"]: raise RuntimeError(f"{model} incomplete")
        raw=[json.loads(x) for x in (out/"predictions.jsonl").read_text().splitlines()]
        keys={(r["id"],r["variant"],r["seed"]) for r in raw}
        cached=ROOT/"data"/f"{model}_dev_seed0_{'logits_numeric' if model=='qwen' else 'numeric'}.jsonl"
        old=[json.loads(x) for x in cached.read_text().splitlines()]
        if len(raw)!=800 or len(keys)!=800 or keys!={(r["id"],r["variant"],r["seed"]) for r in old}: raise ValueError("key set mismatch")
        labels={(r["id"],r["variant"],r["seed"]):r["gold_option"] for r in old}
        for r in raw:
            if r["gold_option"]!=labels[(r["id"],r["variant"],r["seed"])]: raise ValueError("gold mismatch")
            for k,v in r["scores"].items():
                if not np.isfinite(v).all() or len(v)!=len(r["option_S"]): raise ValueError("invalid logits")
                if int(np.argmax(v))!=r["answers"][k]: raise ValueError(f"{model}: saved argmax mismatch: {k}")
        safe_keys=("id","tree","seed","variant","split","gold_option","p","option_S","answers","scores","seconds","total_seconds","tokens","condition_key","condition_reused","condition_seconds")
        public=[{k:r[k] for k in safe_keys} for r in raw]
        with gzip.open(archive/f"{model}.jsonl.gz","wt") as f:
            for r in public: f.write(json.dumps(r)+"\n")
        rows=[prepare(r) for r in raw]
        pred=[predict(r,frozen["parameters"][model]) for r in rows]
        fixed[model]={"metrics":metric(rows,np.array(pred)),
            "paired_vs_original":paired(rows,pred,[r["answers"]["original"] for r in rows]),
            "predictions":pred,"parameters":frozen["parameters"][model]}
        manifest=json.loads((out/"manifest.json").read_text())
        execution["models"][model]={**status,"manifest":manifest,
            "prediction_sha256":hashlib.sha256((out/"predictions.jsonl").read_bytes()).hexdigest(),
            "gpu_tests":json.loads((out/"gpu_tests.json").read_text())}
    execution["runner_sha256"]=hashlib.sha256((ROOT/"fresh_forward.py").read_bytes()).hexdigest()
    (ROOT/"execution.json").write_text(json.dumps(execution,indent=2)+"\n")
    (ROOT/"frozen_development_recheck.json").write_text(json.dumps(fixed,indent=2)+"\n")
    subprocess.run([sys.executable,str(ROOT/"balanced_analysis.py"),"--fresh-root",str(RUN),"--output",str(ROOT/"balanced_fresh.json")],check=True)
    subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT),"-p","test_*.py"],check=True)
    render(ROOT/"balanced_fresh.json",ROOT/"fresh_report.zh.md",True)
    (ROOT/"run_state.json").write_text(json.dumps({"state":"complete_verified","finished_unix":time.time(),"report":"fresh_report.zh.md","independent_new_tree_test":"not_available"},indent=2)+"\n")
if __name__=="__main__":
    try:main()
    except BaseException as e:
        (ROOT/"run_state.json").write_text(json.dumps({"state":"failed","error":str(e),"finished_unix":time.time()},indent=2)+"\n")
        traceback.print_exc();sys.exit(1)
