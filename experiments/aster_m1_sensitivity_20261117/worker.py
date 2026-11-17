"""Queue-owned GPU worker: score only previously unmeasured target/donor/head requests."""
import argparse,fcntl,gzip,json,os,sqlite3,subprocess,time,traceback
from functools import lru_cache
from pathlib import Path
from .prepare import REPO,OUT,RUN,OLDQ,PRIVATE,DATA
from .audit import read,lines
from experiments.aster_cr_retrain_paired_v2_20261111.common import write,load_inputs
from experiments.aster_e678_adaptive_20261116.run import require_queue

def publish(stage):
    with (RUN/"publish.lock").open("a") as lock,(RUN/"publish.log").open("a") as log:
        fcntl.flock(lock,fcntl.LOCK_EX)
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()
        if branch!="audit/aster-m1-sensitivity-20261117":raise RuntimeError("Unexpected publication branch")
        subprocess.run(["git","add","-f","--","reports/aster_m1_sensitivity_20261117"],cwd=REPO,stdout=log,stderr=log,check=True)
        if subprocess.run(["git","diff","--cached","--quiet"],cwd=REPO).returncode:
            subprocess.run(["git","commit","-m","results(aster): M1 sensitivity "+stage],cwd=REPO,stdout=log,stderr=log,check=True)
        subprocess.run(["git","push","origin","HEAD:audit/aster-m1-sensitivity-20261117"],cwd=REPO,stdout=log,stderr=log,check=True)

def run(worker):
    gpu=require_queue()
    root=OUT/"workers"/str(worker);root.mkdir(parents=True,exist_ok=True)
    with (RUN/("worker_"+str(worker)+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        requests=read(RUN/("requests_"+str(worker)+".json"))
        conn=sqlite3.connect(RUN/("new_scores_"+str(worker)+".sqlite"))
        conn.execute("PRAGMA journal_mode=WAL");conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE IF NOT EXISTS scores (target TEXT, donor TEXT, head INTEGER, value REAL, PRIMARY KEY(target,donor,head)) WITHOUT ROWID")
        records={r["uid"]:r for r in lines(PRIVATE/"cache/scores_fit.jsonl")}
        if any(r["target"] not in records or r["donor"] not in records for r in requests):
            raise ValueError("Request outside fit cache")
        @lru_cache(maxsize=96)
        def heads(uid):
            import numpy as np
            with np.load(records[uid]["feature"],allow_pickle=False) as data:
                return data["heads"].astype("float32").reshape(1152,-1)
        import torch,numpy as np
        from experiments.aster_e678.engine import Engine
        write(root/"progress.json",dict(state="loading",pid=os.getpid(),gpu=gpu))
        engine=Engine("qwen",metadata_path=root/"model_metadata.json")
        original=read(OLDQ/"model_metadata.json")
        current=engine.describe()
        fields=["path","model_type","num_layers","num_attention_heads","head_dim","answer_token_ids","prompt_policy","dtype"]
        if any(current[k]!=original[k] for k in fields):
            raise ValueError("Checkpoint/scoring geometry differs from reused cache")
        inputs=load_inputs(DATA)
        use_suffix=os.environ.get("ASTER_M1_SUFFIX_ENABLED")=="1"
        if use_suffix:
            from .suffix import SuffixRunner
            suffix=SuffixRunner(engine)
        else:suffix=None
        started=time.monotonic();completed=0;new=0;baseline=0;current_uid=None;context=None
        for request in requests:
            a,b,h=request["target"],request["donor"],request["head"]
            if conn.execute("SELECT 1 FROM scores WHERE target=? AND donor=? AND head=?",(a,b,h)).fetchone():
                completed+=1;continue
            if current_uid!=a:
                context=engine.prepare(inputs[a]["prompt"])
                fresh=suffix.prepare(context) if suffix is not None else float(engine.run(context,capture=False)["ell"]);baseline+=1
                cached=float(records[a]["ell"])
                if abs(fresh-cached)>1e-4+1e-5*abs(fresh):
                    raise ValueError("Fresh baseline differs from training cache")
                current_uid=a
            value=suffix.score(h,heads(b)[h]) if suffix is not None else float(engine.run(context,replacements={h:heads(b)[h]},capture=False)["ell"])
            if not np.isfinite(value):raise ValueError("Nonfinite intervention score")
            if a==b and abs(value-float(records[a]["ell"]))>1e-4+1e-5*abs(float(records[a]["ell"])):
                raise ValueError("Self-patch identity failed")
            conn.execute("INSERT INTO scores VALUES (?,?,?,?)",(a,b,h,value));conn.commit()
            completed+=1;new+=1
            if completed<=3 or completed%500==0:
                write(root/"progress.json",dict(state="running",pid=os.getpid(),gpu=gpu,
                    completed=completed,total=len(requests),new_scores=new,baseline_forwards=baseline,acceleration="suffix" if use_suffix else "serial",
                    seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30))
        with gzip.open(root/"new_scores.jsonl.gz","wt",encoding="utf-8") as f:
            for a,b,h,v in conn.execute("SELECT target,donor,head,value FROM scores ORDER BY target,donor,head"):
                f.write(json.dumps(dict(target=a,donor=b,head=h,score=v))+"\n")
        write(root/"completion.json",dict(complete=True,worker=worker,gpu=gpu,requested_scores=len(requests),
            stored_scores=conn.execute("SELECT count(*) FROM scores").fetchone()[0],
            new_scores_this_invocation=new,baseline_forwards=baseline,acceleration="suffix" if use_suffix else "serial",seconds=time.monotonic()-started,
            peak_gib=torch.cuda.max_memory_allocated()/2**30))
        conn.close()
        del engine
        # GPU process exits immediately; CPU analysis publishes the completed patch phase.

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker",type=int,choices=range(4),required=True)
    a=p.parse_args()
    try:run(a.worker)
    except Exception as e:
        write(OUT/"workers"/str(a.worker)/"failure.json",dict(error=str(e),traceback=traceback.format_exc()))
        try:publish("worker "+str(a.worker)+" failed")
        except Exception:pass
        raise
