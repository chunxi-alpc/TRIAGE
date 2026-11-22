"""Score missing exact interventions on one authorized GPU."""
import argparse,json,os,sqlite3,time,traceback
import numpy as np
from .common import OUT,RUN,SOURCE,Features,read,write
from experiments.aster_cr_retrain_paired_v2_20261111.common import load_inputs

def run(worker):
    gpu=os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu not in ("0","1","5","6"):
        raise ValueError("Use an authorized GPU: 0, 1, 5, or 6.")
    root=OUT/"workers"/str(worker);root.mkdir(parents=True,exist_ok=True)
    requests=read(RUN/("requests_"+str(worker)+".json"),[])
    if not requests:
        write(root/"completion.json",dict(complete=True,total=0,gpu=gpu));return
    conn=sqlite3.connect(RUN/("new_scores_"+str(worker)+".sqlite"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS scores (target TEXT,donor TEXT,head INTEGER,value REAL,PRIMARY KEY(target,donor,head)) WITHOUT ROWID")
    import torch
    torch.set_num_threads(4)
    from experiments.aster_e678.engine import Engine
    from .suffix import SuffixRunner
    write(root/"progress.json",dict(state="loading",gpu=gpu,pid=os.getpid(),total=len(requests)))
    engine=Engine("qwen",metadata_path=root/"model_metadata.json")
    suffix=SuffixRunner(engine);features=Features(("fit",));inputs=load_inputs(SOURCE)
    start=time.monotonic();current=None;context=None;done=0;computed=0;baselines=0;checked=False
    for request in requests:
        a,b,h=request["target"],request["donor"],int(request["head"])
        if conn.execute("SELECT 1 FROM scores WHERE target=? AND donor=? AND head=?",(a,b,h)).fetchone():
            done+=1;continue
        if current!=a:
            if context is not None:
                suffix.context=None;suffix.inputs={};suffix.attention={};suffix.kwargs={};del context
            context=engine.prepare(inputs[a]["prompt"])
            fresh=suffix.prepare(context);baselines+=1
            if not np.isclose(fresh,features.ell(a),rtol=1e-5,atol=1e-4):
                raise RuntimeError("Replayed baseline differs from the reused score for "+a)
            current=a
        donor=features.heads(b)[h]
        value=suffix.score(h,donor)
        if not checked:
            original=float(engine.run(context,replacements={h:donor},capture=False)["ell"])
            write(root/"replay_check.json",dict(serial=original,suffix=value,absolute_difference=abs(original-value),target=a,donor=b,head=h))
            if original!=value:
                raise RuntimeError("Suffix replay changed the scalar score.")
            checked=True
        if not np.isfinite(value):raise RuntimeError("Nonfinite intervention score")
        conn.execute("INSERT INTO scores VALUES (?,?,?,?)",(a,b,h,value))
        computed+=1;done+=1
        if done<=3 or done%100==0:
            conn.commit()
            write(root/"progress.json",dict(state="running",gpu=gpu,pid=os.getpid(),completed=done,total=len(requests),
               new_scores=computed,baseline_forwards=baselines,seconds=time.monotonic()-start,
               peak_gib=torch.cuda.max_memory_allocated()/2**30))
    conn.commit();conn.close()
    write(root/"completion.json",dict(complete=True,gpu=gpu,total=len(requests),new_scores=computed,
        seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=int,required=True);args=parser.parse_args()
    try:run(args.worker)
    except Exception as e:
        write(OUT/"workers"/str(args.worker)/"failure.json",dict(error=str(e),traceback=traceback.format_exc()))
        raise
