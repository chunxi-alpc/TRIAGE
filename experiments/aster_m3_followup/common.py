import json, os, time, gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/"results/aster_m3_followup_20261107"
OLD=Path("/home/gcx/TRIAGE/results/aster_learnable_m3_20261106")
LEGACY=Path("/home/gcx/aster_existing_data_20261106")
DIAG=Path("/home/gcx/m1_diagnostics_20261106")
SEED=20261107
METHODS=("full","aster","learned_reference","learned_self","direct_reference")
def rows(path):
    p=Path(path)
    if not p.exists(): p=Path(str(p)+".gz")
    if not p.exists(): return []
    op=gzip.open if p.suffix==".gz" else open
    with op(p,"rt",encoding="utf-8") as f:return [json.loads(s) for s in f if s.strip()]
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f".{os.getpid()}.tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def write_rows(path,rs):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rs))
def append(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a") as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+"\n")
def status(name,**v):write(ROOT/"status"/f"{name}.json",dict(updated=time.time(),**v))
def old_predictions():
    out={}
    for p in sorted((OLD/"test").glob("predictions_*.jsonl")):
        for r in rows(p):out[r["case_id"],r["method"]]=r
    return out
def paired(a,b,sources,seed=SEED,resamples=20000):
    from scipy.stats import binomtest
    a,b=np.asarray(a,dtype=int),np.asarray(b,dtype=int)
    names=sorted(set(sources));loc={s:i for i,s in enumerate(names)}
    idx=np.array([loc[s] for s in sources])
    sizes=np.bincount(idx,minlength=len(names))
    delta=np.bincount(idx,weights=a-b,minlength=len(names)).astype(int)
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(0,resamples,1000):
        draw=rng.integers(0,len(names),(min(1000,resamples-len(boot)),len(names)))
        boot.extend((delta[draw].sum(1)/sizes[draw].sum(1)).tolist())
    fixed=int(((a==1)&(b==0)).sum());harmed=int(((a==0)&(b==1)).sum())
    nonzero=delta[delta!=0];obs=abs(int(delta.sum()))
    if len(nonzero)<=20:
        sums=np.array([0])
        for d in nonzero:sums=np.concatenate([sums+d,sums-d])
        p=float((abs(sums)>=obs).mean());permutation_n=len(sums);exact=True
    else:
        signs=rng.choice([-1,1],size=(resamples,len(nonzero)))
        p=float((1+(abs(signs@nonzero)>=obs).sum())/(resamples+1));permutation_n=resamples;exact=False
    return dict(n=len(a),sources=len(names),corrected=fixed,degraded=harmed,
        net=fixed-harmed,difference=float((a-b).mean()),
        source_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
        mcnemar_exact_p=float(binomtest(fixed,fixed+harmed).pvalue) if fixed+harmed else 1.,
        source_permutation_p=p,source_permutation_exact=exact,source_permutation_draws=permutation_n,
        nonzero_source_count=len(nonzero),seed=seed,bootstrap_resamples=resamples)
