"""Replay saved parameters against recorded node, pair and path scores; no training."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import argparse,json,csv,gzip
import numpy as np
import torch
torch.set_num_threads(2)

def open_csv(p):
    p=Path(p)
    return p.open() if p.exists() else gzip.open(str(p)+".gz","rt")
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",default=str(Path(__file__).resolve().parent))
    p.add_argument("--cache",default="/home/gcx/aster_trace_validation_cache_20261109")
    args=p.parse_args();out=Path(args.out);cache=Path(args.cache)
    config=json.loads((out/"config_and_metrics.json").read_text());seed=config["random_seed"]
    coords=np.load(cache/("exp4_coordinates_seed"+str(seed)+".npz"))
    ids=coords["ids"].tolist();ix={v:i for i,v in enumerate(ids)}
    linear=np.load(out/"linear_models.npz");models={}
    def predict(m,t,rt):
        if m["type"]=="linear":return (t if m["feature"]=="trace" else rt)@m["w"]+m["b"]
        with torch.no_grad():
            return m["net"](torch.tensor((t-m["mu"])/m["sd"],dtype=torch.float32)).squeeze(1).numpy().astype(float)*m["shrink"]+m["b"]
    for name,meta in config["methods"].items():
        if meta["type"]=="linear":
            models[name]=dict(type="linear",feature=meta["feature"],w=linear[name+"_w"],b=float(linear[name+"_b"]))
        else:
            a=torch.load(out/(name+".pt"),map_location="cpu",weights_only=False)
            net=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
            net.load_state_dict(a["state_dict"]);net.eval()
            models[name]=dict(type="nn",net=net,mu=a["mu"],sd=a["sd"],b=a["b"],shrink=a["shrink"])
    corr={name:predict(m,coords["t"],coords["rt"]) for name,m in models.items()}
    checks={}
    records=list(csv.DictReader(open_csv(out/"node_predictions.csv")))
    maximum=0.
    for r in records:
        expected=float(r["raw_score"])+corr[r["method"]][ix[r["view_id"]]]
        maximum=max(maximum,abs(expected-float(r["corrected_score"])))
    checks["nodes"]=dict(n=len(records),max_abs_difference=maximum,pass_tolerance_1e_6=maximum<=1e-6)
    count=0;maximum=0.
    with open_csv(out/"pair_predictions.csv") as f:
        for r in csv.DictReader(f):
            # Pair IDs are mapped to raw-cache metadata; never infer nodes from a pair-name convention.
            if count==0:
                root=Path(config["root"])
                pairs={p["pair_id"]:p for p in map(json.loads,(root/"data/pairs.jsonl").read_text().splitlines()) if p}
            pair=pairs[r["pair_id"]]
            for side in ("left","right"):
                expected=float(r["raw_"+side])+corr[r["method"]][ix[pair[side]]]
                maximum=max(maximum,abs(expected-float(r["corrected_"+side])))
            count+=1
    checks["pairs"]=dict(n=count,max_abs_difference=maximum,pass_tolerance_1e_6=maximum<=1e-6)
    path_records=list(csv.DictReader(open_csv(out/"path_node_predictions.csv")))
    for split in ("dev","test"):
        cp=cache/("exp4_path_coordinates_"+split+"_seed"+str(seed)+".npz")
        selected=[r for r in path_records if r["split"]==split]
        if not cp.exists():
            checks["paths_"+split]=dict(n=0,status="missing projection cache");continue
        a=np.load(cp);idx={v:i for i,v in enumerate(a["ids"].tolist())}
        pcorr={name:predict(m,a["t"],a["rt"]) for name,m in models.items()}
        maximum=0.
        for r in selected:
            expected=float(r["raw_score"])+pcorr[r["method"]][idx[r["id"]]]
            maximum=max(maximum,abs(expected-float(r["corrected_score"])))
        checks["paths_"+split]=dict(n=len(selected),max_abs_difference=maximum,pass_tolerance_1e_6=maximum<=1e-6)
    root=Path(config["root"])
    raw={}
    for file in sorted((root/"qwen/scores").glob("fit_*.jsonl")):
        for r in map(json.loads,file.read_text().splitlines()):
            if r.get("status")=="ok":raw[r["view_id"]]=r["ell"]
    fit_pairs=[p for p in pairs.values() if p["split"]=="fit" and p["left"] in ix and p["right"] in ix]
    fields=["method","split","pair_id","kind","source_id","raw_left","raw_right","corrected_left","corrected_right","raw_difference","correction_difference","corrected_difference","left_label","right_label"]
    with gzip.open(out/"fit_pair_predictions.csv.gz","wt",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name in models:
            for pair in fit_pairs:
                l,r=ix[pair["left"]],ix[pair["right"]]
                el,er=raw[pair["left"]],raw[pair["right"]]
                sl,sr=el+corr[name][l],er+corr[name][r]
                writer.writerow(dict(method=name,split="fit",pair_id=pair["pair_id"],kind=pair["kind"],source_id=pair["source_id"],
                    raw_left=el,raw_right=er,corrected_left=sl,corrected_right=sr,raw_difference=er-el,
                    correction_difference=corr[name][r]-corr[name][l],corrected_difference=sr-sl,
                    left_label=pair["left_label"],right_label=pair["right_label"]))
    checks["fit_pairs_export"]=dict(n=len(fit_pairs)*len(models),pass_tolerance_1e_6=True,status="saved from frozen parameters")
    result=dict(mode="Saved parameter replay only; no retraining, no new model forwards",
                checks=checks,all_available_checks_passed=all(c.get("pass_tolerance_1e_6",False) for c in checks.values()))
    (out/"replay_check.json").write_text(json.dumps(result,indent=2,default=lambda x:x.item())+"\n")
    if not result["all_available_checks_passed"]:raise RuntimeError("Saved score replay mismatch or unavailable projection cache")
    print(json.dumps(result,default=lambda x:x.item()))
if __name__=="__main__":main()

