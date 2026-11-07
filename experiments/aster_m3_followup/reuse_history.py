"""Reuse identical historical pair inputs and scoring; no model execution."""
from .common import *

def run():
    views={v["view_id"]:v for v in rows(ROOT/"data/views.jsonl")}
    legacy={v["view_id"]:v for v in rows(LEGACY/"data/views.jsonl")}
    pairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="test"]
    needed={p[s] for p in pairs for s in ["left","right"]}
    summary={}
    for method,prefix in [("full","baseline"),("aster","final")]:
        path=ROOT/"predictions/pair_views"/f"{method}.jsonl"
        cached={r["view_id"]:r for r in rows(path)}
        eligible={}
        for file in sorted((LEGACY/"scores").glob(prefix+"_[01].jsonl")):
            for r in rows(file):
                vid=r.get("view_id")
                if vid not in needed or r.get("method")!=method or r.get("status")!="complete":continue
                a,b=views[vid],legacy[vid]
                if a["prompt"]!=b["prompt"] or a["label"]!=b["label"]:
                    raise ValueError("Historical input differs: "+vid)
                eligible[vid]=dict(view_id=vid,ell=r["ell"],label=a["label"],
                    prediction=int(r["ell"]>0),reused_from=str(file))
        added=0
        for vid,r in eligible.items():
            if vid not in cached:append(path,r);cached[vid]=r;added+=1
        summary[method]=dict(reusable_historical_views=len(eligible),newly_imported=added,
            required_views=len(needed),remaining_views=len(needed-set(cached)),
            source_directory=str(LEGACY/"scores"))
    write(ROOT/"history_reuse.json",dict(state="complete",methods=summary,
        policy="Identical view IDs, full prompts, labels, Qwen3-8B BF16 scoring and frozen closed-form operators. Calibration derives from saved full scores. Historical learned path predictions and checkpoints remain reused."))
    return summary

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False))
