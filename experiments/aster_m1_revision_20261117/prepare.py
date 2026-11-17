"""Read-only E1 data audit and deterministic manifests; no model is loaded."""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import json
from experiments.aster_cr_retrain_paired_v2_20261111.common import rows, read, write
from .revision import Config, KINDS, partition_sources, validate_pairs

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/aster_m1_revision_20261117"
PRIVATE = Path("/home/gcx/aster-m1-revision-runtime-20261117")
SOURCE = Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
HIST = Path("/home/gcx/aster_cr_repair_v2_run_20261108")
OLD = Path("/home/gcx/aster-e678-adaptive-20261116/reports/aster_e678_adaptive_20261116")
KEYS = ("qwen", "llama31", "gemma2", "gemma3", "medgemma", "qwen_base", "gemma2_base")
DATA_FILES = ("E1_pairs.jsonl", "E1_views.jsonl", "natural_views.jsonl",
              "historical_pairs.jsonl", "historical_views.jsonl",
              "path_features.jsonl", "path_interference_views.jsonl")

def datasets():
    return {name: rows(SOURCE / "data" / name) for name in DATA_FILES}

def audit(data):
    pairs=data["E1_pairs.jsonl"]
    validate_pairs(pairs)
    if Counter(p["kind"] for p in pairs) != Counter(dict.fromkeys(KINDS,635)):
        raise ValueError("E1 category counts conflict")
    hist=data["historical_pairs.jsonl"]
    sources={"fit":{p["source_id"] for p in pairs},
             **{s:{p["source_id"] for p in hist if p["split"]==s} for s in ("dev","test")}}
    if [len(sources[s]) for s in ("fit","dev","test")] != [80,20,40]:
        raise ValueError("Source counts conflict with 80/20/40")
    for a,b in (("fit","dev"),("fit","test"),("dev","test")):
        if sources[a] & sources[b]: raise ValueError("Source leakage")
    endpoint={}
    for name in ("E1_views.jsonl","natural_views.jsonl","historical_views.jsonl",
                 "path_features.jsonl","path_interference_views.jsonl"):
        for v in data[name]:
            split,source=v["split"],v["source_id"]
            if source not in sources[split]: raise ValueError("View source/split conflict")
            uid=v["view_id"]
            membership=(split,source)
            if uid in endpoint and endpoint[uid] != membership:
                raise ValueError("Input spans source boundaries: "+uid)
            endpoint[uid]=membership
    labels={}
    for v in data["E1_views.jsonl"]:
        if v["view_id"] in labels and labels[v["view_id"]] != v["label"]:
            raise ValueError("Endpoint labels conflict")
        labels[v["view_id"]]=v["label"]
    for p in pairs:
        for side in ("left","right"):
            if endpoint[p[side]] != ("fit",p["source_id"]) or labels[p[side]]!=p[side+"_label"]:
                raise ValueError("Pair endpoint membership/label conflict")
    screening,confirmation=partition_sources(pairs)
    return dict(passed=True,fit_pairs=len(pairs),fit_category_pairs=dict(Counter(p["kind"] for p in pairs)),
                source_counts={s:len(v) for s,v in sources.items()},
                sources={s:sorted(v) for s,v in sources.items()},source_disjoint=True,
                endpoint_membership_passed=True,label_transitions_passed=True,
                screening_sources=screening,confirmation_sources=confirmation,
                screening_category_sources={k:len({p["source_id"] for p in pairs if p["kind"]==k and p["source_id"] in screening}) for k in KINDS},
                confirmation_category_sources={k:len({p["source_id"] for p in pairs if p["kind"]==k and p["source_id"] in confirmation}) for k in KINDS},
                confirmation_category_pairs=dict(Counter(p["kind"] for p in pairs if p["source_id"] in confirmation)),
                dev_data="historical evaluation population, unchanged; no augmented selection pairs",
                evaluation_scope="adaptive parent branch: dev and 40-source historical holdout",
                test_labels_used_for_selection=False)

def prepare():
    if (OUT/"preparation.json").exists():
        raise FileExistsError("Preparation already exists; inspect it before resuming")
    data=datasets(); result=audit(data)
    fit=data["E1_pairs.jsonl"]
    grads=sorted({p[k] for p in fit for k in ("left","right")})
    stages={"fit":sorted(set(grads)|{v["view_id"] for v in data["natural_views.jsonl"]})}
    for scope in ("dev","test"):
        stages[scope]=sorted({v["view_id"] for name in ("historical_views.jsonl","path_features.jsonl","path_interference_views.jsonl")
                              for v in data[name] if v["split"]==scope})
        pair_ids={p[k] for p in data["historical_pairs.jsonl"] if p["split"]==scope for k in ("left","right")}
        if pair_ids-set(stages[scope]): raise ValueError("Evaluation pair endpoint absent from views")
    specs=read(REPO/"experiments/aster_e678/models.json")
    write(OUT/"data_audit.json",result)
    write(OUT/"stage_ids.json",stages)
    for key in KEYS:
        root=OUT/"models"/key
        old={r.get("uid",r.get("view_id")):r for r in rows(OLD/"models"/key/"cache/scores_fit.jsonl")}
        missing_old=[uid for uid in grads if not old.get(uid,{}).get("has_grads")]
        write(root/"gradient_plan.json",dict(required_ids=grads,required_count=len(grads),
              historical_missing_ids=missing_old,historical_missing_count=len(missing_old),
              extract_ids=grads,extract_count=len(grads),zero_fill=False,
              historical_cache_reused=False,
              reason="Fresh checkpoint-bound extraction; prior cache lacks complete weight identity"))
        cfg=dict(model_key=key,model=specs[key],M1=asdict(Config()),rank=16,delta=.2923349,regularization=.01,
                 allowed_gpus=[0,1,5,6],stage_counts={k:len(v) for k,v in stages.items()},
                 source=str(SOURCE),private_runtime=str(PRIVATE/key),
                 sensitivity_retained_mass=[.80,.95],performance_confidence_intervals=False,
                 fixed_51_rerun=False,automatic_parameter_selection=False,
                 evaluation_scope=result["evaluation_scope"])
        write(root/"configuration.json",cfg)
    manifest=dict(prepared=True,models=list(KEYS),stage_counts={k:len(v) for k,v in stages.items()},
                  unique_fit_pair_endpoints=len(grads),gpu_runs=0,
                  historical_cache_reused=False,queue_gpus=[0,1,5,6])
    write(OUT/"preparation.json",manifest)
    return manifest

if __name__=="__main__":
    print(json.dumps(prepare(),ensure_ascii=False))
