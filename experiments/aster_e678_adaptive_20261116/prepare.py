"""CPU preparation and split audit, with no historical cache or parameter import."""
import argparse
from pathlib import Path
from collections import Counter
from experiments.aster_cr_retrain_paired_v2_20261111.common import rows, read, write, lines
from .adaptive_m1 import representatives, pair_ids, Config
from dataclasses import asdict

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/aster_e678_adaptive_20261116"
SOURCE = Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
KEYS = ("qwen","llama31","gemma2","gemma3","medgemma","qwen_base","gemma2_base")

def audit(data):
    fit = data["E1_pairs.jsonl"]
    hist = data["historical_pairs.jsonl"]
    dev, test = ([p for p in hist if p["split"] == split] for split in ("dev","test"))
    sets = [{p["source_id"] for p in rr} for rr in (fit,dev,test)]
    if any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError("Source split overlap")
    if any(p["split"] != "fit" for p in fit):
        raise ValueError("E1 contains non-training pairs")
    if Counter(p["kind"] for p in fit) != Counter(dict(A=635,P=635,C=635,F=635)):
        raise ValueError("E1 pair counts conflict")
    if list(map(len,sets)) != [80,20,40]:
        raise ValueError("Expected source split 80/20/40")
    if any(v["split"] != "fit" or v["source_id"] not in sets[0]
           for name in ("E1_views.jsonl","natural_views.jsonl") for v in data[name]):
        raise ValueError("Training/calibration metadata outside training sources")
    counts = {k:len(v) for k,v in representatives(dev,"dev").items()}
    return dict(fit_pairs=len(fit), source_counts=dict(zip(("fit","dev","test"),map(len,sets))),
                fit_category_sources={k:len(v) for k,v in representatives(fit,"fit").items()},
                dev_category_sources=counts, disjoint=True,
                insufficient_categories=[k for k,n in counts.items() if n<16],
                missing_five_donor_categories=[k for k,n in counts.items() if n<6],
                protocol_conflict=any(n<16 for n in counts.values()),
                test_labels_used_for_selection=False, raw_cache_imported=False)

def prepare():
    names=("E1_pairs.jsonl","E1_views.jsonl","natural_views.jsonl",
           "historical_pairs.jsonl","historical_views.jsonl","path_features.jsonl","path_interference_views.jsonl")
    data={name:rows(SOURCE/"data"/name) for name in names}
    result=audit(data)
    completed=OUT/"development/dev_pairs.jsonl"
    if completed.exists():
        completed_pairs=rows(completed)
        rep=representatives(completed_pairs,"dev")
        old_sources={p["source_id"] for p in data["historical_pairs.jsonl"] if p["split"]=="dev"}
        if {p["source_id"] for p in completed_pairs} != old_sources:
            raise ValueError("Development completion changed source membership")
        result["original_dev_category_sources"]=result["dev_category_sources"]
        result["dev_category_sources"]={k:len(v) for k,v in rep.items()}
        result["insufficient_categories"]=[k for k,v in rep.items() if len(v)<16]
        result["missing_five_donor_categories"]=[k for k,v in rep.items() if len(v)<6]
        result["protocol_conflict"]=bool(result["insufficient_categories"])
        result["development_completion"]=read(OUT/"development/manifest.json")

    write(OUT/"data_audit.json",result)
    specs=read(REPO/"experiments/aster_e678/models.json")
    cases=rows(Path("/home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl"))
    jobs=[]
    for key in KEYS:
        root=OUT/"models"/key
        for split in ("dev","test"):
            lines(root/"data"/(split+"_views.jsonl"),[v for v in data["historical_views.jsonl"] if v["split"]==split])
            lines(root/"data"/(split+"_evaluation_pairs.jsonl"),[v for v in data["historical_pairs.jsonl"] if v["split"]==split])
            lines(root/"data"/(split+"_path_views.jsonl"),[v for name in ("path_features.jsonl","path_interference_views.jsonl")
                for v in data[name] if v["split"]==split])
            lines(root/"data"/(split+"_cases.jsonl"),[v for v in cases if v["split"]==split])
        # Refuse to replace any prepared configuration; a new data version needs explicit re-preparation.
        if (root/"configuration.json").exists():
            continue
        for name, rr in data.items():
            lines(root/"data"/name, rr)
        dev=[p for p in data["historical_pairs.jsonl"] if p["split"]=="dev" and p["kind"] in "APCF"]
        if completed.exists():
            dev=completed_pairs
        lines(root/"data/dev_pairs.jsonl",dev)
        lines(root/"data/extra_inputs.jsonl",rows(OUT/"development/inputs.jsonl"))
        stages={"fit":sorted(pair_ids(data["E1_pairs.jsonl"]) |
                            {v["view_id"] for v in data["natural_views.jsonl"]})}
        for split in ("dev","test"):
            stages[split]=sorted({v["view_id"] for name in ("historical_views.jsonl","path_features.jsonl",
                "path_interference_views.jsonl") for v in data[name] if v["split"]==split} |
                pair_ids([p for p in data["historical_pairs.jsonl"] if p["split"]==split]))
        stages["dev"]=sorted(set(stages["dev"])|pair_ids(dev))
        # Fit gradients cover all E1 pairs so a fresh matched fixed-51 baseline remains possible.
        write(root/"gradient_ids.json",sorted({p["left"] for p in data["E1_pairs.jsonl"]}))
        write(root/"stage_ids.json",stages)
        configuration=dict(model_key=key,model=specs[key],adaptive=asdict(Config()),
            rank=16,regularization=.01,delta=.2923349,
            score_protocol="native final-token YES-NO; BF16 backbone, FP32 projection, native softcap",
            source_report=str(SOURCE),source_counts=result["source_counts"],
            raw_cache_reused=False,learned_parameters_reused=False,
            state="awaiting_data_resolution_and_unified_queue",allowed_gpus=[0,1,5,6])
        write(root/"configuration.json",configuration)
    return result

if __name__=="__main__":
    prepare()
