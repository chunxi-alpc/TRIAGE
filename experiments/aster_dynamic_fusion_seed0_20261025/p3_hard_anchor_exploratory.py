#!/usr/bin/env python3
"""Exploratory, fixed-parameter hard-anchor fusion on the already-inspected P3 set.

No model inference. Reads p3_direct action support q and original answer.
Labels are used only to score the frozen method. This is post-hoc exploratory,
not a confirmatory test, because the method variant was devised after P3 results
had already been inspected.
"""
import gzip, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "aster_p0_p3_20261025" / "results" / "p3_direct.jsonl.gz"
OUT = ROOT / "p3_hard_anchor_exploratory.json"
CORE = {"NC", "AC", "IC", "SW"}
BETA = 0.8
TEMPERATURE = 0.25

def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])

def hard_anchor(row):
    q = [max(float(v), 1e-30) ** (1.0 / TEMPERATURE) for v in row["q"]]
    z = sum(q)
    q = [v / z for v in q]
    mixed = [BETA*q[i] + (1-BETA)*float(i == int(row["none"])) for i in range(len(q))]
    return argmax(mixed)

def bootstrap(tree_counts, seed, reps=10000):
    rng = random.Random(seed)
    names = sorted(tree_counts)
    draws = {"accuracy_delta": [], "harm_delta": [], "correction_delta": []}
    for _ in range(reps):
        n = cd = cf = hd = hf = fd = ff = 0
        for _ in names:
            t = tree_counts[rng.choice(names)]
            n += t["n"]; cd += t["direct"]; cf += t["fusion"]
            hd += t["harm_direct"]; hf += t["harm_fusion"]
            fd += t["fix_direct"]; ff += t["fix_fusion"]
        draws["accuracy_delta"].append((cf-cd)/n)
        draws["harm_delta"].append((hf-hd)/n)
        draws["correction_delta"].append((ff-fd)/n)
    def ci(vals):
        vals.sort()
        return [vals[int(.025*len(vals))], vals[min(len(vals)-1, int(.975*len(vals)))]]
    return {k: ci(v) for k, v in draws.items()}

def summarize(rows, model):
    rr = [r for r in rows if r["model"] == model]
    trees = defaultdict(lambda: {"n":0,"base":0,"direct":0,"fusion":0,
                                  "harm_direct":0,"harm_fusion":0,
                                  "fix_direct":0,"fix_fusion":0})
    variants = defaultdict(lambda: {"n":0,"base":0,"direct":0,"fusion":0,
                                    "harm_direct":0,"harm_fusion":0,
                                    "fix_direct":0,"fix_fusion":0})
    seeds = defaultdict(lambda: {"n":0,"base":0,"direct":0,"fusion":0})
    mismatch = 0
    q_not_norm = 0
    for r in rr:
        q = [float(v) for v in r["q"]]
        mismatch += int(argmax(q) != int(r["pred"]))
        q_not_norm += int(abs(sum(q)-1.0) > 1e-4)
        pred_f = hard_anchor(r)
        base_ok = int(r["none"]) == int(r["gold"])
        direct_ok = int(r["pred"]) == int(r["gold"])
        fusion_ok = pred_f == int(r["gold"])
        for store, key in ((trees,r["tree"]),(variants,r["variant"]),(seeds,str(r["seed"]))):
            x=store[key];x["n"]+=1
            x["base"]+=base_ok;x["direct"]+=direct_ok;x["fusion"]+=fusion_ok
            if store is not seeds:
                x["harm_direct"]+=base_ok and not direct_ok
                x["harm_fusion"]+=base_ok and not fusion_ok
                x["fix_direct"]+=(not base_ok) and direct_ok
                x["fix_fusion"]+=(not base_ok) and fusion_ok
    assert mismatch == 0, f"{model}: direct argmax mismatch {mismatch}"
    assert q_not_norm == 0, f"{model}: q not normalized on {q_not_norm} rows"
    n=sum(x["n"] for x in variants.values())
    def compact(x):
        return {"n":x["n"],"original_accuracy":x["base"]/x["n"],
                "direct_accuracy":x["direct"]/x["n"],"fusion_accuracy":x["fusion"]/x["n"],
                "direct_harm":x.get("harm_direct",0)/x["n"],
                "fusion_harm":x.get("harm_fusion",0)/x["n"],
                "direct_correction":x.get("fix_direct",0)/x["n"],
                "fusion_correction":x.get("fix_fusion",0)/x["n"]}
    tree_counts={k:v for k,v in trees.items()}
    return {
      "n":n,"tree_count":len(trees),"q_argmax_mismatches":mismatch,"q_normalization_failures":q_not_norm,
      "overall":compact({"n":n,"base":sum(x["base"] for x in variants.values()),
          "direct":sum(x["direct"] for x in variants.values()),
          "fusion":sum(x["fusion"] for x in variants.values()),
          "harm_direct":sum(x["harm_direct"] for x in variants.values()),
          "harm_fusion":sum(x["harm_fusion"] for x in variants.values()),
          "fix_direct":sum(x["fix_direct"] for x in variants.values()),
          "fix_fusion":sum(x["fix_fusion"] for x in variants.values())}),
      "by_variant":{k:compact(v) for k,v in variants.items()},
      "by_seed":{k:compact(v) for k,v in seeds.items()},
      "cluster_bootstrap_95":bootstrap(tree_counts, 20261025+(0 if model=="qwen" else 1))
    }

def main():
    rows=[]
    with gzip.open(SOURCE,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            if r["variant"] in CORE:
                rows.append(r)
    keys={(r["model"],r["id"],r["seed"],r["variant"]) for r in rows}
    assert len(rows)==24000 and len(keys)==24000
    assert {r["seed"] for r in rows}=={0,1,2}
    result={
      "protocol":{"status":"exploratory_post_hoc",
        "source":str(SOURCE),"rows":len(rows),"beta":BETA,
        "support_temperature":TEMPERATURE,
        "formula":"argmax(0.2*onehot(original_answer)+0.8*normalize(q**4))",
        "coverage":1.0,
        "warning":"The method variant was devised after inspecting prior P3 results; this is not a pristine confirmatory test."},
      "models":{m:summarize(rows,m) for m in ("qwen","llama")}
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({m:{"n":v["n"],"overall":v["overall"],
          "delta_vs_direct_ci":v["cluster_bootstrap_95"]} for m,v in result["models"].items()},
          ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
