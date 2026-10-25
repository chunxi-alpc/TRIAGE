#!/usr/bin/env python3
"""Reproduce ASTER seed0 dev-only offline calibration and attention residual study.

Reads numeric, de-identified outputs from PR31/PR32. No GPU, no new inference,
and no use of a held-out test split. Four development trees only.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ATT = ("action_bias_1.0_all", "action_bias_2.0_all",
       "action_bias_2.0_last_half", "mass_1.0_last_half")

def norm(v):
    x = np.maximum(np.asarray(v, dtype=np.float64), 0.)
    t = x.sum()
    return x/t if t else np.full(len(x), 1./len(x))

def temp(v, t):
    a = np.log(np.maximum(np.asarray(v, dtype=float), 1e-30))/t
    return norm(np.exp(a - a.max()))

def softmax(v):
    a = np.asarray(v, dtype=float)
    return norm(np.exp(a-a.max()))

def prepare(row):
    p0 = softmax(row["scores"]["original"]) if "scores" in row else norm(row["none_q"])
    s = norm(row["option_S"])
    ranked = np.sort(s)[::-1]
    row["_orig"] = p0
    row["_support"] = s
    row["_features"] = {
        "sc":float(ranked[0]), "margin":float(ranked[0]-ranked[1]),
        "origmax":float(p0.max()), "sig":float(s.max()-s[np.argmax(p0)]),
        "entropy":float(-np.sum(s*np.log(np.maximum(s,1e-30)))/np.log(len(s))),
        "agree":float(np.argmax(p0)==np.argmax(s)),
    }
    return row

def fold_configurations(kind):
    if kind == "static":
        for beta,t0,ts,geo in itertools.product(
          (.5,.6,.7,.75,.8,.85,.9,.925,.95,.975,1.),
          (.5,1,1.5,2,3,4,6), (.25,1./3,.5,.75,1,1.5,2), (False, True)):
            yield dict(kind=kind,beta=beta,t0=t0,ts=ts,geo=geo)
    elif kind == "gate":
        for feat,threshold,hi,lo in itertools.product(
          ("margin","sc","sig","origmax","entropy","agree"),
          (.1,.25,.5,.75), (.5,.75,.95,1.), (.25,.5,.75)):
            if hi>lo:
                yield dict(kind=kind,feat=feat,threshold=threshold,hi=hi,lo=lo)
    elif kind == "linear":
        for b,w,feat in itertools.product(
          (.25,.5,.75),(-1.,-.5,.5,1.),("margin","sig","origmax","entropy")):
            yield dict(kind=kind,base=b,w=w,feat=feat)
    else:
        methods = ATT if kind=="residual" else ("action_bias_1.0_all",)
        for method,b,t0,g,scale in itertools.product(
            methods,(.75,.85,.9,.95),(1,2,3),(-.5,-.25,.1,.25,.5,1),(2,4)):
            yield dict(kind=kind,method=method,beta=b,t0=t0,gamma=g,scale=scale)

def seed_sign(name):
    h=0
    for ch in name:
        h=(h*33+ord(ch)) & 0xffffffff
    return 1 if h%2 else -1

def predict(row,c):
    p = temp(row["_orig"], c.get("t0",1))
    s = temp(row["_support"], c.get("ts",1))
    k = c["kind"]
    if k=="gate":
        b = c["hi"] if row["_features"][c["feat"]] >= c["threshold"] else c["lo"]
    elif k=="linear":
        b = np.clip(c["base"]+c["w"]*(row["_features"][c["feat"]]-0.5),0,1)
    else:
        b = c["beta"]
    if c.get("geo",False):
        mixed=norm(np.power(np.maximum(p,1e-30),1-b)*np.power(np.maximum(s,1e-30),b))
    else:
        mixed=(1-b)*p+b*s
    if k not in ("static","gate","linear"):
        orig=np.asarray(row["scores"]["original"], dtype=float)
        zero=np.asarray(row["scores"]["mass_zero_control"], dtype=float)
        if k=="numeric_null":
            delta=zero-orig
        else:
            edited=np.asarray(row["scores"][c["method"]],dtype=float)
            delta=edited-zero
        if k=="option_rotation":
            delta=np.roll(delta,-1)
        if k=="sign_flip":
            delta=delta*seed_sign(row["id"])
        shift=c["gamma"]*np.tanh(delta/c["scale"])
        mixed=norm(np.maximum(mixed,1e-30)*np.exp(shift-shift.max()))
    return int(np.argmax(mixed))

def metric(rows,preds):
    y=np.asarray([r["gold_option"] for r in rows])
    base=np.asarray([r["answers"]["original"] if "answers" in r else r["none"] for r in rows])
    hit=preds==y
    split={}
    for variant in ("NC","AC","IC","SW"):
        ix=np.asarray([r["variant"]==variant for r in rows])
        split[variant]=float(hit[ix].mean())
    return dict(correct=int(hit.sum()),accuracy=float(hit.mean()),
       harm_rate=float(np.mean((base==y)&(~hit))),
       correction_rate=float(np.mean((base!=y)&hit)),by_variant=split)

def crossfit(rows,kind):
    gold=np.asarray([r["gold_option"] for r in rows])
    tree=np.asarray([r["tree"] for r in rows])
    groups=sorted(set(tree))
    output=np.zeros(len(rows),dtype=np.int64)
    best_score={t:-1 for t in groups};chosen={}
    seen=0
    for cfg in fold_configurations(kind):
        seen+=1
        preds=np.fromiter((predict(r,cfg) for r in rows),int,count=len(rows))
        correct=preds==gold
        for held in groups:
            others=[t for t in groups if t!=held]
            score=float(np.mean([correct[tree==t].mean() for t in others]))
            if score>best_score[held]+1e-12:
                best_score[held]=score
                chosen[held]=cfg
                output[tree==held]=preds[tree==held]
    return dict(configs=seen,metrics=metric(rows,output),
                folds={t:dict(params=chosen[t],train_score=best_score[t],
                      heldout_score=float((output[tree==t]==gold[tree==t]).mean()))
                       for t in groups})

def main():
    summary={"scope":"seed0 development only; four trees; offline cached logits"}
    for model in ("qwen","llama"):
        src=ROOT/"data"/f"{model}_dev_seed0_{'logits_numeric' if model=='qwen' else 'numeric'}.jsonl"
        rows=[prepare(json.loads(line)) for line in src.read_text().splitlines() if line.strip()]
        assert len(rows)==800 and len({(r["id"],r["variant"]) for r in rows})==800
        assert {r["seed"] for r in rows}=={0} and {r["variant"] for r in rows}=={"NC","AC","IC","SW"}
        out={"trees":sorted({r["tree"] for r in rows}),"patients":len({r["id"] for r in rows})}
        if model=="qwen":
            assert sum(predict(r,dict(kind="static",beta=.75)) != r["answers"]["fusion_0_75"] for r in rows)==0
            assert sum(int(np.argmax(r["_support"])) != r["answers"]["direct_map"] for r in rows)==0
        else:
            assert sum(int(np.argmax(r["_orig"])) != r["none"] for r in rows)==0
        families=["static","gate","linear"]
        if model=="qwen":
            families+=["residual","numeric_null","option_rotation","sign_flip"]
        for family in families:
            out[family]=crossfit(rows,family)
            print(model,family,out[family]["metrics"]["accuracy"])
        summary[model]=out
    (ROOT/"seed0_offline_reproduced.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("Completed: offline development analysis, no new GPU inference")

if __name__=="__main__":
    main()
