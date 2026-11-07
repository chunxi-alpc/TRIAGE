import math
import torch
from .common import *
from experiments.aster_learnable_m3.m3 import LearnableM3
from experiments.aster_learnable_m3.data import target_prompt,reference_prompt,canonical_guide
def frozen():return torch.load(OLD/"frozen_m1_m2.pt",map_location="cpu",weights_only=False)
def load_editor(engine,name,selected=True):
    if name in ["learned_reference","learned_self"]:
        mode=name.removeprefix("learned_");basis=frozen();path=OLD/mode/"selected.pt"
        spec=dict(name=name,reference=mode,paired=False)
    else:
        spec=next(v for v in read(ROOT/"configuration.json")["variants"] if v["name"]==name)
        basis=torch.load(ROOT/"bases"/f"{name}.pt",map_location="cpu",weights_only=False)
        path=ROOT/"models"/name/"selected.pt"
    model=LearnableM3(basis,engine.n_heads).to(engine.device)
    if selected:model.load_state_dict(torch.load(path,map_location=engine.device,weights_only=False)["state_dict"])
    return model,spec
def ref_heads(engine,prompt,task="binary"):
    c=engine.prepare(prompt,task=task);out=engine.run(c)
    return out["heads"].flatten(0,1).to(engine.device)
def diagnostics(result):
    return {k:float(result[k].detach()) for k in ["magnitude","signal_change","gate"] if k in result}
def node(engine,case,gid,name,editor=None,spec=None,basis=None,bias=0.):
    if name=="direct_reference":
        c=engine.prepare(reference_prompt(case,gid));r=engine.run(c,capture=False)
    else:
        ref=None
        if spec and spec["reference"]=="reference":ref=ref_heads(engine,reference_prompt(case,gid))
        c=engine.prepare(target_prompt(case,gid))
        if name in ["full","bias"]:r=engine.run(c,capture=False)
        elif name=="aster":r=engine.run(c,operators=basis,capture=False)
        else:
            if spec["reference"]=="self":ref=engine.run(c)["heads"].flatten(0,1).to(engine.device)
            r=engine.run(c,m3=editor,reference=ref,capture=False)
    return float(r["ell"])+(bias if name=="bias" else 0.),diagnostics(r)
def predict_case(engine,case,name,editor=None,spec=None,basis=None,bias=0.,cached_full=None):
    tree=canonical_guide(case["guideline"]);nodes={n["id"]:n for n in tree["nodes"]}
    gid="G0";path=[];ds=[]
    while gid not in tree["leaves"]:
        if name=="bias" and cached_full is not None and gid in cached_full:
            ell=float(cached_full[gid]["ell"])+bias;diag={}
        else:ell,diag=node(engine,case,gid,name,editor,spec,basis,bias)
        if not math.isfinite(ell):raise ValueError("Nonfinite score")
        label=int(ell>0);path.append(dict(gid=gid,label=label,ell=ell,**diag));ds.append(diag)
        gid=nodes[gid]["yes" if label else "no"]
        if len(path)>len(nodes):raise ValueError("Cyclic tree")
    predicted=[dict(gid=n["gid"],label=n["label"]) for n in path]
    return dict(case_id=case["case_id"],source_id=case["source_id"],method=name,predicted_path=predicted,
        predicted_leaf=gid,predictions=path,correct=predicted==case["gold_path"] and gid==case["gold_leaf"],
        magnitude=sum(d.get("magnitude",0) for d in ds)/max(1,len(ds)),status="complete")
def mcq(engine,case,name,editor=None,spec=None,basis=None,bias=0.):
    if name=="direct_reference":raise ValueError("A profile-only input has no option-letter mapping; use it only as an activation donor")
    r=case["native"];letters=list("ABCDEFGH"[:len(r["options"])])
    ids=[engine.tokenizer.encode(x,add_special_tokens=False) for x in letters]
    if any(len(x)!=1 for x in ids):raise ValueError("MCQ labels must be single tokens")
    ref=None
    # profile is the released patient/question text. Never use native.path or answer_text.
    donor=r["profile"]+"\nAssess the patient's clinical facts. Answer:"
    if spec and spec["reference"]=="reference":ref=ref_heads(engine,donor,task="mcq")
    prompt=donor if name=="direct_reference" else r["prompt"]
    c=engine.prepare(prompt,task="mcq")
    kwargs=dict(capture=False,label_ids=[x[0] for x in ids])
    if name=="aster":out=engine.run(c,operators=basis,**kwargs)
    elif editor is not None:
        if spec["reference"]=="self":ref=engine.run(c)["heads"].flatten(0,1).to(engine.device)
        out=engine.run(c,m3=editor,reference=ref,**kwargs)
    else:out=engine.run(c,**kwargs)
    logits=list(out["label_logits"]);logits[0]+=bias if name=="bias" else 0.
    prediction=letters[int(np.argmax(logits))]
    return dict(case_id=case["sample_id"],source_id=case["source_id"],method=name,prediction=prediction,
        label=r["answer"],correct=prediction==r["answer"],label_logits=logits,tokens=c.length,
        status="complete",**diagnostics(out))
def fit_bias(records,binary=False):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    def loss(b):
        vals=[]
        for r in records:
            logits=np.array([r["ell"],0.] if binary else r["label_logits"],dtype=float)
            logits[0]+=b
            y=(0 if r["label"] else 1) if binary else "ABCDEFGH".index(r["label"])
            vals.append(logsumexp(logits)-logits[y])
        return float(np.mean(vals))
    result=minimize_scalar(loss,bounds=(-20,20),method="bounded")
    return dict(bias=float(result.x),train_n=len(records),objective="mean CE, one additive bias for yes (binary) or A (MCQ)",loss=float(result.fun))
