"""M1: separate A/P attribution, bidirectional exact ranking, all dev prefixes."""
import json
import math
from pathlib import Path
import time
import torch

EPSILON = .1
RHO = .8
CANDIDATES = 64

def _json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)

def _rows(path):
    path=Path(path)
    if not path.exists():
        return []
    result=[]
    lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:
                raise
            raise RuntimeError("Incomplete M1 checkpoint tail; preserve and remove only the partial final line before resuming")
    return result

def _append(handle,row):
    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    handle.flush()

def _heads(feature):
    z=torch.as_tensor(feature["heads"],dtype=torch.float32)
    return z.reshape(-1,z.shape[-1])

def _prompt(views, view_id):
    return views[view_id]["prompt"]

def effective_pairs(kind, split, pairs, features, epsilon=EPSILON):
    selected=[]
    for pair in pairs:
        if pair["kind"]!=kind or pair["split"]!=split:
            continue
        te=float(features[pair["right"]]["ell"])-float(features[pair["left"]]["ell"])
        if not math.isfinite(te):
            raise ValueError("Non-finite original score: "+pair["pair_id"])
        if abs(te)>=epsilon:
            selected.append(pair)
    return selected

def select_prefix(curve,rho=RHO):
    """Empty prefix is a candidate; recovery is not clipped to a probability."""
    valid=[r for r in curve if r.get("recovery") is not None]
    reached=[r for r in valid if r["recovery"]>=rho]
    if reached:
        return min(reached,key=lambda r:r["k"])["k"],True
    if not valid:
        return 0,False
    best=max(r["recovery"] for r in valid)
    return min(r["k"] for r in valid if r["recovery"]==best),False

def _effects(engine, contexts, left, right, heads, te):
    zl,zr=_heads(left),_heads(right)
    forward=engine.run(contexts[0],replacements={h:zr[h] for h in heads},capture=False)["ell"]
    reverse=engine.run(contexts[1],replacements={h:zl[h] for h in heads},capture=False)["ell"]
    ie_forward=forward-float(left["ell"])
    ie_reverse=reverse-float(right["ell"])
    return dict(forward_ell=forward,reverse_ell=reverse,
                ie_forward=ie_forward,ie_reverse=ie_reverse,te=te,
                forward_normalized=ie_forward/te,reverse_normalized=ie_reverse/(-te),
                recovery=.5*(ie_forward/te+ie_reverse/(-te)))

def run_m1(kind,engine,views,pairs,features,outdir):
    if kind not in ("A","P"):
        raise ValueError("M1 is defined for A and P separately")
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    started=time.time()
    fit=effective_pairs(kind,"fit",pairs,features)
    dev=effective_pairs(kind,"dev",pairs,features)
    counts={split:dict(before=sum(p["kind"]==kind and p["split"]==split for p in pairs),
                       after=len(subset)) for split,subset in (("fit",fit),("dev",dev))}
    base=dict(kind=kind,epsilon=EPSILON,rho=RHO,candidate_limit=CANDIDATES,
              effective_pair_counts=counts,normalization="0.5*(IE_forward/TE+IE_reverse/(-TE))",
              ranking_data="fit",prefix_data="dev",unclipped=True)
    if not fit or not dev:
        result=dict(base,H=[],ranking=[],curve=[dict(k=0,recovery=0.)],target_met=False,
                    reason="no_effective_fit_pairs" if not fit else "no_effective_dev_pairs")
        _json(outdir/f"m1_{kind}.json",result)
        return result
    attr=torch.zeros(engine.num_heads,dtype=torch.float64)
    for i,pair in enumerate(fit):
        left,right=features[pair["left"]],features[pair["right"]]
        if left.get("grads") is None:
            context=engine.prepare(_prompt(views,pair["left"]))
            computed=engine.run(context,grad=True)
            left["grads"]=computed["grads"]
            del context
        z0,z1=_heads(left),_heads(right)
        gradients=torch.as_tensor(left["grads"],dtype=torch.float32).reshape_as(z0)
        te=float(right["ell"])-float(left["ell"])
        attr+=((z1-z0)*gradients).sum(-1).double()/te
    attr/=len(fit)
    candidates=sorted(range(engine.num_heads),key=lambda h:(-float(attr[h]),h))[:CANDIDATES]
    _json(outdir/f"attribution_{kind}.json",dict(base,ranking=[
        dict(head=h,layer=h//engine.n_heads,query_head=h%engine.n_heads,
             attribution=float(attr[h])) for h in sorted(range(engine.num_heads),
                   key=lambda h:(-float(attr[h]),h))],candidates=candidates))
    exact_path=outdir/f"exact_{kind}.jsonl"
    exact={(r["pair_id"],r["head"]):r for r in _rows(exact_path)}
    with exact_path.open("a") as handle:
        for i,pair in enumerate(fit):
            needed=[h for h in candidates if (pair["pair_id"],h) not in exact]
            if needed:
                left,right=features[pair["left"]],features[pair["right"]]
                te=float(right["ell"])-float(left["ell"])
                contexts=(engine.prepare(_prompt(views,pair["left"])),
                          engine.prepare(_prompt(views,pair["right"])))
                for h in needed:
                    row=dict(pair_id=pair["pair_id"],head=h,split="fit",kind=kind,
                             **_effects(engine,contexts,left,right,[h],te))
                    _append(handle,row);exact[(pair["pair_id"],h)]=row
                del contexts
            _json(outdir/f"status_{kind}.json",dict(stage="exact_ranking",done=i+1,total=len(fit),
                seconds=time.time()-started,forward_count=engine.counts["final_forwards"]))
    exact_mean={h:sum(exact[(p["pair_id"],h)]["recovery"] for p in fit)/len(fit) for h in candidates}
    ranking=sorted(candidates,key=lambda h:(-exact_mean[h],h))
    ranking_rows=[dict(head=h,layer=h//engine.n_heads,query_head=h%engine.n_heads,
                      attribution=float(attr[h]),nie=exact_mean[h]) for h in ranking]
    curve_path=outdir/f"prefix_pairs_{kind}.jsonl"
    saved={(r["pair_id"],r["k"]):r for r in _rows(curve_path)}
    with curve_path.open("a") as handle:
        for i,pair in enumerate(dev):
            needed=[k for k in range(1,len(ranking)+1) if (pair["pair_id"],k) not in saved]
            if needed:
                left,right=features[pair["left"]],features[pair["right"]]
                te=float(right["ell"])-float(left["ell"])
                contexts=(engine.prepare(_prompt(views,pair["left"])),
                          engine.prepare(_prompt(views,pair["right"])))
                for k in needed:
                    row=dict(pair_id=pair["pair_id"],k=k,split="dev",kind=kind,
                             **_effects(engine,contexts,left,right,ranking[:k],te))
                    _append(handle,row);saved[(pair["pair_id"],k)]=row
                del contexts
            _json(outdir/f"status_{kind}.json",dict(stage="joint_prefixes",done=i+1,total=len(dev),
                seconds=time.time()-started,forward_count=engine.counts["final_forwards"]))
    curve=[dict(k=0,recovery=0.,pairs=len(dev))]
    for k in range(1,len(ranking)+1):
        rows=[saved[(p["pair_id"],k)] for p in dev]
        curve.append(dict(k=k,recovery=sum(r["recovery"] for r in rows)/len(rows),pairs=len(rows),
                          forward_recovery=sum(r["forward_normalized"] for r in rows)/len(rows),
                          reverse_recovery=sum(r["reverse_normalized"] for r in rows)/len(rows)))
    k,met=select_prefix(curve)
    result=dict(base,H=ranking[:k],ranking=ranking_rows,curve=curve,target_met=met,
                reason="target_reached" if met else "highest_recovery_shortest_prefix",
                selected_k=k,selected_recovery=curve[k]["recovery"],
                elapsed_seconds=time.time()-started)
    _json(outdir/f"m1_{kind}.json",result)
    _json(outdir/f"status_{kind}.json",dict(stage="complete",seconds=time.time()-started,
          selected_k=k,target_met=met,forward_count=engine.counts["final_forwards"]))
    return result

def joint_recovery(engine,views,pairs,features,heads,*,epsilon=EPSILON,output=None):
    """Bidirectional joint patching for unions, random controls, or all heads.

    The caller chooses split/kind/subset. Zero-effect pairs remain in main
    behavior tables; this diagnostic reports both eligible and measured counts.
    """
    heads=[int(h) for h in heads]
    completed={r["pair_id"]:r for r in _rows(output)} if output else {}
    rows=[]
    handle=Path(output).open("a") if output else None
    try:
        for p in pairs:
            left,right=features[p["left"]],features[p["right"]]
            te=float(right["ell"])-float(left["ell"])
            if abs(te)<epsilon:
                continue
            if p["pair_id"] in completed:
                row=completed[p["pair_id"]]
            elif not heads:
                row=dict(pair_id=p["pair_id"],recovery=0.,te=te,
                         ie_forward=0.,ie_reverse=0.,forward_normalized=0.,reverse_normalized=0.)
            else:
                contexts=(engine.prepare(_prompt(views,p["left"])),
                          engine.prepare(_prompt(views,p["right"])))
                row=dict(pair_id=p["pair_id"],**_effects(engine,contexts,left,right,heads,te))
                del contexts
            if handle and p["pair_id"] not in completed:
                _append(handle,row)
            rows.append(row)
    finally:
        if handle:handle.close()
    return dict(heads=heads,before=len(pairs),after=len(rows),
                recovery=sum(r["recovery"] for r in rows)/len(rows) if rows else None,
                pairs=rows,epsilon=epsilon)
