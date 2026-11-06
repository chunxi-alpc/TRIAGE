"""H1/H2 interventions and joint-clamp local-theory diagnostics."""
import json
from collections import defaultdict
from pathlib import Path
import time
import torch
from .m1 import EPSILON, _append, _json, _rows, _heads, _effects
from .operators import layer_matched_random

def _types(selection,head):
    value=selection.get("T_h",selection.get("types",{}))
    return value.get(head,value.get(str(head),[]))

def _selection(selection):
    h=selection.get("H",sorted(set(selection.get("H_A",[]))|set(selection.get("H_P",[]))))
    h=sorted(set(int(x) for x in h))
    types={int(x):_types(selection,int(x)) for x in h}
    for kind in ("A","P"):
        for x in selection.get("H_"+kind,[]):
            if kind not in types.setdefault(int(x),[]):
                types[int(x)]=list(types[int(x)])+[kind]
    return h,types

def _fixed_nodes(views,split,limit=20):
    """Fixed source-balanced node IDs, determined from metadata before outcomes."""
    by_source=defaultdict(set)
    for v in views.values():
        if v.get("split")==split and v.get("cohort","main")=="main":
            by_source[v["source_id"]].add(v["group_id"])
    pools={s:sorted(g) for s,g in sorted(by_source.items())}
    chosen=[]
    for depth in range(max((len(x) for x in pools.values()),default=0)):
        for source,pool in pools.items():
            if depth<len(pool):
                chosen.append(pool[depth])
                if len(chosen)>=limit:
                    return chosen
    return chosen

def _representatives(views,nodes,split):
    output={}
    for gid in nodes:
        vs=[(key,v) for key,v in views.items() if v.get("group_id")==gid and v.get("split")==split]
        preferred=[(key,v) for key,v in vs if v.get("original",False)]
        if not preferred:
            preferred=[(key,v) for key,v in vs if v.get("variant") in ("original","base","original_state0","state0")]
        if not preferred:
            raise ValueError("No original view for fixed local-theory node "+gid)
        output[gid]=sorted(preferred,key=lambda kv:kv[0])[0][0]
    return output

def _summarize(rows,keys,value="recovery"):
    grouped=defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(k) for k in keys)].append(r)
    result=[]
    for key,items in sorted(grouped.items(),key=lambda kv:str(kv[0])):
        values=[float(x[value]) for x in items if x.get(value) is not None]
        result.append(dict(zip(keys,key),count=len(items),
                           **{value:sum(values)/len(values) if values else None}))
    return result

def _nuisance_target(z0,z1,operators,heads):
    out={}
    for h in heads:
        op=operators.get(h,operators.get(str(h)))
        if op is None:
            out[h]=z0[h]
            continue
        n=torch.as_tensor(op["N"],dtype=torch.float32)
        b=torch.as_tensor(op["B"],dtype=torch.float32)
        out[h]=z0[h]+n@(b@(z1[h]-z0[h]))
    return out

def _replacement_effects(engine,contexts,left,right,forward_targets,reverse_targets,te):
    forward=engine.run(contexts[0],replacements=forward_targets,capture=False)["ell"]
    reverse=engine.run(contexts[1],replacements=reverse_targets,capture=False)["ell"]
    ie_forward=forward-float(left["ell"]);ie_reverse=reverse-float(right["ell"])
    return dict(forward_ell=forward,reverse_ell=reverse,ie_forward=ie_forward,
                ie_reverse=ie_reverse,te=te,forward_normalized=ie_forward/te,
                reverse_normalized=ie_reverse/(-te),
                recovery=.5*(ie_forward/te+ie_reverse/(-te)))

def run_mechanisms(engine,views,pairs,features,operators,selection,outdir,split="test"):
    """Run planned interventions; only diagnostic normalization uses epsilon.

    Returns aggregate JSON while preserving every pair intervention on disk.
    Main behavior evaluation must independently retain all valid test pairs.
    """
    if split not in ("dev","test"):
        raise ValueError("Mechanism evaluation uses dev or test")
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    started=time.time()
    heads,types=_selection(selection)
    groups=_fixed_nodes(views,split,20)
    representatives=_representatives(views,groups,split)
    fixed=dict(split=split,nodes=groups,original_views=representatives,
               selection="source-balanced sorted node IDs; independent of labels and scores",
               target_node_count=20,actual_node_count=len(groups),epsilon=EPSILON,
               H=heads,T_h=types,automatic_preexperiment=True)
    _json(outdir/f"mechanism_subset_{split}.json",fixed)
    all_pairs=[p for p in pairs if p.get("split")==split and p["kind"] in ("A","P")]
    effective=[p for p in all_pairs if abs(float(features[p["right"]]["ell"])-
                                         float(features[p["left"]]["ell"]))>=EPSILON]
    randoms={}
    for seed in (13,37,73):
        rh,rt=layer_matched_random(heads,types,seed,engine.n_heads)
        randoms[seed]=(rh,rt)
    _json(outdir/f"h1_random_heads_{split}.json",
          {str(seed):dict(H=rh,T_h=rt) for seed,(rh,rt) in randoms.items()})
    # H1: all effective pairs, source/pair identities kept for clustered summaries.
    h1_path=outdir/f"h1_pairs_{split}.jsonl"
    h1={(r["pair_id"],r["control"]):r for r in _rows(h1_path)}
    with h1_path.open("a") as output:
        for i,p in enumerate(effective):
            kind=p["kind"]
            selected=[h for h in heads if kind in types.get(h,[])]
            controls={"type_selected":selected,"selected_union":heads,
                      "all_query_heads":list(range(engine.num_heads))}
            for seed,(rh,rt) in randoms.items():
                controls[f"random_{seed}_type"]=[h for h in rh if kind in rt[h]]
                controls[f"random_{seed}_union"]=rh
            missing=[name for name in controls if (p["pair_id"],name) not in h1]
            if missing:
                left,right=features[p["left"]],features[p["right"]]
                te=float(right["ell"])-float(left["ell"])
                contexts=(engine.prepare(views[p["left"]]["prompt"]),
                          engine.prepare(views[p["right"]]["prompt"]))
                # Reuse equal head sets within this pair, including empty sets.
                computed={}
                for name in missing:
                    hs=controls[name];key=tuple(hs)
                    if key not in computed:
                        if hs:
                            computed[key]=_effects(engine,contexts,left,right,hs,te)
                        else:
                            computed[key]=dict(ie_forward=0.,ie_reverse=0.,te=te,
                                forward_normalized=0.,reverse_normalized=0.,recovery=0.)
                    row=dict(pair_id=p["pair_id"],source_id=p["source_id"],group_id=p["group_id"],
                             cohort=p.get("cohort","main"),kind=kind,subtype=p.get("subtype",""),
                             split=split,control=name,head_count=len(hs),**computed[key])
                    _append(output,row);h1[(p["pair_id"],name)]=row
                del contexts
            _json(outdir/f"mechanism_status_{split}.json",dict(stage="H1",done=i+1,total=len(effective),
                elapsed_seconds=time.time()-started))
    # H2: the fixed node subset, complete relevant-head and joint interventions.
    h2_pairs=[p for p in effective if p["group_id"] in set(groups)]
    h2_path=outdir/f"h2_pairs_{split}.jsonl"
    h2={(r["pair_id"],str(r["head"]),r["operation"]):r for r in _rows(h2_path)}
    with h2_path.open("a") as output:
        for i,p in enumerate(h2_pairs):
            relevant=[h for h in heads if p["kind"] in types.get(h,[])]
            jobs=[(h,"nuisance") for h in relevant]+[(h,"full") for h in relevant]
            jobs += [("joint","nuisance"),("joint","full")]
            jobs=[(h,op) for h,op in jobs if (p["pair_id"],str(h),op) not in h2]
            if jobs:
                left,right=features[p["left"]],features[p["right"]]
                te=float(right["ell"])-float(left["ell"])
                z0,z1=_heads(left),_heads(right)
                contexts=(engine.prepare(views[p["left"]]["prompt"]),
                          engine.prepare(views[p["right"]]["prompt"]))
                for h,operation in jobs:
                    selected=heads if h=="joint" else [h]
                    if operation=="full":
                        effect=_effects(engine,contexts,left,right,selected,te) if selected else dict(
                            ie_forward=0.,ie_reverse=0.,te=te,forward_normalized=0.,
                            reverse_normalized=0.,recovery=0.)
                    else:
                        effect=_replacement_effects(engine,contexts,left,right,
                            _nuisance_target(z0,z1,operators,selected),
                            _nuisance_target(z1,z0,operators,selected),te)
                    row=dict(pair_id=p["pair_id"],source_id=p["source_id"],group_id=p["group_id"],
                             cohort=p.get("cohort","main"),kind=p["kind"],subtype=p.get("subtype",""),
                             split=split,head=h,head_count=len(selected),operation=operation,**effect)
                    _append(output,row);h2[(p["pair_id"],str(h),operation)]=row
                del contexts
            _json(outdir/f"mechanism_status_{split}.json",dict(stage="H2",done=i+1,total=len(h2_pairs),
                elapsed_seconds=time.time()-started))
    # Local theory: gradients under simultaneous independent head clamps.
    local_path=outdir/f"local_theory_{split}.jsonl"
    local={r["view_id"]:r for r in _rows(local_path)}
    for gid,view_id in representatives.items():
        if view_id in local:
            continue
        baseline=features[view_id];z=_heads(baseline)
        context=engine.prepare(views[view_id]["prompt"])
        donors={h:z[h] for h in heads}
        clamped=engine.run(context,replacements=donors,clamp_heads=heads,grad=True)
        gradients=torch.as_tensor(clamped["grads"],dtype=torch.float32).reshape_as(z)
        targets={};linear=0.;per_head=[]
        for h in heads:
            op=operators.get(h,operators.get(str(h)))
            if op is None:
                raise ValueError("Missing operator for selected head")
            n=torch.as_tensor(op["N"],dtype=torch.float32)
            b=torch.as_tensor(op["B"],dtype=torch.float32)
            mu=torch.as_tensor(op["mu"],dtype=torch.float32)
            delta=-n@(b@(z[h]-mu))
            targets[h]=z[h]+delta
            contribution=float(gradients[h]@delta)
            linear+=contribution
            per_head.append(dict(head=h,linear_delta=contribution,edit_norm=float(delta.norm())))
        fixed_score=engine.run(context,replacements=targets,capture=False)["ell"]
        actual_score=engine.run(context,operators=operators,capture=False)["ell"]
        raw=float(baseline["ell"])
        row=dict(view_id=view_id,source_id=views[view_id]["source_id"],group_id=gid,
                 cohort=views[view_id].get("cohort","main"),split=split,baseline_ell=raw,
                 jointly_clamped_baseline_ell=clamped["ell"],linear_delta=linear,
                 linear_prediction=raw+linear,fixed_target_ell=fixed_score,
                 sequential_ell=actual_score,nonlinear_error=fixed_score-raw-linear,
                 cross_layer_change=actual_score-fixed_score,
                 total_prediction_error=actual_score-raw-linear,per_head=per_head,
                 gradient_definition="independent simultaneous clamps at original selected-head outputs")
        with local_path.open("a") as output:
            _append(output,row)
        local[view_id]=row
        del context
        _json(outdir/f"mechanism_status_{split}.json",dict(stage="local_theory",done=len(local),
                    total=len(representatives),elapsed_seconds=time.time()-started))
    summary=dict(split=split,all_interference_pairs=len(all_pairs),effective_pairs=len(effective),
        h1=_summarize(list(h1.values()),["cohort","kind","control"]),
        h2_node_count=len(groups),h2_pair_count=len(h2_pairs),
        h2=_summarize(list(h2.values()),["kind","head","operation"]),
        local_theory=dict(nodes=len(local),
            mean_abs_nonlinear_error=sum(abs(x["nonlinear_error"]) for x in local.values())/len(local) if local else None,
            mean_abs_cross_layer_change=sum(abs(x["cross_layer_change"]) for x in local.values())/len(local) if local else None,
            mean_abs_total_prediction_error=sum(abs(x["total_prediction_error"]) for x in local.values())/len(local) if local else None),
        elapsed_seconds=time.time()-started,
        scope="Mechanism subsets only; no effect filter is applied to primary behavioral tables")
    _json(outdir/f"mechanisms_{split}.json",summary)
    _json(outdir/f"mechanism_status_{split}.json",dict(stage="complete",elapsed_seconds=time.time()-started))
    return summary
