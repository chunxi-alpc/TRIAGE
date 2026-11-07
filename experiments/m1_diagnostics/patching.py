"""Fixed fit/development diagnostics. Every checkpoint retains exact intervention IDs."""
import time
from .common import *
from .engine import Engine
from .statistics import summarize
from .data import alignment

def plans(model,kind):
    f=frozen()[model];cand=f["candidates"][kind];n=64 if model=="qwen" else 110
    sets={"type_selected":f["H_"+kind],"selected_union":f["H"],"candidate128":cand,
          "candidate_fraction":cand[:n]}
    for layer in range(f["total_heads"]//32):sets[f"layer_{layer:02d}"]=list(range(layer*32,(layer+1)*32))
    for h in f["H_"+kind]:sets[f"single_{h}"]=[h]
    return sets

def metadata(p,model):
    return dict(model=model,precision=PRECISION,**{k:p[k] for k in ["pair_id","group_id","source_id","split","kind"]})

def patch_values(e,contexts,left,right,hs,fixed=False):
    zl,zr=[x["heads"].reshape(-1,e.head_dim) for x in [left,right]]
    a=e.run(contexts[0],replacements={h:zr[h] for h in hs},capture=False,
            norm_scales=left.get("rms_scales") if fixed else None)["ell"]
    b=e.run(contexts[1],replacements={h:zl[h] for h in hs},capture=False,
            norm_scales=right.get("rms_scales") if fixed else None)["ell"]
    return effects(left["ell"],right["ell"],a,b)

def run_pair(e,p,model):
    dest=cache_path(model,"pairs",p["pair_id"])
    if dest.exists() and read(dest).get("state")=="complete":return read(dest)
    views={r["view_id"]:r for r in rows(ROOT/"data/diagnostic_views.jsonl")}
    contexts=[e.prepare(views[p[s]]["prompt"]) for s in ["left","right"]]
    left,right=[e.run(c,capture_scales=model=="medgemma") for c in contexts]
    meta=metadata(p,model);allheads=list(range(e.num_heads))
    result=read(dest) if dest.exists() else dict(meta,state="running",effects={})
    if "checks" not in result:
        checks={}
        repeated=[e.run(c,capture=False)["ell"] for c in contexts]
        self_scores=[e.run(c,replacements={h:x["heads"].reshape(-1,e.head_dim)[h] for h in allheads},capture=False)["ell"]
                     for c,x in zip(contexts,[left,right])]
        all_effect=patch_values(e,contexts,left,right,allheads)
        checks.update(replay_errors=[repeated[i]-x["ell"] for i,x in enumerate([left,right])],
          self_errors=[self_scores[i]-x["ell"] for i,x in enumerate([left,right])],
          all_heads=all_effect,all_head_donor_errors=[all_effect["forward_ell"]-right["ell"],all_effect["reverse_ell"]-left["ell"]])
        if max(abs(x) for x in checks["replay_errors"]+checks["self_errors"])>1e-5:
            raise RuntimeError("No-intervention/self-patching replay differs; check cache and hooks.")
        if max(abs(x) for x in checks["all_head_donor_errors"])>.1:
            write(dest,dict(result,checks=checks,state="basic_check_failed"))
            raise RuntimeError("All-head replacement does not reproduce donor scores; inspect positions/cache/hooks before interpretation.")
        if model=="medgemma":
            zero=[e.run(c,capture=False,norm_scales=x["rms_scales"])["ell"]-x["ell"] for c,x in zip(contexts,[left,right])]
            checks["fixed_scale_no_intervention_errors"]=zero
            if max(abs(x) for x in zero)>1e-5:raise RuntimeError("Fixed RMS scaling does not reproduce no intervention.")
        checks["full_forward_errors"]=[e.full_score(views[p[s]]["prompt"])-x["ell"] for s,x in zip(["left","right"],[left,right])]
        result["checks"]=checks;result["tokens"]=[c.length for c in contexts];write(dest,result)
    sets=plans(model,p["kind"])
    schedules=[("native",sets)]
    if model=="medgemma":
        schedules.append(("fixed_rms",{k:v for k,v in sets.items() if k in ["type_selected","selected_union"] or k.startswith("single_")}))
    for mode,msets in schedules:
        for name,hs in msets.items():
            key=mode+"/"+name
            if key in result["effects"]:continue
            # Reuse only precisely identical sets within this pair and this diagnostic mode.
            same=next((r for r in result["effects"].values() if r["mode"]==mode and r["heads"]==sorted(hs)),None)
            val={k:same[k] for k in effects(0,0,0,0)} if same else patch_values(e,contexts,left,right,hs,mode=="fixed_rms")
            result["effects"][key]=dict(meta,mode=mode,set=name,heads=sorted(hs),**val)
            write(dest,result)
    if model=="qwen":
        result["normalization_control"]="Qwen未添加处理；同名post_attention_layernorm为MLP前归一化，未修改。"
    result["state"]="complete";write(dest,result)
    return result

def select_top_layers(model):
    dest=ROOT/"patch"/model/"top_layers.json"
    if dest.exists():return read(dest)
    ps=[p for p in rows(ROOT/"data/diagnostic_pairs.jsonl") if p["split"]=="fit"]
    result={}
    for kind in ["A","P"]:
        vals=defaultdict(list)
        for p in ps:
            if p["kind"]!=kind:continue
            d=read(cache_path(model,"pairs",p["pair_id"]))
            for r in d["effects"].values():
                if r["mode"]=="native" and r["set"].startswith("layer_"):vals[int(r["set"][6:])].append(r)
        # Effect-weighted restoration is predeclared for choosing only three layer diagnostics.
        ranking=[]
        for layer,rs in vals.items():
            val=summarize(rs,bootstrap=False)["estimates"]["weighted_all"]
            ranking.append(dict(layer=layer,effect=val))
        ranking.sort(key=lambda r:(-(r["effect"] if r["effect"] is not None else -float("inf")),r["layer"]))
        result[kind]=dict(layers=[r["layer"] for r in ranking[:3]],ranking=ranking,criterion="fit effect-weighted restoration")
    write(dest,result);return result

def layer_complements(e,p,model):
    dest=cache_path(model,"complements",p["pair_id"])
    if dest.exists() and read(dest).get("state")=="complete":return
    top=select_top_layers(model)[p["kind"]]["layers"];cand=set(frozen()[model]["candidates"][p["kind"]])
    views={r["view_id"]:r for r in rows(ROOT/"data/diagnostic_views.jsonl")}
    contexts=[e.prepare(views[p[s]]["prompt"]) for s in ["left","right"]];left,right=[e.run(c) for c in contexts]
    result=read(dest) if dest.exists() else dict(metadata(p,model),state="running",effects={})
    for layer in top:
        allhs=set(range(layer*32,(layer+1)*32))
        for name,hs in [("inside",sorted(allhs&cand)),("outside",sorted(allhs-cand))]:
            key=f"{layer}_{name}"
            if key in result["effects"]:continue
            val=patch_values(e,contexts,left,right,hs) if hs else effects(left["ell"],right["ell"],left["ell"],right["ell"])
            result["effects"][key]=dict(metadata(p,model),layer=layer,part=name,heads=hs,**val);write(dest,result)
    result["state"]="complete";write(dest,result)

def residual_scan(e,p,model):
    dest=cache_path(model,"residual",p["pair_id"])
    if dest.exists() and read(dest).get("state")=="complete":return
    views={r["view_id"]:r for r in rows(ROOT/"data/diagnostic_views.jsonl")}
    prompts=[views[p[s]]["prompt"] for s in ["left","right"]]
    ids=[e.ids(prompt)[0].cpu().tolist() for prompt in prompts]
    align=alignment(*ids)
    assert align==p["alignment"][model],"Residual anchors differ from frozen token positions"
    positions=[{name:pos[i] for name,pos in align["positions"].items() if pos is not None} for i in range(2)]
    layers=[4,5,28,29,58,59] if model=="medgemma" else [2,3,16,17,33,34]
    left,right=[e.residual_run(prompt,pos,layers) for prompt,pos in zip(prompts,positions)]
    result=read(dest) if dest.exists() else dict(metadata(p,model),state="running",alignment=align,effects={})
    if "self_error" not in result:
        self_run=e.residual_run(prompts[0],positions[0],layers,(layers[0],"judgment",left["residuals"][layers[0],"judgment"]))
        result["self_error"]=self_run["ell"]-left["ell"]
        assert abs(result["self_error"])<1e-5
    for layer in layers:
        for name in positions[0]:
            key=f"{layer}_{name}"
            if key in result["effects"]:continue
            f=e.residual_run(prompts[0],positions[0],layers,(layer,name,right["residuals"][layer,name]))["ell"]
            r=e.residual_run(prompts[1],positions[1],layers,(layer,name,left["residuals"][layer,name]))["ell"]
            result["effects"][key]=dict(metadata(p,model),layer=layer,position=name,
               **effects(left["ell"],right["ell"],f,r));write(dest,result)
    result["state"]="complete";write(dest,result)

def run(model,engine=None):
    e=engine or Engine(model);ps=rows(ROOT/"data/diagnostic_pairs.jsonl");start=time.time()
    # Fixed order alternates A/P, independent of scores.
    groups=[[p for p in ps if p["kind"]==k and p["split"]==s] for s in ["fit","dev"] for k in ["A","P"]]
    ps=[groups[j][i] for i in range(max(map(len,groups))) for j in range(4) if i<len(groups[j])]
    count0=dict(e.counts)
    for i,p in enumerate(ps):
        run_pair(e,p,model)
        status("patch_"+model,state="running",phase="heads",done=i+1,total=len(ps),elapsed=time.time()-start,counts=e.counts)
        if i==19 and not (ROOT/"timing"/f"{model}_pilot20.json").exists():
            elapsed=time.time()-start
            write(ROOT/"timing"/f"{model}_pilot20.json",dict(pairs=20,elapsed_seconds=elapsed,
              estimated_head_stage_seconds=elapsed*len(ps)/20,
              estimated_remaining_head_seconds=elapsed*(len(ps)-20)/20,
              final_forwards=e.counts["final_forwards"]-count0["final_forwards"],
              peak_allocated_gib=float(__import__("torch").cuda.max_memory_allocated()/2**30),
              note="首批20对实测；残差与原文评估另计，续跑时此估时包含已缓存配对。"))
            from .report import run as report
            report()
    select_top_layers(model)
    for i,p in enumerate(ps):
        layer_complements(e,p,model)
        status("patch_"+model,state="running",phase="layer_complements",done=i+1,total=len(ps))
    for i,p in enumerate([x for x in ps if x["residual"]]):
        residual_scan(e,p,model)
        status("patch_"+model,state="running",phase="residual",done=i+1,total=12)
    status("patch_"+model,state="complete",elapsed=time.time()-start,counts=e.counts)
    return e
