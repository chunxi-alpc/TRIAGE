"""Reuse the 200 historical full-tree pairs; preserve source-level splits."""
import copy, csv, json, random, re
from collections import Counter,defaultdict
from pathlib import Path
from .common import *
from experiments.aster_representation.tree import tree_guideline
from experiments.aster_representation.inputs import swap_actions
from experiments.gemma3_head_pilot.fourway_data import common_history
HISTORY=Path("/home/gcx/gemma3_27b_fourway_expanded_20261104/data")
DEPENDENCY=Path("/home/gcx/semantic_masking_followup_20261105/data")
MEGA=Path("/home/gcx/TRIAGE/data/MEGA-CDP/train_val")
DEV={"train_val:2308","train_val:2544","train_val:2788","train_val:688"}
def condition(group,gid=None):
    g=next(g for g in group["guideline"]["groups"] if g["group_id"]==(gid or group["gid"]))
    return re.sub(r"（判断为[是否]）$","",g["rules"][0]["condition"])
def ancestors(group):
    out=[]
    for rid in common_history(group["guideline"],group["gid"]):
        gid="G"+rid[1:-1]
        out.append(dict(gid=gid,condition=condition(group,gid),label=int(rid.endswith("Y"))))
    return out
def canonical_guide(guide):
    """Full conditions, transitions and terminal payloads; no gold path."""
    groups=[];leaves={}
    for g in guide["groups"]:
        q=re.sub(r"（判断为[是否]）$","",g["rules"][0]["condition"])
        node=dict(id=g["group_id"],condition=q)
        for r in g["rules"]:
            m=re.match(r"(L\d+)[：:](.*)",r["action"],re.S)
            if m: target=m[1];leaves[target]=m[2]
            else:
                m=re.search(r"G\d+",r["action"])
                if not m: raise ValueError("Unrecognized transition: "+r["action"])
                target=m[0]
            node["yes" if r["rule_id"].endswith("Y") else "no"]=target
        groups.append(node)
    return dict(nodes=groups,leaves=leaves)
def render_guide(guide,definitions,mode="full",semantic=None,protect=()):
    g=canonical_guide(guide);semantic=semantic or {}
    lines=[f"{n['id']}: {n['condition']}\n  是 -> {n['yes']}；否 -> {n['no']}" for n in g["nodes"]]
    for key,text in g["leaves"].items():
        keep=mode=="full" or mode=="boundary_guideline" and key in protect or mode=="semantic" and key in semantic.get("keep_action_ids",[])
        lines.append(key+": "+(text if keep else "[处置已遮挡]"))
    defs=[]
    for i,d in enumerate(definitions):
        keep=mode in ("full","boundary_guideline","definition_restore") or mode=="semantic" and i in semantic.get("keep_definition_indices",[])
        if keep:defs.append(d["quote"])
    return "\n".join(lines)+"\n\n指南原文定义与说明：\n"+"\n".join(defs)
def prompt(guide,definitions,patient,gid,mode="full",semantic=None,protect=()):
    context=render_guide(guide,definitions,mode,semantic,protect)
    q=next(n["condition"] for n in canonical_guide(guide)["nodes"] if n["id"]==gid)
    return f"完整决策树：\n{context}\n\n病历：\n{patient}\n\n当前节点：{gid}\n当前条件：{q}\n请判断当前条件。只输出“是”或“否”。\n答案："
def replace_all(obj,edits):
    if isinstance(obj,str):
        for e in edits: obj=obj.replace(e["old"],e["new"])
        return obj
    if isinstance(obj,list):return [replace_all(v,edits) for v in obj]
    if isinstance(obj,dict):
        out={}
        for k,v in obj.items():
            if k=="action":
                m=re.match(r"(L\d+[\uff1a:])(.*)",v,re.S)
                out[k]=m[1]+replace_all(m[2],edits) if m else v
            else:
                out[k]=replace_all(v,edits) if k in ("condition","ancestor_conditions","quote") or isinstance(v,(dict,list)) else v
        return out
    return obj
def set_condition(guide,gid,text):
    guide=copy.deepcopy(guide)
    old=condition({"guideline":guide,"gid":gid})
    return replace_all(guide,[dict(old=old,new=text)])
def source_definitions():
    nodes={n["node_id"]:n for n in rows(DEPENDENCY/"nodes.jsonl")}
    ann={a["node_id"]:a["annotation"] for a in rows(DEPENDENCY/"annotations.jsonl")}
    definitions=defaultdict(list)
    for nid,a in ann.items():
        n=nodes[nid]
        if a.get("source_semantic_dependency") is True:
            for d in a.get("missing_definitions",[]):
                if d["quote"] not in [x["quote"] for x in definitions[n["source_id"]]]:
                    definitions[n["source_id"]].append(dict(d,source_file=n["source_file"],annotation_node=nid))
    return nodes,ann,definitions
def prepare(root=DEFAULT_ROOT):
    root=Path(root);(root/"data").mkdir(parents=True,exist_ok=True)
    nodes,anns,defs=source_definitions()
    main=[]
    for old in ("development","confirmation"):
        for g in rows(HISTORY/(old+".jsonl")):
            split="test" if old=="confirmation" else "dev" if g["source_id"] in DEV else "fit"
            g=dict(g,split=split,cohort="main",original_split=old,definitions=defs.get(g["source_filename"],[]))
            g["ancestors"]=ancestors(g);main.append(g)
    assert Counter(g["split"] for g in main)=={"fit":80,"dev":20,"test":100}
    source_splits={g["source_filename"]:g["split"] for g in main}
    raw={r["filename"]:r for r in rows(MEGA/"decision_trees.jsonl")}
    # Choose targets from existing dependency records without reference to scores.
    candidates=[]
    for nid,a in anns.items():
        if a.get("source_semantic_dependency") is not True or not a.get("missing_definitions"):continue
        n=nodes[nid];r=raw.get(n["source_id"])
        if not r:continue
        guide,mapping=tree_guideline(json.loads(r["text"])["decision_tree"])
        m=mapping.get(str(n["index"]))
        if not m or m["leaf"]:continue
        candidates.append((n,a,r,guide,m))
    rng=random.Random(SEED);rng.shuffle(candidates)
    desired={"fit":8,"dev":2,"test":20};chosen=[];counts=Counter()
    # Overlap with the main experiment keeps the assigned source split.
    for n,a,r,guide,m in candidates:
        s=source_splits.get(n["source_id"])
        if s and counts[s]<desired[s]:
            chosen.append((s,n,a,r,guide,m));counts[s]+=1
    used={n["node_id"] for _,n,*_ in chosen}
    extra_splits={}
    for n,a,r,guide,m in candidates:
        if n["node_id"] in used or n["source_id"] in source_splits:continue
        s=extra_splits.get(n["source_id"])
        if s is None:
            available=[s for s in ("test","fit","dev") if counts[s]<desired[s]]
            if not available:break
            s=available[0];extra_splits[n["source_id"]]=s
        if counts[s]>=desired[s]:continue
        chosen.append((s,n,a,r,guide,m));counts[s]+=1;used.add(n["node_id"])
    definition=[]
    for s,n,a,r,guide,m in chosen:
        g=dict(group_id="definition-"+n["node_id"],source_id="train_val:"+str(r["guideline_id"]),
            source_filename=n["source_id"],guideline_id=r["guideline_id"],language=n["language"],
            gid="G"+str(m["node"]),guideline=guide,split=s,cohort="definition",
            definitions=[dict(x,source_file=n["source_file"]) for x in a["missing_definitions"]],
            original_condition=n["condition"],source_rule_evidence=a.get("rule_evidence",[]),
            definition_links=a.get("definition_links",[]),existing_patient=n.get("original_case",""),
            existing_label=n.get("original_gold"),shared_record="",facts=[],
            annotation_origin=str(DEPENDENCY/"annotations.jsonl"))
        g["ancestors"]=ancestors(g);definition.append(g)
    write_rows(root/"data/groups_main.jsonl",main);write_rows(root/"data/groups_definition.jsonl",definition)
    # Path records: five existing cases from each of exactly the 20 test sources.
    ids={g["guideline_id"]:g for g in main if g["split"]=="test"}
    path_items={r["guideline_id"]:r for r in rows(MEGA/"case_paths.jsonl") if r["guideline_id"] in ids}
    pool=defaultdict(list)
    for case in rows(MEGA/"vignette_cases.jsonl"):
        if case["guideline_id"] in ids:pool[case["guideline_id"]].append(case)
    paths=[]
    for sid,g in sorted(ids.items()):
        options=sorted(pool[sid],key=lambda x:int(x["path_id"]));rng.shuffle(options)
        for c in options[:5]:
            raw_path=path_items[sid]["paths"][int(c["path_id"])-1]
            # Convert original node indices to fixed tree IDs.
            raw_tree=json.loads(raw[g["source_filename"]]["text"])["decision_tree"]
            _,mapping=tree_guideline(raw_tree)
            expected=[dict(gid="G"+str(mapping[str(n["index"])]["node"]),label=int(n["decision_outcome"]=="yes"))
                      for n in raw_path if "decision_outcome" in n]
            leaf="L"+str(mapping[str(raw_path[-1]["index"])]["node"])
            paths.append(dict(case_id=f"path-{sid}-{c['path_id']}",source_id=g["source_id"],patient=c["text"],
                              guideline=g["guideline"],definitions=g["definitions"],gold_path=expected,gold_leaf=leaf))
    write_rows(root/"data/path_cases.jsonl",paths)
    manifest=dict(seed=SEED,automatic_pilot=True,clinical_verified=False,main_nodes=dict(Counter(g["split"] for g in main)),
        main_sources={s:len({g["source_id"] for g in main if g["split"]==s}) for s in ("fit","dev","test")},
        dev_sources=sorted(DEV),definition_targets=dict(counts),path_cases=len(paths),
        history="Historical development and confirmation sources have prior use; exploratory validation.",
        score_reuse="No old Y/N path scores reused; current full prompts scored with fixed 是/否 template.",
        source_files=[str(HISTORY/"development.jsonl"),str(HISTORY/"confirmation.jsonl"),str(DEPENDENCY/"annotations.jsonl")])
    write_json(root/"data/manifest.json",manifest)
    return main+definition
def protected_swap(g,protected):
    """Historical rotation among actions that are not part of C."""
    guide=copy.deepcopy(g["guideline"]);terminals=[]
    for node in guide["groups"]:
        for r in node["rules"]:
            m=re.fullmatch(r"(L\d+)([：:])(.*)",r["action"],re.S)
            if m and m[1] not in protected:terminals.append((r,m[1]+m[2],m[3]))
    if len(terminals)>1:
        for i,(r,prefix,_) in enumerate(terminals):r["action"]=prefix+terminals[(i+1)%len(terminals)][2]
    return guide, len(terminals)>1 and len({x[2] for x in terminals})>1
def build(root=DEFAULT_ROOT):
    root=Path(root);groups=rows(root/"data/groups_main.jsonl")+rows(root/"data/groups_definition.jsonl")
    constructions={r["group_id"]:r for r in rows(root/"data/constructions.jsonl")}
    masks={r["key"]:r for r in rows(root/"data/semantic_masks.jsonl")}
    views=[];pairs=[];review=[];omissions=[]
    for ordinal,g in enumerate(groups):
        rec=constructions.get(g["group_id"],{});a=rec.get("value",{})
        isdef=g["cohort"]=="definition"
        if isdef and (rec.get("status")!="complete" or not a.get("definition_valid")):
            omissions.append(dict(group_id=g["group_id"],kind="definition",reason=rec.get("error",a.get("reason","construction unavailable"))));continue
        guide=g["guideline"]
        if isdef:
            guide=set_condition(guide,g["gid"],a["short_condition"])
            shared=a.get("shared_record","");facts=a["fact_sentences"]
        else:shared=g["shared_record"];facts=g["facts"]
        base_patients=[shared+"\n"+f for f in facts]
        definitions=g["definitions"]
        protected=a.get("protected_action_ids",[])
        local=dict(g,guideline=guide)
        swapped,swap_ok=protected_swap(local,protected)
        def add(variant,state,patient=None,guide_=None,defs_=None,gold=None,original=False,semantic_key="base",edits=()):
            id_=g["group_id"]+":"+variant
            guide_=guide_ or guide;defs_=definitions if defs_ is None else defs_
            patient=base_patients[state] if patient is None else patient
            mask=masks.get(g["group_id"]+":"+semantic_key,{}).get("value",{})
            factual=a.get("decisive_sentences",[None,None])
            fact_patient=factual[state] if len(factual)>state and factual[state] else facts[state]+"\n"+shared
            # When the patient includes a P change, annotation boundary upper bound removes it.
            kw=dict(guide=guide_,definitions=defs_,gid=g["gid"],semantic=mask,protect=protected)
            ps={m:prompt(patient=patient,mode=m,**kw) for m in ("full","direct","semantic","boundary_guideline")}
            ps["boundary_facts"]=prompt(patient=fact_patient,mode="boundary_guideline",**kw)
            ps["ignore"]="请忽略不属于当前条件定义的后续处置建议，仅根据条件定义和患者事实判断。\n"+ps["full"]
            if isdef:ps["definition_restore"]=prompt(patient=patient,mode="definition_restore",**kw)
            v=dict(view_id=id_,group_id=g["group_id"],source_id=g["source_id"],split=g["split"],cohort=g["cohort"],
                variant=variant,label=int(state if gold is None else gold),original=original,
                prompt=ps["full"],prompts=ps,patient=patient,gid=g["gid"],condition=condition(dict(g,guideline=guide_)),
                edit_locations=list(edits),annotation_origin="historical pairs + MedGemma automatic construction; physician verification pending",
                semantic_available=masks.get(g["group_id"]+":"+semantic_key,{}).get("status")=="complete",
                boundary_annotation_available=rec.get("status")=="complete",
                decisive_annotation_available=bool(len(factual)>state and factual[state]))
            views.append(v);return id_
        def pair(kind,left,right,subtype="",edits=()):
            p=dict(pair_id=g["group_id"]+":"+kind+":"+str(sum(p["group_id"]==g["group_id"] and p["kind"]==kind for p in pairs)),
                   group_id=g["group_id"],source_id=g["source_id"],split=g["split"],cohort=g["cohort"],
                   kind=kind,subtype=subtype,left=left,right=right,edit_locations=list(edits),
                   annotation_origin=str(HISTORY) if kind in ("A","F") and not isdef else "MedGemma automatic; complete predicate evaluated")
            pairs.append(p)
        base=[add("base"+str(s),s,original=True) for s in (0,1)]
        swap=[add("swap"+str(s),s,guide_=swapped,semantic_key="swap") for s in (0,1)]
        pair("F",base[0],base[1])
        if swap_ok:
            for s in (0,1):pair("A",base[s],swap[s])
            pair("F",swap[0],swap[1])
        else:omissions.append(dict(group_id=g["group_id"],kind="A",reason="Fewer than two distinct independent disposition payloads"))
        c=a.get("condition_change")
        if c and c.get("valid"):
            ed=c["edits"];s=int(c["base_state"])
            cg=replace_all(guide,ed);cd=replace_all(definitions,ed)
            label=int(evaluate(c["predicate"],a["facts"][s]))
            cid=add("condition",s,guide_=cg,defs_=cd,gold=label,semantic_key="condition",edits=ed)
            pair("C",base[s],cid,edits=ed)
        else:omissions.append(dict(group_id=g["group_id"],kind="C",reason=(c or {}).get("reason","No valid constructed change")))
        for subtype,key in (("clinical","clinical_background"),("preference","treatment_preference")):
            change=a.get(key)
            if not change or not change.get("valid"):
                omissions.append(dict(group_id=g["group_id"],kind="P",subtype=subtype,reason=(change or {}).get("reason","No valid background pair")));continue
            s=ordinal%2
            ids=[add("P_"+subtype+str(j),s,patient=base_patients[s]+"\n"+t) for j,t in enumerate(change["texts"])]
            pair("P",*ids,subtype=subtype,edits=change["texts"])
        review.append(dict(group_id=g["group_id"],source_id=g["source_id"],split=g["split"],cohort=g["cohort"],
            current_condition=condition(dict(g,guideline=guide)),ancestors=g["ancestors"],definitions=definitions,
            original_patients=base_patients,construction=a,validation=rec.get("validation"),clinical_review_status="pending",
            reviewer_truth_correct=None,reviewer_ancestor_consistent=None,reviewer_C_change_valid=None,
            reviewer_background_irrelevant=None,reviewer_action_definitions_preserved=None,reviewer_notes=""))
    view_by_id={v["view_id"]:v for v in views}
    for p in pairs:
        p["left_label"]=view_by_id[p["left"]]["label"];p["right_label"]=view_by_id[p["right"]]["label"]
        p["truth_relation"]="preserve" if p["kind"] in ("A","P") else "flip"
    write_rows(root/"data/views.jsonl",views);write_rows(root/"data/pairs.jsonl",pairs)
    write_rows(root/"data/omissions.jsonl",omissions);write_rows(root/"review/physician_review.jsonl",review)
    with (root/"review/physician_review.csv").open("w",encoding="utf-8-sig",newline="") as f:
        columns=list(review[0]) if review else ["group_id"]
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in review)
    count={s:{c:dict(Counter(p["kind"]+(":"+p["subtype"] if p["kind"]=="P" else "") for p in pairs if p["split"]==s and p["cohort"]==c))
              for c in ("main","definition")} for s in ("fit","dev","test")}
    write_json(root/"data/valid_counts.json",dict(views=len(views),pairs=len(pairs),by_split=count,omissions=len(omissions)))
    return views,pairs
