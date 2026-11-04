"""Source-separated 100-group development and confirmation data for saved Gemma checkpoints."""
import json
import re
from collections import Counter
from pathlib import Path
from .core import rows, write_json
from .fourway_data import common_history
from .expanded_specs import SOURCE_SPECS

DEFAULT_ROOT=Path("/home/gcx/gemma3_27b_fourway_expanded_20261104")
PILOT_ROOT=Path("/home/gcx/gemma3_27b_head_pilot_20261103")
DATASET=Path("/home/gcx/TRIAGE/data/MEGA-CDP")

def guide_text_key(guide):
    # Compare the provided guideline text directly; no checksum or file hashing.
    return re.sub(r"\s+","",json.dumps(guide,sort_keys=True,ensure_ascii=False))

def validate(groups, expected=100):
    if len(groups)!=expected or len({g["group_id"] for g in groups})!=expected:
        raise ValueError("Each split requires 100 unique groups")
    counts=Counter(g["source_id"] for g in groups)
    if len(counts)<20:raise ValueError("Each split needs at least 20 guideline sources")
    if len({g["source_filename"] for g in groups})!=len(counts):
        raise ValueError("Source metadata and guideline source count disagree")
    for g in groups:
        if g["history"]!=common_history(g["guideline"],g["gid"]):
            raise ValueError("Invalid common path prefix")
        if len(g["facts"])!=2 or g["facts"][0]==g["facts"][1]:
            raise ValueError("Expected two distinct fact states")
        if g["fact_type"]=="" or not g["shared_record"]:
            raise ValueError("Missing group construction information")
    return counts

def prepare(root=DEFAULT_ROOT,pilot=PILOT_ROOT):
    root,pilot=Path(root),Path(pilot)
    root.mkdir(parents=True,exist_ok=True)
    data=root/"data";data.mkdir(exist_ok=True)
    outputs={s:data/(s+".jsonl") for s in ("development","confirmation")}
    if all(p.exists() for p in outputs.values()):
        out={s:rows(p) for s,p in outputs.items()}
        for g in out.values():validate(g)
        if {g["source_id"] for g in out["development"]}&{g["source_id"] for g in out["confirmation"]}:
            raise ValueError("Development and confirmation source overlap")
        return out
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Partial data preparation exists; preserve it before retrying")
    from experiments.aster_representation.tree import tree_guideline
    raw={int(x["guideline_id"]):x for x in rows(DATASET/"train_val/decision_trees.jsonl")}
    def longest(node):
        if not all(isinstance(node.get(b),dict) for b in ("yes","no")):return []
        return max(([(node,b)]+longest(node[b]) for b in ("yes","no")),key=len)
    by_id={}
    for gid,_,_ in SOURCE_SPECS:
        item=raw[gid];tree=json.loads(item["text"])["decision_tree"]
        guide,mapping=tree_guideline(tree);path=longest(tree)[:5]
        by_id[gid]=dict(guideline_id=gid,source_filename=item["filename"],language=item["language"],
                       guideline=guide,nodes=[dict(gid="G"+str(mapping[str(n["index"])]["node"]),
                       condition=n["description"],continue_branch="Y" if b=="yes" else "N") for n,b in path])
    prior=[c for split in ("train","development","confirmation") for c in rows(pilot/"data"/(split+".jsonl"))]
    training=rows(pilot/"data/train.jsonl")
    prior_keys={guide_text_key(c["guideline"]) for c in prior}
    train_keys={guide_text_key(c["guideline"]) for c in training}
    prior_ids={int(c["case_id"].split(":")[2]) for c in prior}
    prior_files={raw[i]["filename"] for i in prior_ids}
    training_files={raw[int(c["case_id"].split(":")[2])]["filename"] for c in training}
    if len(SOURCE_SPECS)!=40:raise ValueError("Expected 40 pre-specified sources")
    result={"development":[],"confirmation":[]}
    source_records={}
    used_keys=set()
    for ordinal,(gid,base,specs) in enumerate(SOURCE_SPECS):
        selected=by_id[gid]
        key=guide_text_key(selected["guideline"])
        if selected["source_filename"] in prior_files or key in prior_keys:
            raise ValueError("Selected guide overlaps a prior training/selection/evaluation guide: "+str(gid))
        if key in used_keys:raise ValueError("Repeated guideline text across sources")
        used_keys.add(key)
        split="development" if ordinal<20 else "confirmation"
        if len(specs)!=5 or len(selected["nodes"])!=5:
            raise ValueError("Each source requires five distinct nodes")
        prefix=[base]
        for index,(node,spec) in enumerate(zip(selected["nodes"],specs)):
            shared="\n".join(x for x in prefix+[spec["shared"]] if x)
            group_id=f"expanded-{split}-{gid}-{node['gid']}"
            result[split].append(dict(group_id=group_id,source_id="train_val:"+str(gid),
                source_filename=selected["source_filename"],guideline_id=gid,language=selected["language"],
                gid=node["gid"],history=common_history(selected["guideline"],node["gid"]),
                node_depth=len(common_history(selected["guideline"],node["gid"])),
                shared_record=shared,facts=[spec["negative"],spec["positive"]],
                fact_type=spec["fact_type"],guideline=selected["guideline"],
                question="请依据完整指南给出患者的临床决策路径及最终处置。最后输出 PATH: 经过的规则编号；FINAL: 终点处置的L编号。"))
            chosen=spec["positive"] if node["continue_branch"]=="Y" else spec["negative"]
            carry=spec["carry"] if spec["carry"] is not None else "\n".join(x for x in [spec["shared"],chosen] if x)
            prefix.append(carry)
        source_records["train_val:"+str(gid)]=dict(split=split,guideline_id=gid,
            filename=selected["source_filename"],language=selected["language"],
            target_nodes=[n["gid"] for n in selected["nodes"]])
    for groups in result.values():validate(groups)
    dev_sources={g["source_id"] for g in result["development"]}
    confirm_sources={g["source_id"] for g in result["confirmation"]}
    if dev_sources&confirm_sources:raise ValueError("Split overlap")
    dev_keys={guide_text_key(g["guideline"]) for g in result["development"]}
    confirm_keys={guide_text_key(g["guideline"]) for g in result["confirmation"]}
    if dev_keys&confirm_keys or (dev_keys|confirm_keys)&train_keys:
        raise ValueError("Training/development/confirmation guideline text overlaps")
    manifest=dict(groups_per_split=100,views_per_group=4,
        sources=source_records,
        splits={split:dict(groups=len(groups),sources=len({g["source_id"] for g in groups}),
                    input_cells=len(groups)*4,
                    fact_types=dict(Counter(g["fact_type"] for g in groups)),
                    languages=dict(Counter(g["language"] for g in groups)),
                    node_depths=dict(Counter(str(g["node_depth"]) for g in groups))) for split,groups in result.items()},
        source_identity="Provided dataset filename and source guideline ID; equal complete guideline texts are kept together. No checksums are computed.",
        source_separation=dict(train_development=True,train_confirmation=True,development_confirmation=True,
                               excluded_prior_source_files=len(prior_files),training_source_files=len(training_files)),
        construction="Literal constructed clinical/workflow records; each pair replaces one specified fact field. No original patient narrative is copied.",
        nodes="Five different consecutive nodes on a valid path per source; depths 0 to 4. Different groups are different constructed patients.",
        prefix="Prior facts establish the common prefix. Explicit carry evidence may be less specific than an earlier group's chosen example, so it does not fix a later target.",
        labels="State0 -> N; state1 -> Y. Labels stay in the evaluator, not the model prompt.",
        selection="Source/node/fact specifications fixed before expanded model predictions.",
        confirmation_policy="No model prediction on confirmation data before a frozen configuration file has been written.")
    for split,p in outputs.items():
        p.write_text("".join(json.dumps(g,ensure_ascii=False)+"\n" for g in result[split]))
    write_json(root/"data_manifest.json",manifest)
    return result
