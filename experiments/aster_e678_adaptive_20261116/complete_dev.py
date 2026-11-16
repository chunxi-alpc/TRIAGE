"""Author-approved completion of missing categories on the original 20 dev sources.

Counterfactual research pairs, not changes to clinical guidance. Existing pairs
are retained. No training or test source is used to construct new examples.
"""
from pathlib import Path
from collections import Counter
from experiments.aster_cr_retrain_paired_v2_20261111.common import rows, read, write, lines, load_inputs
from experiments.aster_closed_form.data import canonical_guide, set_condition, prompt
from .adaptive_m1 import representatives
from .prepare import SOURCE, OUT

import json
_authored = json.loads(Path(__file__).with_name("dev_edits.json").read_text())
CONDITIONS = _authored["conditions"]
FACT_EDITS = _authored["facts"]
BACKGROUND_EN = _authored["background_en"]
BACKGROUND_ZH = _authored["background_zh"]

def complete():
    root=OUT/"development"
    hist=Path("/home/gcx/aster_cr_repair_v2_run_20261108/data")
    nodes=[r for r in rows(hist/"original_nodes.jsonl") if r["split"]=="dev"]
    old_pairs=[r for r in rows(SOURCE/"data/historical_pairs.jsonl") if r["split"]=="dev" and r["kind"] in ("A","P","C","F")]
    full={r["group_id"]:r for r in rows(SOURCE/"data/historical_views.jsonl")
          if r["split"]=="dev" and r["variant"]=="full"}
    raw={r["group_id"]:r for r in rows(hist/"views.jsonl") if r["split"]=="dev" and r["variant"]=="full"}
    reps=representatives(old_pairs,"dev")
    added=[];inputs=[];evidence=[]
    sources=sorted({r["source_id"] for r in nodes})
    for i,s in enumerate(sources):
        n=min([r for r in nodes if r["source_id"]==s],key=lambda r:r["group_id"])
        sid=s.split(":")[1]
        original=raw[n["group_id"]]["prompt"]
        if prompt(n["guideline"],n["definitions"],n["patient"],n["gid"]) != original:
            raise ValueError("Original prompt differs from canonical rendering: "+s)
        for k in ("P","C","F"):
            if any(p["source_id"]==s for p in reps[k]):continue
            guide=n["guideline"];patient=n["patient"];label=n["original_label"]
            if k=="P":
                addition=(BACKGROUND_EN if n["language"]=="en" else BACKGROUND_ZH)[i%10]
                patient += "\n\n"+addition
                edits=[dict(field="patient",operation="append",new=addition)]
                rationale="Administrative background does not alter the current clinical condition."
            elif k=="C":
                old=next(x["condition"] for x in canonical_guide(guide)["nodes"] if x["id"]==n["gid"])
                guide=set_condition(guide,n["gid"],CONDITIONS[sid])
                edits=[dict(field="condition",old=old,new=CONDITIONS[sid])]
                label=1-label
                rationale="The patient is held fixed; the counterfactual node condition reverses its truth value."
            else:
                edits=[]
                for old,new in FACT_EDITS[sid]:
                    if old not in patient:raise ValueError("Missing authored text: "+s+" "+old)
                    patient=patient.replace(old,new)
                    edits.append(dict(field="patient",old=old,new=new))
                label=1-label
                rationale="The decisive patient fact is changed, including repeated mentions; node condition is held fixed."
            uid="adaptive-dev-"+sid+"-"+k
            text=prompt(guide,n["definitions"],patient,n["gid"])
            p=dict(pair_id=uid,source_id=s,split="dev",kind=k,
                   left=full[n["group_id"]]["view_id"],right=uid,
                   left_label=n["original_label"],right_label=label,
                   case_id=n["case_id"],group_id=n["group_id"],origin="adaptive_dev_completion_20261116")
            added.append(p)
            inputs.append(dict(uid=uid,view_id=uid,source_id=s,split="dev",kind=k,
                               prompt=text,label=label,labels=["\u662f","\u5426"]))
            evidence.append(dict(p,source_document=n["source_document"],node_id=n["gid"],
                                 edits=edits,rationale=rationale,original_patient=n["patient"],
                                 modified_patient=patient,construction="Codex-authored counterfactual pair"))
    combined=old_pairs+added
    selected=representatives(combined,"dev")
    if set(sources)!={p["source_id"] for p in old_pairs} or any(len(selected[k])!=20 for k in "APCF"):
        raise ValueError("Incomplete development-source coverage")
    lines(root/"added_pairs.jsonl",added)
    lines(root/"inputs.jsonl",inputs)
    lines(root/"pair_edits.jsonl",evidence)
    lines(root/"dev_pairs.jsonl",combined)
    lines(root/"representatives.jsonl",[p for k in "APCF" for p in selected[k]])
    manifest=dict(complete=True,added_pairs=len(added),added_by_kind=dict(Counter(p["kind"] for p in added)),
                  selected_pairs=80,sources=20,category_sources={k:20 for k in "APCF"},
                  retained_training_pairs=2540,source_split=[80,20,40],
                  authorization="User confirmed retaining 2540 fit pairs and completing original 20 dev sources",
                  test_sources_used=False,source_documents="Existing development guidelines and patient records",
                  clinical_review=False,selection_per_source="lexicographically smallest pair_id within category/source")
    write(root/"manifest.json",manifest)
    write(OUT/"data_resolution.json",dict(resolved=True,method="complete_original_development_sources",
          manifest=str(root/"manifest.json")))
    return manifest

if __name__=="__main__": complete()
