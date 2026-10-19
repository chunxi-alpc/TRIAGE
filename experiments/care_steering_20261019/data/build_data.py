#!/usr/bin/env python3
"""Reproducible synthetic conditional source-conflict benchmark. No clinical claims."""
from __future__ import annotations
import argparse,collections,hashlib,json,random
from pathlib import Path
SEED=20261019
LETTERS="ABCD"
CONDITIONS=["min_apply","min_not_apply","multi_apply","multi_not_apply","exception","exception_clear","old_invalid_applicable","old_invalid_inapplicable","unknown","no_conflict"]
WORDS=["Alder","Birch","Cedar","Dogwood","Elm","Fir","Ginkgo","Hawthorn","Ironwood","Juniper","Kapok","Larch","Maple","Nutmeg","Oak","Pine","Quince","Redwood","Spruce","Tamarind","Upas","Viburnum","Willow","Xylia","Yew","Zelkova","Ash","Beech","Cypress","Dahlia","Ebony","Fig","Guava","Hazel","Iroko","Jacaranda","Kauri","Laurel","Mango","Neem","Olive","Palm","Rowan","Sycamore"]
def sha(b): return hashlib.sha256(b).hexdigest()
def build(seed=SEED):
    rng=random.Random(seed)
    indexes=list(range(44)); rng.shuffle(indexes)
    assigned={n:("train" if j<12 else "dev" if j<20 else "test" if j<36 else "transfer") for j,n in enumerate(indexes)}
    rows=[]
    for n in range(44):
        split=assigned[n]; task="equipment_routing" if split=="transfer" else "patient_routing"
        noun="device" if task=="equipment_routing" else "patient"
        fam=f"{task}-{WORDS[n].lower()}-{n:02d}"
        threshold=20+n
        new=f"Route {WORDS[n]}"; old=f"Route {WORDS[(n+17)%44]}"; extra=f"Record {WORDS[(n+31)%44]}"
        permutation=list(range(4)); rng.shuffle(permutation)
        for ci,cond in enumerate(CONDITIONS):
            app=cond not in {"min_not_apply","multi_not_apply","exception","old_invalid_inapplicable","unknown"}
            value=None if cond=="unknown" else threshold+(0 if app or cond=="exception" else -1)
            exception=cond=="exception"
            multi=cond.startswith("multi_")
            old_valid=not cond.startswith("old_invalid")
            conflict=cond!="no_conflict"
            current_old=new if not conflict else old
            # All observations are current and explicitly scoped. No hidden label is used by inference.
            measurement=f"The current {noun} age is {value} years." if value is not None else f"The current {noun} age is undocumented."
            patient=measurement+f" The current {noun} has routing authorization. The current {noun} "+("has" if exception else "does not have")+" a suspension flag."
            patient+=f" The current {noun} "+("has" if multi else "does not have")+" an additional recording requirement."
            numeric={"condition_id":"C1","field":"age","text":f"The {noun} age is at least {threshold} years.","numeric":{"field":"age","operator":">=","value":threshold,"unit":"year"}}
            authorization={"condition_id":"C2","field":"authorization","text":f"The current {noun} has routing authorization.","positive":f"The current {noun} has routing authorization.","negative":f"The current {noun} does not have routing authorization."}
            suspended={"condition_id":"E1","field":"suspension","text":f"The current {noun} has a suspension flag.","positive":f"The current {noun} has a suspension flag.","negative":f"The current {noun} does not have a suspension flag."}
            additional={"condition_id":"C3","field":"recording","text":f"The current {noun} has an additional recording requirement.","positive":f"The current {noun} has an additional recording requirement.","negative":f"The current {noun} does not have an additional recording requirement."}
            rules=[
                {"rule_id":"NEW","role":"new","text":f"If age is at least {threshold} years and routing authorization is present, assign {new}, unless a suspension flag is present.","action":new,"conditions":[numeric,authorization],"logic":{"op":"AND","args":["C1","C2"]},"exceptions":[suspended],"exception_logic":"E1"},
                {"rule_id":"OLD","role":"old","text":f"The older rule assigns {current_old}. Its current validity is "+("VALID." if old_valid else "INVALID; it must not be used."),"action":current_old,"valid":old_valid},
                {"rule_id":"ADDITIONAL","role":"independent","text":f"Independently, if an additional recording requirement is present, also perform {extra}.","action":extra,"conditions":[additional],"logic":"C3","exceptions":[]}
            ]
            target=new+(" + "+extra if multi else "")
            fallback=current_old+(" + "+extra if multi else "")
            distractor=new if multi else extra
            candidates=[target,fallback,"Request review",distractor]
            # No-conflict target and fallback coincide; preserve four distinct choices.
            if len(set(candidates))!=4:
                candidates=[target,old,"Request review",extra]
            assert len(set(candidates))==4,(fam,cond,candidates)
            options={LETTERS[j]:candidates[k] for j,k in enumerate(permutation)}
            state="U" if cond=="unknown" else ("T" if app else "F")
            gold_action=target if app else (fallback if old_valid and state=="F" else "Request review")
            gold=next(k for k,v in options.items() if v==gold_action)
            new_label=next(k for k,v in options.items() if v==target)
            old_label=next((k for k,v in options.items() if v==fallback),None)
            question="Which complete action set follows from the current observations and all rules?"
            instruction=("This is a fictional routing simulation for logic research, not medical guidance. "
            "The newer rule overrides the older rule only when all its conditions hold and its exception is absent. "
            "If the newer rule is inapplicable, use the older rule only if it is currently valid. "
            "If applicability is unknown, or neither rule is valid and applicable, choose Request review. "
            "An independent recording rule must be combined with the selected route. "
            "Do not perform a route when Request review is required. Missing facts are unknown.")
            prompt=instruction+"\n\nQuestion: "+question+"\nObservations: "+patient+"\nRules:\n"+ "\n".join(f"{r['rule_id']}: {r['text']}" for r in rules)+"\nOptions:\n"+"\n".join(f"{k}. {v}" for k,v in options.items())+"\nReturn exactly one option letter.\nAnswer:"
            rule_start=prompt.index("NEW: ")
            action_start=prompt.index(new,rule_start)
            pair_kind=("minimal" if cond.startswith("min_") else "multiple" if cond.startswith("multi_") else "exception" if cond in {"exception","exception_clear"} else "invalid_old" if cond.startswith("old_invalid") else cond)
            rows.append({
                "item_id":f"{fam}-{cond}","pair_id":f"{fam}-{pair_kind}","rule_family_id":fam,
                "split":split,"task":task,"condition":cond,"applicable":None if state=="U" else app,
                "applicability_state":state,"old_valid":old_valid,"conflict":conflict,
                "gold":gold,"options":options,"prompt":prompt,"patient":patient,
                "structured_rules":rules,"rule_action_span":[action_start,action_start+len(new)],
                "new_action_label":new_label,"old_action_label":old_label,
                "new_action":new,"gold_action":gold_action,"independent_action":extra,
                "requires_joint_actions":multi,"question":question,
                "provenance":{"synthetic":True,"clinical":False,"source":"new deterministic generator","seed":seed,"conflict_type":"explicit old/new contextual rules; not parametric knowledge"},
                "generation_metadata":{"numeric_threshold":threshold,"numeric_value":value,"exception":exception,"multi":multi,"family_index":n}
            })
    return rows
def validate(rows):
    assert len(rows)==440
    fam_split={}
    for r in rows:
        assert fam_split.setdefault(r["rule_family_id"],r["split"])==r["split"]
        a,b=r["rule_action_span"]; assert r["prompt"][a:b]==r["new_action"]
        assert r["gold"] in r["options"] and r["new_action_label"] in r["options"]
        assert r["prompt"].index(r["patient"])<a
    families=collections.defaultdict(list)
    for r in rows: families[r["rule_family_id"]].append(r)
    for fam,rr in families.items():
        assert len(rr)==10
        d={r["condition"]:r for r in rr}
        for a,b in [("min_apply","min_not_apply"),("multi_apply","multi_not_apply"),("old_invalid_applicable","old_invalid_inapplicable")]:
            x,y=d[a],d[b]
            assert x["options"]==y["options"] and x["structured_rules"]==y["structured_rules"]
            xv=x["generation_metadata"]["numeric_value"];yv=y["generation_metadata"]["numeric_value"]
            assert abs(xv-yv)==1
            assert x["patient"].replace(str(xv),"<VALUE>",1)==y["patient"].replace(str(yv),"<VALUE>",1)
        assert d["exception"]["options"]==d["exception_clear"]["options"]
    return {"items":len(rows),"families":len(families),"split_items":dict(collections.Counter(r["split"] for r in rows)),
            "split_families":dict(collections.Counter(fam_split.values())),
            "pair_tests":"three numeric-only minimal pairs and one exception-only pair per family",
            "family_leakage":False,"gold_counts":dict(collections.Counter(r["gold"] for r in rows))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path(__file__).parent);p.add_argument("--seed",type=int,default=SEED);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    rows=build(a.seed); audit=validate(rows)
    all_bytes="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows).encode()
    (a.out/"cases.jsonl").write_bytes(all_bytes)
    for split in ["train","dev","test","transfer"]:
        (a.out/f"{split}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows if r["split"]==split),encoding="utf-8")
    audit.update({"seed":a.seed,"sha256":sha(all_bytes),"generator_sha256":sha(Path(__file__).read_bytes()),
    "source":"Synthetic, authored for this experiment. No prior outputs or existing cases read.",
    "limitations":["Shared logical templates across splits; only exact rule families/action vocabulary combinations and thresholds are split.","Fictional routing, not clinical validation.","Explicit old-rule context, not verified pretrained parametric knowledge.","Transfer changes domain and held-out rule families but retains logical structure."]})
    (a.out/"manifest.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(audit))
if __name__=="__main__":main()
