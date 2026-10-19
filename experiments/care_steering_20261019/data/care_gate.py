"""Restricted CARE module-one adapter for explicit synthetic observations.

Reuses frozen V6 numeric comparison and strong-Kleene logic. Categorical extraction
is a deliberately narrow exact-sentence parser added for the authored schema.
It does NOT call TrialMatchAI or BioLinkBERT, estimate confidence, parse free-form
rules, or constitute the full CARE module-one model.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any
_SRC=Path(__file__).parents[1]/"sources"/"care_v6_frozen.py"
spec=importlib.util.spec_from_file_location("care_v6_frozen",_SRC)
core=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=core
spec.loader.exec_module(core)

def predict(patient: str, structured_rules: list[dict[str,Any]]) -> dict[str,Any]:
    spans=core.sentences_with_offsets(patient)
    rules=[]
    for r in structured_rules:
        if r["role"]=="old":
            continue
        states={}
        evidence={}
        for c in r.get("conditions",[])+r.get("exceptions",[]):
            cid=c["condition_id"]
            if "numeric" in c:
                obs=core._numeric_observation(c,spans)
                states[cid]=obs["status"] if obs else "U"
                evidence[cid]=obs
            else:
                pos=[s for s in spans if s["text"]==c.get("positive")]
                neg=[s for s in spans if s["text"]==c.get("negative")]
                states[cid]="T" if pos and not neg else "F" if neg and not pos else "U"
                evidence[cid]={"positive":pos,"negative":neg,"extractor":"exact explicit sentence"}
        base=core.eval_logic(r.get("logic",{"op":"AND","args":list(states)}),states)
        exc=core.eval_logic(r["exception_logic"],states) if r.get("exceptions") else "F"
        state=core.tri_and([base,core.tri_not(exc)])
        rules.append({"rule_id":r["rule_id"],"state":state,"atoms":states,"evidence":evidence})
    state=next(r["state"] for r in rules if r["rule_id"]=="NEW")
    old=next(r for r in structured_rules if r["role"]=="old")
    new=next(r for r in structured_rules if r["role"]=="new")
    old_valid=bool(old["valid"])
    conflict=old["action"]!=new["action"]
    gate=(1.0 if state=="T" else -1.0 if state=="F" and old_valid else 0.0) if conflict else 0.0
    return {"method":"CARE_V6_numeric_logic_with_exact_categorical_parser","full_care_module1":False,
            "new_state":state,"old_valid":old_valid,"conflict":conflict,"signed_gate":gate,
            "positive_gate":float(state=="T" and conflict),"rules":rules,
            "input_whitelist":["patient","structured_rules"],
            "limitations":"Explicit authored syntax only; no Phi-4, BioLinkBERT, or confidence calibration."}

def predict_item(item: dict[str,Any])->dict[str,Any]:
    return predict(item["patient"],item["structured_rules"])

def get_gate(item:dict[str,Any])->float:
    return predict_item(item)["signed_gate"]

if __name__=="__main__":
    import json,time
    base=Path(__file__).parent
    rows=[json.loads(s) for s in (base/"cases.jsonl").read_text().splitlines()]
    start=time.perf_counter()
    outputs=[]
    for r in rows:
        pred=predict_item(r)
        # Poison every field outside the whitelist: this must not change predictions.
        poisoned={k:"LABEL_POISON" for k in r}
        poisoned.update(patient=r["patient"],structured_rules=r["structured_rules"])
        assert predict_item(poisoned)==pred
        outputs.append({"item_id":r["item_id"],**pred,"reference_state":r["applicability_state"],"state_correct":pred["new_state"]==r["applicability_state"]})
    elapsed=time.perf_counter()-start
    (base/"module1_reduced_outputs.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in outputs))
    audit={"items":len(rows),"state_correct":sum(x["state_correct"] for x in outputs),"seconds_for_predictions_and_leakage_checks":elapsed,
           "full_care_module1":False,"result_kind":"deterministic interface audit, not pretrained-model accuracy",
           "forbidden_field_poisoning_check_passed":True,"gate_counts":{str(v):sum(x["signed_gate"]==v for x in outputs) for v in [-1,0,1]}}
    (base/"module1_reduced_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))
