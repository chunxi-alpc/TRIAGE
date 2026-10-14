"""Small CPU regressions for diagnostic repairs; no model or reference inputs."""
import json
import re
import tempfile
from pathlib import Path
from .conditions import match_conditions, _numeric, _predicate_rows
from .modeler import Module1
from .rules import select_and_parse

class EvidenceBackend:
    def __init__(self): self.calls=[]
    def score_pairs(self, query, texts):
        return [0.95 if "Patient evidence:" in query and "critical" in text else .8-i*.01 for i,text in enumerate(texts)]
    def json(self, system, user, max_new_tokens=0):
        self.calls.append({"system": system, "user": user})
        if "Independently judge" in system or "Judge whether each complete" in system:
            data=json.loads(user)
            out={}
            for c in data["conditions"]:
                value="unknown" if "Independently judge" in system and "handfeeding" in c["condition"] else "true"
                out[c["condition_id"]]={"condition":c["condition"], "predicate_truth":value,
                    "evidence_relation":"insufficient" if value=="unknown" else "entails",
                    "evidence_sentence_ids":[] if value=="unknown" else [0], "reason":"Controlled regression"}
            return out
        if "clinical decision model" in system:
            return {"decision":"Applicable action", "actions":["Applicable action"],"status":"decided","rationale":"Patient evidence"}
        raise ValueError("Unexpected regression prompt")

def run():
    checks=[]
    backend=EvidenceBackend()
    rule=lambda rid, text: {"rule_id":rid,"text":text,"conditions":[],"action":"Action "+rid}
    rules=[rule("r"+str(i), "critical rule" if i==20 else "rule "+str(i)) for i in range(30)]
    with tempfile.TemporaryDirectory() as cache:
        got, relevance=select_and_parse(rules,"Management?",backend,top_k=12,cache_dir=cache,patient="The patient has the relevant condition.")
        assert len(got)==12 and {"r0","r1","r2","r3","r20"}.issubset({r["rule_id"] for r in got})
        checks.append("question_candidates_preserved_and_patient_candidate_recalled")
        got,_=select_and_parse(rules[:24],"Management?",backend,top_k=12,cache_dir=cache,patient="Patient evidence exists.")
        assert len(got)==24
        checks.append("small_guideline_has_no_early_candidate_truncation")
    patient="Oral nutritional supplementation is feasible and available."
    rule1={"rule_id":"hand","conditions":[{"condition_id":"c","text":"Is staff handfeeding assistance available?"}],"logic":"AND","action":"Handfeeding"}
    relations, apps, spans=match_conditions([rule1],patient,backend)
    assert relations[0]["initial_base_condition_status"]=="satisfied" and relations[0]["status"]=="unknown"
    assert apps[0]["status"]=="unknown"
    checks.append("unrelated_exact_quote_does_not_establish_condition")
    inverse={"rule_id":"inverse","conditions":[{"condition_id":"c","text":"No diabetes","negated":True}],"logic":"AND","action":"Action"}
    relations,apps,_=match_conditions([inverse],"The patient denies diabetes.",backend)
    assert relations[0]["status"]=="not_satisfied"
    checks.append("verified_base_condition_is_negated_once")
    note="A 58-year-old patient became unwell 2 days ago."
    condition={"text":"Age >=18 years","numeric":{"field":"age","operator":">=","value":18,"unit":"years"}}
    assert _numeric(condition,[{"text":note,"start":0,"end":len(note),"sentence_id":0}])[0]=="satisfied"
    checks.append("numeric_age_ignores_unrelated_duration")
    module=Module1(backend=backend)
    before=len(backend.calls)
    result=module._decision(patient,"Management?",[rule1],apps)
    assert result["status"]=="insufficient_evidence" and not result["actions"] and len(backend.calls)==before
    checks.append("no_applicable_rule_cannot_generate_guideline_action")
    active=rule("active","An applicable recommendation")
    module._decision(patient,"Management?",[active,rule1],[{"rule_id":"active","status":"applicable"},{"rule_id":"hand","status":"not_applicable"}],relations)
    prompt=json.loads(backend.calls[-1]["user"])
    assert [r["rule_id"] for r in prompt["current_guideline_rules"]]==["active"] and "condition_evidence" in prompt
    checks.append("blocked_actions_removed_from_guided_prompt")
    groups=[{"text":"Has feeding problems?"}, {"text":"Was OGTT performed?"}]
    spans=[{"text":"No feeding problems. OGTT was not performed.","sentence_id":0,"start":0,"end":48}]
    raw={"condition_0":{"condition":groups[1]["text"],"predicate_truth":"true","evidence_sentence_ids":[0]},
         "condition_1":{"condition":groups[1]["text"],"predicate_truth":"false","evidence_sentence_ids":[0]}}
    assert [x[1] for x in _predicate_rows(raw,groups,spans)]==["unknown","not_satisfied"]
    checks.append("condition_echo_rejects_batch_shift_but_preserves_false")
    class ConflictBackend(EvidenceBackend):
        def json(self,system,user,max_new_tokens=0):
            if "Independently judge" in system or "Judge whether each complete" in system:
                data=json.loads(user)
                truth="true" if "Independently judge" in system else "false"
                return {c["condition_id"]:{"condition":c["condition"],"predicate_truth":truth,
                    "evidence_relation":"entails" if truth=="true" else "contradicts",
                    "evidence_sentence_ids":[0],"reason":"Controlled polarity conflict"} for c in data["conditions"]}
            return super().json(system,user,max_new_tokens)
    r={"rule_id":"feeding","conditions":[{"condition_id":"c","text":"Has feeding problems?"}],"logic":"AND","action":"Treat feeding problems"}
    rel,app,_=match_conditions([r],"No feeding problems.",ConflictBackend())
    assert rel[0]["initial_base_condition_status"]=="not_satisfied" and rel[0]["status"]=="unknown" and app[0]["status"]=="unknown"
    checks.append("evidence_verifier_cannot_reverse_known_negative_to_positive")
    out=Path(__file__).resolve().parents[2]/"results/diagnostic_module1_checks.json"
    out.write_text(json.dumps({"passed":True,"models_loaded":False,"checks":checks},indent=2)+"\n")
    print("Passed "+str(len(checks))+" module1 repair regressions")

if __name__=="__main__": run()
