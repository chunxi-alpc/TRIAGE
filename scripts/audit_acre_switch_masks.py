#!/usr/bin/env python3
"""E0: CPU-only replay. Stop on any historical mask, annotation or metric mismatch."""
import collections, gzip, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from src.acre_pilot.span_mapping import build_prompt,map_annotations,select_targets,options_for,FIELDS
from src.acre_pilot.evaluator import evaluate_cases
from src.acre_pilot.oracle_diagnostic import *

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Safe restoration with the committed script, entirely inside the new directory.
    restored=OUT/"_restored/round2"; dest=restored/"revision3"
    dest.mkdir(parents=True,exist_ok=True)
    for name in ["development_manifest.json.gz","development_oracle.json.gz"]:
        shutil.copyfile(BASE/name,restored/name)
    for name in ["case_results.jsonl.gz","annotations.jsonl.gz"]:
        shutil.copyfile(OLD/name,dest/name)
    subprocess.run([sys.executable,str(ROOT/"scripts/unpack_acre_pilot_artifacts.py"),
                    "--directory",str(dest)],check=True,stdout=subprocess.DEVNULL)
    audit=read(BASE/"data_audit.json")
    for name,key in [("development_manifest.json","manifest_sha256"),("development_oracle.json","oracle_sha256")]:
        if not (BASE/name).exists():shutil.copyfile(restored/name,BASE/name)
        assert sha(restored/name)==sha(BASE/name)==audit[key],name
    assert sha(BASE/"development_references.json")==audit["references_sha256"]
    immutable={}
    for folder in ["results/acre_pilot/round2","results/acre_pilot/round1",
                   "results/algorithm_diagnostic","src/module2","src/module3"]:
        for p in (ROOT/folder).rglob("*"):
            if p.is_file() and "__pycache__" not in str(p):
                immutable[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/"historical_fingerprints.json",immutable)
    items,cases,annotations=normal_inputs()
    assert not ({i["guideline_id"] for i in items}&set(audit["historical_guideline_ids"]))
    assert len({i["guideline_id"] for i in items})==36
    bypair=collections.Counter(i["pair_id"] for i in items)
    assert len(bypair)==40 and set(bypair.values())=={2}
    tokenizer=AutoTokenizer.from_pretrained(read(ROOT/"configs/models.json")["models"]["qwen3_8b"]["path"],local_files_only=True)
    normal=[];mapped_by_id={};mapping_by_id={};prompts={}
    # No reference labels are read until all ordinary B2/B3 masks are reconstructed.
    for item in items:
        cid=item["case_id"];c=cases[cid];a=annotations[c["annotation_key"]]
        assert not a["reference_used"] and not c["normal_reference_used"]
        assert digest(a["input"])==a["cache_key"]
        assert a["input"]["patient"]==item["patient"] and a["input"]["question"]==item["question"]
        assert a["input"]["rules"]==[{k:r[k] for k in ("rule_id","text","action")} for r in item["guideline"]["rules"]]
        prompt,_,mapping=build_prompt(tokenizer,item)
        assert hashlib.sha256(prompt.encode()).hexdigest()==c["base_prompt_sha256"]
        assert options_for(item)==c["options"]
        mapped=map_annotations(item,a["rules"],mapping)
        assert mapped==c["annotations"],cid
        conflict=conflict_from_predictions(mapped,c["options"])
        assert conflict==c["out_of_single_rule_scope"]
        targets={}
        for method in ["B1","B2","B3"]:
            t,errors=select_targets(mapped,mapping,method)
            if conflict:t=[]
            targets[method]=t
            for beta in [0.,.25,.5,1.,2.]:
                v=c["variants"][method+"@"+str(beta)]
                assert t==v["attention"]["targets"],(cid,method,beta)
                assert v["attention"]["enabled"]==bool(t and beta)
                assert v["prompt_sha256"]==c["base_prompt_sha256"]
        d=overlap(targets["B2"],targets["B3"])
        row={"case_id":cid,"guideline_id":c["guideline_id"],"pair_id":item["pair_id"],
             "category":mask_kind(targets["B2"],targets["B3"],conflict),
             "conflict_disabled":conflict,**d,
             "removed_roles":span_roles(mapped,d["removed"]),"added_roles":span_roles(mapped,d["added"]),
             "B0_prediction":c["variants"]["B0"]["prediction"],
             "B2_prediction":c["variants"]["B2@2.0"]["prediction"],
             "B3_prediction":c["variants"]["B3@2.0"]["prediction"],
             "B2_changed_from_B0":c["variants"]["B2@2.0"]["prediction"]!=c["variants"]["B0"]["prediction"],
             "B3_changed_from_B0":c["variants"]["B3@2.0"]["prediction"]!=c["variants"]["B0"]["prediction"],
             "B2_B3_prediction_different":c["variants"]["B2@2.0"]["prediction"]!=c["variants"]["B3@2.0"]["prediction"]}
        normal.append(row);mapped_by_id[cid]=mapped;mapping_by_id[cid]=mapping;prompts[cid]=prompt
    oracle,refs=oracle_inputs()
    assert set(refs)==set(cases)==set(oracle)
    metrics,_,_=evaluate_cases(list(cases.values()),refs)
    assert metrics==read(OLD/"metrics.json")["all_grid_metrics"]
    plan=[];annrows=[];oracle_audit=[]
    for item,row in zip(items,normal):
        cid=item["case_id"];c=cases[cid];ref=refs[cid];mapped=mapped_by_id[cid];mapping=mapping_by_id[cid]
        predicted=next(r for r in mapped if r["rule_id"]==ref["target_rule_id"])
        row.update(target_true_state=ref["target_applicability"],target_predicted_state=predicted["applicability"],
                   target_predicted_relevant=predicted["relevant"])
        assert oracle[cid]["verified"] is True
        om=map_annotations(item,oracle[cid]["rules"],mapping)
        assert not any(r["mapping_errors"] for r in om)
        assert om==c["oracle_annotations"]
        for r in om:
            assert r["applicability"]==ref["rule_states"][r["rule_id"]]
            rule=next(x for x in item["guideline"]["rules"] if x["rule_id"]==r["rule_id"])
            assert r["action_spans"]==[rule["action"]]
        targets={}
        for scope,source in [("",om),("-target",[r for r in om if r["rule_id"]==ref["target_rule_id"]])]:
            for method,strategy in [("OF","B2"),("OS","B3")]:
                t,err=select_targets(source,mapping,strategy)
                assert not err
                targets[method+scope]=t
        for beta in [0.,1.,2.]:
            assert targets["OS"]==c["variants"]["B6@"+str(beta)]["attention"]["targets"]
        for scope in ["","-target"]:
            d=overlap(targets["OF"+scope],targets["OS"+scope])
            oracle_audit.append({"case_id":cid,"scope":scope or "full","target_state":ref["target_applicability"],
                                 "category":mask_kind(targets["OF"+scope],targets["OS"+scope]),**d,
                                 "removed_roles":span_roles(om,d["removed"]),"added_roles":span_roles(om,d["added"])})
        plan.append({"case_id":cid,"guideline_id":item["guideline_id"],"pair_id":item["pair_id"],
                     "base_prompt_sha256":c["base_prompt_sha256"],"prompt_tokens":mapping["prompt_tokens"],
                     "oracle_annotation_sha256":digest(oracle[cid]["rules"]),"targets":targets,
                     "oracle_only_non_deployable":True})
        for pred in mapped:
            gold=next(r for r in om if r["rule_id"]==pred["rule_id"])
            decisive=ref["evidence_span"] if pred["rule_id"]==ref["target_rule_id"] else None
            role="supporting_evidence" if gold["applicability"]=="applicable" else "refuting_evidence"
            selected=pred["mapped"][role]
            recall=any(ref["evidence"] in s["text"] for s in selected) if decisive else None
            selected_chars=set(x for s in selected for x in range(s["source_start"],s["source_end"]))
            gold_chars=set(range(decisive["start"],decisive["end"])) if decisive else set()
            annrows.append({"case_id":cid,"rule_id":pred["rule_id"],"is_target":decisive is not None,
                "true_state":gold["applicability"],"predicted_state":pred["applicability"],
                "state_correct":gold["applicability"]==pred["applicability"],
                "reference_relevant":gold["relevant"],"predicted_relevant":pred["relevant"],
                "reference_relevance_protocol":"frozen diagnostic: all supplied guide rules retained; not semantic relevance truth",
                "reference_rules_not_reselected":True,"conflict_disabled":c["out_of_single_rule_scope"],
                "source_rule":next(r["text"] for r in item["guideline"]["rules"] if r["rule_id"]==pred["rule_id"]),
                "reference_evidence":gold[role],"predicted_support":pred["supporting_evidence"],
                "predicted_refute":pred["refuting_evidence"],"mapped_positions":{k:[s["tokens"] for s in v] for k,v in pred["mapped"].items()},
                "decisive_recall":recall,"selected_role":role,"selected_chars":len(selected_chars),
                "decisive_overlap_chars":len(selected_chars&gold_chars) if decisive else None,
                "decisive_character_precision_proxy":len(selected_chars&gold_chars)/len(selected_chars) if decisive and selected_chars else None,
                "precision_proxy_limitation":"overlap with registered decisive sentence, not full semantic evidence precision",
                "error_flags":[k for k,v in {"state_error":gold["applicability"]!=pred["applicability"],
                    "reference_relevance_miss":gold["relevant"] and not pred["relevant"],
                    "decisive_role_or_evidence_miss":decisive is not None and not recall,
                    "mapping_error":bool(pred["mapping_errors"])}.items() if v]})
    csvwrite(OUT/"mask_audit.csv",normal);csvwrite(OUT/"annotation_error_audit.csv",annrows)
    csvwrite(OUT/"oracle_mask_audit.csv",oracle_audit)
    write(OUT/"oracle_mask_plan.json",plan)
    counts=dict(collections.Counter(r["category"] for r in normal))
    full=[r for r in oracle_audit if r["scope"]=="full"]
    summary={"case_count":80,"pair_count":40,"guideline_count":36,"categories_exclusive":counts,
        "both_nonempty_different":sum(r["category"]=="both_nonempty_different" for r in normal),
        "any_mask_difference":sum(r["symmetric_difference_count"]>0 for r in normal),
        "both_empty_jaccard":None,"new_negative_evidence_cases":sum(any(s["role"] in ["refuting_evidence","exception_spans"] for s in r["added_roles"]) for r in normal),
        "new_action_removed_cases":sum(any(s["role"]=="action_spans" for s in r["removed_roles"]) for r in normal),
        "full_oracle_categories":dict(collections.Counter(r["category"] for r in full)),
        "full_oracle_new_negative_cases":sum(any(s["role"] in ["refuting_evidence","exception_spans"] for s in r["added_roles"]) for r in full),
        "normal_all_targets_and_metrics_exact":True,
        "E1_decision":"proceed with frozen full Oracle comparison and prespecified target-only diagnostic; coverage limitations must be reported"}
    write(OUT/"mask_audit_summary.json",summary)
    check={"E0_passed":True,"normal_masks_recomputed_without_reference":True,"normal_annotations_replayed_exact":80,
        "all_historical_metrics_exact":True,"all_B6_masks_exact":True,"restoration_sha256_verified":True,
        "cases":80,"pairs":40,"guidelines":36,"historical_guideline_overlap":[],
        "unused_annotation_cache_records":len(set(annotations)-{c["annotation_key"] for c in cases.values()}),
        "normal_inputs":{"manifest":sha(BASE/"development_manifest.json"),"annotations_archive":sha(OLD/"annotations.jsonl.gz")},
        "oracle_mask_only_inputs":{"oracle":sha(BASE/"development_oracle.json")},
        "offline_evaluation_inputs":{"references":sha(BASE/"development_references.json"),"action_alias_audit":sha(BASE/"action_alias_audit.json")},
        "archive_sha256":{name:sha(OLD/name) for name in ["case_results.jsonl.gz","annotations.jsonl.gz"]},
        "plan_sha256":sha(OUT/"oracle_mask_plan.json"),"remote_base_commit":REMOTE_BASE}
    write(OUT/"recompute_check.json",check)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()
