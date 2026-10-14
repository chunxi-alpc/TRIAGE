#!/usr/bin/env python3
"""CPU-only aggregation; read completed results, never call a model."""
import argparse
import csv
import collections
import hashlib
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=["original","original_full_guideline","oracle_rule","triage_prompt_only","triage_full"]
NAMES=["无规则条件基线","原始全文指南","正确规则诊断上界","TRIAGE提示，无注意力干预","完整TRIAGE"]
EPS=1e-6
def stats(v):
    return {"n":len(v),"mean":statistics.fmean(v) if v else None,
        "median":statistics.median(v) if v else None,
        "positive":sum(x>EPS for x in v),"zero":sum(abs(x)<=EPS for x in v),
        "negative":sum(x < -EPS for x in v)}
def compare(rows,left,right):
    result={"n":len(rows),"corrected":0,"regressed":0,"corrected_case_ids":[],"regressed_case_ids":[]}
    for row in rows:
        a,b=row["variants"][left],row["variants"][right]
        if not a["correct"] and b["correct"]:
            result["corrected"]+=1;result["corrected_case_ids"].append(row["case_id"])
        if a["correct"] and not b["correct"]:
            result["regressed"]+=1;result["regressed_case_ids"].append(row["case_id"])
    for metric in ["correct_log_probability","correct_probability","correct_option_probability",
                   "correct_minus_prior_log_probability","correct_minus_strongest_incorrect_prior",
                   "correct_minus_raw_guideline_decision"]:
        result[metric+"_delta"]=stats([r["variants"][right][metric]-r["variants"][left][metric] for r in rows])
    wrong=[r for r in rows if not r["evaluation"]["prior_equals_gold"]]
    result["original_incorrect_n"]=len(wrong)
    result["gold_minus_prior_delta_on_original_incorrect"]=stats([r["variants"][right]["correct_minus_prior_log_probability"]-r["variants"][left]["correct_minus_prior_log_probability"] for r in wrong])
    return result
def initial_conditions(row):
    # Only known original-path signed nodes are evaluated. Preserve missing initial fields.
    rels={r["condition_id"]:r for r in row["module1"]["condition_evidence_relations"]}
    result=[]
    for c in row["evaluation"]["conditions"]:
        rel=rels.get(c["condition_id"],{})
        candidates=["initial_status","initial_condition_status","status_before_verification"]
        initial=next((rel[k] for k in candidates if k in rel),None)
        if initial is None and "initial_base_condition_status" in rel:
            initial=rel["initial_base_condition_status"]
            if rel.get("negated"):
                initial={"satisfied":"not_satisfied","not_satisfied":"satisfied"}.get(initial,"unknown")
        if initial is None and isinstance(rel.get("verification"),dict):
            initial=rel["verification"].get("initial_status")
        if initial is None: continue
        result.append({"case_id":row["case_id"],"condition_id":c["condition_id"],
            "expected":c["expected"],"initial":initial,"final":c["predicted"],
            "initial_correct":initial==c["expected"],"final_correct":c["correct"]})
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,default=ROOT/"results/algorithm_diagnostic")
    out=p.parse_args().results
    manifest=json.loads((out/"manifest.json").read_text())
    rows=[];missing=[]
    for item in manifest["samples"]:
        path=out/("case_"+item["case_id"].replace(":","_")+".json")
        if not path.exists():missing.append(item["case_id"]);continue
        r=json.loads(path.read_text())
        if r.get("case_id")!=item["case_id"] or not r.get("evaluation"):missing.append(item["case_id"]);continue
        rows.append(r)
    if missing:raise SystemExit("Results incomplete: "+", ".join(missing))
    n=len(rows)
    active=[r for r in rows if r["intervention_stats"].get("context_changed_elements",0) or r["intervention_stats"].get("prior_changed_elements",0)]
    variants={}
    for v in VARIANTS:
        variants[v]={"correct":sum(r["variants"][v]["correct"] for r in rows),"n":n}
        for metric in ["correct_log_probability","correct_probability","correct_option_probability",
                       "correct_minus_prior_log_probability","correct_minus_strongest_incorrect_prior"]:
            variants[v][metric]=stats([r["variants"][v][metric] for r in rows])
        variants[v]["gold_minus_prior_on_original_incorrect"]=stats([r["variants"][v]["correct_minus_prior_log_probability"] for r in rows if not r["evaluation"]["prior_equals_gold"]])
    heads={}
    for h in sorted({h for r in rows for h in r["heads"]}):
        allrows=[r for r in rows if h in r["heads"]]
        act=[r for r in allrows if r["heads"][h]["stats"].get("context_changed_elements",0) or r["heads"][h]["stats"].get("prior_changed_elements",0)]
        heads[h]={"single_head_delta_all":stats([r["heads"][h]["key_intervention_correct_margin_delta"] for r in allrows]),
          "single_head_delta_active":stats([r["heads"][h]["key_intervention_correct_margin_delta"] for r in act]),
          "retained_head_causal_effect_all":stats([r["heads"][h]["retained_head_correct_margin_effect"] for r in allrows]),
          "negative_active_case_ids":[r["case_id"] for r in act if r["heads"][h]["key_intervention_correct_margin_delta"] < -EPS]}
        st=heads[h]["single_head_delta_active"]
        heads[h]["effect_sign_varies"]=st["positive"]>0 and st["negative"]>0
    cond=[c for r in rows for c in r["evaluation"]["conditions"]]
    initial=[c for r in rows for c in initial_conditions(r)]
    comparisons={name:compare(sub,a,b) for name,sub,a,b in [
        ("full_vs_original",rows,"original","triage_full"),
        ("full_vs_raw_guideline",rows,"original_full_guideline","triage_full"),
        ("attention_vs_prompt_only",rows,"triage_prompt_only","triage_full"),
        ("attention_vs_prompt_only_active",active,"triage_prompt_only","triage_full"),
        ("oracle_vs_raw_guideline",rows,"original_full_guideline","oracle_rule")]}
    summary={"schema_version":"triage_algorithm_summary.v1","n":n,
      "guideline_count":len({r["guideline_id"] for r in rows}),"dataset":"MEGA-CDP","split":"train_val",
      "scope":"Single-letter forced choice over every source-guideline action plus insufficient evidence. No-rule original still sees all action options. Free-text generation is not evaluated.",
      "reference_isolation":{"normal_reference_answers_used":any(r.get("normal_inference_reference_answers_used",True) for r in rows),
        "oracle_exception":"Correct original guideline rule selected offline only after ordinary inference; no gt answer text/path labels in inference or gates"},
      "metric_definitions":{"probability":"Full-vocabulary probability of the exact single-letter token. Option-normalized probability is separately named.",
        "gold_minus_prior":"logP(gold)-logP(original selected option), zero if original is correct; original-incorrect subset reported separately.",
        "head_effect":"Correct-minus-fixed-incorrect margin: single-head edit minus prompt-only; retained-head effect is prompt-only minus head ablation. Positive/zero/negative counts are empirical; no arbitrary stability threshold.",
        "condition_scope":"Known reference-path signed condition IDs, deduplicated per case; off-path nodes excluded; guided_decision free text receives no automatic semantic accuracy label."},
      "variants":variants,
      "retrieval":{"before_top3":sum(r["evaluation"]["key_rule_recalled_before"] for r in rows),
         "after":sum(r["evaluation"]["key_rule_recalled_after"] for r in rows),"n":n,"interpretation":"Small complete guidelines retain every leaf; recall gain does not demonstrate improved ranking on large guidelines."},
      "applicability":{"gold_applicable":sum(r["evaluation"]["gold_rule_applicability"]=="applicable" for r in rows),"n":n,
         "gold_status_counts":dict(collections.Counter(r["evaluation"]["gold_rule_applicability"] for r in rows)),
         "candidate_correct":sum(r["evaluation"]["applicability_correct"] for r in rows),
         "candidate_total":sum(r["evaluation"]["applicability_total"] for r in rows),
         "known_signed_conditions_correct":sum(c["correct"] for c in cond),"known_signed_conditions_total":len(cond),
         "known_signed_conditions_unknown":sum(c["predicted"]=="unknown" for c in cond)},
      "condition_verification":{"initial_fields_available_n":len(initial),
         "initial_correct":sum(c["initial_correct"] for c in initial),"final_correct":sum(c["final_correct"] for c in initial),
         "improved":sum(not c["initial_correct"] and c["final_correct"] for c in initial),
         "regressed":sum(c["initial_correct"] and not c["final_correct"] for c in initial),
         "changed_rows":[c for c in initial if c["initial"]!=c["final"]]},
      "comparisons":comparisons,"active_intervention_n":len(active),"heads":heads,
      "controls":{"zero_gate_exact":sum(r["zero_gate_control"]["exact"] for r in rows),
         "zero_strength_exact":sum(r["zero_strength_control"]["exact"] for r in rows),"n":n},
      "failure_cause_counts":dict(collections.Counter(c for r in rows for c in r["evaluation"]["failure_causes"])),
      "case_results":[{"case_id":r["case_id"],"gold_rule_id":r["evaluation"]["gold_rule_id"],
         "recall_before":r["evaluation"]["key_rule_recalled_before"],"recall_after":r["evaluation"]["key_rule_recalled_after"],
         "gold_applicability":r["evaluation"]["gold_rule_applicability"],"gold_action_gate":r["evaluation"]["gold_action_gate"],
         "decisions":{v:r["variants"][v]["decision_rule_ids"] for v in VARIANTS},
          "correct":{v:r["variants"][v]["correct"] for v in VARIANTS},
         "correct_probability":{v:r["variants"][v]["correct_probability"] for v in VARIANTS},
         "correct_log_probability":{v:r["variants"][v]["correct_log_probability"] for v in VARIANTS},
         "correct_minus_prior_log_probability":{v:r["variants"][v]["correct_minus_prior_log_probability"] for v in VARIANTS},
         "full_correct_probability_delta":r["variants"]["triage_full"]["correct_probability"]-r["variants"]["original"]["correct_probability"],
         "attention_correct_logp_delta":r["variants"]["triage_full"]["correct_log_probability"]-r["variants"]["triage_prompt_only"]["correct_log_probability"],
         "failures":r["evaluation"]["failure_causes"]} for r in rows],
      "reproduction":{"manifest_sha256":hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest(),
         "references_sha256":hashlib.sha256((out/"references.json").read_bytes()).hexdigest(),
         "sum_case_seconds":sum(r["elapsed_seconds"] for r in rows),
         "peak_allocated_gib":max(r["gpu_peak_allocated_gib"] for r in rows)}}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    fields=["case_id","gold_rule_id","recall_before","recall_after","gold_applicability","gold_action_gate"]+[v+"_decision" for v in VARIANTS]+[v+"_correct" for v in VARIANTS]+[v+"_gold_probability" for v in VARIANTS]+[v+"_gold_minus_prior" for v in VARIANTS]+["full_correct_probability_delta","attention_correct_logp_delta","failure_causes"]
    with (out/"cases.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for case in summary["case_results"]:
            flat={k:case[k] for k in fields if k in case}
            for v in VARIANTS:
                flat[v+"_decision"]="|".join(case["decisions"][v])
                flat[v+"_correct"]=case["correct"][v]
                flat[v+"_gold_probability"]=case["correct_probability"][v]
                flat[v+"_gold_minus_prior"]=case["correct_minus_prior_log_probability"][v]
            flat["failure_causes"]="|".join(case["failures"])
            writer.writerow(flat)
    print(json.dumps({"n":n,"correct":{v:variants[v]["correct"] for v in VARIANTS},"summary":str(out/"summary.json")},ensure_ascii=False))
if __name__=="__main__":main()

