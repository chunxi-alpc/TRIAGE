#!/usr/bin/env python3
"""Offline E1 metrics, paired exploratory statistics and complete case effects."""
import collections,copy,csv,datetime,math,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.acre_pilot.oracle_diagnostic import *
from src.acre_pilot.evaluator import evaluate_cases,paired_comparisons

def distribution(values):
    values=sorted(values)
    if not values:return {"n":0,"positive":0,"zero":0,"negative":0,"mean":None,"median":None}
    return {"n":len(values),"positive":sum(x>1e-8 for x in values),"zero":sum(abs(x)<=1e-8 for x in values),
            "negative":sum(x< -1e-8 for x in values),"mean":statistics.mean(values),"median":statistics.median(values),
            "min":min(values),"q025":values[int(.025*(len(values)-1))],"q25":values[int(.25*(len(values)-1))],
            "q75":values[int(.75*(len(values)-1))],"q975":values[int(.975*(len(values)-1))],"max":max(values)}

def paired_endpoint(rows,a,b,endpoint):
    eligible=[r for r in rows if r["method"] in [a,b]]
    index={(r["case_id"],r["method"]):r for r in eligible}
    ids=sorted({r["case_id"] for r in eligible})
    paired=[(index[cid,a],index[cid,b]) for cid in ids]
    if endpoint=="oar":paired=[(x,y) for x,y in paired if x["target_state"]=="inapplicable"]
    if endpoint=="retention":paired=[(x,y) for x,y in paired if x["baseline_correct"]]
    if endpoint=="pair_accuracy":
        groups=collections.defaultdict(list)
        for x,y in paired:groups[x["pair_id"]].append((x,y))
        paired2=[]
        for pair in groups.values():
            if len(pair)!=2:continue
            x0,y0=pair[0]
            paired2.append(({"guideline_id":x0["guideline_id"],"value":all(x["correct"] for x,y in pair)},
                            {"value":all(y["correct"] for x,y in pair)}))
    else:
        key="oar" if endpoint=="oar" else "correct"
        paired2=[({"guideline_id":x["guideline_id"],"value":x[key]},{"value":y[key]}) for x,y in paired]
    only_a=sum(x["value"] and not y["value"] for x,y in paired2)
    only_b=sum(y["value"] and not x["value"] for x,y in paired2)
    both=sum(x["value"] and y["value"] for x,y in paired2)
    neither=len(paired2)-only_a-only_b-both;n=only_a+only_b
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(only_a,only_b)+1))/2**n) if n else 1.
    clusters=collections.defaultdict(list)
    for x,y in paired2:clusters[x["guideline_id"]].append(int(x["value"])-int(y["value"]))
    rng=random.Random(SEED);keys=sorted(clusters);draws=[]
    for _ in range(2000):
        vals=[v for k in rng.choices(keys,k=len(keys)) for v in clusters[k]]
        if vals:draws.append(sum(vals)/len(vals))
    draws.sort()
    return {"method_a":a,"method_b":b,"endpoint":endpoint,"n":len(paired2),"a_only":only_a,"b_only":only_b,
            "both":both,"neither":neither,"difference_a_minus_b":(only_a-only_b)/len(paired2) if paired2 else None,
            "exact_mcnemar_p":p,"guide_cluster_ci95":[draws[50],draws[1950]] if draws else None,
            "clusters":len(clusters),"bootstrap_samples":2000,
            "interpretation":"exploratory observed development data; OAR lower is better; no equivalence claim"}

def main():
    items,old,_=normal_inputs();oracle,refs=oracle_inputs();plans={p["case_id"]:p for p in read(OUT/"oracle_mask_plan.json")}
    archive_mode="--from-archive" in sys.argv or not all((OUT/"workers"/("done_"+str(w)+".json")).exists() for w in range(6))
    if archive_mode:
        receipt=read(OUT/"resource_plan.json");done=receipt["workers"]
        saved=collections.defaultdict(list)
        for row in jsonl(OUT/"case_effects.jsonl.gz"):saved[row["case_id"]].append(row)
    else:done=[read(OUT/"workers"/("done_"+str(w)+".json")) for w in range(6)]
    assert all(d["complete"] for d in done) and sum(d["cases"] for d in done)==80
    effects=[];cases=[]
    for item in items:
        cid=item["case_id"];c=old[cid];ref=refs[cid]
        if archive_mode:
            rows=[r for r in saved[cid] if r["method"]=="O0" or r["method"].startswith(("OF","OS"))]
            assert len(rows)==9
            first=rows[0]
            n={"case_id":cid,"same_oracle_source_for_fixed_and_switch":True,
               "reference_text_in_prompt":False,"base_prompt_sha256":first["base_prompt_sha256"],
               "identity_controls":first["identity_controls"],"variants":{}}
            for row in rows:
                n["variants"][row["method"]]={k:row[k] for k in ["prediction","action","option_logits","option_log_probabilities","attention"]}
                n["variants"][row["method"]].update(prompt_sha256=row["base_prompt_sha256"],prompt_tokens=c["prompt_tokens"],
                    reused_from=row["reuse_provenance"])
            n["oracle_annotation_sha256"]=plans[cid]["oracle_annotation_sha256"]
        else:n=read(OUT/"workers"/("case_"+cid+".json"))
        assert n["same_oracle_source_for_fixed_and_switch"] and not n["reference_text_in_prompt"]
        assert n["base_prompt_sha256"]==c["base_prompt_sha256"]
        assert all(n["identity_controls"].values())
        evaluation={k:c[k] for k in ["case_id","guideline_id","pair_id","options","annotations","annotation_errors","out_of_single_rule_scope"]}
        evaluation["variants"]=copy.deepcopy(n["variants"])
        evaluation["variants"]["B0"]=c["variants"]["B0"]
        for beta in [1.,2.]:
            for method in ["B2","B3"]:evaluation["variants"][method+"@"+str(beta)]=c["variants"][method+"@"+str(beta)]
        cases.append(evaluation)
        gold_i=next(i for i,o in enumerate(c["options"]) if o["action"]==ref["gold_action"])
        comp_i=next(i for i,o in enumerate(c["options"]) if o["action"]==ref["competitor_action"])
        base=c["variants"]["B0"];base_margin=base["option_logits"][gold_i]-base["option_logits"][comp_i]
        gold=c["options"][gold_i]["label"];target_labels={o["label"] for o in c["options"] if o["action"] in ref["target_action_aliases"]}
        pred_target=next(r for r in c["annotations"] if r["rule_id"]==ref["target_rule_id"])
        for method,v in evaluation["variants"].items():
            if method=="B0":continue
            if method=="O0":beta=0.;scope="full";pairdiff=overlap([],[]);targets=[]
            else:
                name,b=method.split("@");beta=float(b)
                if name in ["B2","B3"]:
                    scope="predicted";targets=v["attention"]["targets"]
                    pairdiff=overlap(c["variants"]["B2@"+b]["attention"]["targets"],c["variants"]["B3@"+b]["attention"]["targets"])
                else:
                    scope="target" if name.endswith("-target") else "full";suffix="-target" if scope=="target" else ""
                    targets=plans[cid]["targets"][name]
                    pairdiff=overlap(plans[cid]["targets"]["OF"+suffix],plans[cid]["targets"]["OS"+suffix])
            margin=v["option_logits"][gold_i]-v["option_logits"][comp_i]
            effects.append({"case_id":cid,"guideline_id":item["guideline_id"],"pair_id":item["pair_id"],"method":method,"beta":beta,
                "scope":scope,"target_state":ref["target_applicability"],"predicted_state_correct":pred_target["applicability"]==ref["target_applicability"],
                "conflict_disabled":c["out_of_single_rule_scope"],"prediction":v["prediction"],"action":v["action"],
                "gold_label":gold,"gold_action":ref["gold_action"],"fixed_competitor_label":c["options"][comp_i]["label"],
                "fixed_competitor_action":ref["competitor_action"],"target_action_aliases":ref["target_action_aliases"],
                "correct":v["prediction"]==gold,"baseline_correct":base["prediction"]==gold,
                "oar":ref["target_applicability"]=="inapplicable" and v["prediction"] in target_labels,
                "margin":margin,"delta_margin":margin-base_margin,"base_margin":base_margin,
                "attention":v["attention"],"mask_comparison":pairdiff,
                "both_nonempty_different":pairdiff["fixed_count"]>0 and pairdiff["switch_count"]>0 and pairdiff["symmetric_difference_count"]>0,
                "actual_target_count":len(targets),"base_prompt_sha256":n["base_prompt_sha256"],
                "oracle_annotation_sha256":n["oracle_annotation_sha256"] if scope!="predicted" and method!="O0" else None,
                "oracle_diagnostic_only":scope in ["target","full"] and method!="O0","reference_text_in_prompt":False,
                "option_logits":v["option_logits"],"option_log_probabilities":v["option_log_probabilities"],
                "reuse_provenance":"revision3 B0; all 80 candidate logits rechecked" if method=="O0" else v.get("reused_from","same-mask computation" if v.get("same_mask_reuse") else "new GPU forward"),
                "identity_controls":n["identity_controls"]})
    metrics,_,details=evaluate_cases(cases,refs)
    names=[m for m in metrics if m!="B0"]
    assert metrics["O0"]==metrics["B0"] and metrics["O0"]["accuracy"]["numerator"]==64
    def subset(choose):
        selected=[c for c in cases if choose(c)]
        if not selected:return {"n":0,"metrics":{}}
        m,_,_=evaluate_cases(selected,refs)
        return {"n":len(selected),"metrics":{k:v for k,v in m.items() if k!="B0"},
                "retention_denominator_note":"stratum B0-correct cases; primary all-80 denominator remains 64"}
    strata={"all":{"n":80,"metrics":{k:metrics[k] for k in names}}}
    for state in ["applicable","inapplicable"]:
        strata[state]=subset(lambda c:refs[c["case_id"]]["target_applicability"]==state)
    for value in [True,False]:
        strata["predicted_state_"+("correct" if value else "wrong")]=subset(lambda c:
            (next(r for r in c["annotations"] if r["rule_id"]==refs[c["case_id"]]["target_rule_id"])["applicability"]==refs[c["case_id"]]["target_applicability"])==value)
        strata["conflict_"+str(value).lower()]=subset(lambda c:c["out_of_single_rule_scope"]==value)
    for scope in ["full","target","predicted"]:
        ids={r["case_id"] for r in effects if r["scope"]==scope and r["both_nonempty_different"]}
        strata[scope+"_mask_different"]=subset(lambda c:c["case_id"] in ids)
        strata[scope+"_mask_not_different"]=subset(lambda c:c["case_id"] not in ids)
    indexed={(r["case_id"],r["method"]):r for r in effects}
    comparisons=[];changes=[];margin_summary={};pair_stats=[]
    pairs=[]
    for beta in [2.,1.]:
        b=str(beta);pairs.extend([("OS@"+b,"OF@"+b),("OS-target@"+b,"OF-target@"+b),("B3@"+b,"B2@"+b),("OS@"+b,"B3@"+b)])
    pairs.extend([("OS@2.0","O0"),("OF@2.0","O0")])
    for a,b in pairs:
        comparisons.extend(paired_endpoint(effects,a,b,e) for e in ["accuracy","pair_accuracy","oar","retention"])
        rows=[]
        for item in items:
            cid=item["case_id"];x=indexed[cid,a];y=indexed[cid,b]
            row={"case_id":cid,"guideline_id":item["guideline_id"],"pair_id":item["pair_id"],"method_a":a,"method_b":b,
                 "target_state":refs[cid]["target_applicability"],"a_prediction":x["prediction"],"b_prediction":y["prediction"],
                 "a_action":x["action"],"b_action":y["action"],"gold_action":refs[cid]["gold_action"],
                 "a_correct":x["correct"],"b_correct":y["correct"],"a_oar":x["oar"],"b_oar":y["oar"],
                 "prediction_changed":x["prediction"]!=y["prediction"],"a_minus_b_margin":x["margin"]-y["margin"],
                 "baseline_correct":x["baseline_correct"],"predicted_state_correct":x["predicted_state_correct"],
                 "conflict_disabled":x["conflict_disabled"],"mask_symmetric_difference":len(set(x["attention"]["targets"])^set(y["attention"]["targets"])),
                 "a_target_count":len(x["attention"]["targets"]),"b_target_count":len(y["attention"]["targets"]),
                 "both_masks_nonempty":bool(x["attention"]["targets"] and y["attention"]["targets"])}
            rows.append(row);changes.append(row)
        groups={"all":rows,"applicable":[r for r in rows if r["target_state"]=="applicable"],
                "inapplicable":[r for r in rows if r["target_state"]=="inapplicable"],
                "both_nonempty_different":[r for r in rows if r["mask_symmetric_difference"] and r["both_masks_nonempty"]],
                "inapplicable_both_nonempty_different":[r for r in rows if r["mask_symmetric_difference"] and r["both_masks_nonempty"] and r["target_state"]=="inapplicable"]}
        margin_summary[a+" minus "+b]={name:distribution([r["a_minus_b_margin"] for r in group]) for name,group in groups.items()}
    # Report each method's fixed-competitor delta margin, including real-change strata.
    delta_margins={}
    for method in names:
        rs=[r for r in effects if r["method"]==method]
        delta_margins[method]={"all":distribution([r["delta_margin"] for r in rs]),
            "applicable":distribution([r["delta_margin"] for r in rs if r["target_state"]=="applicable"]),
            "inapplicable":distribution([r["delta_margin"] for r in rs if r["target_state"]=="inapplicable"]),
            "both_nonempty_different":distribution([r["delta_margin"] for r in rs if r["both_nonempty_different"]])}
    a=metrics["OS@2.0"];b=metrics["OF@2.0"]
    behavior_signal=(a["accuracy"]["numerator"]>b["accuracy"]["numerator"] or
                     a["pair_accuracy"]["numerator"]>b["pair_accuracy"]["numerator"] or a["oar"]["numerator"]<b["oar"]["numerator"])
    safety_ok=a["oar"]["numerator"]<=b["oar"]["numerator"] and a["retention"]["numerator"]>=b["retention"]["numerator"]
    gate={"E0_passed":True,"E1_complete":True,"full_oracle_behavior_signal":behavior_signal,"full_oracle_safety_ok":safety_ok,
          "E2_offline_triggered":behavior_signal and safety_ok,"E3_feasibility_triggered":behavior_signal and safety_ok,
          "module1_modification_performed":False,"independent_validation_performed":False,
          "no_beta_reselection":True,"interpretation":"探索性开发集信号；不得视为独立验证或等效性证据"}
    archive(OUT/"case_effects.jsonl.gz",effects)
    csvwrite(OUT/"paired_comparisons.csv",comparisons);csvwrite(OUT/"prediction_changes.csv",changes)
    write(OUT/"oracle_factorial_metrics.json",{"scope":"same observed 40-pair development; Oracle contrasts are non-deployable",
        "primary_beta":2,"sensitivity_beta":1,"O0_reused_from":"revision3 B0","OS_reused_from":"revision3 B6, all masks exact and six representative GPU logit checks",
        "same_oracle_source":True,"all_metrics":{k:metrics[k] for k in names},"strata":strata,
        "delta_margin_distributions":delta_margins,"paired_margin_distributions":margin_summary,
        "paired_endpoint_comparisons":comparisons,"gates":gate,"config":read(ROOT/"configs/acre_oracle_switch.json"),
        "source_relevance_limitation":"Frozen Oracle marks all supplied rules relevant. This is the required common-reference protocol, not independently established semantic relevance."})
    write(OUT/"gate_decisions.json",gate)
    p=read(OUT/"resource_plan.json")
    started=p["full_start_utc"] if archive_mode else (OUT/"logs/full_start.txt").read_text().strip();ended=p["full_end_utc"] if archive_mode else (OUT/"logs/full_end.txt").read_text().strip()
    duration=(datetime.datetime.fromisoformat(ended.replace("Z","+00:00"))-datetime.datetime.fromisoformat(started.replace("Z","+00:00"))).total_seconds()
    p.update(full_start_utc=started,full_end_utc=ended,full_wall_seconds=duration,workers=done,
        actual_peak_allocated_gib=max(d["peak_allocated_gib"] for d in done),actual_peak_reserved_gib=max(d["peak_reserved_gib"] for d in done),
        actual_new_forwards=sum(d["model_forward_calls"] for d in done),identical_mask_reuses=sum(d["identical_mask_reuses"] for d in done),
        smoke_checks=read(OUT/"smoke_checks.json"),runtime_exit_status=p["runtime_exit_status"] if archive_mode else int((OUT/"logs/exit_status.txt").read_text()))
    write(OUT/"resource_plan.json",p)
    write(OUT/"final_code_fingerprints.json",code_fingerprints())
    print(json.dumps({"main":{m:{k:metrics[m][k] for k in ["accuracy","pair_accuracy","oar","retention"]} for m in ["O0","OF@2.0","OS@2.0","OF-target@2.0","OS-target@2.0"]},
                      "OS_minus_OF_margin":margin_summary["OS@2.0 minus OF@2.0"],"gates":gate},ensure_ascii=False))
if __name__=="__main__":main()
