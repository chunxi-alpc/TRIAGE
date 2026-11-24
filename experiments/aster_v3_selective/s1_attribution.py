"""Join existing first errors, rule structures and signed margins without path replay."""
import csv
import gzip
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from experiments.aster_closed_form.data import condition
from experiments.aster_cr_retrain_paired_v2_20261111.common import load_inputs
from experiments.aster_v3_selective.s1_verifier import definition, verify

H = Path.home()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/aster_v3_selective_20261124/s1"
RAW = H / "aster_cr_repair_v2_run_20261108/data"
ARCH = H / "aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111"
PR87 = H / "aster-m3-missing-controls-20261123/reports/aster_m3_missing_controls_20261123"
SJ = H / "aster-m3-joint-20261115/reports/aster_m3_joint_study_20261115"
E1 = H / "aster-path-error-20261123/reports/aster_path_error_20261123/results"


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as f:
        if path.suffix == ".csv":
            yield from csv.DictReader(f)
        else:
            yield from map(json.loads, f)


def write_csv(name, records):
    records = iter(records)
    first = next(records)
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, lineterminator="\n", fieldnames=first)
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(records)


def write_json(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")


def structure(p):
    op = p["op"]
    if op in {"and", "or", "not"}:
        return "复合条件"
    if op in {">", ">=", "<", "<="} or (op=="==" and type(p.get("value")) in (int,float)):
        return "数值条件"
    return "类别或状态条件"


TAGS = {
    "比较符或数值界值":r"[<>≥≤]|\d\s*[~～–]\s*\d",
    "测量对象或单位":r"\b(?:mmHg|mg/dL|g/dL|ng/mL|mmol/L|TOFR|ULN)\b|血压|浓度|评分",
    "时间":r"\b(?:current|within|prior|before|after|postpartum|preoperative|duration|during)\b|已持续|之前|之后|当前|术后|术前|产后|孕期|时间",
    "逻辑、否定或例外":r"\b(?:and|or|not|except|neither|without)\b|且|并且|或|而非|除外|尚未|未使用",
    "缺失":r"\b(?:unknown|unavailable|missing)\b|未知|不明|缺失",
}
FLAGS = {
    ("mega:train_val:2744:10","G1"):("数值与病例文字结论冲突", (65,74), (79,83)),
    ("mega:train_val:1857:4","G3"):("数值与归档分支标签冲突", (336,368), (380,393)),
}


def relation(s, u):
    return "positive" if u>s else "negative" if u<s else "zero"


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {r["case_id"]:r for r in rows(RAW/"path_cases.jsonl") if r["split"]=="fit"}
    srcs = sorted({r["source_id"] for r in cases.values()})
    rules = {r["group_id"]:r for r in rows(RAW/"rules.jsonl") if r["split"]=="fit"}
    nodes = {r["group_id"]:r for r in rows(RAW/"nodes.jsonl") if r["group_id"] in rules}
    rule_index = {}
    for name, r in rules.items():
        n = nodes[name]
        if r["original_condition"] == condition(n):
            rule_index[(n["guideline_id"], n["gid"], r["original_condition"],n["source_id"])] = r
    old = {(r["case_id"],r["gid"]):r for r in rows(E1/"qualitative_cases.csv")}
    e1 = {(r["case_id"],r["variant"],r["first_error_gid"],r["method"]):r
          for r in rows(E1/"first_errors.csv") if r["method"] in {"raw","pair_only"}}
    transfers = {(r["case_id"],r["variant"]):r for r in rows(E1/"original_modified_paths.csv")}
    bounds = {r["fold"]+"::"+r["view_id"]:r for r in rows(SJ/"diagnostics/correction_bounds.jsonl")}
    sj = {r["case_id"]:r for r in rows(SJ/"evaluation/natural_first_error.jsonl.gz") if r["comparison"]=="R1 minus R0"}
    observed = {}
    for r in rows(PR87/"A_first_error_boundaries.jsonl.gz"):
        if (r["lhs"],r["rhs"]) != ("J_zero","R0"):
            continue
        cid,var = r["case_id"],r["variant"]
        for which, method in [("before_first_error","R0"),("after_first_error","J_zero")]:
            z = r[which]
            if z is None:
                continue
            key = (cid,var,z["gid"],z["view_id"])
            item = observed.setdefault(key, {"z":z,"methods":[],"transition":r["transition"]})
            item["methods"].append(method)

    inputs=load_inputs(ARCH)
    output = []
    for (cid,var,gid,uid), item in sorted(observed.items()):
        z = item["z"]
        c=cases[cid]
        q=condition(c,gid)
        prompt=inputs[uid.split("::",1)[1]]["prompt"]
        patient=c["patient"]
        if prompt.count(patient)!=1 or q not in prompt:
            raise ValueError("Archived prompt text differs at "+uid)
        patient_start=prompt.index(patient)
        rc=rule_index.get((c["guideline_id"],gid,q,c["source_id"]))
        old_case=old.get((cid,gid))
        memberships=item["methods"]
        if "R0" in memberships:
            ref=e1[(cid,var,gid,"raw")]
            if int(ref["gold_label"])!=z["label"] or abs(float(ref["score"])-z["before_score"])>1e-5 or ref["condition"]!=q:
                raise ValueError("E1 vs PR87 at "+uid)
        if var=="original" and "R0" in memberships:
            if sj[cid]["first_error_view_id"]!=uid:
                raise ValueError("S/J vs PR87 at "+uid)
        lp=bounds.get(uid)
        if lp and (lp["case_id"]!=cid or lp["gold_label"]!=z["label"]):
            raise ValueError("LP join at "+uid)
        pred,reason,span=verify(q,c["patient"])
        if old_case:
            category=old_case["category"]
            evidence="historical_case_reading"
        elif (cid,gid) in FLAGS:
            category,a,b=FLAGS[cid,gid]
            if not all(0<=lo<hi<=len(patient) for lo,hi in [a,b]) or not re.search(r"\d",patient[slice(*a)]):
                raise ValueError("missing evidence at "+cid)
            evidence="new_text_discrepancy_pending_review"
        elif pred is not None:
            category="可复核的测量值与阈值判断" if pred==z["label"] else "数值与归档分支标签冲突"
            evidence="deterministic_evidence" if pred==z["label"] else "new_text_discrepancy_pending_review"
        else:
            category="无法判定"
            evidence="pending_review"
        trans=transfers.get((cid,var))
        r0=z["before_score"]
        uz=z["after_score"]
        output.append(dict(source_id=c["source_id"],case_id=cid,variant=var,gid=gid,view_id=uid,
            guideline_id=c["guideline_id"],methods="|".join(memberships),gold=z["label"],R0=r0,J_zero=uz,
            R0_gold_margin=z["before_gold_margin"],J_zero_gold_margin=z["after_gold_margin"],
            delta_J=uz-r0,delta_direction=relation(r0,uz),J_zero_path_transition=item["transition"],
            pair_only_is_first_error=(cid,var,gid,"pair_only") in e1,
            E1_pair_clean_to_variant=trans["pair_only_clean_to_modified"] if trans else "",
            R0_clean_to_variant=trans["raw_clean_to_modified"] if trans else "",
            SJ_R0_natural_uid_join=var=="original" and sj[cid]["first_error_view_id"]==uid,
            historical_R1_LP_reachable=("" if not lp else not lp["impossible_at_delta_shared_b0"]),
            rule_type=structure(rc["predicate"]) if rc else "未知",rule_id=rc["group_id"] if rc else "",
            exact_archived_fact_case=bool(rc and rc["case_id"]==cid),
            condition_tags="|".join(tag for tag,pat in TAGS.items() if re.search(pat,q,re.I)) or "其他或未识别",
            attribution=category,evidence_status=evidence,
            patient_verified_in_variant=True,patient_start_in_prompt=patient_start,
            text_start=(span[0] if span else FLAGS[cid,gid][1][0] if (cid,gid) in FLAGS else ""),
            text_end=(span[1] if span else FLAGS[cid,gid][1][1] if (cid,gid) in FLAGS else ""),
            conflict_start=FLAGS[cid,gid][2][0] if (cid,gid) in FLAGS else "",
            conflict_end=FLAGS[cid,gid][2][1] if (cid,gid) in FLAGS else "",
            prior_example=bool(old_case),observation_reason=reason))
    if hasattr(inputs, "close"):
        inputs.close()
    write_csv("first_errors.csv",output)
    table = []
    for level,subset in [('natural',[x for x in output if x["variant"]=="original"]),('modified',[x for x in output if x["variant"]!="original"])]:
        for source in ["ALL"]+srcs:
            cohort = [x for x in subset if source=="ALL" or x["source_id"]==source]
            for dim in ["attribution","rule_type","evidence_status"]:
                counts=Counter(x[dim] for x in cohort)
                for value in sorted({x[dim] for x in output}):
                    table.append(dict(level=level,source=source,dimension=dim,value=value,n=counts[value],total=len(cohort)))
    write_csv("source_counts.csv",table)
    summary={}
    for key,part in [("all",output),("natural",[x for x in output if x["variant"]=="original"]),("modified",[x for x in output if x["variant"]!="original"])]:
        summary[key]=dict(n=len(part),cases=len({x["case_id"] for x in part}),sources=len({x["source_id"] for x in part}),unique_case_gid=len({(x["case_id"],x["gid"]) for x in part}),
            attribution=dict(Counter(x["attribution"] for x in part)),evidence=dict(Counter(x["evidence_status"] for x in part)),structure=dict(Counter(x["rule_type"] for x in part)),
            LP_join=sum(x["historical_R1_LP_reachable"]!="" for x in part),R0=sum("R0" in x["methods"] for x in part),J_zero=sum("J_zero" in x["methods"] for x in part))
    summary.update(archived_rules=len(rules),archived_qualified_blind_reviews=0,archived_case_readings=len(old),LP_total=len(bounds),fit_sources=len(srcs),
        s_j_natural_paths=len(sj),e1_natural_modified_pairs=len(transfers),gpu_hours=0,seconds=time.perf_counter()-start)
    write_json("s1_summary.json",summary)

if __name__ == "__main__":
    main()
