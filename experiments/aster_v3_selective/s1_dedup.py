"""Compare V3 rule, patient and case identities to the named archived protocols."""
import csv
import gzip
import json
import subprocess
import time
from pathlib import Path

from experiments.aster_v3_selective.s1_attribution import H, OUT, RAW, ROOT, rows, write_json
from experiments.aster_v3_selective.s1_verifier import HB, SBP


def git_rows(ref, filename):
    data=subprocess.check_output(["git","show",ref+":"+filename],cwd=ROOT)
    return [json.loads(s) for s in data.splitlines() if s]


def main():
    start=time.perf_counter()
    with gzip.open(OUT/"s3_nodes.csv.gz","rt") as f:
        targets=[r for r in csv.DictReader(f) if r["rule"]!="other"]
    ids={r["case_id"] for r in targets}
    patients={r["patient"] for r in rows(RAW/"path_cases.jsonl") if r["case_id"] in ids}
    def counts(records):
        rule_hits={q:0 for q in [HB,SBP]}
        exact_case=exact_patient=0
        for r in records:
            serialized=json.dumps(r,ensure_ascii=False)
            for q in rule_hits:
                rule_hits[q]+=int(q in serialized)
            exact_case+=int(r.get("id",r.get("case_id",r.get("sample_id"))) in ids)
            exact_patient+=int(r.get("patient") in patients)
        return dict(rows=len(records),case_ids_matching=exact_case,full_rule_string_presence=rule_hits,
                    literal_patient_field_matches=exact_patient,rows_with_patient_field=sum("patient" in r for r in records))

    private=H/"cs44_explicit_check_20261024/private"
    evaluations=list(rows(private/"evaluation.jsonl"))
    case_pool=H/"qwen800_methods_20261023/private/cases.jsonl"
    prior_ids={r["id"] for r in evaluations}
    old=[r for r in rows(case_pool) if r["id"] in prior_ids]
    requests=[r for i in [0,1] for r in rows(private/f"requests{i}.jsonl")]
    completed=[r for i in [0,1] for r in rows(private/f"outputs{i}.jsonl")]
    output=[dict(pr=29,protocol="Qwen3-8B explicit two-branch choice A/B/uncertain then final action; two threshold states and two rule orders",
        files=[str(case_pool),str(private/"evaluation.jsonl"),str(private/"requests{0,1}.jsonl"),str(private/"outputs{0,1}.jsonl")],
        source_identity="MedGuide case IDs, disease and source row; separate namespace from MEGA guideline source IDs",
        case_ids=sorted(prior_ids),cases=counts(old),requests=counts(requests),evaluation_rows=len(evaluations),completed_rows=len(completed),
        same_request_keys=sum(r["cell_key"] in {c["cell_key"] for c in completed} for r in requests),same_protocol_completed=0)]
    groups=[
        (4,"origin/codex/care-evidence-v4-20261018","care_module1_v4",
         ["data/rules_manifest","data/labels","data/model/predictions"],"V04 rule ID, source row and anchor; rule applicability/uncertain with evidence"),
        (5,"origin/codex/care-module1-v5","care_module1_v5",
         ["data/rules_v5","data/labels_public","runs/v5_20261018/parser_v2_final/predictions_public"],
         "V5 rule and sample IDs; multi-condition applicability, entailment, uncertainty and program/numeric adjudication"),
        (12,"origin/pr12","care_module1_semantic_v8",
         ["data/semantic_dev_inputs","data/semantic_confirmation_inputs","data/new72_cases"]+
         [f"results/{arm}/predictions" for arm in ["dev_A0","dev_A1","confirmation_A0","confirmation_A1","new72_A0","new72_A1","new72_A2"]],
         "V8 semantic applicability A0/A1/A2; development, confirmation, new72; dataset-specific case and family IDs"),
        (13,"origin/codex/care-module1-v8-1","care_module1_v8_1",["data/natural72_cases"]+
         [f"results/{arm}/predictions" for arm in ["dev_B1","dev_B2","confirmation_B1","confirmation_B2","new72_B1","new72_B2","natural72_A1","natural72_B1","natural72_B2"]],
         "V8.1 applicability A1/B1/B2; reuses V8 dev/confirmation/new72 plus natural72; case and family IDs")]
    for pr,ref,pack,files,description in groups:
        records={}
        rule_ids=set()
        case_ids=set()
        fam=set()
        datasets=set()
        for short in files:
            name=f"experiments/{pack}/{short}.jsonl"
            data=git_rows(ref,name)
            records[name]=counts(data)
            rule_ids.update(str(x["rule_id"]) for x in data if "rule_id" in x)
            case_ids.update(x.get("case_id",x.get("sample_id")) for x in data if "case_id" in x or "sample_id" in x)
            fam.update(x["family_id"] for x in data if "family_id" in x)
            datasets.update(x["dataset_id"] for x in data if "dataset_id" in x)
        output.append(dict(pr=pr,ref=ref,commit=subprocess.check_output(["git","rev-parse",ref],cwd=ROOT,text=True).strip(),
            protocol=description,files=records,unique_case_or_sample_ids=len(case_ids),first_last_ids=[min(case_ids),max(case_ids)],
            rule_ids=sorted(rule_ids),family_ids=sorted(fam),dataset_ids=sorted(datasets),same_protocol_completed=0))
    old_rules=list(rows(RAW/"rules.jsonl"))
    archived=[]
    for r in targets:
        for old in old_rules:
            if old["case_id"]==r["case_id"] and old["original_condition"] in {HB,SBP} and old["group_id"].endswith("-"+r["gid"]):
                archived.append(dict(case_id=r["case_id"],gid=r["gid"],archive_group=old["group_id"],archived_program=old["method"],
                    archived_program_label=old["program_label"],V3_verifier_decision=int(r["verifier"])))
    write_json("s3_dedup.json",dict(target_eligible_nodes=len(targets),target_distinct_case_ids=len(ids),target_source_ids=sorted({r["source_id"] for r in targets}),
        decision_identity="original current patient + complete condition; frozen V3 measurement-bound verifier or first-unit baseline with R0 fallback, cached Qwen3-8B MEGA natural-node scoring",
        comparison="Exact rule text scanned within serialized archive inputs/rules, case identifier equality, patient equality where the archive exposes the full patient field, and differing target/decision protocol. A missing patient field is counted separately.",
        historical=output,prior_same_protocol_predictions=0,same_case_and_condition_prior_MEGA_rule=archived,
        MEGA_rule_path=str(RAW/"rules.jsonl"),MEGA_rule_role="Earlier patient fact/predicate output provides one corroborating record; V3 adds raw-text extraction with abstention and R0 comparator.",
        cost=dict(seconds=time.perf_counter()-start,gpu_hours=0,model_forward=0,old_replays=0)))
if __name__=="__main__":
    main()
