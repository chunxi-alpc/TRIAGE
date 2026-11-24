"""Aggregate S1 per distinct case and node for review, treating variants as repeats."""
import csv
from collections import Counter,defaultdict
from experiments.aster_v3_selective.s1_attribution import OUT, RAW, rows, write_csv

def main():
    groups=defaultdict(list)
    for row in rows(OUT/"first_errors.csv"):
        groups[(row["case_id"],row["gid"])].append(row)
    queue=[]
    counts=defaultdict(Counter)
    sources=sorted({r["source_id"] for r in rows(RAW/"sources.jsonl") if r["split"]=="fit"})
    for (cid,gid),items in sorted(groups.items()):
        r=items[0]
        if len({(x["evidence_status"],x["attribution"]) for x in items})!=1:
            raise ValueError("Inconsistent underlying evidence across guideline-action revisions")
        for source in ["ALL",r["source_id"]]:
            counts[source][r["evidence_status"]]+=1
        queue.append(dict(source_id=r["source_id"],case_id=cid,guideline_id=r["guideline_id"],gid=gid,
            case_file="data/path_cases.jsonl",condition_field="guideline.groups[gid].rules[0].condition",
            patient_field="patient",definition_field="definitions"))
    write_csv("blinded_review_index.csv",queue)
    stages=["historical_case_reading","deterministic_evidence","new_text_discrepancy_pending_review","pending_review"]
    write_csv("case_node_counts.csv",(
        dict(source_id=source,evidence_status=stage,n=counts[source][stage],total=sum(counts[source].values()))
        for source in ["ALL"]+sources for stage in stages))
if __name__=="__main__":
    main()
