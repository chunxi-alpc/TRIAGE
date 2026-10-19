#!/usr/bin/env python3
"""Summarize actual completed inference and package runtime logs."""
import datetime, hashlib, json, pathlib, tarfile
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    metrics=json.loads((HERE/"analysis/metrics.json").read_text())
    completion=json.loads((HERE/"analysis/completion.json").read_text())
    assert completion["complete"], completion
    provenance=json.loads((HERE/"execution_manifest.json").read_text())
    matches={p:sha(REPO/p)==v for p,v in provenance["frozen_inputs_and_code_sha256"].items()}
    assert all(matches.values()), matches
    runs=[]; calls=0; seconds=0; retries=0; invalid=0
    for path in sorted((HERE/"results").glob("*/manifest.json")):
        m=json.loads(path.read_text()); assert m["status"]=="completed", str(path)
        rows=[json.loads(s) for s in (path.parent/"predictions.jsonl").read_text().splitlines()]
        assert len(rows)==m["total_cases"] and len({r["case_id"] for r in rows})==len(rows)
        attempts=[a for r in rows for a in r["attempts"] if not a.get("reused",False)]
        calls+=len(attempts); seconds+=sum(a["generation_seconds"] for a in attempts)
        retries+=sum(a.get("attempt",1)>1 for a in attempts)
        invalid+=sum(r["status"]!="ok" for r in rows)
        runs.append({"run":path.parent.name,"cases":len(rows),"added_calls":len(attempts),"gpu":m["cuda_visible_devices"],"elapsed_seconds":m["elapsed_seconds"],"peak_allocated_gib":m["peak_allocated_gib"],"peak_reserved_gib":m["peak_reserved_gib"],"predictions_sha256":sha(path.parent/"predictions.jsonl")})
    artifacts=HERE/"artifacts"; artifacts.mkdir(exist_ok=True)
    archive=artifacts/"runtime_logs.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        for p in sorted(HERE.rglob("*.log")):
            tar.add(p,arcname=str(p.relative_to(HERE)))
        if (HERE/"logs/operations.txt").exists():tar.add(HERE/"logs/operations.txt",arcname="logs/operations.txt")
    report={"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"all_results_complete":True,"selected":completion["selected"],"added_gpu_case_requests":calls,"added_gpu_generation_seconds":seconds,"format_retries":retries,"failed_cases":invalid,"historical_A1_reused_cases":360,"frozen_hash_checks":matches,"runs":runs,"log_archive_sha256":sha(archive),"evaluated_rule_predictions":sum(m["rules"] for m in metrics.values()),"time_note":"Generated-seconds are summed device batch time, not wall-clock duration or online latency."}
    (HERE/"completion_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("all_results_complete","selected","added_gpu_case_requests","format_retries","failed_cases")}))
if __name__=="__main__":main()
