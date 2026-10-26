"""CPU-only replay audit of the actual source-rule extraction v2 run."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

OUT = Path(__file__).resolve().parent
P = OUT.parent
sys.path.insert(0, str(P))
from mega_trainval_reader import iter_method_tasks
from source_anchored_rules import parse_source_anchored_rules, occurrences, VERSION
from source_rule_prompt import INSTRUCTION

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    runtime = P / "runtime/source_rule_extraction_v2"
    manifest_path = P / "runtime/source_rule_extraction_v2_job/run_manifest.json"
    run = json.loads(manifest_path.read_text())
    protocol = json.loads((runtime / "protocol.json").read_text())
    assert run["process_state"] == "exited" and run["exit_code"] == 0
    assert protocol["completed_cases"] == protocol["expected_cases"] == 16
    assert protocol["generation_errors"] == 0
    assert protocol["prompt_sha256"] == hashlib.sha256(INSTRUCTION.encode()).hexdigest()
    assert protocol["parser_version"] == VERSION
    args = run["module_args"]
    value = lambda key: args[args.index("--" + key) + 1]
    frozen_path = Path(value("input-manifest"))
    frozen = json.loads(frozen_path.read_text())
    assert frozen["split"] == "train_val"
    tasks = list(iter_method_tasks(Path(value("dataset-root")), frozen))
    assert len(tasks) == 16
    records, rule_rows, statuses, failures = [], [], Counter(), Counter()
    manual = {
        "86f0bf4": {"rule_ids": ["R0001","R0002","R0003","R0004","R0005","R0006","R0007","R0008","R0009","R0010","R0013","R0020"],
                     "finding": "action_is_evidence_grade_not_patient_action"},
        "8e4c7da": {"rule_ids": ["R0006","R0007","R0008","R0009","R0010","R0011"],
                     "finding": "condition_is_document_taxonomy_heading_not_patient_predicate"},
        "cce3a84": {"rule_ids": ["R0001","R0002","R0003","R0004","R0006"],
                     "finding": "training_or_examination_requirement_instead_of_current_case_imaging_quality_rule"},
    }
    for identity, task in tasks:
        ghash = hashlib.sha256(task.guideline.encode()).hexdigest()
        source = runtime / (ghash + ".private.json")
        saved = json.loads(source.read_text())
        assert saved["record"]["sample_id"] == identity.sample_id
        assert saved["record"]["guideline_sha256"] == ghash
        parsed, audit = parse_source_anchored_rules(saved["generation"]["text"], task.guideline)
        assert json.loads(json.dumps([asdict(r) for r in parsed], ensure_ascii=False)) == saved["rules"]
        assert audit == saved["item_audit"]
        assert len(parsed) == saved["record"]["rule_count"]
        statuses.update(item["status"] for item in audit)
        failures.update(item.get("source_span_reconstruction_error", item.get("error", "unknown"))
                        for item in audit if item["status"] == "rejected")
        section_review = manual.get(identity.sample_id[:7])
        for rule in parsed:
            assert task.guideline[rule.start:rule.end] == rule.rule_quote
            anchors = [rule.condition_quote, rule.action_quote, *rule.exception_quotes]
            assert all(anchor and anchor in rule.rule_quote and anchor in task.guideline for anchor in anchors)
            assert rule.action_quote not in rule.condition_quote
            unique_occurrences = [len(occurrences(task.guideline, q)) for q in anchors]
            flags = []
            if re.fullmatch(r"(?:[1-3][ABC]|GPP)", rule.action_quote.strip(), re.I):
                flags.append("action_is_evidence_grade")
            if re.search(r"(?:考试|题库|培训|观摩|完成\s*\d+\s*例|training|examination|curriculum)", rule.condition_quote+" "+rule.action_quote, re.I):
                flags.append("training_or_exam_keywords_present_not_automatic_exclusion")
            if section_review and rule.rule_id in section_review["rule_ids"]:
                flags.append("manual_review:" + section_review["finding"])
            if rule.end - rule.start > 4096:
                flags.append("source_span_over_4096_characters")
            if max(unique_occurrences) > 1:
                flags.append("nonunique_anchor_occurrence")
            rule_rows.append({
                "sample_id": identity.sample_id, "guideline_sha256": ghash, "rule_id": rule.rule_id,
                "source_span": [rule.start, rule.end], "span_characters": rule.end-rule.start,
                "span_sha256": hashlib.sha256(rule.rule_quote.encode()).hexdigest(),
                "condition_characters": len(rule.condition_quote), "action_characters": len(rule.action_quote),
                "exception_count": len(rule.exception_quotes), "anchor_occurrence_counts": unique_occurrences,
                "duplicate_passage_occurrences": rule.duplicate_quote_occurrences, "flags": flags,
                "source_anchors_exact": True, "patient_applicability_verified": False,
            })
        records.append({
            "sample_id": identity.sample_id, "guideline_sha256": ghash, "saved_file_sha256": digest(source),
            "rule_count": len(parsed), "output_tokens": saved["generation"].get("output_tokens"),
            "output_at_budget": saved["generation"].get("output_tokens") == protocol["max_new_tokens"],
            "max_span_characters": max((r.end-r.start for r in parsed), default=0),
            "array_closed": all(item.get("array_closed",False) for item in audit) if audit else None,
            "audit_status_counts": dict(Counter(item["status"] for item in audit)),
            "elapsed_seconds": saved["record"]["wall_seconds"],
            "forward_calls": saved["record"]["new_forward_calls"],
        })
    assert sum(row["forward_calls"] for row in records) == protocol["forward_calls"] == run["forward_completed"]
    lengths = [row["span_characters"] for row in rule_rows]
    flag_counts = Counter(flag for row in rule_rows for flag in row["flags"])
    # These are audit summaries, not a clinical quality estimate or new filtering.
    result = {
        "status": "complete_verified",
        "scope": "CPU replay of all 16 actual extraction outputs, exact source anchors and provenance; semantic suitability remains incomplete",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "new_gpu_forwards": 0,
        "test_or_reference_labels_read": False, "source_text_published": False,
        "guidelines": 16, "cases_with_rules": sum(row["rule_count"] > 0 for row in records),
        "total_rules": len(rule_rows), "cases_at_output_budget": sum(row["output_at_budget"] for row in records),
        "replayed_rule_sets_exactly_equal": 16, "all_retained_anchors_exact": True,
        "span_characters": {"minimum": min(lengths), "median": statistics.median(lengths),
                            "maximum": max(lengths), "above_1000": sum(v>1000 for v in lengths),
                            "above_4096": sum(v>4096 for v in lengths)},
        "rules_with_exceptions": sum(row["exception_count"]>0 for row in rule_rows),
        "rules_with_nonunique_anchors": sum("nonunique_anchor_occurrence" in row["flags"] for row in rule_rows),
        "rules_with_repeated_passage": sum(row["duplicate_passage_occurrences"]>1 for row in rule_rows),
        "audit_status_counts": dict(statuses), "rejection_reasons": dict(failures), "flag_counts": dict(flag_counts),
        "reviewer_type": "AI assistant; not a human clinical expert",
        "human_expert_review_completed": False,
        "manual_review_scope": "AI-assisted targeted review of evidence-grade actions, taxonomy headings, training/exam cases and longest intervals; not exhaustive expert annotation",
        "clinical_suitability": "not_verified; observed semantic counterexamples prevent direct adoption",
        "recommendation": "Keep originals. Bind condition/action/exception inside the same source paragraph/table row, explicitly reject evidence-grade-only actions and document headings, and validate extraction on development before attention. Do not treat longer recovered spans as evidence of higher quality or create gold test rules.",
        "original_coverage_note": "Old actual extraction covered 6/16. Prior CPU replay of OLD generated outputs covered 12/16. This NEW model-generated extraction covers 11/16; these are different artifacts.",
        "run_manifest_sha256": digest(manifest_path), "input_manifest_sha256": digest(frozen_path),
        "protocol_sha256": digest(runtime / "protocol.json"), "auditor_sha256": digest(Path(__file__)),
        "current_source_sha256": {name: digest(P/name) for name in ["run_source_rule_extraction.py","source_anchored_rules.py","source_rule_prompt.py","clinical_core.py","mega_trainval_reader.py"]},
        "source_binding_limit": "Current prompt hash equals run protocol; retained parsed rules/items exactly replay. Custom runner source hashes are recorded here at audit time, not newly asserted to have been captured at inference start.",
        "run_costs": {k:run[k] for k in ["forward_completed","elapsed_seconds","peak_allocated_gib","peak_reserved_gib","gpu_physical"]},
        "cases": records,
    }
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    (OUT / "rule_metadata.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rule_rows))
    print(json.dumps({k:result[k] for k in ["status","guidelines","cases_with_rules","total_rules","cases_at_output_budget","span_characters","rules_with_exceptions","rules_with_nonunique_anchors","audit_status_counts","flag_counts"]},ensure_ascii=False))

if __name__ == "__main__":
    main()
