#!/usr/bin/env python3
"""Export frozen A1 T/F/U rule predictions as module-two research signals.

No model inference, gold labels, confidence scores, evidence gates or selection
on accuracy. Preserve every input case in the output, including no-signal ones.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

STATES = {"T": 1, "F": -1, "U": 0}
SOURCE = "care.module1.semantic_a1.frozen"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("case_id"), str):
                    raise ValueError(f"Missing case_id in {path}:{number}")
                yield value


def index_predictions(predictions):
    indexed = {}
    for prediction in predictions:
        cid = prediction["case_id"]
        if cid in indexed:
            raise ValueError(f"Duplicate A1 prediction case_id: {cid}")
        if prediction.get("arm") != "A1":
            raise ValueError(f"Non-A1 prediction for {cid}; expected frozen A1")
        indexed[cid] = prediction
    return indexed


def export(cases, predictions):
    """Return (case records, accounting). Never consult reference labels."""
    predicted = index_predictions(predictions)
    case_ids = set()
    records = []
    counts = Counter()
    for case in cases:
        cid = case["case_id"]
        if cid in case_ids:
            raise ValueError(f"Duplicate input case_id: {cid}")
        case_ids.add(cid)
        original_rules = case.get("rules", [])
        if not isinstance(original_rules, list) or not original_rules:
            raise ValueError(f"Case {cid} has no rules")
        pred = predicted.get(cid)
        pred_rules = pred.get("rules", []) if pred and pred.get("status") == "ok" else []
        if not isinstance(pred_rules, list):
            pred_rules = []
        ids = Counter(r.get("rule_id") for r in pred_rules if isinstance(r, dict))
        by_id = {r.get("rule_id"): r for r in pred_rules if isinstance(r, dict)}
        rows = []
        for source_rule in original_rules:
            rid = source_rule.get("rule_id")
            if not isinstance(rid, str) or not rid:
                raise ValueError(f"Case {cid} has an invalid rule ID")
            action = source_rule.get("action")
            action = action if isinstance(action, str) else ""
            got = by_id.get(rid, {})
            raw_state = got.get("state")
            state = raw_state if ids[rid] == 1 and raw_state in STATES else None
            sign = STATES.get(state)
            usable = sign in (-1, 1) and bool(action.strip())
            rows.append({
                "rule_id": rid,
                "state": state,
                "sign": sign,
                "action_text": action,
                "action_requires_mapping_in_final_prompt": bool(action.strip()),
                "usable_signal": usable,
                "optional_evidence_quote": got.get("evidence_quote", "") if state else "",
                "technical_failure": state is None,
            })
            counts[state if state else "technical"] += 1
        has_signal = any(row["usable_signal"] for row in rows)
        if pred is None:
            status = "missing_prediction"
        elif pred.get("status") != "ok":
            status = "technical_failure"
        elif any(row["technical_failure"] for row in rows):
            status = "partial_technical_failure"
        else:
            status = "ok"
        records.append({
            "case_id": cid,
            "split": case.get("split"),
            "family_id": case.get("family_id"),
            "signal_source": SOURCE,
            "prediction_status": status,
            "eligible_for_module2_signal_study": has_signal,
            "rules": rows,
        })
        counts["cases"] += 1
        counts["eligible_cases" if has_signal else "no_signal_cases"] += 1
        if status != "ok":
            counts["cases_with_technical_failure"] += 1
    unknown_ids = set(predicted) - case_ids
    if unknown_ids:
        raise ValueError(f"A1 prediction records not present in the input cases: {len(unknown_ids)}")
    summary = {
        "signal_source": SOURCE,
        "policy": "T:+1, F:-1, U:0; invalid/missing:null (no edit). At least one T/F with an action text makes a case signal-eligible.",
        "reference_labels_used": False,
        "confidence_or_evidence_gate": False,
        "case_selection_uses_gold_or_downstream_outcomes": False,
        "all_cases_retained": True,
        "action_token_spans": "Resolve by rule_id in the final module-two prompt and tokenizer; action_text is not a token position.",
        "counts": {key: counts[key] for key in (
            "cases", "eligible_cases", "no_signal_cases", "cases_with_technical_failure",
            "T", "F", "U", "technical")},
    }
    return records, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True, help="Same case inputs used for the A1 predictions")
    ap.add_argument("--predictions", type=Path, required=True, help="Frozen A1 predictions.jsonl")
    ap.add_argument("--output", type=Path, required=True, help="New JSONL path; will not overwrite")
    args = ap.parse_args()
    records, summary = export(read_jsonl(args.cases), read_jsonl(args.predictions))
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        ap.error("Output or summary path already exists; choose fresh paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": str(summary_path), **summary["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
