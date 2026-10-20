#!/usr/bin/env python3
"""Recompute frozen-cache evidence without running either model module.

All reported accuracy/coverage values concern saved module-one signals.  This
does not create downstream model predictions, recover missing patient text, or
join unrelated datasets by row order.  The input Git blobs are pinned to the
audited repository commit; a changed input fails before results are written.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from frozen_adapter import (
    ADAPTER_MODES, LABEL_SIGNS, FrozenSignalAdapter, FrozenSignalError,
    exact_key, unique_index,
)


SOURCE_COMMIT = "7557d3150214a3894f2d340df6d9964de75efce1"
M1_DIR = "experiments/care_module1_v6/data"
SIGNED_DIR = "experiments/care_module2_signed_20261019"
PINNED_BLOBS = {
    f"{M1_DIR}/dataset_protocol.json": "aae96ffeedc643abfa8f8394191045b4507bb4b5",
    f"{M1_DIR}/labels_public.jsonl": "ea5b077c4911d906f017878ec20264c177c07535",
    f"{M1_DIR}/new_100_predictions_v6.jsonl": "277da3631d988c9ec552b6c004c5ded511398054",
    f"{M1_DIR}/new_labels_100.jsonl": "9e0e196ba582fba6986c5333822fd7539170c8cf",
    f"{M1_DIR}/new_rules_100.jsonl": "539916dfd5534d27a17bdb3a39f6b73dc2dd7aa0",
    f"{M1_DIR}/new_set_manifest.json": "5a9300cb1c0eeea3d09be31bd1211237b9cb0558",
    f"{M1_DIR}/old_192_predictions_v6.jsonl": "672ab927651889aca5bbae27cc1737e090d763e4",
    f"{M1_DIR}/rules_v5.jsonl": "97777b2a86ca6859b37703aa6a39c61f28d42c83",
    f"{SIGNED_DIR}/results/analysis/selected_predictions.jsonl.gz": (
        "b550f55163a9097331948000dce6a95c52ec75e0"
    ),
    f"{SIGNED_DIR}/results/fold0/provenance.json": (
        "ebeec6c2de3303b4bdd7ae50ed15ee2cbf538773"
    ),
}
DATASETS = {
    "old192": ("old_192_predictions_v6.jsonl", "labels_public.jsonl", "rules_v5.jsonl"),
    "new100": ("new_100_predictions_v6.jsonl", "new_labels_100.jsonl", "new_rules_100.jsonl"),
}
LABELS = ("T", "F", "U")
POLICY_FIELDS = {
    "source_target", "source_mode", "source_label", "decidability_target",
    "decidability_label", "can_decide", "mode_target", "v_src", "v_det",
}
RAW_INPUT_FIELDS = {"patient", "sentences", "query", "question", "prompt"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty input")
    return rows


def fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": hashlib.sha1(
            f"blob {len(data)}\0".encode("ascii") + data
        ).hexdigest(),
    }


def verify_inputs(repo: Path) -> dict[str, Any]:
    fingerprints = {}
    for relative, expected in PINNED_BLOBS.items():
        record = fingerprint(repo / relative)
        record["expected_git_blob_sha1"] = expected
        record["matches_pinned_commit"] = record["git_blob_sha1"] == expected
        if not record["matches_pinned_commit"]:
            raise ValueError(f"pinned input changed: {relative}")
        fingerprints[relative] = record
    return fingerprints


def label_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for expected, actual in zip(gold, predicted, strict=True):
        if expected not in LABELS or actual not in LABELS:
            raise ValueError("invalid T/F/U label")
        confusion[expected][actual] += 1
    correct = sum(confusion[label][label] for label in LABELS)
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        per_class[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": sum(confusion[label].values()),
        }
    return {
        "n": len(gold), "correct": correct, "accuracy": correct / len(gold),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(LABELS),
        "confusion_rows_gold_columns_prediction": confusion,
        "per_class": per_class,
    }


def count_labels(values) -> dict[str, int]:
    counts = Counter(values)
    return {label: counts[label] for label in LABELS}


def rule_audit(rules: list[dict[str, Any]], expected_ids: set[str]) -> dict[str, Any]:
    indexed = {}
    for rule in rules:
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id or rule_id in indexed:
            raise ValueError("invalid or duplicate rule_id in rule manifest")
        indexed[rule_id] = rule
    if set(indexed) != expected_ids:
        raise ValueError("rule manifest and frozen case rule IDs differ")
    entries = []
    for rule_id in sorted(indexed):
        rule = indexed[rule_id]
        if "action" not in rule:
            action_status = "field_missing"
        elif rule["action"] is None:
            action_status = "null"
        elif rule["action"] in ("", [], {}):
            action_status = "empty"
        else:
            action_status = "present"
        entries.append({
            "rule_id": rule_id, "action_status": action_status,
            "condition_count": len(rule.get("conditions", [])),
            "exception_count": len(rule.get("exceptions", [])),
        })
    return {
        "n_rules": len(entries),
        "action_status_counts": dict(Counter(r["action_status"] for r in entries)),
        "rules": entries,
        "nonempty_action_field_count": sum(r["action_status"] == "present" for r in entries),
        "any_nonempty_action_field_present": any(r["action_status"] == "present" for r in entries),
        "independent_clinical_action_gold_verified": False,
        "interpretation": (
            "Research predicates; absent action fields do not define clinical action targets. "
            "Even a present action field would not establish independently verified clinical action gold."
        ),
    }


def audit_dataset(repo: Path, name: str, files: tuple[str, str, str]):
    predictions_path, labels_path, rules_path = [repo / M1_DIR / f for f in files]
    predictions = read_jsonl(predictions_path)
    labels = read_jsonl(labels_path)
    rules = read_jsonl(rules_path)
    prediction_index = unique_index(predictions, context=f"{name} predictions")
    label_index = unique_index(labels, context=f"{name} labels")
    if set(prediction_index) != set(label_index):
        raise FrozenSignalError(f"{name}: predictions and gold keys are not a bijection")
    adapter = FrozenSignalAdapter(predictions)
    # Gold rows supply only exact IDs to the adapter. It ignores their labels.
    adapted = {
        mode: {exact_key(row): row for row in adapter.adapt(labels, mode=mode)}
        for mode in ADAPTER_MODES
    }
    sample_rows = []
    gate_strata = Counter()
    disputed_open = []
    for key, label in label_index.items():
        prediction = prediction_index[key]
        gold = label["label"]
        if gold not in LABELS:
            raise ValueError(f"{name}: invalid gold at {key!r}")
        sample_id, rule_id = key
        variant = label.get("variant")
        anchor = label.get("anchor_id", sample_id.split("_", 1)[0])
        if not isinstance(variant, str) or not variant:
            raise ValueError(f"{name}: missing variant at {key!r}")
        gate = prediction["allow_attention_intervention"]
        evidence = prediction["evidence_status"]
        gate_strata[(prediction["C"], gold, evidence, gate)] += 1
        row = {
            "dataset": name, "sample_id": sample_id, "rule_id": rule_id,
            "anchor_id": anchor, "variant": variant, "gold_label": gold,
            "A": prediction["A"], "B": prediction["B"], "C": prediction["C"],
            "final_C_correct": prediction["C"] == gold,
            "evidence_status": evidence, "allow_attention_intervention": gate,
            "condition_count": len(prediction["conditions"]),
            "strict_gate_route_sign": adapted["strict_gate"][key]["route_sign"],
            "label_only_diagnostic_route_sign": adapted["label_only_diagnostic"][key]["route_sign"],
            "source_target": None, "decidability_target": None,
        }
        sample_rows.append(row)
        if gate and evidence == "disputed":
            disputed_open.append({k: row[k] for k in (
                "sample_id", "rule_id", "C", "gold_label", "evidence_status",
                "allow_attention_intervention",
            )})
    gold_values = [row["gold_label"] for row in sample_rows]
    metrics = {
        field: label_metrics(gold_values, [row[field] for row in sample_rows])
        for field in ("A", "B", "C")
    }
    grouped_metrics = {}
    for field in ("variant", "rule_id"):
        grouped = defaultdict(list)
        for row in sample_rows:
            grouped[row[field]].append(row)
        grouped_metrics[field] = {
            value: label_metrics([r["gold_label"] for r in rows], [r["C"] for r in rows])
            for value, rows in sorted(grouped.items())
        }
    mode_stats = {}
    for mode in ADAPTER_MODES:
        signs = [row[f"{mode}_route_sign"] for row in sample_rows]
        active = [row for row in sample_rows if row[f"{mode}_route_sign"] != 0]
        correct_active = sum(
            row[f"{mode}_route_sign"] == LABEL_SIGNS[row["gold_label"]]
            for row in active
        )
        mode_stats[mode] = {
            "negative": signs.count(-1), "zero": signs.count(0), "positive": signs.count(1),
            "active": len(active), "active_fraction": len(active) / len(signs),
            "active_sign_matches_gold": correct_active,
            "active_sign_agreement": correct_active / len(active) if active else None,
            "metric_scope": "Routing signal only; no downstream model inference or accuracy.",
        }
    gate_rows = [row for row in sample_rows if row["allow_attention_intervention"]]
    by_anchor = defaultdict(set)
    for row in sample_rows:
        by_anchor[row["anchor_id"]].add(row["variant"])
    schema = set().union(*(row.keys() for row in predictions))
    stats = {
        "prediction_file": str(predictions_path.relative_to(repo)),
        "labels_file": str(labels_path.relative_to(repo)),
        "rules_file": str(rules_path.relative_to(repo)),
        "n_cases": len(sample_rows), "n_rule_ids": len({r["rule_id"] for r in sample_rows}),
        "n_anchor_ids": len(by_anchor),
        "variants_per_anchor_histogram": dict(sorted(Counter(len(v) for v in by_anchor.values()).items())),
        "original_counterfactual_anchor_pairs": sum(
            {"original", "counterfactual"}.issubset(variants) for variants in by_anchor.values()
        ),
        "variant_counts": dict(sorted(Counter(r["variant"] for r in sample_rows).items())),
        "join": {
            "key": ["sample_id", "rule_id"], "exact_bijection": True,
            "duplicate_prediction_keys": 0, "duplicate_label_keys": 0,
            "missing_prediction_keys": 0, "missing_label_keys": 0,
        },
        "prediction_top_level_fields": sorted(schema),
        "explicit_source_or_decidability_fields_present": sorted(schema & POLICY_FIELDS),
        "raw_model_input_fields_present": sorted(schema & RAW_INPUT_FIELDS),
        "condition_count": sum(r["condition_count"] for r in sample_rows),
        "condition_fields": sorted({
            field for prediction in predictions for condition in prediction["conditions"]
            for field in condition
        }),
        "gold_counts": count_labels(gold_values),
        "prediction_counts": {field: count_labels(r[field] for r in sample_rows) for field in ("A", "B", "C")},
        "B_C_mismatches": [r["sample_id"] for r in sample_rows if r["B"] != r["C"]],
        "metrics": metrics, "final_C_grouped_metrics": grouped_metrics,
        "gate": {
            "true": len(gate_rows), "false": len(sample_rows) - len(gate_rows),
            "gold_counts_when_true": count_labels(r["gold_label"] for r in gate_rows),
            "C_counts_when_true": count_labels(r["C"] for r in gate_rows),
            "disputed_gate_true_count": len(disputed_open),
            "disputed_gate_true_rows_preserved": disputed_open,
            "interpretation": "Recorded values are preserved, including disputed evidence with an open gate.",
        },
        "adapter_signal_statistics": mode_stats,
        "rules": rule_audit(rules, {r["rule_id"] for r in sample_rows}),
        "frozen_provenance_note": (
            "Offline V6 recomposition of saved V5 conditions; not a new full V6 inference run."
            if name == "old192" else "Saved new100 V6 run; no inference rerun by this audit."
        ),
    }
    strata_rows = [
        {"dataset": name, "C": label, "gold_label": gold, "evidence_status": evidence,
         "allow_attention_intervention": gate, "count": count}
        for (label, gold, evidence, gate), count in sorted(gate_strata.items())
    ]
    stats["gate_strata"] = strata_rows
    return stats, sample_rows, strata_rows


def signed_join_audit(repo: Path, sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    relative = f"{SIGNED_DIR}/results/analysis/selected_predictions.jsonl.gz"
    rows = read_jsonl(repo / relative)
    provenance = json.loads((repo / SIGNED_DIR / "results/fold0/provenance.json").read_text())
    seen = set()
    cases = {}
    methods = defaultdict(set)
    invariant_fields = ("pair_id", "group_id", "rule_id", "applicable", "gold_answer", "fold", "role")
    for row in rows:
        record_key = row["case_id"], row["method"]
        if record_key in seen:
            raise ValueError("duplicate selected signed (case_id, method) record")
        seen.add(record_key)
        methods[row["method"]].add(row["case_id"])
        case = {field: row[field] for field in invariant_fields}
        if type(case["applicable"]) is not bool:
            raise ValueError("signed applicable field must be boolean")
        if row["case_id"] in cases and cases[row["case_id"]] != case:
            raise ValueError("signed methods disagree on case identity or gold metadata")
        cases[row["case_id"]] = case
    if any(ids != set(cases) for ids in methods.values()):
        raise ValueError("signed selected methods do not cover the same cases")
    pairs = defaultdict(list)
    for case_id, case in cases.items():
        pairs[case["pair_id"]].append(case_id)
    if any(len(ids) != 2 or {cases[i]["applicable"] for i in ids} != {True, False}
           for ids in pairs.values()):
        raise ValueError("signed pairs must have one applicable and one nonapplicable member")
    case_keys = {(case_id, case["rule_id"]) for case_id, case in cases.items()}
    intersections = {}
    for dataset in DATASETS:
        frozen = [r for r in sample_rows if r["dataset"] == dataset]
        frozen_keys = {(r["sample_id"], r["rule_id"]) for r in frozen}
        intersections[dataset] = {
            "frozen_case_count": len(frozen),
            "sample_id_overlap": len({r["sample_id"] for r in frozen} & set(cases)),
            "sample_id_rule_id_overlap": len(frozen_keys & case_keys),
            "signed_cases_without_matching_frozen_signal": len(case_keys - frozen_keys),
        }
    old_keys = {(r["sample_id"], r["rule_id"]) for r in sample_rows if r["dataset"] == "old192"}
    new_keys = {(r["sample_id"], r["rule_id"]) for r in sample_rows if r["dataset"] == "new100"}
    schema = set().union(*(r.keys() for r in rows))
    return {
        "selected_journal_file": relative, "journal_records": len(rows),
        "unique_cases": len(cases), "unique_pairs": len(pairs),
        "families": len({c["group_id"] for c in cases.values()}),
        "rules": len({c["rule_id"] for c in cases.values()}),
        "method_case_counts": {method: len(ids) for method, ids in sorted(methods.items())},
        "roles": dict(Counter(row["role"] for row in rows)),
        "case_counts_by_fold": dict(sorted(Counter(c["fold"] for c in cases.values()).items())),
        "applicable_cases": sum(c["applicable"] for c in cases.values()),
        "nonapplicable_cases": sum(not c["applicable"] for c in cases.values()),
        "duplicate_case_method_keys": 0, "all_pairs_have_app_and_nonapp": True,
        "gold_applicability_oracle": provenance["gold_applicability_oracle"],
        "explicit_source_or_decidability_fields_present": sorted(schema & POLICY_FIELDS),
        "frozen_v6_label_field_C_present": "C" in schema,
        "joins_against_frozen_v6": intersections,
        "old192_new100_key_overlap": len(old_keys & new_keys),
        "can_relabel_as_current_v6_end_to_end": False,
        "interpretation": "192 module-one cases and 192 module-two pairs are unrelated ID sets; no row-order or count-based joins are valid.",
    }


def raw_input_audit(repo: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for filename, expected in manifest["raw_inputs_server_only"].items():
        relative = f"{M1_DIR}/{filename}"
        path = repo / relative
        row = {
            "path": relative, "declared_server_only_in_source_manifest": True,
            "expected_sha256": expected, "present_in_current_checkout": path.is_file(),
            "reconstructed_from_gold_or_predictions": False,
        }
        if path.is_file():
            observed = fingerprint(path)
            row["observed_sha256"] = observed["sha256"]
            row["matches_expected_sha256"] = observed["sha256"] == expected
            if not row["matches_expected_sha256"]:
                raise ValueError(f"original input hash mismatch: {relative}")
        else:
            row["status"] = "missing_from_current_checkout; original inputs required for model replay"
        records.append(row)
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(repo: Path, output: Path) -> dict[str, Any]:
    inputs_before = verify_inputs(repo)
    manifest = json.loads((repo / M1_DIR / "new_set_manifest.json").read_text())
    protocol = json.loads((repo / M1_DIR / "dataset_protocol.json").read_text())
    datasets = {}
    cases = []
    gate_rows = []
    for name, files in DATASETS.items():
        dataset, sample_rows, strata = audit_dataset(repo, name, files)
        datasets[name] = dataset
        cases.extend(sample_rows)
        gate_rows.extend(strata)
    if datasets["old192"]["n_cases"] != protocol["sample_count"]:
        raise ValueError("old192 protocol case count mismatch")
    if datasets["old192"]["gold_counts"] != protocol["label_counts"]:
        raise ValueError("old192 protocol label count mismatch")
    if datasets["old192"]["variant_counts"] != protocol["scenario_counts"]:
        raise ValueError("old192 protocol variant count mismatch")
    new_ids = [row["sample_id"] for row in cases if row["dataset"] == "new100"]
    if len(new_ids) != manifest["n"] or set(new_ids) != set(manifest["sample_ids"]):
        raise ValueError("new100 manifest sample ID mismatch")
    result = {
        "schema_version": "care.module2.frozen_signal_audit.v1",
        "status": "completed_offline_frozen_cache_audit",
        "source_repository": "chunxi-alpc/TRIAGE", "source_commit": SOURCE_COMMIT,
        "no_new_model_inference": True,
        "gold_labels_used_only_for_evaluation": True,
        "frozen_module1_modified": False,
        "source_and_decidability_inferred_from_TFU": False,
        "adapter_definitions": {
            "strict_gate": "sign(C) * saved_boolean_gate; T/F/U -> +1/-1/0",
            "label_only_diagnostic": "sign(C); gate preserved but not applied; frozen predicted signal, not oracle labels",
        },
        "datasets": datasets, "signed_dataset_join_audit": signed_join_audit(repo, cases),
        "original_input_availability": raw_input_audit(repo, manifest),
        "input_fingerprints": inputs_before,
        "protocol_checks": {
            "old192_case_label_variant_counts_match": True,
            "new100_manifest_sample_ids_match": True,
            "old192_independent_clinician_review": protocol["independent_clinician_review"],
            "old192_independent_v5_validation_available": protocol["independent_v5_validation_available"],
        },
        "limitations": [
            "Saved M1 predictions and historical signed journals do not form an end-to-end dataset.",
            "Routing-sign metrics are not answer accuracy or evidence of successful model intervention.",
            "Missing clinical actions, source targets and decidability targets are not imputed.",
            "Original inputs are not reconstructed from gold observations, spans or saved predictions.",
            "Variants share base cases; counts must not be interpreted as independent clinical records.",
            "The audit pins historical files; this is not a new M1 or M2 model experiment.",
        ],
    }
    # Ensure every frozen source is unchanged before persisting audit artifacts.
    if verify_inputs(repo) != inputs_before:
        raise RuntimeError("a frozen input changed during the audit")
    confusion_rows = []
    for name, dataset in datasets.items():
        for method, metric in dataset["metrics"].items():
            for gold, predictions in metric["confusion_rows_gold_columns_prediction"].items():
                for predicted, count in predictions.items():
                    confusion_rows.append({"dataset": name, "method": method,
                                           "gold_label": gold, "predicted_label": predicted,
                                           "count": count})
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_signal_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output / "frozen_signal_cases.csv", cases)
    write_csv(output / "frozen_signal_confusion.csv", confusion_rows)
    write_csv(output / "frozen_signal_gate_strata.csv", gate_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    result = run(args.repo_root.resolve(), args.output.resolve())
    print(json.dumps({
        "status": result["status"], "no_new_model_inference": True,
        "final_C": {name: {"n": d["n_cases"], "correct": d["metrics"]["C"]["correct"],
                            "accuracy": d["metrics"]["C"]["accuracy"],
                            "gate_true": d["gate"]["true"]}
                    for name, d in result["datasets"].items()},
        "signed_unique_cases": result["signed_dataset_join_audit"]["unique_cases"],
        "frozen_v6_signed_key_overlaps": {
            name: audit["sample_id_rule_id_overlap"]
            for name, audit in result["signed_dataset_join_audit"]["joins_against_frozen_v6"].items()
        },
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
