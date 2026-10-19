"""Export text-redacted labels, historical predictions, and explicit pairs."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_by_id(rows, name):
    ids = [row.get("sample_id") for row in rows]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} sample IDs are missing or duplicated")
    return set(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v4-labels", type=Path, required=True)
    p.add_argument("--v4-predictions", type=Path, required=True)
    p.add_argument("--v4-pairs", type=Path, required=True)
    p.add_argument("--labels-out", type=Path, required=True)
    p.add_argument("--qwen-reference-out", type=Path, required=True)
    p.add_argument("--pairs-out", type=Path, required=True)
    a = p.parse_args()
    source_labels = read_jsonl(a.v4_labels)
    source_predictions = read_jsonl(a.v4_predictions)
    source_pairs = read_jsonl(a.v4_pairs)
    label_ids = unique_by_id(source_labels, "V4 labels")
    prediction_ids = unique_by_id(source_predictions, "V4 predictions")
    if label_ids != prediction_ids or len(label_ids) != 192:
        raise ValueError(f"V4 label/prediction ID mismatch or unexpected size: labels={len(label_ids)}, predictions={len(prediction_ids)}")
    if len(source_pairs) != 32 or len({row.get("pair_id") for row in source_pairs}) != len(source_pairs):
        raise ValueError("expected 32 unique explicit V4 counterfactual pairs")
    variant_by_id = {row["sample_id"]: row["variant"] for row in source_labels}
    pair_by_id, pairs = {}, []
    for row in source_pairs:
        original, counter = row["original_id"], row["counterfactual_id"]
        if original not in label_ids or counter not in label_ids:
            raise ValueError(f"pair {row['pair_id']} references unknown sample ID")
        if variant_by_id[original] != "original" or variant_by_id[counter] != "counterfactual":
            raise ValueError(f"pair {row['pair_id']} has unexpected variant assignment")
        if original in pair_by_id or counter in pair_by_id:
            raise ValueError("sample ID appears in more than one explicit pair")
        pair_by_id[original] = row["pair_id"]
        pair_by_id[counter] = row["pair_id"]
        pairs.append({"pair_id": row["pair_id"], "original_id": original, "counterfactual_id": counter})
    labels = []
    for row in source_labels:
        item = {k: row[k] for k in ("sample_id", "rule_id", "variant", "label",
            "acceptable_evidence_proofs", "all_relevant_evidence_ids", "conflict_evidence_ids") if k in row}
        if item["sample_id"] in pair_by_id:
            item["pair_id"] = pair_by_id[item["sample_id"]]
        labels.append(item)
    qwen = []
    for row in source_predictions:
        direct = row.get("direct") or {}
        qwen.append({"sample_id": row["sample_id"], "rule_id": row["rule_id"],
                     "label": direct.get("label", "U"),
                     "evidence_sentence_ids": direct.get("evidence_sentence_ids", []),
                     "valid": bool(direct.get("valid", False))})
    for target, rows in ((a.labels_out, labels), (a.qwen_reference_out, qwen), (a.pairs_out, pairs)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    print(f"public labels: {len(labels)}; historical predictions: {len(qwen)}; explicit pairs: {len(pairs)}")

if __name__ == "__main__":
    main()
