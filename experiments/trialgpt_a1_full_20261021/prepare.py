#!/usr/bin/env python3
"""Prepare *all* expert-labeled TrialGPT criteria for the untouched PR12 A1 runner.

Only data fields question/patient/rules reach the model. Gold is separate.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPO_ID = "ncbi/TrialGPT-Criterion-Annotations"
PARQUET_FILE = "data/train-00000-of-00001.parquet"
SOURCE_COMMIT = "8fb87d15576b88500f44a88f04799267d9f2b5ee"
COUNT = 1015

LABEL_MAP = {
    ("inclusion", "included"): "T",
    ("inclusion", "not included"): "F",
    ("inclusion", "not enough information"): "U",
    ("inclusion", "not applicable"): None,
    ("exclusion", "excluded"): "T",
    ("exclusion", "not excluded"): "F",
    ("exclusion", "not enough information"): "U",
    ("exclusion", "not applicable"): None,
}
QUESTION = "Does this clinical trial eligibility criterion apply to the described patient?"
RULE_ID = "R_CRITERION"


def canonical(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen_a1(root=ROOT):
    config = json.loads((root / "configs/care_module1_for_module2.json").read_text())
    assert config["method"] == "A1" and config["source_commit"] == SOURCE_COMMIT
    assert config["prompt_sha256"] == sha256(root / config["prompt_path"]), "frozen A1 prompt changed"
    assert config["fixed_examples_sha256"] == sha256(root / config["fixed_examples_path"]), "frozen few-shot changed"
    assert config["example_count"] == {"demonstrations": 3, "patient_narratives": 4}
    runner = root / config["inference_entrypoint"]
    assert runner.is_file(), "frozen A1 runner not found"
    original = subprocess.run(["git", "show", SOURCE_COMMIT + ":" + config["inference_entrypoint"]],
                              cwd=root, capture_output=True, check=False)
    if original.returncode:
        raise RuntimeError("PR12 source commit unavailable; fetch the exact immutable commit first")
    assert runner.read_bytes() == original.stdout, "runner differs from frozen PR12"
    return {"source_commit": SOURCE_COMMIT,
            "prompt_sha256": config["prompt_sha256"],
            "fewshot_sha256": config["fixed_examples_sha256"],
            "runner_sha256": sha256(runner),
            "base_model": config["base_model"],
            "adapter_model": config["adapter_model"]}


def evidence_present(raw):
    if isinstance(raw, (tuple, list)):
        return bool(raw)
    s = canonical(raw)
    if not s or s in ("[]", "null"):
        return False
    try:
        parsed = ast.literal_eval(s)
        return bool(parsed)
    except (ValueError, SyntaxError):
        return None


def records_to_inputs_and_labels(records):
    """Pure function, independently unit-testable without GPU/pandas/network."""
    inputs, labels = [], []
    ids = set()
    for r in records:
        rid = int(r["annotation_id"])
        if rid in ids:
            raise ValueError("duplicate official annotation_id " + str(rid))
        ids.add(rid)
        t = canonical(r["criterion_type"]).lower()
        g = canonical(r["expert_eligibility"]).lower()
        if (t, g) not in LABEL_MAP:
            raise ValueError("unexpected official eligibility label: " + repr((t, g)))
        patient, criterion = canonical(r["note"]), canonical(r["criterion_text"])
        valid = bool(patient and criterion)
        cid = "trialgpt_%05d" % rid
        if valid:
            # No source gold, GPT-4 annotations, expert rationale or evidence can enter.
            inputs.append({
                "case_id": cid, "question": QUESTION, "patient": patient,
                "rules": [{"rule_id": RULE_ID, "text": criterion,
                           "action": "criterion assessment"}],
            })
        labels.append({
            "case_id": cid, "annotation_id": rid,
            "patient_id": canonical(r["patient_id"]),
            "trial_id": canonical(r["trial_id"]),
            "criterion_type": t,
            "expert_eligibility": g,
            "mapped_gold": LABEL_MAP[(t, g)],
            "training_flag": bool(r["training"]),
            "evidence_present": evidence_present(r.get("expert_sentences")),
            "valid_input": valid,
            "note_sha256": hashlib.sha256(patient.encode()).hexdigest(),
            "criterion_sha256": hashlib.sha256(criterion.encode()).hexdigest(),
        })
    return inputs, labels


def jsonl_write(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_dataframe(dataframe, out, source):
    if len(dataframe) != COUNT:
        raise ValueError("Official source must contain all 1015 rows, not a subset")
    required = {"annotation_id", "patient_id", "note", "trial_id",
                "criterion_type", "criterion_text", "expert_eligibility",
                "expert_sentences", "training"}
    if not required.issubset(set(dataframe.columns)):
        raise ValueError("Official source columns missing: " +
                         repr(sorted(required - set(dataframe.columns))))
    inputs, labels = records_to_inputs_and_labels(dataframe.to_dict("records"))
    if len(labels) != COUNT:
        raise AssertionError("expert labels population lost rows")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    jsonl_write(out / "inputs_all.jsonl", inputs)
    jsonl_write(out / "expert_labels_all.jsonl", labels)
    for gpu in range(4):
        # Deterministic sharding, sharded case IDs are never treated as new cases.
        jsonl_write(out / ("inputs_%d.jsonl" % gpu),
                    [row for i, row in enumerate(inputs) if i % 4 == gpu])
    counts = dict(collections.Counter(r["expert_eligibility"] for r in labels))
    manifest = {
        "source_dataset": REPO_ID, "source_split": "train",
        **source,
        "expert_records": COUNT, "model_input_records": len(inputs),
        "technical_invalid_input": COUNT - len(inputs),
        "mapped_label_records": sum(r["mapped_gold"] is not None for r in labels),
        "unmapped_not_applicable_records": sum(r["mapped_gold"] is None for r in labels),
        "patients": len({r["patient_id"] for r in labels}),
        "trials": len({r["trial_id"] for r in labels}),
        "expert_native_labels": counts,
        "expert_evidence_available": sum(r["evidence_present"] is True for r in labels),
        "expert_evidence_empty": sum(r["evidence_present"] is False for r in labels),
        "no_gold_ever_in_model_inputs": True,
        "source_rows_never_filtered": True,
        "file_sha256": {name: sha256(out / name)
            for name in ["inputs_all.jsonl", "expert_labels_all.jsonl"]
            + ["inputs_%d.jsonl" % i for i in range(4)]},
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-parquet", type=Path)
    p.add_argument("--revision", default="main",
                   help="Hugging Face dataset revision, resolved and pinned to commit SHA")
    p.add_argument("--out", type=Path, default=HERE / "private")
    args = p.parse_args()
    frozen = check_frozen_a1()
    if args.local_parquet:
        file = args.local_parquet
        source = {"revision": "local_verified_file", "parquet_sha256": sha256(file)}
    else:
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().repo_info(repo_id=REPO_ID, repo_type="dataset",
                                 revision=args.revision)
        rev = info.sha
        if not rev:
            raise ValueError("Dataset commit sha missing")
        file = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                    revision=rev, filename=PARQUET_FILE))
        source = {"revision": rev, "parquet_sha256": sha256(file)}
    import pandas as pd
    df = pd.read_parquet(file)
    manifest = prepare_dataframe(df, args.out, source)
    manifest["frozen_a1"] = frozen
    (args.out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in
          ("expert_records", "model_input_records", "mapped_label_records",
           "unmapped_not_applicable_records", "patients", "trials")},
          ensure_ascii=False))


if __name__ == "__main__":
    main()

