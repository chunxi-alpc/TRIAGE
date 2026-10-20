"""Staged fit/select/evaluate runner for N1--N5."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .io import canonical_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .model import Intervention, OptionScorer, capture_hidden, default_layer, hidden_dim, load_qwen


def _rows(data_dir: Path, task: str, split_name: str) -> list[dict[str, Any]]:
    inputs = read_jsonl(data_dir / task / "inputs.jsonl")
    labels = read_jsonl(data_dir / task / "labels.jsonl")
    splits = read_jsonl(data_dir / task / "splits.jsonl")
    return [{**i, **label, **split} for i, label, split in zip(inputs, labels, splits) if split["split"] == split_name]


def _all_rows(data_dir: Path, task: str) -> list[dict[str, Any]]:
    inputs = read_jsonl(data_dir / task / "inputs.jsonl")
    labels = read_jsonl(data_dir / task / "labels.jsonl")
    splits = read_jsonl(data_dir / task / "splits.jsonl")
    return [{**i, **label, **split} for i, label, split in zip(inputs, labels, splits)]


def _prompt(row: dict[str, Any], method: str, control: dict[str, Any] | None = None) -> str:
    prompt = row["prompt_text"]
    if method != "text":
        return prompt
    target = control or row.get("control_target", {})
    if row["task"] == "N1":
        policy = "Focus on the target question." if target.get("R") else "The target rule is irrelevant to this question."
        policy += " The rule is applicable." if target.get("A") else " The rule is not applicable."
    elif row["task"] == "N2":
        policy = "Use the context note." if target.get("source") == "C" else "Use the closed-book record."
    elif row["task"] == "N3":
        policy = "Execute or do not execute when the decision is fixed." if target.get("DQ") == "D" else "Ask the decisive missing attribute when needed."
    elif row["task"] == "N4":
        policy = f"Resolve {target.get('DQ')} first, then use source {target.get('source')} and applicability {target.get('A')}."
    else:
        signal = row.get("a1_signal")
        policy = {1: "The frozen upstream rule is true.", -1: "The frozen upstream rule is false.", 0: "The frozen upstream rule is unknown.", None: "The frozen upstream result is unavailable."}.get(signal, "")
    return prompt + "\nPolicy for this run: " + policy + "\nAnswer with the option letter only."


def _options(row: dict[str, Any]) -> dict[str, str]:
    mapping = row.get("options") or row.get("option_map") or {"A": "A", "B": "B"}
    # Prompts explicitly request option letters.  The human-readable action
    # text remains in the input/label artifacts; scoring the letters avoids a
    # hidden mismatch where a model answers "A" but is scored against a route
    # name.
    return {str(key): str(key) for key in mapping}


def _target_key(row: dict[str, Any], control: dict[str, Any] | None = None) -> str:
    control = control or row.get("control_target", {})
    if row["task"] == "N1":
        if not control.get("R"):
            return "C"
        return "A" if control.get("A") else "B"
    if row["task"] == "N2":
        return "A" if control.get("source") == "C" else "B"
    if row["task"] == "N3":
        # D/Q is the externally supplied mechanism bit; the action for D must
        # still be read from the receiver facts, so direct bias only has a
        # defined target for the Q branch.
        return "Q" if control.get("DQ") == "Q" else None
    if row["task"] == "N4":
        if control.get("DQ") == "Q":
            return "Q"
        if control.get("A") == "F":
            return "SKIP"
        return control.get("source", "C")
    # N5 is evaluated against the A1 signal.  U/null are kept in the data and
    # evaluated as a no-signal arm rather than silently discarded.
    return {1: "A", -1: "B", 0: "D", None: "D"}.get(row.get("a1_signal"), "D")


def _control_for_row(row: dict[str, Any], control_key: str | None = None) -> dict[str, Any]:
    control = dict(row.get("control_target") or {})
    if row["task"] == "N2" and control_key:
        control["source"] = control_key
    elif row["task"] == "N3" and control_key:
        control["DQ"] = control_key
    return control


def _vector_for(row: dict[str, Any], vectors: dict[str, torch.Tensor], control: dict[str, Any]) -> torch.Tensor | None:
    if row["task"] == "N1":
        items = []
        if control.get("R"):
            items.append(vectors.get("R"))
        if control.get("A"):
            items.append(vectors.get("A"))
        else:
            items.append(None if vectors.get("A") is None else -vectors["A"])
        result = [item for item in items if item is not None]
        return torch.stack(result).mean(0) if result else None
    if row["task"] == "N2":
        v = vectors.get("C_P")
        return v if control.get("source") == "C" else None if v is None else -v
    if row["task"] == "N3":
        v = vectors.get("D_Q")
        return v if control.get("DQ") == "D" else None if v is None else -v
    if row["task"] == "N4":
        items = []
        if "C_P" in vectors:
            items.append(vectors["C_P"] if control.get("source") == "C" else -vectors["C_P"])
        if "D_Q" in vectors:
            items.append(vectors["D_Q"] if control.get("DQ") == "D" else -vectors["D_Q"])
        if "R" in vectors:
            items.append(vectors["R"] if control.get("A") == "T" else -vectors["R"])
        return torch.stack(items).mean(0) if items else None
    v = vectors.get("A1")
    signal = row.get("a1_signal")
    return None if v is None or signal in (None, 0) else v * float(signal)


def _fit_direction(model, tokenizer, rows: list[dict[str, Any]], layer: int, device: torch.device, key_fn):
    groups: dict[Any, list[torch.Tensor]] = defaultdict(list)
    meta = []
    for row in rows:
        h, info = capture_hidden(model, tokenizer, row["prompt_text"], layer, device)
        key = key_fn(row)
        groups[key].append(h)
        meta.append({"case_id": row["case_id"], **info})
    if not groups:
        return None, meta
    if set(groups) >= {0, 1}:
        v = torch.stack(groups[1]).mean(0) - torch.stack(groups[0]).mean(0)
    else:
        keys = sorted(groups, key=repr)
        v = torch.stack(groups[keys[-1]]).mean(0) - torch.stack(groups[keys[0]]).mean(0)
    norm = float(v.norm().item())
    return (v / norm if norm else v), meta


def fit(args) -> int:
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, fingerprint = load_qwen(args.model, args.device)
    task = args.experiment
    rows = _rows(data_dir, task, "train")
    layer = int(args.layer) if args.layer is not None else default_layer(model)
    vectors: dict[str, torch.Tensor] = {}
    captures: list[dict[str, Any]] = []
    if task == "N1":
        vectors["R"], meta = _fit_direction(model, tokenizer, rows, layer, device, lambda r: int(bool(r["gold_R"])))
        vectors["A"], meta_a = _fit_direction(model, tokenizer, rows, layer, device, lambda r: int(bool(r["gold_A"])))
        captures.extend(meta + meta_a)
    elif task == "N2":
        # Auxiliary policy sentences provide a train-only control direction;
        # receiver prompts remain neutral during every evaluation call.
        aux = [{**r, "prompt_text": r["prompt_text"] + "\nTraining policy: use context." if i % 2 == 0 else r["prompt_text"] + "\nTraining policy: use closed-book knowledge."} for i, r in enumerate(rows)]
        vectors["C_P"], meta = _fit_direction(model, tokenizer, aux, layer, device, lambda r: int("context." in r["prompt_text"]))
        captures.extend(meta)
    elif task == "N3":
        aux = [{**r, "prompt_text": r["prompt_text"] + ("\nTraining policy: execute when determined." if r["gold_DQ"] == "D" else "\nTraining policy: ask when unresolved.")} for r in rows]
        vectors["D_Q"], meta = _fit_direction(model, tokenizer, aux, layer, device, lambda r: int(r["gold_DQ"] == "D"))
        captures.extend(meta)
    elif task == "N4":
        # N4 freezes the vectors selected in the single-axis stages.  The CLI
        # copies those vectors before running the joint stage.
        raise ValueError("Fit N4 by supplying frozen N2/N3/N1 vectors")
    elif task == "N5":
        aux = [{**r, "prompt_text": r["prompt_text"] + f"\nTraining policy: frozen signal {r.get('a1_signal')}."} for r in rows]
        vectors["A1"], meta = _fit_direction(model, tokenizer, aux, layer, device, lambda r: int((r.get("a1_signal") or 0) > 0))
        captures.extend(meta)
    tensor_path = out_dir / "route_vectors.pt"
    torch.save({key: value.cpu() for key, value in vectors.items() if value is not None}, tensor_path)
    fit_forwards = len(captures)
    result = {"schema_version": "care.m2.fit.v1", "experiment": task, "split": "train", "layer": layer,
              "vectors": {key: {"norm": float(value.norm().item()), "dim": int(value.numel())} for key, value in vectors.items() if value is not None},
              "model": fingerprint, "forward_count": fit_forwards, "capture_count": len(captures),
              "train_case_ids": [row["case_id"] for row in rows], "route_vectors": str(tensor_path),
              "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(out_dir / "train_fit.json", result)
    return 0


def model_forward_count(model) -> int:
    # OptionScorer counts options; capture-only fit uses one call per row.
    return int(getattr(model, "_care_m2_forward_count", 0))


def _load_vectors(path: str | Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    return {str(key): value.float() for key, value in raw.items()}


def _score_row(scorer: OptionScorer, row: dict[str, Any], method: str, vectors: dict[str, torch.Tensor], layer: int, beta: float,
              control: dict[str, Any] | None = None) -> dict[str, Any]:
    control = control or row.get("control_target", {})
    prompt = _prompt(row, "text" if method == "text" else "baseline", control)
    intervention = None
    if method == "route":
        vector = _vector_for(row, vectors, control)
        if vector is not None:
            intervention = Intervention(layer=layer, vector=vector, scale=beta)
    scores, details = scorer.score(prompt, _options(row), intervention)
    if method == "direct_bias":
        target = _target_key(row, control)
        for key in scores:
            if target is not None and key == target:
                scores[key] += beta
    prediction = max(scores, key=scores.get)
    return {"case_id": row["case_id"], "task": row["task"], "method": method, "prediction": prediction,
            "scores": scores, "prompt_sha256": details["prompt_sha256"], "control": control, "gold_answer": row.get("gold_answer"),
            "gold_DQ": row.get("gold_DQ"), "gold_A": row.get("gold_A"), "a1_signal": row.get("a1_signal"),
            "option_labels": row.get("options") or row.get("option_map"), "forward_count": len(_options(row)),
            "hook": details["options"].get(next(iter(details["options"])), {}).get("hook")}


def evaluate_stage(args) -> int:
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, fingerprint = load_qwen(args.model, args.device)
    rows = _rows(data_dir, args.experiment, "test")
    vectors = _load_vectors(args.vectors) if args.vectors else {}
    selection = read_json(args.selection) if args.selection else {"layer": default_layer(model), "beta": 0.5, "methods": ["baseline", "text", "direct_bias", "route"]}
    layer, beta = int(selection.get("layer", default_layer(model))), float(selection.get("beta", 0.5))
    methods = selection.get("methods", ["baseline", "text", "direct_bias", "route"])
    predictions: list[dict[str, Any]] = []
    if args.experiment == "N2":
        for row in rows:
            for source in ("C", "P"):
                for method in methods:
                    predictions.append(_score_row(OptionScorer(model, tokenizer, device), row, method, vectors, layer, beta, {"source": source}))
    else:
        scorer = OptionScorer(model, tokenizer, device)
        for row in rows:
            for method in methods:
                predictions.append(_score_row(scorer, row, method, vectors, layer, beta))
    write_jsonl(out_dir / "test_predictions.jsonl", predictions)
    summary = summarize(args.experiment, predictions, rows)
    write_json(out_dir / "summary.json", summary)
    actual_forwards = sum(len(_options(next(row for row in rows if row["case_id"] == item["case_id"]))) for item in predictions)
    write_json(out_dir / "run_manifest.json", {"schema_version": "care.m2.run.v1", "experiment": args.experiment,
              "split": "test", "model": fingerprint, "layer": layer, "beta": beta,
              "forward_count": actual_forwards, "prediction_records": len(predictions), "rows": len(rows), "git_commit": args.git_commit,
              "created_at": datetime.now(timezone.utc).isoformat()})
    return 0


def summarize(task: str, predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_method[row["method"]].append(row)
    metrics = {}
    for method, items in by_method.items():
        if task == "N2":
            grouped = defaultdict(list)
            for item in items:
                grouped[(item["case_id"], item["control"].get("source"))].append(item)
            # Both source policies are required for the same receiver.
            per_receiver = defaultdict(dict)
            for item in items:
                per_receiver[item["case_id"]][item["control"].get("source")] = item["prediction"] == ("A" if item["control"].get("source") == "C" else "B")
            paired = [all(values.values()) for values in per_receiver.values()]
            metrics[method] = {"paired_both_correct": sum(paired), "paired_total": len(paired),
                               "paired_rate": sum(paired) / len(paired) if paired else None,
                               "rows_correct": sum(item["prediction"] == ("A" if item["control"].get("source") == "C" else "B") for item in items),
                               "rows_total": len(items)}
        else:
            correct = [item["prediction"] == item.get("gold_answer") for item in items]
            family = defaultdict(list)
            for item in items:
                family[next((row["family_id"] for row in rows if row["case_id"] == item["case_id"]), item["case_id"])].append(item["prediction"] == item.get("gold_answer"))
            metrics[method] = {"correct": sum(correct), "total": len(correct), "accuracy": sum(correct) / len(correct) if correct else None,
                               "all_correct_families": sum(all(values) for values in family.values()), "families": len(family)}
            if task == "N3":
                metrics[method]["q_wrong_decision"] = sum(item.get("gold_DQ") == "Q" and item["prediction"] != "Q" for item in items)
                metrics[method]["d_unnecessary_query"] = sum(item.get("gold_DQ") == "D" and item["prediction"] == "Q" for item in items)
    return {"schema_version": "care.m2.summary.v1", "task": task, "methods": metrics, "n_predictions": len(predictions)}


def select_stage(args) -> int:
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Selection is intentionally small and frozen before test: the dev grid is
    # beta only; layer and vector family are fixed by fit.
    dev_rows = _rows(data_dir, args.experiment, "dev")
    candidates = [0.0, 0.25, 0.5, 1.0]
    best_beta = 0.5
    candidate_metrics = []
    if getattr(args, "model", None) and getattr(args, "vectors", None):
        model, tokenizer, device, fingerprint = load_qwen(args.model, args.device)
        vectors = _load_vectors(args.vectors)
        layer = int(args.layer) if args.layer is not None else default_layer(model)
        for beta in candidates:
            scorer = OptionScorer(model, tokenizer, device)
            items = []
            for row in dev_rows:
                controls = ({"source": "C"}, {"source": "P"}) if args.experiment == "N2" else (None,)
                for control in controls:
                    item = _score_row(scorer, row, "route", vectors, layer, beta, control)
                    expected = ("A" if control["source"] == "C" else "B") if control else row.get("gold_answer")
                    items.append(item["prediction"] == expected)
            metric = sum(items) / len(items) if items else 0.0
            candidate_metrics.append({"beta": beta, "dev_accuracy": metric, "dev_rows": len(items)})
        best_beta = sorted(candidate_metrics, key=lambda item: (-item["dev_accuracy"], item["beta"]))[0]["beta"]
        selection_fingerprint = fingerprint
    else:
        selection_fingerprint = None
    best = {"beta": best_beta, "dev_case_count": len(dev_rows), "selection_rule": "predeclared beta grid; maximize dev row accuracy; ties choose smallest beta", "candidates": candidate_metrics or [{"beta": beta} for beta in candidates]}
    best["layer"] = int(args.layer) if args.layer is not None else None
    if selection_fingerprint:
        best["model"] = selection_fingerprint
    best["methods"] = ["baseline", "text", "direct_bias", "route"]
    write_json(out_dir / "dev_selection.json", best)
    return 0
