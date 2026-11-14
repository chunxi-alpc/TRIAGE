"""CPU evaluation of one independently fitted E1 model; never loads a backbone."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy

HIST = Path("/home/gcx/aster_cr_repair_v2_run_20261108")
TARGET = Path("/home/gcx/aster_cross_task_transfer_20261109")
SCOPES = ("dev", "test", "nli4ct", "medpic", "nli4ct_official5500", "medpic_official467")


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def uid(row):
    for key in ("uid", "view_id", "id"):
        if row.get(key) is not None:
            return row[key]
    raise KeyError("Record has no uid/view_id/id")


class DynamicStore:
    """Read score records and requested feature files, without loading inputs.jsonl.

    Features may contain [layers, heads, dim], [all_heads, dim], or selected
    [head_ids, dim]. Sparse arrays are expanded only to the declared/needed size.
    """
    def __init__(self, root, required_heads=None):
        self.out = Path(root)
        self.required_heads = np.asarray(required_heads if required_heads is not None else [], dtype=int)
        self.records = {}
        paths = [self.out / "data/reused_scores.jsonl", self.out / "cache/scores.jsonl"]
        paths += sorted((self.out / "cache").glob("scores*.jsonl"))
        paths += sorted((self.out / "cache").glob("*/scores*.jsonl"))
        for path in dict.fromkeys(paths):
            for record in rows(path):
                if record.get("status", "ok") == "ok" and "ell" in record and "feature" in record:
                    self.records[uid(record)] = record
        self.scores = self.records

    def record(self, key):
        return self.records[key]

    def ell(self, key):
        return float(self.record(key)["ell"])

    @lru_cache(maxsize=24)
    def get(self, key):
        record = self.record(key)
        path = Path(record["feature"])
        if not path.is_absolute():
            path = self.out / path
        result = dict(record)
        with np.load(path, allow_pickle=False) as arrays:
            heads = arrays["heads"].astype(np.float32)
            heads = heads.reshape(-1, heads.shape[-1])
            ids = arrays["head_ids"].astype(int).reshape(-1) if "head_ids" in arrays else None
            if ids is not None:
                if len(ids) != len(heads):
                    raise ValueError("head_ids and feature rows differ: " + str(path))
                needed = max(int(ids.max(initial=-1)), int(self.required_heads.max(initial=-1))) + 1
                needed = max(needed, int(record.get("total_heads", needed)))
                full = np.full((needed, heads.shape[-1]), np.nan, np.float32)
                full[ids] = heads
                heads = full
            result["heads"] = heads
            if "grads" in arrays:
                grads = arrays["grads"].astype(np.float32)
                grads = grads.reshape(-1, grads.shape[-1])
                if ids is not None:
                    full = np.full_like(heads, np.nan)
                    full[ids] = grads
                    grads = full
                result["grads"] = grads
            if "residual" in arrays:
                result["residual"] = arrays["residual"].astype(np.float32)
        result["ell"] = self.ell(key)
        return result

    def heads(self, key, selected=None):
        value = self.get(key)["heads"]
        return value if selected is None else value[np.asarray(selected, dtype=int)].reshape(-1)


def load_models(root):
    root = Path(root)
    model = legacy.model_parameters(root, "E1")
    models = {"E1": model}
    calibration_path = root / "models/E1/full/calibration.json"
    calibration = read(calibration_path, {})
    if calibration:
        intercept = next((calibration[k] for k in ("b", "b0", "intercept") if k in calibration), None)
        if intercept is None:
            raise ValueError("calibration.json must contain b, b0 or intercept")
        if calibration.get("target_labels_used") or calibration.get("test_labels_used"):
            raise ValueError("Source-frozen evaluation cannot use target/test calibration")
        models["calibration"] = dict(model, alpha=np.zeros_like(model["alpha"]),
                                    offset=float(intercept), b=float(intercept),
                                    directory=str(calibration_path),
                                    metadata=dict(model["metadata"], calibration=calibration))
    return models


def _data_rows(root, name):
    # Keep the established metadata-only reader: it discards large prompt strings.
    return legacy.data_rows(Path(root), name)


def _case_rows(root):
    own = Path(root) / "data/path_cases.jsonl"
    return rows(own if own.exists() else HIST / "data/path_cases.jsonl")


def scope_inputs(root, scope):
    root = Path(root)
    if scope in ("dev", "test"):
        views = [v for v in _data_rows(root, "historical_views.jsonl") if v.get("split") == scope]
        pairs = [p for p in _data_rows(root, "historical_pairs.jsonl") if p.get("split") == scope]
        path_views = [v for name in ("path_features.jsonl", "path_interference_views.jsonl")
                      for v in _data_rows(root, name) if v.get("split") == scope]
        cases = [c for c in _case_rows(root) if c.get("split") == scope]
        return dict(views=views, pairs=pairs, cases=cases, path_views=path_views)
    if scope in ("nli4ct", "medpic"):
        return dict(views=[v for v in _data_rows(root, scope + "_views.jsonl") if v.get("split") == "test"],
                    pairs=[p for p in _data_rows(root, scope + "_pairs.jsonl") if p.get("split") == "test"],
                    native_task=scope)
    if scope == "nli4ct_official5500":
        return dict(views=_data_rows(root, "nli4ct_official5500_views.jsonl"),
                    pairs=_data_rows(root, "nli4ct_official5500_pairs.jsonl"), native_task="nli4ct")
    if scope == "medpic_official467":
        explicit = root / "data/medpic_official467_views.jsonl"
        if explicit.exists():
            views = _data_rows(root, explicit.name)
            pairs = _data_rows(root, "medpic_official467_pairs.jsonl")
        else:
            # All 467 original questions exist in the existing split metadata.
            # No target fitting is performed; development questions remain labeled
            # as part of a descriptive all-question evaluation.
            views = [v for v in _data_rows(root, "medpic_views.jsonl") if v.get("original")]
            available = {uid(v) for v in views}
            pairs = [p for p in _data_rows(root, "medpic_pairs.jsonl")
                     if p["left"] in available and p["right"] in available]
        return dict(views=views, pairs=pairs, native_task="medpic")
    raise ValueError("Unsupported scope: " + scope)


def _decisions(views, scores, task=None, items_path=None):
    decision = {name: {key: int(value > 0) for key, value in values.items()}
                for name, values in scores.items()}
    questions = []
    if task != "medpic":
        return decision, questions
    items = {r["id"]: r for r in rows(items_path or TARGET / "medpic/items.jsonl")}
    grouped = defaultdict(dict)
    for view in views:
        if view.get("item_id") and view.get("option_letter"):
            grouped[(view["item_id"], view.get("variant", "full"))][view["option_letter"]] = view
    for (qid, variant), options in grouped.items():
        item = items[qid]
        if set(options) != set(item["options"]):
            raise ValueError("Incomplete MedPIC option set: " + qid + "/" + variant)
        for name, values in scores.items():
            chosen = legacy.decode_medpic(item, {letter: values[uid(v)] for letter, v in options.items()})
            for letter, view in options.items():
                decision[name][uid(view)] = int(letter in chosen)
            sample = next(iter(options.values()))
            questions.append(dict(method=name, item_id=qid, variant=variant, source_id=sample["source_id"],
                                  split=sample.get("split"), predicted=chosen, gold=item["answer"],
                                  correct=set(chosen) == set(item["answer"])))
    return decision, questions


def _population_count(scope, views):
    if scope == "nli4ct":
        return sum(bool(v.get("original")) for v in views), 500
    if scope == "nli4ct_official5500":
        return len({uid(v) for v in views}), 5500
    if scope in ("medpic", "medpic_official467"):
        count = len({v["item_id"] for v in views if v.get("original") and v.get("item_id")})
        return count, 188 if scope == "medpic" else 467
    return None, None


def _native_metrics(scope, views, result, root):
    """Add whole-population accuracy and original-statement macro-F1 separately."""
    if not scope.startswith("nli4ct"):
        return
    predictions = rows(Path(root) / "evaluation" / scope / "predictions.jsonl")
    original = {uid(v) for v in views if v.get("original")}
    for method, metrics in result["models"].items():
        for prefix, population in (("all_statement", predictions),
                                   ("original_statement", [r for r in predictions if r["view_id"] in original])):
            available = [r for r in population if r.get("label") is not None]
            y = np.asarray([r["label"] for r in available], dtype=int)
            p = np.asarray([r["predictions"][method] for r in available], dtype=int)
            f1 = []
            for label in (0, 1):
                tp = int(np.sum((y == label) & (p == label)))
                fp = int(np.sum((y != label) & (p == label)))
                fn = int(np.sum((y == label) & (p != label)))
                f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            metrics[prefix + "_count"] = len(available)
            metrics[prefix + "_correct"] = int(np.sum(y == p))
            metrics[prefix + "_accuracy"] = float(np.mean(y == p)) if len(y) else None
            metrics[prefix + "_macro_f1"] = float(np.mean(f1)) if len(y) else None


def population_status(root):
    result = {}
    descriptions = {
        "nli4ct": dict(original_statements=500, expected_views=2864),
        "medpic": dict(questions=188, original_options=940, expected_views=2820),
        "nli4ct_official5500": dict(statements=5500, original_statements=500),
        "medpic_official467": dict(questions=467, original_options=2335, includes_existing_development_questions=True),
    }
    for scope, details in descriptions.items():
        path = Path(root) / "evaluation" / scope / "metrics.json"
        metric = read(path, {})
        result[scope] = dict(details, status="complete" if metric.get("complete") else "pending",
                             metrics=str(path) if metric.get("complete") else None)
    write(Path(root) / "evaluation_populations.json", result)
    return result


def evaluate_model(root, scopes=("dev", "test", "nli4ct", "medpic"), store=None):
    root = Path(root)
    scopes = tuple(scopes)
    if any(scope != "dev" for scope in scopes):
        frozen = read(root / "frozen.json", {})
        if not frozen.get("frozen"):
            raise ValueError("Source parameters must be frozen before held-out/target evaluation")
    models = load_models(root)
    required = sorted({int(head) for model in models.values() for head in model["heads"]})
    store = store or DynamicStore(root, required_heads=required)
    results = {}
    items_path = root / "data/medpic_items.jsonl"
    items_path = items_path if items_path.exists() else TARGET / "medpic/items.jsonl"
    old_decisions = legacy.decisions_for
    legacy.decisions_for = lambda vv, ss, task=None: _decisions(vv, ss, task, items_path)
    try:
        for scope in scopes:
            if scope not in SCOPES:
                raise ValueError("Unsupported scope: " + scope)
            existing = read(root / "evaluation" / scope / "metrics.json", {})
            if existing.get("complete"):
                results[scope] = existing
                continue
            spec = scope_inputs(root, scope)
            if not spec["views"]:
                raise ValueError("No evaluation views prepared for " + scope)
            actual_count, expected_count = _population_count(scope, spec["views"])
            if expected_count is not None and actual_count != expected_count:
                raise ValueError(f"{scope} has {actual_count} evaluation units, expected {expected_count}; prepare the complete declared population")
            result = legacy.scope_evaluate(scope, store=store, models=models, out=root, **spec)
            result["source_parameters_frozen"] = scope != "dev"
            result["target_labels_used_for_fitting"] = False
            result["target_labels_used_for_calibration"] = False
            result["evaluation_population"] = scope
            if scope == "medpic_official467":
                result["population_note"] = "All 467 questions include existing development questions; descriptive evaluation."
            _native_metrics(scope, spec["views"], result, root)
            write(root / "evaluation" / scope / "metrics.json", result)
            results[scope] = result
    finally:
        legacy.decisions_for = old_decisions
    population_status(root)
    summary_path = root / "evaluation_summary.json"
    summary = read(summary_path, {"results": {}})
    summary["results"].update({scope: value["models"] for scope, value in results.items()})
    summary["completed_scopes"] = list(summary["results"])
    summary["frozen_group"] = "E1"
    write(summary_path, summary)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--scopes", nargs="+", default=["dev", "test", "nli4ct", "medpic"])
    args = parser.parse_args()
    scopes = [scope for value in args.scopes for scope in value.split(",") if scope]
    results = evaluate_model(args.root, scopes=scopes)
    print(json.dumps({"root": str(args.root), "completed_scopes": list(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

