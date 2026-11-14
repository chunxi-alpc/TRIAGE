"""Official MEGA dynamic paths using this checkpoint's frozen E1 parameters.

No model is loaded on import. The outer scheduler assigns CUDA_VISIBLE_DEVICES.
Historical corrected results and historical selected-head caches are not read.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
import numpy as np
from experiments.aster_closed_form.data import canonical_guide, prompt

OFFICIAL_CASES = Path("/home/gcx/aster_next_20261109/inventory/official_all_path_cases.jsonl")
METHODS = ("raw", "bias_only", "aster_cr")


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def inventory(cases_path=OFFICIAL_CASES):
    cases = rows(cases_path)
    return dict(cases_file=str(cases_path), paths=len(cases),
                gold_path_nodes=sum(len(case["gold_path"]) for case in cases),
                sources=len({case["source_id"] for case in cases}),
                evaluation="official MEGA test, with earlier source/input-use metadata retained",
                methods=list(METHODS))


class Frozen:
    def __init__(self, model_root):
        self.root = Path(model_root)
        frozen = read(self.root / "frozen.json")
        if not frozen.get("frozen"):
            raise RuntimeError("Source-task parameter freezing must precede official evaluation")
        parameter_root = self.root / "models/E1/full"
        with np.load(parameter_root / "m2.npz", allow_pickle=False) as values:
            self.p = {key: values[key] for key in values.files}
        with np.load(parameter_root / "m3.npz", allow_pickle=False) as values:
            self.w = np.asarray(values["w"], np.float64).reshape(-1)
            self.b = float(values["b"])
        self.calibration = read(parameter_root / "calibration.json")
        self.bias_only = float(self.calibration["b"])
        self.heads = np.asarray(self.p["heads"], int).reshape(-1)
        self.coef = np.asarray(self.p["N"], np.float64) @ self.w
        self.parameter_root = str(parameter_root)

    def scores(self, ell, selected_heads):
        x = np.asarray(selected_heads, np.float32).reshape(-1)
        z = (x - self.p["mu"]) / self.p["scale"]
        return dict(raw=float(ell), bias_only=float(ell + self.bias_only),
                    aster_cr=float(ell + z @ self.coef + self.b))


def score_selected(engine, text, selected):
    """Original single-row final-token replay, copying only the selected heads."""
    import torch
    context = engine.prepare(text)
    by_layer, captured, handles = {}, {}, []
    for head in selected:
        by_layer.setdefault(int(head) // engine.n_heads, []).append(int(head) % engine.n_heads)
    def hook_for(layer, indices):
        def hook(module, args):
            captured[layer] = args[0][0, -1].reshape(engine.n_heads, engine.head_dim)[indices].detach()
        return hook
    if engine._active:
        raise RuntimeError("Overlapping model forwards")
    started = time.perf_counter()
    engine._active = True
    try:
        for layer, indices in by_layer.items():
            handles.append(engine.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook_for(layer, indices)))
        with torch.no_grad():
            embeddings = engine.model.get_input_embeddings()(context.input_ids[:, -1:]).detach()
            output = engine.model.model(inputs_embeds=embeddings, attention_mask=torch.ones_like(context.input_ids),
                                        past_key_values=engine._cache(context), use_cache=True,
                                        cache_position=torch.tensor([context.length - 1], device=engine.device),
                                        position_ids=torch.tensor([[context.length - 1]], device=engine.device))
            ell = float(engine._score_hidden(output.last_hidden_state[0, -1]))
            position = {(layer, head): i for layer, indices in by_layer.items() for i, head in enumerate(indices)}
            features = torch.stack([captured[int(head) // engine.n_heads][position[int(head) // engine.n_heads, int(head) % engine.n_heads]]
                                    for head in selected]).float().cpu().numpy().astype(np.float16)
        engine.counts["final_forwards"] += 1
        engine.counts["final_seconds"] += time.perf_counter() - started
        if not np.isfinite(ell) or not np.isfinite(features).all():
            raise ValueError("Nonfinite official MEGA score or activation")
        return dict(ell=ell, tokens=context.length, selected_heads=features)
    finally:
        for handle in handles:
            handle.remove()
        engine._active = False


def traverse(case, node_scores):
    """Evaluate all methods, including nodes reached only after a wrong branch."""
    tree = canonical_guide(case["guideline"])
    nodes = {node["id"]: node for node in tree["nodes"]}
    states = {method: dict(gid=tree["nodes"][0]["id"], path=[], error=None) for method in METHODS}
    while True:
        active = {method: state["gid"] for method, state in states.items()
                  if state["error"] is None and state["gid"] in nodes}
        if not active:
            break
        for gid in dict.fromkeys(active.values()):
            scores = node_scores(gid)
            for method, requested in active.items():
                if requested != gid:
                    continue
                state = states[method]
                if scores is None:
                    state["error"] = "unavailable_score"
                elif len(state["path"]) >= len(nodes):
                    state["error"] = "cycle"
                else:
                    label = int(scores[method] > 0)
                    state["path"].append(dict(gid=gid, label=label))
                    state["gid"] = nodes[gid]["yes" if label else "no"]
    results = []
    for method, state in states.items():
        results.append(dict(method=method, case_id=case["case_id"], source_id=case["source_id"],
                            correct=bool(state["error"] is None and state["path"] == case["gold_path"] and state["gid"] == case["gold_leaf"]),
                            leaf_correct=bool(state["error"] is None and state["gid"] == case["gold_leaf"]),
                            predicted_path=state["path"], gold_path=case["gold_path"],
                            predicted_leaf=state["gid"], gold_leaf=case["gold_leaf"],
                            error=state["error"], status="ok" if state["error"] is None else "unavailable"))
    return results


def run(model_key, model_root, shard=0, shards=1, engine=None, cases_path=OFFICIAL_CASES):
    """Run one scheduler-assigned shard; no physical GPU selection occurs here."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Expected 0 <= shard < shards")
    root = Path(model_root)
    output = root / "evaluation/mega_official"
    work = output / ("shard" + str(shard))
    work.mkdir(parents=True, exist_ok=True)
    (work / "features").mkdir(exist_ok=True)
    with (work / ".execution.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(model_key, root, work, shard, shards, engine, cases_path)


def _run(model_key, root, work, shard, shards, engine, cases_path):
    frozen = Frozen(root)
    all_cases = rows(cases_path)
    cases = all_cases[shard::shards]
    config = dict(model=model_key, parameter_directory=frozen.parameter_root, cases_file=str(cases_path),
                  shard=shard, shards=shards, selected_heads=frozen.heads.tolist(), methods=list(METHODS),
                  target_labels_used_for_fitting=False, cache_policy="this run, this checkpoint only",
                  physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    prior_config = read(work / "configuration.json") if (work / "configuration.json").exists() else None
    if prior_config is not None:
        for key in ("model", "parameter_directory", "cases_file", "shard", "shards", "selected_heads"):
            if prior_config[key] != config[key]:
                raise ValueError("Output directory belongs to a different evaluation: " + key)
    dump(work / "configuration.json", config)
    own = {record["id"]: record for record in rows(work / "nodes.jsonl")}
    completed = {(record["case_id"], record["method"]) for record in rows(work / "path_predictions.jsonl")}
    naturals = {record["id"] for record in rows(work / "natural_nodes.jsonl")}
    pending = [case for case in cases if any((case["case_id"], method) not in completed for method in METHODS)]
    started, forwards, reused = time.monotonic(), 0, 0
    for case_index, case in enumerate(pending):
        case_cache = {}
        def node(gid):
            nonlocal engine, forwards, reused
            if gid in case_cache:
                return case_cache[gid]
            key = case["case_id"] + ":" + gid
            cached = own.get(key)
            if cached is not None and cached.get("status") == "ok" and Path(cached["feature"]).exists():
                with np.load(cached["feature"], allow_pickle=False) as values:
                    if not np.array_equal(values["head_ids"], frozen.heads):
                        raise ValueError("Cached heads differ from the frozen model")
                    scores = frozen.scores(cached["ell"], values["selected_heads"])
                reused += 1
            else:
                if engine is None:
                    from .engine import Engine
                    engine = Engine(model_key, metadata_path=root / "official_model_metadata.json")
                text = prompt(case["guideline"], case.get("definitions", []), case["patient"], gid)
                record = dict(id=key, case_id=case["case_id"], source_id=case["source_id"], gid=gid)
                try:
                    result = score_selected(engine, text, frozen.heads)
                    feature = work / "features" / ("node_" + str(len(own)).zfill(8) + ".npz")
                    np.savez_compressed(feature, selected_heads=result.pop("selected_heads"), head_ids=frozen.heads)
                    record.update(result, feature=str(feature), status="ok")
                    with np.load(feature, allow_pickle=False) as values:
                        scores = frozen.scores(record["ell"], values["selected_heads"])
                except ValueError as exc:
                    # TooLong and other input-specific scoring failures retain the full denominator.
                    record.update(status="unavailable", error=str(exc))
                    scores = None
                append(work / "nodes.jsonl", record)
                own[key] = record
                forwards += 1
            case_cache[gid] = scores
            return scores
        for gold in case["gold_path"]:
            key = case["case_id"] + ":" + gold["gid"]
            if key in naturals:
                continue
            scores = node(gold["gid"])
            append(work / "natural_nodes.jsonl", dict(id=key, case_id=case["case_id"], source_id=case["source_id"],
                   gid=gold["gid"], gold_label=gold["label"], scores=scores,
                   status="ok" if scores is not None else "unavailable"))
            naturals.add(key)
        for record in traverse(case, node):
            if (record["case_id"], record["method"]) not in completed:
                record.update(model=model_key, evaluation_group="official_test",
                              usage_metadata={key: case.get(key) for key in
                                              ("current_source_role", "history_group", "known_earlier_input_use")})
                append(work / "path_predictions.jsonl", record)
                completed.add((record["case_id"], record["method"]))
        if case_index < 3 or (case_index + 1) % 20 == 0:
            dump(work / "status.json", dict(state="running", model=model_key, cases_completed=case_index + 1,
                 pending_at_start=len(pending), new_forwards=forwards, reused_nodes=reused,
                 seconds=time.monotonic() - started, last_case=case["case_id"]))
    peak = None
    if engine is not None:
        import torch
        peak = torch.cuda.max_memory_allocated(engine.device) / 2**30
    result = dict(state="complete", model=model_key, cases=len(cases), new_forwards=forwards,
                  reused_nodes=reused, seconds=time.monotonic() - started, peak_gib=peak,
                  counts=engine.counts if engine is not None else {}, shard=shard, shards=shards)
    dump(work / "done.json", result)
    return summarize(root, cases_path)


def _binary_summary(records, method, metric):
    grouped = defaultdict(lambda: np.zeros(3))
    for record in records:
        grouped[record["source_id"]] += [1, int(record["correct"]), int(record["raw_correct"])]
    ids = sorted(grouped)
    values = np.asarray([grouped[key] for key in ids])
    total = values.sum(0)
    rng = np.random.default_rng(20261111)
    draws = values[rng.integers(len(ids), size=(2000, len(ids)))].sum(1)
    low, high = np.percentile((draws[:, 1] - draws[:, 2]) / draws[:, 0], [2.5, 97.5])
    return dict(method=method, metric=metric, n=int(total[0]), correct=int(total[1]),
                accuracy=float(total[1] / total[0]), raw_correct=int(total[2]),
                delta_accuracy=float((total[1] - total[2]) / total[0]), sources=len(ids),
                paired_source_ci_low=float(low), paired_source_ci_high=float(high))


def summarize(model_root, cases_path=OFFICIAL_CASES):
    output = Path(model_root) / "evaluation/mega_official"
    paths, naturals = {}, {}
    for work in sorted(output.glob("shard*")):
        for record in rows(work / "path_predictions.jsonl"):
            paths[record["case_id"], record["method"]] = record
        for record in rows(work / "natural_nodes.jsonl"):
            naturals[record["id"]] = record
    raw = {case_id: record for (case_id, method), record in paths.items() if method == "raw"}
    stats = []
    for method in METHODS:
        records = [dict(record, raw_correct=raw[case_id]["correct"]) for (case_id, name), record in paths.items()
                   if name == method and case_id in raw]
        if records:
            summary = _binary_summary(records, method, "dynamic_path")
            summary["failed"] = sum(record.get("error") is not None for record in records)
            stats.append(summary)
        records = [dict(source_id=record["source_id"],
                        correct=record["scores"] is not None and (record["scores"][method] > 0) == bool(record["gold_label"]),
                        raw_correct=record["scores"] is not None and (record["scores"]["raw"] > 0) == bool(record["gold_label"]))
                   for record in naturals.values()]
        if records:
            summary = _binary_summary(records, method, "natural_gold_path_node")
            summary["failed"] = sum(record["scores"] is None for record in naturals.values())
            stats.append(summary)
    expected = inventory(cases_path)
    complete = (len(paths) == expected["paths"] * len(METHODS)
                and len(naturals) == expected["gold_path_nodes"])
    result = dict(state="complete" if complete else "partial", expected=expected,
                  path_records=len(paths), natural_node_records=len(naturals), metrics=stats)
    dump(output / "metrics.json", result)
    if stats:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
            writer.writeheader()
            writer.writerows(stats)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if args.inventory:
        dump(args.root / "official_mega_inventory.json", inventory())
    elif args.summarize:
        summarize(args.root)
    else:
        if not args.model:
            parser.error("--model is required for inference")
        run(args.model, args.root, args.shard, args.shards)
