"""Autonomous root-to-leaf evaluation with the same fixed node scorer.

Gold paths are used only after the predicted traversal has completed.
Original patient records and their per-case outputs stay in the server run directory.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

from .common import append, rows, write_json
from .data import canonical_guide, prompt
from .evaluation import margin, path_metrics, predict_fusion

METHODS = ("full", "direct", "semantic", "best_output", "aster")


def enumerate_contexts(cases):
    """Unique source/node mask requests, without patients or reference decisions."""
    contexts = {}
    for case in cases:
        guide = canonical_guide(case["guideline"])
        for node in guide["nodes"]:
            key = str(case["source_id"]) + ":" + str(node["id"])
            contexts.setdefault(key, dict(key=key, source_id=case["source_id"], gid=node["id"],
                condition=node["condition"], guideline=case["guideline"],
                definitions=case.get("definitions", [])))
    return list(contexts.values())


def _masks_map(masks):
    if masks is None:
        return {}
    if isinstance(masks, dict):
        return masks
    return {str(m["key"]): m for m in masks}


def _mask(masks, source_id, gid):
    key = str(source_id) + ":" + str(gid)
    if key not in masks:
        raise ValueError("Semantic mask is unavailable for " + key)
    item = masks[key]
    if isinstance(item, dict) and "value" in item:
        if item.get("status", "complete") != "complete":
            raise ValueError("Semantic mask construction failed for " + key)
        item = item["value"]
    if not isinstance(item, dict):
        raise ValueError("Malformed semantic mask for " + key)
    if "keep_action_ids" not in item or "keep_definition_indices" not in item:
        raise ValueError("Semantic mask lacks explicit keep lists for " + key)
    return item


def _routes(method, fusion_spec):
    if method in ("full", "aster"):
        return {"full"}
    if method in ("direct", "semantic"):
        return {method}
    if method != "best_output":
        raise ValueError("Unknown path method: " + method)
    if not fusion_spec:
        raise ValueError("No frozen development-selected output method")
    if fusion_spec["family"] == "calibration":
        return {fusion_spec["route"]}
    return {"full", fusion_spec["masked_route"]}


def run_paths(engine, cases, operators, fusion_spec, semantic_masks, outdir, shard=0, shards=1):
    """Score original cases with five autonomous methods, resuming whole paths.

    Calls on separate shards write separate files. An existing complete case-method
    record is reused; failed records are retained and may be superseded by a retry.
    The outdir must belong to the same frozen configuration when resuming.
    """
    if int(shards) < 1 or not 0 <= int(shard) < int(shards):
        raise ValueError("Invalid path worker shard")
    all_cases = list(cases)
    cases = all_cases[int(shard)::int(shards)]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"predictions_{shard}.jsonl"
    progress = outdir / f"status_{shard}.json"
    masks = _masks_map(semantic_masks)
    existing = {(r["case_id"], r["method"]): r for r in rows(output)}
    start = time.perf_counter()
    engine_start = dict(getattr(engine, "counts", {}))
    planned = len(cases) * len(METHODS)
    finished = sum(1 for c in cases for method in METHODS
                   if existing.get((c["case_id"], method), {}).get("status") == "complete")
    write_json(progress, dict(status="running", planned_paths=planned, finished_paths=finished,
                              cases=len(cases), shard=shard, shards=shards))
    for case_index, case in enumerate(cases):
        before = time.perf_counter()
        tree = canonical_guide(case["guideline"])
        nodes = {n["id"]: n for n in tree["nodes"]}
        if "G0" not in nodes:
            raise ValueError("Expected the shared controller root G0")
        active = {method: dict(gid="G0", path=[], visited=set(), logical_forwards=0)
                  for method in METHODS
                  if existing.get((case["case_id"], method), {}).get("status") != "complete"}
        cache, failures = {}, {}

        def finish(method, state, status="complete", error=None):
            nonlocal finished
            if status=="failed" and error is not None and (
                "Semantic mask is unavailable" in str(error) or "Semantic mask construction failed" in str(error)):
                status="unavailable_input"
            path = [dict(gid=x["gid"], label=x["label"]) for x in state["path"]]
            leaf = state["gid"] if state["gid"] in tree["leaves"] else None
            correct = status == "complete" and path == case["gold_path"] and leaf == case["gold_leaf"]
            record = dict(method=method, case_id=case["case_id"], source_id=case["source_id"],
                status=status, predictions=state["path"], predicted_path=path,
                predicted_leaf=leaf, returned_action=tree["leaves"].get(leaf),
                gold_path=case["gold_path"], gold_leaf=case["gold_leaf"], correct=bool(correct),
                logical_label_forwards=state["logical_forwards"],
                node_count=len(state["path"]), seconds_since_case_start=time.perf_counter()-before,
                protocol="autonomous traversal; gold path only used after prediction",
                raw_case_location="server-only path_cases.jsonl", shard=int(shard))
            if method == "best_output":
                record["frozen_fusion"] = fusion_spec
            if error is not None:
                record["error"] = str(error)
            append(output, record)
            existing[(case["case_id"], method)] = record
            finished += 1
            del active[method]

        while active:
            for method, state in list(active.items()):
                if state["gid"] in tree["leaves"]:
                    finish(method, state)
                elif state["gid"] not in nodes:
                    finish(method, state, "failed", "Predicted transition is absent from the tree")
                elif state["gid"] in state["visited"]:
                    finish(method, state, "failed", "Cycle encountered in the supplied tree")
            if not active:
                break
            gid = min((state["gid"] for state in active.values()),
                      key=lambda x: (int(x[1:]) if x[1:].isdigit() else math.inf, x))
            methods_here = [m for m, s in active.items() if s["gid"] == gid]
            requests = {}
            for method in methods_here:
                try:
                    requests[method] = _routes(method, fusion_spec)
                except (KeyError, ValueError) as exc:
                    finish(method, active[method], "failed", exc)
            needed = set().union(*requests.values()) if requests else set()
            for mode in sorted(needed):
                ast_needed = mode == "full" and "aster" in requests and (gid, "aster") not in cache
                natural_needed = any(mode in r and method != "aster" for method, r in requests.items())
                natural_needed = natural_needed and (gid, mode) not in cache
                if not natural_needed and not ast_needed:
                    continue
                context = None
                try:
                    mask = _mask(masks, case["source_id"], gid) if mode == "semantic" else None
                    text = prompt(case["guideline"], case.get("definitions", []),
                                  case["patient"], gid, mode=mode, semantic=mask)
                    context = engine.prepare(text)
                    if natural_needed:
                        result = engine.run(context, capture=False)
                        score = margin(result.get("ell"))
                        if score is None:
                            raise ValueError("Nonfinite natural label score")
                        cache[(gid, mode)] = score
                    if ast_needed:
                        result = engine.run(context, operators=operators, capture=False)
                        score = margin(result.get("ell"))
                        if score is None:
                            raise ValueError("Nonfinite ASTER label score")
                        cache[(gid, "aster")] = score
                except Exception as exc:
                    if natural_needed and (gid, mode) not in cache:
                        failures[(gid, mode)] = str(exc)
                    if ast_needed and (gid, "aster") not in cache:
                        failures[(gid, "aster")] = str(exc)
                finally:
                    del context
            for method in list(requests):
                if method not in active:
                    continue
                state = active[method]
                try:
                    if method == "aster":
                        key = (gid, "aster")
                        if key not in cache:
                            raise ValueError(failures.get(key, "ASTER score unavailable"))
                        score = cache[key]
                    elif method in ("full", "direct", "semantic"):
                        key = (gid, method)
                        if key not in cache:
                            raise ValueError(failures.get(key, "Reading score unavailable"))
                        score = cache[key]
                    else:
                        for route in requests[method]:
                            if (gid, route) not in cache:
                                raise ValueError(failures.get((gid, route), "Required fusion score unavailable"))
                        if fusion_spec["family"] == "calibration":
                            value = cache[(gid, fusion_spec["route"])]
                            score = predict_fusion(fusion_spec, value, value)
                        else:
                            score = predict_fusion(fusion_spec, cache[(gid, "full")],
                                                   cache[(gid, fusion_spec["masked_route"])])
                    score = margin(score)
                    if score is None:
                        raise ValueError("Nonfinite fused label score")
                    label = int(score > 0)
                    state["path"].append(dict(gid=gid, label=label, ell=score))
                    state["visited"].add(gid)
                    state["logical_forwards"] += len(requests[method])
                    state["gid"] = nodes[gid]["yes" if label else "no"]
                except Exception as exc:
                    finish(method, state, "failed", exc)
        write_json(progress, dict(status="running", completed_cases=case_index+1, cases=len(cases),
            planned_paths=planned, finished_paths=finished, elapsed_seconds=time.perf_counter()-start,
            shard=shard, shards=shards))
    records = [existing[(c["case_id"], m)] for c in cases for m in METHODS
               if (c["case_id"], m) in existing]
    costs = {k: value-engine_start.get(k, 0) for k, value in getattr(engine, "counts", {}).items()
             if isinstance(value, (int, float))}
    write_json(outdir / f"metrics_{shard}.json", path_metrics(records))
    write_json(progress, dict(status="complete", cases=len(cases), planned_paths=planned,
        completed_paths=sum(r.get("status") == "complete" for r in records),
        failed_paths=sum(r.get("status") != "complete" for r in records),
        elapsed_seconds=time.perf_counter()-start, engine_costs=costs, shard=shard, shards=shards))
    return records
