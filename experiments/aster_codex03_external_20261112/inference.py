"""Collect pair scores and complete dynamic paths from a supplied official-method scorer.

This file does not implement any steering/erasure/PEFT method. The caller supplies
a scorer backed by the audited implementation. Each prompt is read unchanged.
Paths are chosen from model predictions only; gold labels are not read here.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Callable

from .data import get_prompt, load_bundle
from experiments.aster_closed_form.data import canonical_guide


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _uid(view):
    return view.get("uid", view.get("view_id", view.get("id")))


def score_split(method: str, split: str, score_one: Callable,
                scores_path: str | Path, metadata: dict | None = None,
                execution_kind: str = "model_inference"):
    """Score exact prompts with score_one(prompt, uid) -> float or score dict.

    A score dict requires ell and may include forward_count, tokens, and
    peak_memory_gib. The scorer must finish device work before returning for
    accurate wall time. Its intervention must cover the intended prompt tokens;
    no prefix/hidden state from the raw model is supplied by this collector.
    Reusing an output requires exactly the same method/split/configuration.
    """
    if not method:
        raise ValueError("A method name is required")
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = dict(method=method, split=split, metadata=metadata or {},
                         execution_kind=execution_kind)
    config_path = scores_path.with_suffix(".configuration.json")
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Existing score configuration differs; use a separate output")
    elif scores_path.exists() and scores_path.stat().st_size:
        raise ValueError("Existing scores have no configuration; use a separate output")
    else:
        _write_json(config_path, configuration)

    records = {}
    if scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["uid"]
            if key in records:
                raise ValueError("Duplicate saved score: " + key)
            if row.get("method") != method or row.get("split") != split:
                raise ValueError("Saved score belongs to a different run")
            if not math.isfinite(float(row["ell"])):
                raise ValueError("Nonfinite saved score: " + key)
            records[key] = row

    bundle = load_bundle(split)
    started = time.perf_counter()
    new_inputs = 0
    with scores_path.open("a", encoding="utf-8", buffering=1) as output:
        def score(key):
            nonlocal new_inputs
            if key not in records:
                before = time.perf_counter()
                answer = score_one(get_prompt(key), key)
                if not isinstance(answer, dict):
                    answer = dict(ell=float(answer))
                ell = float(answer["ell"])
                if not math.isfinite(ell):
                    raise ValueError("Nonfinite model score: " + key)
                row = dict(uid=key, method=method, split=split, ell=ell,
                           prediction=int(ell > 0), status="ok",
                           execution_kind=execution_kind,
                           scoring_wall_seconds=time.perf_counter() - before,
                           forward_count=answer.get("forward_count"),
                           tokens=answer.get("tokens"),
                           peak_memory_gib=answer.get("peak_memory_gib"))
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                records[key] = row
                new_inputs += 1
            return float(records[key]["ell"])

        for view in bundle["views"]:
            score(_uid(view))

        path_maps = defaultdict(dict)
        for view in bundle["path_views"]:
            variant = view.get("variant", "original")
            if variant == "full":
                variant = "original"
            path_maps[(view["case_id"], variant)][view["gid"]] = _uid(view)
        paths = []
        variants = ("original", "action_presence", "action_neutral", "action_swap")
        for case in bundle["cases"]:
            tree = canonical_guide(case["guideline"])
            nodes = {n["id"]: n for n in tree["nodes"]}
            for variant in variants:
                node_uids = path_maps[(case["case_id"], variant)]
                gid = tree["nodes"][0]["id"]
                visited, seen = [], set()
                while gid in nodes:
                    if gid in seen:
                        raise ValueError("Cycle in path for " + str(case["case_id"]))
                    seen.add(gid)
                    if gid not in node_uids:
                        raise ValueError("No input for visited node: " + str(gid))
                    key = node_uids[gid]
                    label = int(score(key) > 0)
                    visited.append(dict(gid=gid, uid=key, label=label))
                    gid = nodes[gid]["yes" if label else "no"]
                paths.append(dict(case_id=case["case_id"],
                                  source_id=case["source_id"], variant=variant,
                                  path=visited, leaf=gid, visited_nodes=len(visited)))

    path_file = scores_path.with_suffix(".traversed_paths.jsonl")
    with path_file.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(json.dumps(path, ensure_ascii=False, allow_nan=False) + "\n")
    current = list(records.values())
    def total_if_known(field):
        values = [r.get(field) for r in current]
        return sum(values) if values and all(v is not None for v in values) else None
    peaks = [r.get("peak_memory_gib") for r in current]
    result = dict(metadata or {})
    result.update(
        method=method, split=split, execution_kind=execution_kind,
        actual_experiment_results=execution_kind == "model_inference",
        inference_wall_seconds=(sum(r["scoring_wall_seconds"] for r in current)
                                if execution_kind == "model_inference" else None),
        inference_forward_count=(total_if_known("forward_count")
                                 if execution_kind == "model_inference" else None),
        inference_peak_memory_gib=(max(peaks) if peaks and all(p is not None for p in peaks)
                                   and execution_kind == "model_inference" else None),
        inference_input_count=len(current), inference_token_count=total_if_known("tokens"),
        candidate_input_count=len({_uid(v) for v in bundle["views"] + bundle["path_views"]}),
        pair_view_records=len(bundle["views"]),
        path_count=len(paths), path_visited_node_count=sum(p["visited_nodes"] for p in paths),
        new_inputs_this_call=new_inputs, collection_wall_seconds_this_call=time.perf_counter()-started,
        scoring_policy="All pair views plus all dynamically visited natural/interference path nodes; no gold labels used to choose nodes.",
        timing_scope="Prompt lookup and scorer callback only; model loading and training are excluded. Cached validation has no model inference cost.",
        scores_file=str(scores_path), traversed_paths_file=str(path_file),
    )
    _write_json(scores_path.with_suffix(".metadata.json"), result)
    return result
