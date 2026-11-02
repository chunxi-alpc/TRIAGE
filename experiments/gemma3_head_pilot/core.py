"""Shared data, head grouping, intervention and selection utilities."""
import json, math, os, random
from pathlib import Path
import torch
from torch import nn
from experiments.aster_representation.inputs import nodes, swap_actions, deployment, direct_prompt, terminal_paths
from experiments.aster_representation.intervention import ConditionalIntervention, objective

SEED = 20261101

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)

def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()

def journal(path, key):
    path = Path(path)
    if not path.exists(): return {}
    result = {}
    # Only an interrupted final write can be incomplete. Preserve it separately.
    with path.open("rb+") as handle:
        while True:
            start = handle.tell(); line = handle.readline()
            if not line: break
            try: value = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if handle.read(1): raise
                path.with_name(path.name + ".interrupted-line").write_bytes(line)
                handle.truncate(start); break
            k = key(value)
            if k in result: raise ValueError("Duplicate journal cell: " + str(k))
            result[k] = value
    return result

def group_indices(groups, config):
    shared = config.num_attention_heads // config.num_key_value_heads
    return [i for g in sorted(groups) for i in range(g * shared * config.head_dim, (g + 1) * shared * config.head_dim)]

def screen_cases(cases):
    by_source = {}
    for case in cases: by_source.setdefault(case["source_id"], []).append(case)
    rng = random.Random(SEED); output = []
    for source in sorted(by_source):
        values = sorted(by_source[source], key=lambda c: c["case_id"])
        rng.shuffle(values); output.extend(values[:2])
    return output

def separation(sums, squares, counts):
    means = sums / counts[:, None, None, None].clamp_min(1)
    variances = (squares / counts[:, None, None, None].clamp_min(1) - means.square()).clamp_min(0)
    return ((means[0] - means[1]).norm(dim=-1) / (variances.mean(0).sum(-1).sqrt() + 1e-8)).tolist()

def rank_heads(records, distinctness, layer_types, group_count):
    layers = len(layer_types); by_group = {}
    for r in records:
        by_group.setdefault((r["layer"], r["group"]), []).append(r)
    ranked = []
    for layer in range(layers):
        group_rows = []
        for group in range(group_count):
            values = by_group.get((layer, group), [])
            good = [r for r in values if r.get("status") == "complete"]
            failed = len(values) - len(good)
            positive = sum(r["plus_margin"] - r["base_margin"] for r in good) / len(good) if good else None
            reverse = sum(r["minus_margin"] - r["base_margin"] for r in good) / len(good) if good else None
            group_rows.append(dict(layer=layer, group=group, attention_type=layer_types[layer],
                                   positive_gain=positive, reverse_gain=reverse,
                                   separation=float(distinctness[layer][group]), cases=len(values), failures=failed))
        group_rows.sort(key=lambda r: (r["failures"] > 0 or r["positive_gain"] is None,
                                      -(r["positive_gain"] if r["positive_gain"] is not None else -1e30),
                                      -r["separation"], r["group"]))
        chosen = group_rows[:4]
        score = sum(r["positive_gain"] for r in chosen) / 4 if all(r["positive_gain"] is not None for r in chosen) else None
        ranked.append(dict(layer=layer, attention_type=layer_types[layer], groups=[r["group"] for r in chosen],
                           score=score, separation=sum(r["separation"] for r in chosen) / 4,
                           failures=sum(r["failures"] for r in chosen), ranking=group_rows))
    ranked.sort(key=lambda r: (r["failures"] > 0 or r["score"] is None,
                              -(r["score"] if r["score"] is not None else -1e30), -r["separation"], r["layer"]))
    return dict(candidates=ranked[:3], layer_ranking=ranked, rule="positive margin gain; separation and ascending indices break ties")

class HeadIntervention(nn.Module):
    """Learn only in selected pre-o_proj query-head coordinates."""
    def __init__(self, config, groups, rank=4):
        super().__init__()
        self.groups = sorted(groups)
        self.register_buffer("indices", torch.tensor(group_indices(groups, config), dtype=torch.long), persistent=False)
        self.edit = ConditionalIntervention(len(self.indices), rank=rank, variant="no_js")

    def forward(self, x, blind, projection, force_zero=False):
        if force_zero: return x, x.float().sum() * 0, x.float().sum() * 0
        selected = x.index_select(-1, self.indices)
        reference = blind.to(x.device).index_select(-1, self.indices)
        changed, _, gate = self.edit(selected, reference)
        result = x.index_copy(-1, self.indices, changed)
        # Measure edit in the same post-projection space as the whole-layer method.
        delta = torch.nn.functional.linear((changed.float() - selected.float()),
                                            projection.weight[:, self.indices].detach().float())
        with torch.no_grad():
            original = torch.nn.functional.linear(x.to(projection.weight.dtype), projection.weight, projection.bias).float()
        magnitude = delta.square().sum() / original.square().sum().clamp_min(1e-8)
        return result, magnitude, gate

def make_module(spec, config, device):
    if spec["kind"] == "whole":
        return ConditionalIntervention(config.hidden_size, rank=4, variant="no_js").to(device)
    return HeadIntervention(config, spec["groups"], rank=4).to(device)

def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def prediction_summary(cases, records):
    lookup = {(r["case_id"], r["view"]): r for r in records}
    orig, both, failed, magnitudes = 0, 0, 0, []
    for case in cases:
        successes = []
        for view in ("original", "swapped"):
            row = lookup.get((case["case_id"], view))
            good = row is not None and row["status"] == "complete"
            correct = good and row["path"] == case["gold_rules"] and row["leaf"] == case["gold_leaf"]
            successes.append(bool(correct))
            failed += not good
            if good and view == "original": magnitudes.extend(row.get("magnitudes", []))
        orig += successes[0]; both += all(successes)
    return dict(cases=len(cases), original_correct=orig, both_views_correct=both,
                path_accuracy=orig / len(cases), failures=int(failed),
                mean_magnitude=sum(magnitudes) / len(magnitudes) if magnitudes else 0.)

def selection_key(summary, layer, epoch):
    return (summary["original_correct"], summary["both_views_correct"], -summary["mean_magnitude"], -layer, -epoch)

def paired(cases, methods, baseline="zero"):
    by_method = {name: {(r["case_id"], r["view"]): r for r in records} for name, records in methods.items()}
    def correct(name, case):
        r = by_method[name].get((case["case_id"], "original"))
        return bool(r and r["status"] == "complete" and r["path"] == case["gold_rules"] and r["leaf"] == case["gold_leaf"])
    output = {}
    for name in methods:
        cells = [[0, 0], [0, 0]]
        for case in cases: cells[int(correct(baseline, case))][int(correct(name, case))] += 1
        output[name] = dict(pair_table=cells, corrections=cells[0][1], regressions=cells[1][0], net=cells[0][1] - cells[1][0])
    return output
