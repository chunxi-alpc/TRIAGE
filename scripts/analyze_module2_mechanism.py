#!/usr/bin/env python3
"""Reproducible analysis of the 2026-10-28 module-two mechanism pilot.

The fitting/selection split is never changed here.  Examples are explicitly
exploratory illustrations selected from a fully exported confirmation census.
This script does not run, fit, or tune the language model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, float) and not math.isfinite(v)
                             else v for k, v in row.items()})


def group_mean(values, groups):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    return np.asarray([np.mean(values[groups == g], axis=0)
                       for g in sorted(set(groups.tolist()))])


def clustered_interval(values, groups, seed=261028, repeats=5000):
    """Guide-level percentile resampling; not a model-selection interval."""
    means = group_mean(values, groups)
    if len(means) < 2:
        return float(np.mean(means)), None, None
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, len(means), size=(repeats, len(means)))].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    return float(means.mean()), float(low), float(high)


def config_id(row):
    return row["id"].rsplit(":target", 1)[0]


def selected_heads(value):
    if isinstance(value, dict):
        for key in ("selected_heads", "heads", "selected", "accepted_heads"):
            if key in value:
                return selected_heads(value[key])
        raise ValueError("Selection JSON does not contain selected heads")
    result = []
    for head in value:
        if isinstance(head, dict):
            result.append((int(head["layer"]), int(head.get("kv_head", head.get("head")))))
        else:
            result.append(tuple(map(int, head)))
    if any(len(x) != 2 for x in result) or len(result) != len(set(result)):
        raise ValueError("Invalid or duplicate selected head")
    return result


def take_array(pack, *names):
    for name in names:
        if name in pack:
            return np.asarray(pack[name])
    raise ValueError(f"Missing array; expected one of {names}")


def analyze_representations(offline, online, operators, rows, heads, output):
    """Compute all split/head/method statistics without choosing new heads."""
    keys = take_array(offline, "offline_keys", "offline_means", "keys").astype(np.float64)
    q_online = take_array(online, "online_effective_queries", "effective_queries").astype(np.float64)
    q_offline = take_array(offline, "offline_effective_queries", "effective_queries").astype(np.float64)
    if keys.ndim != 5 or keys.shape[0] != len(rows) or keys.shape[1] != 3:
        raise ValueError("Offline keys must have [triplet, positive/negative/neutral, layer, KV, dim]")
    if q_online.shape[:2] != (len(rows), 2) or q_online.shape[2:] != keys.shape[2:]:
        raise ValueError("Online queries must have [triplet, positive/negative, layer, KV, dim]")
    if q_offline.shape[0] != len(rows) or q_offline.shape[1] < 2 or q_offline.shape[2:] != keys.shape[2:]:
        raise ValueError("Offline query array does not match keys")
    if not np.isfinite(keys).all() or not np.isfinite(q_offline).all():
        raise ValueError("Incomplete or nonfinite offline representation measurement")
    fit_mask = np.asarray([r["split"] == "fit" for r in rows])
    if not np.isnan(q_online[fit_mask]).all():
        raise ValueError("This frozen protocol does not collect online queries on fitting guides")
    if not np.isfinite(q_online[~fit_mask]).all():
        raise ValueError("Incomplete or nonfinite selection/confirmation online query measurement")
    delta = take_array(operators, "delta").astype(np.float64)
    raw = take_array(operators, "raw_direction", "raw_unit").astype(np.float64)
    if delta.shape != keys.shape[2:] or raw.shape != delta.shape:
        raise ValueError("Operator dimensions do not match representation dimensions")
    if not np.isfinite(delta).all() or not np.isfinite(raw).all():
        raise ValueError("Nonfinite frozen operator")
    if any(not (0 <= l < delta.shape[0] and 0 <= h < delta.shape[1]) for l, h in heads):
        raise ValueError("Selected head is outside the model geometry")
    if "random_direction" in operators:
        random = operators["random_direction"].astype(np.float64)
        random_source = "saved operator"
    else:
        random = np.random.default_rng(261028).normal(size=delta.shape)
        random /= np.linalg.norm(random, axis=-1, keepdims=True)
        random_source = "analysis-only, one fixed draw, numpy.default_rng(261028)"
    directions = {"projected": delta, "raw_mean": raw, "random": random}
    d = keys[:, 0] - keys[:, 1]
    missing = .5 * (keys[:, 0] + keys[:, 1]) - keys[:, 2]
    dim = keys.shape[-1]
    groups = np.asarray([r["group_id"] for r in rows])
    splits = np.asarray([r["split"] for r in rows])
    norm_d = np.linalg.norm(d, axis=-1)
    headset = set(heads)
    summary, points, interactions = [], [], []
    all_values = {}
    for method, direction in directions.items():
        norm = np.linalg.norm(direction, axis=-1)
        z = np.einsum("nlhd,lhd->nlh", d, direction)
        u = np.einsum("nlhd,lhd->nlh", missing, direction)
        cosine = np.divide(z, norm_d * norm, out=np.full_like(z, np.nan),
                           where=(norm_d * norm) > 1e-12)
        read_on = np.einsum("nslhd,lhd->nslh", q_online, direction) / np.sqrt(dim)
        read_off = np.einsum("nslhd,lhd->nslh", q_offline[:, :2], direction) / np.sqrt(dim)
        all_values[method] = {"z": z, "u": u, "cosine": cosine,
                              "read_online": read_on, "read_offline": read_off}
        for split in ("fit", "select", "confirm"):
            idx = np.flatnonzero(splits == split)
            if not len(idx):
                continue
            for layer in range(delta.shape[0]):
                for head in range(delta.shape[1]):
                    zz, uu, cc = z[idx, layer, head], u[idx, layer, head], cosine[idx, layer, head]
                    rp, rn = read_on[idx, 0, layer, head], read_on[idx, 1, layer, head]
                    valid_on = bool(np.isfinite(rp).all() and np.isfinite(rn).all())
                    point = {"split": split, "method": method, "layer": layer, "kv_head": head,
                             "selected": (layer, head) in headset, "n_triplets": len(idx),
                             "n_guides": len(set(groups[idx])), "mean_z": float(zz.mean()),
                             "std_z": float(zz.std()), "mean_abs_unknown_offset": float(np.abs(uu).mean()),
                             "separation": float(zz.mean() / (zz.std() + np.abs(uu).mean() + 1e-8)),
                             "positive_z_count": int((zz > 0).sum()),
                             "near_zero_z_count": int((np.abs(zz) <= 1e-9).sum()),
                             "positive_z_fraction": float((zz > 0).mean()),
                             "mean_cosine": float(np.nanmean(cc)) if np.isfinite(cc).any() else float("nan"),
                             "near_zero_difference_count": int((norm_d[idx, layer, head] <= 1e-12).sum()),
                             "online_available": valid_on,
                             "read_positive": float(rp.mean()) if valid_on else float("nan"),
                             "read_negative": float(rn.mean()) if valid_on else float("nan"),
                             "positive_read_positive_fraction": float((rp > 0).mean()) if valid_on else float("nan"),
                             "positive_read_negative_fraction": float((rn > 0).mean()) if valid_on else float("nan"),
                             "offline_read_positive": float(read_off[idx, 0, layer, head].mean()),
                             "offline_read_negative": float(read_off[idx, 1, layer, head].mean())}
                    summary.append(point)
                    if (layer, head) in headset:
                        for i in idx:
                            points.append({"triplet_id": rows[i]["id"], "config_id": config_id(rows[i]),
                                           "group_id": groups[i], "split": split,
                                           "target_rule_id": rows[i]["target_rule_id"],
                                           "action_permutation": rows[i]["controls"]["action_permutation"],
                                           "rule_order_index": rows[i]["controls"]["rule_order_index"],
                                           "method": method, "layer": layer, "kv_head": head,
                                           "z": float(z[i, layer, head]), "cosine": float(cosine[i, layer, head]),
                                           "unknown_offset": float(u[i, layer, head]),
                                           "read_positive": float(read_on[i, 0, layer, head]),
                                           "read_negative": float(read_on[i, 1, layer, head])})
        pairs = {}
        for i, row in enumerate(rows):
            pairs.setdefault(config_id(row), {})[row["target_rule_id"]] = i
        for identity, pair in sorted(pairs.items()):
            if set(pair) != {"A", "B"}:
                raise ValueError(f"Expected exactly A/B target rows for {identity}")
            a, b = pair["A"], pair["B"]
            if groups[a] != groups[b] or splits[a] != splits[b]:
                raise ValueError("A/B interaction crosses guide or split")
            for layer, head in heads:
                za, zb = z[a, layer, head], z[b, layer, head]
                interactions.append({"config_id": identity, "group_id": groups[a], "split": splits[a],
                                     "method": method, "layer": layer, "kv_head": head,
                                     "action_permutation": rows[a]["controls"]["action_permutation"],
                                     "rule_order_index": rows[a]["controls"]["rule_order_index"],
                                     "z_A": float(za), "z_B": float(zb),
                                     "applicability_interaction": float((za + zb) / 2),
                                     "common_fact_component": float((za - zb) / 2)})
    write_csv(output / "head_direction_metrics.csv", summary)
    write_csv(output / "heldout_direction_points.csv", points)
    write_csv(output / "rule_relative_effects.csv", interactions)
    from src.module2.mechanism import enumerate_group_sign_directions
    fit = np.flatnonzero(splits == "fit")
    confirm = np.flatnonzero(splits == "confirm")
    sign_results = enumerate_group_sign_directions(d[fit], groups[fit], take_array(operators, "basis", "U"))
    sign_rows = []
    for si, signs in enumerate(sign_results["signs"]):
        for layer, head in heads:
            direction = sign_results["directions"][si, layer, head]
            zz = d[confirm, layer, head] @ direction
            rr = np.einsum("nsd,d->ns", q_online[confirm, :, layer, head], direction) / np.sqrt(dim)
            dn, original_norm = np.linalg.norm(direction), np.linalg.norm(delta[layer, head])
            sign_rows.append({"configuration_index": si, "group_signs": " ".join("+" if x == 1 else "-" for x in signs),
                              "all_original_labels": si == sign_results["all_positive_index"],
                              "layer": layer, "kv_head": head,
                              "projected_mean_norm": float(sign_results["projected_norms"][si, layer, head]),
                              "near_zero_mean": bool(sign_results["projected_norms"][si, layer, head] <= 1e-6),
                              "cosine_to_original": float(direction @ delta[layer, head] / (dn * original_norm)) if dn * original_norm > 0 else float("nan"),
                              "confirmation_mean_z": float(zz.mean()), "confirmation_positive_z_fraction": float((zz > 0).mean()),
                              "confirmation_read_positive": float(rr[:, 0].mean()),
                              "confirmation_read_negative": float(rr[:, 1].mean())})
    write_csv(output / "group_sign_controls.csv", sign_rows)
    write_json(output / "group_sign_controls_metadata.json", {
        "n_configurations": len(sign_results["signs"]), "fit_guideline_order": sign_results["group_ids"],
        "all_original_index": sign_results["all_positive_index"],
        "interpretation": "Exhaustive directional sensitivity diagnostic, holding the fitted spectral basis and selected heads fixed. Pair sign changes leave token outer products invariant. No spectral null, randomization p-value, or head reselection is claimed.",
        "near_zero_projected_mean_threshold": 1e-6})
    query_rows = []
    for method in ("projected", "raw_mean", "random", "reversed"):
        key = "online_per_query_" + method
        if key not in online:
            continue
        per_query = np.asarray(online[key], dtype=np.float64)
        rms = np.asarray(online["online_unit_rms_" + method], dtype=np.float64)
        token_positive = np.asarray(online["online_token_positive_" + method], dtype=np.float64)
        for split in ("select", "confirm"):
            idx = np.flatnonzero(splits == split)
            if not np.isfinite(per_query[idx]).all() or not np.isfinite(rms[idx]).all() or not np.isfinite(token_positive[idx]).all():
                raise ValueError("Incomplete per-query/token diagnostics")
            for layer, head in heads:
                for state, state_name in enumerate(("positive", "negative")):
                    b = per_query[idx, state, layer, head]
                    unit_rms = rms[idx, state, layer, head]
                    cosine_to_constant = np.divide(b.mean(axis=-1), unit_rms, out=np.full(len(idx), np.nan), where=unit_rms > 0)
                    for local_query in range(b.shape[-1]):
                        query_rows.append({"split": split, "method": method, "layer": layer, "kv_head": head,
                                           "query_head": head * b.shape[-1] + local_query, "state": state_name,
                                           "n_target_triplets": len(idx), "mean_query_response": float(b[:, local_query].mean()),
                                           "min_query_response": float(b[:, local_query].min()),
                                           "positive_query_fraction": float((b[:, local_query] > 0).mean()),
                                           "mean_token_positive_fraction_across_shared_queries": float(token_positive[idx, state, layer, head].mean()),
                                           "min_token_positive_fraction_across_shared_queries": float(token_positive[idx, state, layer, head].min()),
                                           "mean_score_pattern_cosine_to_constant": float(np.nanmean(cosine_to_constant)) if np.isfinite(cosine_to_constant).any() else float("nan")})
    write_csv(output / "online_query_diagnostics.csv", query_rows)
    return {"summary": summary, "points": points, "interactions": interactions,
            "values": all_values, "random_source": random_source, "sign_controls": sign_rows,
            "query_diagnostics": query_rows}


COLORS = {"projected": "#007F86", "raw_mean": "#DD8733", "random": "#818B9D",
          "reverse": "#B55B86", "direct_score_bias": "#4773B3"}
LABELS = {"projected": "Spectral mean", "raw_mean": "Raw mean", "random": "Random unit direction"}


METHOD_ALIASES = {"projected_mean": "projected", "spectral": "projected", "raw": "raw_mean",
                  "random_direction": "random", "reversed": "reverse", "reversed_direction": "reverse",
                  "direct_bias": "direct_score_bias", "vanilla": "baseline", "none": "baseline"}

# Frozen in PROTOCOL.zh.md before this run.  Absence of an entire condition
# must not be mistaken for a complete observed subset.
FROZEN_CONDITIONS = [
    {"method": "baseline", "mode": "none", "gain": 0},
    {"method": "projected", "mode": "both", "gain": 1},
    {"method": "projected", "mode": "both", "gain": 4},
    {"method": "projected", "mode": "enhance", "gain": 4},
    {"method": "projected", "mode": "suppress", "gain": 4},
    {"method": "raw_mean", "mode": "both", "gain": 4},
    {"method": "random", "mode": "both", "gain": 4},
    {"method": "reverse", "mode": "both", "gain": 4},
    {"method": "direct_score_bias", "mode": "both", "gain": 4},
]


def causal_rows(path):
    """Normalize explicitly named scalar fields, without inferring absent effects."""
    rows, unique = [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        source = json.loads(line)
        method = METHOD_ALIASES.get(source["method"], source["method"])
        gain = float(source["gain"])
        mode = source.get("mode", "none" if gain == 0 else "both")
        if gain == 0:
            method, mode = "baseline", "none"
        row = {"case_id": source["case_id"], "group_id": source.get("group_id"),
               "patient_state": source["patient_state"], "gold_rule": source["gold_rule"],
               "method": method, "mode": mode, "gain": gain}
        if not isinstance(row["patient_state"], bool):
            raise ValueError("Patient state must be the observed Boolean decisive fact")
        readout = source.get("readout_spec", {})
        row["readout_schema"] = readout.get("schema_version", "unspecified")
        row["candidate_token_ids"] = json.dumps(readout.get("candidate_token_ids"))
        row["candidate_continuations"] = json.dumps(readout.get("candidate_continuations"))
        if row["gold_rule"] not in ("A", "B"):
            raise ValueError("Expected A/B gold rule in this pilot")
        for role in ("mass", "logits"):
            for rule in ("A", "B"):
                key = f"{role}_{rule}"
                value = source.get(key)
                if value is None and isinstance(source.get(role), dict):
                    value = source[role].get(rule)
                if value is None:
                    raise ValueError(f"Missing {key} in causal record")
                row[key] = float(value)
                if not math.isfinite(row[key]):
                    raise ValueError("Nonfinite causal measurement")
        if not 0 < row["mass_A"] <= 1 or not 0 < row["mass_B"] <= 1:
            raise ValueError("Rule attention masses must lie in (0,1]")
        gold = row["gold_rule"]
        other = "B" if gold == "A" else "A"
        row["gold_mass"], row["other_mass"] = row[f"mass_{gold}"], row[f"mass_{other}"]
        row["log_mass_ratio"] = math.log(row["gold_mass"] / row["other_mass"])
        row["gold_logit_margin"] = row[f"logits_{gold}"] - row[f"logits_{other}"]
        row["prediction"] = "A" if row["logits_A"] >= row["logits_B"] else "B"
        row["correct"] = row["prediction"] == gold
        if "prediction" in source and source["prediction"] != row["prediction"]:
            raise ValueError("Saved prediction disagrees with the supplied candidate logits")
        if "correct" in source and source["correct"] != row["correct"]:
            raise ValueError("Saved correctness disagrees with the supplied candidate logits")
        key = (row["case_id"], gold, method, mode, gain)
        if key in unique:
            for field in ("mass_A", "mass_B", "logits_A", "logits_B"):
                if not math.isclose(row[field], unique[key][field], rel_tol=0, abs_tol=1e-8):
                    raise ValueError("Conflicting duplicate causal condition")
            continue
        unique[key] = row
        rows.append(row)
    return rows


def analyze_causal(path, triplets, output, *, protocol=None):
    """Export the whole confirmation census before choosing illustrations."""
    rows = causal_rows(path)
    configs = {config_id(r): r for r in triplets if r["split"] == "confirm"}
    expected = {(identity, rule) for identity in configs for rule in ("A", "B")}
    baseline = {(r["case_id"], r["gold_rule"]): r for r in rows if r["method"] == "baseline"}
    if set(baseline) != expected:
        raise ValueError(f"Baseline coverage mismatch: {len(baseline)}/{len(expected)}")
    for row in rows:
        if row["case_id"] not in configs:
            raise ValueError("A non-confirmation case entered causal analysis")
        row["group_id"] = configs[row["case_id"]]["group_id"]
        metadata = configs[row["case_id"]]
        required = [c["required"] for c in metadata["rules"][row["gold_rule"]]["conditions"]
                    if str(c["node_index"]) == str(metadata["decisive_node_index"])]
        if len(required) != 1 or row["patient_state"] is not required[0]:
            raise ValueError("Gold branch and actual decisive patient fact disagree")
        row["action_permutation"] = configs[row["case_id"]]["controls"]["action_permutation"]
        row["rule_order_index"] = configs[row["case_id"]]["controls"]["rule_order_index"]
        base = baseline[(row["case_id"], row["gold_rule"])]
        for field in ("log_mass_ratio", "gold_logit_margin", "gold_mass", "other_mass"):
            row[f"change_{field}"] = row[field] - base[field]
        row["correct_to_wrong"] = base["correct"] and not row["correct"]
        row["wrong_to_correct"] = not base["correct"] and row["correct"]
        row["answer_changed"] = row["prediction"] != base["prediction"]
        row["signed_mass_change"] = .5 * (row["change_gold_mass"] - row["change_other_mass"])
    conditions = sorted({(r["method"], r["mode"], r["gain"]) for r in rows})
    if protocol is None:
        protocol = FROZEN_CONDITIONS
    if protocol is not None:
        required = set()
        for c in protocol:
            method = METHOD_ALIASES.get(c["method"], c["method"])
            gain = float(c["gain"])
            mode = c["mode"]
            if gain == 0:
                method, mode = "baseline", "none"
            required.add((method, mode, gain))
        if set(conditions) != required:
            raise ValueError("Observed causal conditions differ from the frozen condition list")
    coverage, summary, pairs = [], [], []
    for condition in conditions:
        method, mode, gain = condition
        current = [r for r in rows if (r["method"], r["mode"], r["gain"]) == condition]
        actual = {(r["case_id"], r["gold_rule"]) for r in current}
        coverage.append({"method": method, "mode": mode, "gain": gain,
                         "expected": len(expected), "actual": len(actual), "complete": actual == expected})
        if actual != expected:
            raise ValueError(f"Incomplete causal condition {condition}: {len(actual)}/{len(expected)}")
        values = {"method": method, "mode": mode, "gain": gain,
                  "n_patients": len(current), "n_paired_configs": len(configs),
                  "n_unique_prompts": len({(r["case_id"], r["patient_state"]) for r in current}),
                  "n_base_patient_states": len({(r["group_id"], r["patient_state"]) for r in current}),
                  "n_guides": len({r["group_id"] for r in current}),
                  "accuracy": float(np.mean([r["correct"] for r in current])),
                  "answer_changes": sum(r["answer_changed"] for r in current),
                  "corrections": sum(r["wrong_to_correct"] for r in current),
                  "correct_to_wrong": sum(r["correct_to_wrong"] for r in current),
                  "gold_mass_increased_count": sum(r["change_gold_mass"] > 0 for r in current),
                  "other_mass_decreased_count": sum(r["change_other_mass"] < 0 for r in current),
                  "both_mass_directions_count": sum(r["change_gold_mass"] > 0 and r["change_other_mass"] < 0 for r in current),
                  "positive_ratio_change_fraction": float(np.mean([r["change_log_mass_ratio"] > 0 for r in current]))}
        for field in ("change_log_mass_ratio", "change_gold_logit_margin", "signed_mass_change", "change_gold_mass", "change_other_mass"):
            mean, low, high = clustered_interval([r[field] for r in current], [r["group_id"] for r in current])
            values[field], values[field + "_ci_low"], values[field + "_ci_high"] = mean, low, high
        values["geometric_mass_ratio_multiplier"] = math.exp(values["change_log_mass_ratio"])
        pair_values = []
        for identity in sorted(configs):
            pair = sorted([r for r in current if r["case_id"] == identity], key=lambda x: x["gold_rule"])
            if len(pair) != 2 or {r["gold_rule"] for r in pair} != {"A", "B"}:
                raise ValueError("A case does not contain both opposite-patient gold branches")
            entry = {"case_id": identity, "group_id": pair[0]["group_id"], "method": method,
                     "mode": mode, "gain": gain,
                     "minimum_patient_ratio_change": min(r["change_log_mass_ratio"] for r in pair),
                     "mean_patient_ratio_change": float(np.mean([r["change_log_mass_ratio"] for r in pair])),
                     "both_patients_positive": all(r["change_log_mass_ratio"] > 0 for r in pair),
                     "pair_correct": all(r["correct"] for r in pair),
                     "mean_margin_change": float(np.mean([r["change_gold_logit_margin"] for r in pair]))}
            pairs.append(entry)
            pair_values.append(entry)
        values["both_patients_positive_configs"] = sum(x["both_patients_positive"] for x in pair_values)
        values["pair_accuracy"] = float(np.mean([x["pair_correct"] for x in pair_values]))
        summary.append(values)
    primary = [x for x in pairs if x["method"] == "projected" and x["mode"] == "both" and x["gain"] > 0]
    if not primary:
        raise ValueError("No primary projected/both causal conditions")
    highest_gain = max(r["gain"] for r in primary)
    census = [x for x in primary if x["gain"] == highest_gain]
    success = [x for x in census if x["both_patients_positive"]]
    chosen = []
    used = set()
    if success:
        median = float(np.median([r["minimum_patient_ratio_change"] for r in success]))
        typical = min(success, key=lambda x: (abs(x["minimum_patient_ratio_change"] - median), x["case_id"]))
        chosen.append({"illustration_role": "typical_success", **typical})
        used.add(typical["case_id"])
        remaining = [r for r in success if r["case_id"] not in used]
        if remaining:
            strong = min(remaining, key=lambda x: (-x["minimum_patient_ratio_change"], x["case_id"]))
            chosen.append({"illustration_role": "strong_success", **strong})
            used.add(strong["case_id"])
    remaining = [r for r in census if r["case_id"] not in used]
    if remaining:
        boundary = min(remaining, key=lambda x: (x["minimum_patient_ratio_change"], x["case_id"]))
        chosen.append({"illustration_role": "boundary", **boundary})
    for row in census:
        row["chosen_as"] = ";".join(c["illustration_role"] for c in chosen if c["case_id"] == row["case_id"])
    write_csv(output / "causal_patient_effects.csv", rows)
    write_csv(output / "causal_paired_effects.csv", pairs)
    write_csv(output / "causal_summary.csv", summary)
    write_csv(output / "causal_coverage.csv", coverage)
    strata = []
    for method, mode, gain in conditions:
        current = [r for r in rows if (r["method"], r["mode"], r["gain"]) == (method, mode, gain)]
        for field in ("group_id", "action_permutation", "rule_order_index"):
            for value in sorted({r[field] for r in current}):
                part = [r for r in current if r[field] == value]
                local_pairs = [r for r in pairs if r["method"] == method and r["mode"] == mode and r["gain"] == gain
                               and r["case_id"] in {p["case_id"] for p in part}]
                strata.append({"method": method, "mode": mode, "gain": gain, "stratum": field, "stratum_value": value,
                               "n_patients": len(part), "n_configs": len(local_pairs), "n_guides": len({r["group_id"] for r in part}),
                               "mean_ratio_change": float(np.mean([r["change_log_mass_ratio"] for r in part])),
                               "positive_ratio_change_fraction": float(np.mean([r["change_log_mass_ratio"] > 0 for r in part])),
                               "both_patients_positive_configs": sum(r["both_patients_positive"] for r in local_pairs),
                               "mean_margin_change": float(np.mean([r["change_gold_logit_margin"] for r in part])),
                               "accuracy": float(np.mean([r["correct"] for r in part])),
                               "pair_accuracy": float(np.mean([r["pair_correct"] for r in local_pairs])),
                               "corrections": sum(r["wrong_to_correct"] for r in part),
                               "correct_to_wrong": sum(r["correct_to_wrong"] for r in part)})
    write_csv(output / "causal_strata.csv", strata)
    write_csv(output / "case_selection_census.csv", census)
    write_json(output / "selected_cases.json", {
        "purpose": "Exploratory mechanism illustrations; not an efficacy estimate from a selected subset.",
        "selection": "On all confirmation configurations at highest preset projected/both gain: a success requires both opposite-patient delta log(gold/other mass)>0. Typical is nearest to the success median of the minimum patient effect; strong maximizes that minimum among remaining successes; boundary minimizes it over all remaining configurations. Ties use ascending case_id. No outcome accuracy enters selection.",
        "n_candidates": len(census), "n_success": len(success), "gain": highest_gain,
        "cases": chosen})
    lines = ["# 可控读取：机制示范样本", "", "这些解释性示范由运行前固定的规则从本轮结果中选出；总体分母和全部候选保存在 `case_selection_census.csv`。示范的成功标准只针对规则读取，不等于回答纠正，也不把精选子集作为总体收益估计。", "",
             f"共有 {len(census)} 个指南/处置映射/顺序配置，每个配置包含两名金标相反的模拟患者。最高预设增益为 {highest_gain:g}；其中 {len(success)} 个配置的两名患者均提高了正确/兄弟规则注意力质量比。", ""]
    titles = {"typical_success": "典型成功：接近成功配置的中位数", "strong_success": "清晰成功：两名患者中的较小效应最大", "boundary": "边界示范：其余配置中较小患者效应最弱"}
    for case in chosen:
        row = configs[case["case_id"]]
        decisive = row["decisive_node_index"]
        condition = next(c["text"] for c in row["rules"]["A"]["conditions"] if str(c["node_index"]) == str(decisive))
        lines.extend([f"## {titles[case['illustration_role']]}", "", f"- 配置：`{case['case_id']}`；来源：{row['group_id']}。",
                      f"- 处置置换：{row['controls']['action_permutation']}；规则顺序：{' → '.join(row['controls']['rule_order'])}。",
                      f"- 决定性问题：{condition}", f"- 两名患者均保留相同祖先条件，决定性事实分别为 YES 和 NO。",
                      f"- 规则 A 处置：{row['rules']['A']['action']}", f"- 规则 B 处置：{row['rules']['B']['action']}",
                      f"- 两患者最小 Δlog(正确/兄弟规则质量)：{case['minimum_patient_ratio_change']:.6g}。", ""])
        for r in rows:
            if r["case_id"] == case["case_id"] and r["method"] == "projected" and r["mode"] == "both" and r["gain"] == highest_gain:
                lines.append(f"- 金标 {r['gold_rule']}：Δlog 质量比 {r['change_log_mass_ratio']:.6g}；正确编号 logit 间隔变化 {r['change_gold_logit_margin']:.6g}；编号答案{'改变' if r['answer_changed'] else '未改变'}。")
        lines.append("")
    (output / "CASE_STUDIES.zh.md").write_text("\n".join(lines), encoding="utf-8")
    return {"rows": rows, "pairs": pairs, "summary": summary, "chosen": chosen,
            "gain": highest_gain, "n_candidates": len(census), "n_success": len(success),
            "readout_schema": sorted({r["readout_schema"] for r in rows}),
            "candidate_token_ids": sorted({r["candidate_token_ids"] for r in rows}),
            "candidate_continuations": sorted({r["candidate_continuations"] for r in rows})}


def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "triage-module2-mechanism-20261028", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    return plt


def save_figure(fig, directory, name):
    for suffix in ("png", "svg", "pdf"):
        kw = {"dpi": 180} if suffix == "png" else {}
        if suffix == "pdf":
            kw["metadata"] = {"CreationDate": None, "ModDate": None}
        elif suffix == "svg":
            kw["metadata"] = {"Date": None}
        fig.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", **kw)


def plot_directions(result, heads, directory):
    plt = mpl_setup()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    ax = axes[0]
    points = [x for x in result["points"] if x["split"] == "confirm"]
    for offset, method in zip((-.22, 0, .22), ("projected", "raw_mean", "random")):
        for number, (layer, head) in enumerate(heads):
            p = [x for x in points if x["method"] == method and x["layer"] == layer and x["kv_head"] == head]
            by_guide = {}
            for row in p:
                by_guide.setdefault(row["group_id"], []).append(row["cosine"])
            means = [float(np.mean(v)) for _, v in sorted(by_guide.items())]
            ax.scatter(np.full(len(means), number + offset), means, c=COLORS[method],
                       s=33, alpha=.65, label=LABELS[method] if number == 0 else None)
            ax.plot([number + offset - .06, number + offset + .06], [np.mean(means)] * 2,
                    color=COLORS[method], linewidth=2.7)
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=.8)
    ax.set_xticks(range(len(heads)), [f"L{l} / KV{h}" for l, h in heads])
    ax.set_ylabel("Cosine with positive − negative rule-key change")
    ax.set_title("A  Frozen directions on held-out guides", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.text(.02, .02, "Dots: guide means   |   bars: mean of guides", transform=ax.transAxes,
            color="#596577", fontsize=8)
    ax = axes[1]
    palette = ["#007F86", "#497BB3", "#DD8733", "#B55B86", "#6B8C4F"]
    for number, (layer, head) in enumerate(heads):
        p = [x for x in result["interactions"] if x["split"] == "confirm" and
             x["method"] == "projected" and x["layer"] == layer and x["kv_head"] == head]
        xx = [r["common_fact_component"] for r in p]
        yy = [r["applicability_interaction"] for r in p]
        ax.scatter(xx, yy, color=palette[number % len(palette)], alpha=.75, s=36,
                   edgecolor="white", linewidth=.35,
                   label=f"L{layer}/KV{head}: {sum(y > 0 for y in yy)}/{len(yy)} positive")
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=.8)
    ax.axvline(0, color="#64748B", linestyle="--", linewidth=.8)
    ax.set_xlabel(r"Common fact component: $(z_A-z_B)/2$")
    ax.set_ylabel(r"Rule-relative component: $(z_A+z_B)/2$")
    ax.set_title("B  Relative rule response after A/B pairing", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Module 2 · Shared rule-key directions across guidelines", fontsize=15, fontweight="bold", y=1.04)
    fig.text(.01, -.02, "Qwen3-0.6B-Base · token-level second moments · ID + action keys · training-fixed directions\n"
             "Confirmation split only. A/B pairing cancels an identical common fact response; it does not prove removal of all lexical confounds.",
             fontsize=9, color="#4B5563")
    fig.tight_layout(w_pad=3)
    save_figure(fig, directory, "shared_direction")
    plt.close(fig)


def plot_readability(result, heads, directory):
    plt = mpl_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))
    rows = [x for x in result["summary"] if x["method"] == "projected"]
    for ax, split, label in zip(axes[0], ("select", "confirm"), ("A  Selection guides", "B  Held-out confirmation guides")):
        p = [x for x in rows if x["split"] == split and x["online_available"]]
        ax.scatter([x["read_positive"] for x in p], [x["read_negative"] for x in p],
                   color="#BBC3CE", s=17, alpha=.65, label="All candidate KV heads")
        label_offsets = [(-44, 20), (-46, 22), (10, 10), (14, -19)]
        for head_index, (layer, head) in enumerate(heads):
            q = [x for x in p if x["layer"] == layer and x["kv_head"] == head]
            if len(q) != 1:
                raise ValueError(f"Missing online data for selected {layer}/{head} in {split}")
            q = q[0]
            ax.scatter(q["read_positive"], q["read_negative"], c=COLORS["projected"], s=60,
                       edgecolor="white", linewidth=.7, zorder=4)
            ax.annotate(f"L{layer}/KV{head}", (q["read_positive"], q["read_negative"]),
                        xytext=label_offsets[head_index % len(label_offsets)], textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": .65})
        ax.axhline(0, color="#6B7280", linewidth=.8)
        ax.axvline(0, color="#6B7280", linewidth=.8)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel(r"Online $Read^+$")
        ax.set_ylabel(r"Online $Read^-$")
        ax.text(.02, .98, "Upper right: both mean responses positive", ha="left", va="top",
                transform=ax.transAxes, fontsize=8, color="#596577")
    ax = axes[1, 0]
    for i, (layer, head) in enumerate(heads):
        p = next(x for x in rows if x["split"] == "confirm" and x["layer"] == layer and x["kv_head"] == head)
        off = .5 * (p["offline_read_positive"] + p["offline_read_negative"])
        on = .5 * (p["read_positive"] + p["read_negative"])
        ax.plot([i - .16, i + .16], [off, on], color="#B4BFCC", linewidth=1.4)
        ax.scatter(i - .16, off, c="#DD8733", s=48, label="Patient first" if i == 0 else None)
        ax.scatter(i + .16, on, c=COLORS["projected"], s=48, label="Guideline first" if i == 0 else None)
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=.8)
    ax.set_xticks(range(len(heads)), [f"L{l}/KV{h}" for l, h in heads])
    ax.set_ylabel("Mean direct score response per unit key shift")
    ax.set_title("C  Same direction, changed prompt order", loc="left", fontweight="bold", pad=32)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    ax = axes[1, 1]
    for i, (layer, head) in enumerate(heads):
        p = next(x for x in rows if x["split"] == "confirm" and x["layer"] == layer and x["kv_head"] == head)
        ax.bar(i - .17, 100 * p["positive_read_positive_fraction"], .30, color=COLORS["projected"],
               label="Applicable state" if i == 0 else None)
        ax.bar(i + .17, 100 * p["positive_read_negative_fraction"], .30, color="#497BB3",
               label="Inapplicable state" if i == 0 else None)
    ax.set_xticks(range(len(heads)), [f"L{l}/KV{h}" for l, h in heads])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Triplets with positive online response (%)")
    ax.set_title("D  Per-example sign consistency", loc="left", fontweight="bold", pad=32)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    fig.suptitle("Module 2 · Transfer to the guideline-prefix protocol", fontsize=15, fontweight="bold", y=1.01)
    fig.text(.01, -.025, "Confirmation: 24 target triplets / 3 guidelines. Guide-first measurements use the unedited model; the head set is frozen on selection guides.\n"
             "Read is a direct attention-score sensitivity; it is not an attention-mass or final-decision effect.", fontsize=9, color="#4B5563")
    fig.tight_layout(h_pad=3, w_pad=3)
    save_figure(fig, directory, "online_readability")
    plt.close(fig)


def plot_controlled_reading(result, directory):
    plt = mpl_setup()
    cases = result["chosen"]
    if not cases:
        return
    fig, axes = plt.subplots(2, max(3, len(cases)), figsize=(15.6, 9.0))
    titles = {"typical_success": "Typical successful pair", "strong_success": "Strong successful pair",
              "boundary": "Weakest remaining pair"}
    patient_colors = {"A": "#007F86", "B": "#DD8733"}
    for col, case in enumerate(cases):
        ax = axes[0, col]
        for gold in ("A", "B"):
            records = [r for r in result["rows"] if r["case_id"] == case["case_id"] and r["gold_rule"] == gold and
                       (r["method"] == "baseline" or (r["method"] == "projected" and r["mode"] == "both"))]
            records.sort(key=lambda x: x["gain"])
            ax.plot([r["gain"] for r in records], [r["change_log_mass_ratio"] for r in records],
                    marker="o" if gold == "A" else "s", markersize=4.5, linewidth=1.7,
                    color=patient_colors[gold], label=f"Patient with gold rule {gold}")
        ax.axhline(0, color="#64748B", linewidth=.8, linestyle="--")
        ax.set_xlabel("Fixed key-shift gain")
        ax.set_ylabel(r"$\Delta\log(M_{gold}/M_{other})$")
        short = case["case_id"].rsplit(":", 2)
        ax.set_title(f"{'ABC'[col]}  {titles[case['illustration_role']]}\n{case['group_id']} · {short[-2]} · {short[-1]}",
                     loc="left", fontweight="bold", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8)
        changed = sum(r["answer_changed"] for r in result["rows"] if r["case_id"] == case["case_id"] and
                      r["method"] == "projected" and r["mode"] == "both" and r["gain"] == result["gain"])
        ax.text(.03, .04, f"A/B answer changes at gain {result['gain']:g}: {changed}/2", transform=ax.transAxes,
                fontsize=8, color="#526072", bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    for col in range(len(cases), axes.shape[1]):
        axes[0, col].set_axis_off()
    visible = [r for r in result["rows"] if r["case_id"] in {c["case_id"] for c in cases} and
               (r["method"] == "baseline" or (r["method"] == "projected" and r["mode"] == "both"))]
    upper = max(r["change_log_mass_ratio"] for r in visible)
    lower = min(0., min(r["change_log_mass_ratio"] for r in visible))
    extent = max(upper - lower, .01)
    for ax in axes[0, :len(cases)]:
        ax.set_ylim(lower - .05 * extent, upper + .06 * extent)
        ax.set_xticks(sorted({r["gain"] for r in visible}))
    ax = axes[1, 0]
    methods = ("projected", "raw_mean", "random", "reverse", "direct_score_bias")
    labels = {"projected": "Spectral", "raw_mean": "Raw mean", "random": "Random", "reverse": "Reverse", "direct_score_bias": "Score bias"}
    selected = [next((r for r in result["summary"] if r["method"] == m and r["mode"] == "both" and r["gain"] == result["gain"]), None) for m in methods]
    for i, (method, row) in enumerate(zip(methods, selected)):
        if row is None:
            continue
        y = row["change_log_mass_ratio"]
        low, high = row["change_log_mass_ratio_ci_low"], row["change_log_mass_ratio_ci_high"]
        ax.bar(i, y, .62, color=COLORS.get(method, "#818B9D"), alpha=.85)
        if low is not None:
            ax.vlines(i, low, high, color="#334155", linewidth=1)
            ax.plot([i - .08, i + .08], [low, low], color="#334155", linewidth=1)
            ax.plot([i - .08, i + .08], [high, high], color="#334155", linewidth=1)
    ax.axhline(0, color="#64748B", linewidth=.8)
    ax.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel(r"Mean $\Delta\log(M_{gold}/M_{other})$")
    ax.set_title(f"D  All 24 confirmation prompts · gain {result['gain']:g}", loc="left", fontweight="bold", fontsize=10.5)
    ax = axes[1, 1]
    modes = ("enhance", "suppress", "both")
    for i, mode in enumerate(modes):
        row = next((r for r in result["summary"] if r["method"] == "projected" and r["mode"] == mode and r["gain"] == result["gain"]), None)
        if row is None:
            continue
        ax.bar(i, row["change_log_mass_ratio"], .62, color=("#43929C", "#497BB3", "#007F86")[i])
        low, high = row["change_log_mass_ratio_ci_low"], row["change_log_mass_ratio_ci_high"]
        if low is not None:
            ax.vlines(i, low, high, color="#334155", linewidth=1)
            ax.plot([i - .08, i + .08], [low, low], color="#334155", linewidth=1)
            ax.plot([i - .08, i + .08], [high, high], color="#334155", linewidth=1)
    ax.axhline(0, color="#64748B", linewidth=.8)
    ax.set_xticks(range(3), ["Enhance gold", "Suppress other", "Both"], rotation=12, ha="right")
    ax.set_ylabel(r"Mean $\Delta\log(M_{gold}/M_{other})$")
    ax.set_title("E  Separate the two routing operations", loc="left", fontweight="bold", fontsize=10.5)
    ax = axes[1, 2]
    for i, (method, row) in enumerate(zip(methods, selected)):
        if row is None:
            continue
        y = row["change_gold_logit_margin"]
        low, high = row["change_gold_logit_margin_ci_low"], row["change_gold_logit_margin_ci_high"]
        ax.bar(i, y, .62, color=COLORS.get(method, "#818B9D"), alpha=.85)
        if low is not None:
            ax.vlines(i, low, high, color="#334155", linewidth=1)
            ax.plot([i - .08, i + .08], [low, low], color="#334155", linewidth=1)
            ax.plot([i - .08, i + .08], [high, high], color="#334155", linewidth=1)
    ax.axhline(0, color="#64748B", linewidth=.8)
    ax.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Mean correct-label logit-margin change")
    ax.set_title("F  Decision readout · secondary endpoint", loc="left", fontweight="bold", fontsize=10.5)
    fig.suptitle("Module 2 · Controlled reading in the actual prefix-cache protocol", fontsize=15, fontweight="bold", y=1.01)
    fig.text(.01, -.04, f"Top: transparently selected illustrative pairs ({result['n_success']}/{result['n_candidates']} pairs improve the mass ratio for both opposite patients).\n"
             "Bottom: all 24 prompts / 12 paired configurations / 3 guides; intervals resample guides. Masses average selected KV heads and their shared queries.\n"
             "Oracle routing signs · simultaneous selected-head editing · fixed unit-key gains. Score-bias calibration is fixed on selection data, not matched per confirmation example.",
             fontsize=8.5, color="#4B5563")
    fig.tight_layout(h_pad=3.5, w_pad=2.5)
    save_figure(fig, directory, "controlled_reading")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", type=Path, required=True)
    p.add_argument("--online", type=Path, required=True)
    p.add_argument("--operators", type=Path, required=True)
    p.add_argument("--triplets", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    p.add_argument("--causal", type=Path)
    p.add_argument("--status", type=Path, help="Required for causal outputs: completed-run JSON")
    p.add_argument("--conditions", type=Path, help="Optional frozen list of method/mode/gain objects")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.triplets.read_text(encoding="utf-8").splitlines() if line]
    heads = selected_heads(json.loads(args.selection.read_text(encoding="utf-8")))
    with np.load(args.offline, allow_pickle=False) as offline, \
            np.load(args.online, allow_pickle=False) as online, \
            np.load(args.operators, allow_pickle=False) as operators:
        result = analyze_representations(offline, online, operators, rows, heads, args.output)
    if heads:
        plot_directions(result, heads, args.figures)
        plot_readability(result, heads, args.figures)
    causal = None
    if args.causal:
        if not args.status:
            raise ValueError("A completed status file is required for causal analysis")
        status = json.loads(args.status.read_text(encoding="utf-8"))
        if status.get("status", status.get("state")) not in ("completed", "complete"):
            raise ValueError("Refusing to summarize an incomplete or interrupted causal run")
        conditions = json.loads(args.conditions.read_text(encoding="utf-8")) if args.conditions else None
        if isinstance(conditions, dict):
            conditions = conditions["conditions"]
        causal = analyze_causal(args.causal, rows, args.output, protocol=conditions)
        plot_controlled_reading(causal, args.figures)
    inputs = [args.offline, args.online, args.operators, args.triplets, args.selection]
    inputs.extend(p for p in (args.causal, args.status, args.conditions) if p)
    write_json(args.output / "analysis_manifest.json", {
        "schema_version": "module2.mechanism.analysis.v1",
        "inputs": {str(p): digest(p) for p in inputs},
        "selected_heads": heads, "n_triplets": len(rows), "random_direction_source": result["random_source"],
        "source_sha256": digest(__file__),
        "direction_sign_rule": "Strict >0; abs(z)<=1e-9 and difference_norm<=1e-12 separately flagged.",
        "intervals": "5000 guide-cluster percentile bootstrap draws, numpy seed 261028; conditional on frozen operator, only 3 confirmation guides.",
        "scope": "Descriptive held-out mechanism analysis; no new head, direction, dose, or model selection.",
        "empty_expert": not bool(heads),
        "causal": None if causal is None else {"n_pairs": causal["n_candidates"], "n_reading_success_pairs": causal["n_success"],
                                                "n_records": len(causal["rows"]), "n_selected_examples": len(causal["chosen"]),
                                                "n_unique_prompts": len({(r["case_id"], r["patient_state"]) for r in causal["rows"]}),
                                                "n_base_patient_states": len({(r["group_id"], r["patient_state"]) for r in causal["rows"]}),
                                                "readout_schema": causal["readout_schema"],
                                                "candidate_token_ids": causal["candidate_token_ids"],
                                                "candidate_continuations": causal["candidate_continuations"],
                                                "illustration_selection_is_exploratory": True}})
    print(json.dumps({"output": str(args.output), "figures": str(args.figures), "heads": heads}))


if __name__ == "__main__":
    main()
