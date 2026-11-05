#!/usr/bin/env python3
"""Summarize fixed test predictions without fitting parameters or invoking models.

Example:
  python summarize_predictions.py --run-dir /path/to/snapshot \
      --output /path/to/supplementary_analysis.json
The run directory must contain the original data/ and results/ subdirectories.
"""
import argparse
import collections
import json
from pathlib import Path


VIEWS = ("full", "direct", "condition_ancestors")
LEVELS = ("below", "above")
EXAMPLES = (
    "83cf1b073d03d1221b88fcd00c45f468:2",
    "48b354078dca3fbaf7855e527e8d1637:13",
)


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def primary_groups(rows):
    groups = collections.defaultdict(dict)
    for row in rows:
        if (row["split"] == "test" and row["population"] == "constructed"
                and row["background"] == "neutral" and row["wording"] == "original"
                and row["metric_level"] in LEVELS):
            groups[row["node_id"]][row["metric_level"]] = row
    for node_id, pair in groups.items():
        if set(pair) != set(LEVELS):
            raise ValueError("Incomplete primary pair: " + node_id)
        if pair["below"]["gold"] == pair["above"]["gold"]:
            raise ValueError("Primary pair does not cross the full predicate: " + node_id)
    return dict(groups)


def score_for(row, scores):
    score = scores[row["score_id"]]
    if score["status"] != "complete":
        raise ValueError("Incomplete saved score: " + row["score_id"])
    return score


def correct_prediction(row, probability):
    return int(probability >= 0.5) == row["gold"]


def describe_example(row, score):
    construction = row["construction"]
    return {
        "node_id": row["node_id"], "example_id": row["example_id"],
        "condition": row["condition"], "metric": construction["metric"],
        "value": row["metric_value"], "unit": row.get("unit", construction.get("unit")),
        "fixed_facts": construction.get("fixed_fact_sentences", {}),
        "gold": row["gold"], "prediction": score["prediction"], "p_yes": score["p_yes"],
    }


def fact_summary(rows, scores, view):
    selected = [row for row in rows if row["view"] == view
                and row["population"] == "constructed" and row["split"] == "test"]
    groups = primary_groups(selected)
    counts = collections.Counter()
    direction = collections.Counter()
    failed_direction = collections.Counter()
    failed_nodes = []
    examples = []
    for node_id, pair in groups.items():
        left, right = pair["below"], pair["above"]
        first, second = score_for(left, scores), score_for(right, scores)
        is_correct = (first["prediction"] == left["gold"]
                      and second["prediction"] == right["gold"])
        counts["both_correct" if is_correct else "at_least_one_wrong"] += 1
        same = first["prediction"] == second["prediction"]
        counts["prediction_unchanged" if same else "prediction_changed"] += 1
        if same:
            counts["constant_yes" if first["prediction"] else "constant_no"] += 1
        elif not is_correct:
            counts["reverse_flip"] += 1
        delta = (second["p_yes"] - first["p_yes"]) * (right["gold"] - left["gold"])
        category = "correct" if delta > 0 else "opposite" if delta < 0 else "equal_saved_probability"
        direction[category] += 1
        if not is_correct:
            failed_direction[category] += 1
            failed_nodes.append(node_id)
        if view == "full" and node_id in EXAMPLES:
            examples.append({
                "node_id": node_id,
                "both_correct": is_correct,
                "below": describe_example(left, first),
                "above": describe_example(right, second),
            })
    backgrounds = collections.defaultdict(dict)
    noncrossing = collections.defaultdict(dict)
    for row in selected:
        if row["wording"] != "original":
            continue
        backgrounds[(row["node_id"], row["metric_level"])][row["background"]] = row
        if row["background"] == "neutral" and row["metric_level"] in ("above", "same_side"):
            noncrossing[row["node_id"]][row["metric_level"]] = row
    background_changes = []
    background_pairs = neutral_correct = cue_correct = 0
    for (_, _), pair in backgrounds.items():
        if "neutral" not in pair or "cue" not in pair:
            continue
        first_row, second_row = pair["neutral"], pair["cue"]
        first = score_for(first_row, scores)
        second = score_for(second_row, scores)
        background_pairs += 1
        neutral_correct += first["prediction"] == first_row["gold"]
        cue_correct += second["prediction"] == second_row["gold"]
        if first["prediction"] != second["prediction"]:
            background_changes.append({
                "node_id": first_row["node_id"], "metric_level": first_row["metric_level"],
                "condition": first_row["condition"], "value": first_row["metric_value"],
                "gold": first_row["gold"],
                "neutral_prediction": first["prediction"], "cue_prediction": second["prediction"],
                "neutral_p_yes": first["p_yes"], "cue_p_yes": second["p_yes"],
            })
    noncrossing_pairs = stable = both_correct = 0
    for pair in noncrossing.values():
        if set(pair) != {"above", "same_side"}:
            continue
        left, right = pair["above"], pair["same_side"]
        first, second = score_for(left, scores), score_for(right, scores)
        noncrossing_pairs += 1
        stable += first["prediction"] == second["prediction"]
        both_correct += first["prediction"] == left["gold"] and second["prediction"] == right["gold"]
    required_counts = ("both_correct", "at_least_one_wrong", "prediction_unchanged",
                       "prediction_changed", "constant_yes", "constant_no", "reverse_flip")
    return {
        "nodes": len(groups),
        "P_b": fraction(counts["at_least_one_wrong"], len(groups)),
        "failure_types": {key: counts[key] for key in required_counts},
        "probability_direction": {key: direction[key] for key in
                                  ("correct", "opposite", "equal_saved_probability")},
        "probability_direction_among_failed": {key: failed_direction[key] for key in
                                               ("correct", "opposite", "equal_saved_probability")},
        "probability_note": "Directions use stored binary probabilities; equality near 0 or 1 may reflect floating-point saturation.",
        "failed_node_ids": sorted(failed_nodes),
        "background": {
            "paired_examples": background_pairs, "changed_predictions": len(background_changes),
            "change_rate": fraction(len(background_changes), background_pairs),
            "neutral_correct": neutral_correct, "cue_correct": cue_correct,
            "changed_examples": background_changes,
        },
        "noncrossing": {
            "pairs": noncrossing_pairs, "stable_predictions": stable, "both_correct": both_correct,
        },
        "examples": sorted(examples, key=lambda item: EXAMPLES.index(item["node_id"])),
    }


def gains_and_losses(rows, first_method, baseline):
    gain, loss = [], []
    first_correct = baseline_correct = 0
    for row in rows:
        probabilities = row["probabilities"]
        first = correct_prediction(row, probabilities[first_method])
        second = correct_prediction(row, probabilities[baseline])
        first_correct += first
        baseline_correct += second
        if first and not second:
            gain.append(row)
        elif second and not first:
            loss.append(row)
    return {
        "n": len(rows), "method": first_method, "baseline": baseline,
        "method_accuracy": fraction(first_correct, len(rows)),
        "baseline_accuracy": fraction(baseline_correct, len(rows)),
        "gained_examples": len(gain), "lost_examples": len(loss),
        "net_correct_difference": len(gain) - len(loss),
        "gain_node_count": len({row["node_id"] for row in gain}),
        "loss_node_count": len({row["node_id"] for row in loss}),
        "gain_node_counts": dict(sorted(collections.Counter(row["node_id"] for row in gain).items())),
        "loss_node_counts": dict(sorted(collections.Counter(row["node_id"] for row in loss).items())),
        "gain_example_ids": sorted(row["example_id"] for row in gain),
        "loss_example_ids": sorted(row["example_id"] for row in loss),
    }


def fusion_summary(predictions, metadata):
    test = [row for row in predictions if row["split"] == "test"]
    comparisons = {}
    for population in ("original", "constructed"):
        selected = [row for row in test if row["population"] == population]
        comparisons[population] = {
            baseline: gains_and_losses(selected, "logistic", baseline)
            for baseline in ("calibration_full", "calibration_blind")
        }
    groups = collections.defaultdict(dict)
    for row in test:
        if row["population"] != "constructed":
            continue
        item = metadata[row["example_id"]]
        if (item["background"] == "neutral" and item["wording"] == "original"
                and item["metric_level"] in LEVELS):
            groups[row["node_id"]][item["metric_level"]] = row
    if any(set(pair) != set(LEVELS) for pair in groups.values()):
        raise ValueError("Incomplete primary fusion pair")
    methods = sorted(next(iter(groups.values()))["below"]["probabilities"]) if groups else []
    pair_success = {}
    pair_metrics = {}
    for method in methods:
        succeeded = {node_id for node_id, pair in groups.items()
                     if all(correct_prediction(row, row["probabilities"][method]) for row in pair.values())}
        pair_success[method] = succeeded
        pair_metrics[method] = {
            "P_b": fraction(len(groups) - len(succeeded), len(groups)),
            "both_correct_pairs": len(succeeded),
        }
    pair_comparisons = {}
    for baseline in ("full", "calibration_full", "calibration_blind"):
        gains = pair_success["logistic"] - pair_success[baseline]
        losses = pair_success[baseline] - pair_success["logistic"]
        pair_comparisons[baseline] = {
            "gained_complete_pairs": len(gains), "lost_complete_pairs": len(losses),
            "gain_node_ids": sorted(gains), "loss_node_ids": sorted(losses),
            "gain_conditions": {
                node_id: metadata[groups[node_id]["below"]["example_id"]]["condition"]
                for node_id in sorted(gains)
            },
        }
    return {
        "single_example_comparisons": comparisons,
        "primary_pair_population": "test sources, neutral background, original condition, one below/above pair per node",
        "primary_pair_metrics_all_methods": pair_metrics,
        "primary_pair_logistic_comparisons": pair_comparisons,
    }


def summarize(run_dir):
    inputs = read_rows(run_dir / "data" / "inputs_full.jsonl")
    scores = {}
    for shard in (0, 1):
        for row in read_rows(run_dir / "results" / f"scores_full_{shard}.jsonl"):
            scores[row["score_id"]] = row
    metadata = {row["example_id"]: row for row in inputs if row["view"] == "full"}
    result = {
        "analysis_type": "Supplementary descriptive analysis of fixed test predictions",
        "parameter_refitting": False, "new_inference": False,
        "case_independence_note": "Constructed test examples are six variants of each of 28 nodes, not 168 independent conditions.",
        "source_files": [
            "data/inputs_full.jsonl", "results/scores_full_0.jsonl", "results/scores_full_1.jsonl",
            "results/fusion_predictions_full_direct.jsonl",
            "results/fusion_predictions_full_condition_ancestors.jsonl",
        ],
        "key_facts": {view: fact_summary(inputs, scores, view) for view in VIEWS},
        "fusion": {},
    }
    for view in ("direct", "condition_ancestors"):
        predictions = read_rows(run_dir / "results" / f"fusion_predictions_full_{view}.jsonl")
        result["fusion"][view] = fusion_summary(predictions, metadata)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = summarize(arguments.run_dir)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
    print(json.dumps({
        "output": str(arguments.output),
        "full_primary_P_b": result["key_facts"]["full"]["P_b"],
        "condition_ancestors_fusion_primary_P_b":
            result["fusion"]["condition_ancestors"]["primary_pair_metrics_all_methods"]["logistic"]["P_b"],
        "new_inference": False,
    }))


if __name__ == "__main__":
    main()
