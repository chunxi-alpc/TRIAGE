"""Diagnose saved primary clinical development outputs; CPU only, no inference."""
from pathlib import Path
import collections
import hashlib
import json
import statistics

P = Path(__file__).resolve().parent
R = P / "runtime/mega_dev16_seed0_gpu01"

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg = json.loads((P / "mega_dev16_primary_scores_verified/evaluation_config.json").read_text())
    predictions = {}
    for method, expected in cfg["prediction_sha256"].items():
        path = R / f"{method}.predictions.private.jsonl"
        assert digest(path) == expected, method
        predictions[method] = {row["sample_id"]: row for row in read_rows(path)}
        assert len(predictions[method]) == 16
    scores_path = P / "mega_dev16_primary_scores_verified/case_metrics.jsonl"
    scores = {(row["method"], row["sample_id"]): row for row in read_rows(scores_path)}
    assert len(scores) == 176
    stage = read_rows(R / "stage_predictions.private.jsonl")
    conditions = {(row["sample_id"], row.get("rule_id"), row["stage_configuration"]): row
                  for row in stage}
    transitions = collections.Counter()
    for row in stage:
        if row["stage_configuration"] == "A0":
            other = conditions[row["sample_id"], row["rule_id"], "A1"]
            transitions[row["prediction"]["label"] + "->" + other["prediction"]["label"]] += 1
    metrics = ("acc_normalized", "pcc_dtw_cost")
    arms = {}
    for method, rows in predictions.items():
        arms[method] = {
            "n": len(rows),
            "steering_path_calls": sum(bool(row["generation"].get("steer")) for row in rows.values()),
            "zero_gain_note": "steer=True records hook path, not nonzero effect; inspect gains",
            "at_output_budget": sum(row["generation"].get("output_tokens") == row["max_new_tokens"] for row in rows.values()),
            "parse_failures": sum(not scores[method, sid]["parsed_success"] for sid in rows),
        }
    primary = predictions["A1B1"]
    groups = {
        "active": [sid for sid, row in primary.items() if row["generation"]["steer"]],
        "inactive": [sid for sid, row in primary.items() if not row["generation"]["steer"]],
        "active_parse_success": [sid for sid, row in primary.items() if row["generation"]["steer"] and scores["A1B1", sid]["parsed_success"]],
        "parse_failed": [sid for sid in primary if not scores["A1B1", sid]["parsed_success"]],
    }
    grouped = {}
    for name, ids in groups.items():
        grouped[name] = {
            "n": len(ids),
            "scope": "Post-hoc diagnostic subset, not a replacement primary denominator",
            "means": {method: {metric: statistics.mean(scores[method, sid][metric] for sid in ids)
                              for metric in metrics} for method in predictions},
        }
    paired = {}
    for method, reference in [("A1B1_zero", "A1B0"), ("A1B1", "A1B1_zero")]:
        rows = []
        for sid, row in predictions[method].items():
            other = predictions[reference][sid]
            rows.append({
                "sample_id": sid,
                "same_answer": row["llm_output"] == other["llm_output"],
                "same_input_ids": row["generation"]["input_ids_sha256"] == other["generation"]["input_ids_sha256"],
                "main_steer": primary[sid]["generation"]["steer"],
                "rule_count": row["condition_rule_count"],
                "highlighted_tokens": row["generation"].get("highlighted_tokens"),
                "input_tokens": row["generation"].get("input_tokens"),
                "output_tokens": row["generation"].get("output_tokens"),
                "parsed_success": scores[method, sid]["parsed_success"],
                "numerical_control": row["generation"].get("numerical_control"),
                "delta": {metric: scores[method, sid][metric] - scores[reference, sid][metric] for metric in metrics},
            })
        paired[method + "_vs_" + reference] = rows
    zero = paired["A1B1_zero_vs_A1B0"]
    assert all(row["same_answer"] and row["same_input_ids"] for row in zero)
    assert all(row["numerical_control"]["max_abs_first_token_score_difference"] == 0 for row in zero)
    cond_zero = [row["generation"]["numerical_control"] for row in stage if row["stage_configuration"] == "A1_zero"]
    assert len(cond_zero) == 60
    assert all(row["max_abs_first_token_score_difference"] == 0 for row in cond_zero)
    output = {
        "status": "complete_verified", "scope": "Post-hoc diagnostics of fixed 16-case development run",
        "new_gpu_forwards": 0, "primary_denominator_unchanged": 16, "raw_text_published": False,
        "prediction_sha256": cfg["prediction_sha256"], "scores_sha256": digest(scores_path),
        "stage_predictions_sha256": digest(R / "stage_predictions.private.jsonl"),
        "diagnostic_script_sha256": digest(Path(__file__)),
        "arms": arms, "groups": grouped, "pairs": paired, "condition_label_transitions": dict(transitions),
        "condition_zero_max_abs_logit_difference": max(row["max_abs_first_token_score_difference"] for row in cond_zero),
        "execution_zero_identical_full_answers": len(zero),
        "judge_noise_note": "Identical generated answers were independently sampled by the local judge. Differences between A1B0 and A1B1_zero are not generator effects. Existing CIs condition on saved judge draws and do not integrate full judge uncertainty.",
        "interpretation": "Current nonzero execution steering harms generation on the six activated cases. This establishes failure of the present configuration, not all attention methods. Exact-zero identity provides evidence against hook drift but is not a proof of nonzero steering correctness.",
    }
    (P / "primary_failure_diagnosis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "new_gpu_forwards": 0, "active": len(groups["active"]), "zero_identical": len(zero), "condition_transitions": dict(transitions)}))

if __name__ == "__main__":
    main()
