"""One-shot CPU snapshot of completed v3 records; never launches evaluation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from mega_development_evaluation import load_official, parse_with_official
from clinical_v3_identity import index_predictions

P = Path(__file__).resolve().parent

def sha(value):
    return hashlib.sha256(value).hexdigest()

def main():
    plan = json.loads((P / "clinical_v3_plan.json").read_text())
    root = P / "runtime/clinical_v3_generation"
    frozen = json.loads(Path(plan["input_manifest"]).read_text())
    official, sources = load_official(Path(plan["official_root"]) / "mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py")
    timestamp = datetime.now(timezone.utc)
    output = P / "clinical_v3_snapshots" / timestamp.strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    snapshots, metadata, summaries, fingerprints = {}, [], {}, {}
    for method in plan["methods"]:
        path = root / (method + ".predictions.private.jsonl")
        data = path.read_bytes() if path.exists() else b""
        end = data.rfind(b"\n") + 1
        complete = data[:end]
        rows = [json.loads(line) for line in complete.splitlines()]
        records = index_predictions(rows, frozen)
        assert len(records) == len(rows)
        snapshots[method] = records
        fingerprints[method] = {"private_source": str(path), "complete_prefix_bytes": len(complete),
                                "complete_prefix_sha256": sha(complete), "ignored_incomplete_tail_bytes": len(data) - end}
        for sid, row in records.items():
            parsed, failure = parse_with_official(official["parser"], row)
            generation = row["generation"]
            metadata.append({
                "sample_id": sid, "method": method, "generation_status": row["status"],
                "prediction_text_sha256": sha(row["llm_output"].encode()),
                "base_prompt_sha256": row["base_prompt_sha256"], "mapping_sha256": row["mapping_sha256"],
                "parsed_success": parsed["parsed_success"], "parse_failure": failure,
                "selected_span_count": row["selected_span_count"],
                **{key: generation.get(key) for key in ["output_tokens", "actual_forward_calls", "wall_seconds",
                                                       "peak_allocated_gib", "peak_reserved_gib", "numerical_control"]},
                "answer_score": None, "pathway_cost": None, "evaluation_status": "not_run",
            })
        mine = [row for row in metadata if row["method"] == method]
        summaries[method] = {
            "completed_predictions": len(rows), "expected_predictions": 16,
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "parse_failures": sum(not row["parsed_success"] for row in mine),
            "at_output_budget": sum(row["output_tokens"] == 1024 for row in mine),
            "recorded_forward_calls": sum(row["actual_forward_calls"] for row in mine),
            "recorded_wall_seconds": sum(row["wall_seconds"] for row in mine),
        }
    paired = {}
    zero = snapshots["gain_0"]
    for method in plan["methods"]:
        common = sorted(zero.keys() & snapshots[method].keys())
        paired[method] = {
            "completed_pairs_with_zero": len(common),
            "same_complete_answer_as_zero": sum(zero[sid]["llm_output"] == snapshots[method][sid]["llm_output"] for sid in common),
            "note": "Text equality is an implementation diagnostic, not clinical quality or accuracy",
        }
    checks = [row["generation"].get("numerical_control") for row in zero.values()]
    assert all(check and check["same_full_answer"] and check["max_abs_first_token_score_difference"] == 0 for check in checks)
    common_all = set.intersection(*(set(rows) for rows in snapshots.values()))
    audit = {
        "status": "running", "snapshot_verification_status": "complete_verified",
        "scope": "Partial generation and parsing snapshot only; clinical effects unscored",
        "observed_at_utc": timestamp.isoformat(), "expected_predictions": 112,
        "completed_predictions": len(metadata), "complete_cases_across_all_methods": len(common_all),
        "methods": summaries, "paired_text_diagnostics": paired,
        "zero_control_checked_pairs": len(checks), "zero_control_passed": True,
        "prediction_prefix_fingerprints": fingerprints, "official_source_sha256": sources,
        "new_gpu_forwards": 0, "raw_text_published": False,
        "all_case_denominator_remains": 16, "winner_claim": None,
        "pipeline_state_at_snapshot": json.loads((P / "runtime/clinical_v3_pipeline_state.json").read_text()),
        "generator_manifest_at_snapshot": {key: value for key, value in json.loads((P / "runtime/clinical_v3_generation_job/run_manifest.json").read_text()).items()
                                           if key in ["pid", "started_at_utc", "process_state", "gpu_physical", "stage"]},
        "script_sha256": sha(Path(__file__).read_bytes()),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "prediction_metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata))
    note = ["# v3 部分进度（一次快照）", "",
            f"快照时间：{timestamp.isoformat()}。已完成 {len(metadata)}/112 条生成，所有方法均完成的案例为 {len(common_all)}/16。",
            "本轮实际于北京时间 2026-10-26 15:25 开始占用 GPU1，之前为等待前置任务。",
            "统一临床评分尚未开始，不能判断 0.2、0.6 是否超过零强度、文本强调或概率融合。", "",
            "| 方法 | 完成生成 | 解析失败 | 写满输出上限 |",
            "|---|---:|---:|---:|"]
    for method, values in summaries.items():
        note.append(f"| {method} | {values['completed_predictions']}/16 | {values['parse_failures']} | {values['at_output_budget']} |")
    note.extend(["", f"已完成的 {len(checks)} 个零强度/无干预配对，首 token 分数差均为零，完整答案相同。",
                 "部分生成文本相同与否只用于排查实现，不代表优劣；尚无答案评分、路径分数或配对置信区间。",
                 "后台任务保持原安排，完成后自动核验并提交最终结果。本次不启动新实验或额外 GPU 评分。"])
    (P / "CLINICAL_V3_PROGRESS.zh.md").write_text("\n".join(note) + "\n")
    ledger = P / "task_ledger.json"; state = json.loads(ledger.read_text())
    for task in state["subtasks"]:
        if task["id"] == "clinical_strength_v3":
            task.update(status="running", execution_state="generation_v3", completed_predictions=len(metadata),
                        expected_predictions=112, evidence=str((output / "audit.json").relative_to(P)),
                        native_evaluation_status="not_started")
    ledger.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "total": len(metadata), "methods": summaries, "pairs": paired}))

if __name__ == "__main__":
    main()
