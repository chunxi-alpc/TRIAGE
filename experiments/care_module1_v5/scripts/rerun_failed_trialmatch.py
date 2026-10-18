"""Regenerate only parser-failed TrialMatchAI atoms with the required Phi-4 LoRA.

All patient text and raw completions stay under a mode-0700 run directory.
This script never reads labels, Qwen outputs, or gold facts during generation.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/care_module1_v5"
sys.path.insert(0, str(ROOT))
DEFAULT_RAW = EXP / "runs/v5_20261018/private/trialmatch_precomputed.jsonl"
DEFAULT_CONDITIONS = EXP / "runs/v5_20261018/parser_v2_final/conditions_public.jsonl"
DEFAULT_REQUESTS = EXP / "runs/v5_parallel_generation/private/requests_private.jsonl"
DEFAULT_BASELINE = EXP / "runs/v5_20261018/predictions_public.jsonl"
BASE = "/home/gcx/models/care_module1_v5/phi-4"
ADAPTER = "/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora"
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 8
WORLD_SIZE = 4
PROMPT_VERSION = "retry_exact_single_condition_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha_bytes(value: bytes):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity(row):
    return tuple(row[key] for key in ("sample_id", "rule_id", "condition_id", "criterion_kind"))


def prepare(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    os.chmod(run, 0o700)
    private = run / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    os.chmod(private, 0o700)
    prior = read(args.conditions)
    originals = read(args.raw)
    requests = read(args.requests)
    parsed_by_key = {identity(row): row for row in prior}
    raw_by_key = {identity(row): row for row in originals}
    request_by_key = {identity(row): row for row in requests}
    if any(len(index) != len(rows) for index, rows in ((parsed_by_key, prior), (raw_by_key, originals), (request_by_key, requests))):
        raise ValueError("duplicate condition identity in source files")
    if set(parsed_by_key) != set(raw_by_key) or set(raw_by_key) != set(request_by_key) or len(prior) != 492:
        raise ValueError("expected exact matching set of 492 saved conditions and requests")
    selected = [row for row in prior if row["parse_status"] in {"truncated", "ambiguous_target"}]
    if len(selected) != 69:
        raise ValueError(f"expected 69 parser failures, found {len(selected)}")
    targets = []
    for parsed in selected:
        key = identity(parsed)
        raw = raw_by_key[key]
        request = request_by_key[key]
        completion = raw["trialmatch"]
        digest = sha_bytes(completion["raw_output"].encode())
        if parsed["raw_output_sha256"] != digest:
            raise ValueError("saved raw completion hash does not match frozen parser record")
        if request["condition"].get("text") != parsed["criterion_text"]:
            raise ValueError("request condition differs from frozen target criterion")
        targets.append({"request_id": request["request_id"], **{k: request[k] for k in (
            "sample_id", "rule_id", "condition_id", "criterion_kind", "patient", "condition")},
            "old_parse_status": parsed["parse_status"], "old_parse_error": parsed["parse_error"],
            "old_raw_output_sha256": digest})
    targets.sort(key=lambda row: row["request_id"])
    assignment_counts = []
    for rank in range(WORLD_SIZE):
        assigned = targets[rank::WORLD_SIZE]
        assignment_counts.append(len(assigned))
        write_rows(private / f"retry_assignment_rank{rank}_private.jsonl", assigned)
    write_rows(private / "retry_targets_private.jsonl", targets)
    manifest = {
        "status": "prepared", "created_utc": now(), "target_count": len(targets),
        "source_condition_count": len(prior), "source_parse_status_counts": dict(collections.Counter(row["parse_status"] for row in prior)),
        "retry_parse_status_counts": dict(collections.Counter(row["parse_status"] for row in selected)),
        "world_size": WORLD_SIZE, "physical_gpus": args.gpus, "assignment_counts": assignment_counts,
        "batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS,
        "base_model": BASE, "adapter_model": ADAPTER, "prompt_version": PROMPT_VERSION,
        "labels_read": False, "qwen_called": False,
        "source_sha256": {str(Path(p)): sha_file(p) for p in (args.conditions, args.raw, args.requests, args.baseline)},
    }
    write_json(private / "retry_manifest_private.json", manifest)
    print(json.dumps({"target_count": len(targets), "assignment_counts": assignment_counts,
                      "statuses": manifest["retry_parse_status_counts"], "labels_read": False}, ensure_ascii=False))


def retry_messages(patient: str, criterion_text: str, criterion_kind: str):
    inclusion = criterion_kind != "exclusion"
    target_key = "Inclusion_Criteria_Evaluation" if inclusion else "Exclusion_Criteria_Evaluation"
    other_key = "Exclusion_Criteria_Evaluation" if inclusion else "Inclusion_Criteria_Evaluation"
    allowed = "Met | Not Met | Unclear | Irrelevant" if inclusion else "Violated | Not Violated | Unclear | Irrelevant"
    system = "Classify only the single requested clinical-trial condition from the supplied patient note. Follow the exact output constraints."
    user = (
        "Return one compact JSON object only. Do not add markdown, preface, analysis, recap, or a trial-wide decision.\n"
        "Assess only the one requested condition. Copy its Criterion text exactly, character for character; do not paraphrase, split, shorten, or replace it.\n"
        f"Classification must be exactly one of: {allowed}. Use Unclear when the note lacks enough information.\n"
        "Keep Justification to one brief sentence grounded in the note. The requested array must contain exactly one object; the other array must be empty.\n"
        f'Output schema: {{"{target_key}":[{{"Criterion":"exact requested text","Classification":"one allowed label","Justification":"brief evidence-based reason"}}],"{other_key}":[]}}\n\n'
        "Requested condition (copy this text exactly into Criterion):\n"
        f"{criterion_text}\n\n"
        "Patient note (use only this information):\n"
        f"{patient}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def worker(args):
    work = Path(args.run_dir) / "private"
    rank = args.rank
    assignment = read(work / f"retry_assignment_rank{rank}_private.jsonl")
    status_path = work / f"retry_worker_rank{rank}_status_private.json"
    output_path = work / f"retry_worker_rank{rank}_outputs_private.jsonl"
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError("worker output already exists; refusing duplicate generation")
    status = {"status": "loading", "rank": rank, "physical_gpu": args.device,
              "assigned_conditions": len(assignment), "completed_conditions": 0,
              "start_utc": now(), "max_new_tokens": MAX_NEW_TOKENS,
              "batch_size": BATCH_SIZE, "prompt_version": PROMPT_VERSION,
              "labels_read": False, "qwen_called": False}
    write_json(status_path, status)
    started = time.perf_counter()
    try:
        import torch
        from src.module1.care_v5 import TrialMatchAIAdapter
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        TrialMatchAIAdapter.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        TrialMatchAIAdapter._messages = staticmethod(retry_messages)
        load_start = time.perf_counter()
        model = TrialMatchAIAdapter.from_pretrained(
            device=f"cuda:{args.device}", base_model=args.base_model, adapter_model=args.adapter_model)
        active = getattr(model.model, "active_adapters", None)
        if callable(active):
            active = active()
        if not active:
            raise RuntimeError("required TrialMatchAI LoRA is not active; no fallback is permitted")
        status.update(status="running", load_seconds=time.perf_counter() - load_start,
            active_adapters=list(active) if not isinstance(active, str) else [active],
            parameter_count=sum(p.numel() for p in model.model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in model.model.parameters() if p.requires_grad),
            gpu_name=torch.cuda.get_device_name(args.device))
        write_json(status_path, status)
        with output_path.open("w") as handle:
            os.chmod(output_path, 0o600)
            for start in range(0, len(assignment), BATCH_SIZE):
                batch = assignment[start:start + BATCH_SIZE]
                results = model.judge_many(batch, batch_size=BATCH_SIZE)
                for request, result in zip(batch, results):
                    result.update(prompt_version=PROMPT_VERSION,
                        prompt_style=PROMPT_VERSION, prompt_commit="custom_retry_prompt",
                        prompt_source="experiments/care_module1_v5/scripts/rerun_failed_trialmatch.py")
                    row = {"request_id": request["request_id"], **{k: request[k] for k in (
                        "sample_id", "rule_id", "condition_id", "criterion_kind")},
                        "old_parse_status": request["old_parse_status"],
                        "old_parse_error": request["old_parse_error"],
                        "old_raw_output_sha256": request["old_raw_output_sha256"],
                        "trialmatch": result}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                status.update(completed_conditions=status["completed_conditions"] + len(batch),
                    last_request_id=batch[-1]["request_id"], elapsed_seconds=time.perf_counter() - started)
                write_json(status_path, status)
        status["status"] = "completed"
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    finally:
        if "torch" in locals() and torch.cuda.is_available():
            status.update(peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(args.device) / 1024**3)
        status.update(end_utc=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(status_path, status)
    if status["status"] != "completed":
        raise SystemExit(2)


def merge_retry(args):
    run = Path(args.run_dir)
    private = run / "private"
    original = read(args.raw)
    new = []
    for rank in range(WORLD_SIZE):
        status = json.loads((private / f"retry_worker_rank{rank}_status_private.json").read_text())
        if status.get("status") != "completed":
            raise RuntimeError(f"retry worker {rank} did not complete")
        new.extend(read(private / f"retry_worker_rank{rank}_outputs_private.jsonl"))
    by_key = {identity(row): row for row in new}
    if len(by_key) != 69:
        raise ValueError("expected exactly 69 unique retry completions")
    merged = []
    replaced = set()
    for row in original:
        key = identity(row)
        if key in by_key:
            retry = by_key[key]
            if retry["old_raw_output_sha256"] != sha_bytes(row["trialmatch"]["raw_output"].encode()):
                raise ValueError("retry parent raw hash changed")
            merged.append({**row, "trialmatch": retry["trialmatch"], "retry_source": PROMPT_VERSION})
            replaced.add(key)
        else:
            merged.append(row)
    if replaced != set(by_key):
        raise ValueError("retry output identities do not match original saved records")
    merged_path = private / "trialmatch_merged_retry_private.jsonl"
    write_rows(merged_path, merged)
    # Update only public digests/timing metadata for the 69 regenerated outputs.
    baseline = read(args.baseline)
    replacement = {identity(row): row["trialmatch"] for row in merged if identity(row) in by_key}
    for sample in baseline:
        old_total = sample.get("trialmatch_seconds", 0.0)
        old_elapsed = sample.get("elapsed_seconds", 0.0)
        condition_times = 0.0
        for condition in sample["conditions"]:
            key = (sample["sample_id"], sample["rule_id"], condition["condition_id"],
                   next((identity(row)[3] for row in merged if identity(row)[:3] ==
                         (sample["sample_id"], sample["rule_id"], condition["condition_id"])), "inclusion"))
            if key not in replacement:
                condition_times += condition.get("trialmatch_seconds", 0.0)
                continue
            output = replacement[key]
            old_time = condition.get("trialmatch_seconds", 0.0)
            new_time = output.get("inference_seconds", 0.0)
            condition_times += new_time
            condition.update(trialmatch_seconds=new_time,
                trialmatch_output_sha256=sha_bytes(output.get("raw_output", "").encode()),
                trialmatch_output_tokens=output.get("output_tokens", 0),
                trialmatch_truncated=bool(output.get("truncated", False)),
                trialmatch_parse_error=output.get("parse_error"),
                trialmatch_raw_label=output.get("raw_label"),
                trialmatch_timing_method=output.get("timing_method", "retry_batch_wall_time_divided_by_batch_size"),
                trialmatch_generation_batch_id=output.get("generation_batch_id"),
                trialmatch_generation_batch_size=output.get("generation_batch_size"),
                trialmatch_input_tokens=output.get("input_tokens"),
                trialmatch_cache_lookup_seconds=0.0)
            sample["elapsed_seconds"] = sample.get("elapsed_seconds", 0.0) + new_time - old_time
        sample["trialmatch_seconds"] = condition_times
    baseline_path = private / "baseline_retry_public.jsonl"
    write_rows(baseline_path, baseline)
    return merged_path, baseline_path, {"retry_count": len(by_key), "replaced_count": len(replaced),
        "unchanged_count": len(merged)-len(replaced), "merged_sha256": sha_file(merged_path),
        "baseline_sha256": sha_file(baseline_path)}


def report(run, original_metrics, conditions, metrics, statuses, failed):
    old_cond = json.loads((EXP / "runs/v5_20261018/parser_v2_final/condition_eval/condition_metrics.json").read_text())
    new_cond = json.loads((conditions / "condition_eval/condition_metrics.json").read_text())
    old_rule = json.loads(original_metrics.read_text())
    new_rule = json.loads((conditions / "metrics.json").read_text())
    old_status = json.loads((EXP / "runs/v5_20261018/parser_v2_final/run_status.json").read_text())
    counts = collections.Counter(row["parse_status"] for row in read(conditions / "conditions_public.jsonl"))
    failures = [row for row in read(conditions / "conditions_public.jsonl") if row["parse_status"] != "accepted"]
    methods = new_rule["methods"]
    metrics = new_cond["new_parser"]
    retry_statuses = collections.Counter(row["parse_status"] for row in read(conditions / "conditions_public.jsonl")
        if row.get("sample_id") and (row.get("old_parse_status") if "old_parse_status" in row else None))
    # Derive retries directly from the private worker output identities without emitting patient text.
    retry_rows = []
    for rank in range(WORLD_SIZE):
        retry_rows.extend(read(run / "private" / f"retry_worker_rank{rank}_outputs_private.jsonl"))
    accepted_retry = sum(1 for row in retry_rows if row["trialmatch"].get("parse_status") == "accepted")
    counts_before = old_status["parse_status_counts"]
    cm_before, cm_after = old_cond["new_parser"], new_cond["new_parser"]
    a_old = old_rule["methods"]["trialmatch_model_only"]
    a_new = methods["trialmatch_model_only"]
    b_old = old_rule["methods"]["combined"]
    b_new = methods["combined"]
    qwen = methods["qwen_v4"]
    worker_status = [json.loads((run / "private" / f"retry_worker_rank{i}_status_private.json").read_text()) for i in range(WORLD_SIZE)]
    wall = max(s.get("elapsed_seconds", 0.0) for s in worker_status)
    allocated = sum(row["trialmatch"].get("inference_seconds", 0.0) for row in retry_rows)
    lines = ["# TrialMatchAI 失败条件定向重生成与复评", "",
        f"生成状态：{statuses.get('status')}；只重跑原先失败的 69/492 条条件（截断 29、目标不明确 40），原 423 条成功输出逐字保留。模型调用只使用本地 Phi-4 + TrialMatchAI LoRA；Qwen3 未调用。生成提示版本 `{PROMPT_VERSION}`，上限 {MAX_NEW_TOKENS} token，批大小 {BATCH_SIZE}，使用 4 张 A100。", "",
        "## 解析结果", "",
        "| 指标 | 重跑前 | 重跑后 |", "|---|---:|---:|"]
    for label, key in (("成功解析", "accepted"), ("真正截断", "truncated"), ("目标不明确", "ambiguous_target")):
        before = counts_before.get(key, 0)
        after = counts.get(key, 0)
        lines.append(f"| {label} | {before}/492 ({before/4.92:.2f}%) | {after}/492 ({after/4.92:.2f}%) |")
    lines += ["", f"69 条重跑中严格解析成功 {accepted_retry}/69；没有放宽原文/标签核验。重跑后全体解析状态：`{json.dumps(dict(counts), ensure_ascii=False)}`。", "",
        "## 条件判断与规则适用性", "",
        "| 指标 | 重跑前 | 重跑后 |", "|---|---:|---:|"]
    for label, old, new in (("原子条件准确率", cm_before, cm_after), ("原子条件宏 F1", cm_before, cm_after)):
        if label == "原子条件准确率":
            lines.append(f"| {label} | {old['correct']}/{old['n']} ({old['accuracy']:.2%}) | {new['correct']}/{new['n']} ({new['accuracy']:.2%}) |")
        else:
            lines.append(f"| {label} | {old['macro_f1']:.4f} | {new['macro_f1']:.4f} |")
    lines += ["", "| 规则级方法 | 重跑前准确率 / 宏 F1 | 重跑后准确率 / 宏 F1 |", "|---|---:|---:|"]
    for key, name in (("trialmatch_model_only", "TrialMatchAI 单独"), ("trialmatch", "TrialMatchAI + 数值比较"),
                      ("combined", "TrialMatchAI + BioLinkBERT（缓存证据）"), ("qwen_v4", "V4 已保存 Qwen3 对照")):
        old, new = old_rule["methods"][key], methods[key]
        lines.append(f"| {name} | {old['correct']}/192 ({old['accuracy']:.2%}) / {old['macro_f1']:.4f} | {new['correct']}/192 ({new['accuracy']:.2%}) / {new['macro_f1']:.4f} |")
    lines += ["", f"新 TrialMatchAI 准确率相对历史 Qwen3 为 {(a_new['accuracy']-qwen['accuracy'])*100:+.2f} 个百分点。加入 BioLinkBERT 后相对单独 TrialMatchAI {(b_new['accuracy']-a_new['accuracy'])*100:+.2f} 个百分点；证据定位 {b_new.get('known_evidence_correct')}/{b_new.get('known_evidence_n')} ({b_new.get('known_evidence_accuracy',0):.2%})，判定和证据同时正确 {b_new.get('joint_correct')}/192。证据关系沿用同一批已缓存 MedNLI 输出，本次没有重跑 BioLinkBERT，因此不能把证据指标变化归因于重生成。", "",
        f"本次 4 卡并行墙钟 {wall/60:.1f} 分钟；69 条输出分摊生成 GPU 工作 {allocated/60:.1f} GPU 分钟。单卡峰值显存：" + ", ".join(f"GPU {s['physical_gpu']}: {s.get('peak_allocated_gib',0):.1f} GiB allocated / {s.get('peak_reserved_gib',0):.1f} GiB reserved" for s in worker_status) + ".", "",
        "## 重跑后仍未严格解析的条件", ""]
    if failures:
        lines += ["| 样本 | 条件 | 原解析状态 | 新解析状态 | 错误 |", "|---|---|---|---|---|"]
        retry_keys = {identity(row) for row in retry_rows}
        for row in failures:
            if identity(row) in retry_keys:
                lines.append(f"| {row['sample_id']} | {row['condition_id']} | 重跑 | {row['parse_status']} | {row.get('parse_error') or '—'} |")
    else:
        lines.append("无。")
    lines += ["", "## 产物", "",
        f"- 逐条件解析及冻结记录：`{conditions}/conditions_public.jsonl`、`{conditions}/REPARSE_FREEZE.json`。",
        f"- 原子条件评价和错误：`{conditions}/condition_eval/condition_metrics.json`、`{conditions}/condition_eval/condition_predictions_public.jsonl`。",
        f"- 规则级指标和逐例错误：`{conditions}/metrics.json`、`{conditions}/errors_public.jsonl`。",
        f"- 私有重跑原文与合并输出：`{run}/private/`；不包含在公开预测文件中。", "",
        "192 条数据已用于方法开发，属于同一集合的探索性回归，不是独立验证。"]
    out = run / "RETRY_REPORT.zh.md"
    out.write_text("\n".join(lines) + "\n")
    os.chmod(out, 0o600)


def orchestrate(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    os.chmod(run, 0o700)
    private = run / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    os.chmod(private, 0o700)
    start = time.perf_counter()
    global_args = argparse.Namespace(**vars(args))
    prepare(global_args)
    worker_logs = []
    processes = []
    for rank, device in enumerate(args.gpus):
        log_path = private / f"retry_worker_rank{rank}.log"
        log_handle = log_path.open("w")
        os.chmod(log_path, 0o600)
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_XET="1")
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--run-dir", str(run),
                   "--rank", str(rank), "--device", str(device), "--base-model", args.base_model,
                   "--adapter-model", args.adapter_model]
        processes.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT), log_handle))
        worker_logs.append(str(log_path))
    write_json(private / "retry_controller_status_private.json", {
        "status": "workers_launched", "start_utc": now(), "pid": os.getpid(),
        "physical_gpus": args.gpus, "worker_pids": [p.pid for p, _ in processes],
        "target_count": 69, "worker_logs": worker_logs})
    deadline = time.monotonic() + 240
    stable = False
    while time.monotonic() < deadline:
        states = []
        for rank in range(WORLD_SIZE):
            path = private / f"retry_worker_rank{rank}_status_private.json"
            states.append(json.loads(path.read_text()).get("status") if path.exists() else "not_started")
        if all(value in {"running", "completed"} for value in states):
            stable = True
            break
        if any(value == "failed" for value in states):
            break
        time.sleep(3)
    write_json(private / "retry_controller_status_private.json", {
        "status": "stable_running" if stable else "startup_incomplete", "startup_observed_utc": now(),
        "physical_gpus": args.gpus, "worker_pids": [p.pid for p, _ in processes],
        "worker_states": states, "continuous_monitoring": False})
    return_code = 0
    for process, log_handle in processes:
        return_code = max(return_code, process.wait())
        log_handle.close()
    worker_statuses = [json.loads((private / f"retry_worker_rank{i}_status_private.json").read_text()) for i in range(WORLD_SIZE)]
    if return_code or any(s.get("status") != "completed" for s in worker_statuses):
        write_json(private / "retry_controller_status_private.json", {
            "status": "failed", "finished_utc": now(), "worker_statuses": worker_statuses,
            "error": "one or more required-model workers failed; no fallback was attempted"})
        raise RuntimeError("TrialMatchAI retry workers failed; see private worker status and logs")
    merged_path, baseline_path, merge_summary = merge_retry(argparse.Namespace(
        run_dir=run, raw=args.raw, baseline=args.baseline))
    parse_dir = run / "parser_v3_retry"
    if parse_dir.exists():
        raise FileExistsError(f"refusing to overwrite {parse_dir}")
    parser_script = EXP / "scripts/reparse_trialmatch.py"
    commands = [
        [sys.executable, str(parser_script), "--raw", str(merged_path), "--rules", str(args.rules),
         "--baseline", str(baseline_path), "--output", str(parse_dir)],
        [sys.executable, str(EXP / "scripts/evaluate_reparsed_conditions.py"),
         "--conditions", str(parse_dir / "conditions_public.jsonl"),
         "--predictions", str(parse_dir / "predictions_public.jsonl"),
         "--reference-facts", args.reference_facts,
         "--reparse-freeze", str(parse_dir / "REPARSE_FREEZE.json"),
         "--output", str(parse_dir / "condition_eval")],
        [sys.executable, str(EXP / "scripts/evaluate.py"), "--labels", str(args.labels),
         "--pairs", str(args.pairs), "--predictions", str(parse_dir / "predictions_public.jsonl"),
         "--qwen-reference", str(args.qwen_reference), "--output", str(parse_dir / "metrics.json"),
         "--errors", str(parse_dir / "errors_public.jsonl")],
    ]
    for index, command in enumerate(commands):
        log = private / f"offline_evaluation_{index}.log"
        with log.open("w") as handle:
            os.chmod(log, 0o600)
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"offline evaluation stage {index} failed; see private log")
    run_status = {
        "status": "completed", "created_utc": now(), "startup_stable": stable,
        "retry_conditions": 69, "unchanged_original_conditions": 423,
        "retry_parse_successes": sum(1 for rank in range(WORLD_SIZE)
            for row in read(private / f"retry_worker_rank{rank}_outputs_private.jsonl")
            if row["trialmatch"].get("parse_status") == "accepted"),
        "worker_statuses": worker_statuses, "merge": merge_summary,
        "retry_prompt_version": PROMPT_VERSION, "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE, "qwen_called": False, "labels_read_during_generation": False,
        "cached_mednli_reused": True, "mednli_rerun": False,
        "run_wall_seconds": time.perf_counter() - start,
    }
    write_json(run / "run_status.json", run_status)
    report(run, EXP / "runs/v5_20261018/metrics.json", parse_dir,
           parse_dir / "metrics.json", run_status, None)
    write_json(private / "retry_controller_status_private.json", {
        "status": "completed", "finished_utc": now(), "run_wall_seconds": time.perf_counter()-start,
        "retry_successes": run_status["retry_parse_successes"], "report": str(run / "RETRY_REPORT.zh.md")})
    print(json.dumps({"status": "completed", "retry_parse_successes": run_status["retry_parse_successes"],
        "parse_status_counts": json.loads((parse_dir / "run_status.json").read_text())["parse_status_counts"],
        "rule_metrics": {k: {field: v.get(field) for field in ("correct", "accuracy", "macro_f1", "known_evidence_accuracy", "joint_correct")}
            for k, v in json.loads((parse_dir / "metrics.json").read_text())["methods"].items()},
        "report": str(run / "RETRY_REPORT.zh.md")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "orchestrate"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
        p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
        p.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
        p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
        p.add_argument("--rules", type=Path, default=EXP / "data/rules_v5.jsonl")
        p.add_argument("--labels", type=Path, default=EXP / "data/labels_public.jsonl")
        p.add_argument("--pairs", type=Path, default=EXP / "data/pairs_public.jsonl")
        p.add_argument("--qwen-reference", type=Path, default=EXP / "data/v4_qwen_direct_public.jsonl")
        p.add_argument("--reference-facts", type=Path, default=Path("/home/gcx/care_module1_20261018_v4/data/labels.jsonl"))
        p.add_argument("--base-model", default=BASE)
        p.add_argument("--adapter-model", default=ADAPTER)
        p.add_argument("--gpus", nargs=4, type=int, default=[4, 5, 6, 7])
        p.set_defaults(function=prepare if name == "prepare" else orchestrate)
    p = sub.add_parser("worker")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--base-model", default=BASE)
    p.add_argument("--adapter-model", default=ADAPTER)
    p.set_defaults(function=worker)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
