"""Prepare, never execute, official Base AdaSEKA queue commands on the server.

The coordinator reviews the generated JSON and announces resources before using
run_baseline_queue.py. All paths are actual server paths supplied by the caller.
"""
import argparse
import hashlib
import json
from pathlib import Path


PIN = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, name, task, subset, strength):
    output = args.output_root / name
    module = "benchmarks.eval_bias_gen" if task == "biasbios" else "benchmarks.eval_fact_gen"
    native = ["--model", str(args.model), "--data_path", str(args.biasbios if task == "biasbios" else args.counterfact_dir),
              "--output_dir", str(output), "--example_subset", subset, "--batch_size", "1",
              "--max_new_tokens", "64" if task == "biasbios" else "1"]
    if task != "biasbios":
        benchmarks = ["efficacy", "paraphrase"] if strength == 3. else ["efficacy"]
        native += ["--benchmarks", *benchmarks, "--add_unmediated_fact", "True"]
    if strength is not None:
        native += ["--adaptive-seka", "--adaptive-expert-path", str(args.expert_config),
                   "--adaptive_amplify_factor", str(strength), "--top_k_singular", "5",
                   "--combination_method", "weighted_top_k", "--layers", "last10"]
    if task == "biasbios":
        native = ["--report", str(output / "scipy_compat_report.json"), "--", *native]
        module = "biasbios_compat"
    if strength is None:
        native = ["--torch-dtype", "float32", "--upstream-module", module,
                  "--report", str(output / "dtype_load_report.json"), "--", *native]
        module = "dtype_load_compat"
    argv = [str(args.python), str(args.runner), "--gpu", str(args.gpu),
            "--source", str(args.source), "--model", str(args.model), "--output", str(output),
            "--seed", "0", "--module", module, "--", *native]
    count = int(subset.split(":")[1]) - int(subset.split(":")[0])
    return {"name": name, "status": "planned", "command": argv, "log": str(args.output_root / (name + ".log")),
            "checkpoint": "Qwen3-8B-Base (actual files bound by run manifest)", "task": task,
            "benchmarks": (["efficacy", "paraphrase"] if strength == 3. else ["efficacy"]) if task == "counterfact" else None,
            "subset": subset, "expected_case_rows": count, "batch_size": 1,
            "expected_dtype": "torch.float32", "adaptive_strength": strength,
            "max_new_tokens": 64 if task == "biasbios" else 1,
            "biasbios_compatibility": ({"module": "biasbios_compat", "report": str(output / "scipy_compat_report.json"),
                                        "scope": "Restore csr_matrix.A as toarray alias; official scoring unchanged"}
                                       if task == "biasbios" else None),
            "expected_forward_calls": (count * (6 if strength == 3. else 2 if strength is not None else 1)) if task == "counterfact" else None,
            "maximum_forward_calls": count * (65 if strength is not None else 64) if task == "biasbios" else None,
            "forward_cost_note": "Ada routes once per input and then generates; EOS may stop BiasBios before64 tokens. CF positive covers efficacy+paraphrase; 30000-call expectation uses the coordinator's actual same-data two deduplicated paraphrase prompts per case; record actual count."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    for name in ("python", "runtime", "source", "model", "counterfact-dir", "biasbios", "expert-config", "expert-audit", "output-root", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True,
                        help="Actual experiment-root run_official_job.py; --runtime remains the data/artifact directory")
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--wait-for-pids", nargs="+", type=int, required=True,
                        help="Current Base and parser-v2 attention parents:425321 427918; explicitly pass both current PIDs")
    parser.add_argument("--smoke-positive-manifest", type=Path)
    parser.add_argument("--smoke-zero-manifest", type=Path)
    args = parser.parse_args()
    for name in ("python", "runtime", "runner", "source", "model", "counterfact_dir", "biasbios", "expert_config", "expert_audit", "output_root", "plan"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.runner.is_file():
        raise FileNotFoundError("Actual official-job runner is missing: " + str(args.runner))
    audit = json.loads(args.expert_audit.read_text())
    if audit.get("status") != "complete_verified" or audit.get("config_ready") is not True:
        raise ValueError("Use the completed Base four-expert audit")
    if audit["expert_config_sha256"] != sha(args.expert_config):
        raise ValueError("Base expert config differs from its audit")
    experts = json.loads(args.expert_config.read_text())
    if set(experts) != {"synthetic", "counterfact", "biasbios", "hotpotqa"}:
        raise ValueError("Use all four audited official expert sources")
    for name, path in experts.items():
        detail = audit["experts"][name]
        if detail["sha256"] != sha(path) or detail["source_commit"] != PIN:
            raise ValueError("Expert changed since the Base audit: " + name)
        manifest = json.loads(Path(detail["manifest_path"]).read_text())
        if Path(manifest["model_path"]).resolve() != args.model:
            raise ValueError("Expert belongs to a different model directory: " + name)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "smoke":
        definitions = [("base_ada_bios_val16_g05", "biasbios", "4500:4516", .5),
                       ("base_ada_bios_val16_zero", "biasbios", "4500:4516", 0.),
                       ("base_bios_val16_nohook_fp32", "biasbios", "4500:4516", None)]
    elif args.phase == "full":
        definitions = [("base_ada_cf5000_g3", "counterfact", "5000:10000", 3.),
                       ("base_ada_cf5000_zero", "counterfact", "5000:10000", 0.),
                       ("base_cf5000_nohook_fp32", "counterfact", "5000:10000", None),
                       ("base_ada_bios5000_g05", "biasbios", "5000:10000", .5),
                       ("base_ada_bios5000_zero", "biasbios", "5000:10000", 0.)]
    estimates = {}
    for name, path in [("positive", args.smoke_positive_manifest), ("zero", args.smoke_zero_manifest)]:
        if path is None:
            continue
        measured = json.loads(path.read_text())
        if measured.get("exit_code") != 0 or measured.get("process_state") != "exited":
            raise ValueError("Do not extrapolate cost from an incomplete/failed smoke")
        estimates[name] = {"source_manifest_sha256": sha(path), "measured_cases": 16,
                           "measured_elapsed_seconds": measured["elapsed_seconds"],
                           "measured_forward_calls": measured["forward_completed"],
                           "measured_peak_allocated_gib": measured["peak_allocated_gib"],
                           "rough_full5000_seconds": measured["elapsed_seconds"] * 5000 / 16,
                           "limitation": "Simple planning extrapolation includes fixed loading/hash costs and a different validation subset; not a guaranteed full-split runtime or memory bound"}
    plan = {"status": "planned", "phase": args.phase, "gpu": args.gpu, "wait_for_pids": args.wait_for_pids,
            "runner_path": str(args.runner), "runner_sha256": sha(args.runner), "runtime_artifact_directory": str(args.runtime),
            "resource_plan": {"gpu_count": 1, "expected_gpu": "A100 80GiB", "planning_memory_gib": [40, 60],
                              "memory_is_measured": False,
                              "total_forward_upper_bound": 3104 if args.phase == "smoke" else 695000,
                              "time_estimate": "No Ada Bio batch1 timing before smoke; use measured smoke estimates below before launching full. Never substitute batch16 throughput."},
            "source_commit": PIN, "seed": 0, "batch_size": 1, "dtype": "official Ada default FP32; measured dtype must confirm",
            "expert_audit_sha256": sha(args.expert_audit), "expert_config_sha256": sha(args.expert_config),
            "expert_sources": list(experts), "calibration_limit": "Released synthetic file has 100 context_1 rows, not paper random-200 calibration; other expert sources use first 200 requests with official filtering",
            "scheduling": "Wait for BOTH existing Base construction queue and already scheduled clinical attention queue; never start this generator's commands directly while either owns the GPU",
            "launch_authority": "Plan only; coordinator runs reviewed queue after announcing resource/time estimates. This program never launches experiments.",
            "small_run_scope": "16 validation examples verify execution/cost; cannot replace either complete 5000-example task",
            "biasbios_cost_estimates": estimates,
            "official_metrics": {"counterfact": "First generation-token mediated vs unmediated target probabilities, official efficacy; not textual exact-match",
                                 "biasbios": "Official first-step case-variant profession top1/top3 plus generated-text fluency/consistency with max_new_tokens64; smoke candidate label count may be below28",
                                 "paraphrase": "Included by default in full positive g3 run; official deduplicated prompts scored by original case. Zero/nohook full efficacy provides the matched numerical reference."},
            "jobs": [command(args, *definition) for definition in definitions],
            "not_completed_by_plan": ["Actual Base Ada inference", "Independent per-example exports/metrics and grouped confidence intervals",
                                       "Matched numerical comparison", "CounterFact efficacy and paraphrase from the full positive run",
                                       "Paper random-200 calibration reproduction", "Clinical effectiveness"]}
    with args.plan.open("x") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
