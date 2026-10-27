#!/usr/bin/env python3
"""Run the unchanged causal protocol with validated batch-one suffix execution.

Arguments are exactly those of run_module2_causal_pilot.py. This wrapper changes
only the in-process execution adapter. It does not rewrite that script or its
adapter, and does not use the optional approximate multirow suffix-batch path.
Every prompt receives exact full-vocabulary and attention zero validation.
The first nonzero key/bias condition for each sign/target support is additionally
compared to the original full-depth execution with zero absolute tolerance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    import run_module2_causal_pilot as canonical
    if "--help" in sys.argv or "-h" in sys.argv:
        canonical.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    location, _ = parser.parse_known_args()
    import torch
    import src.module3.applicability_shift as standard
    from src.module3.applicability_tail import prepare_tail, tail_decision_logits, compare_results
    reference = standard.decision_logits
    reference_prefix = standard.prepare_prefix
    runtime = {"last_state": None, "tail": None, "validated": set()}
    profiler_handles, profile_clock = [], []
    phase = {"value": "decision"}
    audit = {
        "schema_version": "module2.suffix_execution.v1", "status": "running",
        "protocol": "unchanged canonical causal runner",
        "optimization": "reuse decision-token inputs below the one edited layer; share immutable prefix cache tensors",
        "batch_size": 1, "zero_logit_atol": 0., "zero_attention_atol": 0.,
        "thread_policy": {"prefill": 4, "decision": location.threads,
                          "scope": "execution only; scientific configuration unchanged"},
        "lm_head_profile": {"clock": "CPU synchronous wall time; begins at first baseline",
                            "prefill": {"calls": 0, "seconds": 0.},
                            "decision": {"calls": 0, "seconds": 0.}},
        "nonzero_validation_logit_atol": 0., "nonzero_validation_attention_atol": 0.,
        "candidate_and_full_vocabulary_argmax": "unchanged by bitwise logit equality",
        "source_files": {str(path.relative_to(ROOT)): canonical.sha256(path) for path in (
            Path(__file__), ROOT / "src/module3/applicability_tail.py",
            ROOT / "scripts/run_module2_causal_pilot.py", ROOT / "src/module3/applicability_shift.py")},
        "checks": [], "optimized_decision_calls": 0, "reference_validation_calls": 0,
    }
    started = time.monotonic()
    def save():
        if location.output.exists():
            audit["elapsed_seconds"] = time.monotonic() - started
            canonical.write_json(location.output / "execution_optimization.json", audit)
    def prefix_with_stage_threads(*args, **kwargs):
        previous = torch.get_num_threads()
        phase["value"] = "prefill"
        torch.set_num_threads(4)
        try:
            return reference_prefix(*args, **kwargs)
        finally:
            torch.set_num_threads(previous)
            phase["value"] = "decision"
    def accelerated(model, state, shifts=(), score_bias=(), *, return_diagnostics=False,
                    diagnostic_targets=()):
        shifts, score_bias, targets = list(shifts), list(score_bias), list(diagnostic_targets)
        if not profiler_handles and model.get_input_embeddings().weight.device.type == "cpu":
            def begin_head(module, args):
                profile_clock.append((time.monotonic(), phase["value"]))
            def end_head(module, args, output):
                clock, name = profile_clock.pop()
                profile = audit["lm_head_profile"][name]
                profile["calls"] += 1
                profile["seconds"] += time.monotonic() - clock
            profiler_handles.extend((model.lm_head.register_forward_pre_hook(begin_head),
                                     model.lm_head.register_forward_hook(end_head)))
        if runtime["last_state"] is not state:
            if shifts or score_bias or not targets:
                raise ValueError("A suffix prompt must start with the unchanged diagnostic baseline")
            heads = {(int(t["layer"]), int(t["kv_head"])) for t in targets}
            if len(heads) != 1:
                raise ValueError("The accelerated pilot supports exactly one frozen layer/KV head")
            layer, head = next(iter(heads))
            # Release the preceding prompt: retaining every prefix would waste memory.
            runtime.update(last_state=None, tail=None, validated=set())
            tail = prepare_tail(model, state, layer, head, targets)
            reproduced = tail_decision_logits(model, tail, return_diagnostics=True,
                                              diagnostic_targets=targets)
            check = compare_results(tail.baseline, reproduced)
            audit["reference_validation_calls"] += 1
            audit["checks"].append({"kind": "exact_baseline_suffix_identity", "layer": layer,
                "kv_head": head, "prefix_length": state.prefix_length, **check})
            runtime.update(last_state=state, tail=tail)
            save()
            return tail.baseline if return_diagnostics else tail.baseline["logits"]
        result = tail_decision_logits(model, runtime["tail"], shifts, score_bias,
                                     return_diagnostics=return_diagnostics, diagnostic_targets=targets)
        audit["optimized_decision_calls"] += 1
        nonzero_shifts = [s for s in shifts if float(s.get("coefficient", 1.)) != 0.]
        nonzero_biases = [b for b in score_bias if float(b["bias"]) != 0.]
        if not nonzero_shifts and not nonzero_biases:
            original = runtime["tail"].baseline
            if not return_diagnostics:
                original = original["logits"]
            compare_results(original, result)
        else:
            signature = tuple(
                (kind, 1 if float(entry.get("coefficient", 1.) if kind == "shift" else entry["bias"]) > 0 else -1,
                 tuple(entry.get("positions", state.target_positions)))
                for kind, entries in (("shift", nonzero_shifts), ("bias", nonzero_biases))
                for entry in entries)
            if signature not in runtime["validated"]:
                original = reference(model, state, shifts, score_bias,
                    return_diagnostics=return_diagnostics, diagnostic_targets=targets)
                check = compare_results(original, result)
                audit["reference_validation_calls"] += 1
                audit["checks"].append({"kind": "exact_nonzero_full_depth_comparison",
                    "intervention_signature": signature, **check})
                runtime["validated"].add(signature)
                save()
        return result
    standard.decision_logits = accelerated
    standard.prepare_prefix = prefix_with_stage_threads
    try:
        canonical.main()
        audit["status"] = "completed"
    except BaseException as error:
        audit["status"] = "failed"
        audit["error_type"] = type(error).__name__
        audit["error"] = str(error)
        raise
    finally:
        standard.decision_logits = reference
        standard.prepare_prefix = reference_prefix
        for handle in profiler_handles:
            handle.remove()
        save()


if __name__ == "__main__":
    main()
