"""Complete native MedPIC orchestration, using preloaded inference callbacks.

No dataset access in this module. Formal input reading lives behind a frozen
configuration check in run_medpic.py. Fusion reuses native_generation_baselines.
"""
from __future__ import annotations

from dataclasses import replace
import json
import time

from aster_seka_adapter import Control, PhasePolicy
from medpic_native_adapter import case_rng_seed, fingerprint, parse_final_answer
from medpic_option_transfer import (ARMS, PROTOCOL, assessment, condition_prompt, direct_applicability_set,
    execution_prompt, option_evidence, summarize_costs)


CONTROL_NAMES = {"zero": Control.MATCHED_ZERO, "wrong": Control.WRONG_SPAN,
                 "misaligned": Control.MISALIGNED_RULE, "reverse": Control.REVERSE}
FUSION_SOURCES = ("A0", "A1")


def method_names(fusion_sources=FUSION_SOURCES):
    if len(set(fusion_sources)) != len(fusion_sources) or any(x not in FUSION_SOURCES for x in fusion_sources):
        raise ValueError("Fusion sources must be a unique subset of A0/A1")
    methods = ["native_full_text", *ARMS]
    for a in FUSION_SOURCES:
        methods += [f"{a}B1_{name}" for name in CONTROL_NAMES]
        methods += [f"{a}_direct_mapping", f"{a}_text_emphasis"]
        if a in fusion_sources:
            methods += [f"{a}_fusion_{kind}{suffix}" for kind in ("static", "dynamic") for suffix in ("", "_zero")]
    return methods


def support_prompt(task, assessments, evidence):
    base = execution_prompt(task, assessments, evidence)
    quotes = [task.patient_vignette[span.start:span.end] for span in base.evidence.spans]
    if not quotes:
        return base
    return replace(base, text=base.text + "\n\nAttend to these predicted relevant patient quotations (unverified):\n" +
                   "\n\n".join(quotes) + "\nPreserve the required FINAL_ANSWER format.")


def numerical_drift(reference, candidate, *, reference_name, logprob=False, temperature=1.):
    if reference is None or candidate is None:
        return {"status": "unavailable", "reference": reference_name, "reason": "missing first-token score tensor"}
    import torch
    reference, candidate = reference.float(), candidate.float()
    if logprob:
        reference = (reference / temperature).log_softmax(-1)
    if reference.shape != candidate.shape or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return {"status": "unavailable", "reference": reference_name, "reason": "shape mismatch or nonfinite scores"}
    return {"status": "complete_verified", "reference": reference_name,
            "max_abs_first_token_difference": float((reference - candidate).abs().max()),
            "argmax_changed": bool(reference.argmax() != candidate.argmax()),
            "semantics": "log probabilities" if logprob else "first-token generation scores"}


def run_case(task, *, sample_id, seed, config, lock_sha256, infer, generate_baseline,
             fusion_sources=FUSION_SOURCES, fusion_base_temperature=1., on_trace=None):
    """Run fixed core, all phase controls, text references and vocabulary fusion.

    infer accepts raw PreparedPrompt, PhasePolicy, Control, budget, seed and is
    responsible for final-tokenization control construction. generate_baseline
    accepts primary/auxiliary PreparedPrompt and kind/control/budget/seed.
    Both retain measured failed calls. Every method emits a row on every case.
    """
    config.validate()
    evidence = option_evidence(task)
    allowed = [key for key, _ in task.options]
    case_seed = case_rng_seed(seed, sample_id)
    traces, score_tensors, rows = [], {}, []
    by_id, decision, a_refs = {}, {}, {}

    def invoke(trace_id, call, *, prompt=None, auxiliary=None):
        started = time.perf_counter()
        try:
            result = dict(call())
        except Exception as error:
            result = {"status": "error", "text": "", "error_type": type(error).__name__, "error": str(error),
                      "wall_seconds": time.perf_counter() - started}
        # Raw scores belong in separate torch files, never JSON or public data.
        values = {}
        for key in ("_first_token_scores_tensor", "first_log_probs", "first_raw_base"):
            value = result.pop(key, None)
            if value is not None:
                values[key] = value
        if values:
            score_tensors[trace_id] = values
        result.setdefault("status", "ok")
        result.setdefault("text", "")
        if "wall_seconds" not in result and "elapsed_seconds" in result:
            result["wall_seconds"] = result["elapsed_seconds"]
        if "input_tokens" not in result and "input_tokens_per_path" in result:
            result["input_tokens"] = sum(result["input_tokens_per_path"])
        if trace_id in by_id:
            raise ValueError("Duplicate inference trace ID")
        entry = {"trace_id": trace_id, "generation": result}
        if prompt is not None:
            entry["prompt"] = prompt.text
        if auxiliary is not None:
            entry["auxiliary_prompt"] = auxiliary.text
        traces.append(entry)
        by_id[trace_id] = result
        if on_trace is not None:
            # Persist completed calls and their tensors immediately. Final traces
            # additionally contain later numerical/fallback annotations.
            on_trace(entry, values)
        return result

    def score(trace_id, key="_first_token_scores_tensor"):
        return score_tensors.get(trace_id, {}).get(key)

    def add_row(method, result, refs, prompt, *, direct=None, extra=None):
        if direct is None:
            prediction = parse_final_answer(result["text"], allowed) if result["status"] in {"ok", "control_unavailable"} else None
        else:
            prediction = direct
        valid = prediction is not None and bool(prediction)
        row = {"sample_id": str(sample_id), "method": method, "seed": seed, "protocol": PROTOCOL,
               "prediction": prediction, "parse_valid": valid, "status": result["status"],
               "config_lock_sha256": lock_sha256, "case_rng_seed": case_seed,
               "prompt_sha256": fingerprint(prompt.text), "output_sha256": fingerprint(result["text"]),
               "cost_references": list(dict.fromkeys(refs)),
               "control_executed": result.get("control_executed"),
               "control_unavailable_reason": result.get("control_unavailable_reason"),
               "reused_from": result.get("reused_from"),
               "numerical_control": result.get("numerical_control")}
        if row["status"] == "ok" and not valid:
            row["status"] = "parse_error"
        if extra:
            row.update(extra)
        rows.append(row)

    # Plain native reference has no A-stage predictions, no additional documents.
    native = execution_prompt(task)
    result = invoke("native_full_text", lambda: infer(native, PhasePolicy(False, False), Control.NO_HOOK,
                                                      config.execution_max_new_tokens, case_seed), prompt=native)
    add_row("native_full_text", result, ["native_full_text"], native)

    # A0/A1 judgments shared by B, fusion and text comparators; no repeated fitting.
    for a in (0, 1):
        key, current, refs = f"A{a}", [], []
        for label in allowed:
            prompt = condition_prompt(task, label, evidence)
            trace_id = f"{key}/{label}"
            result = invoke(trace_id, lambda: infer(prompt, PhasePolicy(bool(a), False),
                Control.NORMAL if a else Control.NO_HOOK, 1, case_seed), prompt=prompt)
            item = assessment(label, result)
            traces[-1]["assessment"] = item
            current.append(item)
            refs.append(trace_id)
        decision[key], a_refs[key] = current, refs

    # All A controls remain diagnostics; they never silently replace normal A1.
    for label in allowed:
        prompt = condition_prompt(task, label, evidence)
        for suffix, control in CONTROL_NAMES.items():
            trace_id = f"A1_{suffix}/{label}"
            result = invoke(trace_id, lambda: infer(prompt, PhasePolicy(True, False), control, 1, case_seed), prompt=prompt)
            if result["status"] == "control_unavailable":
                result.update(reused_from=f"A0/{label}", fallback_assessment=assessment(label, by_id[f"A0/{label}"]))
            else:
                result["assessment"] = assessment(label, result)
            if suffix == "zero":
                result["numerical_control"] = numerical_drift(score(f"A0/{label}"), score(trace_id),
                                                              reference_name=f"A0/{label}")

    for a in (0, 1):
        key = f"A{a}"
        base = execution_prompt(task, decision[key], evidence)
        support = support_prompt(task, decision[key], evidence)
        nohook_id = f"{key}B0"
        result = invoke(nohook_id, lambda: infer(base, PhasePolicy(bool(a), False), Control.NO_HOOK,
                         config.execution_max_new_tokens, case_seed), prompt=base)
        add_row(nohook_id, result, a_refs[key] + [nohook_id], base)
        regular_id = f"{key}B1"
        result = invoke(regular_id, lambda: infer(base, PhasePolicy(bool(a), True), Control.NORMAL,
                         config.execution_max_new_tokens, case_seed), prompt=base)
        add_row(regular_id, result, a_refs[key] + [regular_id], base)

        for suffix, control in CONTROL_NAMES.items():
            trace_id = f"{key}B1_{suffix}"
            result = invoke(trace_id, lambda: infer(base, PhasePolicy(bool(a), True), control,
                config.execution_max_new_tokens, case_seed), prompt=base)
            refs = a_refs[key] + [trace_id]
            if result["status"] == "control_unavailable":
                # Preserve full-row prediction from the actual same-A no-hook run.
                fallback = by_id[nohook_id]
                result.update(text=fallback["text"], reused_from=nohook_id,
                              fallback_status=fallback["status"])
                refs.append(nohook_id)
            if suffix == "zero":
                result["numerical_control"] = numerical_drift(score(nohook_id), score(trace_id), reference_name=nohook_id)
            add_row(trace_id, result, refs, base)

        mapped = direct_applicability_set(task, decision[key])
        trace_id = f"{key}_direct_mapping"
        result = invoke(trace_id, lambda: {"text": "FINAL_ANSWER: " + json.dumps(mapped), "status": "ok",
            "actual_forward_calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0.,
            "peak_allocated_gib": 0., "peak_reserved_gib": 0.})
        add_row(trace_id, result, a_refs[key] + [trace_id], base, direct=mapped,
                extra={"direct_mapping_not_probability_fusion": True})

        trace_id = f"{key}_text_emphasis"
        result = invoke(trace_id, lambda: infer(support, PhasePolicy(False, False), Control.NO_HOOK,
                         config.execution_max_new_tokens, case_seed), prompt=support)
        add_row(trace_id, result, a_refs[key] + [trace_id], support)

        if key in fusion_sources:
            for kind in ("static", "dynamic"):
                for suffix, control in (("", "normal"), ("_zero", "matched_zero")):
                    trace_id = f"{key}_fusion_{kind}{suffix}"
                    result = invoke(trace_id, lambda: generate_baseline(base, support, kind=kind, control=control,
                        max_new_tokens=config.execution_max_new_tokens, seed=case_seed), prompt=base, auxiliary=support)
                    if suffix:
                        result["numerical_control"] = numerical_drift(score(nohook_id), score(trace_id, "first_log_probs"),
                            reference_name=nohook_id + " at declared base temperature", logprob=True,
                            temperature=fusion_base_temperature)
                        result["raw_primary_first_token_drift"] = numerical_drift(score(nohook_id),
                            score(trace_id, "first_raw_base"), reference_name=nohook_id)
                    add_row(trace_id, result, a_refs[key] + [trace_id], base,
                            extra={"auxiliary_prompt_sha256": fingerprint(support.text),
                                   "fusion_kind": kind, "fusion_condition_source": key,
                                   "additional_vocabulary_path": True})

    expected = set(method_names(fusion_sources))
    if {r["method"] for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Fixed method coverage mismatch")
    return {"predictions": rows, "private_traces": traces, "assessments": decision,
            "cost_summary": summarize_costs(traces, rows), "score_tensors": score_tensors}
