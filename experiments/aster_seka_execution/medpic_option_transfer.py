"""Native MedPIC option-applicability transfer; no dataset I/O or model loading.

Prepared using public schema and synthetic fixtures only. The coordinator owns
the model, GPU allocation, final configuration freeze and test-file access.
This module does not call an option a clinical guideline or an action-blind rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time

from aster_seka_adapter import Control, PhasePolicy, PreparedPrompt, PredictedSpans, Span
from clinical_core import counted_control_spans, lexical_patient_spans, verify_tfu_tokens
from medpic_native_adapter import NativeInput, case_rng_seed, fingerprint, parse_final_answer, render_native


PROTOCOL = "medpic_native_option_applicability_seka_v1"
SELECTOR = "original_patient_sentences_lexical_option_overlap_v1"
ARMS = ("A0B0", "A0B1", "A1B0", "A1B1")
CONDITION_INSTRUCTION = (
    "Assess whether the target answer option is applicable to this patient and question. "
    "Use the complete patient, question, and original options below. "
    "Answer T if the target option is supported as a correct selection, F if it is contradicted "
    "or inapplicable, and U if the available information is insufficient. "
    "Do not assume an unmentioned patient fact. Return only T, F, or U. "
    "This is an option-applicability assessment, not an external guideline."
)
ASSESSMENT_SUFFIX = (
    "\n\nAuxiliary predicted option-applicability assessments (unverified; unknown is not false):\n"
)


@dataclass(frozen=True)
class TransferConfig:
    execution_max_new_tokens: int = 512
    context_limit: int = 32768
    condition_max_new_tokens: int = 1
    chat_role: str = "user"
    enable_thinking: bool = False
    greedy: bool = True

    def validate(self):
        if self.execution_max_new_tokens <= 0 or self.context_limit <= 0:
            raise ValueError("Positive fixed token budgets are required")
        if self.condition_max_new_tokens != 1 or self.chat_role != "user" or self.enable_thinking or not self.greedy:
            raise ValueError("This transfer has one-token T/F/U and native user-role greedy decoding")


def source_manifest():
    """Caller can freeze these hashes on the server before opening test files."""
    from pathlib import Path
    import aster_seka_adapter, clinical_core, medpic_native_adapter
    return {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in
            (__file__, aster_seka_adapter.__file__, clinical_core.__file__, medpic_native_adapter.__file__)}


def require_lock(lock, config, seed):
    config.validate()
    if lock.get("protocol") != PROTOCOL or lock.get("frozen") is not True or not lock.get("frozen_at"):
        raise ValueError("The coordinator must freeze this native-transfer protocol before inference")
    if lock.get("seeds") != [0, 1, 2] or seed not in lock["seeds"]:
        raise ValueError("All predeclared seeds 0/1/2 must be retained")
    if lock.get("arms") != list(ARMS) or lock.get("config") != asdict(config):
        raise ValueError("Configuration or factorial arms differ from the frozen declaration")
    if lock.get("selector") != SELECTOR or lock.get("condition_instruction_sha256") != fingerprint(CONDITION_INSTRUCTION):
        raise ValueError("Selector/prompt differs from the frozen declaration")
    if lock.get("source_hashes") != source_manifest():
        raise ValueError("Source files differ from the frozen declaration")


def option_evidence(task: NativeInput):
    """Deterministic lexical matches, not verified clinical support.

    Keep complete original patient sentences, including negation and units. Only
    the target original option is the lexical query; no generated rule is added.
    Shared ranges are deduplicated when constructing the execution mask.
    """
    return {label: lexical_patient_spans(task.patient_vignette, text, label)
            for label, text in task.options}


def condition_prompt(task: NativeInput, label: str, evidence=None):
    if label not in dict(task.options):
        raise ValueError("Target is not an original option")
    evidence = option_evidence(task) if evidence is None else evidence
    text = CONDITION_INSTRUCTION + "\n\nPatient:\n"
    fields = {"patient_history": (len(text), task.patient_vignette)}
    text += task.patient_vignette + "\n\nQuestion:\n" + task.question + "\n\nOptions:\n"
    text += "\n".join(f"{key}. {value}" for key, value in task.options)
    text += f"\n\nTarget option: {label}\nAssessment:"
    return PreparedPrompt(text, fields, PredictedSpans(fingerprint(SELECTOR), tuple(evidence[label])),
                          "condition", "tfu")


def validate_assessments(task: NativeInput, assessments):
    if [x["option"] for x in assessments] != [x[0] for x in task.options]:
        raise ValueError("Assessments must cover every original option in original order")
    for item in assessments:
        if item["label"] not in {"T", "F", "U"}:
            raise ValueError("Unknown applicability value")
        p = item.get("candidate_normalized_probabilities")
        if p is not None and (set(p) != {"T", "F", "U"} or
                any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()) or
                abs(sum(p.values()) - 1) > 1e-5):
            raise ValueError("Invalid T/F/U probability vector")


def support_spans(task, assessments, evidence=None):
    validate_assessments(task, assessments)
    evidence = option_evidence(task) if evidence is None else evidence
    selected, seen = [], set()
    for item in assessments:
        if item["label"] != "T" or item.get("status") != "ok":
            continue
        for span in evidence[item["option"]]:
            interval = (span.source, span.start, span.end)
            if interval not in seen:
                seen.add(interval)
                selected.append(span)
    return tuple(selected)


def execution_prompt(task: NativeInput, assessments=None, evidence=None):
    native = render_native(task)
    spans = () if assessments is None else support_spans(task, assessments, evidence)
    text = native.text
    if assessments is not None:
        # Only model predictions are appended; the original complete input stays.
        public = [{k: item.get(k) for k in ("option", "label", "status", "candidate_normalized_probabilities")}
                  for item in assessments]
        text += ASSESSMENT_SUFFIX + json.dumps(public, ensure_ascii=False, sort_keys=True)
        text += "\nUse the complete original question and options. Preserve the required FINAL_ANSWER format."
    return PreparedPrompt(text, {"patient_history": (native.patient_char_span[0], task.patient_vignette)},
                          PredictedSpans(fingerprint(SELECTOR), spans), "execution", "multi_select")


def chat_prompt(prompt, tokenizer):
    """Preserve the original MedPIC user role, without modifying the MEGA core."""
    messages = [{"role": "user", "content": prompt.text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    if rendered.count(prompt.text) != 1:
        raise ValueError("Chat template must retain exactly one unchanged source prompt")
    shift = rendered.index(prompt.text)
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False)
    # The shared StageSEKAAdapter uses add_special_tokens=True. Prove that it
    # agrees with the actual tokenizer rather than silently adding duplicate BOS.
    actual = tokenizer(rendered, add_special_tokens=True, truncation=False)["input_ids"]
    if actual != expected:
        raise ValueError("SEKA encoding and native chat-template tokens disagree")
    return replace(prompt, text=rendered,
                   fields={name: (offset + shift, value) for name, (offset, value) in prompt.fields.items()})


def attach_controls(prompt, adapter, evidence, option_order, target=None):
    """Patient-only wrong spans and rotated option-to-patient evidence.

    'misaligned_rule' is an existing adapter enum; here it means option/evidence
    misalignment, not the existence of a guideline. Empty controls are unavailable.
    """
    _, patient = prompt.fields["patient_history"]
    pool = (Span("patient_history", 0, len(patient), "wrong"),) if patient else ()
    wrong = counted_control_spans(prompt, adapter, pool, disjoint=True)
    misplaced = []
    if len(option_order) > 1:
        rotate = dict(zip(option_order, (*option_order[1:], option_order[0])))
        selected_labels = [target] if target is not None else list(dict.fromkeys(s.rule_id for s in prompt.evidence.spans))
        for label in selected_labels:
            misplaced.extend(replace(span, rule_id=label) for span in evidence[rotate[label]])
    aligned = counted_control_spans(prompt, adapter, tuple(misplaced), disjoint=False) if misplaced else ()
    return replace(prompt, evidence=replace(prompt.evidence, wrong_spans=wrong, misaligned_rule_spans=aligned))


def assessment(label, result):
    """U on failure is missing information, never a made-up confidence of 1."""
    scores = result.get("label_logits")
    item = {"option": label, "label": "U", "status": result.get("status", "error"),
            "candidate_normalized_probabilities": None,
            "label_probabilities": result.get("label_probabilities")}
    if result.get("status") == "ok" and isinstance(scores, dict) and set(scores) == {"T", "F", "U"} and all(math.isfinite(v) for v in scores.values()):
        values = [math.exp(scores[k] - max(scores.values())) for k in ("T", "F", "U")]
        probabilities = dict(zip(("T", "F", "U"), (v / sum(values) for v in values)))
        item.update(label=max(probabilities, key=probabilities.get),
                    candidate_normalized_probabilities=probabilities,
                    score_definition="within-option first-token T/F/U scores, not calibrated confidence")
    else:
        item.update(status="error", fallback_reason="Missing/nonfinite first-token scores; U fallback")
    return item


def run_factorial(task: NativeInput, *, sample_id, seed, lock, infer, config=TransferConfig()):
    """Return private traces + public predictions for exactly the four core arms.

    infer(prompt, policy, control, token_budget, case_seed) owns actual inference
    and costs. It receives no raw dataset row, label, taxonomy or pair metadata.
    Shared A calls are logged once; each method references its full standalone
    cost. Conditions are not recomputed separately for B0 and B1.
    """
    require_lock(lock, config, seed)
    evidence = option_evidence(task)
    rng = case_rng_seed(seed, sample_id)
    traces, conditions, predictions, score_tensors = [], {}, [], {}
    def invoke(trace_id, prompt, policy, control, budget):
        started = time.perf_counter()
        try:
            result = dict(infer(prompt, policy, control, budget, rng))
        except Exception as error:
            result = {"status": "error", "text": "", "error_type": type(error).__name__,
                      "error": str(error), "wall_seconds": time.perf_counter() - started}
        tensor = result.pop("_first_token_scores_tensor", None)
        if tensor is not None:
            score_tensors[trace_id] = tensor
        return result
    for a in (0, 1):
        current, refs = [], []
        for label, _ in task.options:
            prompt = condition_prompt(task, label, evidence)
            trace_id = f"A{a}/{label}"
            result = invoke(trace_id, prompt, PhasePolicy(bool(a), False), Control.NORMAL if a else Control.NO_HOOK,
                            config.condition_max_new_tokens)
            current.append(assessment(label, result))
            traces.append({"trace_id": trace_id, "prompt": prompt.text, "generation": result,
                           "assessment": current[-1]})
            refs.append(trace_id)
        conditions[a] = current
        prepared = execution_prompt(task, current, evidence)
        for b in (0, 1):
            arm = f"A{a}B{b}"
            result = invoke(arm, prepared, PhasePolicy(bool(a), bool(b)), Control.NORMAL if b else Control.NO_HOOK,
                            config.execution_max_new_tokens)
            selected = parse_final_answer(result.get("text", ""), [x[0] for x in task.options]) if result.get("status") == "ok" else None
            traces.append({"trace_id": arm, "prompt": prepared.text, "generation": result})
            predictions.append({"sample_id": str(sample_id), "method": arm, "seed": seed,
                "protocol": PROTOCOL, "prediction": selected,
                "status": "ok" if selected is not None else ("parse_error" if result.get("status") == "ok" else "error"),
                "parse_valid": selected is not None, "input_sha256": fingerprint(asdict(task)),
                "prompt_sha256": fingerprint(prepared.text), "output_sha256": fingerprint(result.get("text", "")),
                "config_lock_sha256": fingerprint(lock), "cost_references": refs + [arm],
                "condition_failure_count": sum(item["status"] != "ok" for item in current),
                "no_support_fallback": not bool(prepared.evidence.spans)})
    return {"predictions": predictions, "private_traces": traces,
            "assessments": conditions, "cost_summary": summarize_costs(traces, predictions),
            "numerical_score_tensors_save_separately": score_tensors}


def direct_applicability_set(task, assessments):
    """Cheap direct-mapping comparator; U stays U, never an inferred negative.

    It has no execution call and therefore a lower cost than the factorial arms.
    This is not temperature-calibrated probability fusion and is not described as
    such. An empty set is retained as a prediction, not repaired with an argmax.
    """
    validate_assessments(task, assessments)
    return sorted(item["option"] for item in assessments if item["status"] == "ok" and item["label"] == "T")


def summarize_costs(traces, predictions):
    keys = ("actual_forward_calls", "input_tokens", "output_tokens", "wall_seconds")
    by_id = {item["trace_id"]: item["generation"] for item in traces}
    if len(by_id) != len(traces):
        raise ValueError("Duplicate cost references")
    def total(ids):
        answer = {}
        for key in keys:
            values = [by_id[value].get(key) for value in ids]
            answer[key] = sum(values) if all(x is not None for x in values) else None
        for key in ("peak_allocated_gib", "peak_reserved_gib"):
            values = [by_id[value].get(key) for value in ids]
            answer[key] = max(values) if values and all(x is not None for x in values) else None
        return answer
    return {"actual_physical_run": total(list(by_id)),
            "standalone_per_method": {p["method"]: total(p["cost_references"]) for p in predictions},
            "note": "Cached A costs count once physically and in full for every standalone method; missing cost is not zero"}


def seka_infer(adapter, config=TransferConfig(), *, control_builder=None):
    """Attach the existing loaded official SEKA adapter, no GPU/model creation."""
    import torch
    config.validate()
    tokenizer = adapter.operator.tok
    labels = verify_tfu_tokens(tokenizer)
    def infer(prompt, policy, control, budget, seed):
        calls, handle = [0], None
        device = adapter.operator.device
        cuda = str(device).startswith("cuda")
        start = time.perf_counter()
        result = {"status": "error", "text": ""}
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            prepared = chat_prompt(prompt, tokenizer)
            if control_builder is not None:
                # Build token-count-matched controls after the final chat
                # template; this avoids matching a different tokenization.
                prepared = control_builder(prepared)
            ids, _, _ = adapter._encode(prepared, prepared.evidence.spans)
            model_limit = int(getattr(adapter.operator.model.config, "max_position_embeddings", config.context_limit))
            if ids.numel() + budget > min(config.context_limit, model_limit):
                raise ValueError("Fixed context budget exceeded; no input is truncated")
            def forward_counter(*unused):
                calls[0] += 1
            handle = adapter.operator.model.register_forward_hook(forward_counter)
            result = adapter.generate(prepared, policy, control=control, max_new_tokens=budget,
                                      generation_seed=seed, capture_first_token_scores=True)
            scores = result.pop("first_token_scores")
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite first-token scores")
            logprobs = scores.log_softmax(-1)
            result.update(status="ok", label_logits={k: float(scores[v]) for k, v in labels.items()},
                          label_probabilities={k: float(logprobs[v].exp()) for k, v in labels.items()})
            # The caller may persist this tensor separately for zero/no-hook
            # numerical drift. Never serialize it as a vocabulary-size JSON list.
            result["_first_token_scores_tensor"] = scores
        except Exception as error:
            result.update(status="error", text="", error_type=type(error).__name__, error=str(error))
        finally:
            if handle is not None:
                handle.remove()
            adapter.operator.remove_projection()
        result.update(actual_forward_calls=calls[0], wall_seconds=time.perf_counter() - start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None)
        return result
    return infer
