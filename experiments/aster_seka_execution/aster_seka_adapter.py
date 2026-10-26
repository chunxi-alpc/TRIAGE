"""Small two-stage adapter around the official SEKA API.

This module does not extract clinical rules, predict applicability, or score a
benchmark. Callers must supply automatically predicted spans and a frozen public
input. No gold answer, path, or applicability label is accepted by the schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch


def _keys(value: Mapping[str, Any], required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"Missing input fields: {sorted(missing)}; forbidden/unknown fields: {sorted(unknown)}")


@dataclass(frozen=True)
class Span:
    """Character range in the exact unmodified source field, not in a token list."""
    source: str
    start: int
    end: int
    rule_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Span":
        _keys(value, {"source", "start", "end"}, {"rule_id"})
        if type(value["start"]) is not int or type(value["end"]) is not int:
            raise TypeError("Span boundaries must be integers")
        return cls(**value)


@dataclass(frozen=True)
class PredictedSpans:
    """Provenance declaration; the selector itself must be audited separately.

    Control ranges are optional until that particular control is requested.
    misaligned_rule_spans must come from an explicitly permuted rule mapping.
    """
    selector_fingerprint: str
    spans: tuple[Span, ...]
    wrong_spans: tuple[Span, ...] = ()
    misaligned_rule_spans: tuple[Span, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PredictedSpans":
        _keys(value, {"selector_fingerprint", "spans"}, {"wrong_spans", "misaligned_rule_spans"})
        if not isinstance(value["selector_fingerprint"], str) or not value["selector_fingerprint"]:
            raise ValueError("An automatic selector fingerprint is required")
        return cls(value["selector_fingerprint"], **{
            k: tuple(Span.from_mapping(s) for s in value.get(k, ()))
            for k in ("spans", "wrong_spans", "misaligned_rule_spans")
        })


@dataclass(frozen=True)
class ConditionRequest:
    patient_history: str
    conditions: tuple[tuple[str, str], ...]
    evidence: PredictedSpans

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConditionRequest":
        # No action choices, action mapping, complete guideline, or gold fields.
        _keys(value, {"patient_history", "conditions", "evidence"})
        conditions = []
        for item in value["conditions"]:
            _keys(item, {"rule_id", "condition"})
            conditions.append((item["rule_id"], item["condition"]))
        if len({r for r, _ in conditions}) != len(conditions):
            raise ValueError("Duplicate condition rule IDs")
        return cls(value["patient_history"], tuple(conditions), PredictedSpans.from_mapping(value["evidence"]))


@dataclass(frozen=True)
class ExecutionRequest:
    patient_history: str
    question: str
    guideline: str
    predicted_tfu: tuple[tuple[str, float, float, float], ...]
    output_kind: str
    public_options: tuple[tuple[str, str], ...]
    evidence: PredictedSpans

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionRequest":
        _keys(value, {"patient_history", "question", "guideline", "predicted_tfu", "output_kind", "evidence"}, {"public_options"})
        if value["output_kind"] not in {"single_choice", "multi_select", "path"}:
            raise ValueError("Unknown native output contract")
        predictions = []
        for item in value["predicted_tfu"]:
            _keys(item, {"rule_id", "true_score", "false_score", "unknown_score"})
            scores = tuple(float(item[k]) for k in ("true_score", "false_score", "unknown_score"))
            if not all(0 <= v <= 1 for v in scores):
                raise ValueError("T/F/U scores must be finite and between zero and one")
            predictions.append((item["rule_id"], *scores))
        if len({r[0] for r in predictions}) != len(predictions):
            raise ValueError("Duplicate predicted rule IDs")
        options = []
        for item in value.get("public_options", ()):
            _keys(item, {"label", "text"})
            options.append((item["label"], item["text"]))
        if len({k for k, _ in options}) != len(options):
            raise ValueError("Duplicate option labels")
        if value["output_kind"] != "path" and not options:
            raise ValueError("Choice tasks require the original public options")
        return cls(value["patient_history"], value["question"], value["guideline"],
                   tuple(predictions), value["output_kind"], tuple(options),
                   PredictedSpans.from_mapping(value["evidence"]))


@dataclass(frozen=True)
class PhasePolicy:
    condition_on: bool
    execution_on: bool

    @classmethod
    def factorial(cls) -> tuple["PhasePolicy", ...]:
        return tuple(cls(a, b) for a in (False, True) for b in (False, True))


class Control(str, Enum):
    NORMAL = "normal"
    NO_HOOK = "no_hook"
    MATCHED_ZERO = "matched_zero"
    WRONG_SPAN = "wrong_span"
    MISALIGNED_RULE = "misaligned_rule"
    REVERSE = "reverse"


@dataclass(frozen=True)
class PreparedPrompt:
    text: str
    fields: Mapping[str, tuple[int, str]]
    evidence: PredictedSpans
    stage: str
    output_kind: str

    def absolute_spans(self, spans: Sequence[Span]) -> list[tuple[int, int]]:
        absolute = []
        for span in spans:
            if span.source not in self.fields:
                raise ValueError(f"Span source is not visible in this stage: {span.source}")
            offset, source = self.fields[span.source]
            if not (0 <= span.start < span.end <= len(source)):
                raise ValueError("Span outside source text")
            absolute.append((offset + span.start, offset + span.end))
        return absolute


def _assemble(header: str, items: Sequence[tuple[str, str]], tail: str, evidence: PredictedSpans,
              stage: str, output_kind: str) -> PreparedPrompt:
    text, fields = header, {}
    for key, content in items:
        if key in fields or not isinstance(content, str):
            raise ValueError("Duplicate or invalid public source field")
        text += f"\n[{key}]\n"
        fields[key] = (len(text), content)
        text += content
    return PreparedPrompt(text + tail, fields, evidence, stage, output_kind)


def prepare_condition(request: ConditionRequest) -> PreparedPrompt:
    # Treatment history remains visible; callers must remove proposed actions
    # from the condition fields. This serializer does not prove semantic blindness.
    return _assemble(
        "Assess each rule condition from the patient history. Use true, false, or unknown. "
        "Do not infer missing facts. Return a JSON object keyed by rule ID with T/F/U "
        "scores and supporting or contradicting quotations from the visible text.",
        [("patient_history", request.patient_history)] +
        [(f"condition:{rid}", text) for rid, text in request.conditions],
        "\nAssessment:\n", request.evidence, "condition", "tfu")


def prepare_execution(request: ExecutionRequest) -> PreparedPrompt:
    suffix = {
        "single_choice": "Return the label of the selected option.",
        "multi_select": "Return a JSON array containing all and only the selected option labels.",
        "path": "Generate the decision path using the guideline's native nodes and give the final outcome.",
    }[request.output_kind]
    tfu = [{"rule_id": r, "true_score": t, "false_score": f, "unknown_score": u}
           for r, t, f, u in request.predicted_tfu]
    extra = "\nPredicted applicability (not verified facts):\n" + json.dumps(tfu, ensure_ascii=False, sort_keys=True)
    if request.public_options:
        extra += "\nOptions:\n" + json.dumps(dict(request.public_options), ensure_ascii=False)
    extra += "\n" + suffix + "\nAnswer:\n"
    return _assemble("Use the complete guideline and the supplied predicted applicability. "
                     "Multiple rules may apply together. Unknown is not false.",
                     [("patient_history", request.patient_history), ("question", request.question),
                      ("guideline", request.guideline)], extra, request.evidence, "execution", request.output_kind)


def projection_manifest(operator: Any, kind: str, paths: Sequence[str]) -> list[dict[str, Any]]:
    """Validate the actual saved layer order, KV heads, dimension, and numbers."""
    config = operator.model.config
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("This first adapter is validated structurally only for Qwen3")
    n_q, n_kv, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    if n_q % n_kv != 0:
        raise ValueError("Invalid GQA mapping")
    if kind == "seka":
        spec = operator.layers
        if spec.startswith("last"):
            selected = list(range(config.num_hidden_layers - int(spec[4:]), config.num_hidden_layers))
        elif spec == "all":
            selected = list(range(config.num_hidden_layers))
        else:
            selected = [int(x) for x in spec.split(",")]
    else:
        selected = list(operator.sel_layers)
    manifest = []
    if kind == "adaseka" and len(paths) != len(operator.task_projector.expert_names):
        raise ValueError("Provide every loaded expert path in expert_names order")
    for index, path in enumerate(paths):
        p = Path(path)
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or obj.get("layers") != selected:
            raise ValueError("Saved projection layer order must exactly match inference layers")
        tensor = obj.get("proj") if kind == "seka" else obj.get("U_matrices")
        if tensor is None or tuple(tensor.shape) != (len(selected), n_kv, head_dim, head_dim):
            raise ValueError("Projection shape does not match the checkpoint's KV heads")
        disabled_heads = 0
        if kind == "seka" and not torch.isfinite(tensor).all():
            raise ValueError("Non-finite projection")
        if kind == "adaseka":
            singular = obj.get("singular_values")
            if singular is None or tuple(singular.shape) != (len(selected), n_kv, head_dim):
                raise ValueError("Invalid expert singular values")
            # The official adaptive builder uses complete NaN blocks for filtered
            # heads. Partial NaNs or infinities do not follow that convention.
            valid = torch.isfinite(tensor).all(dim=-1).all(dim=-1) & torch.isfinite(singular).all(dim=-1)
            disabled = torch.isnan(tensor).all(dim=-1).all(dim=-1) & torch.isnan(singular).all(dim=-1)
            if not (valid | disabled).all():
                raise ValueError("Malformed adaptive head: expected finite values or complete NaN sentinel")
            disabled_heads = int(disabled.sum())
            expert_name = operator.task_projector.expert_names[index]
            actual_u = operator.task_projector.expert_U_matrices[expert_name].detach().cpu()
            actual_s = operator.task_projector.expert_singular_values[expert_name].detach().cpu()
            if not torch.allclose(tensor.to(actual_u.dtype), actual_u, rtol=0, atol=0, equal_nan=True) or not torch.allclose(singular.to(actual_s.dtype), actual_s, rtol=0, atol=0, equal_nan=True):
                raise ValueError("SVD manifest does not match the actual loaded expert")
        manifest.append({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "layers": selected, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                         "disabled_head_count": disabled_heads})
    return manifest


class StageSEKAAdapter:
    """Reuses one loaded upstream SEKALLM/AdaptiveSEKALLM, one example per call."""

    def __init__(self, operator: Any, *, kind: str, source_commit: str,
                 projection_paths: Sequence[str]):
        if kind not in {"seka", "adaseka"}:
            raise ValueError("kind must identify the actual official operator")
        if len(source_commit) != 40 or not projection_paths:
            raise ValueError("Pinned source commit and actual projection files are required")
        if kind == "seka":
            expected = [operator.pos_pt] + ([operator.neg_pt] if operator.neg_pt else [])
            if [str(Path(p).resolve()) for p in expected] != [str(Path(p).resolve()) for p in projection_paths]:
                raise ValueError("Manifest files do not match the loaded SEKA operator configuration")
        self.operator, self.kind, self.source_commit = operator, kind, source_commit
        self.projections = projection_manifest(operator, kind, projection_paths)
        self.dtype = next(operator.model.parameters()).dtype
        if self.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("Unsupported model dtype")
        if not operator.tok.is_fast:
            raise ValueError("A fast tokenizer with original character offsets is required")

    def _encode(self, prompt: PreparedPrompt, spans: Sequence[Span]):
        # No marker tokens are inserted. All configurations share exactly these IDs.
        encoded = self.operator.tok(prompt.text, return_tensors="pt", return_offsets_mapping=True,
                                    add_special_tokens=True, truncation=False, padding=False)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        ranges = prompt.absolute_spans(spans)
        mask = torch.tensor([[any(ce > a and cs < b and ce > cs for a, b in ranges)
                              for cs, ce in offsets]], dtype=torch.bool)
        attention = encoded["attention_mask"]
        mask &= attention.bool()
        if spans and not mask.any():
            raise ValueError("Nonempty predicted spans produced no visible tokens")
        return encoded["input_ids"], attention, mask

    @torch.inference_mode()
    def generate(self, prompt: PreparedPrompt, policy: PhasePolicy, *, max_new_tokens: int,
                 control: Control = Control.NORMAL, generation_seed: int = 0,
                 capture_first_token_scores: bool = False) -> dict[str, Any]:
        if prompt.stage not in {"condition", "execution"} or max_new_tokens <= 0:
            raise ValueError("Invalid stage or generation budget")
        enabled = policy.condition_on if prompt.stage == "condition" else policy.execution_on
        control = Control(control)
        if not enabled and control not in {Control.NORMAL, Control.NO_HOOK}:
            raise ValueError("An intervention control needs this stage switched on")
        main = prompt.evidence.spans
        selected = main
        if control == Control.WRONG_SPAN:
            selected = prompt.evidence.wrong_spans
        elif control == Control.MISALIGNED_RULE:
            selected = prompt.evidence.misaligned_rule_spans
            if not selected or any(s.rule_id is None for s in (*main, *selected)):
                raise ValueError("Misalignment requires declared original and permuted rule IDs")
            association = lambda items: {(s.rule_id, s.source, s.start, s.end) for s in items}
            if association(main) == association(selected):
                raise ValueError("Rule-to-span associations were not misaligned")
        ids, attention, main_mask = self._encode(prompt, main)
        _, _, mask = self._encode(prompt, selected)
        if control in {Control.WRONG_SPAN, Control.MISALIGNED_RULE}:
            if not selected or torch.equal(mask, main_mask):
                raise ValueError("Negative control must change the selected tokens")
            if int(mask.sum()) != int(main_mask.sum()):
                raise ValueError("Control must preserve highlighted-token count")
            if control == Control.WRONG_SPAN and (mask & main_mask).any():
                raise ValueError("Wrong-span control must be disjoint")
        # No support has an explicit original-context fallback, with no fabricated span.
        steer = enabled and control != Control.NO_HOOK and bool(mask.any())
        gain_names = ("amplify_pos", "amplify_neg") if self.kind == "seka" else ("amplify_factor",)
        original_gains = {name: getattr(self.operator, name) for name in gain_names}
        effective_gains = dict(original_gains)
        if control == Control.MATCHED_ZERO:
            effective_gains = {name: 0.0 for name in gain_names}
        elif control == Control.REVERSE:
            effective_gains = {name: -value for name, value in original_gains.items()}
        torch.manual_seed(generation_seed)
        device = self.operator.device
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            for name, value in effective_gains.items():
                setattr(self.operator, name, value)
            output = self.operator.generate(ids=ids.to(device), steer=steer,
                steer_mask=mask.to(device), attention_mask=attention.to(device), return_raw=True,
                max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_k=None, top_p=None,
                output_scores=capture_first_token_scores, return_dict_in_generate=capture_first_token_scores,
                pad_token_id=self.operator.tok.eos_token_id)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
        finally:
            self.operator.remove_projection()
            for name, value in original_gains.items():
                setattr(self.operator, name, value)
        elapsed = time.perf_counter() - started
        first_token_scores = None
        if capture_first_token_scores:
            if not hasattr(output, "scores") or not output.scores:
                raise ValueError("Requested generation scores are missing")
            first_token_scores = output.scores[0][0].detach().float().cpu()
        if hasattr(output, "sequences"):
            output = output.sequences
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError("Only one example per call is supported")
        continuation = output[0, ids.shape[-1]:]
        result = {"text": self.operator.tok.decode(continuation, skip_special_tokens=True),
                "stage": prompt.stage, "output_kind": prompt.output_kind, "control": control.value,
                "policy": {"condition_on": policy.condition_on, "execution_on": policy.execution_on},
                "prompt_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
                "input_ids_sha256": hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest(),
                "span_mask_sha256": hashlib.sha256(json.dumps(mask.tolist()).encode()).hexdigest(),
                "input_tokens": ids.numel(), "output_tokens": continuation.numel(),
                "highlighted_tokens": int(mask.sum()), "steer": steer,
                "no_support_fallback": not bool(mask.any()), "gains": effective_gains,
                "source_commit": self.source_commit, "projection_manifest": self.projections,
                "model_dtype": str(self.dtype), "generation_seed": generation_seed,
                "greedy_decoding": True, "selector_fingerprint": prompt.evidence.selector_fingerprint,
                "elapsed_seconds": elapsed}
        if first_token_scores is not None:
            # Save this tensor separately from the JSON ledger for matched-zero
            # numerical analysis; output scores include generation processors.
            result["first_token_scores"] = first_token_scores
            result["first_token_scores_semantics"] = "generation scores after any configured logits processors"
        return result
