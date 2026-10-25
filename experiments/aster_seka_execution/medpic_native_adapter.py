"""Schema-only MedPIC native multi-select adapter; no dataset files are opened.

Developed from previously audited public schema and synthetic fixtures only.
The coordinator freezes this protocol before any external test inference.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable

from evaluate_medpic import option_set


NATIVE_PROTOCOL = "medpic_native_v1"
AUGMENTED_PROTOCOL = "medpic_augmented_open_book_v1"
INSTRUCTION = (
    "Answer the clinical multiple-choice question using the patient information. "
    "Select every correct option; more than one option may be correct. "
    "Give a brief explanation, then write one final line in exactly this form: "
    'FINAL_ANSWER: ["A", "C"]. Replace the example letters with your selected option labels. '
    "Do not write anything after that final line."
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class NativeInput:
    patient_vignette: str
    question: str
    options: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PreparedPrompt:
    text: str
    patient_char_span: tuple[int, int]
    allowed_options: tuple[str, ...]
    protocol: str
    additional_context_sha256: str | None = None
    additional_context_origin: str | None = None


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    process_peak_allocated_gib: float | None = None
    generation_token_ids_sha256: str | None = None


@dataclass(frozen=True)
class DecodeConfig:
    max_new_tokens: int = 512
    context_limit: int = 32768
    do_sample: bool = False
    temperature: float = .7
    top_p: float = .8
    top_k: int = 20
    enable_thinking: bool = False


def native_input(row: dict) -> NativeInput:
    """Only these three fields reach the method; labels/taxonomies are ignored."""
    patient, question, options = row["patient_vignette"], row["question"], row["options"]
    if not isinstance(patient, str) or not isinstance(question, str) or not patient.strip() or not question.strip():
        raise ValueError("Native input requires nonempty patient_vignette and question strings")
    if (not isinstance(options, dict) or not options
            or any(not isinstance(key, str) or not re.fullmatch(r"[A-Z]", key)
                   or not isinstance(value, str) for key, value in options.items())):
        raise ValueError("Options must be the official letter-to-text mapping")
    # Preserve the published option order; do not reorder according to labels.
    return NativeInput(patient, question, tuple(options.items()))


def render_native(task: NativeInput) -> PreparedPrompt:
    prefix = INSTRUCTION + "\n\nPatient:\n"
    start = len(prefix)
    text = prefix + task.patient_vignette + "\n\nQuestion:\n" + task.question + "\n\nOptions:\n"
    text += "\n".join(f"{label}. {option}" for label, option in task.options)
    return PreparedPrompt(text, (start, start + len(task.patient_vignette)),
                          tuple(label for label, _ in task.options), NATIVE_PROTOCOL)


def render_augmented_open_book(task: NativeInput, additional_context: str, origin: str) -> PreparedPrompt:
    """Separate track; generated rules are never labeled official guidelines."""
    if origin not in {"model_derived_rules", "retrieved_document"} or not additional_context.strip():
        raise ValueError("Additional context needs an explicit supported origin")
    native = render_native(task)
    description = ("Model-derived rules (not an official guideline)" if origin == "model_derived_rules"
                   else "Retrieved document (source authority is not verified by this adapter)")
    text = native.text + "\n\nAdditional context — " + description + ":\n" + additional_context
    return PreparedPrompt(text, native.patient_char_span, native.allowed_options, AUGMENTED_PROTOCOL,
                          fingerprint(additional_context), origin)


def parse_final_answer(text: str, allowed_options) -> list[str] | None:
    """Read one designated final field only; never infer choices from rationale."""
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = [line for line in lines if line.startswith("FINAL_ANSWER:")]
    if not lines or len(candidates) != 1 or lines[-1] != candidates[0]:
        return None
    encoded = candidates[0].split(":", 1)[1].strip()
    if not encoded.startswith("["):
        return None
    selected = option_set(encoded, allowed_options)
    return sorted(selected) if selected is not None else None


def require_frozen_native(lock, method, seed):
    if lock.get("frozen") is not True or not lock.get("frozen_at"):
        raise ValueError("Freeze the external method/configuration before inference")
    if lock.get("protocol") != NATIVE_PROTOCOL:
        raise ValueError("Native and augmented-open-book protocols require separate locks and reports")
    if sorted(lock.get("seeds", [])) != [0, 1, 2] or seed not in lock["seeds"] or method not in lock.get("methods", []):
        raise ValueError("Method/seed is outside the frozen confirmation declaration")
    if lock.get("prompt_template_sha256") != fingerprint(INSTRUCTION):
        raise ValueError("Native instruction template differs from the frozen protocol")


def case_rng_seed(seed: int, instance_id) -> int:
    # Same case/seed uses the same random stream across methods; no outcome-based seeds.
    return int(fingerprint(["MedPIC-case-seed-v1", seed, str(instance_id)])[:8], 16)


def predict_native_row(row, generate: Callable[[PreparedPrompt, int], Generation], *, method, seed, lock):
    """Return (public prediction, server-private trace); never pass the raw row."""
    require_frozen_native(lock, method, seed)
    if lock.get("decode_config_sha256") is not None and lock["decode_config_sha256"] != getattr(generate, "decode_config_sha256", None):
        raise ValueError("Actual generation configuration differs from the frozen configuration")
    sample_id = str(row["instance_id"])
    public = {"sample_id": sample_id, "method": method, "seed": seed, "protocol": NATIVE_PROTOCOL,
              "prediction": None, "status": "error", "parse_valid": False,
              "case_rng_seed": case_rng_seed(seed, sample_id), "config_lock_sha256": fingerprint(lock),
              "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "answer_parser_sha256": hashlib.sha256(Path(__file__).with_name("evaluate_medpic.py").read_bytes()).hexdigest(),
              "decode_config_sha256": getattr(generate, "decode_config_sha256", None)}
    private = {"sample_id": sample_id, "method": method, "seed": seed}
    started = time.monotonic()
    try:
        task = native_input(row)
        prepared = render_native(task)
        # Neither answer, family, taxonomy, pair identities nor source row is passed.
        generated = generate(prepared, public["case_rng_seed"])
        selected = parse_final_answer(generated.text, prepared.allowed_options)
        public.update(prediction=selected, status="ok" if selected is not None else "parse_error",
                      parse_valid=selected is not None, prompt_sha256=fingerprint(prepared.text),
                      output_sha256=fingerprint(generated.text),
                      patient_span_sha256=fingerprint(prepared.patient_char_span),
                      costs={key: value for key, value in asdict(generated).items() if key != "text"})
        private.update(prompt=prepared.text, patient_char_span=prepared.patient_char_span,
                       raw_output=generated.text)
    except Exception as exc:
        public.update(error_type=type(exc).__name__, error_sha256=fingerprint(str(exc)))
        private["error"] = {"type": type(exc).__name__, "message": str(exc)}
    public["adapter_elapsed_seconds"] = time.monotonic() - started
    return public, private


def prepare_chat_tokens(task: PreparedPrompt, tokenizer):
    """Return identical native text tokens plus a mask for patient tokens only."""
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": task.text}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if rendered.count(task.text) != 1:
        raise ValueError("Chat template did not preserve one exact task text span")
    offset = rendered.index(task.text)
    start, end = (offset + value for value in task.patient_char_span)
    encoded = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False,
                        truncation=False)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    patient_mask = [right > left and right > start and left < end for left, right in offsets]
    return encoded, patient_mask


def hf_native_generator(model, tokenizer, config: DecodeConfig):
    """Minimal plain-text baseline generator; model ownership/GPU locks stay outside.

    Attention methods can consume the same PreparedPrompt and patient token mask
    through their already-audited hooks, preserving prompt and budget equality.
    """
    if config.max_new_tokens <= 0 or config.context_limit <= 0 or config.enable_thinking:
        raise ValueError("The frozen minimal adapter uses a positive budget and disabled thinking")
    def generate(task, seed):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        encoded, _patient_mask = prepare_chat_tokens(task, tokenizer)
        length = encoded["input_ids"].shape[1]
        limit = min(config.context_limit, int(getattr(model.config, "max_position_embeddings", config.context_limit)))
        if length + config.max_new_tokens > limit:
            raise ValueError("Frozen context budget exceeded; input is not truncated")
        device = next(model.parameters()).device
        encoded = encoded.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.monotonic()
        kwargs = {"max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample,
                  "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                  "return_dict_in_generate": True}
        if config.do_sample:
            kwargs.update(temperature=config.temperature, top_p=config.top_p, top_k=config.top_k)
        model.eval()
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        ids = output.sequences[0, length:].tolist()
        return Generation(tokenizer.decode(ids, skip_special_tokens=True), length, len(ids), elapsed,
                          torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                          fingerprint(ids))
    generate.decode_config_sha256 = fingerprint(asdict(config))
    return generate
