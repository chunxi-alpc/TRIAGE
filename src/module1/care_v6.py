"""CARE module-one V6: TrialMatchAI condition assessment + BioLinkBERT-MedNLI evidence.

This module never imports or calls the legacy Qwen backend. Guidelines must be
prepared as structured rules before inference; free-text rule parsing is refused.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

TRIALMATCH_BASE = "microsoft/phi-4"
TRIALMATCH_ADAPTER = "majdabd33/trialmatchai-phi4-reasoning-lora"
BIOLINK_MEDNLI = "cnut1648/biolinkbert-mednli"

TRUE, FALSE, UNKNOWN = "T", "F", "U"


def tri_and(values: Iterable[str]) -> str:
    values = list(values)
    if FALSE in values:
        return FALSE
    if UNKNOWN in values:
        return UNKNOWN
    return TRUE


def tri_or(values: Iterable[str]) -> str:
    values = list(values)
    if TRUE in values:
        return TRUE
    if UNKNOWN in values:
        return UNKNOWN
    return FALSE


def tri_not(value: str) -> str:
    return {TRUE: FALSE, FALSE: TRUE}.get(value, UNKNOWN)


def eval_logic(expr: Any, states: dict[str, str]) -> str:
    """Evaluate a rule's AND/OR/NOT tree with strong Kleene three-valued logic."""
    if isinstance(expr, str):
        return states.get(expr, UNKNOWN)
    if not isinstance(expr, dict):
        return UNKNOWN
    op = str(expr.get("op", "AND")).upper()
    args = [eval_logic(arg, states) for arg in expr.get("args", [])]
    if op == "AND":
        return tri_and(args)
    if op == "OR":
        return tri_or(args)
    if op == "NOT" and len(args) == 1:
        return tri_not(args[0])
    return UNKNOWN


def sentences_with_offsets(text: str) -> list[dict[str, Any]]:
    """Split without losing exact [start,end) Python character offsets."""
    result: list[dict[str, Any]] = []
    cursor = 0
    for boundary in re.finditer(r"(?<=[.!?])(?=\s)|(?<=[。！？])|\n+", text):
        start, end = cursor, boundary.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            result.append({"sentence_id": len(result) + 1, "start": start,
                           "end": end, "text": text[start:end]})
        cursor = boundary.end()
    start, end = cursor, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        result.append({"sentence_id": len(result) + 1, "start": start,
                       "end": end, "text": text[start:end]})
    return result


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "or", "for", "with", "from", "that", "this", "was", "were",
            "are", "is", "has", "have", "had", "patient", "condition", "present", "absent"}
    return {x for x in re.findall(r"[a-z0-9]+", text.lower()) if len(x) > 1 and x not in stop}


def _active_spans(condition: dict[str, Any], spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Honor explicit withdrawal/correction boundaries; keep unresolved co-temporal conflicts."""
    aliases = FIELD_ALIASES.get(condition.get("field"), ())
    boundaries = []
    for span in spans:
        text = span["text"]
        corrected = re.search(r"(?i)(earlier statements.*withdrawn|copied in error.*withdrawn|withdrawn value cannot be used)", text)
        if corrected and (not aliases or any(alias.casefold() in text.casefold() for alias in aliases)):
            boundaries.append(span["end"])
    if not boundaries:
        return spans
    boundary = min(boundaries)
    return [span for span in spans if span["start"] >= boundary and not re.search(
        r"(?i)(earlier statements.*withdrawn|copied in error.*withdrawn|withdrawn value cannot be used)", span["text"])]


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "ato_prior": ("arsenic trioxide", "ATO"), "relapse_months": ("relapse", "months after", "interval"),
    "flt3": ("FLT3",), "pt": ("pT", "pathological T stage", "pathologic T stage"),
    "pn": ("pN", "pathological N stage", "pathologic N stage"),
    "ct": ("cT", "clinical T stage"), "cn": ("cN", "clinical N stage"),
    "preop_chemo": ("preoperative chemotherapy", "neoadjuvant chemotherapy", "before surgery"),
    "sln_count": ("sentinel node", "sentinel lymph node"), "wbrt": ("whole breast radiotherapy", "whole breast radiation"),
    "ax_nodes": ("axillary node", "axillary lymph node"), "tumor_cm": ("tumor diameter", "tumour diameter", "invasive tumor", "invasive tumour"),
    "margin_mm": ("resection margin", "surgical margin", "closest margin"), "ypn": ("ypN", "postoperative N stage", "post-neoadjuvant N stage"),
    "distant": ("distant metastasis", "distant metastatic"), "dmmr": ("mismatch repair", "microsatellite"),
    "pole": ("POLE", "POLD1"), "tmb": ("tumor mutational burden", "tumour mutational burden", "TMB"),
    "csf": ("CSF cytology", "cerebrospinal fluid cytology"), "margin": ("lobectomy margin", "thyroid margin"),
    "contra": ("contralateral thyroid lesion",), "susp_nodes": ("suspicious cervical lymph node", "suspicious cervical nodes"),
    "age": ("patient age", "aged", "years old", "months old", "years-old", "months-old", "year-old", "month-old", "age:", "age at", "age is", "age was"), "ret634": ("RET codon 634", "RET 634"),
    "v804m": ("RET V804M", "V804M"), "strong_family": ("strong family history",),
    "gross_margin": ("gross positive margin",), "gross_ete": ("gross extrathyroidal extension",),
    "thyroid_nodes": ("pathology-confirmed cervical nodal", "confirmed cervical nodal metastasis"),
    "confirmed_contra": ("confirmed contralateral thyroid carcinoma",), "vascular": ("vascular invasion",),
    "resectable": ("resectable", "surgically resectable"), "iodine_avid": ("radioiodine-avid", "radioiodine uptake", "iodine uptake"),
    "progressive": ("progressive", "progression"), "symptoms": ("cancer-related symptoms", "symptomatic"),
    "indolent": ("indolent systemic metastatic",), "brain_mets": ("brain metastasis", "brain metastases"),
}


def candidate_sentences(condition: dict[str, Any], spans: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    text = condition.get("text", "")
    query_tokens = _tokens(text)
    field = condition.get("field")
    aliases = FIELD_ALIASES.get(field, ())
    ranked = []
    for span in _active_spans(condition, spans):
        sentence = span["text"]
        sentence_tokens = _tokens(sentence)
        overlap = len(query_tokens & sentence_tokens)
        alias_hit = any(alias.casefold() in sentence.casefold() for alias in aliases)
        # Avoid sending unrelated sentences to NLI when neither concept nor field is present.
        if overlap == 0 and not alias_hit:
            continue
        ranked.append((2 * int(alias_hit) + overlap, -span["sentence_id"], span))
    ranked.sort(reverse=True, key=lambda row: (row[0], row[1]))
    return [row[2] for row in ranked[:limit]]


def map_trialmatch_label(raw: str, criterion_kind: str = "inclusion") -> str:
    """Map only an exact official verdict; prose and reasoning are never labels.

    A negative exclusion verdict does not prove that the positive atom is false.
    """
    text = re.sub(r"\s+", " ", str(raw or "")).strip().casefold()
    if criterion_kind == "exclusion":
        return {"violated": TRUE, "not violated": UNKNOWN,
                "unclear": UNKNOWN, "irrelevant": UNKNOWN}.get(text, UNKNOWN)
    return {"met": TRUE, "not met": FALSE, "unclear": UNKNOWN,
            "irrelevant": UNKNOWN}.get(text, UNKNOWN)


def _rule_list(guideline: Any) -> list[dict[str, Any]]:
    if isinstance(guideline, dict):
        if "rules" in guideline:
            rows = guideline["rules"]
        elif "conditions" in guideline:
            rows = [guideline]
        else:
            raise ValueError("V6 requires offline-prepared structured rules; no model-based rule parsing is available")
    elif isinstance(guideline, (list, tuple)):
        rows = list(guideline)
    else:
        raise ValueError("V6 requires offline-prepared structured rules; free-text guideline parsing is disabled")
    if not isinstance(rows, (list, tuple)):
        raise TypeError("guideline.rules must be a list")
    normalized = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("conditions"), list):
            raise ValueError("each prepared rule must include a conditions list")
        item = dict(row)
        item["rule_id"] = str(item.get("rule_id") or f"R{i+1}")
        if not isinstance(item.get("text", ""), str):
            raise ValueError(f"{item['rule_id']} text must be a string")
        if not item["conditions"] and item.get("explicitly_unconditional") is not True:
            raise ValueError(f"{item['rule_id']} has no conditions but is not explicitly marked unconditional")
        item.setdefault("logic", {"op": "AND", "args": [c.get("condition_id", f"c{j+1}") for j, c in enumerate(item["conditions"])]})
        item.setdefault("exceptions", [])
        item.setdefault("exception_logic", {"op": "OR", "args": [c.get("condition_id", f"e{j+1}") for j, c in enumerate(item["exceptions"])]})
        if item.get("action") is None:
            item["action"] = ""
        if item.get("source_span") is None:
            item["source_span"] = {"start": 0, "end": len(item.get("text", "")),
                                   "unit": "character", "source": "caller_supplied_rule_text"}
        for prefix, group in (("c", item["conditions"]), ("e", item["exceptions"])):
            for j, condition in enumerate(group):
                if condition.get("negated"):
                    raise ValueError("Use an explicit logic NOT node instead of negated=true")
                condition.setdefault("condition_id", f"{prefix}{j+1}")
                if not isinstance(condition.get("text"), str) or not condition["text"].strip():
                    raise ValueError(f"{item['rule_id']}:{condition['condition_id']} needs condition text")
        normalized.append(item)
    return normalized


def _patient_text(patient: Any) -> str:
    if isinstance(patient, str):
        return patient
    if isinstance(patient, (list, tuple)) and all(isinstance(x, str) for x in patient):
        return "\n".join(patient)
    if isinstance(patient, dict):
        for key in ("patient", "text", "patient_note", "note", "case", "description", "sentences"):
            if key in patient:
                return _patient_text(patient[key])
    raise TypeError("patient must be text, a sentence list, or a note dictionary")


def _patient_spans(patient: Any, text: str) -> list[dict[str, Any]]:
    """Preserve provided sentence IDs; otherwise split text while retaining offsets."""
    rows = patient
    if isinstance(rows, dict):
        rows = rows.get("sentences")
    if isinstance(rows, (list, tuple)) and all(isinstance(x, str) for x in rows):
        spans, cursor = [], 0
        for index, sentence in enumerate(rows, 1):
            start = cursor
            end = start + len(sentence)
            spans.append({"sentence_id": index, "start": start, "end": end, "text": sentence})
            cursor = end + 1
        return spans
    return sentences_with_offsets(text)


def _numeric_observation(condition: dict[str, Any], spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compare all active values/range endpoints after explicit unit conversion."""
    spec = condition.get("numeric")
    if not isinstance(spec, dict):
        return None
    field = condition.get("field", spec.get("field", ""))
    aliases = FIELD_ALIASES.get(field, ())
    op = str(spec.get("operator", ""))
    try:
        threshold = float(spec["value"])
    except (KeyError, TypeError, ValueError):
        return {"status": UNKNOWN, "reason": "invalid_numeric_spec", "evidence": []}
    if not math.isfinite(threshold):
        return {"status": UNKNOWN, "reason": "invalid_numeric_spec", "evidence": []}
    number = r"\d+(?:\.\d+)?"
    measurement_units = {
        "age": r"years?|months?|yrs?|mos?|y/o|yo",
        "relapse_months": r"years?|months?|yrs?|mos?",
        "tumor_cm": r"cm|mm|inches|inch|in",
        "margin_mm": r"mm|cm|inches|inch|in",
    }
    canonical = {"age": "year", "relapse_months": "month", "tumor_cm": "cm", "margin_mm": "mm"}
    def unit_name(unit):
        value = re.sub(r"[- ]old$", "", (unit or "").casefold()).strip()
        return {"years": "year", "yr": "year", "yrs": "year", "y/o": "year", "yo": "year",
                "months": "month", "mo": "month", "mos": "month", "inches": "in", "inch": "in"}.get(value, value)
    def convert(value, source):
        source = unit_name(source)
        target = unit_name(spec.get("unit") or canonical[field])
        factors = {"year": 12.0, "month": 1.0} if field in {"age", "relapse_months"} else {"cm": 10.0, "mm": 1.0, "in": 25.4}
        if source not in factors or target not in factors:
            return None
        return float(value) * factors[source] / factors[target]
    patterns = {
        "pt": r"(?i)\bpT\s*([1-4])(?:\s*[-–—/]\s*(?:pT\s*)?([1-4]))?\b",
        "pn": r"(?i)\bpN\s*([0-3])(?:\s*[-–—/]\s*(?:pN\s*)?([0-3]))?\b",
        "ct": r"(?i)\bcT\s*([1-4])(?:\s*[-–—/]\s*(?:cT\s*)?([1-4]))?\b",
        "cn": r"(?i)(?:\bcN\s*([0-3])(?:\s*[-–—/]\s*(?:cN\s*)?([0-3]))?\b|\bcN\s*\+)",
        "ypn": r"(?i)\bypN\s*([0-3])(?:\s*[-–—/]\s*(?:ypN\s*)?([0-3]))?\b",
        "tmb": r"(?i)(\d+(?:\.\d+)?)\s*(?:mutations?|mut)\s*/\s*Mb\b",
        "sln_count": r"(?i)(\d+)\s+positive\s+(?:sentinel|sentinal)\s+(?:lymph\s+)?nodes?",
        "ax_nodes": r"(?i)(\d+)\s+positive\s+axillary\s+(?:lymph\s+)?nodes?",
    }
    age_single = r"(?i)(?:(?:aged?\s*(?:is|at)?\s*|patient\s+age\s+(?:is|of|was)?\s*|age\s+(?:at\s+\w+\s+)?(?:is|of|was)\s*)(\d+(?:\.\d+)?)\s*(years?|months?|yrs?|mos?|y/o|yo)?|(\d+(?:\.\d+)?)\s*(years?[- ]old|months?[- ]old|y/o|yo))"
    values: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for span in spans:
        sentence = span["text"]
        if aliases and not any(a.casefold() in sentence.casefold() for a in aliases):
            continue
        if field in measurement_units:
            units = measurement_units[field]
            unit_pattern = r"(?:" + units + r")\b"
            range_pattern = (r"(?P<low>" + number + r")\s*(?P<u1>" + unit_pattern + r")?\s*"
                             r"(?:[-–—]|\bto\b|\bthrough\b)\s*(?P<high>" + number + r")\s*"
                             r"(?P<u2>" + unit_pattern + r")?")
            consumed = []
            for match in re.finditer(range_pattern, sentence, re.I):
                u1, u2 = match.group("u1"), match.group("u2")
                if not u1 and not u2:
                    if field != "age" or not re.search(r"(?i)\b(?:age|aged)\b[^\d]{0,24}$", sentence[:match.start()]):
                        continue
                    u1 = u2 = "years"
                low, high = convert(match.group("low"), u1 or u2), convert(match.group("high"), u2 or u1)
                if low is None or high is None:
                    continue
                values.append((tuple(sorted((low, high))), span))
                consumed.append((match.start(), match.end()))
            pattern = age_single if field == "age" else r"(?i)(" + number + r")\s*(" + units + r")\b"
            for match in re.finditer(pattern, sentence):
                if any(match.start() < end and match.end() > start for start, end in consumed):
                    continue
                value_text = match.group(1) or (match.group(3) if field == "age" else None)
                source_unit = match.group(2) or (match.group(4) if field == "age" else None)
                value = convert(value_text, source_unit or "years")
                if value is not None:
                    values.append(((value,), span))
            continue
        pattern = patterns.get(field)
        if pattern is None:
            alias_pattern = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
            if not alias_pattern:
                continue
            pattern = r"(?i)(?:" + alias_pattern + r")[^\d]{0,24}([-+]?\d+(?:\.\d+)?)"
        for match in re.finditer(pattern, sentence):
            if field == "cn" and match.group(0).endswith("+"):
                values.append(((1.0,), span))
                continue
            try:
                endpoints = [float(match.group(1))]
                if field in {"pt", "pn", "ct", "cn", "ypn"} and match.lastindex and match.lastindex >= 2 and match.group(2):
                    endpoints.append(float(match.group(2)))
                values.append((tuple(sorted(endpoints)), span))
            except (TypeError, ValueError, IndexError):
                continue
    active_ids = {s["sentence_id"] for s in _active_spans(condition, spans)}
    dedup = {}
    for endpoints, span in values:
        if span["sentence_id"] in active_ids:
            dedup.setdefault((endpoints, span["sentence_id"]), (endpoints, span))
    values = list(dedup.values())
    if not values:
        return None
    compare = {"<": lambda x: x < threshold, "<=": lambda x: x <= threshold,
               ">": lambda x: x > threshold, ">=": lambda x: x >= threshold,
               "==": lambda x: x == threshold, "!=": lambda x: x != threshold}.get(op)
    if compare is None:
        return {"status": UNKNOWN, "reason": "unsupported_numeric_operator", "evidence": [s for _, s in values]}
    outcomes = {compare(endpoint) for endpoints, _ in values for endpoint in endpoints}
    if op in {"==", "!="} and any(len(endpoints) > 1 and endpoints[0] < threshold < endpoints[-1]
                                  for endpoints, _ in values):
        outcomes.add(compare(threshold))
    state = TRUE if outcomes == {True} else FALSE if outcomes == {False} else UNKNOWN
    observed = [list(endpoints) for endpoints, _ in values]
    return {"status": state, "reason": "deterministic_numeric_comparison" if state != UNKNOWN else "numeric_range_or_values_cross_threshold",
            "values": observed, "value": observed[0][0] if len(observed) == 1 and len(observed[0]) == 1 else None,
            "evidence": list({s["sentence_id"]: s for _, s in values}.values())}


class TrialMatchAIAdapter:
    """Phi-4 + the required LoRA, using the official eligibility chat interface.

    Prompt source (verbatim CoT message content): TrialMatchAI commit
    5c2df1654eed2894c526ef433e8f49bba189011b,
    src/trialmatchai/matching/eligibility_base.py:63-155.
    One atomic condition is the only criterion supplied to that interface.
    """
    MAX_NEW_TOKENS = 2048
    OFFICIAL_COMMIT = "5c2df1654eed2894c526ef433e8f49bba189011b"
    OFFICIAL_SOURCE = "src/trialmatchai/matching/eligibility_base.py"
    _INCLUSION_LABELS = {"Met", "Not Met", "Unclear", "Irrelevant"}
    _EXCLUSION_LABELS = {"Violated", "Not Violated", "Unclear", "Irrelevant"}

    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda:0"):
        self.model, self.tokenizer, self.device = model, tokenizer, device

    @classmethod
    def from_pretrained(cls, device: str = "cuda:0", cache_dir: str | None = None,
                        base_model: str = TRIALMATCH_BASE,
                        adapter_model: str = TRIALMATCH_ADAPTER):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftConfig, PeftModel
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError("TrialMatchAI requires torch, transformers, peft, accelerate, and bitsandbytes; no fallback is permitted") from exc
        config = PeftConfig.from_pretrained(adapter_model, cache_dir=cache_dir)
        base = getattr(config, "base_model_name_or_path", None)
        if base != TRIALMATCH_BASE:
            raise RuntimeError(f"TrialMatchAI adapter base mismatch: expected {TRIALMATCH_BASE}, got {base!r}")
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, use_fast=True)
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError("TrialMatchAI requires the Phi-4 chat template; plain-text fallback is disabled")
        base_model_instance = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir,
            quantization_config=qconfig, torch_dtype=torch.bfloat16, device_map={"": device})
        model = PeftModel.from_pretrained(base_model_instance, adapter_model, cache_dir=cache_dir, is_trainable=False)
        lora_parameters = [(name, value) for name, value in model.named_parameters() if ".lora_" in name]
        if not lora_parameters or not getattr(model, "active_adapters", None):
            raise RuntimeError("TrialMatchAI LoRA was not active after loading; refusing base-model inference")
        if not any(bool(torch.count_nonzero(value).item()) for name, value in lora_parameters if ".lora_B." in name):
            raise RuntimeError("TrialMatchAI loaded no nonzero LoRA B weights; refusing base-model inference")
        model.eval()
        return cls(model, tokenizer, device=device)

    @staticmethod
    def _messages(patient: str, criterion_text: str, criterion_kind: str) -> list[dict[str, str]]:
        # Preserve the official CoT prompt and its schema. Only the input criteria
        # block changes: it contains exactly the single atom under assessment.
        heading = "Exclusion Criteria" if criterion_kind == "exclusion" else "Inclusion Criteria"
        criteria_text_formatted = f"Eligibility Criteria:\n{heading}:\n{criterion_text}"
        system_msg = (
            "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
            "Answer the following question. Before answering, create a concise chain of thoughts reasoning to ensure a logical and accurate response.\n"
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": (
                "Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.\n\n"
                "### INCLUSION CRITERIA ASSESSMENT\n"
                "For each inclusion criterion, classify it as one of:\n"
                "- **Met:** The patient's data explicitly and unequivocally satisfies the criterion.\n"
                "- **Not Met:** The patient's data explicitly and unequivocally contradicts or fails to satisfy the criterion.\n"
                "- **Unclear:** Insufficient or missing patient data to verify.\n"
                "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
                "### EXCLUSION CRITERIA ASSESSMENT\n"
                "For each exclusion criterion, classify it as one of:\n"
                "- **Violated:** The patient's data explicitly and unequivocally violates the criterion.\n"
                "- **Not Violated:** The patient's data confirms compliance with the criterion.\n"
                "- **Unclear:** Insufficient or missing patient data to verify.\n"
                "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
                "### IMPORTANT INSTRUCTIONS\n"
                "- Ensure all criteria are assessed one-by-one.\n"
                "- Use **only** the provided patient data; **do not infer, assume, or extrapolate beyond the given information.**\n"
                "- Justifications must be strictly based on direct evidence from the patient profile.\n"
                "### RESPONSE FORMAT (STRICTLY FOLLOW)\n"
                "{\n"
                '  "Inclusion_Criteria_Evaluation": [\n'
                '    {"Criterion": "Exact inclusion criterion text", "Classification": "Met | Not Met | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
                "  ],\n"
                '  "Exclusion_Criteria_Evaluation": [\n'
                '    {"Criterion": "Exact exclusion criterion text", "Classification": "Violated | Not Violated | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
                "  ],\n"
                '  "Recap": "Concise summary of key qualifying/disqualifying factors",\n'
                '  "Final Decision": "Eligible | Likely Eligible (leaning toward inclusion) | Likely Ineligible (leaning toward exclusion) | Ineligible"\n'
                "}\n\n"
                "### INPUT\n"
                "---Start of Clinical Trial Criteria---\n"
                f"{criteria_text_formatted}\n"
                "---End of Clinical Trial Criteria---\n\n"
                "----\n"
                "---Start of Patient Description---\n"
                f"{patient}\n"
                "---End of Patient Description---\n"
                "## IMPORTANT REMINDER:\n"
                "NEVER make assumptions, inferences, or extrapolations beyond the explicitly stated patient information."
            )},
        ]

    @staticmethod
    def _strip_thinking(raw: str) -> str:
        # Match the official handling of Phi reasoning LoRA's second <think>
        # closing delimiter as well as the conventional </think> delimiter.
        clean = re.sub(r"<think>.*?(?:</think>|<think>)", "", raw, flags=re.DOTALL)
        cut = clean.find("<think>")
        if cut != -1:
            clean = clean[:cut]
        return clean.strip()

    @classmethod
    def _parse_response(cls, raw: str, criterion_text: str, criterion_kind: str = "inclusion",
                        *, truncated: bool = False) -> dict[str, Any]:
        from .trialmatch_parser import parse_trialmatch_response
        return parse_trialmatch_response(raw, criterion_text, criterion_kind, truncated=truncated)

    @staticmethod
    def _eos_ids(tokenizer: Any, model: Any) -> set[int]:
        eos_ids: set[int] = set()
        for eos_value in (getattr(tokenizer, "eos_token_id", None),
                          getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
            values = eos_value if isinstance(eos_value, (list, tuple, set)) else [] if eos_value is None else [eos_value]
            eos_ids.update(int(value) for value in values)
        return eos_ids

    def judge_many(self, requests: list[dict[str, Any]], batch_size: int = 8) -> list[dict[str, Any]]:
        """Batch the same single-criterion prompts without changing their semantics.

        Each request has patient, condition, and optional criterion_kind. Recorded
        inference_seconds is shared batch wall time / actual generated batch size.
        It is a throughput allocation, not a measurement of isolated latency.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not requests:
            return []
        import torch
        self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            raise RuntimeError("TrialMatchAI requires a valid left-padding token")
        results: list[dict[str, Any] | None] = [None] * len(requests)
        pending: list[dict[str, Any]] = []
        limit = getattr(self.model.config, "max_position_embeddings", None)
        for index, request in enumerate(requests):
            patient, condition = request["patient"], request["condition"]
            if not isinstance(patient, str) or not isinstance(condition.get("text"), str):
                raise TypeError("TrialMatchAI requests require string patient and condition.text")
            kind = "exclusion" if request.get("criterion_kind", "inclusion") == "exclusion" else "inclusion"
            criterion_text = condition["text"]
            prompt = self.tokenizer.apply_chat_template(self._messages(patient, criterion_text, kind),
                tokenize=False, add_generation_prompt=True)
            ids = self.tokenizer(prompt, truncation=False, add_special_tokens=False)["input_ids"]
            input_tokens = len(ids)
            metadata = {"criterion_kind": kind, "input_tokens": input_tokens,
                        "max_new_tokens": self.MAX_NEW_TOKENS, "prompt_commit": self.OFFICIAL_COMMIT,
                        "prompt_source": self.OFFICIAL_SOURCE, "prompt_style": "official_cot_single_criterion"}
            if limit and input_tokens + self.MAX_NEW_TOKENS > limit:
                results[index] = {**metadata, "raw_output": "", "raw_label": None, "state": UNKNOWN,
                    "output_tokens": 0, "truncated": False, "parse_error": "input_context_limit",
                    "inference_seconds": 0.0, "batch_elapsed_seconds": 0.0, "generation_batch_size": 0,
                    "timing_method": "not_generated_input_context_limit"}
            else:
                pending.append({"index": index, "prompt": prompt, "criterion": criterion_text,
                                "kind": kind, "metadata": metadata})
        eos_ids = self._eos_ids(self.tokenizer, self.model)
        model_device = getattr(self.model, "device", self.device)
        is_cuda = str(model_device).startswith("cuda")
        for batch_id, start in enumerate(range(0, len(pending), batch_size), 1):
            batch = pending[start:start + batch_size]
            if is_cuda:
                torch.cuda.synchronize(model_device)
            batch_start = time.perf_counter()
            encoded = self.tokenizer([item["prompt"] for item in batch], padding=True,
                truncation=False, add_special_tokens=False, return_tensors="pt")
            if "attention_mask" not in encoded:
                raise RuntimeError("TrialMatchAI batch tokenizer did not return attention_mask")
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                output = self.model.generate(**encoded, do_sample=False, use_cache=True,
                    max_new_tokens=self.MAX_NEW_TOKENS, pad_token_id=self.tokenizer.pad_token_id)
            for batch_index, item in enumerate(batch):
                token_ids = output[batch_index, prompt_width:].detach().cpu().tolist()
                # Other rows may continue after this row's EOS, adding padding.
                # Count/decode only this row's actual completion, including its EOS.
                eos_at = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)
                if eos_at is not None:
                    token_ids = token_ids[:eos_at + 1]
                output_tokens = len(token_ids)
                truncated = output_tokens >= self.MAX_NEW_TOKENS and eos_at is None
                raw = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                parsed = self._parse_response(raw, item["criterion"], item["kind"], truncated=truncated)
                results[item["index"]] = {**item["metadata"], "raw_output": raw,
                    "output_tokens": output_tokens, **parsed, "generation_batch_id": batch_id,
                    "generation_batch_size": len(batch), "padded_input_tokens": prompt_width}
            if is_cuda:
                torch.cuda.synchronize(model_device)
            elapsed = time.perf_counter() - batch_start
            for item in batch:
                results[item["index"]].update(inference_seconds=elapsed / len(batch),
                    batch_elapsed_seconds=elapsed, timing_method="shared_batch_wall_time_divided_by_batch_size")
        if any(result is None for result in results):
            raise RuntimeError("TrialMatchAI batch did not return every requested condition")
        return list(results)

    def judge(self, patient: str, condition: dict[str, Any], criterion_kind: str = "inclusion") -> dict[str, Any]:
        return self.judge_many([{"patient": patient, "condition": condition,
                                 "criterion_kind": criterion_kind}], batch_size=1)[0]


class BioLinkBERTMedNLI:
    """MedNLI classifier; class indices are read from the loaded config."""
    def __init__(self, model: Any, tokenizer: Any, id2label: dict[int, str], device: str = "cuda:0"):
        self.model, self.tokenizer, self.id2label, self.device = model, tokenizer, id2label, device
        self.label2id = {label.casefold(): idx for idx, label in id2label.items()}
        self.required = {name: next((idx for idx, label in id2label.items() if label.casefold() == name), None)
                         for name in ("entailment", "neutral", "contradiction")}
        if any(value is None for value in self.required.values()):
            raise ValueError(f"MedNLI config lacks required labels: {id2label}")

    @classmethod
    def from_pretrained(cls, device: str = "cuda:0", cache_dir: str | None = None,
                        model_name: str = BIOLINK_MEDNLI):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        id2label = {int(k): str(v) for k, v in config.id2label.items()}
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
            config=config, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map={"": device})
        model.eval()
        return cls(model, tokenizer, id2label, device=device)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[dict[str, Any]]:
        import torch
        results: list[dict[str, Any] | None] = [None] * len(pairs)
        valid: list[tuple[int, tuple[str, str]]] = []
        for index, pair in enumerate(pairs):
            premise_ids = self.tokenizer(pair[0], add_special_tokens=False, truncation=False)["input_ids"]
            hypothesis_ids = self.tokenizer(pair[1], add_special_tokens=False, truncation=False)["input_ids"]
            special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
            if len(premise_ids) + len(hypothesis_ids) + special_tokens > 512:
                results[index] = {"relation": "neutral", "class_id": -1, "scores": {},
                                  "skipped_reason": "candidate_sentence_exceeds_512_tokens; no truncation applied"}
            else:
                valid.append((index, pair))
        with torch.inference_mode():
            for start in range(0, len(valid), batch_size):
                indexed_batch = valid[start:start + batch_size]
                batch = [row for _, row in indexed_batch]
                encoded = self.tokenizer([p for p, _ in batch], [h for _, h in batch],
                    padding=True, truncation=False, return_tensors="pt")
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits.float()
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
                for (original_index, _), row in zip(indexed_batch, probs):
                    idx = max(range(len(row)), key=row.__getitem__)
                    label = self.id2label[idx].casefold()
                    results[original_index] = {"relation": label, "class_id": idx,
                        "scores": {self.id2label[j].casefold(): float(v) for j, v in enumerate(row)}}
        return [row or {"relation": "neutral", "class_id": -1, "scores": {},
                        "skipped_reason": "missing NLI output"} for row in results]


@dataclass
class CAREModule1V6:
    trialmatch: Any
    mednli: Any
    nli_batch_size: int = 32
    candidate_limit: int = 8

    @classmethod
    def from_pretrained(cls, device: str = "cuda:0", cache_dir: str | None = None,
                        nli_batch_size: int = 32, candidate_limit: int = 8,
                        base_model: str = TRIALMATCH_BASE, adapter_model: str = TRIALMATCH_ADAPTER,
                        evidence_model: str = BIOLINK_MEDNLI):
        trialmatch = TrialMatchAIAdapter.from_pretrained(device=device, cache_dir=cache_dir,
            base_model=base_model, adapter_model=adapter_model)
        mednli = BioLinkBERTMedNLI.from_pretrained(device=device, cache_dir=cache_dir,
            model_name=evidence_model)
        return cls(trialmatch, mednli, nli_batch_size=nli_batch_size, candidate_limit=candidate_limit)

    def _evaluate_condition(self, patient: str, spans: list[dict[str, Any]], condition: dict[str, Any], kind: str) -> dict[str, Any]:
        tm_start = time.perf_counter()
        criterion_kind = condition.get("criterion_kind", kind)
        tm_error = None
        try:
            tm = self.trialmatch.judge(patient, condition, criterion_kind=criterion_kind)
        except Exception as exc:
            tm_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            tm = {"raw_output": "", "raw_label": None, "state": UNKNOWN, "criterion_kind": criterion_kind,
                  "parse_status": "ERROR", "parse_error": "generation_error"}
        trialmatch_seconds = time.perf_counter() - tm_start
        numeric_start = time.perf_counter()
        numeric = _numeric_observation(condition, spans)
        numeric_seconds = time.perf_counter() - numeric_start
        evidence_start = time.perf_counter()
        candidates = candidate_sentences(condition, spans, limit=self.candidate_limit)
        pairs = [(span["text"], "For this patient, the following condition is true: " + condition["text"]) for span in candidates]
        selection_seconds = time.perf_counter() - evidence_start
        nli_start = time.perf_counter()
        nli_rows = self.mednli.predict(pairs, batch_size=self.nli_batch_size) if pairs else []
        mednli_seconds = time.perf_counter() - nli_start
        nli_evidence = [dict(span, nli_relation=row["relation"], scores=row["scores"], class_id=row["class_id"]) for span, row in zip(candidates, nli_rows)]
        support = [row for row in nli_evidence if row["nli_relation"] == "entailment"]
        refute = [row for row in nli_evidence if row["nli_relation"] == "contradiction"]
        if support and refute:
            evidence_status, evidence_state, evidence_conflict = "disputed", UNKNOWN, True
        elif support or refute:
            evidence_status, evidence_state, evidence_conflict = "supported", TRUE if support else FALSE, False
        else:
            evidence_status, evidence_state, evidence_conflict = "missing", UNKNOWN, False
        numeric_dispute = bool(numeric is not None and numeric.get("status") == UNKNOWN and numeric.get("reason") == "numeric_range_or_values_cross_threshold")
        if numeric is not None and numeric.get("status") in (TRUE, FALSE):
            applicability, applicability_source, resolution = numeric["status"], "deterministic_numeric", "deterministic_numeric"
        else:
            applicability, applicability_source = tm["state"], "trialmatch"
            resolution = "trialmatch_retained" if numeric is None or not condition.get("numeric") else "trialmatch_retained_numeric_missing_or_disputed"
        conflict = evidence_conflict or numeric_dispute or (applicability in (TRUE, FALSE) and evidence_state in (TRUE, FALSE) and applicability != evidence_state)
        if conflict:
            resolution += "+evidence_disputed" if evidence_conflict else "+numeric_disputed"
        if numeric is not None and numeric.get("reason") == "deterministic_numeric_comparison":
            target = support if numeric["status"] == TRUE else refute
            present = {row["sentence_id"] for row in target}
            for span in numeric.get("evidence", []):
                if span["sentence_id"] not in present: target.append(dict(span, evidence_source="deterministic_numeric"))
            if not support and not refute: evidence_status = "supported"
        return {"condition_id": condition["condition_id"], "condition": condition["text"], "role": kind,
            "trialmatch": tm, "numeric": numeric, "nli_candidates": nli_evidence, "evidence_state": evidence_state,
            "evidence_status": evidence_status, "status": applicability, "applicability": applicability,
            "applicability_source": applicability_source, "resolution": resolution, "conflict": conflict,
            "trialmatch_seconds": trialmatch_seconds, "numeric_seconds": numeric_seconds,
            "evidence_selection_seconds": selection_seconds, "mednli_seconds": mednli_seconds,
            "support_evidence": support, "refute_evidence": refute,
            "unknown_evidence": [row for row in nli_evidence if row["nli_relation"] == "neutral"],
            "allow_attention_intervention": applicability in (TRUE, FALSE) and evidence_status == "supported" and not conflict,
            "generation_status": "ERROR" if tm_error else tm.get("parse_status", "accepted" if tm.get("raw_label") else "unparsed"),
            "technical_error": tm_error}

    def run(self, guideline: Any, patient: Any, question: str) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip(): raise ValueError("question must be a nonempty string")
        patient_text, rules, spans = _patient_text(patient), _rule_list(guideline), _patient_spans(patient, _patient_text(patient))
        started, output_rules, relations = time.perf_counter(), [], []
        for rule in rules:
            all_conditions, all_exceptions, raw_states, numeric_states = [], [], {}, {}
            for kind, group, dest in (("inclusion", rule["conditions"], all_conditions), ("exclusion", rule["exceptions"], all_exceptions)):
                for condition in group:
                    row = self._evaluate_condition(patient_text, spans, condition, kind); dest.append(row); cid = condition["condition_id"]
                    raw_states[cid], numeric_states[cid] = row["trialmatch"]["state"], row["status"]
                    relations.append({"rule_id": rule["rule_id"], "condition_id": cid, "condition": condition["text"], "role": kind,
                        "status": {TRUE:"satisfied",FALSE:"not_satisfied",UNKNOWN:"unknown"}[row["status"]], "applicability": row["applicability"],
                        "applicability_source": row["applicability_source"], "trialmatch_status": row["trialmatch"]["state"],
                        "evidence_status": row["evidence_status"], "evidence": row["support_evidence"]+row["refute_evidence"]+row["unknown_evidence"],
                        "refute_evidence": row["refute_evidence"], "support_evidence": row["support_evidence"], "unknown_evidence": row["unknown_evidence"],
                        "allow_attention_intervention": row["allow_attention_intervention"],
                        "confidence": 1.0 if row["allow_attention_intervention"] else 0.0,
                        "uncertain": row["status"] == UNKNOWN})
            model_base, numeric_base = eval_logic(rule["logic"], raw_states), eval_logic(rule["logic"], numeric_states)
            if all_exceptions:
                model_state = tri_and([model_base, tri_not(eval_logic(rule["exception_logic"], raw_states))])
                applicability = tri_and([numeric_base, tri_not(eval_logic(rule["exception_logic"], numeric_states))])
            else: model_state, applicability = model_base, numeric_base
            specs = {c["condition_id"]: c for c in rule["conditions"] + rule["exceptions"]}; rows = all_conditions + all_exceptions
            cond_spans = [{"condition_id": r["condition_id"], **(specs[r["condition_id"]].get("source_span") or {"start":None,"end":None})} for r in rows]
            support = [e for x in rows for e in x["support_evidence"]]; refute = [e for x in rows for e in x["refute_evidence"]]
            decisive = [x for x in all_conditions if x["status"] == TRUE] or [x for x in rows if x["status"] == TRUE]
            allow = applicability == TRUE and bool(decisive) and any(x["allow_attention_intervention"] for x in decisive) and not any(x["evidence_status"] == "disputed" for x in decisive)
            evidence_status = "disputed" if any(x["evidence_status"] == "disputed" for x in rows) else "supported" if support or refute else "missing"
            output_rules.append({"rule_id": rule["rule_id"], "rule_text": rule.get("text", ""), "applicability": applicability,
                "trialmatch_applicability": applicability, "trialmatch_model_only_applicability": model_state, "conditions": all_conditions,
                "exceptions": all_exceptions, "condition_spans": cond_spans, "action": rule.get("action"),
                "action_spans": [rule["action_span"]] if rule.get("action_span") else [], "source_span": rule.get("source_span"),
                "support_evidence": support, "refute_evidence": refute, "evidence_status": evidence_status,
                "allow_attention_intervention": allow, "uncertain": applicability == UNKNOWN or evidence_status == "disputed"})
        applicability_rows = [{"rule_id": r["rule_id"], "state": r["applicability"], "status": {TRUE:"applicable",FALSE:"not_applicable",UNKNOWN:"unknown"}[r["applicability"]],
            "trialmatch_state": r["trialmatch_applicability"], "evidence_status": r["evidence_status"],
            "decisive_condition_ids": [c["condition_id"] for c in r["conditions"] if c["status"] in (TRUE, FALSE)],
            "allow_attention_intervention": r["allow_attention_intervention"], "uncertain": r["uncertain"]} for r in output_rules]
        critical_spans = [{"rule_id": x["rule_id"], "condition_id": x["condition_id"], **e} for x in relations for e in x["evidence"]]
        binding_gate, action_gate, conflict_scores, relation_weights = {}, {}, [], []
        for rule in output_rules:
            rid = rule["rule_id"]; action_gate[rid] = 1.0 if rule["allow_attention_intervention"] else 0.0; binding_gate[rid] = {}
            for rel in [x for x in relations if x["rule_id"] == rid]:
                gate = 1.0 if rel["allow_attention_intervention"] else 0.0; binding_gate[rid][rel["condition_id"]] = gate
                sign = 1 if rel["status"] == "satisfied" else -1 if rel["status"] == "not_satisfied" else 0
                relation_weights.append({"rule_id":rid,"condition_id":rel["condition_id"],"condition":rel["condition"],"status":rel["status"],"evidence":rel["evidence"],"weight":gate,"signed_weight":sign*gate,"binding_gate":gate,"action_gate":action_gate[rid],"prior_competition_weight":0.0,"decisive":rel["evidence_status"] == "disputed"})
            conflict_scores.append({"rule_id":rid,"score":1.0 if rule["evidence_status"] == "disputed" else 0.0,"status":rule["evidence_status"],"prior_support":"unknown","guided_support":"unknown","reason":"BioLinkBERT is evidence-only; it never changes applicability"})
        result = {"schema_version":"module1.v6","question":question,"rules":output_rules,"candidate_rules":rules,"prior_decision":None,"guideline_decision":None,
            "conflict_scores":conflict_scores,"relevance_scores":[{"rule_id":r["rule_id"],"score":1.0,"source":"caller_supplied_rule"} for r in rules],"condition_evidence_relations":relations,
            "applicability":applicability_rows,"critical_spans":critical_spans,"relation_weights":relation_weights,"binding_gate":binding_gate,"action_gate":action_gate,
            "metadata":{"elapsed_seconds":round(time.perf_counter()-started,6),"trialmatch_base":TRIALMATCH_BASE,"trialmatch_adapter":TRIALMATCH_ADAPTER,"evidence_model":BIOLINK_MEDNLI,
                "reference_answers_used":False,"qwen_called":False,"scores_are_calibrated":False,"rule_parsing":"offline_structured_only","evidence_offsets":"Python character offsets [start,end)","biolinkbert_role":"evidence_only"}}
        json.dumps(result, ensure_ascii=False, allow_nan=False); return result


_default_v6: CAREModule1V6 | None = None


def validate_v6_input(guideline: Any, patient: Any, question: str) -> None:
    """Check the existing V6 input contract without loading models or mutating rules."""
    from copy import deepcopy

    if not isinstance(question, str) or not question.strip():
        raise ValueError("V6 question must be a nonempty string")
    try:
        _rule_list(deepcopy(guideline))
    except AttributeError as exc:
        raise ValueError("V6 conditions and exceptions must be objects with nonempty text fields") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid V6 structured rules: " + str(exc)) from exc
    try:
        _patient_text(patient)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid V6 patient: " + str(exc)) from exc


def create_module1_v6(*, device: str | None = None, cache_dir: str | None = None,
                      base_model: str | None = None, adapter_model: str | None = None,
                      evidence_model: str | None = None, nli_batch_size: int = 32,
                      candidate_limit: int = 8) -> CAREModule1V6:
    """Create V6 with one configuration path for package, CLI, and pipeline.

    Explicit arguments take precedence over CARE_MODULE1_V6_* environment
    variables, which in turn take precedence over the original model defaults.
    This constructs model adapters; it does not run inference.
    """
    def configured(value: str | None, key: str, default: str | None = None):
        return value if value is not None else os.environ.get("CARE_MODULE1_V6_" + key) or default

    return CAREModule1V6.from_pretrained(
        device=configured(device, "DEVICE", "cuda:0"),
        cache_dir=configured(cache_dir, "CACHE"),
        base_model=configured(base_model, "BASE_MODEL", TRIALMATCH_BASE),
        adapter_model=configured(adapter_model, "ADAPTER_MODEL", TRIALMATCH_ADAPTER),
        evidence_model=configured(evidence_model, "EVIDENCE_MODEL", BIOLINK_MEDNLI),
        nli_batch_size=nli_batch_size, candidate_limit=candidate_limit)


def set_default_v6(instance: CAREModule1V6 | None) -> None:
    global _default_v6
    _default_v6 = instance


def run(guideline: Any, patient: Any, question: str) -> dict[str, Any]:
    """V6 entry point with the stable run(guideline, patient, question) call shape.

    Construct and reuse V6 lazily through create_module1_v6. Rules must already
    be structured. The legacy implementation remains available as run_legacy.
    """
    global _default_v6
    validate_v6_input(guideline, patient, question)
    if _default_v6 is None:
        _default_v6 = create_module1_v6()
    return _default_v6.run(guideline, patient, question)
