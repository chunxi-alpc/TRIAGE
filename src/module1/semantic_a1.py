"""CARE A1: frozen lightweight T/F/U applicability signals for module-two research.

This interface uses the exact PR #12 A1 system prompt and fixed demonstrations.
It does not run atom decomposition, NLI, confidence calibration, retrieval, or
evidence admission gates. Technical failures stay distinct from medical U.

The original multi-case experiment runner is kept for batch reproduction.
This class adds a single-case interface using the same messages and decoding
parameters; numerical identity across differing batch shapes is not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .care_v6 import TrialMatchAIAdapter, TRIALMATCH_BASE, TRIALMATCH_ADAPTER
from experiments.care_module1_semantic_v8.prompts import (
    messages_for, parse_output, prediction_input,
)

A1_SCHEMA = "module1.a1"
A1_SOURCE = "care.module1.semantic_a1.frozen"
_ASSISTANT_PREFIX = '{"rules":['
_EXAMPLES_PATH = (Path(__file__).resolve().parents[2] /
                  "experiments/care_module1_semantic_v8/data/fewshot_examples.json")
_RETRY = (
    "The answer did not match the JSON output format. Return the complete "
    "JSON object with exactly the supplied rule IDs, T/F/U states, short "
    "evidence_quote and reason_short strings. Keep quotes and reasons short. "
    "No extra text."
)
SIGNS = {"T": 1, "F": -1, "U": 0}


def _inputs(guideline: Any, patient: str, question: str) -> dict:
    """A1 accepts full natural-language rule entries with stable IDs/actions.

    This is not automatic segmentation of an unstructured guideline document.
    Existing structured rules can also be verbalized by the frozen A1 prompt
    helper, provided their logic, atoms, ID and action are already supplied.
    """
    if not isinstance(patient, str) or not patient.strip():
        raise ValueError("A1 patient must be a nonempty narrative string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A1 question must be a nonempty string")
    rules = guideline.get("rules") if isinstance(guideline, dict) else guideline
    if not isinstance(rules, list) or not rules:
        raise ValueError("A1 guideline must be a nonempty list of rule objects (or {'rules': [...]})")
    ids = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Every A1 rule must be an object")
        rid, action = rule.get("rule_id"), rule.get("action")
        if not isinstance(rid, str) or not rid.strip() or not isinstance(action, str) or not action.strip():
            raise ValueError("Every A1 rule needs a stable rule_id and nonempty original action")
        if not (isinstance(rule.get("text"), str) and rule["text"].strip()
                or isinstance(rule.get("natural_rule_text"), str) and rule["natural_rule_text"].strip()
                or "logic" in rule):
            raise ValueError("Every A1 rule needs natural text or an already parsed logic expression")
        ids.append(rid)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate A1 rule_id")
    return prediction_input({"question": question, "patient": patient, "rules": rules})


@dataclass
class CAREModule1A1:
    adapter: Any
    examples: dict
    max_new_tokens: int = 384

    @classmethod
    def from_pretrained(cls, device: str = "cuda:0", cache_dir: str | None = None,
                        base_model: str = TRIALMATCH_BASE, adapter_model: str = TRIALMATCH_ADAPTER,
                        examples_path: str | Path | None = None, max_new_tokens: int = 384):
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        location = Path(examples_path) if examples_path is not None else _EXAMPLES_PATH
        examples = json.loads(location.read_text(encoding="utf-8"))
        if (examples.get("count_demonstrations") != 3 or
                examples.get("count_patient_narratives") != 4):
            raise ValueError("A1 requires PR #12's original three demonstrations / four narratives")
        adapter = TrialMatchAIAdapter.from_pretrained(
            device=device, cache_dir=cache_dir, base_model=base_model, adapter_model=adapter_model)
        return cls(adapter=adapter, examples=examples, max_new_tokens=max_new_tokens)

    def _generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        tokenizer = self.adapter.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True) + _ASSISTANT_PREFIX
        inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(self.adapter.device)
        with torch.inference_mode():
            output = self.adapter.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        generated = output[0, inputs["input_ids"].shape[1]:].tolist()
        if tokenizer.eos_token_id in generated:
            generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
        return _ASSISTANT_PREFIX + tokenizer.decode(generated, skip_special_tokens=True)

    def run(self, guideline: Any, patient: str, question: str) -> dict:
        inputs = _inputs(guideline, patient, question)
        messages = messages_for(inputs, "A1", self.examples)
        ids = [rule["rule_id"] for rule in inputs["rules"]]
        first = self._generate(messages)
        parsed, error = parse_output(first, ids)
        attempts = 1
        if parsed is None:
            retry_messages = messages + [
                {"role": "assistant", "content": first},
                {"role": "user", "content": _RETRY},
            ]
            second = self._generate(retry_messages)
            parsed, error = parse_output(second, ids)
            attempts = 2
        predicted = {rule["rule_id"]: rule for rule in parsed} if parsed else {}
        rows = []
        for rule in inputs["rules"]:
            rid = rule["rule_id"]
            raw = predicted.get(rid, {})
            state = raw.get("state") if raw.get("state") in SIGNS else None
            quote = raw.get("evidence_quote", "") if state else ""
            offset = patient.find(quote) if quote else -1
            rows.append({
                "rule_id": rid,
                "state": state,
                "sign": SIGNS.get(state),
                "action_text": rule["action"],
                "evidence_quote": quote,
                "reason_short": raw.get("reason_short", "") if state else "",
                "optional_evidence_char_span": [offset, offset + len(quote)] if offset >= 0 else None,
                "technical_failure": state is None,
            })
        return {
            "schema_version": A1_SCHEMA,
            "source": A1_SOURCE,
            "method": "A1",
            "status": "ok" if parsed is not None else "parse_failure",
            "error": error if parsed is None else None,
            "attempts": attempts,
            "rules": rows,
            "has_module2_signal": any(r["sign"] in (-1, 1) for r in rows),
            "action_token_spans_must_be_mapped_in_final_prompt": True,
        }


def create_module1_a1(**options: Any) -> CAREModule1A1:
    """Mainline module-one factory for new module-two experiments."""
    return CAREModule1A1.from_pretrained(**options)


def signal_targets(result: dict) -> list[dict]:
    """Return T/F targets; U or technical errors provide no edit signal.

    The returned action text is never a tokenizer span; it must be located by
    rule_id in the final module-two prompt, after chat templating/tokenization.
    """
    if result.get("schema_version") != A1_SCHEMA:
        raise ValueError("Expected module1.a1 output; do not silently reinterpret V6 graphs")
    return [
        {"rule_id": row["rule_id"], "sign": SIGNS[row["state"]],
         "action_text": row["action_text"]}
        for row in result.get("rules", [])
        if row.get("state") in ("T", "F") and isinstance(row.get("action_text"), str)
        and row["action_text"].strip()
    ]
