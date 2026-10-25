"""FaithEval public README semantics, with no claim of a released full evaluator."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import string


OFFICIAL_COMMIT = "58d35840e3fbc7bf4b7672582a417b3b3a327dec"
TASKS = ("unanswerable", "inconsistent", "counterfactual")
PHRASES = {
    "unanswerable": ["unknown", "no answer", "no information", "not", "unclear"],
    "inconsistent": ["conflict", "multiple answers", "disagreement", "inconsistent", "contradictory",
                     "contradiction", "inconsistency", "two answers", "2 answers", "conflicting"],
}
TASK_PROMPTS = {
    "unanswerable": "If there is no information available from the context, the answer should be 'unknown'. ",
    "inconsistent": "If there is conflict information or multiple answers from the context, the answer should be 'conflict'.",
    "counterfactual": "",  # Paper's context-only QA task; not an invented fallback label.
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class FaithEvalInput:
    context: str
    question: str


def method_input(row):
    context, question = row["context"], row["question"]
    if not isinstance(context, str) or not isinstance(question, str):
        raise ValueError("The published native input fields must be strings")
    return FaithEvalInput(context, question)


def messages(task: FaithEvalInput, kind):
    if kind not in TASKS:
        raise ValueError("Unknown FaithEval task")
    text = ("You are an expert in retrieval question answering. \n"
            "Please respond with the exact answer only. Do not be verbose or provide extra information.\n" +
            TASK_PROMPTS[kind] + "\nContext: " + task.context + "\nQuestion: " + task.question + "\nAnswer:")
    return [{"role": "user", "content": text}]


def normalize_answer(value):
    # Same transformations/order as the pinned README's public function.
    value = value.replace("_", " ").lower()
    punctuation = set(string.punctuation + "‘’´`")
    value = "".join(char if char not in punctuation else " " for char in value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split()).strip()


def score_prediction(prediction, kind, *, strict=False, reference_answers=None):
    if kind not in TASKS:
        raise ValueError("Unknown FaithEval task")
    if not isinstance(prediction, str):
        return {"status": "invalid_prediction", "correct": 0}
    normalized = normalize_answer(prediction)
    if kind in PHRASES:
        valid = PHRASES[kind][:1] if strict else PHRASES[kind]
        return {"status": "ok", "correct": int(any(phrase in normalized for phrase in valid)),
                "metric": "README_normalized_phrase_containment", "strict_phrase_list": strict,
                "not_exact_whole_answer_match": True}
    if not reference_answers or any(not isinstance(value, str) for value in reference_answers):
        return {"status": "blocked", "correct": None,
                "reason": "Explicit verified counterfactual reference mapping is unavailable; do not guess a gold field"}
    valid = [normalize_answer(value) for value in reference_answers]
    if not all(valid):
        return {"status": "blocked", "correct": None, "reason": "Empty normalized reference"}
    return {"status": "ok", "correct": int(any(value in normalized for value in valid)),
            "metric": "paper_answer_mention_accuracy_adaptation", "official_full_script_reproduced": False}


def source_overlap_report(calibration_records, evaluation_records):
    """Consume previously verified public ID/context hashes, never emit text.

    ID hashes must have the same declared namespace; a failed exact-context match
    cannot exclude overlap with modified FaithEval contexts.
    """
    def ids(records):
        return {(row["source_namespace"], row["source_id_sha256"]) for row in records
                if row.get("source_namespace") and row.get("source_id_sha256")}
    def texts(records):
        return {row["context_sha256"] for row in records if row.get("context_sha256")}
    a, b = ids(calibration_records), ids(evaluation_records)
    return {"shared_public_source_ids": len(a & b),
            "shared_exact_context_hashes": len(texts(calibration_records) & texts(evaluation_records)),
            "calibration_ids_available": len(a), "evaluation_ids_available": len(b),
            "source_independence": "not_established",
            "reason": "Modified contexts can share HotpotQA origins without exact text overlap; absent IDs remain unknown"}
