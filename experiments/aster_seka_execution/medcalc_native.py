"""Pinned MedCalc-Verified native prompt/parser/correctness adapter.

Only the named upstream functions are loaded, avoiding run.py's model imports,
HF login, file reads and main program. No datasets or models are opened here.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import operator
from pathlib import Path
import re
import numpy as np


OFFICIAL_COMMIT = "801592132bfd833f049b517a4e013e00a5fc40fa"
SOURCE_BLOBS = {"evaluation/run.py": "8eaefa6dcfa281f0d073a230785dc3674bc5733e",
                "evaluation/evaluate.py": "41d17934c00ce187b1dccc125856529e797fd43b"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class MedCalcInput:
    patient_note: str
    question: str


def method_input(row):
    # Calculator ID, entities, limits, answer, explanation and provenance never
    # reach the model. Calculator-specific parsing belongs to the evaluator.
    note, question = row["Patient Note"], row["Question"]
    if not isinstance(note, str) or not note.strip() or not isinstance(question, str) or not question.strip():
        raise ValueError("Nonempty native Patient Note and Question are required")
    return MedCalcInput(note, question)


def arithmetic_eval(expression, _globals=None, _locals=None):
    """Numeric-only implementation of eval used by the original parser.

    It preserves scalar arithmetic/rounding and numeric tuple behavior, but never
    executes arbitrary generated Python. Unsupported expressions become upstream
    N/A or explicit parse failures; a separate code-execution track is not enabled.
    """
    tree = ast.parse(str(expression), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Arithmetic expression too large")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
              ast.Pow: operator.pow}
    calls = {"min": min, "pow": pow, "round": round, "abs": abs, "int": int, "float": float}
    calls.update({"math." + name: getattr(math, name) for name in
                  ("sqrt", "exp", "log", "log10", "log2", "floor", "ceil", "fabs")})
    calls.update({prefix + "." + name: getattr(np, name) for prefix in ("np", "numpy") for name in
                  ("sqrt", "exp", "log", "log10", "abs", "power", "round")})
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(visit(value) for value in node.elts)
        if isinstance(node, ast.List):
            return [visit(value) for value in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1e100):
                raise ValueError("Arithmetic power exceeds numeric parser limits")
            return binary[type(node.op)](left, right)
        if isinstance(node, ast.Name) and node.id == "pi":
            raise ValueError("Upstream extraction does not expose a bare pi constant")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr in {"pi", "e"}:
            return getattr(math, node.attr)
        if isinstance(node, ast.Call) and not node.keywords:
            function = node.func
            name = function.id if isinstance(function, ast.Name) else (
                function.value.id + "." + function.attr if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) else None)
            if name in calls:
                return calls[name](*(visit(arg) for arg in node.args))
        raise ValueError("Only numeric expressions are permitted by this native evaluator")
    return visit(tree.body)


class OfficialMedCalc:
    def __init__(self, official_root):
        self.root = Path(official_root)
        namespace = {"re": re, "json": json, "math": math, "datetime": datetime,
                     "eval": arithmetic_eval, "np": np}
        self.source_hashes = {}
        required = {"evaluation/run.py": {"zero_shot", "direct_answer", "extract_answer"},
                    "evaluation/evaluate.py": {"check_correctness"}}
        for relative, names in required.items():
            path = self.root / relative
            data = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != SOURCE_BLOBS[relative]:
                raise ValueError("Official evaluator source differs from the pinned file: " + relative)
            self.source_hashes[relative] = hashlib.sha256(data).hexdigest()
            tree = ast.parse(data.decode("utf-8"), filename=str(path))
            selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            if {node.name for node in selected} != names:
                raise ValueError("Missing upstream function")
            code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
            exec(compile(code, str(path), "exec"), namespace)
        self.functions = namespace

    def messages(self, task: MedCalcInput, style="direct_answer"):
        if style not in {"direct_answer", "zero_shot"}:
            raise ValueError("Only native no-demonstration styles are prepared; one-shot is a separate track")
        system, user = self.functions[style](task.patient_note, task.question)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def score(self, evaluation_row, raw_output):
        # All evaluation-only fields remain in this function, after generation.
        try:
            output = re.sub(r"\s+", " ", raw_output)
            answer, explanation = self.functions["extract_answer"](output, evaluation_row["Calculator ID"])
            correct = self.functions["check_correctness"](answer, str(evaluation_row["Ground Truth Answer"]),
                evaluation_row["Calculator ID"], str(evaluation_row["Upper Limit"]), str(evaluation_row["Lower Limit"]))
            return {"correct": int(correct), "status": "ok", "parsed_answer_sha256": digest(answer),
                    "explanation_sha256": digest(explanation)}
        except Exception as error:
            return {"correct": 0, "status": "parse_or_evaluation_error", "error_type": type(error).__name__,
                    "error_sha256": digest(str(error))}


def predict(task: MedCalcInput, generate, official: OfficialMedCalc, *, sample_id, style="direct_answer"):
    messages = official.messages(task, style)
    try:
        result = dict(generate(messages))
        result.setdefault("status", "ok")
    except Exception as error:
        result = {"status": "error", "text": "", "error_type": type(error).__name__, "error_sha256": digest(str(error))}
    public = {"sample_id": str(sample_id), "status": result["status"], "prompt_sha256": digest(messages),
              "output_sha256": digest(result.get("text", "")), "prompt_style": style,
              "costs": {key: result.get(key) for key in ("actual_forward_calls", "input_tokens", "output_tokens",
                          "elapsed_seconds", "peak_allocated_gib", "peak_reserved_gib")}}
    return public, {"sample_id": str(sample_id), "messages": messages, "generation": result}


def grouped_point_metrics(scored):
    """Accuracy uses every row; upstream's stale hardcoded-1047 std is not copied."""
    groups = {}
    for row in scored:
        groups.setdefault("overall", []).append(row["correct"])
        groups.setdefault("category:" + str(row.get("category", "unknown")), []).append(row["correct"])
    return {name: {"n": len(values), "correct": sum(values), "accuracy": sum(values) / len(values)}
            for name, values in groups.items()}
