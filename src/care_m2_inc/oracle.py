"""Small three-valued logic oracle used to construct N1--N4 gold labels.

`None` is unknown.  All N3 labels are derived by enumerating every admissible
completion of unknown atoms; no model output is consulted.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

Tri = bool | None


def _as_tri(value: Any) -> Tri:
    if value is None or value == "U":
        return None
    if value is True or value == "T" or value == 1:
        return True
    if value is False or value == "F" or value == 0:
        return False
    raise ValueError(f"Unsupported truth value: {value!r}")


def _kleene_not(value: Tri) -> Tri:
    return None if value is None else not value


def _kleene_and(values: Iterable[Tri]) -> Tri:
    values = list(values)
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def _kleene_or(values: Iterable[Tri]) -> Tri:
    values = list(values)
    if any(value is True for value in values):
        return True
    if any(value is None for value in values):
        return None
    return False


def evaluate(ast: Any, values: dict[str, Tri]) -> Tri:
    """Evaluate a boolean AST using strong Kleene semantics."""
    if isinstance(ast, str):
        return _as_tri(values.get(ast))
    if isinstance(ast, bool) or ast is None:
        return _as_tri(ast)
    if not isinstance(ast, dict):
        raise TypeError(f"AST node must be a string, bool, or dict: {ast!r}")
    op = str(ast.get("op", "")).upper()
    args = ast.get("args", [])
    if op == "NOT":
        if len(args) != 1:
            raise ValueError("NOT expects one argument")
        return _kleene_not(evaluate(args[0], values))
    if op == "AND":
        return _kleene_and(evaluate(arg, values) for arg in args)
    if op == "OR":
        return _kleene_or(evaluate(arg, values) for arg in args)
    if op == "XOR":
        evaluated = [evaluate(arg, values) for arg in args]
        if any(item is None for item in evaluated):
            return None
        return sum(bool(item) for item in evaluated) % 2 == 1
    if op in {"ATLEAST", "COUNT_GE"}:
        evaluated = [evaluate(arg, values) for arg in args]
        threshold = int(ast.get("threshold", 1))
        if sum(item is True for item in evaluated) >= threshold:
            return True
        if sum(item is True for item in evaluated) + sum(item is None for item in evaluated) < threshold:
            return False
        return None
    if op == "EQUIV":
        evaluated = [evaluate(arg, values) for arg in args]
        if any(item is None for item in evaluated):
            return None
        return len(set(evaluated)) == 1
    raise ValueError(f"Unsupported AST operator: {op}")


def atoms(ast: Any) -> list[str]:
    found: list[str] = []
    if isinstance(ast, str):
        return [ast]
    if isinstance(ast, dict):
        for arg in ast.get("args", []):
            found.extend(atoms(arg))
    seen: set[str] = set()
    return [item for item in found if not (item in seen or seen.add(item))]


def canonical_ast(ast: Any) -> Any:
    if isinstance(ast, str) or isinstance(ast, bool) or ast is None:
        return ast
    if not isinstance(ast, dict):
        raise TypeError(ast)
    op = str(ast.get("op", "")).upper()
    args = [canonical_ast(arg) for arg in ast.get("args", [])]
    if op in {"AND", "OR", "XOR", "EQUIV"}:
        args = sorted(args, key=lambda item: repr(item))
    result = {"op": op, "args": args}
    if "threshold" in ast:
        result["threshold"] = int(ast["threshold"])
    return result


def enumerate_completions(ast: Any, values: dict[str, Tri], unknown_atoms: Iterable[str] | None = None) -> list[dict[str, Any]]:
    names = list(unknown_atoms or [name for name in atoms(ast) if values.get(name) is None])
    rows = []
    for assignment in product((False, True), repeat=len(names)):
        completed = dict(values)
        completed.update(dict(zip(names, assignment)))
        rows.append({"assignment": dict(zip(names, assignment)), "value": evaluate(ast, completed), "values": completed})
    return rows


def action_set(ast: Any, values: dict[str, Tri], actions: tuple[str, str] = ("A", "B")) -> set[str]:
    rows = enumerate_completions(ast, values)
    return {actions[0] if row["value"] else actions[1] for row in rows}


def decisive_attributes(ast: Any, values: dict[str, Tri], actions: tuple[str, str] = ("A", "B")) -> list[str]:
    unknown = [name for name in atoms(ast) if values.get(name) is None]
    decisive: list[str] = []
    for name in unknown:
        outcomes: set[str] = set()
        for row in enumerate_completions(ast, values):
            if name not in row["assignment"]:
                continue
            action = actions[0] if row["value"] else actions[1]
            outcomes.add(f"{row['assignment'][name]}:{action}")
        if len({item.split(":", 1)[1] for item in outcomes}) > 1:
            decisive.append(name)
    return decisive


def dq_label(ast: Any, values: dict[str, Tri], actions: tuple[str, str] = ("A", "B")) -> tuple[str, set[str], list[str]]:
    possible = action_set(ast, values, actions)
    attrs = decisive_attributes(ast, values, actions)
    return ("D" if len(possible) == 1 else "Q", possible, attrs)
