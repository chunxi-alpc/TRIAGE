"""Explicit, immutable routing adapters for saved module-one V6 predictions.

This module does not run module one, read gold labels, infer source preference,
or infer decidability.  A cache key is exactly (sample_id, rule_id).  Its two
adapters are alternative module-two interface semantics, not two module-one
versions or downstream model results.

``strict_gate`` maps C=T/F/U to +1/-1/0 and multiplies by the saved boolean
allow_attention_intervention.  ``label_only_diagnostic`` uses the same sign
mapping without that gate.  The latter is a diagnostic of the frozen predicted
signal, not an oracle and not an end-to-end performance claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


LABEL_SIGNS = {"T": 1, "F": -1, "U": 0}
ADAPTER_MODES = ("strict_gate", "label_only_diagnostic")


class FrozenSignalError(ValueError):
    """A frozen cache or requested join violates the declared interface."""


def exact_key(row: Mapping[str, Any], *, context: str = "row") -> tuple[str, str]:
    """Read the exact key without coercion, trimming, aliases, or row fallback."""
    if not isinstance(row, Mapping):
        raise FrozenSignalError(f"{context}: expected a mapping")
    values = []
    for name in ("sample_id", "rule_id"):
        value = row.get(name)
        if not isinstance(value, str) or not value or not value.strip():
            raise FrozenSignalError(f"{context}: missing or invalid {name}")
        values.append(value)
    return values[0], values[1]


def unique_index(
    rows: Iterable[Mapping[str, Any]], *, context: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index a collection, failing instead of choosing among duplicate keys."""
    result = {}
    for position, row in enumerate(rows):
        key = exact_key(row, context=f"{context}[{position}]")
        if key in result:
            raise FrozenSignalError(f"{context}: duplicate key {key!r}")
        result[key] = row
    return result


@dataclass(frozen=True)
class FrozenSignal:
    sample_id: str
    rule_id: str
    label: str
    allow_attention_intervention: bool
    evidence_status: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FrozenSignal":
        sample_id, rule_id = exact_key(row, context="prediction")
        label = row.get("C")
        if not isinstance(label, str) or label not in LABEL_SIGNS:
            raise FrozenSignalError(
                f"prediction {(sample_id, rule_id)!r}: C must be T, F, or U"
            )
        gate = row.get("allow_attention_intervention")
        # bool is checked exactly: neither 'false' nor the integer 0 is a gate.
        if type(gate) is not bool:
            raise FrozenSignalError(
                f"prediction {(sample_id, rule_id)!r}: "
                "allow_attention_intervention must be a saved boolean"
            )
        evidence_status = row.get("evidence_status")
        if not isinstance(evidence_status, str) or not evidence_status:
            raise FrozenSignalError(
                f"prediction {(sample_id, rule_id)!r}: missing evidence_status"
            )
        return cls(sample_id, rule_id, label, gate, evidence_status)


class FrozenSignalAdapter:
    """Snapshot saved predictions and adapt only explicitly matching requests.

    ``adapt`` defaults to an exact set match.  Callers selecting a documented
    subset may explicitly set ``require_exact_keys=False``; missing requested
    signals and duplicate requests still raise an error.  Only immutable scalar
    fields are copied into the adapter, so later edits to caller dictionaries do
    not alter its saved signals.
    """

    def __init__(self, predictions: Iterable[Mapping[str, Any]]) -> None:
        indexed = unique_index(predictions, context="predictions")
        if not indexed:
            raise FrozenSignalError("predictions: empty cache")
        self._signals = {
            key: FrozenSignal.from_row(row) for key, row in indexed.items()
        }

    @property
    def keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._signals)

    def adapt(
        self,
        request_rows: Iterable[Mapping[str, Any]],
        *,
        mode: str,
        require_exact_keys: bool = True,
    ) -> list[dict[str, Any]]:
        if mode not in ADAPTER_MODES:
            raise FrozenSignalError(f"unknown adapter mode: {mode!r}")
        if type(require_exact_keys) is not bool:
            raise FrozenSignalError("require_exact_keys must be boolean")
        requested = unique_index(request_rows, context="requests")
        if not requested:
            raise FrozenSignalError("requests: empty selection")
        missing = set(requested) - self.keys
        if missing:
            raise FrozenSignalError(
                f"missing frozen prediction keys ({len(missing)}): "
                f"{sorted(missing)[:5]!r}"
            )
        extra = self.keys - set(requested)
        if require_exact_keys and extra:
            raise FrozenSignalError(
                f"unrequested frozen prediction keys ({len(extra)}): "
                f"{sorted(extra)[:5]!r}; explicitly opt into a subset if intended"
            )
        output = []
        for key in requested:
            signal = self._signals[key]
            ungated_sign = LABEL_SIGNS[signal.label]
            sign = ungated_sign
            if mode == "strict_gate" and not signal.allow_attention_intervention:
                sign = 0
            output.append(
                {
                    "sample_id": signal.sample_id,
                    "rule_id": signal.rule_id,
                    "adapter_mode": mode,
                    "signal_source": "frozen_module1_prediction.C",
                    "frozen_label": signal.label,
                    "frozen_allow_attention_intervention": (
                        signal.allow_attention_intervention
                    ),
                    "frozen_evidence_status": signal.evidence_status,
                    "ungated_route_sign": ungated_sign,
                    "route_sign": sign,
                    "route_active": sign != 0,
                    "saved_gate_applied": mode == "strict_gate",
                    "source_target": None,
                    "decidability_target": None,
                    "new_module1_inference": False,
                }
            )
        return output
