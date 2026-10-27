"""Fit one global temperature on held-out development/calibration label logits.

No model parameters are trained. Split labels are caller declarations, not a
proof of data isolation: the caller must also keep trees/patients disjoint from
the locked test set. Threshold selection is a separate development-set step.
"""

from collections.abc import Mapping
import math

from .action_blind import NONE, _number


def fit_temperature(records, *, split, lower=0.05, upper=20.0, max_iterations=100):
    """Minimize pooled multiclass NLL over one bounded scalar temperature.

    Each record is {"logits": {rule_id: value, ..., "NONE": value}, "label": id}.
    Groups may have different class counts. Test/unknown split names are rejected.
    Boundary solutions are reported so the chosen interval can be reviewed on
    development data. This fit does not certify accuracy or downstream harm.
    """
    if split not in {"development", "calibration"}:
        raise ValueError("Temperature fitting requires split='development' or 'calibration'")
    lower = _number(lower, "lower", positive=True)
    upper = _number(upper, "upper", positive=True)
    if lower >= upper:
        raise ValueError("Temperature search bounds must satisfy lower < upper")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    rows = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Every calibration record must be an object")
        logits, label = record.get("logits"), record.get("label")
        if (not isinstance(logits, Mapping) or NONE not in logits or len(logits) < 2
                or not isinstance(label, str) or label not in logits
                or not all(isinstance(k, str) and k.strip() for k in logits)):
            raise ValueError("Each record needs rule logits, NONE and a valid gold label")
        values = {k: _number(v, "calibration logit") for k, v in logits.items()}
        maximum = max(values.values())
        shifted = {k: v - maximum for k, v in values.items()}
        if not all(math.isfinite(v / lower) for v in shifted.values()):
            raise ValueError("Logit scale exceeds the finite temperature-fitting range")
        rows.append((tuple(shifted.values()), shifted[label]))
    if not rows:
        raise ValueError("At least one calibration record is required")

    def nll(temperature):
        # Divide before summation: a finite mean can have an overflowing raw sum.
        try:
            mean = math.fsum((math.log(math.fsum(math.exp(v / temperature) for v in values))
                              - gold / temperature) / len(rows) for values, gold in rows)
        except (OverflowError, ValueError):
            raise ValueError("Calibration NLL must remain finite") from None
        if not math.isfinite(mean):
            raise ValueError("Calibration NLL must remain finite")
        return mean

    # NLL is convex in inverse temperature, hence unimodal on positive T.
    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = nll(math.exp(c)), nll(math.exp(d))
    for _ in range(max_iterations):
        if right - left < 1e-10:
            break
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = nll(math.exp(c))
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = nll(math.exp(d))
    # Prefer identity scaling when the objective is flat and T is unidentifiable.
    candidates = [1.0] if lower <= 1 <= upper else []
    candidates.extend([lower, upper, math.exp((left + right) / 2)])
    temperature = min(candidates, key=nll)
    return {"temperature": temperature, "split": split, "n_records": len(rows),
            "nll_before": nll(1.0), "nll_after": nll(temperature),
            "bounds": [lower, upper], "at_boundary": temperature in (lower, upper),
            "method": "pooled_multiclass_nll_single_temperature"}
