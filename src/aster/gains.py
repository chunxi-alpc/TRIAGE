"""M1-to-M3 gain contract only; no attention editing or tree traversal here."""

from collections.abc import Mapping
import math

from .action_blind import NONE, _number, _text


def rule_gains(group_result, *, g_plus=1.0, g_minus=1.0, eligible=True):
    """Choose signed, fixed gains for one eligible mutually exclusive group.

    The caller must determine reachability from the actual tree. Set eligible=False
    for a group whose ancestor path is unconfirmed. This function does not infer
    descendant suppression. Zero gains mean this group has no direct edit; they do
    not imply whole-example fallback if some other group remains active.
    """
    g_plus = _number(g_plus, "g_plus", minimum=0)
    g_minus = _number(g_minus, "g_minus", minimum=0)
    if not isinstance(eligible, bool):
        raise ValueError("eligible must be a boolean reachability decision")
    if not isinstance(group_result, Mapping):
        raise ValueError("group_result must be an object")
    probabilities = group_result.get("probabilities")
    if not isinstance(probabilities, Mapping) or NONE not in probabilities or len(probabilities) < 2:
        raise ValueError("probabilities must contain rules and NONE")
    probs = {_text(k, "rule_id"): _number(v, "probability", minimum=0, maximum=1)
             for k, v in probabilities.items()}
    if not math.isclose(sum(probs.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Probabilities must sum to one")
    prediction, abstain = group_result.get("prediction"), group_result.get("abstain")
    _text(prediction, "prediction")
    if prediction not in probs or not isinstance(abstain, bool):
        raise ValueError("A valid prediction and boolean abstain are required")
    if probs[prediction] != max(probs.values()):
        raise ValueError("prediction must remain the probability argmax")
    ranked = sorted(probs.values(), reverse=True)
    summaries = {"probability": ranked[0], "margin": ranked[0] - ranked[1],
                 "entropy": -math.fsum(p * math.log(p) for p in probs.values() if p > 0)}
    for field, expected in summaries.items():
        if field in group_result:
            observed = _number(group_result[field], field, minimum=0)
            if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(f"{field} disagrees with probabilities")
    allowed = not abstain and prediction != NONE
    if group_result.get("intervention_allowed", allowed) is not allowed:
        raise ValueError("intervention_allowed disagrees with prediction/abstain")
    if not abstain and sum(p == probs[prediction] for p in probs.values()) > 1:
        raise ValueError("Tied predictions must abstain")
    if not eligible or not allowed:
        return {rule: 0.0 for rule in probs if rule != NONE}
    return {rule: g_plus if rule == prediction else -g_minus
            for rule in probs if rule != NONE}
