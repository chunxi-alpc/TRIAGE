"""Question-aware rule/patient relation modeler. No training or reference inputs."""
import json
from pathlib import Path
from .backend import LocalBackend, ROOT


class Module1:
    def __init__(self, config_path=None, device="cuda:0", top_k=12, cache_dir=None, backend=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.backend = backend or LocalBackend(config_path=config_path, device=device)
        self.top_k = top_k
        self.cache_dir = Path(cache_dir or ROOT / "results/module1_parse_cache")

    def _decision(self, patient, question, rules=None, applicability=None, relations=None):
        system = ("You are a clinical decision model. Return only a concise JSON object with "
                  "decision (a concrete recommendation), actions (list of strings), "
                  "status (decided or insufficient_evidence), and rationale (one brief sentence). "
                  "Do not invent patient evidence. If a decision depends on missing evidence, say so.")
        context = {"patient": patient, "question": question}
        if rules is None:
            system += " Decide using your existing knowledge; no current guideline is supplied."
        else:
            system += (" Use the supplied current guideline rules and their patient applicability. "
                       "Do not activate inapplicable rules. Preserve uncertainty for unknown applicability.")
            by_id = {row["rule_id"]: row for row in applicability or []}
            active = [rule for rule in rules if by_id.get(rule["rule_id"], {}).get("status") == "applicable"]
            context["current_guideline_rules"] = active
            context["applicability"] = applicability
            context["condition_evidence"] = [{key: row.get(key) for key in
                ("rule_id", "condition_id", "condition", "status", "evidence")}
                for row in relations or []]
            system += (" Only applicable rule actions may be recommended as current guideline decisions. "
                "Check each recommendation against the explicit patient evidence, including negative evidence and contraindications. "
                "Do not substitute an unknown or blocked branch. Keep every qualification of an applicable action. "
                "If no current rule is applicable, report insufficient_evidence and identify the missing or blocking condition.")
            if not active:
                return {"decision": "Insufficient evidence for an applicable current guideline recommendation",
                    "actions": [], "status": "insufficient_evidence",
                    "rationale": "No retrieved rule has all required conditions established and exceptions excluded."}
        obj = self.backend.json(system, json.dumps(context, ensure_ascii=False), max_new_tokens=640)
        if not isinstance(obj.get("decision"), str) or not isinstance(obj.get("actions"), list):
            raise ValueError("Decision response requires decision string and actions list")
        if obj.get("status") not in {"decided", "insufficient_evidence"}:
            obj["status"] = "insufficient_evidence"
        return obj

    def _conflicts(self, prior, guided, rules):
        if not rules:
            return []
        system = ("Compare two independently obtained clinical decisions. Return JSON: "
                  "{\"rules\":[{\"rule_id\":string,\"prior_support\":\"support\"|\"oppose\"|\"unknown\","
                  "\"guided_support\":\"support\"|\"oppose\"|\"unknown\",\"reason\":string}]}. "
                  "For each rule action classify each decision support. Support means endorsing that action; "
                  "oppose means a materially incompatible action, not merely absent mention. Unknown covers "
                  "missing or ambiguous recommendations. Wording changes alone are not disagreement. Avoiding a drug also supports avoiding its standard dose; compatible qualifications or additional recommendations are not opposition. Only label oppose when the decision expressly recommends an incompatible action. "
                  "Use only the supplied decisions, not a reference answer. Keep reasons concise.")
        comparison = self.backend.json(system, json.dumps({"prior_decision": prior,
            "guideline_decision": guided, "rule_actions": [{"rule_id": r["rule_id"],
            "action": r.get("action", "")} for r in rules]}, ensure_ascii=False), max_new_tokens=max(1024, len(rules) * 128))
        rows = comparison.get("rules", [])
        by_id = {r.get("rule_id"): r for r in rows if isinstance(r, dict)} if isinstance(rows, list) else {}
        result = []
        for rule in rules:
            row = by_id.get(rule["rule_id"], {})
            left, right = row.get("prior_support", "unknown"), row.get("guided_support", "unknown")
            if left not in {"support", "oppose", "unknown"}: left = "unknown"
            if right not in {"support", "oppose", "unknown"}: right = "unknown"
            known = left != "unknown" and right != "unknown"
            result.append({"rule_id": rule["rule_id"], "score": float(left != right) if known else 0.5,
                "status": ("disagreement" if left != right else "agreement") if known else "unknown",
                "prior_support": left, "guided_support": right, "reason": row.get("reason", "Comparison unavailable"),
                "method": "Qwen action compatibility; disagreement=1, agreement=0, unknown=0.5"})
        return result

    def run(self, guideline, patient, question):
        """Return a JSON-serializable graph of Q-rule-condition-patient evidence relations.

        Inputs contain only guideline content, a patient note string, and a question.
        All scores/gates are heuristic weights, not calibrated probabilities.
        """
        import time
        from .rules import select_and_parse
        from .conditions import match_conditions
        if not isinstance(patient, str) or not patient.strip():
            raise ValueError("patient must be a nonempty note string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        started = time.time()
        call_start = len(getattr(self.backend, "calls", []))
        # Closed-book decision is generated before any current guideline is supplied to Qwen.
        prior = self._decision(patient, question)
        rules, relevance = select_and_parse(guideline, question, self.backend,
            top_k=self.top_k, cache_dir=self.cache_dir, patient=patient)
        relations, applicability, spans = match_conditions(rules, patient, self.backend)
        guided = self._decision(patient, question, rules, applicability, relations)
        conflicts = self._conflicts(prior, guided, rules)
        rel_by_id = {r["rule_id"]: float(r["score"]) for r in relevance}
        conflict_by_id = {r["rule_id"]: r["score"] for r in conflicts}
        applies = {r["rule_id"]: r for r in applicability}
        binding, action, weights = {}, {}, []
        for rule in rules:
            rid = rule["rule_id"]
            relevance_score = rel_by_id.get(rid, 0.0)
            conflict = conflict_by_id.get(rid, 0.5)
            base = relevance_score * (0.5 + 0.5 * conflict)
            app = applies.get(rid, {"status": "unknown"})
            action[rid] = round(base if app["status"] == "applicable" else 0.0, 6)
            decisive_ids = app.get("decisive_condition_ids", [])
            for row in relations:
                if row["rule_id"] != rid:
                    continue
                cid = row["condition_id"]
                known = row["status"] in {"satisfied", "not_satisfied"} and bool(row.get("evidence"))
                confidence = max(0.0, min(float(row.get("confidence", 1.0)), 1.0)) if known else 0.0
                gate = round(base * confidence, 6)
                binding.setdefault(rid, {})[cid] = gate
                decisive = cid in decisive_ids
                sign = 1 if row["status"] == "satisfied" else -1 if row["status"] == "not_satisfied" else 0
                weights.append({"rule_id": rid, "condition_id": cid, "question": question,
                    "condition": row["condition"], "role": row.get("role", "condition"),
                    "status": row["status"], "evidence": row.get("evidence", []),
                    "weight": gate, "signed_weight": round(sign * gate, 6), "decisive": decisive,
                    "binding_gate": gate, "action_gate": action[rid],
                    "prior_competition_weight": round(action[rid] * conflict, 6)})
        output = {"schema_version": "module1.v1", "question": question,
            "candidate_rules": rules, "prior_decision": prior, "guideline_decision": guided,
            "conflict_scores": conflicts, "relevance_scores": relevance,
            "condition_evidence_relations": relations, "applicability": applicability,
            "critical_spans": spans, "relation_weights": weights,
            "decisive_relations": [r for r in weights if r["decisive"]],
            "binding_gate": binding, "action_gate": action,
            "metadata": {"elapsed_seconds": round(time.time() - started, 3),
                "backend": "Qwen3-8B predicate/evidence verification + MedCPT-MedPair",
                "reference_answers_used": False, "scores_are_calibrated": False,
                "gate_formula": "binding=relevance*(0.5+0.5*conflict)*evidence_confidence; action=relevance*(0.5+0.5*conflict) iff applicable; prior_competition=action*conflict",
                "unknown_binding": 0, "unknown_action": 0,
                "model_calls": getattr(self.backend, "calls", [])[call_start:]}}
        # Enforce the downstream serialization contract, including finite scores.
        json.dumps(output, ensure_ascii=False, allow_nan=False)
        return output


_default_instance = None


def run(guideline, patient, question):
    """Unified convenient entry: from src import module1; module1.run(R, P, Q)."""
    global _default_instance
    if _default_instance is None:
        _default_instance = Module1()
    return _default_instance.run(guideline=guideline, patient=patient, question=question)
