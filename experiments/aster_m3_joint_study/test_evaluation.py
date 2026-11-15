"""Synthetic decoder/statistics checks; run on the designated experiment server."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy
from .evaluation import evaluate, selection_metrics, _success


def fixture():
    views, pairs, cases, path_views = [], [], [], []
    raw, adjusted = {}, {}
    for source in ("source-a", "source-b"):
        cid = "synthetic-m3-" + source
        ids = {name: cid + ":" + name for name in ("g1", "g2", "a1", "a2", "cf")}
        guide = {"groups": [
            {"group_id": "G1", "rules": [
                {"rule_id": "R1Y", "condition": "First condition", "action": "G2"},
                {"rule_id": "R1N", "condition": "First condition", "action": "L0: stop"}]},
            {"group_id": "G2", "rules": [
                {"rule_id": "R2Y", "condition": "Second condition", "action": "L1: yes"},
                {"rule_id": "R2N", "condition": "Second condition", "action": "L2: no"}]}]}
        cases.append(dict(case_id=cid, source_id=source, guideline=guide,
                          gold_path=[dict(gid="G1", label=1), dict(gid="G2", label=0)], gold_leaf="L2"))
        for name, gid, label, variant in (("g1", "G1", 1, "original"),
                                         ("g2", "G2", 0, "original"),
                                         ("a1", "G1", 1, "action_presence"),
                                         ("a2", "G2", 0, "action_presence")):
            view = dict(view_id=ids[name], source_id=source, case_id=cid, gid=gid,
                        label=label, variant=variant, kind="original" if variant == "original" else "A",
                        original=variant == "original")
            path_views.append(view)
            if name != "a2":
                views.append(view)
        views.append(dict(view_id=ids["cf"], source_id=source, label=0, variant="F", kind="F", original=False))
        pairs.extend([
            dict(pair_id=cid + ":AP", source_id=source, kind="A", left=ids["g1"], right=ids["a1"], left_label=1, right_label=1),
            dict(pair_id=cid + ":CF", source_id=source, kind="F", left=ids["g1"], right=ids["cf"], left_label=1, right_label=0)])
        raw.update({ids["g1"]: -0.7, ids["g2"]: -0.3, ids["a1"]: -0.6, ids["a2"]: 0.3, ids["cf"]: -1.0})
        adjusted.update({ids["g1"]: 0.2, ids["g2"]: -0.3, ids["a1"]: 0.1, ids["a2"]: -0.2, ids["cf"]: -0.1})
    scores = {name: dict(raw) for name in ("raw", "R0", "R1", "R2_shared", "R2_independent", "S_shared", "S_independent")}
    scores.update({name: dict(adjusted) for name in ("J_shared", "J_independent")})
    return views, pairs, cases, path_views, scores


class EvaluationTests(unittest.TestCase):
    def test_dynamic_paths_match_existing_scorer_and_follow_new_branch(self):
        data = fixture()
        expected = legacy.path_predictions(data[2], data[3], data[4])
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(*data, directory)
            actual = [json.loads(line) for line in (Path(directory) / "path_predictions.jsonl").read_text().splitlines()]
            self.assertEqual(expected, actual)
            self.assertEqual(result["models"]["R1"]["natural_path_correct"], 0)
            self.assertEqual(result["models"]["J_independent"]["natural_path_correct"], 2)
            changed = [r for r in actual if r["method"] == "J_independent" and not r["interference"]]
            self.assertTrue(all(len(r["path"]) == 2 and r["leaf"] == "L2" for r in changed))

    def test_selection_matches_final_point_estimates_and_first_error_margins(self):
        data = fixture()
        selected = selection_metrics(*data)
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(*data, directory)
            for name in selected:
                for metric in ("natural_path_accuracy", "interference_path_accuracy", "cf_correction_rms"):
                    self.assertEqual(selected[name][metric], result["models"][name][metric])
            rows = [json.loads(line) for line in (Path(directory) / "natural_first_error.jsonl").read_text().splitlines()]
            row = next(r for r in rows if r["comparison"] == "J_independent minus R1")
            self.assertTrue(row["first_error_repaired"])
            self.assertEqual(row["before_gold_margin"], -0.7)
            self.assertEqual(row["after_gold_margin"], 0.2)
            interval = result["comparisons"]["S_independent minus R1"]["metrics"]["natural_path_accuracy"]
            self.assertEqual(interval["defined_draws"], 2000)
            self.assertEqual((interval["estimate"], interval["low"], interval["high"]), (0.0, 0.0, 0.0))

    def test_fixed_sequence_stops_formal_second_comparison(self):
        comparisons = {
            "J_independent minus S_independent": {"metrics": {
                "natural_path_accuracy": {"low": 0.0}, "interference_path_accuracy": {"low": 0.0}}},
            "J_independent minus R1": {"metrics": {
                "natural_path_accuracy": {"low": 0.0}, "interference_path_accuracy": {"low": 0.1}}}}
        result = _success(comparisons)["sequence"]
        self.assertFalse(result[0]["passes_predeclared_sequence"])
        self.assertTrue(result[1]["both_conditions"])
        self.assertFalse(result[1]["formal_gate_open"])
        self.assertFalse(result[1]["passes_predeclared_sequence"])

    def test_decision_at_zero_remains_negative(self):
        data = fixture()
        for view in data[3]:
            if view["variant"] == "original" and view["gid"] == "G1":
                data[4]["J_independent"][view["view_id"]] = 0.0
        result = selection_metrics(*data)
        self.assertEqual(result["J_independent"]["natural_path_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
