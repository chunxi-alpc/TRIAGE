"""Small CPU tests of evaluation semantics, source resampling and fit/dev isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .evaluation import (evaluate, fit_fusions, path_metrics, predict_fusion,
                         signal_probe, source_bootstrap)
from .report import render_report


def fixture():
    views, pairs, full, direct = {}, [], {}, {}
    for split in ("fit", "dev", "test"):
        for source in range(2):
            group = f"{split}-{source}"
            common = dict(group_id=group, source_id=group, split=split, cohort="main")
            for variant, label in (("base0", 0), ("base1", 1), ("swap0", 0), ("swap1", 1),
                                   ("condition0", 1), ("clinical0", 0), ("preference0", 0)):
                vid = group + ":" + variant
                views[vid] = dict(common, view_id=vid, variant=variant, label=label,
                                  original=variant in ("base0", "base1"), prompt="unused")
                full[vid] = 2.0 if label else -2.0
                direct[vid] = 1.0 if label else -1.0
            for kind, left, right, subtype in (
                ("A", "base0", "swap0", ""), ("A", "base1", "swap1", ""),
                ("C", "base0", "condition0", ""), ("F", "base0", "base1", ""),
                ("P", "base0", "clinical0", "clinical"),
                ("P", "base0", "preference0", "preference"),
            ):
                pairs.append(dict(common, pair_id=group+":"+kind+":"+right, kind=kind,
                                  subtype=subtype, left=group+":"+left, right=group+":"+right))
    return views, pairs, {"full": full, "direct": direct, "semantic": direct.copy()}


class EvaluationTests(unittest.TestCase):
    def test_missing_scores_stay_in_denominator_and_tie_is_no(self):
        views, pairs, scores = fixture()
        del scores["full"]["test-0:swap1"]
        scores["full"]["test-0:base0"] = 0.0
        result = evaluate(views, pairs, scores, bootstrap=0)["methods"]["full"]
        self.assertEqual(result["PA_A"]["denominator"], 4)
        self.assertEqual(result["PA_A"]["numerator"], 3)
        self.assertEqual(result["PA_A"]["missing_score_pairs"], 1)
        self.assertEqual(result["score_coverage"]["missing"], 1)
        self.assertEqual(result["original_accuracy"]["rate"], 1)
        self.assertEqual(result["historical_fourway"]["rate"], .5)

    def test_p_subtypes_have_equal_weight_and_missing_type_is_not_zero(self):
        views, pairs, scores = fixture()
        scores["full"]["test-0:preference0"] = 4.0
        scores["full"]["test-1:preference0"] = 4.0
        clinical = [p for p in pairs if p["split"] == "test" and p["subtype"] == "clinical"]
        pairs.extend([dict(p, pair_id=p["pair_id"]+":repeat") for p in clinical])
        result = evaluate(views, pairs, scores, bootstrap=0)["methods"]["full"]
        self.assertEqual(result["PA_P"]["rate"], .5)
        self.assertAlmostEqual(result["PA_P"]["pooled_rate"], 2/3)
        filtered = [p for p in pairs if p["kind"] != "C"]
        missing = evaluate(views, filtered, scores, bootstrap=0)["methods"]["full"]
        self.assertIsNone(missing["macro_PA"])
        self.assertEqual(missing["missing_pair_types"], ["C"])

    def test_same_node_double_wrong_is_not_path_or_union_error(self):
        views, pairs, scores = fixture()
        candidate = scores["full"].copy()
        scores["full"]["test-0:base0"] = 5
        scores["direct"]["test-0:base0"] = 4
        scores["direct"]["test-0:base1"] = -4
        del scores["direct"]["test-1:base0"]
        scores["aster"] = candidate
        result = evaluate(views, pairs, scores, bootstrap=0)["methods"]["aster"]
        d = result["double_wrong_correction"]["direct"]
        self.assertEqual((d["numerator"], d["denominator"]), (1, 1))
        self.assertEqual(d["baseline_missing"], 1)
        self.assertEqual(result["change_vs_full"]["corrected"]["numerator"], 1)

    def test_source_bootstrap_resamples_whole_source_with_shared_differences(self):
        views, pairs, scores = fixture()
        for vid, v in views.items():
            if v["split"] == "test" and v["source_id"] == "test-0":
                scores["direct"][vid] *= -1
        result = evaluate(views, pairs, scores, bootstrap=500, seed=7)
        intervals = result["confidence_intervals"]
        self.assertEqual(intervals["n_sources"], 2)
        self.assertEqual(intervals["methods"]["full"]["macro_PA"]["low"], 1)
        delta = intervals["methods"]["direct"]["differences"]["full"]["macro_PA"]
        self.assertEqual(delta["low"], -1)
        self.assertEqual(delta["high"], 0)
        self.assertEqual(intervals, evaluate(views, pairs, scores, bootstrap=500, seed=7)["confidence_intervals"])

    def test_profile_difference_in_differences_measured_from_scores(self):
        views, pairs, scores = fixture()
        for v in ("clinical0", "preference0"):
            scores["full"]["test-0:"+v] = -1
            scores["direct"]["test-0:"+v] = 1
        result = evaluate(views, pairs, scores, bootstrap=0)["profile_response"]["direct"]["all"]
        self.assertEqual(result["paired_scored"], 4)
        self.assertEqual(result["difference_in_differences"]["mean"], -.5)
        self.assertEqual(result["difference_in_differences"]["mean_abs"], .5)

    def test_cad_directions_probability_wrong_pair_and_ties(self):
        self.assertEqual(predict_fusion({"family": "cad", "alpha": 2}, 3, 1), 7)
        self.assertEqual(predict_fusion({"family": "reverse_cad", "alpha": 2}, 3, 1), -3)
        for w in (0, .25, .5, .75, 1):
            self.assertLess(predict_fusion({"family": "probability", "weight": w}, -1, -3), 0)
        self.assertEqual(predict_fusion({"family": "probability", "weight": .5}, 0, 0), 0)
        np.testing.assert_allclose(predict_fusion({"family": "logit", "weight": .25},
                                                 np.array([2., -2.]), np.array([4., 8.])), [3.5, 5.5])

    def test_fusion_fit_and_selection_do_not_use_test_labels_or_scores(self):
        views, pairs, scores = fixture()
        first = fit_fusions(views, pairs, scores)
        changed_views, changed_scores = copy.deepcopy(views), copy.deepcopy(scores)
        for vid, v in changed_views.items():
            if v["split"] == "test":
                v["label"] = 1-v["label"]
                for method in changed_scores:
                    changed_scores[method][vid] *= -10
        second = fit_fusions(changed_views, pairs, changed_scores)
        self.assertEqual(first["specs"], second["specs"])
        self.assertEqual(first["selection"], second["selection"])
        self.assertEqual(first["best_method"], second["best_method"])
        self.assertIn("cad_direct", first["specs"])
        self.assertIn("reverse_cad_semantic", first["specs"])
        self.assertEqual(first["specs"]["logistic_direct"]["training"]["C"], 1.0)
        self.assertTrue(all(s.startswith("fit-") for s in
                            first["specs"]["logistic_direct"]["training"]["sources"]))
        json.dumps(first, allow_nan=False)

    def test_probe_is_fit_only_and_evaluates_all_full_errors_with_original_subset(self):
        import torch
        views, pairs, scores = fixture()
        features = {}
        for vid, v in views.items():
            features[vid] = {"heads": torch.tensor([[[2.0*v["label"]-1, .25]]])}
        operators = {0: {"S": torch.tensor([[1.], [0.]]), "mu": torch.zeros(2)}}
        scores["full"]["test-0:base0"] = 3
        scores["full"]["test-0:base1"] = -3
        scores["full"]["test-1:swap0"] = 3
        result = signal_probe(views, features, operators, scores["full"])
        self.assertEqual(result["full_error_count"], 3)
        self.assertEqual(result["original_full_error_count"], 2)
        self.assertEqual(result["accuracy"]["denominator"], 3)
        self.assertEqual(result["original_error_accuracy"]["denominator"], 2)
        self.assertEqual(result["accuracy"]["rate"], 1)
        self.assertEqual(result["original_error_accuracy"]["rate"], 1)
        self.assertEqual(result["coordinate_dimension"], 1)
        self.assertFalse(result["online_use"])

    def test_path_missing_method_case_is_a_failure(self):
        rows = [{"method": "full", "case_id": "a", "source_id": "s", "correct": True},
                {"method": "full", "case_id": "b", "source_id": "s", "correct": False},
                {"method": "aster", "case_id": "a", "source_id": "s", "correct": True}]
        result = path_metrics(rows, n_resamples=50)
        self.assertEqual(result["methods"]["aster"]["denominator"], 2)
        self.assertEqual(result["methods"]["aster"]["missing_or_failed"], 1)
        self.assertEqual(result["methods"]["aster"]["rate"], .5)

    def test_report_marks_pending_and_preserves_automatic_status(self):
        with tempfile.TemporaryDirectory() as d:
            p = render_report(d)
            text = Path(p).read_text(encoding="utf-8")
            self.assertIn("自动预实验", text)
            self.assertIn("主测试结果尚未完成", text)
            views, pairs, scores = fixture()
            p = render_report(d, {"main_test": evaluate(views, pairs, scores, bootstrap=10)})
            text = Path(p).read_text(encoding="utf-8")
            self.assertIn("PA_P 临床", text)
            self.assertIn("医生核验尚未完成", text)
            self.assertIn("历史四输入全对", text)



    def test_report_reads_actual_mechanism_artifact_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/"results").mkdir(); (root/"review").mkdir()
            def save(name, value):
                (root/"results"/name).write_text(json.dumps(value), encoding="utf-8")
            save("h1.json", {
                "A": {"H": [2], "selected_k": 1, "selected_recovery": .85, "target_met": True,
                      "curve": [{"k": 0, "recovery": 0}, {"k": 1, "recovery": .85}]},
                "P": {"H": [], "selected_k": 0, "target_met": False, "curve": []},
                "interventions": {"test": [{"cohort": "main", "kind": "A", "control": "selected_union",
                                           "count": 3, "recovery": .6}]}})
            save("h2.json", {"number_of_heads": 1, "total_edit_rank": 1, "heads": {},
                "interventions": {"test": {"nodes": 20, "pairs": 3, "effects": [
                    {"kind": "A", "head": "joint", "operation": "nuisance", "count": 3, "recovery": .4}]}}})
            save("h3.json", {"probe_main": {"status": "complete", "fit_inputs": 10,
                  "coordinate_dimension": 2, "original_full_error_count": 4, "scored_errors": 4,
                  "accuracy": {"numerator": 3, "denominator": 4, "rate": .75}},
                  "amplification": {"amplification": 1.25, "development": {}}})
            save("local_theory.json", {"test": {"nodes": 20, "mean_abs_nonlinear_error": .1,
                                               "mean_abs_cross_layer_change": .2,
                                               "mean_abs_total_prediction_error": .3}})
            save("operator_checks.json", {"2": {"signal_preservation_error": 1e-7,
                  "erasure_error": 2e-7, "idempotence_error": 1e-7,
                  "edit_range_error": 1e-7, "edit_operator_norm": 2,
                  "angle_bound_satisfied": True}})
            save("fusion.json", {"best_method": "cad_direct"})
            (root/"review/physician_review.jsonl").write_text("{}\n")
            text = Path(render_report(root)).read_text(encoding="utf-8")
            self.assertIn("selected_union", text)
            self.assertIn("nuisance", text)
            self.assertIn("3/4 (75.0%)", text)
            self.assertIn("review/physician_review.jsonl", text)
            self.assertIn("results/fusion.json", text)

    def test_completion_status_and_conditional_scores_preserve_primary_denominators(self):
        from .report import coverage_status
        views, pairs, scores = fixture()
        scores["boundary_guideline"] = scores["full"].copy()
        for vid, view in views.items():
            if view["split"] == "test" and view["source_id"] == "test-0":
                del scores["boundary_guideline"][vid]
        evaluations = {"main_test": evaluate(views, pairs, scores, bootstrap=0),
                       "definition_test": evaluate(views, pairs, scores, cohort="definition", bootstrap=0)}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for folder in ("data", "results", "scores"):
                (root/folder).mkdir()
            (root/"data/views.jsonl").write_text("".join(json.dumps(v)+"\n" for v in views.values()))
            records = [dict(method=m, view_id=vid, ell=ell, status="complete")
                       for m, current in scores.items() for vid, ell in current.items()]
            (root/"scores/baseline_0.jsonl").write_text("".join(json.dumps(v)+"\n" for v in records))
            (root/"results/status.json").write_text(json.dumps({"status": "complete"}))
            (root/"results/path_results.json").write_text(json.dumps({"methods": {
                "semantic": {"numerator": 3, "denominator": 4, "rate": .75,
                             "completed": 3, "missing_or_failed": 1}}}))
            status = coverage_status(root, evaluations)
            self.assertEqual(status["pipeline_state"], "complete")
            self.assertEqual(status["experiment_state"], "complete_with_gaps")
            self.assertEqual({x["code"] for x in status["issues"]},
                             {"definition_subset_empty", "node_scores_missing", "path_runs_failed"})
            before = copy.deepcopy(evaluations)
            text = Path(render_report(root, evaluations)).read_text(encoding="utf-8")
            self.assertEqual(before, evaluations)
            self.assertIn("计算流程：已结束；计划实验：存在未完成内容", text)
            self.assertIn("定义测试子集为 0/20", text)
            self.assertIn("2/4 (50.0%)", text)
            self.assertIn("2/2 (100.0%)", text)
            self.assertIn("3/4 (75.0%)", text)
            self.assertIn("3/3 (100.0%)", text)
            self.assertIn("不能把缺分造成的低分解释为纯判断退化", text)

class AutonomousPathTests(unittest.TestCase):
    @staticmethod
    def case():
        def rules(i, yes, no):
            return dict(group_id="G"+str(i), rules=[
                dict(rule_id=f"R{i}Y", condition="condition "+str(i), action=yes),
                dict(rule_id=f"R{i}N", condition="condition "+str(i), action=no)])
        guide = {"groups": [rules(0, "continue G1", "continue G2"),
                            rules(1, "L3:correct action", "L4:alternative action"),
                            rules(2, "L5:wrong-branch action", "L6:other action")]}
        return dict(case_id="case1", source_id="source1", guideline=guide, patient="original patient",
                    definitions=[], gold_path=[dict(gid="G0", label=1), dict(gid="G1", label=1)],
                    gold_leaf="L3")

    @staticmethod
    def masks(case):
        from .paths import enumerate_contexts
        return {c["key"]: {"keep_action_ids": [], "keep_definition_indices": []}
                for c in enumerate_contexts([case])}

    @staticmethod
    def fake_prompt(guide, definitions, patient, gid, mode="full", semantic=None):
        return gid+"|"+mode+"|"+patient

    class Engine:
        def __init__(self):
            self.calls = []
            self.counts = {"prefills": 0, "final_forwards": 0}
        def prepare(self, text):
            self.counts["prefills"] += 1
            return text
        def run(self, context, operators=None, capture=True):
            self.counts["final_forwards"] += 1
            gid, mode, patient = context.split("|")
            edited = operators is not None
            self.calls.append((gid, mode, edited, patient))
            score = 2 if gid != "G0" or mode == "direct" or edited else -2
            return {"ell": score}

    def test_wrong_root_follows_wrong_child_without_gold_guidance_and_resumes(self):
        from unittest.mock import patch
        from .paths import run_paths
        case, engine = self.case(), self.Engine()
        spec = dict(family="logit", weight=.5, masked_route="direct")
        with tempfile.TemporaryDirectory() as d, patch(
                "experiments.aster_closed_form.paths.prompt", self.fake_prompt):
            records = run_paths(engine, [case], {"marker": {}}, spec, self.masks(case), d)
            by_method = {r["method"]: r for r in records}
            self.assertFalse(by_method["full"]["correct"])
            self.assertEqual(by_method["full"]["predicted_path"],
                             [{"gid": "G0", "label": 0}, {"gid": "G2", "label": 1}])
            self.assertEqual(by_method["full"]["returned_action"], "wrong-branch action")
            self.assertTrue(by_method["aster"]["correct"])
            self.assertTrue(by_method["direct"]["correct"])
            self.assertFalse(by_method["best_output"]["correct"])
            # The tie at root takes no and then the wrong branch autonomously.
            self.assertEqual(by_method["best_output"]["predictions"][0]["ell"], 0)
            self.assertEqual(sum(g == "G0" and m == "full" and not e
                                 for g, m, e, p in engine.calls), 1)
            resumed = self.Engine()
            again = run_paths(resumed, [case], {"marker": {}}, spec, self.masks(case), d)
            self.assertEqual(again, records)
            self.assertEqual(resumed.calls, [])

    def test_mask_requests_exclude_patient_and_gold_and_missing_mask_is_failure(self):
        from unittest.mock import patch
        from .paths import enumerate_contexts, run_paths
        case, engine = self.case(), self.Engine()
        contexts = enumerate_contexts([case, case])
        self.assertEqual(len(contexts), 3)
        self.assertTrue(all("patient" not in c and "gold_path" not in c and "gold_leaf" not in c
                            for c in contexts))
        spec = dict(family="calibration", route="full", coefficients=[1], intercept=0,
                    masked_route="full")
        with tempfile.TemporaryDirectory() as d, patch(
                "experiments.aster_closed_form.paths.prompt", self.fake_prompt):
            records = run_paths(engine, [case], {}, spec, {}, d)
        by_method = {r["method"]: r for r in records}
        self.assertEqual(by_method["semantic"]["status"], "failed")
        self.assertIn("mask", by_method["semantic"]["error"])
        self.assertEqual(by_method["full"]["status"], "complete")
        self.assertEqual(by_method["best_output"]["status"], "complete")

if __name__ == "__main__":
    unittest.main()
