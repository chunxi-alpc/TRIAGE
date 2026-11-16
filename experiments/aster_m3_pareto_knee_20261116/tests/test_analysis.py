"""Geometric edge cases and integration against the existing PR76 CSVs."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.aster_m3_pareto_knee_20261116 import analyze as a


def point(name, x, y):
    return {"id": name, a.X: x, a.Y: y, "eta": 1., "regularization": 100.}


class GeometryTests(unittest.TestCase):
    def test_strict_dominance_and_exact_duplicates(self):
        rows = [point("origin", 0, 0), point("a", 1, 3), point("copy", 1, 3),
                point("same_x_worse_y", 1, 2), point("same_y_worse_x", 2, 3),
                point("tip", 3, 4), point("dominated", 4, 1)]
        self.assertEqual([r["id"] for r in a.pareto_front(rows)],
                         ["origin", "a", "copy", "tip"])

    def test_known_perpendicular_distance(self):
        front, chosen = a.analyze_grid(
            [point("lo", 2, 4), point("knee", 4, 11), point("hi", 12, 14)])
        knee = next(r for r in front if r["id"] == "knee")
        self.assertAlmostEqual(knee["normalized_cf_distortion"], .2)
        self.assertAlmostEqual(knee["normalized_ap_vr"], .7)
        self.assertAlmostEqual(knee["distance_A"], 1 / math.sqrt(8))
        self.assertAlmostEqual(knee["balance_score_B"], .5)
        self.assertEqual(chosen["method_A"]["id"], "knee")
        self.assertEqual(chosen["method_B"]["id"], "knee")

    def test_affine_rescaling_does_not_change_choice(self):
        rows = [point("lo", 0, 0), point("knee", 2, 7), point("hi", 10, 10)]
        transformed = [dict(r, **{a.X: 300 + 17*r[a.X], a.Y: -91 + .03*r[a.Y]}) for r in rows]
        _, before = a.analyze_grid(rows)
        _, after = a.analyze_grid(transformed)
        self.assertEqual(before["method_A"]["id"], after["method_A"]["id"])
        self.assertAlmostEqual(before["method_A"]["distance_A"], after["method_A"]["distance_A"])

    def test_unsigned_A_and_signed_B_can_disagree(self):
        rows = [point("lo", 0, 0), point("below", .3, .1),
                point("above", .7, .8), point("hi", 1, 1)]
        _, selection = a.analyze_grid(rows)
        self.assertEqual(selection["method_A"]["id"], "below")
        self.assertEqual(selection["method_B"]["id"], "above")
        self.assertFalse(selection["methods_agree"])
        self.assertFalse(selection["all_on_or_above_chord"])

    def test_numerical_tie_prefers_lower_distortion(self):
        rows = [point("lo", 0, 0), point("early", .2, .5),
                point("late", .6, .9), point("hi", 1, 1)]
        _, selection = a.analyze_grid(rows)
        self.assertEqual(selection["method_A"]["id"], "early")
        self.assertEqual(set(selection["method_A"]["tied_ids"]), {"early", "late"})

    def test_degenerate_front_has_no_interior_knee(self):
        cases = [
            [point("a", 1, 2)],
            [point("a", 1, 2), point("b", 1, 2)],
            [point("a", 0, 0), point("b", 1, 1)],
            [point("a", 0, 0), point("b", .3, .3), point("c", 1, 1)],
            [point("a", 0, 1), point("b", 1, 1)],
            [point("a", 1, 0), point("b", 1, 1)],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                _, selection = a.analyze_grid(rows)
                self.assertFalse(selection["informative_knee"])
                self.assertIsNone(selection["method_A"])
                self.assertIsNone(selection["method_B"])

    def test_empty_or_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            a.pareto_front([])
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                a.pareto_front([point("bad", value, 1)])

    def test_input_order_and_unrelated_metrics_do_not_select(self):
        rows = [point("lo", 0, 0), point("knee", .2, .7), point("hi", 1, 1)]
        _, expected = a.analyze_grid(rows)
        alternate = [dict(r, natural_path_accuracy=1-r[a.Y], test_accuracy=999) for r in reversed(rows)]
        _, actual = a.analyze_grid(alternate)
        self.assertEqual(actual, expected)

    def test_dominated_outlier_cannot_change_normalization(self):
        rows = [point("lo", 0, 0), point("knee", .2, .7), point("hi", 1, 1)]
        _, expected = a.analyze_grid(rows)
        _, actual = a.analyze_grid(rows + [point("outlier", 1000, -1000)])
        self.assertEqual(actual, expected)

    def test_input_is_not_mutated(self):
        rows = [point("lo", 0, 0), point("knee", .2, .7), point("hi", 1, 1)]
        original = copy.deepcopy(rows)
        a.analyze_grid(rows)
        self.assertEqual(rows, original)

    def test_source_output_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            for output in (source, source / "nested", Path(tmp)):
                with self.assertRaises(ValueError):
                    a.protected_output(source, output)
            self.assertEqual(a.protected_output(source, Path(tmp)/"other"), Path(tmp)/"other")


SOURCE = os.environ.get("M3_PR76_SOURCE")


@unittest.skipUnless(SOURCE, "Set M3_PR76_SOURCE to the existing PR76 report directory.")
class ExistingResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(SOURCE)
        cls.rows = a.read_csv(cls.source / "oof_grid_70.csv")
        cls.front, cls.selection = a.analyze_grid(cls.rows)
        cls.prior = json.loads((cls.source / "selection.json").read_text())
        cls.evaluations = {
            scope: {r["method"]: r for r in a.read_csv(cls.source/f"evaluation/{scope}/metrics.csv")}
            for scope in ("oof_selection", "dev", "test")
        }

    def test_known_grid_selection_and_front(self):
        self.assertEqual(len(self.rows), 70)
        self.assertEqual(len(self.front), 59)
        self.assertEqual(self.selection["method_A"]["id"], "e05_l06")
        self.assertEqual(self.selection["method_B"]["id"], "e05_l06")
        self.assertAlmostEqual(self.selection["method_A"]["distance_A"], .14488038246226825)
        self.assertEqual({r["id"] for r in self.front},
                         {r["id"] for r in a.read_csv(self.source/"pareto_front.csv")})
        for r in self.front:
            self.assertAlmostEqual(r["distance_A"], r["balance_score_B"]/math.sqrt(2))

    def test_key_points_and_unevaluated_values(self):
        keys, evaluated = a.collect_key_points(
            self.rows, self.front, self.selection, self.prior, self.evaluations)
        self.assertEqual(len(keys), 6)
        self.assertEqual(len(evaluated), 18)
        for row in evaluated:
            if row["scope"] in ("dev", "test") and row["role"] in ("knee_A", "knee_B", "conservative"):
                self.assertEqual(row["status"], "not_evaluated")
                self.assertTrue(all(row[m] is None for m in a.METRICS))
            else:
                self.assertEqual(row["status"], "evaluated")
        auto = next(r for r in keys if r["role"] == "auto_eta0")
        aggressive = next(r for r in keys if r["role"] == "aggressive")
        self.assertEqual(auto["candidate_id"], aggressive["candidate_id"])

    def test_existing_intervals_are_copied(self):
        keys, _ = a.collect_key_points(
            self.rows, self.front, self.selection, self.prior, self.evaluations)
        cis = a.reused_bootstrap(self.source, keys, self.prior)
        ci = next(r for r in cis if r["role"] == "knee_A"
                  and r["scope"] == "oof_selection" and r["metric"] == a.Y)
        self.assertEqual(ci["defined_draws"], 2000)
        self.assertAlmostEqual(ci["low"], .040389059348875045)
        self.assertAlmostEqual(ci["high"], .1570125040572212)

    def test_dev_holdout_changes_cannot_change_knee(self):
        original_read = a.read_csv
        original_analyze = a.analyze_grid
        selection_finished = []
        def record_selection(rows):
            result = original_analyze(rows)
            selection_finished.append(True)
            return result
        def changed_read(path):
            result = original_read(path)
            if "/evaluation/dev/" in str(path) or "/evaluation/test/" in str(path):
                self.assertTrue(selection_finished, "Evaluation was read before knee selection")
                for row in result:
                    for metric in a.METRICS:
                        if metric in row:
                            row[metric] = -999.
            return result
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(a, "analyze_grid", side_effect=record_selection), \
                 patch.object(a, "read_csv", side_effect=changed_read), \
                 patch.object(a, "make_figures"), patch.object(a, "make_report"):
                a.run(self.source, Path(tmp)/"output")
            saved = json.loads((Path(tmp)/"output/knee_selection.json").read_text())
            self.assertEqual(saved["method_A"], self.selection["method_A"])
            self.assertEqual(saved["method_B"], self.selection["method_B"])
            self.assertFalse(saved["dev_test_used_for_selection"])
            key_csv = a.read_csv(Path(tmp)/"output/key_points_evaluation.csv")
            missing = [r for r in key_csv if r["role"] == "knee_A" and r["scope"] == "test"][0]
            self.assertIsNone(missing[a.Y])
            self.assertEqual(missing["status"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
