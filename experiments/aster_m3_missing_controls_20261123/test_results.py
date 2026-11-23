"""Integration checks on the released OOF scores and source-level path counts."""
import csv
import os
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import SourceBootstrap
from .replay_a import archived_scores

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("ASTER_CONTROLS_OUTPUT", ROOT / "reports/aster_m3_missing_controls_20261123"))
ARCHIVE = ROOT / "reports/aster_m3_joint_study_20261115"


def rows(name, delimiter=","):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


class ControlsReplayTests(unittest.TestCase):
    def test_zero_uses_full_correction_with_no_intercept(self):
        row = {"view_id": "x", "raw_score": -0.75, "correction": {"J": 0.5},
               "uncalibrated_score": {"J": -0.25}}
        values = archived_scores(row, {"b_shared": -0.5, "b_independent": 0.25})
        self.assertEqual(values, {"R0": -0.75, "R1": -1.25, "J_zero": -0.25,
                                  "J_shared": -0.75, "J_independent": 0.0})
        checks = json.loads((OUT / "A_verification.json").read_text())
        self.assertEqual(checks["zero_reconstruction_max_abs_gap"], 0)
        self.assertEqual(checks["archive_path_mismatches"], 0)
        self.assertEqual(checks["compared_archived_path_records"], 4 * (1712 + 5136))
        check = checks["pair_intercept_invariance"]
        self.assertEqual(check["pair_counts"], {"AP": 5003, "CF": 1342})
        self.assertLess(max(check["max_abs_difference"].values()), 1e-12)

    def test_registered_folds_and_calibration_sets(self):
        folds = json.loads((ARCHIVE / "folds.json").read_text())["folds"]
        held_out = []
        for fold in folds:
            path = OUT / "b_models" / f"fold{fold['fold']}"
            choice = json.loads((path / "selection.json").read_text())
            train, valid = set(fold["train_sources"]), set(fold["validation_sources"])
            small, tuning = set(choice["inner_train_sources"]), set(choice["inner_validation_sources"])
            self.assertEqual((len(train), len(valid), len(small), len(tuning)), (64, 16, 48, 16))
            self.assertTrue(train.isdisjoint(valid) and small.isdisjoint(tuning))
            self.assertEqual(train, small | tuning)
            held_out.extend(valid)
            parent = json.loads((ARCHIVE / "models" / path.name / "J.json").read_text())
            for method in ("Direction", "Asymmetric"):
                model = json.loads((path / (method + ".json")).read_text())
                self.assertEqual(model["calibration_ids"], parent["calibration_ids"])
                self.assertEqual(set(model["train_sources"]), train)
                self.assertEqual(model["lambda_ap"], parent["lambda_ap"])
        self.assertEqual(len(held_out), len(set(held_out)))
        self.assertEqual(len(held_out), 80)

    def test_common_source_intervals_from_completed_transitions(self):
        # Reconstruction from per-source four-way counts checks pairing and the
        # pooled path denominator for each bootstrap draw, including interaction.
        folds = json.loads((ARCHIVE / "folds.json").read_text())
        boot = SourceBootstrap(folds["all_sources"], draws=2000)
        groups = defaultdict(list)
        for prefix, file, sep in (("A", "A_source_transitions.tsv", "\t"),
                                  ("B", "b_source_transitions.csv", ",")):
            for r in rows(file, sep):
                level = r["level"]
                if level not in ("natural_path", "interference_path"):
                    continue
                comparison = r.get("comparison") or r["lhs"] + "-" + r["rhs"]
                key = (prefix, r.get("calibration", ""), level, comparison)
                groups[key].append(r)
        draws = {}
        for key, entries in groups.items():
            for r in entries:
                flows = [int(r[name]) for name in ("wrong_to_right", "right_to_wrong",
                                                   "right_to_right", "wrong_to_wrong")]
                self.assertEqual(sum(flows), int(r["total"]))
                self.assertEqual(flows[0] - flows[1], int(r["net"]))
            src = [r["source_id"] for r in entries]
            self.assertCountEqual(src, boot.sources)
            denom = boot.sums([int(r["total"]) for r in entries], src)
            draws[key] = boot.sums([int(r["net"]) for r in entries], src) / denom
        for cal in ("shared", "independent"):
            for metric in ("natural_path", "interference_path"):
                key = ("B", cal, metric)
                draws[key + ("interaction",)] = draws[key + ("J2-Direction",)] - draws[key + ("Asymmetric-J",)]
        archived = {}
        for r in rows("A_comparisons.tsv", "\t"):
            archived["A", "", r["level"], r["lhs"]+"-"+r["rhs"]] = r
        for r in rows("b_paired_intervals.csv"):
            archived["B", r["calibration"], r["metric"].removesuffix("_accuracy"), r["comparison"]] = r
        for key, samples in draws.items():
            summary = boot.summary(samples)
            expected = archived[key]
            np.testing.assert_allclose([summary[k] for k in ("estimate", "low", "high")],
                                       [float(expected[k]) for k in ("estimate", "low", "high")],
                                       atol=2e-14, rtol=0)
            self.assertEqual(int(expected["defined_draws"]), 2000)

    def test_archived_cells_replay_every_path(self):
        checks = json.loads((OUT / "b_evaluation_checks.json").read_text())
        for method in ("J", "J2"):
            result = checks["archive_replay"][method]
            self.assertTrue(result["all_paths_identical"])
            self.assertEqual(result["archived_path_records_compared"], 2 * (1712 + 5136))
        self.assertEqual((checks["draws"], checks["seed"]), (2000, 20261111))

