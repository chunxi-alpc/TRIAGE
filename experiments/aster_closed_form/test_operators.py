"""Focused CPU checks for the closed-form maps and fit-data boundaries."""
import json
import math
import unittest
from collections import Counter

import torch

from .operators import (
    algebra_diagnostics, apply_operator, configuration_grid, energy_basis,
    fit_operators, layer_matched_random, leace_factors, oblique_factors, orth,
)

torch.set_num_threads(2)


def fixture():
    features, pairs = {}, []
    def add(pid, kind, left, right, effect=1.0, split="fit", subtype=""):
        for side, value, ell in (("l", left, 0.0), ("r", right, effect)):
            features[pid + side] = {
                "heads": torch.tensor(value, dtype=torch.float64).reshape(1, 1, -1),
                "ell": ell, "split": split, "original": side == "l",
            }
        pairs.append({"pair_id": pid, "kind": kind, "left": pid+"l", "right": pid+"r",
                      "split": split, "subtype": subtype})
    # Nuisance e1+e2 overlaps protected e1, yet has a separable component.
    add("a", "A", [0, 0, 0, 0], [1, 1, 0, 0])
    add("p", "P", [0, 0, 0, 0], [0, 0, 1, 0], subtype="clinical")
    add("c", "C", [0, 0, 0, 0], [1, 0, 0, 0], effect=0.0)
    add("f", "F", [0, 0, 0, 0], [0, 0, 0, 1], effect=0.0)
    return features, pairs, add


class OperatorTests(unittest.TestCase):
    def test_two_dimensional_oblique_example(self):
        angle = math.radians(60)
        n = torch.tensor([[math.cos(angle)], [math.sin(angle)]], dtype=torch.float64)
        s = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        nb, b, geometry = oblique_factors(n, s, theta=30)
        e = torch.eye(2, dtype=torch.float64) - nb @ b
        expected = torch.tensor([[1.0, -1/math.tan(angle)], [0.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(e, expected, atol=1e-12, rtol=1e-12)
        checks = algebra_diagnostics(nb, b, s, 30)
        self.assertLess(checks["signal_preservation_error"], 1e-12)
        self.assertLess(checks["erasure_error"], 1e-12)
        self.assertLess(checks["idempotence_error"], 1e-12)
        self.assertTrue(checks["angle_bound_satisfied"])
        self.assertAlmostEqual(geometry["principal_angles_degrees"][0], 60)

    def test_overlap_and_extra_nuisance_dimensions(self):
        n = torch.eye(4, dtype=torch.float64)[:, :3]
        s = n[:, :1]
        nb, b, geom = oblique_factors(n, s, 45)
        self.assertEqual(nb.shape, (4, 2))
        self.assertEqual(geom["discarded_rank"], 1)
        torch.testing.assert_close((torch.eye(4) - nb@b) @ s, s)
        empty, empty_b, _ = oblique_factors(n[:, :1], s, 15)
        self.assertEqual(empty.shape[1], 0)
        self.assertEqual(empty_b.shape, (0, 4))

    def test_uncentered_energy_sign_invariance_and_rank_cap(self):
        constant = torch.tensor([[2.0, 0], [2.0, 0]], dtype=torch.float64)
        basis, moment, detail = energy_basis(constant, .95, 4)
        self.assertEqual(basis.shape[1], 1)
        self.assertEqual(detail["total_energy"], 4.0)
        _, reversed_moment, _ = energy_basis(-constant, .95, 4)
        torch.testing.assert_close(moment, reversed_moment)
        data = torch.diag(torch.tensor([math.sqrt(6), math.sqrt(3), 1.0], dtype=torch.float64))
        _, _, cap = energy_basis(data, .90, 1)
        self.assertAlmostEqual(cap["retained_fraction"], .6)
        self.assertTrue(cap["rank_limited"])
        self.assertFalse(cap["target_reached"])
        zeros, _, zero = energy_basis(torch.zeros(3, 4), .90, 4)
        self.assertEqual(zeros.shape[1], 0)
        self.assertTrue(zero["zero_energy"])

    def test_separate_type_ranks_merge_above_single_type_cap(self):
        features, pairs, _ = fixture()
        operators, diag = fit_operators(features, pairs, [0], {0: ["A", "P"]}, kmax=1)
        self.assertEqual(diag["heads"]["0"]["merged_nuisance_rank"], 2)
        self.assertEqual(diag["heads"]["0"]["merged_signal_rank"], 2)
        self.assertEqual(operators[0]["N"].shape[1], 2)
        self.assertFalse(diag["signal_pairs_filtered_by_effect"])
        self.assertEqual(diag["pair_counts_after_effect_filter"]["C"], 1)

    def test_interference_filter_and_unfiltered_reference(self):
        features, pairs, add = fixture()
        add("small", "A", [10, 0, 0, 0], [12, 0, 0, 0], effect=.01)
        add("heldout", "A", [1000, 0, 0, 0], [2000, 0, 0, 0], split="test")
        operators, diag = fit_operators(features, pairs, [0], {0: ["A"]})
        self.assertEqual(diag["pair_counts_before_effect_filter"]["A"], 2)
        self.assertEqual(diag["pair_counts_after_effect_filter"]["A"], 1)
        torch.testing.assert_close(operators[0]["mu"], torch.tensor([5.75, .25, 0, 0]))
        self.assertEqual(diag["heads"]["0"]["types"], ["A"])
        self.assertEqual(diag["heads"]["0"]["merged_nuisance_rank"], 1)
        self.assertEqual(diag["heads"]["0"]["anchor_observations"], 4)

    def test_anchor_shifts_do_not_change_pair_response(self):
        features, pairs, _ = fixture()
        paired, _ = fit_operators(features, pairs, [0], {0: ["A"]}, anchor="paired")
        zero, _ = fit_operators(features, pairs, [0], {0: ["A"]}, anchor="zero")
        left = torch.tensor([2., 3., 4., 5.])
        right = torch.tensor([6., 7., 8., 9.])
        pa = apply_operator(right, paired[0]) - apply_operator(left, paired[0])
        pz = apply_operator(right, zero[0]) - apply_operator(left, zero[0])
        torch.testing.assert_close(pa, pz)
        self.assertGreater(float(torch.linalg.vector_norm(apply_operator(left, paired[0])-apply_operator(left, zero[0]))), 0)

    def test_protection_and_orthogonal_replacements_rebuild(self):
        features, pairs, _ = fixture()
        full, _ = fit_operators(features, pairs, [0], {0: ["A"]})
        removed, rd = fit_operators(features, pairs, [0], {0: ["A"]}, variant="no_C")
        self.assertEqual(removed[0]["S"].shape[1], 1)
        self.assertFalse(rd["heads"]["0"]["per_type"]["C"]["included_as_signal"])
        perpendicular, pd = fit_operators(features, pairs, [0], {0: ["A"]}, variant="orthogonal")
        torch.testing.assert_close(full[0]["N"] @ full[0]["N"].T,
                                   perpendicular[0]["N"] @ perpendicular[0]["N"].T)
        self.assertGreater(pd["heads"]["0"]["algebra"]["signal_preservation_error"], .1)
        self.assertLess(rd["heads"]["0"]["algebra"]["signal_preservation_error"], 1e-6)

    def test_oriented_mean_cancellation_is_reported(self):
        delta = torch.tensor([[1., 0], [-1., 0]])
        mean_basis, _, detail = energy_basis(delta, mean_direction=True)
        moment_basis, _, _ = energy_basis(delta)
        self.assertEqual(mean_basis.shape[1], 0)
        self.assertEqual(moment_basis.shape[1], 1)
        self.assertEqual(detail["total_energy"], 1.0)
        self.assertEqual(detail["retained_energy"], 0.0)

    def test_leace_erases_empirical_cross_covariance_without_truth(self):
        generator = torch.Generator().manual_seed(13)
        concept = torch.randint(0, 2, (120, 2), generator=generator).double()
        noise = torch.randn(120, 4, generator=generator, dtype=torch.float64)
        mixing = torch.tensor([[1., 2.], [3., 0.], [0., 2.], [1., -1.]], dtype=torch.float64)
        x = noise + concept @ mixing.T
        n, b, detail = leace_factors(x, concept)
        xc, zc = x-x.mean(0), concept-concept.mean(0)
        erased = xc - (xc @ b.T) @ n.T
        self.assertLess(float(torch.linalg.matrix_norm(erased.T @ zc / len(x))), 1e-12)
        self.assertLess(algebra_diagnostics(n, b, torch.empty(4, 0))["idempotence_error"], 1e-12)
        self.assertFalse(detail["node_truth_used_as_concept"])
        features, pairs, _ = fixture()
        _, actual = fit_operators(features, pairs, [0], {0: ["A", "P"]}, variant="leace")
        leace = actual["heads"]["0"]["leace"]
        self.assertEqual(leace["concept_categories"], ["A:", "P:clinical"])
        self.assertFalse(leace["node_truth_used_as_concept"])

    def test_randomization_preserves_layer_type_and_union_counts(self):
        heads = [0, 1, 4, 32, 33, 35]
        types = {0: ["A"], 1: ["P"], 4: ["A", "P"], 32: ["A"], 33: ["A"], 35: ["P"]}
        for seed in (13, 37, 73):
            new, nt = layer_matched_random(heads, types, seed)
            def counts(hs, ts):
                return Counter((h//32, tuple(ts[h])) for h in hs)
            self.assertEqual(counts(heads, types), counts(new, nt))
            self.assertEqual(len(new), len(set(new)))
            self.assertEqual((new, nt), layer_matched_random(heads, types, seed))

    def test_grid_empty_set_and_original_fallback(self):
        self.assertEqual(len(configuration_grid()), 18)
        empty, detail = fit_operators({}, [], [], {})
        self.assertEqual(empty, {})
        self.assertEqual(detail["total_edit_rank"], 0)
        features, pairs, _ = fixture()
        pairs = [p for p in pairs if p["kind"] != "A"]
        ops, diag = fit_operators(features, pairs, [0], {0: ["P"]})
        self.assertEqual(diag["heads"]["0"]["anchor_source"], "explicit_fit_original_inputs")
        json.dumps(diag, allow_nan=False)
        self.assertTrue(torch.isfinite(ops[0]["B"]).all())


if __name__ == "__main__":
    unittest.main()
