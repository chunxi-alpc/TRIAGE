"""CPU-only arithmetic tests for the cached attention-head quadrant analysis."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "aster_head_quadrant" / "analyze.py"
SPEC = importlib.util.spec_from_file_location("aster_head_quadrant_analysis", MODULE)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def example_pairs():
    # A has two sources, with two pairs in source s1; the other kinds have one.
    return [
        {"kind": "A", "source_id": "s1", "left": "shared", "right": "r1"},
        {"kind": "A", "source_id": "s1", "left": "shared", "right": "r2"},
        {"kind": "A", "source_id": "s2", "left": "shared", "right": "r3"},
        {"kind": "P", "source_id": "s1", "left": "shared", "right": "r4"},
        {"kind": "C", "source_id": "s2", "left": "shared", "right": "r5"},
        {"kind": "F", "source_id": "s3", "left": "shared", "right": "r6"},
    ]


def test_percentile_average_ties_and_missing():
    actual = analysis.percentile_ranks(np.array([10.0, 10.0, 30.0, np.nan]))
    np.testing.assert_allclose(actual[:3], [100 / 3, 100 / 3, 250 / 3])
    assert np.isnan(actual[3])


def test_percentile_constant_and_singleton():
    np.testing.assert_array_equal(analysis.percentile_ranks([2, 2, 2]), [50, 50, 50])
    np.testing.assert_array_equal(analysis.percentile_ranks([9]), [50])
    assert analysis.percentile_ranks([]).size == 0


def test_percentile_order_and_permutation():
    values = np.array([30.0, 10.0, 40.0, 20.0])
    expected = np.array([62.5, 12.5, 87.5, 37.5])
    np.testing.assert_allclose(analysis.percentile_ranks(values), expected)
    permutation = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(analysis.percentile_ranks(values[permutation]), expected[permutation])


def test_pair_weights_balance_categories_sources_and_pairs():
    pairs = example_pairs()
    actual = analysis.pair_weights(pairs)
    np.testing.assert_allclose(actual, [1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 4, 1 / 4])
    np.testing.assert_allclose(actual.sum(), 1.0, rtol=1e-12, atol=1e-12)
    for kind in "APCF":
        np.testing.assert_allclose(sum(w for p, w in zip(pairs, actual) if p["kind"] == kind), 0.25, rtol=1e-12, atol=1e-12)


def test_single_category_weights_are_renormalized():
    np.testing.assert_allclose(analysis.pair_weights(example_pairs()[:3]), [0.25, 0.25, 0.5])


def test_endpoint_weights_split_pairs_and_aggregate_shared_inputs():
    pairs = example_pairs()
    uid_to_index = {"shared": 0, **{f"r{i}": i for i in range(1, 7)}, "held_out": 7}
    actual = analysis.endpoint_weights(pairs, uid_to_index, 8)
    np.testing.assert_allclose(actual, [0.5, 1 / 32, 1 / 32, 1 / 16, 1 / 8, 1 / 8, 1 / 8, 0])
    np.testing.assert_allclose(actual.sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_weighted_population_sigma_is_channelwise():
    # Each item is [heads, channels]; the unused fourth endpoint has zero weight.
    activations = np.array([
        [[0.0, 7.0], [1.0, 1.0]],
        [[2.0, 7.0], [3.0, 3.0]],
        [[4.0, 7.0], [5.0, 9.0]],
        [[1e6, -1e6], [1e6, -1e6]],
    ])
    weights = np.array([0.25, 0.5, 0.25, 0.0])
    sigma, floored = analysis.weighted_sigma(activations, weights, batch_size=2)
    # Weighted means: [[2, 7], [3, 4]]; variances: [[2, 0], [2, 9]].
    np.testing.assert_allclose(sigma, [[np.sqrt(2), 1e-5], [np.sqrt(2), 3]], rtol=1e-12)
    np.testing.assert_array_equal(floored, [[False, True], [False, False]])


def test_weighted_sigma_batch_size_and_held_out_values_do_not_change_result():
    activations = np.array([[[0.0, 3.0]], [[2.0, 3.0]], [[4.0, 3.0]], [[999.0, 999.0]]])
    weights = np.array([0.25, 0.5, 0.25, 0.0])
    expected, _ = analysis.weighted_sigma(activations, weights, batch_size=1)
    activations[3] = -1e12
    actual, _ = analysis.weighted_sigma(activations, weights, batch_size=8)
    np.testing.assert_array_equal(actual, expected)


def test_weighted_sigma_retains_m2_small_scale_floor():
    activations = np.array([[[0.0]], [[2e-7]]])
    sigma, floored = analysis.weighted_sigma(activations, np.array([0.5, 0.5]))
    np.testing.assert_allclose(sigma, [[[1e-5]]][0])
    np.testing.assert_array_equal(floored, [[True]])


def test_quadrant_labels_and_exact_threshold_boundary():
    labels, thresholds = analysis.quadrants(
        np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0])
    )
    assert thresholds == {"x": 1.0, "y": 1.0}
    assert list(labels) == ["low", "CF_selective", "AP_selective", "mixed"]


def test_quadrants_missing_and_median_sensitivity():
    labels, _ = analysis.quadrants(
        np.array([0.0, 0.0, 1.0, 1.0, np.nan]),
        np.array([0.0, 1.0, 0.0, 1.0, np.nan]),
        quantile=0.5,
    )
    assert list(labels) == ["low", "CF_selective", "AP_selective", "mixed", "missing"]


def test_patch_ranks_are_within_each_category_and_missing_stays_missing():
    exact = {
        "A": {"0": 1.0, "1": 3.0, "2": 2.0},
        "P": {"0": 2.0, "1": 1.0},
        "C": {"0": 10.0, "2": 20.0},
        "F": {"0": 30.0, "2": 10.0},
    }
    patch = analysis.combine_patch(exact, 4)
    np.testing.assert_allclose(patch["A_pct"][:3], [100 / 6, 500 / 6, 50])
    np.testing.assert_allclose(patch["AP_pct"][0], (100 / 6 + 75) / 2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(patch["CF_pct"][0], 50.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(patch["AP_raw"][0], 1.5, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(patch["CF_raw"][0], 20.0, rtol=1e-12, atol=1e-12)
    assert np.isnan(patch["AP_raw"][2])  # No P measurement for head 2.
    assert np.isnan(patch["CF_raw"][1])  # Neither C nor F measured head 1.
    assert np.isnan(patch["A_raw"][3])   # Unsampled head has no zero-filled effect.
    np.testing.assert_array_equal(patch["both_available"], [True, False, False, False])


def test_patch_empty_category_yields_no_false_bilateral_coverage():
    patch = analysis.combine_patch({"A": {"0": 4.0}, "P": {}, "C": {"0": 3.0}, "F": {"0": 1.0}}, 2)
    assert np.isnan(patch["AP_pct"]).all()
    assert not patch["both_available"].any()
    np.testing.assert_allclose(patch["CF_pct"][0], 50.0, rtol=1e-12, atol=1e-12)


def test_five_fold_consensus_uses_identical_bilateral_fold_sets():
    x = np.array([[10, 10, 10], [20, 20, 20], [30, 30, np.nan], [40, np.nan, 40], [50, np.nan, 50]])
    y = np.array([[60, 60, np.nan], [70, 70, 70], [80, 80, 80], [90, 90, np.nan], [100, 100, np.nan]])
    mean_x, mean_y, counts = analysis.fold_consensus(x, y, min_folds=3)
    np.testing.assert_array_equal(counts, [5, 3, 1])
    np.testing.assert_allclose(mean_x[:2], [30, 20])
    np.testing.assert_allclose(mean_y[:2], [80, 70])
    assert np.isnan(mean_x[2]) and np.isnan(mean_y[2])


def test_five_fold_consensus_order_is_irrelevant():
    x = np.array([[1, 2], [3, np.nan], [5, 6], [7, 8], [9, 10]], dtype=float)
    y = 100 - x
    expected = analysis.fold_consensus(x, y, min_folds=3)
    actual = analysis.fold_consensus(x[::-1], y[::-1], min_folds=3)
    for actual_item, expected_item in zip(actual, expected):
        np.testing.assert_allclose(actual_item, expected_item)




def test_movement_rms_uses_source_weights_and_mean_squared_channels():
    pairs = example_pairs()[:3]
    index = {"shared": 0, "r1": 1, "r2": 2, "r3": 3}
    activations = np.array([
        [[0, 0], [0, 0]],
        [[2, 0], [6, 0]],
        [[0, 4], [0, 8]],
        [[2, 4], [6, 8]],
    ], dtype=float)
    sigma = np.array([[2, 4], [2, 4]], dtype=float)
    actual = analysis.movement_rms(activations, pairs, index, sigma, batch_size=2)
    # Pair weights [.25, .25, .5]; per-head squared distances
    # [.5, .5, 1] and [4.5, 2, 6.5], respectively.
    np.testing.assert_allclose(actual, np.sqrt([0.75, 4.875]), rtol=1e-12)
    channel_scale = np.array([[100, 0.1], [0.25, 3]])
    scaled = analysis.movement_rms(activations * channel_scale, pairs, index, sigma * channel_scale)
    np.testing.assert_allclose(scaled, actual, rtol=1e-12)


def test_weighted_sigma_never_reads_held_out_activation_rows():
    class TrainingRowsOnly:
        shape = (4, 1, 1)

        def __getitem__(self, indices):
            assert set(np.asarray(indices).tolist()) <= {0, 1, 2}
            return np.array([[[0.0]], [[2.0]], [[4.0]], [[1e12]]])[indices]

    sigma, _ = analysis.weighted_sigma(TrainingRowsOnly(), np.array([0.25, 0.5, 0.25, 0]), batch_size=2)
    np.testing.assert_allclose(sigma, [[np.sqrt(2)]])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    )


if __name__ == "__main__":
    unittest.main()
