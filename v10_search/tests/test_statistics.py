import unittest

import numpy as np

from v10_search.statistics import white_style_max_test, walkforward_selection_diagnostic


class MaxStatisticTests(unittest.TestCase):
    def test_matches_naive_paired_centered_resampling_with_remainder(self):
        # Includes a negative initial-entry observation, not discarded as warmup.
        a = np.array([[-.001, -.002, -.001], [.03, -.01, .01], [-.04, .03, -.02],
                      [.02, .01, .01], [-.01, -.03, .02], [.04, .01, -.01], [.01, .02, .03]])
        b = np.array([-.001, .01, -.02, .01, -.01, .02, .01])
        n, block, draws, seed = 7, 3, 101, 12
        output = white_style_max_test(a, b, block, draws, seed, batch_size=8)
        d = np.log1p(a) - np.log1p(b)[:, None]
        observed = np.sqrt(n) * max(0, d.mean(axis=0).max())
        centered = d - d.mean(axis=0)
        starts = np.random.default_rng(seed).integers(0, n, size=(draws, 3))
        reference = []
        for row in starts:
            indices = [(int(s) + j) % n for s in row[:2] for j in range(block)]
            indices.append(int(row[-1]))
            self.assertEqual(len(indices), n)
            reference.append(max(0, np.sqrt(n) * centered[indices].mean(axis=0).max()))
        expected_p = (1 + np.count_nonzero(np.asarray(reference) >= observed)) / (draws + 1)
        self.assertAlmostEqual(output["observed_statistic"], observed, places=13)
        self.assertEqual(output["p_value"], expected_p)
        self.assertAlmostEqual(output["bootstrap_95_critical"], np.quantile(reference, .95), places=13)
        self.assertEqual(output["observations"], n)
        self.assertEqual(output["tested_candidate_count"], 3)
        self.assertEqual(output["remainder_observations"], 1)

    def test_identical_paths_have_no_evidence_and_retain_all_columns(self):
        b = np.array([-.001, .01, -.02, .03, -.01, .01])
        result = white_style_max_test(np.column_stack((b, b, b)), b, block_size=2, draws=50)
        self.assertEqual(result["p_value"], 1)
        self.assertEqual(result["observed_statistic"], 0)
        self.assertEqual(result["tested_candidate_count"], 3)

    def test_zero_centering_and_finite_draw_pvalue(self):
        b = np.zeros(12)
        a = np.expm1(np.full((12, 2), [.002, -.003]))
        result = white_style_max_test(a, b, block_size=4, draws=99)
        self.assertAlmostEqual(result["p_value"], .01)
        self.assertLess(result["bootstrap_95_critical"], 1e-12)

    def test_batch_size_and_duplicate_columns_do_not_change_test(self):
        rng = np.random.default_rng(77)
        a = rng.normal(.0002, .015, size=(31, 4))
        b = rng.normal(.0001, .01, size=31)
        first = white_style_max_test(a, b, block_size=6, draws=103, batch_size=1)
        other = white_style_max_test(a, b, block_size=6, draws=103, batch_size=32)
        duplicate = white_style_max_test(np.column_stack((a, a[:, 0])), b,
                                         block_size=6, draws=103, batch_size=9)
        self.assertEqual(first["p_value"], other["p_value"])
        self.assertEqual(first["p_value"], duplicate["p_value"])
        self.assertAlmostEqual(first["bootstrap_95_critical"], duplicate["bootstrap_95_critical"])

    def test_invalid_or_incomplete_inputs_fail(self):
        for a, b in ((np.zeros((3, 0)), np.zeros(3)), (np.zeros((3, 2)), np.zeros(2)),
                     (np.array([[np.nan]]), np.zeros(1)), (np.array([[-1.0]]), np.zeros(1))):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                white_style_max_test(a, b, block_size=1)
        with self.assertRaises(ValueError):
            white_style_max_test(np.zeros((3, 2)), np.zeros(3), candidate_ids=["same", "same"])


def walk_data():
    dates = ["%d-%s" % (y, suffix) for y in range(2014, 2027) for suffix in ("01-04", "07-01")]
    a = np.zeros((len(dates), 2))
    b = np.full(len(dates), .001)
    for i, day in enumerate(dates):
        a[i] = [.01, .005] if day < "2018" else [-.005, .008]
    return a, b, dates


class WalkForwardTests(unittest.TestCase):
    def test_four_folds_choose_using_training_not_test_and_exclude_2026(self):
        a, b, dates = walk_data()
        result = walkforward_selection_diagnostic(a, b, dates, ["first", "second"])
        self.assertEqual(len(result["folds"]), 4)
        self.assertEqual(result["folds"][0]["selected_id"], "first")
        self.assertLess(result["folds"][0]["test_cagr_difference"], 0)
        changed = a.copy()
        changed_b = b.copy()
        future = np.asarray(dates) >= "2026"
        changed[future] = [-.9, 9]
        changed_b[future] = 3
        self.assertEqual(result, walkforward_selection_diagnostic(changed, changed_b, dates, ["first", "second"]))
        self.assertFalse(result["realizable_strategy"])
        self.assertFalse(result["clean_oos"])

    def test_later_data_cannot_change_earlier_fold_choices(self):
        a, b, dates = walk_data()
        expected = walkforward_selection_diagnostic(a, b, dates)
        a[np.asarray(dates) >= "2022"] = [-.9, 9]
        changed = walkforward_selection_diagnostic(a, b, dates)
        for i in range(3):
            self.assertEqual(expected["folds"][i]["selected_index"], changed["folds"][i]["selected_index"])

    def test_benchmark_fallback_and_fixed_id_tie_break(self):
        a, b, dates = walk_data()
        result = walkforward_selection_diagnostic(np.full_like(a, -.002), b, dates)
        self.assertTrue(all(f["selected_benchmark"] for f in result["folds"]))
        self.assertEqual(result["selected_stream_cagr"], result["benchmark_stream_cagr"])
        tied = walkforward_selection_diagnostic(np.full_like(a, .003), b, dates, ["z", "a"])
        self.assertTrue(all(f["selected_id"] == "a" for f in tied["folds"]))


if __name__ == "__main__":
    unittest.main()
