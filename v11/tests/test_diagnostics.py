"""Synthetic economic/statistical identities; no V11 result files are read."""
from datetime import date, timedelta
import math
import unittest

import numpy as np

from v11.diagnostics import (advantage_concentration, development_family_tests,
                             event_cluster_diagnostics, white_style_test)


def calendar(n):
    return [(date(2014, 1, 2) + timedelta(days=i)).isoformat() for i in range(n)]


class FamilyBootstrapTests(unittest.TestCase):
    def compare_naive(self, method):
        # First observation deliberately includes an entry-cost difference.
        candidates = np.array([[-.003, .03, -.04, .02, -.01, .04, .01],
                               [-.002, -.01, .03, .01, -.03, .01, .02],
                               [-.001, .01, -.02, .01, .02, -.01, .03]])
        benchmark = np.array([-.001, .01, -.02, .01, -.01, .02, .01])
        saved = candidates.copy()
        candidates.flags.writeable = False
        t, block, draws, seed = 7, 3, 1000, 728
        out = white_style_test(candidates, benchmark, block, draws, seed, method,
                               batch_size=17, return_bootstrap=True, expected_candidate_count=3)
        delta = np.log1p(candidates) - np.log1p(benchmark)
        centered = delta - delta.mean(axis=1)[:, None]
        expected_observed = math.sqrt(t) * max(0, delta.mean(axis=1).max())
        rng = np.random.default_rng(seed)
        statistics = []
        for _ in range(draws):
            indices = []
            if method == 'circular':
                starts = rng.integers(0, t, size=3)
                for start in starts:
                    indices.extend((int(start) + k) % t for k in range(block))
                indices = indices[:t]
            else:
                while len(indices) < t:
                    length = min(int(rng.geometric(1 / block)), t - len(indices))
                    start = int(rng.integers(0, t))
                    indices.extend((start + k) % t for k in range(length))
            self.assertEqual(len(indices), t)
            statistics.append(math.sqrt(t) * max(0, centered[:, indices].mean(axis=1).max()))
        expected_p = (1 + np.count_nonzero(np.asarray(statistics) >= expected_observed)) / (draws + 1)
        self.assertAlmostEqual(out['observed_statistic'], expected_observed, places=14)
        np.testing.assert_allclose(out['bootstrap_statistics'], statistics, rtol=1e-12, atol=1e-14)
        self.assertEqual(out['p_value'], expected_p)
        self.assertEqual(out['observations'], 7)
        self.assertEqual(out['trials'], 3)
        self.assertTrue(out['supplied_family_count_checked'])
        np.testing.assert_array_equal(candidates, saved)
        if method == 'circular':
            self.assertEqual(out['circular_full_blocks'], 2)
            self.assertEqual(out['circular_remainder'], 1)

    def test_circular_counts_match_naive_paired_resampling_with_remainder(self):
        self.compare_naive('circular')

    def test_stationary_geometric_counts_match_naive_paired_resampling(self):
        self.compare_naive('stationary')

    def test_same_seed_and_sampling_are_independent_of_batch_size_and_duplicate_paths(self):
        rng = np.random.default_rng(51)
        candidates = rng.normal(.0004, .012, (3, 43))
        benchmark = rng.normal(.0001, .008, 43)
        first = white_style_test(candidates, benchmark, 6, seed=10, batch_size=1, return_bootstrap=True)
        batch = white_style_test(candidates, benchmark, 6, seed=10, batch_size=23, return_bootstrap=True)
        duplicated = white_style_test(np.vstack((candidates, candidates[0])), benchmark, 6,
                                      seed=10, batch_size=11, return_bootstrap=True)
        for other in (batch, duplicated):
            self.assertEqual(first['resampling_count_sha256'], other['resampling_count_sha256'])
            self.assertEqual(first['p_value'], other['p_value'])
            np.testing.assert_allclose(first['bootstrap_statistics'], other['bootstrap_statistics'], atol=1e-14)
        changed_seed = white_style_test(candidates, benchmark, 6, seed=11)
        self.assertNotEqual(first['resampling_count_sha256'], changed_seed['resampling_count_sha256'])
        self.assertNotEqual(first['bootstrap_95_critical'], changed_seed['bootstrap_95_critical'])

    def test_identical_zero_and_negative_constant_paths_give_no_evidence(self):
        benchmark = np.linspace(-.01, .01, 60)
        same = white_style_test(np.tile(benchmark, (3, 1)), benchmark, 20)
        self.assertEqual(same['observed_statistic'], 0)
        self.assertEqual(same['p_value'], 1)
        negative = white_style_test(np.expm1(np.full((1, 60), -.003)), np.zeros(60), 20)
        self.assertEqual(negative['observed_statistic'], 0)
        self.assertEqual(negative['p_value'], 1)
        self.assertLess(negative['bootstrap_tail_probability_wilson95'][0], 1)
        self.assertEqual(negative['candidates'], 1)

    def test_constant_positive_growth_is_centered_and_has_finite_simulation_p_floor(self):
        candidates = np.expm1(np.vstack((np.full(60, .002), np.full(60, -.003))))
        out = white_style_test(candidates, np.zeros(60), 20, candidate_ids=['winner', 'poor'])
        self.assertEqual(out['observed_winner_id'], 'winner')
        self.assertAlmostEqual(out['observed_statistic'], math.sqrt(60) * .002)
        self.assertLess(out['bootstrap_95_critical'], 1e-12)
        self.assertEqual(out['bootstrap_exceedances'], 0)
        self.assertEqual(out['p_value'], 1 / 1001)
        self.assertGreater(out['bootstrap_tail_probability_wilson95'][1], out['p_value'])
        self.assertFalse(out['annual_log_excess_is_cagr_difference'])

    def test_single_strategy_longer_blocks_preserve_positive_serial_dependence(self):
        rng = np.random.default_rng(34)
        z = np.zeros(300)
        for i in range(1, len(z)):
            z[i] = .95 * z[i - 1] + rng.normal(0, .003)
        z = z - z.mean() + .0002
        candidates = np.expm1(z)[None, :]
        iid = white_style_test(candidates, np.zeros(300), 1, method='circular')
        blocks = white_style_test(candidates, np.zeros(300), 20, method='circular')
        self.assertGreater(blocks['bootstrap_95_critical'], 1.5 * iid['bootstrap_95_critical'])
        self.assertGreaterEqual(blocks['p_value'], iid['p_value'])

    def test_full_length_circular_block_does_not_drop_or_repeat_days(self):
        row = np.expm1(np.array([-.004, .009, -.001, .005, .003]))
        out = white_style_test(row[None, :], np.zeros(5), 5, method='circular', return_bootstrap=True)
        self.assertLess(max(out['bootstrap_statistics']), 1e-14)
        self.assertEqual(out['circular_remainder'], 0)
        self.assertEqual(out['p_value'], 1 / 1001)

    def test_ties_use_explicit_id_and_family_is_not_internally_shortlisted(self):
        candidates = np.expm1(np.tile(np.linspace(-.01, .011, 60), (4, 1)))
        out = white_style_test(candidates, np.zeros(60), 20, candidate_ids=['z', 'b', 'a', 'c'])
        self.assertEqual(out['observed_winner_id'], 'a')
        self.assertEqual(out['observed_winner_index'], 2)
        self.assertEqual(out['candidates'], 4)
        with self.assertRaisesRegex(ValueError, 'TopK'):
            white_style_test(candidates, np.zeros(60), 20, expected_candidate_count=5)

    def test_invalid_inputs_cannot_be_silently_dropped(self):
        for candidates, benchmark in ((np.zeros((0, 3)), np.zeros(3)), (np.zeros((3, 0)), np.zeros(1)),
                                      (np.zeros((1, 1)), np.zeros(1)), (np.zeros((3, 4)), np.zeros(3)),
                                      (np.array([[0., np.nan]]), np.zeros(2)),
                                      (np.array([[0., -1.]]), np.zeros(2))):
            with self.subTest(shape=candidates.shape), self.assertRaises(ValueError):
                white_style_test(candidates, benchmark, 1)
        for kwargs in ({'draws': 999}, {'block_length': 61}, {'block_length': 0}, {'seed': -1},
                       {'batch_size': True}, {'method': 'independent_candidates'},
                       {'candidate_ids': ['duplicate', 'duplicate']}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                white_style_test(np.zeros((2, 60)), np.zeros(60), **kwargs)

    def test_development_wrapper_refuses_confirmation_and_2026_data(self):
        values, benchmark = np.zeros((2, 61)), np.zeros(61)
        result = development_family_tests(values, benchmark, calendar(61), block_lengths=(20, 60))
        self.assertEqual(set(result['tests']), {'20', '60'})
        self.assertFalse(result['known_confirmation_or_2026_included'])
        for future in ('2022-01-04', '2026-01-05'):
            dates = calendar(61); dates[-1] = future
            with self.subTest(future=future), self.assertRaisesRegex(ValueError, 'development'):
                development_family_tests(values, benchmark, dates)


class ConcentrationTests(unittest.TestCase):
    def test_net_share_and_positive_mass_share_have_different_denominators(self):
        delta = np.array([.10, .04, -.12, .01, -.02])
        benchmark = np.expm1(np.full(5, .02))
        selected = np.expm1(delta + .02)
        out = advantage_concentration(selected, benchmark, calendar(5), top_k=(1, 10))
        self.assertAlmostEqual(out['net_log_excess'], .01)
        self.assertAlmostEqual(out['positive_log_excess'], .15)
        self.assertAlmostEqual(out['negative_log_excess'], -.14)
        self.assertAlmostEqual(out['top']['1']['share_of_net_log_excess'], 10.)
        self.assertAlmostEqual(out['top']['1']['share_of_positive_log_excess'], 2 / 3)
        self.assertEqual(out['top']['10']['actual_positive_days'], 3)
        self.assertEqual(out['top']['10']['share_of_positive_log_excess'], 1.)
        self.assertFalse(out['exclusion_is_tradable'])
        self.assertNotEqual(out['net_log_excess'], out['selected_net_log_growth'])

    def test_zero_or_negative_net_excess_does_not_publish_a_misleading_net_share(self):
        negative = advantage_concentration(np.expm1([.02, -.03]), np.zeros(2))
        self.assertLess(negative['net_log_excess'], 0)
        self.assertIsNone(negative['top']['10']['share_of_net_log_excess'])
        self.assertEqual(negative['top']['10']['share_of_positive_log_excess'], 1.)
        identical = advantage_concentration(np.zeros(4), np.zeros(4))
        self.assertIsNone(identical['top']['10']['share_of_net_log_excess'])
        self.assertIsNone(identical['top']['10']['share_of_positive_log_excess'])
        self.assertEqual(identical['top_positive_days'], [])

    def test_clusters_use_market_indices_and_flag_overlapping_surrounding_windows(self):
        dates = calendar(20)
        delta = np.arange(20) * .001
        out = event_cluster_diagnostics(np.expm1(delta), np.zeros(20), dates,
                                        [dates[3], dates[5], dates[10], dates[10]],
                                        merge_gap=2, pre=2, post=3)
        self.assertEqual(out['supplied_event_count'], 4)
        self.assertEqual(out['unique_event_dates'], 3)
        self.assertEqual(out['cluster_count'], 2)
        first = out['clusters'][0]
        self.assertEqual(first['pre']['observations'], 2)
        self.assertAlmostEqual(first['pre']['net_log_excess'], delta[1:3].sum())
        self.assertAlmostEqual(first['core']['net_log_excess'], delta[3:6].sum())
        self.assertAlmostEqual(first['post']['net_log_excess'], delta[6:9].sum())
        self.assertEqual(out['overlapping_surrounding_window_pairs'], [(0, 1)])
        self.assertFalse(out['causal_event_effect'])
        self.assertFalse(out['independent_event_samples'])

    def test_event_edges_are_clipped_and_unknown_dates_are_not_shifted(self):
        dates = calendar(5)
        out = event_cluster_diagnostics(np.zeros(5), np.zeros(5), dates,
                                        [dates[0], dates[-1]], merge_gap=0, pre=3, post=3)
        self.assertIsNone(out['clusters'][0]['pre'])
        self.assertTrue(out['clusters'][0]['pre_truncated'])
        self.assertIsNone(out['clusters'][-1]['post'])
        self.assertTrue(out['clusters'][-1]['post_truncated'])
        empty = event_cluster_diagnostics(np.zeros(5), np.zeros(5), dates, [])
        self.assertEqual(empty['cluster_count'], 0)
        with self.assertRaisesRegex(ValueError, 'observed date'):
            event_cluster_diagnostics(np.zeros(5), np.zeros(5), dates, ['2014-03-01'])


if __name__ == '__main__':
    unittest.main()
