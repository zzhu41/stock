import unittest

from v10_round2.timing import run_same_close, for_same_close


class SameCloseTests(unittest.TestCase):
    def test_current_close_signal_does_not_capture_already_elapsed_intraday_return(self):
        dates = ['2020-01-02', '2020-01-03', '2020-01-06']
        rows = {'A': [(dates[0], 50, 100, 1), (dates[1], 110, 120, 1), (dates[2], 120, 132, 1)]}
        r = run_same_close(rows, dates, lambda d, h, s: {'A': 1} if d == dates[0] else None,
                           dates[0], dates[-1], fee=0)
        self.assertAlmostEqual(r['daily'][0][1], 1)
        self.assertAlmostEqual(r['nav'], 1.32)
        self.assertEqual(r['trades'][0]['date'], dates[0])

    def test_sell_after_close_receives_old_asset_return_and_charges_both_sides(self):
        dates = ['2020-01-02', '2020-01-03']
        rows = {'A': [(dates[0], 50, 100, 1), (dates[1], 110, 120, 1)],
                'B': [(dates[0], 200, 200, 1), (dates[1], 200, 220, 1)]}
        r = run_same_close(rows, dates, lambda d, h, s: {'A' if d == dates[0] else 'B': 1},
                           dates[0], dates[-1], fee=.001)
        self.assertAlmostEqual(r['nav'], (1 / 1.001) * 1.2 * .999 / 1.001)
        self.assertEqual(r['diagnostics']['fill_count'], 3)

    def test_timing_adaptation_is_independent_and_preserves_economic_five_day_lock(self):
        c = {'kind': 'legacy', 'overrides': {}, 'score_windows': [25]}
        adjusted = for_same_close(c)
        self.assertEqual(adjusted['overrides']['crash_lock'], 6)
        self.assertEqual(c['overrides'], {})


if __name__ == '__main__':
    unittest.main()
