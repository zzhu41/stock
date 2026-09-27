import math
import unittest
from datetime import date, timedelta

from research_overfitting import concentration, paired_block_bootstrap, path_returns


def make_path(returns):
    nav, rows = 1.0, []
    for i, value in enumerate(returns):
        nav *= 1 + value
        rows.append(((date(2020, 1, 1) + timedelta(days=i)).isoformat(), nav, "A"))
    return rows


class DiagnosticTests(unittest.TestCase):
    def test_identical_paths_have_zero_difference_in_every_paired_draw(self):
        path = make_path([.03, -.02, .04, -.05, .01] * 21)
        r = paired_block_bootstrap(path, path, block_size=20, draws=300)
        self.assertEqual(r["observed_cagr_difference"], 0)
        self.assertEqual(r["conditional_percentile_95"], [0.0, 0.0])

    def test_constant_growth_has_analytic_result_even_with_truncated_last_block(self):
        a, b = make_path([.001] * 103), make_path([.0002] * 103)
        expected = 1.001 ** 244 - 1.0002 ** 244
        r = paired_block_bootstrap(a, b, block_size=20, draws=300)
        self.assertAlmostEqual(r["observed_cagr_difference"], expected, places=11)
        for bound in r["conditional_percentile_95"]:
            self.assertAlmostEqual(bound, expected, places=11)

    def test_initial_return_and_losses_survive_concentration_attribution(self):
        path = make_path([.10, -.20, .05])
        dates, logs = path_returns(path)
        self.assertAlmostEqual(sum(logs), math.log(1.1 * .8 * 1.05))
        r = concentration(path)
        self.assertEqual(r["best_days"][0]["date"], dates[0])
        self.assertEqual(r["neutralize_best_days"][0]["days"], 2)
        self.assertAlmostEqual(r["neutralize_best_days"][0]["annualized"], .8 ** (244 / 3) - 1)

    def test_dates_must_align_and_nav_must_be_valid(self):
        a, b = make_path([.01] * 4), make_path([.01] * 3)
        with self.assertRaises(ValueError):
            paired_block_bootstrap(a, b, block_size=2)
        with self.assertRaises(ValueError):
            path_returns([(a[0][0], 0, "A")])
        with self.assertRaises(ValueError):
            path_returns([a[0], a[0]])


if __name__ == "__main__":
    unittest.main()
