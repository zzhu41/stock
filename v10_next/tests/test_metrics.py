import math
import unittest

from v10_next.execution import run
from v10_next.metrics import attribution, summarize


class MetricsTests(unittest.TestCase):
    def test_period_boundary_keeps_first_day_return_and_compounds_years(self):
        result = dict(daily=[("2020-12-31", 1.1, {}), ("2021-01-04", .99, {}),
                             ("2021-01-05", 1.188, {})],
                      trades=[], diagnostics=dict(deferred_rebalances=[], missing_held_bars=[]))
        m = summarize(result, "2021-01-01", "2021-12-31")
        self.assertAlmostEqual(m["total_return"], .08)
        self.assertAlmostEqual(m["max_dd"], -.10)
        full = summarize(result)
        self.assertAlmostEqual(math.prod(1 + v for v in full["yearly"].values()), 1.188)

    def test_asset_and_cost_attribution_reconstructs_a_switch(self):
        dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
        histories = {"A": [(dates[0], 100, 100, 1), (dates[1], 100, 110, 1),
                            (dates[2], 121, 100, 1), (dates[3], 100, 100, 1)],
                     "B": [(dates[0], 200, 200, 1), (dates[1], 200, 200, 1),
                            (dates[2], 200, 220, 1), (dates[3], 220, 242, 1)]}
        targets = {dates[0]: {"A": 1}, dates[1]: {"B": .5}, dates[2]: None, dates[3]: None}
        result = run(histories, dates, lambda d, h, s: targets[d], dates[0], dates[-1], fee=.001, slippage=.002)
        a = attribution(result, histories)
        self.assertGreater(a["log_growth_by_asset"]["A"], 0)
        self.assertGreater(a["log_growth_by_asset"]["B"], 0)
        self.assertLess(a["log_growth_by_asset"]["transaction_costs"], 0)
        self.assertAlmostEqual(sum(a["log_growth_by_asset"].values()), math.log(result["nav"]))


if __name__ == "__main__":
    unittest.main()
