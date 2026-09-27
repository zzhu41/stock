import unittest
from unittest.mock import patch

import backtest
import strategy
from market_data import CASH, GOLD


def indicator(**changes):
    values = strategy.indicators([100 + i * .1 for i in range(300)])
    values.update(changes)
    return values


class StrategyCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.defaults = patch.multiple(strategy, PANIC_DROP=.04, VOL_PANIC_ON=False,
                                       MIN_HOLD=0, NEVER_EMPTY=True, BULL_VOTE=False,
                                       BULL_HYST=0, BULL_CONFIRM=1)
        self.defaults.start()
        self.addCleanup(self.defaults.stop)

    def test_live_price_requires_explicit_date(self):
        with self.assertRaises(ValueError):
            strategy.rank({}, live_prices={"513100": 2.0})

    def test_live_price_appends_after_yesterday(self):
        rows = [("2025-%03d" % i, 100 + i, 100 + i, 1000) for i in range(300)]
        expected = strategy.indicators([r[2] for r in rows] + [420])
        actual = dict(strategy.rank({"513100": rows}, on_date="2026-01-01",
                                    live_prices={"513100": 420}))["513100"]
        self.assertAlmostEqual(actual["mom20"], expected["mom20"])
        self.assertEqual(rows[-1][2], 399)

    def test_top_ranked_panic_exits_despite_minimum_hold_and_buffer(self):
        table = [("510300", indicator(score=10, mom20=.2, ret1=-.08, above_ma=True)),
                 ("159915", indicator(score=5, mom20=.1, ret1=.01))]
        with patch.multiple(strategy, MIN_HOLD=10, BUFFER=.5, POOL_BUFFER={}):
            target, reason = strategy.decide(table, "510300", holding_days=1)
        self.assertEqual(target, "159915")
        self.assertIn("急跌", reason)

    def test_panic_gold_cannot_reenter_through_fallback_same_day(self):
        table = [(GOLD, indicator(score=10, mom20=-.1, ret1=-.05))]
        self.assertEqual(strategy.decide(table, GOLD)[0], CASH)
        table[0][1].update(mom20=.1, ret1=.01)
        self.assertEqual(strategy.decide(table, CASH)[0], GOLD)

    def test_negative_fallback_without_panic_is_preserved(self):
        table = [(GOLD, indicator(mom20=-.1, ret1=-.01))]
        self.assertEqual(strategy.decide(table, GOLD)[0], GOLD)

    def test_crash_lock_overrides_panic_until_fifth_session(self):
        days = ["2026-01-%02d" % i for i in range(1, 8)]
        histories = {c: [(d, 100, 100, 1000) for d in days]
                     for c in ("159915", GOLD, CASH)}
        def rank(_, on_date=None):
            return [("159915", indicator(mom5=-.1 if on_date == days[0] else 0,
                                         dist_ma250=-.3, ret1=-.05, mom20=.1,
                                         score=10)),
                    (GOLD, indicator(score=1, mom20=.1, ret1=.01))]
        with patch.object(strategy, "rank", side_effect=rank):
            result = backtest.backtest(histories, days, start=days[0])
        self.assertEqual([r[2] for r in result["daily"][:5]], ["159915"] * 5)
        self.assertEqual(result["daily"][5][2], GOLD)

    def test_removed_stock_cannot_be_bought_by_crash_branch(self):
        days = ["2026-01-02", "2026-01-05"]
        histories = {c: [(d, 100, 100, 1000) for d in days] for c in ("159915", CASH)}
        table = [("159915", indicator(mom5=-.1, dist_ma250=-.3))]
        with patch.object(strategy, "STOCK_POOL", []), \
                patch.object(strategy, "rank", return_value=table):
            result = backtest.backtest(histories, days, start=days[0])
        self.assertEqual(result["crash_buys"], [])
        self.assertTrue(all(row[2] == CASH for row in result["daily"]))

    def test_stale_historical_bar_does_not_rank_as_today(self):
        rows = [("2025-%03d" % i, 100, 100, 1000) for i in range(300)]
        self.assertEqual(strategy.rank({"513100": rows}, on_date="2026-01-01"), [])

    def test_missing_bar_defers_sale_and_catches_up_return_on_resumption(self):
        days = ["2026-01-02", "2026-01-05", "2026-01-06"]
        histories = {"159915": [(days[0], 100, 100, 1000), (days[2], 120, 120, 1000)],
                     CASH: [(d, 100, 100, 1000) for d in days]}
        with patch.object(strategy, "rank", return_value=[]), \
                patch.object(strategy, "decide", side_effect=[("159915", ""), (CASH, ""), (CASH, "")]):
            result = backtest.backtest(histories, days, start=days[0], crash_mom5=0)
        self.assertEqual(result["daily"][1][2], "159915")
        self.assertEqual(result["switches"], 1)
        self.assertAlmostEqual(result["nav"], 1.2 * (1 - 2 * backtest.FEE))


if __name__ == "__main__":
    unittest.main()
