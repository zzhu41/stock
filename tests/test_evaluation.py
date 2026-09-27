import unittest

from evaluate_versions import next_open_replay


DATES = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")


def signals(*targets):
    # Signal NAV is deliberately unrelated: replay must use targets and prices.
    return [(day, 999.0 + i, target) for i, (day, target) in enumerate(zip(DATES, targets))]


def bars(*rows):
    return [(DATES[i], opening, close, 1000.0) for i, opening, close in rows]


class NextOpenReplayTests(unittest.TestCase):
    def test_none_target_sells_to_uninvested_cash(self):
        daily = [("2026-01-02", 1, "A"), ("2026-01-05", 1, None),
                 ("2026-01-06", 1, None)]
        histories = {"A": [("2026-01-05", 100, 110, 1), ("2026-01-06", 120, 130, 1)]}
        result = next_open_replay(daily, histories, fee=0)
        self.assertAlmostEqual(result["nav"], 1.2)
        self.assertIsNone(result["daily"][-1][2])

    def test_switch_gives_old_holding_overnight_and_new_holding_intraday(self):
        histories = {
            "A": bars((0, 90, 100), (1, 100, 110), (2, 121, 99)),
            "B": bars((0, 80, 90), (1, 90, 100), (2, 200, 220)),
        }
        result = next_open_replay(signals("A", "B", "B"), histories, fee=0)
        self.assertEqual([h for _, _, h in result["daily"]], [None, "A", "B"])
        self.assertAlmostEqual(result["daily"][1][1], 1.1)
        # Buy A at 100, mark 110; sell at 121; buy B at 200, mark 220.
        self.assertAlmostEqual(result["nav"], 1.21 * 1.1)
        self.assertEqual(result["trades"], [(DATES[1], None, "A"), (DATES[2], "A", "B")])

    def test_initial_buy_and_each_switch_side_charge_costs(self):
        histories = {
            "A": bars((0, 100, 100), (1, 100, 100), (2, 100, 100)),
            "B": bars((0, 100, 100), (1, 100, 100), (2, 100, 100)),
        }
        fee, slippage = .001, .002
        result = next_open_replay(signals("A", "B", "B"), histories,
                                  fee=fee, slippage=slippage)
        initial = (1 - fee) / (1 + slippage)
        switch = (1 - fee) * (1 - slippage) * (1 - fee) / (1 + slippage)
        self.assertAlmostEqual(result["daily"][1][1], initial)
        self.assertAlmostEqual(result["nav"], initial * switch)

    def test_missing_held_bar_carries_mark_then_catches_up(self):
        histories = {"A": bars((0, 100, 100), (1, 100, 110), (3, 121, 133.1))}
        result = next_open_replay(signals("A", "A", "A", "A"), histories, fee=0)
        self.assertAlmostEqual(result["daily"][2][1], 1.1)
        self.assertAlmostEqual(result["nav"], 1.331)
        self.assertEqual(len(result["trades"]), 1)

    def test_missing_target_defers_whole_switch_without_spurious_fees(self):
        histories = {
            "A": bars((0, 100, 100), (1, 100, 110), (2, 110, 121), (3, 133.1, 140)),
            "B": bars((0, 80, 90), (1, 90, 100), (3, 100, 110)),
        }
        fee = .001
        result = next_open_replay(signals("A", "B", "B", "B"), histories, fee=fee)
        self.assertEqual([h for _, _, h in result["daily"]], [None, "A", "A", "B"])
        self.assertEqual(result["trades"], [(DATES[1], None, "A"), (DATES[3], "A", "B")])
        self.assertEqual(result["deferred_count"], 1)
        self.assertAlmostEqual(result["daily"][2][1], (1 - fee) * 1.21)
        self.assertAlmostEqual(result["nav"], (1 - fee) ** 3 * 1.4641)

    def test_missing_held_open_prevents_sale_even_when_target_is_available(self):
        histories = {
            "A": bars((0, 100, 100), (1, 100, 110), (3, 121, 99)),
            "B": bars((0, 80, 90), (1, 90, 100), (2, 100, 110), (3, 110, 121)),
        }
        result = next_open_replay(signals("A", "B", "B", "B"), histories, fee=0)
        self.assertEqual([h for _, _, h in result["daily"]], [None, "A", "A", "B"])
        self.assertAlmostEqual(result["nav"], 1.331)

    def test_latest_target_cancels_unfilled_previous_switch(self):
        histories = {
            "A": bars((0, 100, 100), (1, 100, 110), (2, 110, 121), (3, 121, 133.1)),
            "B": bars((0, 80, 90), (1, 90, 100), (3, 100, 120)),
        }
        result = next_open_replay(signals("A", "B", "A", "A"), histories, fee=0)
        self.assertEqual([h for _, _, h in result["daily"]], [None, "A", "A", "A"])
        self.assertEqual(result["trades"], [(DATES[1], None, "A")])
        self.assertAlmostEqual(result["nav"], 1.331)

    def test_delayed_initial_buy_keeps_uninvested_cash_until_actual_fill(self):
        histories = {"A": bars((0, 100, 100), (2, 120, 132))}
        result = next_open_replay(signals("A", "A", "A"), histories, fee=.001)
        self.assertEqual([h for _, _, h in result["daily"]], [None, None, "A"])
        self.assertEqual(result["trades"], [(DATES[2], None, "A")])
        self.assertAlmostEqual(result["nav"], .999 * 1.1)

    def test_existing_invalid_prices_fail_instead_of_being_treated_as_missing(self):
        for opening, close in ((0, 100), (100, 0), (float("nan"), 100),
                               (100, float("inf"))):
            histories = {"A": bars((0, 100, 100), (1, opening, close))}
            with self.subTest(opening=opening, close=close), self.assertRaises(ValueError):
                next_open_replay(signals("A", "A"), histories)


if __name__ == "__main__":
    unittest.main()
