"""Deterministic accounting and chronology tests; no market I/O."""
import copy
import unittest

from v10_next.execution import run


D = ["2026-01-%02d" % i for i in range(1, 7)]


def bars(prices, dates=None):
    dates = dates or D[:len(prices)]
    return [(date, opening, close, 100) for date, (opening, close) in zip(dates, prices)]


class ExecutionTests(unittest.TestCase):
    def test_old_asset_gets_overnight_new_asset_gets_intraday(self):
        histories = {"A": bars([(100, 100), (100, 110), (121, 999)]),
                     "B": bars([(200, 200), (200, 200), (200, 220)])}
        received = []

        def policy(date, observed, state):
            received.append((date, state["holding"], state["nav"]))
            return {"A": 1} if date == D[0] else {"B": 1}

        result = run(histories, D[:3], policy, D[0], D[2], fee=0, slippage=0)
        self.assertEqual(result["daily"][0][1], 1)
        self.assertAlmostEqual(result["daily"][1][1], 1.1)
        self.assertAlmostEqual(result["daily"][2][1], 1.331)
        self.assertEqual([row[1] for row in received], [None, "A", "B"])
        self.assertEqual(result["trades"][1]["signal_date"], D[1])

    def test_post_cost_full_investment_never_borrows(self):
        fee, slip = .001, .01
        histories = {"A": bars([(100, 100)] * 3)}
        result = run(histories, D[:3], lambda d, h, s: {"A": 1} if d == D[0] else {},
                     D[0], D[2], fee=fee, slippage=slip)
        entry_nav = 1 / ((1 + fee) * (1 + slip))
        self.assertAlmostEqual(result["daily"][1][1], entry_nav)
        self.assertAlmostEqual(result["nav"], entry_nav * (1 - fee) * (1 - slip))
        self.assertTrue(all(t["cash_after"] >= 0 for t in result["trades"]))
        self.assertEqual(result["final_state"]["units"], {})
        self.assertAlmostEqual(1 - result["nav"], result["diagnostics"]["total_commission"]
                               + result["diagnostics"]["total_slippage_cost"])

    def test_two_assets_equal_weights_and_idle_none_keep_units(self):
        histories = {"A": bars([(100, 100), (100, 120), (120, 150)]),
                     "B": bars([(100, 100)] * 3)}
        result = run(histories, D[:3], lambda d, h, s: {"A": .5, "B": .5} if d == D[0] else None,
                     D[0], D[2], fee=0, slippage=0)
        self.assertEqual(len(result["trades"]), 1)
        self.assertAlmostEqual(result["nav"], 1.25)
        self.assertAlmostEqual(result["final_state"]["units"]["A"], .005)
        self.assertAlmostEqual(result["daily"][-1][2]["A"], .6)

    def test_repeated_full_position_target_does_not_charge_daily_trade_costs(self):
        histories = {"A": bars([(100, 100), (100, 110), (115, 120), (125, 130)])}
        result = run(histories, D[:4], lambda d, h, s: {"A": 1}, D[0], D[3])
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["diagnostics"]["completed_noop_orders"], 2)
        self.assertAlmostEqual(result["nav"], 1.3 / (1.0001 * 1.001))

    def test_two_asset_target_uses_post_cost_nav(self):
        histories = {"A": bars([(10, 10)] * 2), "B": bars([(200, 200)] * 2)}
        result = run(histories, D[:2], lambda d, h, s: {"A": .5, "B": .5}, D[0], D[1])
        trade = result["trades"][0]
        self.assertAlmostEqual(trade["weights_after"]["A"], .5)
        self.assertAlmostEqual(trade["weights_after"]["B"], .5)
        self.assertGreaterEqual(trade["cash_after"], 0)
        self.assertAlmostEqual(trade["nav_after"], 1 / (1.0001 * 1.001))

    def test_partial_target_retains_uninvested_cash(self):
        histories = {"A": bars([(100, 100), (100, 200)])}
        result = run(histories, D[:2], lambda d, h, s: {"A": .5} if d == D[0] else None,
                     D[0], D[1], fee=0, slippage=0)
        self.assertAlmostEqual(result["final_state"]["cash"], .5)
        self.assertAlmostEqual(result["nav"], 1.5)

    def test_missing_target_bar_defers_entire_rebalance_and_none_retains_pending(self):
        histories = {"A": bars([(100, 100), (100, 100), (100, 110), (120, 120)]),
                     "B": bars([(100, 100), (100, 100)], [D[0], D[3]]),
                     "C": bars([(100, 100)] * 4)}
        states = {}

        def policy(date, observed, state):
            states[date] = state
            if date == D[0]:
                return {"A": 1}
            if date == D[1]:
                return {"B": .5, "C": .5}
            return None

        result = run(histories, D[:4], policy, D[0], D[3], fee=0, slippage=0)
        self.assertEqual(states[D[2]]["holdings"], ("A",))
        self.assertTrue(states[D[2]]["execution_deferred"])
        self.assertEqual(states[D[3]]["holdings"], ("B", "C"))
        self.assertEqual(result["diagnostics"]["deferred_count"], 1)
        self.assertEqual([t["date"] for t in result["trades"]], [D[1], D[3]])
        self.assertEqual(result["trades"][-1]["signal_date"], D[1])
        self.assertAlmostEqual(result["nav"], 1.2)

    def test_missing_held_bar_carries_mark_and_resume_counts_entire_gap(self):
        histories = {"A": bars([(100, 100), (100, 110), (132, 132)], [D[0], D[1], D[3]]),
                     "B": bars([(200, 200), (200, 200), (200, 200), (200, 210)])}
        states = {}

        def policy(date, observed, state):
            states[date] = state
            return {"A": 1} if date == D[0] else ({"B": 1} if date == D[1] else None)

        result = run(histories, D[:4], policy, D[0], D[3], fee=0, slippage=0)
        self.assertAlmostEqual(result["daily"][2][1], 1.1)
        self.assertAlmostEqual(result["nav"], 1.386)
        self.assertEqual(states[D[2]]["missing_held_bars"], ("A",))
        self.assertEqual(states[D[3]]["holding_since"], {"B": D[3]})

    def test_new_instruction_replaces_deferred_order_using_actual_state(self):
        histories = {"A": bars([(100, 100)] * 4),
                     "B": bars([(100, 100), (100, 100)], [D[0], D[3]])}

        def policy(date, observed, state):
            if date == D[0]:
                return {"A": 1}
            if date == D[1]:
                return {"B": 1}
            if state["execution_deferred"]:
                self.assertEqual(state["holding"], "A")
                return {}  # Abandon the unavailable B order and liquidate next open.
            return None

        result = run(histories, D[:4], policy, D[0], D[3], fee=0, slippage=0)
        self.assertEqual(result["final_state"]["units"], {})
        self.assertFalse(any(f["code"] == "B" for t in result["trades"] for f in t["fills"]))

    def test_observed_history_is_dated_read_only_and_future_invariant(self):
        histories = {"A": bars([(100, 100)] * 4)}
        original = copy.deepcopy(histories)
        seen = []

        def policy(date, observed, state):
            self.assertEqual(observed["A"][-1][0], date)
            self.assertTrue(all(row[0] <= date for row in observed["A"][:]))
            with self.assertRaises(IndexError):
                observed["A"][len(observed["A"])]
            with self.assertRaises(TypeError):
                observed["A"][0][1] = 999
            seen.append((date, len(observed["A"]), state["previous_date"], state["trading_index"]))
            return {"A": 1}

        result = run(histories, D[:4], policy, D[1], D[2], fee=0, slippage=0)
        self.assertEqual(seen[0], (D[1], 2, D[0], 1))
        self.assertEqual(histories, original)
        histories["A"][-1] = (D[3], 9999, 9999, 100)
        changed = run(histories, D[:4], lambda d, h, s: {"A": 1}, D[1], D[2], fee=0, slippage=0)
        self.assertEqual(result["daily"], changed["daily"])

    def test_invalid_targets_and_trailing_data_cutoff_are_rejected(self):
        histories = {"A": bars([(100, 100)] * 3)}
        for target in ({"A": -1}, {"A": 1.01}, {"A": float("nan")}, {"X": 1}, {"A": True}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                run(histories, D[:3], lambda d, h, s: target, D[0], D[2])
        with self.assertRaisesRegex(ValueError, "coverage"):
            run(histories, D[:4], lambda d, h, s: None, D[0], D[3])

    def test_policy_mutating_its_state_cannot_change_portfolio(self):
        histories = {"A": bars([(100, 100)] * 3)}

        def policy(date, observed, state):
            state["units"]["A"] = 1e10
            state["cash"] = -1e10
            state["weights"]["A"] = 100
            return {"A": 1} if date == D[0] else None

        result = run(histories, D[:3], policy, D[0], D[2], fee=0, slippage=0)
        self.assertAlmostEqual(result["nav"], 1)
        self.assertAlmostEqual(result["final_state"]["units"]["A"], .01)


if __name__ == "__main__":
    unittest.main()
