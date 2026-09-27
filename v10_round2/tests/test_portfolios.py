"""Synthetic independent-account arithmetic; never runs registered market data."""
import copy
import math
import unittest

from v10_next.execution import run
from v10_next.metrics import attribution, summarize
from v10_round2.portfolios import run_accounts


D = ("2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07")


def bars(prices, dates=D):
    return [(day, opened, closed, 1000.0) for day, (opened, closed) in zip(dates, prices)]


def config():
    return dict(id="test_accounts", kind="accounts", components=({"id": "one", "kind": "legacy"},
                {"id": "two", "kind": "legacy"}), initial_weights=(.5, .5),
                rebalance="never", netting=False, capital_transfers=False)


def children(histories, instructions, fee=0, slippage=0):
    results = {}
    for name, signals in instructions.items():
        results[name] = run(histories, D, lambda day, observed, state: signals.get(day),
                            D[0], D[-1], fee=fee, slippage=slippage)
    return results


class PortfolioTests(unittest.TestCase):
    def aggregate(self, histories, results, specification=None):
        return run_accounts(specification or config(), histories, D,
                            lambda component: results[component["id"]])

    def test_wealth_weights_drift_without_resets_or_new_rebalancing(self):
        histories = {"A": bars([(100, 100), (100, 200), (200, 400), (400, 400)]),
                     "B": bars([(100, 100)] * 4)}
        result = self.aggregate(histories, children(histories, {"one": {D[0]: {"A": 1}},
                                                               "two": {D[0]: {"B": 1}}}))
        self.assertAlmostEqual(result["nav"], 2.5)
        self.assertAlmostEqual(result["daily"][-1][2]["A"], .8)
        self.assertEqual(result["account_wealth_shares"]["initial"], {"one": .5, "two": .5})
        self.assertAlmostEqual(result["account_wealth_shares"]["final"]["one"], .8)
        self.assertEqual(result["account_wealth_shares"]["periods"]["2018-2021"]["start_wealth_shares"],
                         {"one": .5, "two": .5})
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["diagnostics"]["capital_transfers"], 0)
        self.assertEqual(summarize(result)["fill_legs"], 2)
        self.assertAlmostEqual(sum(attribution(result, histories)["log_growth_by_asset"].values()), math.log(2.5))

    def test_opposing_fills_are_not_netted_and_all_costs_are_paid(self):
        histories = {"A": bars([(100, 100)] * 4), "B": bars([(100, 100)] * 4)}
        books = children(histories, {"one": {D[0]: {"A": 1}, D[1]: {"B": 1}},
                                     "two": {D[0]: {"B": 1}, D[1]: {"A": 1}}}, .0001, .001)
        original = copy.deepcopy(books)
        result = self.aggregate(histories, books)
        self.assertEqual(books, original)
        self.assertEqual(len(result["trades"][1]["fills"]), 4)
        self.assertEqual(result["diagnostics"]["fill_count"], 6)
        self.assertEqual(result["diagnostics"]["rebalance_count"], 2)
        self.assertEqual(result["diagnostics"]["component_rebalance_count"], 4)
        expected_cost = .5 * sum(r["diagnostics"]["total_commission"] for r in books.values())
        self.assertAlmostEqual(result["diagnostics"]["total_commission"], expected_cost)
        self.assertLess(result["nav"], result["daily"][1][1])
        a = attribution(result, histories)
        self.assertLess(a["log_growth_by_asset"]["transaction_costs"], 0)
        self.assertAlmostEqual(sum(a["log_growth_by_asset"].values()), math.log(result["nav"]))

    def test_turnover_uses_opening_wealth_of_the_nontrading_book_too(self):
        histories = {"A": bars([(100, 100), (100, 100), (200, 200), (200, 200)]),
                     "B": bars([(100, 100)] * 4), "C": bars([(100, 100)] * 4)}
        books = children(histories, {"one": {D[0]: {"A": 1}},
                                     "two": {D[0]: {"B": 1}, D[1]: {"C": 1}}})
        result = self.aggregate(histories, books)
        trade = result["trades"][1]
        self.assertAlmostEqual(trade["nav_before"], 1.5)
        self.assertAlmostEqual(trade["turnover"], 2 / 3)
        self.assertAlmostEqual(result["nav"], 1.5)
        attribution(result, histories)

    def test_missing_bar_in_one_account_does_not_block_another_account(self):
        histories = {"A": bars([(100, 100), (100, 100), (120, 130)], (D[0], D[1], D[3])),
                     "B": bars([(100, 100)] * 4), "C": bars([(100, 100)] * 4)}
        books = children(histories, {"one": {D[0]: {"A": 1}},
                                     "two": {D[0]: {"B": 1}, D[1]: {"C": 1}}})
        result = self.aggregate(histories, books)
        self.assertEqual(result["trades"][1]["date"], D[2])
        self.assertAlmostEqual(result["nav"], 1.15)
        self.assertEqual(result["diagnostics"]["missing_held_bars"][0]["codes"], ["A"])
        self.assertEqual(result["diagnostics"]["missing_held_bars"][0]["accounts"], ["one"])
        attribution(result, histories)

    def test_book_pending_instructions_remain_explicit_at_the_endpoint(self):
        histories = {"A": bars([(100, 100)] * 4), "B": bars([(100, 100)] * 4)}
        books = children(histories, {"one": {D[0]: {"A": 1}, D[-1]: {"B": 1}},
                                     "two": {D[0]: {"B": 1}}})
        result = self.aggregate(histories, books)
        state = result["final_state"]
        self.assertTrue(state["independent_accounts"])
        self.assertIsNone(state["pending_target"])
        self.assertEqual(state["pending_account_instructions"],
                         {"one": {"signal_date": D[-1], "target_weights": {"B": 1.0}}})

    def test_rebalanced_or_misaligned_accounts_are_rejected(self):
        histories = {"A": bars([(100, 100)] * 4)}
        books = children(histories, {"one": {D[0]: {"A": 1}}, "two": {D[0]: {"A": 1}}})
        for key, value in (("netting", True), ("rebalance", "monthly"),
                           ("capital_transfers", True), ("initial_weights", (.6, .4))):
            bad = config()
            bad[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.aggregate(histories, books, bad)
        bad_results = copy.deepcopy(books)
        bad_results["two"]["daily"] = bad_results["two"]["daily"][1:]
        with self.assertRaisesRegex(ValueError, "identical trading-date"):
            self.aggregate(histories, bad_results)
        bad_results = copy.deepcopy(books)
        bad_results["two"]["execution_clock"] = "ideal_close"
        with self.assertRaisesRegex(ValueError, "mix execution clocks"):
            self.aggregate(histories, bad_results)


if __name__ == "__main__":
    unittest.main()
