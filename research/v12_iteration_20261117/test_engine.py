"""Portfolio behavior, timing and baseline fidelity; writes no old files."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from research.v12_iteration_20261117.engine import run_portfolio, ResearchPolicy, LOWVOL
from v10_deep.reference import DatedValues, Portfolio
from v10_deep.tests.test_native import inputs, put, prices, config, AI, A, B, X, CASH, GOLD, BENCH
from v10_deep.features import F
from v12.reference import run_reference

ROOT = Path(__file__).resolve().parents[2]


def crash_fixture(days=9):
    arrays, meta = inputs(days)
    arrays["features"][1, AI[B], F["mom5"]] = -.1
    arrays["features"][1, AI[B], F["ma250"]] = -.3
    arrays["scores"][:, 1:, AI[B]] = 80.
    return arrays, meta, config(crash_mask=1, crash_lock=5, panic=0.)


class PortfolioTests(unittest.TestCase):
    def compare(self, arrays, meta, configuration, fee=.0001):
        old = run_reference(arrays, meta, configuration, fee=fee)
        new = run_portfolio(arrays, meta, configuration, fee=fee)
        np.testing.assert_array_equal(new["holdings"], old["holdings"])
        np.testing.assert_allclose(new["returns"], old["returns"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(new["summary"], old["summary"], rtol=1e-12, atol=1e-12)
        return new

    def test_baseline_missing_prices_delayed_entry_and_lock_exit(self):
        for lag in (0, 1):
            for fee in (.0001, .0011):
                arrays, meta, c = crash_fixture()
                prices(arrays, B, [100., 100., 94., 90., np.nan, 92., 93., 94., 95.])
                prices(arrays, CASH, [100., 100., np.nan, 100., 100., 100., 100., 100., 100.])
                for flag in (0, 1):
                    with self.subTest(lag=lag, fee=fee, flag=flag):
                        self.compare(arrays, meta, dict(c, lag=lag, panic=.04, locked_panic_exit=flag), fee)

    def test_seeded_baseline_daily_fidelity(self):
        rng = np.random.default_rng(20261117)
        arrays, meta = inputs(80)
        for code in (A, B, X, GOLD):
            prices(arrays, code, 100*np.exp(np.cumsum(rng.normal(0, .04, 80))))
            for feature in ("mom5", "mom20", "ma250"):
                put(arrays, code, feature, rng.normal(0, .2, 80))
            arrays["scores"][:, :, AI[code]] = rng.normal(0, 20, (2, 80))
        arrays["features"][[12, 33], AI[B], F["close"]] = np.nan
        for lag in (0, 1):
            for fee in (.0001, .0011):
                self.compare(arrays, meta, config(lag=lag, crash_mask=5, locked_panic_exit=1), fee)

    def test_fixed_units_cash_growth_and_no_daily_rebalance_after_lock(self):
        arrays, meta, c = crash_fixture()
        prices(arrays, B, [100., 100., 120., 140., 160., 180., 200., 220., 240.])
        prices(arrays, CASH, [100., 100., 110., 110., 110., 110., 110., 110., 110.])
        out = run_portfolio(arrays, meta, c, {"crash_weight": .5}, fee=0.)
        np.testing.assert_array_equal(out["units"][1:], np.tile(out["units"][1], (8, 1)))
        self.assertAlmostEqual(out["trace"][2]["nav"], 1.15)
        self.assertAlmostEqual(out["weights"][2, AI[B]], .6/1.15)
        self.assertAlmostEqual(out["trace"][-1]["nav"], 1.75)
        self.assertEqual(out["trace"][6]["lock_until"], 6)
        self.assertEqual(out["trace"][6]["holding"], B)
        self.assertFalse(out["trace"][6]["filled"])

    def test_partial_risk_to_cash_cost_uses_drifted_security_turnover(self):
        arrays, meta, c = crash_fixture(5)
        c.update(crash_lock=1)
        prices(arrays, B, [100., 100., 120., 120., 120.])
        prices(arrays, CASH, [100., 100., 110., 110., 110.])
        for code in (A, B, X, GOLD):
            arrays["features"][2:, AI[code], F["mom20"]] = -.1
        out = run_portfolio(arrays, meta, c, {"crash_weight": .5}, fee=.0011)
        entry = 1 - .0022
        self.assertAlmostEqual(out["trace"][2]["pretrade_nav"], entry*1.15)
        self.assertAlmostEqual(out["turnover"][2], 1.2/1.15)
        self.assertAlmostEqual(out["costs"][2], .0011*entry*1.2)
        self.assertEqual(out["trace"][2]["holding"], CASH)
        self.assertAlmostEqual(out["trace"][2]["nav"], entry*1.15-.0011*entry*1.2)

    def test_partial_cash_missing_carries_mark_and_blocks_trade(self):
        arrays, meta, c = crash_fixture(5)
        c.update(crash_lock=1)
        prices(arrays, B, [100., 100., 120., 130., 140.])
        prices(arrays, CASH, [100., 100., np.nan, 110., 110.])
        arrays["scores"][:, 2:, AI[A]] = 200.
        out = run_portfolio(arrays, meta, c, {"crash_weight": .5}, fee=0.)
        self.assertEqual(out["trace"][2]["holding"], B)
        self.assertAlmostEqual(out["trace"][2]["nav"], 1.1)
        self.assertEqual(out["trace"][2]["blocked_missing"], [CASH])
        self.assertEqual(out["trace"][3]["holding"], A)
        self.assertAlmostEqual(out["trace"][3]["nav"], 1.2)
        self.assertEqual(out["cash_missing"][0]["kind"], "held_cash_mark_carried")

    def test_missing_cash_prevents_fractional_entry_and_no_lock_is_invented(self):
        arrays, meta, c = crash_fixture(5)
        prices(arrays, CASH, [100., np.nan, 100., 100., 100.])
        out = run_portfolio(arrays, meta, c, {"crash_weight": .5}, fee=0.)
        self.assertEqual(out["trace"][1]["holding"], A)
        self.assertFalse(out["trace"][1]["filled"])
        self.assertEqual(out["trace"][1]["lock_until"], -1)
        self.assertEqual(out["trace"][2]["holding"], B)
        self.assertFalse(out["trace"][2]["crash_requested"])
        self.assertEqual(out["weights"][2, AI[B]], 1.)

    def test_first_session_crash_is_free_and_weight_only_changes_on_target_fill(self):
        arrays, meta, c = crash_fixture(5)
        arrays["features"][0, AI[B], F["mom5"]] = -.1
        arrays["features"][0, AI[B], F["ma250"]] = -.3
        out = run_portfolio(arrays, meta, c, {"crash_weight": .75}, fee=.0011)
        self.assertEqual(out["returns"][0], 0.)
        self.assertEqual(out["costs"][0], 0.)
        self.assertEqual(out["weights"][0, AI[B]], .75)
        self.assertEqual(out["weights"][0, AI[CASH]], .25)

    def test_lock_panic_exit_reuses_policy_and_only_clears_after_fill(self):
        arrays, meta, c = crash_fixture()
        c.update(panic=.04, locked_panic_exit=1)
        prices(arrays, B, [100., 100., 94., 90., 91., 92., 93., 94., 95.])
        prices(arrays, CASH, [100., 100., np.nan, 100., 100., 100., 100., 100., 100.])
        out = run_portfolio(arrays, meta, c, {"crash_weight": .5}, fee=0.)
        self.assertTrue(out["trace"][2]["locked_panic_exit_requested"])
        self.assertEqual(out["trace"][2]["lock_until"], 6)
        self.assertFalse(out["trace"][2]["filled"])
        self.assertEqual(out["trace"][3]["holding"], CASH)
        self.assertEqual(out["trace"][3]["lock_until"], -1)

    def test_future_changes_and_lag1_signal_clock(self):
        arrays, meta, c = crash_fixture()
        c["lag"] = 1
        spec = {"crash_weight": .5, "fallback_hysteresis": .02}
        first = run_portfolio(arrays, meta, c, spec, fee=.0011)
        changed = deepcopy(arrays)
        changed["features"][5:, :, :] *= 3
        changed["scores"][:, 5:, :] *= -4
        second = run_portfolio(changed, meta, c, spec, fee=.0011)
        np.testing.assert_array_equal(first["returns"][:5], second["returns"][:5])
        np.testing.assert_array_equal(first["weights"][:5], second["weights"][:5])
        self.assertEqual(first["trace"][1]["holding"], A)
        self.assertEqual(first["trace"][2]["holding"], B)
        self.assertEqual(first["trace"][2]["signal_date"], meta["dates"][1])
        self.assertEqual(first["trace"][2]["lock_until"], 7)

    def test_fee_approximation_disclosed_and_difference_bounded_for_two_scenarios(self):
        arrays, meta = inputs(3)
        for fee in (.0001, .0011):
            out = run_portfolio(arrays, meta, config(), fee=fee)
            m = out["metadata"]
            diff = m["full_switch_exact_self_financing_factor"]-m["full_switch_legacy_factor"]
            self.assertAlmostEqual(diff, 2*fee*fee/(1+fee), places=15)
            self.assertLess(diff, .0000025)


class PolicyOverlayTests(unittest.TestCase):
    def setup_bear(self):
        arrays, meta = inputs(6)
        put(arrays, BENCH, "ma250", -.1)
        for code in (A, B, X, GOLD):
            put(arrays, code, "mom20", .01)
        c = config(fallback_mode="rank", fallback_floor=-9.9, buffer=0., bear_entry=.07)
        return arrays, meta, c

    def test_hysteresis_entry_exit_thresholds_and_preserved_fallback_ranking(self):
        arrays, meta, c = self.setup_bear()
        put(arrays, X, "mom20", [.02, .021, .01, 0., -.001, .02])
        put(arrays, GOLD, "mom20", [-.1]*6)
        out = run_portfolio(arrays, meta, c, {"fallback_hysteresis": .02}, fee=0.)
        self.assertEqual([x["holding"] for x in out["trace"]], [CASH, X, X, X, CASH, CASH])
        # Eligible lower-ranked gold wins while higher-ranked new X is weak.
        put(arrays, GOLD, "mom20", [.03]*6)
        self.assertEqual(run_portfolio(arrays, meta, c, {"fallback_hysteresis": .02}, fee=0.)["trace"][0]["holding"], GOLD)

    def test_hysteresis_does_not_filter_regular_entrants(self):
        arrays, meta = inputs(3)
        put(arrays, A, "mom20", .01)
        out = run_portfolio(arrays, meta, config(), {"fallback_hysteresis": .02}, fee=0.)
        self.assertEqual(out["trace"][0]["holding"], A)
        self.assertFalse(out["trace"][0]["fallback_used"])

    def test_defensive_uses_own_momentum_keeps_stock_crash_role_and_exit_rules(self):
        arrays, meta, c = self.setup_bear()
        c["stock_pool"].append(LOWVOL)
        put(arrays, LOWVOL, "valid", 1.)
        put(arrays, LOWVOL, "mom20", .08)
        arrays["scores"][:, :, AI[LOWVOL]] = 40.
        old = run_portfolio(arrays, meta, c, fee=0.)
        new = run_portfolio(arrays, meta, c, {"defensive_lowvol": True}, fee=0.)
        self.assertNotEqual(old["trace"][0]["holding"], LOWVOL)
        self.assertEqual(new["trace"][0]["holding"], LOWVOL)
        policy = ResearchPolicy(DatedValues(arrays, meta, c), c, {"defensive_lowvol": True})
        self.assertIn(LOWVOL, policy.stocks)
        self.assertIn(LOWVOL, policy.safe_pool)
        self.assertFalse(policy.must_exit(0, LOWVOL, False))
        put(arrays, LOWVOL, "mom20", -.01)
        self.assertTrue(policy.must_exit(0, LOWVOL, False))
        put(arrays, LOWVOL, "mom20", .5)
        put(arrays, LOWVOL, "mom5", -.01)
        self.assertTrue(policy.must_exit(0, LOWVOL, False))


class RealBaselineTests(unittest.TestCase):
    def test_full_frozen_r2_all_four_execution_scenarios_match(self):
        from v12_r2.features import effective_config, view
        registry = json.loads((ROOT/"v12_r2/registered_candidates.json").read_text())
        record = next(r for r in registry["records"] if r["id"] == "v12r2_08b522c2e1aa291f054d")
        meta = json.loads((ROOT/"v12_r2/results/feature_metadata.json").read_text())
        with np.load(ROOT/"v12_r2/results/features.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays = view(arrays, meta, record)
        for lag in (0, 1):
            for fee in (.0001, .0011):
                with self.subTest(lag=lag, fee=fee):
                    c = effective_config(record, lag)
                    old = run_reference(arrays, meta, c, start="2014-01-02", end="2026-09-24", fee=fee)
                    new = run_portfolio(arrays, meta, c, start="2014-01-02", end="2026-09-24", fee=fee)
                    self.assertEqual(len(new["returns"]), 3097)
                    np.testing.assert_array_equal(new["holdings"], old["holdings"])
                    np.testing.assert_allclose(new["returns"], old["returns"], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
