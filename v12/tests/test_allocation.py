"""Synthetic-only units, time alignment and native boundary checks."""
from copy import deepcopy
from datetime import date, timedelta
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v12.allocation import CASH, prior_annualized_volatility, run_allocation
from v10_deep.portfolios import run_weights, prices_from_features
from v10_deep import native
from v10_deep.tests.test_native import inputs, prices as set_prices, finish, config, A, B, AI


def dates(n):
    return [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]


def run(prices, tape, **kwargs):
    count = len(tape)
    kwargs.setdefault("provenance", {"base_lag": kwargs.get("lag", 0)})
    start, end = kwargs.pop("start", 0), kwargs.pop("end", count - 1)
    return run_allocation(prices, dates(count), tape, start, end, **kwargs)


class AllocationAccountingTests(unittest.TestCase):
    def test_progressive_uses_actual_drift_and_pays_real_security_turnover(self):
        result = run({"A": [100, 120], CASH: [100, 100]}, ["A", "A"], mode="progressive", speed=.5, fee=.01)
        first, second = result["trades"]
        self.assertTrue(first["initial_session_free"])
        self.assertEqual(first["model_fee"], 0.)
        self.assertAlmostEqual(second["marked_weights"]["A"], 6/11)
        self.assertAlmostEqual(second["target"]["A"], 17/22)
        self.assertAlmostEqual(second["target"][CASH], 5/22)
        self.assertAlmostEqual(second["turnover"], 5/11)
        self.assertAlmostEqual(second["model_fee"], .005)
        self.assertAlmostEqual(result["nav"], 1.095)
        self.assertNotAlmostEqual(result["nav"], 1.1)  # Free constant-weight return averaging.
        self.assertNotAlmostEqual(second["target"]["A"], .75)  # Old desired weights are not current weights.

    def test_skip_preserves_units_while_explicit_cash_moves_toward_cash_etf(self):
        prices = {"A": [100, 120, 120], CASH: [100, 101, 101]}
        result = run(prices, ["A", None, CASH], mode="progressive", speed=.5, fee=0.)
        self.assertEqual(result["allocation_decisions"][1]["status"], "skip_no_signal")
        self.assertAlmostEqual(result["daily"][1][1], 1.105)
        self.assertEqual(len(result["trades"]), 2)
        self.assertGreater(result["final_state"]["weights"]["A"], 0.)
        self.assertAlmostEqual(result["final_state"]["weights"]["A"], .5 * (.6 / 1.105))
        self.assertEqual(result["final_state"]["cash"], 0.)
        self.assertFalse(result["overlay_metadata"]["panic_or_crash_execution_exception"])

    def test_missing_target_blocks_whole_trade_and_does_not_queue_old_target(self):
        result = run({"A": [100, 110, 120], "B": [100, None, 200], CASH: [100]*3},
                     ["A", "B", "A"], fee=.001)
        self.assertEqual(result["diagnostics"]["blocked_rebalance_count"], 1)
        self.assertEqual(result["diagnostics"]["blocked_rebalances"][0]["missing_codes"], ["B"])
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(set(result["final_state"]["units"]), {"A"})
        self.assertAlmostEqual(result["nav"], 1.2)
        self.assertEqual(result["diagnostics"]["total_model_fee"], 0.)

    def test_missing_held_leg_carries_mark_and_resume_catches_up(self):
        result = run({"A": [100, None, 130], "B": [100]*3, CASH: [100]*3}, ["A", "B", "B"], fee=.001)
        self.assertEqual(result["diagnostics"]["blocked_rebalance_count"], 1)
        self.assertAlmostEqual(result["daily"][1][1], 1.)
        self.assertAlmostEqual(result["nav"], 1.3 * .998)
        self.assertEqual(set(result["final_state"]["units"]), {"B"})

    def test_cash_etf_missing_blocks_fractional_start_instead_of_inventing_cash_return(self):
        result = run({"A": [100, 100], CASH: [None, 100]}, ["A", "A"],
                     mode="progressive", speed=.5, fee=.001)
        self.assertEqual(result["diagnostics"]["blocked_rebalance_count"], 1)
        self.assertEqual(result["daily"][0][2], {})
        self.assertAlmostEqual(result["nav"], .999)
        self.assertFalse(result["trades"][0]["initial_session_free"])

    def test_full_and_speed_one_match_frozen_units_engine_with_same_tape_semantics(self):
        prices = {"A": [100, 120, None, 140, 150], "B": [100, 100, 100, 120, 130], CASH: [100,101,102,103,104]}
        tape = ["A", "A", "B", "B", CASH]
        baseline = run_weights(prices, dates(5), lambda i, state: {tape[i]: 1.}, 0, 4, fee=.001)
        before = deepcopy((prices, tape))
        for mode in ("full_exposure", "progressive"):
            result = run(prices, tape, mode=mode, speed=1., fee=.001)
            np.testing.assert_allclose(result["returns"], baseline["returns"], rtol=0, atol=1e-14)
            self.assertEqual(result["final_state"], baseline["final_state"])
            self.assertTrue(result["overlay_metadata"]["native_baseline_fidelity_check_eligible"])
        self.assertEqual((prices, tape), before)


class VolatilityAndClockTests(unittest.TestCase):
    def market(self, n=80):
        a = [100.]
        for i in range(1, n):
            a.append(a[-1] * (1 + (.01 if i % 2 else -.01)))
        return {"A": a, "B": [100. + i for i in range(n)], CASH: [100.]*n}

    def test_prior_sigma_needs_window_returns_and_excludes_signal_day_close(self):
        p = self.market(65)
        for context, window in (("prior20", 20), ("prior60", 60)):
            values = prior_annualized_volatility(p, dates(65), context)["A"]
            self.assertIsNone(values[window])
            expected = np.std(np.diff(p["A"][:window+1]) / np.asarray(p["A"][:window]), ddof=0) * math.sqrt(244)
            self.assertAlmostEqual(values[window+1], expected, places=14)
            changed = deepcopy(p);changed["A"][window+1] *= 4
            other = prior_annualized_volatility(changed, dates(65), context)["A"]
            self.assertEqual(other[window+1], values[window+1])
            if window+2 < 65:self.assertNotEqual(other[window+2], values[window+2])

    def test_prior_own_quotes_skip_missing_bars_instead_of_zero_filling_returns(self):
        p = self.market(25);p["A"][10] = None
        series = prior_annualized_volatility(p, dates(25), "prior20")["A"]
        self.assertIsNone(series[21])
        observed = np.asarray([v for v in p["A"][:22] if v is not None])
        expected = np.std(observed[1:] / observed[:-1] - 1) * math.sqrt(244)
        self.assertAlmostEqual(series[22], expected, places=14)

    def test_vol_target_caps_one_and_records_warmup_and_zero_sigma_cash(self):
        n = 24
        flat = {"A": [100.]*n, CASH: [100.]*n}
        result = run(flat, ["A"]*n, mode="vol_target", target_vol=.2)
        self.assertEqual(set(result["final_state"]["units"]), {CASH})
        self.assertEqual(result["diagnostics"]["sigma_fallback_counts"],
                         {"sigma_warmup_incomplete": 21, "sigma_nonpositive": 3})
        p = self.market(n)
        full = run(p, ["A"]*n, mode="vol_target", target_vol=.50, start=21)
        self.assertEqual(full["allocation_decisions"][0]["desired_weights"], {"A": 1.})
        fractional = run(p, ["A"]*n, mode="vol_target", target_vol=.15, start=21)
        first = fractional["allocation_decisions"][0]
        self.assertAlmostEqual(first["desired_weights"]["A"], .15 / first["sigma_annual"])
        self.assertAlmostEqual(sum(first["desired_weights"].values()), 1.)
        self.assertIn(CASH, first["desired_weights"])

    def test_execution_aligned_lag_is_not_applied_twice_and_sigma_uses_same_signal_day(self):
        n = 26;p = self.market(n)
        tape = [CASH]*n;tape[23] = "A";tape[24] = "B"
        result = run(p, tape, start=23, mode="vol_target", target_vol=.15, lag=1,
                     target_alignment="execution_day", provenance={"base_lag": 1})
        first = result["allocation_decisions"][0]
        self.assertEqual(first["base_target"], "A")
        self.assertEqual(first["target_index"], 23)
        self.assertEqual(first["signal_index"], 22)
        self.assertEqual(first["sigma_last_quote_date"], dates(n)[21])
        observed = np.asarray(p["A"][1:22])
        self.assertAlmostEqual(first["sigma_annual"], np.std(observed[1:]/observed[:-1]-1)*math.sqrt(244), places=14)
        self.assertEqual(result["allocation_decisions"][1]["base_target"], "B")

    def test_explicit_signal_aligned_tape_shifts_once_and_all_bad_provenance_rejects(self):
        p = {"A": [100,110,120], "B": [100,100,100], CASH: [100]*3}
        result = run(p, ["A","B",CASH], lag=1, target_alignment="time_of_signal", provenance={"base_lag":0})
        self.assertEqual([r["base_target"] for r in result["allocation_decisions"]], [None,"A","B"])
        self.assertEqual(result["daily"][0][2], {})
        self.assertFalse(result["overlay_metadata"]["native_baseline_fidelity_check_eligible"])
        for alignment, lag, provenance in (("execution_day",1,{"base_lag":0}),
                ("time_of_signal",1,{"base_lag":1}), ("execution_day",0,{})):
            with self.subTest(alignment=alignment,provenance=provenance), self.assertRaises(ValueError):
                run(p, ["A","B",CASH], lag=lag, target_alignment=alignment, provenance=provenance)

    def test_future_price_and_target_changes_do_not_change_return_prefix(self):
        p = self.market(40);tape = ["A"]*40
        before = run(p, tape, mode="vol_target", target_vol=.15, start=21)
        other = deepcopy(p);other["A"][31:] = [value*2 for value in other["A"][31:]]
        changed_tape = tape[:31] + ["B"]*9
        after = run(other, changed_tape, mode="vol_target", target_vol=.15, start=21)
        np.testing.assert_array_equal(before["returns"][:10], after["returns"][:10])
        self.assertEqual(before["allocation_decisions"][:10], after["allocation_decisions"][:10])

    def test_invalid_prices_params_or_missing_cash_are_rejected(self):
        p = {"A": [100,101], CASH: [100,100]}
        for kwargs in ({"mode":"vol_target","target_vol":.35}, {"mode":"progressive","speed":.2},
                       {"mode":"vol_target","target_vol":.2,"speed":.5}, {"risk_context":"current20"},
                       {"fee":float("nan")}, {"lag":True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):run(p,["A","A"],**kwargs)
        for prices in ({"A":[100,101]}, {"A":[100,0],CASH:[100,100]}, {"A":[100,math.inf],CASH:[100,100]}):
            with self.assertRaises(ValueError):run(prices,["A","A"])

    def test_predeclared_volatility_neighbors_are_allowed_without_new_main_grid(self):
        from v12.allocation import MAIN_TARGET_VOLS
        self.assertEqual(MAIN_TARGET_VOLS, (.15,.20,.25,.30,.40,.50))
        p = self.market(24)
        for target in MAIN_TARGET_VOLS:
            for scale in (.8,1.2):
                result = run(p,["A"]*24,start=21,mode="vol_target",target_vol=target*scale)
                self.assertLessEqual(sum(result["final_state"]["weights"].values()),1+1e-12)
                self.assertFalse(result["overlay_metadata"]["candidate_selection_performed"])


class NativeReplayTests(unittest.TestCase):
    def native_run(self, arrays, meta, configs, start=0):
        # Compilation uses a disposable directory; no old source/cache changes.
        source = (native.BASE / "native.cpp").read_bytes()
        with tempfile.TemporaryDirectory(prefix="v12-native-synthetic-") as directory, patch.object(native,"BASE",Path(directory)):
            (Path(directory)/"native.cpp").write_bytes(source)
            return native.Simulator(arrays,meta).run(configs,meta["dates"][start],meta["dates"][-1],fee=.001,workers=1)

    def test_execution_aligned_full_speed_one_matches_native_with_lag_and_missing_held_day(self):
        arrays,meta = inputs(8)
        set_prices(arrays,A,[100.,101.,102.,104.,np.nan,107.,109.,110.])
        arrays["scores"][:,4:,AI[B]] = 80.
        arrays = finish(arrays)
        result = self.native_run(arrays,meta,[config(lag=0),config(lag=1)],start=2)
        p = prices_from_features(arrays,meta)
        for i,lag in enumerate((0,1)):
            tape = [None,None]+[meta["assets"][h] if h>=0 else None for h in result["holdings"][i]]
            for mode in ("full_exposure","progressive"):
                replay = run_allocation(p,meta["dates"],tape,2,7,mode=mode,speed=1.,lag=lag,fee=.001,
                                        target_alignment="execution_day",provenance={"base_lag":lag})
                np.testing.assert_allclose(replay["returns"],result["returns"][i],rtol=0,atol=1e-13)
                self.assertAlmostEqual(replay["nav"],result["summary"][i,0],places=12)
                self.assertTrue(replay["overlay_metadata"]["native_baseline_fidelity_check_eligible"])

    def test_delayed_first_entry_keeps_true_l1_and_discloses_native_boundary(self):
        arrays,meta = inputs(3)
        arrays["features"][0,:,meta["feature_names"].index("close")] = np.nan
        arrays["features"][0,:,meta["feature_names"].index("valid")] = 0.
        arrays = finish(arrays)
        result = self.native_run(arrays,meta,[config()])
        tape = [meta["assets"][h] if h>=0 else None for h in result["holdings"][0]]
        self.assertIsNone(tape[0])
        replay = run_allocation(prices_from_features(arrays,meta),meta["dates"],tape,0,2,
                                provenance={"base_lag":0},fee=.001)
        self.assertAlmostEqual(replay["returns"][1],-.001)
        self.assertAlmostEqual(result["returns"][0,1],-.002)
        self.assertFalse(replay["overlay_metadata"]["native_baseline_fidelity_check_eligible"])
        self.assertTrue(replay["overlay_metadata"]["delayed_first_fill_uses_true_security_l1_fee"])
        self.assertFalse(replay["overlay_metadata"]["first_successful_fill_was_free"])


if __name__ == "__main__":unittest.main()
