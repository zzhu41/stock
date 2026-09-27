"""Regression checks for research baselines and effective perturbations (offline)."""
import contextlib
import importlib
import inspect
import io
import unittest
from unittest.mock import patch

import backtest
import crash_check
import experiments15
import market_data
import pool_test_v91
import strategy
import v9_robustness
from strategy_versions import V9, V91, BASE_CODES, backtest_kwargs


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.original_globals = {
            name: value for name, value in vars(strategy).items()
            if name.isupper() or name == "_state_bull"
        }
        self.days = ["2024-01-02", "2024-01-03", "2024-01-04"]
        self.histories = {
            code: [(day, 1.0, 1.0, 1.0) for day in self.days]
            for code in BASE_CODES
        }

    def tearDown(self):
        for name, value in self.original_globals.items():
            setattr(strategy, name, value)

    def test_configs_pin_every_engine_parameter(self):
        expected = set(inspect.signature(backtest.backtest).parameters) - {
            "histories", "calendar", "start", "end"
        }
        self.assertEqual(set(V9), expected)
        self.assertEqual(set(V91), expected)
        self.assertEqual({key for key in V9 if V9[key] != V91[key]}, {"pool_buffer"})
        self.assertEqual(V9["pool_buffer"], {})
        self.assertEqual(V91["pool_buffer"], {"stock": .02, "global": .03, "gold": .03})

    def test_config_mutation_cannot_contaminate_next_run(self):
        first = backtest_kwargs("v9.1")
        first["pool_buffer"]["global"] = .99
        self.assertEqual(backtest_kwargs("v9.1")["pool_buffer"]["global"], .03)
        with self.assertRaises(TypeError):
            V91["pool_buffer"]["global"] = .99
        with self.assertRaises(ValueError):
            backtest_kwargs("v9", crash_lok=3)

    def test_importing_scripts_does_not_fetch_or_backtest(self):
        with patch.object(market_data, "fetch_history", side_effect=AssertionError("network")), \
                patch.object(backtest, "backtest", side_effect=AssertionError("execution")):
            for module in (crash_check, experiments15, pool_test_v91, v9_robustness):
                importlib.reload(module)

    def test_crash_comparison_has_a_real_disabled_control(self):
        table = [("159915", {"mom5": -.10, "dist_ma250": -.25})]
        with patch.object(strategy, "rank", return_value=table), \
                patch.object(strategy, "decide", return_value=(market_data.CASH, "fixture")):
            control = crash_check.run(self.histories, self.days, include_crash=False)
            candidate = crash_check.run(self.histories, self.days)
        self.assertEqual(control["crash_buys"], [])
        self.assertEqual(candidate["crash_buys"], [(self.days[0], "159915")])

    @staticmethod
    def indicator(momentum, score):
        return dict(mom20=momentum, mom20_max=momentum, mom5=.01,
                    dist_ma250=.1, ret1=0.0, above_ma=True, ma_rising=True,
                    pos_frac=1.0, max_ret=0.0, bias20=0.0, pctb=.5,
                    cci=0.0, vol_in_ratio=1.0, score=score)

    def rotation(self, challenger, difference, kwargs):
        incumbent = "510300"
        first = [(incumbent, self.indicator(.10, 2.0)),
                 (challenger, self.indicator(.09, 1.0))]
        later = [(challenger, self.indicator(.10 + difference, 2.0)),
                 (incumbent, self.indicator(.10, 1.0))]
        tables = [first, later, later]
        with patch.object(strategy, "rank", side_effect=tables):
            result = backtest.backtest(self.histories, self.days, start=self.days[0], **kwargs)
        return result["daily"][1][2]

    def test_buffer_axis_changes_actual_decision(self):
        low, _ = v9_robustness.perturbation_parameters((1, 1, 1, .8, 1))
        high, _ = v9_robustness.perturbation_parameters((1, 1, 1, 1.2, 1))
        self.assertEqual(self.rotation("159915", .02, low), "159915")
        self.assertEqual(self.rotation("159915", .02, high), "510300")

    def test_v9_and_v91_pool_buffers_produce_distinct_decisions(self):
        self.assertEqual(self.rotation("513100", .025, backtest_kwargs("v9")), "513100")
        self.assertEqual(self.rotation("513100", .025, backtest_kwargs("v9.1")), "510300")

    def test_experiment15_control_is_v9(self):
        fake = dict(ann=.1, max_dd=-.1, nav=1.1, daily=[])
        with patch.object(backtest, "backtest", return_value=fake) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            experiments15.show("baseline", self.histories, self.days)
        for call in run.call_args_list:
            self.assertEqual(call[1]["pool_buffer"], {})

    def test_pool_validation_rejects_outside_holding_and_restores_globals(self):
        original_stock, original_global = strategy.STOCK_POOL, strategy.GLOBAL_POOL
        invalid = {"daily": [(self.days[0], 1.0, "159915")]}
        with patch.object(backtest, "backtest", return_value=invalid), \
                self.assertRaisesRegex(AssertionError, "excluded"):
            pool_test_v91.run_pool(self.histories, self.days, [], ["513100"])
        self.assertIs(strategy.STOCK_POOL, original_stock)
        self.assertIs(strategy.GLOBAL_POOL, original_global)

    def test_wls_patch_restores_after_error(self):
        original = strategy.indicators
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with v9_robustness.wls_window(30):
                self.assertIsNot(strategy.indicators, original)
                raise RuntimeError("fixture")
        self.assertIs(strategy.indicators, original)


if __name__ == "__main__":
    unittest.main()
