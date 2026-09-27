"""Small independent-state oracle tests; never build or alter real feature data."""
from copy import deepcopy
from datetime import date, timedelta
import unittest

import numpy as np

from v10_deep.data import ASSET_ORDER
from v10_deep.features import FEATURE_NAMES, F
from v10_deep.native import Simulator
from v10_deep.reference import run_reference
from v10_deep.schema import baseline


A, B, X, GOLD, CASH, BENCH = "159915", "510500", "513100", "518880", "511880", "510300"
AI = {code: index for index, code in enumerate(ASSET_ORDER)}


def inputs(days=8):
    count = len(ASSET_ORDER)
    cube = np.full((days, count, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    cube[:, :, F["close"]] = 100.
    cube[:, :, F["valid"]] = 0.
    for code in (A, B, X, GOLD, BENCH, CASH):
        cube[:, AI[code], F["valid"]] = 1.
    for name, index in F.items():
        if name.startswith("mom"):
            cube[:, :, index] = .01 if name == "mom5" else .10
        elif name.startswith("vol") and name != "volume_ratio":
            cube[:, :, index] = .02
        elif name.startswith("ma") or name.startswith("ema"):
            cube[:, :, index] = .10
        elif name.startswith("dd"):
            cube[:, :, index] = 0.
    cube[:, :, F["ret1"]] = 0.
    cube[:, :, F["volume_ratio"]] = 1.
    cube[:, :, F["er20"]] = .5
    scores = np.full((2, days, count), -50., dtype=np.float64)
    for code, score in ((A, 30.), (B, 20.), (X, 10.), (GOLD, 0.), (BENCH, 100.)):
        scores[:, :, AI[code]] = score
    arrays = dict(features=cube, scores=scores, fear=np.zeros(days, dtype=np.int32))
    dates = [(date(2020, 1, 2) + timedelta(days=i)).isoformat() for i in range(days)]
    meta = dict(assets=list(ASSET_ORDER), dates=dates, feature_names=list(FEATURE_NAMES),
                score_names=["wls25_v20", "wls30_v20"])
    return arrays, meta


def put(arrays, code, feature, values):
    arrays["features"][:, AI[code], F[feature]] = values


def prices(arrays, code, values):
    put(arrays, code, "close", values)
    last, rets = None, []
    for price in values:
        rets.append(price / last - 1 if last is not None and np.isfinite(price) else 0.)
        if np.isfinite(price):
            last = price
    put(arrays, code, "ret1", rets)
    missing = ~np.isfinite(np.asarray(values))
    arrays["features"][missing, AI[code], F["valid"]] = 0.


def finish(arrays):
    valid = (arrays["features"][:, :, F["valid"]] > .5) & np.isfinite(arrays["features"][:, :, F["close"]])
    arrays["scores"][:, ~valid] = -1e100
    arrays["scores"][:, :, AI[CASH]] = -1e100
    arrays["orders"] = np.ascontiguousarray(np.argsort(-arrays["scores"], axis=2, kind="stable"), dtype=np.int32)
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def config(**changes):
    result = baseline()
    result.update(stock_pool=[A, B], global_pool=[X], buffer=0., global_buffer=-1., gold_buffer=-1.,
                  fallback_mode="cash", crash_mask=0)
    result.update(changes)
    return result


class NativeOracleTests(unittest.TestCase):
    def compare(self, arrays, meta, configs, begin=0, stop=None, fee=.0001, workers=2):
        arrays = finish(arrays)
        stop = len(meta["dates"]) - 1 if stop is None else stop
        native = Simulator(arrays, meta).run(configs, start=meta["dates"][begin], end=meta["dates"][stop],
                                             fee=fee, workers=workers, capture=True)
        references = []
        for index, candidate in enumerate(configs):
            reference = run_reference(arrays, meta, candidate, meta["dates"][begin], meta["dates"][stop], fee)
            message = "configuration %d: %r" % (index, candidate)
            np.testing.assert_array_equal(native["holdings"][index], reference["holdings"], err_msg=message)
            np.testing.assert_allclose(native["returns"][index], reference["returns"], rtol=1e-11, atol=1e-12, err_msg=message)
            np.testing.assert_allclose(native["summary"][index], reference["summary"], rtol=1e-10, atol=1e-11, err_msg=message)
            references.append(reference)
        return native, references

    @staticmethod
    def names(holding):
        return [ASSET_ORDER[index] if index >= 0 else None for index in holding]

    def test_missing_held_bar_preserves_mark_and_initial_session_is_free(self):
        arrays, meta = inputs(7)
        prices(arrays, A, [100., 105., np.nan, 120., 121., 122., 123.])
        arrays["scores"][:, 2:, AI[B]] = 80.
        for code in (A, B, X, GOLD):
            arrays["features"][4:, AI[code], F["mom20"]] = -.1
        native, _ = self.compare(arrays, meta, [config()])
        self.assertEqual(self.names(native["holdings"][0]), [A, A, A, B, CASH, CASH, CASH])
        self.assertEqual(native["returns"][0, 0], 0.)
        wealth = np.cumprod(1 + native["returns"][0])
        self.assertAlmostEqual(wealth[3], 1.2 * .9998)
        self.assertAlmostEqual(wealth[4], 1.2 * .9998 ** 2)
        self.assertEqual(tuple(native["summary"][0, 3:7]), (2., 3., 1., 1.))
        later, _ = self.compare(arrays, meta, [config()], begin=3)
        self.assertEqual(later["returns"][0, 0], 0.)

    def test_confirmation_requires_consecutive_healthy_proposals_and_resets(self):
        arrays, meta = inputs(6)
        arrays["scores"][:, [1, 3, 4, 5], AI[B]] = 80.
        native, _ = self.compare(arrays, meta, [config(switch_confirm=2)])
        self.assertEqual(self.names(native["holdings"][0]), [A, A, A, A, B, B])
        arrays, meta = inputs(6)
        arrays["scores"][:, 1:, AI[B]] = 80.
        put(arrays, B, "mom20", [.1, .13, .11, .13, .13, .13])
        native, _ = self.compare(arrays, meta, [config(switch_confirm=2, buffer=.02)])
        self.assertEqual(self.names(native["holdings"][0]), [A, A, A, A, B, B])

    def test_panic_negative_momentum_and_regime_exit_bypass_confirmation_and_min_hold(self):
        for event in ("panic", "momentum", "bear"):
            with self.subTest(event=event):
                arrays, meta = inputs(5)
                arrays["scores"][:, 1:, AI[B]] = 80.
                if event == "panic":
                    prices(arrays, A, [100., 100., 94., 94., 94.])
                elif event == "momentum":
                    arrays["features"][2:, AI[A], F["mom20"]] = -.1
                else:
                    arrays["features"][2:, AI[BENCH], F["ma250"]] = -.1
                native, _ = self.compare(arrays, meta, [config(switch_confirm=4, min_hold=10)])
                self.assertEqual(self.names(native["holdings"][0])[:3], [A, A, X if event == "bear" else B])

    def test_crash_override_has_five_close_intervals_and_ignores_normal_exit_while_locked(self):
        arrays, meta = inputs(8)
        arrays["features"][1, AI[B], F["mom5"]] = -.1
        arrays["features"][1, AI[B], F["ma250"]] = -.3
        arrays["features"][2:, AI[B], F["mom20"]] = -.1
        prices(arrays, B, [100., 100., 95., 96., 97., 98., 99., 100.])
        native, _ = self.compare(arrays, meta, [config(crash_mask=1, crash_lock=5, switch_confirm=3, min_hold=10)])
        self.assertEqual(self.names(native["holdings"][0]), [A, B, B, B, B, B, A, A])
        self.assertEqual(native["summary"][0, 7], 1.)

    def test_lagged_crash_without_current_buy_quote_blocks_whole_order_and_creates_no_lock(self):
        arrays, meta = inputs(6)
        arrays["features"][1, AI[B], F["mom5"]] = -.1
        arrays["features"][1, AI[B], F["ma250"]] = -.3
        arrays["scores"][:, 1:, AI[X]] = 90.
        put(arrays, X, "mom20", [.1, .2, .2, .2, .2, .2])
        prices(arrays, B, [100., 100., np.nan, 100., 100., 100.])
        native, reference = self.compare(arrays, meta, [config(lag=1, crash_mask=1)])
        self.assertEqual(self.names(native["holdings"][0])[:4], [None, A, A, X])
        self.assertEqual(native["summary"][0, 5], 1.)
        self.assertEqual(native["summary"][0, 7], 0.)
        self.assertTrue(reference[0]["trace"][2]["crash_requested"])
        self.assertEqual(reference[0]["trace"][2]["lock_until"], -1)

    def test_missing_lagged_risk_features_do_not_pass_cash_or_health_guards(self):
        arrays, meta = inputs(6)
        prices(arrays, A, [100., 100., np.nan, 100., 100., 100.])
        arrays["features"][2, AI[B], F["mom5"]] = -.1
        arrays["features"][2, AI[B], F["ma250"]] = -.3
        arrays["scores"][:, 2:, AI[X]] = 90.
        put(arrays, X, "mom20", [.1, .1, .2, .2, .2, .2])
        guards = ("always", "cash_only", "held_negative", "candidate_better", "held_shock")
        native, _ = self.compare(arrays, meta, [config(lag=1, crash_mask=1, crash_guard=guard) for guard in guards])
        self.assertEqual(self.names(native["holdings"][0])[3], B)
        for i in range(1, len(guards)):
            self.assertEqual(self.names(native["holdings"][i])[3], X)
            self.assertEqual(native["summary"][i, 7], 0.)

    def test_cash_uses_zero_signal_value_not_its_score_sentinel_or_own_momentum(self):
        arrays, meta = inputs(4)
        arrays["scores"][:, :, AI[A]] = -.5
        put(arrays, A, "mom20", [.3, .1, .1, .1])
        put(arrays, GOLD, "mom20", -.1)
        put(arrays, CASH, "mom20", .9)  # Even a falsely marked valid cash row is not a ranking asset.
        variants = [config(bull_entry=.2, buffer_mode=mode, buffer=.1 if mode != "momentum" else 0.)
                    for mode in ("score_gap", "score_relative", "momentum")]
        native, _ = self.compare(arrays, meta, variants)
        self.assertEqual(self.names(native["holdings"][0])[:2], [A, CASH])
        self.assertEqual(self.names(native["holdings"][1])[:2], [A, CASH])
        self.assertEqual(self.names(native["holdings"][2])[:2], [A, A])

    def test_trailing_stop_and_cooldown_are_based_on_actual_fills_for_both_lags(self):
        arrays, meta = inputs(8)
        prices(arrays, A, [100., 120., 113., 114., 116., 118., 121., 124.])
        native, _ = self.compare(arrays, meta, [config(lag=lag, panic=0., trail_stop=.05, panic_cooldown=2) for lag in (0, 1)])
        self.assertEqual(self.names(native["holdings"][0])[:6], [A, A, B, B, B, A])
        self.assertEqual(self.names(native["holdings"][1])[:7], [None, A, A, B, B, B, A])

    def test_breadth_confirmation_and_ma_hysteresis_preserve_regime_state(self):
        arrays, meta = inputs(5)
        put(arrays, A, "ma250", [.1, -.1, -.1, .1, .1])
        native, _ = self.compare(arrays, meta, [config(regime="breadth", breadth_threshold=.75, regime_confirm=2)])
        self.assertEqual(self.names(native["holdings"][0]), [A, A, X, X, A])
        arrays, meta = inputs(5)
        put(arrays, BENCH, "ma250", [.01, -.01, -.03, .01, .03])
        native, _ = self.compare(arrays, meta, [config(regime_hyst=.02)])
        self.assertEqual(self.names(native["holdings"][0]), [A, A, X, X, A])

    def test_fallback_floor_low_volatility_and_price_vs_feature_warmup(self):
        arrays, meta = inputs(5)
        for code in (A, B, X, GOLD):
            put(arrays, code, "mom20", -.01)
        put(arrays, GOLD, "vol20", .005)
        put(arrays, GOLD, "mom20", -.03)
        native, _ = self.compare(arrays, meta, [config(fallback_mode="rank", fallback_floor=-.02),
                                                 config(fallback_mode="low_vol", fallback_floor=-.02),
                                                 config(fallback_mode="low_vol", fallback_floor=-.04)])
        self.assertEqual([self.names(h)[0] for h in native["holdings"]], [X, CASH, GOLD])
        arrays, meta = inputs(5)
        for code in (A, B, X, GOLD, CASH):
            put(arrays, code, "valid", 0.)
        arrays["features"][2:, AI[A], F["valid"]] = 1.
        native, _ = self.compare(arrays, meta, [config()])
        self.assertEqual(self.names(native["holdings"][0]), [CASH, CASH, A, A, A])

    def test_randomized_combinations_match_independent_decision_state_and_future_prefix(self):
        rng = np.random.default_rng(20260927)
        arrays, meta = inputs(64)
        active = (A, B, X, GOLD, BENCH)
        for code in active:
            path = 100 * np.exp(np.cumsum(rng.normal(0., .035, len(meta["dates"]))))
            prices(arrays, code, path)
            for name in FEATURE_NAMES:
                if name.startswith("mom"):
                    put(arrays, code, name, rng.normal(.025, .13, len(path)))
                elif name.startswith("ma") or name.startswith("ema"):
                    put(arrays, code, name, rng.normal(0., .20, len(path)))
            put(arrays, code, "vol20", rng.uniform(.008, .045, len(path)))
            put(arrays, code, "volume_ratio", rng.uniform(.3, 4., len(path)))
            arrays["scores"][:, :, AI[code]] = rng.integers(-3, 4, (2, len(path))) * 10.0
        for code, positions in ((A, (9, 31)), (B, (12, 43)), (X, (18, 44)), (GOLD, (17,))):
            arrays["features"][list(positions), AI[code], F["close"]] = np.nan
            arrays["features"][list(positions), AI[code], F["valid"]] = 0.
        arrays["fear"] = np.ascontiguousarray(rng.integers(0, 2, len(meta["dates"])), dtype=np.int32)
        variants = []
        choices = lambda values: values[int(rng.integers(len(values)))]
        for _ in range(72):
            variants.append(config(
                score=choices(meta["score_names"]), mom=choices((10, 20, 40)), exit_mom=choices((10, 20, 60)), fast_mom=choices((3, 5, 10)),
                ma=choices(("ma100", "ma250", "ema120")), regime=choices(("ma", "open_stock", "always_bull", "all_assets", "breadth", "dual")),
                regime_hyst=choices((0., .02)), regime_confirm=choices((1, 2, 3)),
                buffer_mode=choices(("momentum", "score_gap", "score_relative", "rank")), buffer=choices((0., .02, .2)),
                global_buffer=choices((-1., .03)), gold_buffer=choices((-1., .03)), rank_keep=choices((1, 2, 3)),
                min_hold=choices((0, 2, 5)), switch_confirm=choices((1, 2, 3)),
                bull_entry=choices((0., .03)), bear_entry=choices((0., .07)), exit_floor=choices((-.02, 0., .02)),
                panic_mode=choices(("fixed", "volatility")), panic=choices((0., .04, 2.)), overheat=choices((.25, .4, 9.9)),
                fallback_mode=choices(("rank", "cash", "low_vol")), fallback_floor=choices((-9.9, -.05, 0.)),
                crash_mask=choices(tuple(range(8))), crash_lock=choices((1, 3, 5)),
                deep_mom=choices((-.06, -.08)), deep_below=choices((.15, .20)), relaxed_mom=choices((-.03, -.04)),
                volume_below=choices((.10, .15)), volume_ratio=choices((1.5, 2.)),
                crash_guard=choices(("always", "held_negative", "candidate_better", "held_shock", "cash_only")),
                crash_pick=choices(("score", "deepest", "least_volatile")), crash_stock_only=choices((0, 1)),
                trail_stop=choices((0., .05, .1)), lag=choices((0, 1)), breadth_threshold=choices((.5, .75)),
                trend_buffer=choices((0., .1)), panic_cooldown=choices((0, 2, 4))))
        original, _ = self.compare(arrays, meta, variants)
        altered = deepcopy(arrays)
        cutoff = 40
        altered["features"][cutoff + 1:, :, F["close"]] *= 3
        altered["scores"][:, cutoff + 1:, :] *= -10
        future, _ = self.compare(altered, meta, variants[:8], stop=cutoff, workers=1)
        np.testing.assert_array_equal(original["holdings"][:8, :cutoff + 1], future["holdings"])
        np.testing.assert_allclose(original["returns"][:8, :cutoff + 1], future["returns"], rtol=0., atol=0.)


if __name__ == "__main__":
    unittest.main()
