"""Independent trailing-regression and dated-feature tests; synthetic inputs only.

Each build writes solely to a temporary cache. Real cache/data/core files are
never changed, and no candidate-return matrix or native strategy is executed.
"""
from copy import deepcopy
from datetime import date, timedelta
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v10_deep import features as feature_module
from v10_deep.features import F, MA_WINDOWS, MOM_WINDOWS, SCORE_WINDOWS, VOL_WINDOWS, _regression


ASSETS = ("510300", "159915", "588080", "511880")


def fixture(n=410, very_late=False):
    calendar, day = [], date(2018, 1, 2)
    while len(calendar) < n:
        if day.weekday() < 5:
            calendar.append(day.isoformat())
        day += timedelta(days=1)
    histories = {}
    for ai, code in enumerate(ASSETS):
        rows = []
        for i, d in enumerate(calendar):
            if code == "159915" and i in (8, 16, 305, 306, 333):
                continue
            if code == "588080" and i < (n - 17 if very_late else 30):
                continue
            p = (10 + ai) * (1 + .0015 * i + .035 * math.sin(i / 8.0 + ai)
                             + .012 * math.cos(i / 19.0))
            rows.append((d, p * .999, p, 1000 + 17 * i + (i % 11) * 23))
        histories[code] = rows
    return histories, calendar


def direct_regression(values, window, weighted=True):
    sample = list(values[-window:])
    weights = list(range(1, window + 1)) if weighted else [1] * window
    total = math.fsum(weights)
    mx = math.fsum(w * i for i, w in enumerate(weights)) / total
    my = math.fsum(w * y for w, y in zip(weights, sample)) / total
    xx = math.fsum(w * (i - mx) ** 2 for i, w in enumerate(weights))
    xy = math.fsum(w * (i - mx) * (y - my) for i, (w, y) in enumerate(zip(weights, sample)))
    yy = math.fsum(w * (y - my) ** 2 for w, y in zip(weights, sample))
    slope = xy / xx
    r2 = xy * xy / xx / yy if yy > 0 else 0.0
    return my, slope, min(1.0, max(0.0, r2))


def direct_vol(closes, index, window):
    returns = [closes[j] / closes[j - 1] - 1 for j in range(index - window + 1, index + 1)]
    mean = math.fsum(returns) / window
    return math.sqrt(math.fsum((r - mean) ** 2 for r in returns) / window)


def direct_score(closes, index, family, window, vol_window=20):
    prefix = closes[:index + 1]
    source = [math.log(p) for p in prefix] if family == "logwls" else prefix
    mean, slope, r2 = direct_regression(source, window, family != "ols")
    trend = slope * 250 if family == "logwls" else slope / mean * 250
    volatility = direct_vol(closes, index, vol_window)
    return trend / volatility if volatility > 0 else 0.0, trend, r2


def build_fixture(histories, calendar):
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(feature_module, "BASE", Path(directory)), \
            patch.object(feature_module, "ASSET_ORDER", ASSETS), \
            patch.object(feature_module, "sha", return_value="synthetic-fingerprint"), \
            patch.object(feature_module, "inputs", return_value=(histories, calendar, [False] * len(calendar))):
        return feature_module.build(force=True)


class DatedFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.histories, cls.calendar = fixture()
        cls.arrays, cls.meta = build_fixture(cls.histories, cls.calendar)
        cls.score_index = {name: i for i, name in enumerate(cls.meta["score_names"])}
        cls.date_index = {d: i for i, d in enumerate(cls.calendar)}

    def assert_close(self, actual, expected):
        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
                        "%r != %r" % (actual, expected))

    def test_regression_tail_orientation_matches_independent_weighted_and_ordinary_ols(self):
        prices = np.asarray([10 + .03 * i + .4 * math.sin(i / 7) + .15 * math.cos(i / 13)
                             for i in range(180)])
        for window in (10, 25, 35, 60, 120):
            for family in ("wls", "ols", "logwls"):
                source = np.log(prices) if family == "logwls" else prices
                mean, slope, quality = _regression(source, window, family != "ols")
                self.assertEqual(len(mean), len(prices) - window + 1)
                for endpoint in (window - 1, window + 5, len(prices) - 1):
                    expected = direct_regression(source[:endpoint + 1], window, family != "ols")
                    j = endpoint - window + 1
                    with self.subTest(window=window, family=family, endpoint=endpoint):
                        self.assert_close(mean[j], expected[0])
                        self.assert_close(slope[j], expected[1])
                        self.assertAlmostEqual(quality[j], expected[2], delta=1e-8)

    def test_all_155_score_families_exist_and_three_estimator_families_match_raw_calculation(self):
        self.assertEqual(len(self.meta["score_names"]), 155)
        for code in ("510300", "159915"):
            ai = ASSETS.index(code)
            rows = self.histories[code]
            closes = [r[2] for r in rows]
            for own_i in (269, 301, 365):
                ti = self.date_index[rows[own_i][0]]
                for window in SCORE_WINDOWS:
                    for family in ("wls", "ols", "logwls"):
                        for vol_window in (20, 40, 60):
                            name = "%s%d_v%d" % (family, window, vol_window)
                            actual = self.arrays["scores"][self.score_index[name], ti, ai]
                            expected = direct_score(closes, own_i, family, window, vol_window)[0]
                            with self.subTest(code=code, own_i=own_i, score=name):
                                self.assert_close(actual, expected)
                    score, trend, r2 = direct_score(closes, own_i, "wls", window)
                    self.assert_close(self.arrays["scores"][self.score_index["wls%d_raw" % window], ti, ai], trend)
                    self.assertAlmostEqual(self.arrays["scores"][self.score_index["wls%d_r2" % window], ti, ai],
                                           score * r2, delta=1e-7)

    def test_smooth_three_uses_own_observations_across_missing_market_bars(self):
        code, ai = "159915", ASSETS.index("159915")
        rows = self.histories[code]
        closes = [r[2] for r in rows]
        # Include the first legal row and the first bar after a two-session gap.
        for own_i in (269, next(i for i, r in enumerate(rows) if r[0] == self.calendar[307]), 345):
            ti = self.date_index[rows[own_i][0]]
            for window in (20, 40, 60):
                expected = math.fsum(direct_score(closes, j, "wls", window)[0]
                                     for j in range(own_i - 2, own_i + 1)) / 3
                actual = self.arrays["scores"][self.score_index["wls%d_smooth3" % window], ti, ai]
                self.assert_close(actual, expected)
                self.assertGreater(actual, -1e99)
        for ti in (305, 306, 333):
            self.assertTrue(np.isnan(self.arrays["features"][ti, ai, F["close"]]))
            self.assertEqual(self.arrays["features"][ti, ai, F["valid"]], 0)
            self.assertTrue(np.all(self.arrays["scores"][:, ti, ai] == -1e100))

    def test_first_legal_data_is_270_own_observations_and_cash_is_not_ranked(self):
        for code in ASSETS[:-1]:
            ai, rows = ASSETS.index(code), self.histories[code]
            before, first = self.date_index[rows[268][0]], self.date_index[rows[269][0]]
            self.assertEqual(self.arrays["features"][before, ai, F["valid"]], 0)
            self.assertEqual(self.arrays["features"][first, ai, F["valid"]], 1)
            self.assertTrue(np.all(self.arrays["scores"][:, before, ai] == -1e100))
            self.assertTrue(np.all(self.arrays["scores"][:, first, ai] > -1e99))
        cash = ASSETS.index("511880")
        self.assertTrue(np.all(self.arrays["features"][:, cash, F["valid"]] == 0))
        self.assertTrue(np.all(self.arrays["scores"][:, :, cash] == -1e100))
        self.assertTrue(np.isfinite(self.arrays["features"][:, cash, F["close"]]).all())

    def test_volume_and_efficiency_windows_exclude_wrapped_future_observations(self):
        for code in ASSETS:
            ai, rows = ASSETS.index(code), self.histories[code]
            for r in rows[:20]:
                ti = self.date_index[r[0]]
                self.assertTrue(np.isnan(self.arrays["features"][ti, ai, F["volume_ratio"]]))
                self.assertTrue(np.isnan(self.arrays["features"][ti, ai, F["er20"]]))
            for j in (20, 269, 320):
                ti = self.date_index[rows[j][0]]
                prior_volume = math.fsum(r[3] for r in rows[j - 20:j]) / 20
                expected_volume = rows[j][3] / prior_volume
                distance = abs(rows[j][2] - rows[j - 20][2])
                path = math.fsum(abs(rows[k][2] - rows[k - 1][2]) for k in range(j - 19, j + 1))
                self.assert_close(self.arrays["features"][ti, ai, F["volume_ratio"]], expected_volume)
                self.assert_close(self.arrays["features"][ti, ai, F["er20"]], distance / path if path > 0 else 1)

    def test_momentum_volatility_ma_drawdown_and_ema_use_the_declared_trailing_bars(self):
        for code in ("510300", "159915"):
            ai, rows = ASSETS.index(code), self.histories[code]
            prices = [r[2] for r in rows]
            for j in (269, 301, 365):
                ti = self.date_index[rows[j][0]]
                values = self.arrays["features"][ti, ai]
                for window in MOM_WINDOWS:
                    self.assert_close(values[F["mom%d" % window]], prices[j] / prices[j - window] - 1)
                for window in VOL_WINDOWS:
                    self.assert_close(values[F["vol%d" % window]], direct_vol(prices, j, window))
                for window in MA_WINDOWS:
                    actual = values[F["ma%d" % window]]
                    if j + 1 < window:
                        self.assertTrue(np.isnan(actual))
                    else:
                        mean = math.fsum(prices[j - window + 1:j + 1]) / window
                        self.assert_close(actual, prices[j] / mean - 1)
                for window in (20, 60, 120):
                    expected = prices[j] / max(prices[j - window + 1:j + 1]) - 1
                    self.assert_close(values[F["dd%d" % window]], expected)
                for window in (120, 250):
                    alpha = 2 / (window + 1)
                    # Direct closed-form exponentially weighted sum, independent
                    # of the builder's sequential EMA recurrence.
                    average = prices[0] * (1 - alpha) ** j
                    average += math.fsum(alpha * (1 - alpha) ** (j - k) * prices[k]
                                         for k in range(1, j + 1))
                    self.assert_close(values[F["ema%d" % window]], prices[j] / average - 1)

    def test_future_price_and_volume_changes_leave_the_entire_prefix_unchanged(self):
        cutoff = self.calendar[325]
        changed = deepcopy(self.histories)
        for code, rows in changed.items():
            changed[code] = [(d, o * 7 + 100, c * 5 + 200, v * 99)
                             if d > cutoff else (d, o, c, v) for d, o, c, v in rows]
        later, meta = build_fixture(changed, self.calendar)
        self.assertEqual(meta["score_names"], self.meta["score_names"])
        np.testing.assert_allclose(later["features"][:326], self.arrays["features"][:326],
                                   rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(later["scores"][:, :326], self.arrays["scores"][:, :326])
        np.testing.assert_array_equal(later["orders"][:, :326], self.arrays["orders"][:, :326])

    def test_short_listing_and_short_regression_are_ineligible_without_broadcast_errors(self):
        for length in (0, 1, 9):
            result = _regression(np.ones(length), 10)
            self.assertTrue(all(len(x) == 0 for x in result))
        h, calendar = fixture(n=310, very_late=True)
        arrays, _ = build_fixture(h, calendar)
        ai = ASSETS.index("588080")
        self.assertEqual(len(h["588080"]), 17)
        self.assertTrue(np.all(arrays["features"][:, ai, F["valid"]] == 0))
        self.assertTrue(np.all(arrays["scores"][:, :, ai] == -1e100))

    def test_changed_qvix_fingerprint_invalidates_the_cache(self):
        h, calendar = fixture(n=280)
        q_version = [0]

        def fingerprint(path):
            return "qvix-%d" % q_version[0] if str(path).endswith("snapshots/qvix50.csv") else "unchanged"

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(feature_module, "BASE", Path(directory)), \
                patch.object(feature_module, "ASSET_ORDER", ASSETS), \
                patch.object(feature_module, "sha", side_effect=fingerprint), \
                patch.object(feature_module, "inputs", return_value=(h, calendar, [False] * len(calendar))) as inputs:
            first, _ = feature_module.build(force=True)
            cached, _ = feature_module.build(force=False)
            self.assertEqual(inputs.call_count, 1)
            np.testing.assert_array_equal(first["fear"], cached["fear"])
            q_version[0] = 1
            inputs.return_value = (h, calendar, [True] * len(calendar))
            replaced, _ = feature_module.build(force=False)
            self.assertEqual(inputs.call_count, 2)
            self.assertTrue(np.all(replaced["fear"] == 1))


if __name__ == "__main__":
    unittest.main()
