"""Continuous-account seam checks; no new candidate search or 2026 evaluation."""
from copy import deepcopy
import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from v10_deep.features import FEATURE_NAMES
from v10_deep.reference import DatedValues, run_reference
from v10_deep.schema import baseline
from v11 import walkforward


ASSETS = ("159915", "588080", "510300", "510500", "563300", "512400", "512890",
          "513100", "513120", "518880", "511880")
A, B, C = "159915", "588080", "513100"
F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def fixture(count=18):
    dates, current = [], date(2020, 1, 6)
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current.isoformat())
        current += timedelta(days=1)
    values = np.zeros((count, len(ASSETS), len(FEATURE_NAMES)), dtype=np.float64)
    for i in range(count):
        for j, code in enumerate(ASSETS):
            values[i, j, F["close"]] = 100 * (1 + .002 * i * (j + 1))
            values[i, j, F["valid"]] = float(code != "511880")
            values[i, j, F["mom5"]] = .01
            values[i, j, F["mom20"]] = .3 if code == B else .1
            values[i, j, F["mom60"]] = .2
            values[i, j, F["ma250"]] = .1
            values[i, j, F["ma180"]] = .1
            values[i, j, F["vol20"]] = .02
            values[i, j, F["volume_ratio"]] = 1.
            values[i, j, F["ret1"]] = 0. if not i else values[i, j, F["close"]] / values[i - 1, j, F["close"]] - 1
    scores = np.ones((3, count, len(ASSETS)), dtype=np.float64)
    scores[:2, :, ASSETS.index(A)] = 100.
    scores[2, :, ASSETS.index(B)] = 200.
    arrays = dict(features=values, scores=scores, fear=np.zeros(count, dtype=np.int32))
    meta = dict(dates=dates, assets=list(ASSETS), feature_names=list(FEATURE_NAMES),
                score_names=["wls25_v20", "wls20_smooth3", "alternate"])
    config = baseline()
    config.update(crash_mask=0, buffer=0., global_buffer=0., gold_buffer=0.)
    return arrays, meta, config


def segments(meta, configs, cuts=None, contexts=None):
    cuts = cuts or [0, len(meta["dates"]) // 2, len(meta["dates"])]
    return [dict(start=meta["dates"][a], end=meta["dates"][b - 1], config=deepcopy(configs[i]),
                 risk_context=(contexts[i] if contexts else {}))
            for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]


class ContinuousWalkForwardTests(unittest.TestCase):
    def test_same_model_preserves_confirmation_and_matches_reference_for_both_lags(self):
        arrays, meta, config = fixture()
        config.update(switch_confirm=2, min_hold=0, crash_mask=7)
        arrays["scores"][0, 1:, ASSETS.index(B)] = 300.
        arrays["features"][5, ASSETS.index(C), F["mom5"]] = -.09
        arrays["features"][5, ASSETS.index(C), F["ma250"]] = -.30
        arrays["scores"][0, 5, ASSETS.index(C)] = 1000.
        arrays["features"][7, ASSETS.index(C), F["close"]] = np.nan
        arrays["features"][7, ASSETS.index(C), F["valid"]] = 0.
        for lag in (0, 1):
            with self.subTest(lag=lag):
                config["lag"] = lag
                schedule = segments(meta, [config] * 3, [0, 2, 8, len(meta["dates"])])
                actual = walkforward.run(schedule, arrays, meta)
                expected = run_reference(arrays, meta, config)
                np.testing.assert_array_equal(actual["returns"], expected["returns"])
                np.testing.assert_array_equal(actual["holdings"], expected["holdings"])
                np.testing.assert_array_equal(actual["summary"], expected["summary"])
                self.assertTrue(all(row["policy_reused"] for row in actual["boundaries"][1:]))
                self.assertFalse(any(row["pending_reset"] for row in actual["trace"]))

    def test_model_change_marks_old_asset_and_charges_real_boundary_switch(self):
        arrays, meta, first = fixture(7)
        second = deepcopy(first)
        second["score"] = "alternate"
        schedule = segments(meta, [first, second], [0, 3, 7])
        fee = .0011
        result = walkforward.run(schedule, arrays, meta, fee=fee)
        price = arrays["features"][:, :, F["close"]]
        expected = (price[3, ASSETS.index(A)] / price[0, ASSETS.index(A)] * (1 - 2 * fee)
                    * price[-1, ASSETS.index(B)] / price[3, ASSETS.index(B)])
        self.assertAlmostEqual(result["summary"][0], expected, places=13)
        self.assertEqual(result["summary"][3], 1)
        boundary_trade = result["trades"][1]
        self.assertTrue(boundary_trade["at_model_change"])
        self.assertEqual((boundary_trade["from"], boundary_trade["to"]), (A, B))
        self.assertAlmostEqual(boundary_trade["cost"], boundary_trade["nav_before_cost"] * 2 * fee)
        self.assertTrue(result["trades"][0]["first_session_free"])
        self.assertEqual(result["trades"][0]["cost"], 0.)
        self.assertEqual(result["boundaries"][1]["holding"], A)
        self.assertEqual(result["boundaries"][1]["age"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            walkforward.write_trace_csv(path, result)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[3]["model_changed"], "True")
            self.assertEqual(rows[3]["previous_holding"], A)
            self.assertEqual(rows[3]["holding"], B)
            self.assertGreater(float(rows[3]["cost"]), 0.)

    def test_changed_model_keeps_actual_crash_lock_until_its_original_expiry(self):
        arrays, meta, first = fixture(9)
        first["crash_mask"] = 1
        arrays["features"][1, ASSETS.index(C), F["mom5"]] = -.09
        arrays["features"][1, ASSETS.index(C), F["ma250"]] = -.3
        arrays["scores"][0, 1, ASSETS.index(C)] = 1000.
        second = deepcopy(first)
        second.update(score="alternate", crash_mask=0)
        result = walkforward.run(segments(meta, [first, second], [0, 2, 9]), arrays, meta)
        self.assertEqual(result["daily"][1][2], C)
        self.assertEqual(result["boundaries"][1]["lock_until"], 6)
        self.assertTrue(all(row[2] == C for row in result["daily"][2:6]))
        self.assertEqual(result["daily"][6][2], B)

    def test_changed_model_resets_pending_but_identical_model_does_not(self):
        arrays, meta, first = fixture(6)
        first["switch_confirm"] = 2
        arrays["scores"][0, 1:, ASSETS.index(B)] = 300.
        second = deepcopy(first)
        second["panic"] = .05
        changed = walkforward.run(segments(meta, [first, second], [0, 2, 6]), arrays, meta)
        same = walkforward.run(segments(meta, [first, first], [0, 2, 6]), arrays, meta)
        self.assertEqual(changed["daily"][2][2], A)
        self.assertEqual(changed["daily"][3][2], B)
        self.assertEqual(same["daily"][2][2], B)
        self.assertTrue(changed["trace"][2]["pending_reset"])
        self.assertFalse(same["trace"][2]["pending_reset"])

    def test_asset_peak_and_age_survive_model_change_without_a_trade(self):
        arrays, meta, first = fixture(5)
        first["panic"] = 0.
        p = [100., 150., 110., 120., 121.]
        arrays["features"][:, ASSETS.index(A), F["close"]] = p
        arrays["features"][:, ASSETS.index(A), F["ret1"]] = [0.] + [p[i] / p[i - 1] - 1 for i in range(1, len(p))]
        second = deepcopy(first)
        second["panic"] = .03
        result = walkforward.run(segments(meta, [first, second], [0, 3, 5]), arrays, meta)
        self.assertEqual(result["boundaries"][1]["asset_peak"], 150.)
        self.assertEqual(result["trace"][3]["asset_peak"], 150.)
        self.assertEqual(result["trace"][3]["age"], 3)
        self.assertFalse(result["trace"][3]["filled"])
        self.assertEqual(result["summary"][3], 0)

    def test_lagged_signal_never_uses_current_rank_and_current_missing_bar_blocks_fill(self):
        arrays, meta, config = fixture(5)
        config["lag"] = 1
        arrays["scores"][0, 2:, ASSETS.index(B)] = 300.
        arrays["features"][3, ASSETS.index(B), F["close"]] = np.nan
        arrays["features"][3, ASSETS.index(B), F["valid"]] = 0.
        result = walkforward.run(segments(meta, [config, config]), arrays, meta)
        self.assertEqual(result["daily"][2][2], A)  # Today's B score cannot execute until the next observation.
        self.assertEqual(result["trace"][3]["signal_date"], meta["dates"][2])
        self.assertEqual(result["trace"][3]["intended"], B)
        self.assertFalse(result["trace"][3]["filled"])
        self.assertEqual(result["daily"][3][2], A)
        self.assertEqual(result["summary"][5], 1)

    def test_nonempty_risk_context_requires_a_factory_and_changes_context_explicitly(self):
        arrays, meta, config = fixture(6)
        contexts = [{"risk_view": "prior20"}, {"risk_view": "prior60"}]
        schedule = segments(meta, [config, config], contexts=contexts)
        with self.assertRaisesRegex(ValueError, "view_factory"):
            walkforward.run(schedule, arrays, meta)
        seen = []
        def factory(values, metadata, selected, context):
            seen.append(context)
            return DatedValues(values, metadata, selected)
        result = walkforward.run(schedule, arrays, meta, view_factory=factory)
        self.assertEqual(seen, contexts)
        self.assertTrue(result["boundaries"][1]["pending_reset"])
        self.assertEqual(result["boundaries"][1]["holding"], A)

    def test_future_array_perturbations_do_not_change_past_prefix(self):
        arrays, meta, config = fixture()
        schedule = segments(meta, [config, config], [0, 6, len(meta["dates"])])
        changed = deepcopy(arrays)
        changed["features"][10:, :, F["close"]] *= 100.
        changed["scores"][:, 10:] *= -100.
        first = walkforward.run(schedule, arrays, meta, end=meta["dates"][9])
        second = walkforward.run(schedule, changed, meta, end=meta["dates"][9])
        np.testing.assert_array_equal(first["returns"], second["returns"])
        self.assertEqual(first["trace"], second["trace"])

    def test_four_fixed_folds_and_invalid_schedule_coverage(self):
        self.assertEqual([row["train_end"] for row in walkforward.training_folds()],
                         ["2017-12-31", "2019-12-31", "2021-12-31", "2023-12-31"])
        arrays, meta, config = fixture(6)
        schedule = segments(meta, [config, config])
        overlap = deepcopy(schedule)
        overlap[1]["start"] = overlap[0]["end"]
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            walkforward.run(overlap, arrays, meta)
        gap = deepcopy(schedule)
        gap[0]["end"] = meta["dates"][1]
        with self.assertRaisesRegex(ValueError, "Every evaluated"):
            walkforward.run(gap, arrays, meta)


class FrozenHDevelopmentFidelityTests(unittest.TestCase):
    def test_same_h_across_development_segments_matches_one_continuous_reference(self):
        from v11.data import load_frozen
        bundle = load_frozen()
        arrays, meta = bundle["arrays"], bundle["meta"]
        config = deepcopy(bundle["profiles"]["h"])
        for lag in (0, 1):
            with self.subTest(lag=lag):
                config["lag"] = lag
                schedule = [dict(start=start, end=end, config=deepcopy(config)) for start, end in
                            (("2014-01-02", "2015-12-31"), ("2016-01-01", "2017-12-31"),
                             ("2018-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"))]
                actual = walkforward.run(schedule, arrays, meta)
                expected = run_reference(arrays, meta, config, start="2014-01-02", end="2021-12-31")
                np.testing.assert_array_equal(actual["returns"], expected["returns"])
                np.testing.assert_array_equal(actual["holdings"], expected["holdings"])
                np.testing.assert_array_equal(actual["summary"], expected["summary"])
                self.assertLessEqual(actual["dates"][-1], "2021-12-31")
                self.assertTrue(all(row["policy_reused"] for row in actual["boundaries"][1:]))


if __name__ == "__main__":
    unittest.main()
