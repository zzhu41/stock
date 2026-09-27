"""Offline tests of frozen-H feature/state adaptation; no network or real state writes."""
from copy import deepcopy
import csv
from datetime import date, timedelta
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v10_live import policy

ROOT = Path(__file__).resolve().parent.parent
PROFILE = json.loads((ROOT / "v10_deep/profiles.json").read_text())["variants"]["growth"]


def weekdays(count=315):
    day, result = date(2024, 1, 1), []
    while len(result) < count:
        if day.weekday() < 5:
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


DATES = weekdays()


def histories():
    prices = [100 * (1 + .001 * i + .004 * math.sin(i * .31)) for i in range(len(DATES))]
    rows = [(day, p, p, p, p, 100 + (i % 9)) for i, (day, p) in enumerate(zip(DATES, prices))]
    return {code: list(rows) for code in policy.ASSETS}


def independent_score(closes):
    values = closes[-20:]
    weights = list(range(1, 21))
    mx = sum(i * w for i, w in enumerate(weights)) / sum(weights)
    my = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    slope = sum(w * (i - mx) * (v - my) for i, (w, v) in enumerate(zip(weights, values))) / sum(w * (i - mx) ** 2 for i, w in enumerate(weights))
    returns = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 20, len(closes))]
    mean = sum(returns) / 20
    sigma = (sum((r - mean) ** 2 for r in returns) / 20) ** .5
    return slope / my * 250 / sigma if sigma else 0


class LiveHPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.qvix_path = Path(self.temporary.name) / "qvix.csv"
        self.loader = patch.object(policy, "load_profile", side_effect=lambda version: deepcopy(PROFILE)).start()
        self.addCleanup(patch.stopall)

    def call(self, hs=None, state=None, signal_date=None, calendar=None):
        return policy.decide(hs or histories(), calendar or DATES, signal_date or DATES[-1], state, self.qvix_path)

    def test_growth_profile_only_cold_start_and_no_account_side_effect(self):
        import strategy
        before = {key: deepcopy(value) for key, value in vars(strategy).items() if key.isupper()}
        state = {"holding": None, "nav": 9.9, "last_date": None}
        original = deepcopy(state)
        result = self.call(state=state)
        self.loader.assert_called_once_with("growth")
        self.assertEqual(result["candidate_id"], policy.CANDIDATE_ID)
        self.assertEqual(result["target"], "159915")
        self.assertTrue(result["executable"])
        self.assertTrue(result["research_observation_only"])
        self.assertFalse(result["order_submission"])
        self.assertNotIn("nav", result)
        self.assertEqual(state, original)
        self.assertEqual(before, {key: value for key, value in vars(strategy).items() if key.isupper()})
        altered = deepcopy(PROFILE)
        altered["config"]["id"] = "some_other_profile"
        with patch.object(policy, "load_profile", return_value=altered):
            with self.assertRaisesRegex(ValueError, "main H"):
                self.call()

    def test_reference_source_is_pinned_by_verified_fidelity_receipt(self):
        receipt = Path(self.temporary.name) / "fidelity.json"
        receipt.write_text(json.dumps({"source_and_cache_sha256": {
            "/frozen/v10_deep/reference.py": "0" * 64}}))
        with patch.object(policy, "REFERENCE_RECEIPT", receipt):
            with self.assertRaisesRegex(ValueError, "reference.py changed"):
                self.call()

    def test_smoothed_score_uses_three_own_observations_not_market_days(self):
        hs = histories()
        hs["159915"].pop(-2)
        data, dates = policy.snapshot(hs, DATES, DATES[-1])
        closes = [row[2] for row in hs["159915"]]
        expected = sum(independent_score(closes[:len(closes) - lag]) for lag in (0, 1, 2)) / 3
        self.assertAlmostEqual(data.score(len(dates) - 1, "159915"), expected, places=9)
        row = data.indicators["159915"]
        self.assertAlmostEqual(row["ma180"], closes[-1] / (sum(closes[-180:]) / 180) - 1, places=14)
        self.assertAlmostEqual(row["ma250"], closes[-1] / (sum(closes[-250:]) / 250) - 1, places=14)
        previous = [r[5] for r in hs["159915"][-21:-1]]
        self.assertEqual(row["volume_ratio"], hs["159915"][-1][5] / (sum(previous) / 20))

    def test_future_values_cannot_change_features_or_decision(self):
        hs = histories()
        asof = DATES[-8]
        first = self.call(hs, signal_date=asof)
        changed = deepcopy(hs)
        for code in policy.ASSETS:
            changed[code] = [row if row[0] <= asof else (row[0], 1e9, float("nan"), 1e9, 1e9, -1) for row in changed[code]]
        second = self.call(changed, signal_date=asof)
        self.assertEqual(first, second)

    def test_warmup_cash_and_stable_tie_order(self):
        hs = histories()
        data, dates = policy.snapshot(hs, DATES, DATES[268])
        self.assertEqual(data.ranked(len(dates) - 1), [])
        data, dates = policy.snapshot(hs, DATES, DATES[269])
        self.assertEqual(data.ranked(len(dates) - 1), list(policy.ASSETS[:-1]))
        self.assertFalse(data.informed(len(dates) - 1, policy.CASH))
        self.assertTrue(math.isfinite(data.price(len(dates) - 1, policy.CASH)))

    def test_ma180_regime_does_not_replace_ma250_crash_distance(self):
        data, dates = policy.snapshot(histories(), DATES, DATES[-1])
        data.indicators[policy.BENCHMARK].update(ma180=.01, ma250=-.3)
        candidate = data.indicators["159915"]
        candidate.update(score=100000, mom5=-.09, ma180=-.5, ma250=-.19, volume_ratio=1.)
        with patch.object(policy, "snapshot", return_value=(data, dates)):
            result = self.call()
        self.assertTrue(result["diagnostics"]["bull"])
        self.assertFalse(result["crash"])
        candidate["ma250"] = -.21
        with patch.object(policy, "snapshot", return_value=(data, dates)):
            result = self.call()
        self.assertTrue(result["crash"])
        self.assertEqual(result["diagnostics"]["crash_channels"], ["深跌"])

    def test_dynamic_panic_clips_at_two_and_ten_percent_and_uses_inclusive_boundary(self):
        for sigma, threshold in ((.001, .02), (.04, .06), (.2, .10)):
            for delta, expected in ((0, True), (.00001, False)):
                with self.subTest(sigma=sigma, delta=delta):
                    data, dates = policy.snapshot(histories(), DATES, DATES[-1])
                    data.indicators["159915"].update(vol20=sigma, ret1=-threshold + delta)
                    with patch.object(policy, "snapshot", return_value=(data, dates)):
                        result = self.call(state={"holding": "159915", "entry_date": DATES[-9]})
                    self.assertEqual(result["panic"], expected)
                    self.assertEqual(result["diagnostics"]["panic_threshold"], threshold)

    def test_three_crash_channels_and_proposed_lock_only_after_executable_action(self):
        cases = ((-.09, -.21, 1., False, "深跌"), (-.05, -.21, 1., True, "QVIX恐慌"), (-.05, -.11, 2., False, "量能恐慌"))
        for mom5, distance, volume, fear, label in cases:
            with self.subTest(channel=label):
                data, dates = policy.snapshot(histories(), DATES, DATES[-1], fear)
                data.indicators["159915"].update(score=100000, mom5=mom5, ma250=distance, volume_ratio=volume)
                with patch.object(policy, "snapshot", return_value=(data, dates)):
                    result = self.call()
                self.assertEqual(result["target"], "159915")
                self.assertTrue(result["crash"])
                self.assertEqual(result["crash_trigger_date"], DATES[-1])
                self.assertEqual(result["crash_code"], "159915")
                self.assertEqual(result["lock_remaining_sessions"], 5)
                self.assertEqual(result["diagnostics"]["crash_channels"], [label])
        data.indicators["510500"]["close"] = float("nan")
        original = {"holding": "510500", "entry_date": DATES[-8]}
        with patch.object(policy, "snapshot", return_value=(data, dates)):
            blocked = self.call(state=original)
        self.assertFalse(blocked["executable"])
        self.assertFalse(blocked["crash"])
        self.assertIsNone(blocked["crash_trigger_date"])
        self.assertEqual(blocked["target"], "510500")
        self.assertEqual(original, {"holding": "510500", "entry_date": DATES[-8]})

    def test_five_trading_day_crash_lock_bypasses_negative_momentum_then_expires(self):
        hs = histories()
        hs[policy.GOLD] = [(day, 100-i*.15, 100-i*.15, 100-i*.15, 100-i*.15, 100.) for i, day in enumerate(DATES)]
        trigger = DATES[-6]
        state = dict(holding=policy.GOLD, entry_date=trigger, crash_trigger_date=trigger, crash_code=policy.GOLD)
        for day in DATES[-6:-1]:
            result = self.call(hs, state, day)
            self.assertEqual(result["target"], policy.GOLD)
            self.assertTrue(result["lock_active"])
        result = self.call(hs, state)
        self.assertFalse(result["lock_active"])
        self.assertNotEqual(result["target"], policy.GOLD)
        self.assertIsNone(result["crash_trigger_date"])
        self.assertIsNone(result["crash_code"])

    def test_bad_state_or_date_and_missing_held_quote_cannot_create_a_trade(self):
        with self.assertRaisesRegex(ValueError, "together"):
            self.call(state=dict(holding="510300", crash_code="510300"))
        with self.assertRaisesRegex(ValueError, "match"):
            self.call(state=dict(holding="510500", crash_code="510300", crash_trigger_date=DATES[-1]))
        with self.assertRaisesRegex(ValueError, "predates"):
            self.call(state=dict(last_date="2099-01-01"))
        hs = histories()
        hs["159915"] = hs["159915"][:-1]
        result = self.call(hs, state={"holding": "159915"})
        self.assertFalse(result["executable"])
        self.assertEqual(result["target"], "159915")

    def test_local_qvix_requires_same_day_and_120_strict_previous_observations(self):
        with self.qvix_path.open("w", newline="") as stream:
            csv.writer(stream).writerows((day, 10 + i % 5) for i, day in enumerate(DATES[:-1]))
        result = self.call()
        self.assertFalse(result["diagnostics"]["qvix"]["available"])
        self.assertIn("停用", result["diagnostics"]["qvix"]["note"])
        self.assertTrue(result["executable"])
        with self.qvix_path.open("a", newline="") as stream:
            csv.writer(stream).writerow((DATES[-1], 30))
        q = policy.qvix_state(DATES[-1], self.qvix_path)
        self.assertTrue(q["available"])
        self.assertTrue(q["active"])
        self.assertEqual(q["previous_samples"], 250)
        with self.qvix_path.open("w", newline="") as stream:
            csv.writer(stream).writerows((day, 10 + i % 5) for i, day in enumerate(DATES[-120:]))
        self.assertFalse(policy.qvix_state(DATES[-1], self.qvix_path)["available"])


class FrozenHistoryParityTests(unittest.TestCase):
    def test_fixed_asof_features_and_targets_match_frozen_cache_and_reference(self):
        from v10_deep.reference import run_reference
        self.assertEqual(policy.load_profile("growth")["config"], PROFILE["config"])
        meta = json.loads((ROOT / "v10_deep/cache/features.json").read_text())
        with np.load(ROOT / "v10_deep/cache/features.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("features", "scores", "fear")}
        hs = {}
        for code in policy.ASSETS:
            with (ROOT / "v10_h_close/corrected_snapshots" / (code + ".csv")).open(newline="") as stream:
                rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(stream) if r]
            hs[code] = [(d, o, c, c, c, volume) for d, o, c, volume in rows]
        calendar = [row[0] for row in hs[policy.BENCHMARK]]
        # Deterministic mechanism dates, unrelated to model performance.
        dates = ["2014-01-02", "2021-10-25", "2026-09-24"]
        reference = run_reference(arrays, meta, PROFILE["config"], start="2014-01-02", end="2026-09-24")
        crash_dates = [row["date"] for row in reference["trace"] if row["filled"] and row["crash_requested"]]
        dates += [crash_dates[0], crash_dates[-1]]
        entry = trigger = crash_code = None
        states = {}
        for i, trace in enumerate(reference["trace"]):
            holding = meta["assets"][reference["holdings"][i-1]] if i and reference["holdings"][i-1] >= 0 else None
            if trace["date"] in dates:
                states[trace["date"]] = dict(holding=holding, entry_date=entry,
                                             crash_trigger_date=trigger, crash_code=crash_code)
            if trace["filled"]:
                entry = trace["date"]
                if trace["crash_requested"]:
                    trigger, crash_code = trace["date"], trace["holding"]
                else:
                    trigger = crash_code = None
            if trigger and calendar.index(trace["date"]) >= calendar.index(trigger) + 5:
                trigger = crash_code = None
        score_index = meta["score_names"].index("wls20_smooth3")
        with patch.object(policy, "load_profile", return_value=deepcopy(PROFILE)):
            for asof in dict.fromkeys(dates):
                with self.subTest(date=asof):
                    date_index = meta["dates"].index(asof)
                    data, live_dates = policy.snapshot(hs, calendar, asof)
                    for code in policy.ASSETS:
                        asset_index = meta["assets"].index(code)
                        for key in ("close", "valid"):
                            expected = arrays["features"][date_index, asset_index, meta["feature_names"].index(key)]
                            actual = data.value(len(live_dates)-1, code, key)
                            if math.isnan(expected):
                                self.assertTrue(math.isnan(actual))
                            else:
                                self.assertAlmostEqual(actual, expected, places=12)
                        if not data.informed(len(live_dates)-1, code):
                            continue
                        for key in ("ret1", "mom5", "mom20", "mom60", "vol20", "ma180", "ma250", "volume_ratio"):
                            expected = arrays["features"][date_index, asset_index, meta["feature_names"].index(key)]
                            self.assertAlmostEqual(data.value(len(live_dates)-1, code, key), expected, places=12)
                        expected_score = arrays["scores"][score_index, date_index, asset_index]
                        self.assertAlmostEqual(data.score(len(live_dates)-1, code), expected_score, places=10)
                    result = policy.decide(hs, calendar, asof, states[asof], ROOT / "v10_h_close/snapshots/qvix50.csv")
                    ri = reference["dates"].index(asof)
                    expected = meta["assets"][reference["holdings"][ri]]
                    self.assertTrue(result["executable"])
                    self.assertEqual(result["target"], expected)
                    self.assertEqual(result["crash"], reference["trace"][ri]["crash_requested"] and reference["trace"][ri]["filled"])


if __name__ == "__main__":
    unittest.main()
