"""Offline V9.2+ fidelity and isolated forward-account tests."""
from bisect import bisect_right
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import shadow_v92_plus
from v92_plus_live import policy, ledger, runtime

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / "v10_deep/profiles.json").read_text())["variants"]["simple"]
ASSETS = policy.ASSETS
A, B = "159915", "510500"
DAYS = ["2026-09-24", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-09", "2026-10-12"]


def decision(target=A, **fields):
    return dict(dict(candidate_id=policy.CANDIDATE_ID, signal_date=DAYS[0], target=target,
                     executable=True, reason="测试建议", crash_trigger_date=None,
                     crash_code=None, diagnostics={"qvix": {"note": "当日QVIX缺失，通道已停用"}}), **fields)


def fixtures(day, price=10.):
    calendar = [d for d in DAYS if d <= day]
    quotes = {c: dict(price=price, open=price, volume=100., date=day, timestamp=day+" 14:50:00") for c in ASSETS}
    return quotes, dict(calendar=calendar,
        histories={c: [(d, 100., 100., 100., 100., 100.) for d in calendar] for c in ASSETS},
        raw_histories={c: [(d, price, price, 100.) for d in calendar] for c in ASSETS},
        actions={c: {d: dict(split_ratio=1., cash_per_old_share=0.) for d in calendar} for c in ASSETS},
        metadata=dict(provisional_date=day, quote_timestamps={c: quotes[c]["timestamp"] for c in ASSETS}))


def snapshot(day, **a_fields):
    indicators = {c: dict(close=100., valid=float(c != policy.CASH), score=0., bars=300,
        ret1=.01, mom5=.02, mom20=.1, mom60=.1, vol20=.01, ma250=.1, ma180=.1, volume_ratio=1.) for c in ASSETS}
    indicators[B].update(score=10., mom20=.5, mom5=.1)
    indicators[A].update(a_fields)
    return policy.Snapshot(indicators, day, False), DAYS[:day + 1]


class PolicyTests(unittest.TestCase):
    def call(self, day, state, **fields):
        with patch.object(policy, "verified_profile", return_value=deepcopy(PROFILE)), \
                patch.object(policy, "snapshot", return_value=snapshot(day, **fields)), \
                patch.object(policy, "qvix_state", return_value={"active": False}):
            return policy.decide({}, DAYS[:day + 1], DAYS[day], state)

    def test_healthy_minhold_two_uses_actual_entry_and_risk_exit_is_exempt(self):
        state = dict(holding=A, entry_date=DAYS[0], last_date=DAYS[0])
        self.assertEqual(self.call(1, state)["target"], A)
        self.assertEqual(self.call(2, state)["target"], B)
        risk = self.call(1, state, ret1=-.04)
        self.assertTrue(risk["panic"])
        self.assertEqual(risk["target"], B)
        negative = self.call(1, state, mom20=-.01)
        self.assertEqual(negative["target"], B)
        # A failed/late entry cannot inherit an earlier proposed signal date.
        delayed = dict(holding=A, entry_date=DAYS[1], last_date=DAYS[1])
        self.assertEqual(self.call(2, delayed)["target"], A)

    def test_crash_lock_takes_priority_and_expires_on_fifth_benchmark_observation(self):
        state = dict(holding=A, entry_date=DAYS[0], last_date=DAYS[0],
                     crash_trigger_date=DAYS[0], crash_code=A)
        for day in (1, 2, 3, 4):
            out = self.call(day, state, ret1=-.08, mom20=-.1)
            self.assertEqual(out["target"], A)
            self.assertTrue(out["lock_active"])
        out = self.call(5, state, ret1=-.08, mom20=-.1)
        self.assertEqual(out["target"], B)
        self.assertIsNone(out["crash_trigger_date"])

    def test_missing_held_quote_cannot_create_fill_or_crash_lock(self):
        state = dict(holding=A, entry_date=DAYS[0], last_date=DAYS[0])
        out = self.call(2, state, close=float("nan"), valid=0.)
        self.assertFalse(out["executable"])
        self.assertEqual(out["target"], A)
        self.assertFalse(out["crash"])
        with self.assertRaisesRegex(ValueError, "actual entry"):
            self.call(2, {"holding": A})

    def test_wls25_exact_old_arithmetic_and_future_values_are_ignored(self):
        from datetime import date, timedelta
        dates = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(290)]
        prices = [100. + i*.08 + .3*np.sin(i*.31) for i in range(290)]
        hs = {c: [(d, p, p, p, p, 100.+i) for i, (d, p) in enumerate(zip(dates, prices))] for c in ASSETS}
        first, ds = policy.snapshot(hs, dates, dates[-5])
        for c in ASSETS:
            hs[c][-4:] = [(d, 1., float("nan"), 1., 1., -1.) for d in dates[-4:]]
        second, unused = policy.snapshot(hs, dates, dates[-5])
        self.assertEqual(first.indicators, second.indicators)
        self.assertFalse(first.informed(len(ds)-1, policy.CASH))
        early, unused = policy.snapshot(hs, dates, dates[268])
        self.assertEqual(early.ranked(268), [])


class LedgerTests(unittest.TestCase):
    def start(self, day=DAYS[0], price=10.):
        q, v = fixtures(day, price)
        return ledger.advance(None, decision(), v, q, day)

    def test_first_current_entry_nav_one_and_new_target_has_no_ex_date_entitlement(self):
        q, v = fixtures(DAYS[0], 9.)
        v["actions"][A][DAYS[0]]["cash_per_old_share"] = 1.
        state = ledger.advance(None, decision(), v, q, DAYS[0])
        self.assertEqual(state["nav"], 1.)
        self.assertEqual(state["events"][0]["actions"], [])
        self.assertEqual(state["candidate_id"], policy.CANDIDATE_ID)
        q, v = fixtures(DAYS[1])
        v["actions"][B][DAYS[1]]["cash_per_old_share"] = 2.
        later = ledger.advance(state, decision(B), v, q, DAYS[1])
        self.assertAlmostEqual(later["nav"], 10/9*.9998)
        self.assertEqual(later["events"][-1]["actions"], [])

    def test_split_cash_entitlement_is_per_old_share_and_before_sell(self):
        state = self.start()
        q, v = fixtures(DAYS[1], 4.5)
        v["actions"][A][DAYS[1]].update(split_ratio=2., cash_per_old_share=1.)
        result = ledger.advance(state, decision(B), v, q, DAYS[1])
        self.assertAlmostEqual(result["nav"], .9998)
        self.assertAlmostEqual(result["events"][-1]["actions"][0]["cash_received"], .1)
        self.assertEqual(result["entry_date"], DAYS[1])
        self.assertEqual(state["nav"], 1.)

    def test_same_date_and_foreign_account_never_inherit_or_retrade(self):
        state = self.start()
        q, v = fixtures(DAYS[0], 100.)
        self.assertEqual(ledger.advance(state, decision(B), v, q, DAYS[0]), state)
        foreign = deepcopy(state)
        foreign["candidate_id"] = "vd_7d52c290adc6c4503f7d"
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            ledger.advance(foreign, decision(), v, q, DAYS[1])

    def test_gap_carry_and_unknown_or_revised_actions_fail_without_mutation(self):
        state = self.start()
        q, v = fixtures(DAYS[2], 11.)
        v["raw_histories"][A] = [r for r in v["raw_histories"][A] if r[0] != DAYS[1]]
        v["actions"][A][DAYS[1]].update(not_observed=True, verification="bracketed_no_action_interval",
                                      previous_quote_date=DAYS[0], next_quote_date=DAYS[2])
        result = ledger.advance(state, decision(), v, q, DAYS[2])
        self.assertAlmostEqual(result["nav"], 1.1)
        v["actions"][A][DAYS[1]]["cash_per_old_share"] = .1
        with self.assertRaisesRegex(ValueError, "unsafe unobserved"):
            ledger.advance(state, decision(), v, q, DAYS[2])
        q, v = fixtures(DAYS[3])
        v["actions"][A][DAYS[2]]["cash_per_old_share"] = .1
        with self.assertRaisesRegex(ValueError, "previously booked"):
            ledger.advance(result, decision(), v, q, DAYS[3])


class RuntimeTests(unittest.TestCase):
    def test_independent_temp_start_sealed_repeat_and_read_only_cached_block(self):
        day = DAYS[1]
        q, v = fixtures(day)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"shadow_v92_plus.json"
            now = datetime(2026, 9, 28, 14, 50)
            first = runtime.run(q, day, path, now, lambda *a, **k: v, lambda *a, **k: decision())
            original = path.read_bytes()
            state = json.loads(original)
            self.assertEqual(state["nav"], 1.)
            self.assertEqual(state["candidate_id"], policy.CANDIDATE_ID)
            self.assertIn("【影子 V9.2+】", "\n".join(first))
            never = Mock(side_effect=AssertionError("sealed day recalculated"))
            self.assertEqual(runtime.run({}, day, path, now, never, never), first)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(shadow_v92_plus.saved_block(day, path), first)
            self.assertIsNone(shadow_v92_plus.saved_block(DAYS[2], path))

    def test_failed_validation_policy_or_commit_never_creates_or_mutates_account(self):
        day, later = DAYS[1:3]
        q, v = fixtures(day)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"state.json"
            with self.assertRaisesRegex(ValueError, "policy failed"):
                runtime.run(q, day, path, datetime(2026,9,28,14,50), lambda *a, **k: v,
                            Mock(side_effect=ValueError("policy failed")))
            self.assertFalse(path.exists())
            runtime.run(q, day, path, datetime(2026,9,28,14,50), lambda *a, **k: v, lambda *a, **k: decision())
            original = path.read_bytes()
            q, v = fixtures(later)
            bad = deepcopy(v)
            bad["raw_histories"][A][-1] = (later, 1., 1., 100.)
            with self.assertRaisesRegex(ValueError, "differs"):
                runtime.run(q, later, path, datetime(2026,9,29,14,50), lambda *a, **k: bad,
                            lambda *a, **k: decision(B))
            with patch.object(runtime, "atomic_json", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    runtime.run(q, later, path, datetime(2026,9,29,14,50), lambda *a, **k: v,
                                lambda *a, **k: decision(B))
            self.assertEqual(path.read_bytes(), original)

    def test_stale_after_slow_policy_and_concurrent_writer_fail_safely(self):
        from v10_live.runtime import locked
        q, v = fixtures(DAYS[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"state.json"
            moments = iter([datetime(2026,9,28,14,50)]*3 + [datetime(2026,9,28,14,54)]*3)
            with patch("v10_live.runtime.runtime_clock", return_value=lambda: next(moments)):
                with self.assertRaisesRegex(ValueError, "陈旧|window|窗口"):
                    runtime.run(q, DAYS[1], path, build_view=lambda *a, **k: v, decide=lambda *a, **k: decision())
            self.assertFalse(path.exists())
            with locked(path):
                with self.assertRaises(BlockingIOError):
                    runtime.run(q, DAYS[1], path, datetime(2026,9,28,14,50))

    def test_fsync_crossing_quote_age_boundary_cannot_commit(self):
        q, v = fixtures(DAYS[1])
        moment = [datetime(2026,9,28,14,50)]
        def slow_fsync(unused_fd):
            moment[0] = datetime(2026,9,28,14,54)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"state.json"
            with patch("v10_live.runtime.runtime_clock", return_value=lambda: moment[0]), \
                    patch("v10_live.runtime.os.fsync", side_effect=slow_fsync):
                with self.assertRaisesRegex(ValueError, "陈旧"):
                    runtime.run(q, DAYS[1], path, build_view=lambda *a, **k: v,
                                decide=lambda *a, **k: decision())
            self.assertFalse(path.exists())


class FrozenHistoryFidelityTests(unittest.TestCase):
    def test_full_simple_and_baseline_features_targets_and_actual_age_match_frozen_reference(self):
        """Two frozen controls only, 3097 dates; no search, network or state I/O."""
        from v10_deep.cli import evaluate
        from v10_deep.reference import run_reference, Policy, Portfolio, DatedValues
        import strategy as mutable_strategy
        profile, native, meta = evaluate("simple")
        self.assertEqual(profile["config"], policy.verified_profile()["config"])
        with np.load(ROOT/"v10_deep/cache/features.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("features", "scores", "fear")}
        base = policy.load_profile("v92")["config"]
        histories, date_axes = {}, {}
        for code in ASSETS:
            with (ROOT/"v10_h_close/corrected_snapshots"/(code+".csv")).open() as stream:
                histories[code] = [(r[0], float(r[2]), float(r[3])) for r in csv.reader(stream) if r]
            date_axes[code] = [r[0] for r in histories[code]]
        expected = run_reference(arrays, meta, profile["config"], start="2014-01-02", end="2026-09-24")
        np.testing.assert_array_equal(expected["holdings"], native["holdings"][0])
        np.testing.assert_array_equal(expected["returns"], native["returns"][0])
        frozen_data = DatedValues(arrays, meta, base)
        simple_data = DatedValues(arrays, meta, profile["config"])
        before = {k: deepcopy(v) for k, v in vars(mutable_strategy).items() if k.isupper()}
        state = {}
        with patch.object(policy, "verified_profile", return_value=profile), \
                patch.object(policy, "qvix_state", return_value={"active": False}):
            for trace in expected["trace"]:
                date = trace["date"]
                index = bisect_right(meta["dates"], date)-1
                values = {}
                for code in ASSETS:
                    j = bisect_right(date_axes[code], date)
                    values[code] = policy.latest_features(histories[code][max(0,j-270):j], date)
                    if code == policy.CASH:
                        values[code].update(valid=0., score=0.)
                    elif values[code]["valid"]:
                        self.assertEqual(values[code]["score"], simple_data.score(index, code), (date, code))
                        for field in ("ret1","mom5","mom20","mom60","vol20","ma250","volume_ratio"):
                            self.assertEqual(values[code][field], frozen_data.value(index, code, field), (date,code,field))
                data = policy.Snapshot(values, index, arrays["fear"][index])
                calendar = meta["dates"][:index+1]
                with patch.object(policy, "snapshot", return_value=(data,calendar)):
                    result = policy.decide({}, calendar, date, state)
                self.assertEqual(result["intended_target"], trace["intended"], date)
                self.assertEqual(result["target"], trace["holding"], date)
                self.assertEqual(result["crash_requested"], trace["crash_requested"], date)
                # Baseline minhold0 uses exactly the same features under the
                # same actual account; independently compare its native cache.
                entry = state.get("entry_date")
                age = index-meta["dates"].index(entry) if entry else 0
                trigger = state.get("crash_trigger_date")
                lock = meta["dates"].index(trigger)+5 if trigger else -1
                account = Portfolio(holding=state.get("holding"), age=age, lock_until=lock)
                can_sell = account.holding is None or np.isfinite(data.price(index,account.holding))
                self.assertEqual(Policy(data,base).intent(index,deepcopy(account),can_sell),
                                 Policy(frozen_data,base).intent(index,deepcopy(account),can_sell), date)
                if result["executable"]:
                    entry = date if result["target"] != state.get("holding") else entry
                    state = dict(holding=result["target"], entry_date=entry, last_date=date,
                                 crash_trigger_date=result["crash_trigger_date"], crash_code=result["crash_code"])
        self.assertEqual(before, {k:v for k,v in vars(mutable_strategy).items() if k.isupper()})


if __name__ == "__main__":
    unittest.main()
