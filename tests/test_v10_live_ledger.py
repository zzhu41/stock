from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import shadow_v10
from v10_live.ledger import advance, CANDIDATE_ID
from v10_live.runtime import run, locked
from v10_live.data import CODES
from v10_live import runtime

DAYS = ["2026-09-24", "2026-09-28", "2026-09-29"]
A, B = "513100", "518880"


def decision(target=A, **kwargs):
    result = dict(target=target, candidate_id=CANDIDATE_ID, executable=True,
                  reason="测试信号", crash_trigger_date=None, crash_code=None,
                  diagnostics={"qvix": {"warning": "当日QVIX缺失，渠道已停用"}})
    result.update(kwargs)
    return result


def fixtures(day, price=10):
    quotes = {code: dict(price=price, open=price, volume=1000., date=day, timestamp=day + " 14:50:00") for code in CODES}
    view = dict(calendar=DAYS, histories={}, metadata={},
                raw_histories={code: [(d, 10, 10, 100) for d in DAYS] for code in (A, B)},
                actions={code: {d: dict(split_ratio=1, cash_per_old_share=0) for d in DAYS}
                         for code in (A, B)})
    return quotes, view


class LedgerTests(unittest.TestCase):
    def start(self):
        quotes, view = fixtures(DAYS[0])
        return advance(None, decision(), view, quotes, DAYS[0])

    def test_first_fill_starts_at_one_without_backtest_return_or_today_dividend(self):
        quotes, view = fixtures(DAYS[0])
        view["actions"][A][DAYS[0]]["cash_per_old_share"] = 1
        state = advance(None, decision(), view, quotes, DAYS[0])
        self.assertEqual(state["nav"], 1)
        self.assertEqual(state["units"], 0.1)
        self.assertEqual(state["events"][0]["actions"], [])

    def test_continuing_holder_receives_ex_dividend_before_switch(self):
        state = self.start()
        quotes, view = fixtures(DAYS[1], price=9)
        view["actions"][A][DAYS[1]]["cash_per_old_share"] = 1
        result = advance(state, decision(B), view, quotes, DAYS[1])
        self.assertAlmostEqual(result["nav"], 0.9998)
        self.assertEqual(result["events"][-1]["actions"][0]["cash_received"], 0.1)
        self.assertEqual(result["holding"], B)
        self.assertEqual(state, self.start())

    def test_new_target_does_not_receive_its_ex_dividend(self):
        state = self.start()
        quotes, view = fixtures(DAYS[1])
        view["actions"][B][DAYS[1]]["cash_per_old_share"] = 2
        result = advance(state, decision(B), view, quotes, DAYS[1])
        self.assertAlmostEqual(result["nav"], 0.9998)
        self.assertEqual(result["events"][-1]["actions"], [])

    def test_intraday_buy_after_ex_date_earns_only_following_raw_price_return(self):
        quotes, view = fixtures(DAYS[0], price=9.2)
        view["actions"][A][DAYS[0]]["cash_per_old_share"] = 1
        state = advance(None, decision(), view, quotes, DAYS[0])
        quotes, view = fixtures(DAYS[1], price=9.66)
        result = advance(state, decision(), view, quotes, DAYS[1])
        self.assertAlmostEqual(result["nav"], 1.05)

    def test_split_preserves_wealth_and_cash_is_per_old_share(self):
        state = self.start()
        quotes, view = fixtures(DAYS[1], price=4.5)
        view["actions"][A][DAYS[1]] = dict(split_ratio=2, cash_per_old_share=1)
        result = advance(state, decision(), view, quotes, DAYS[1])
        self.assertAlmostEqual(result["nav"], 1)
        self.assertAlmostEqual(result["units"], 1 / 4.5)

    def test_skipped_day_cash_reinvests_at_completed_close_without_invented_trades(self):
        state = self.start()
        quotes, view = fixtures(DAYS[2], price=11)
        view["actions"][A][DAYS[1]]["cash_per_old_share"] = 1
        result = advance(state, decision(), view, quotes, DAYS[2])
        self.assertAlmostEqual(result["nav"], 1.21)
        self.assertEqual(len(result["events"]), 2)

    def test_duplicate_date_does_not_mark_switch_or_restart_crash_lock(self):
        state = self.start()
        quotes, view = fixtures(DAYS[0], price=100)
        result = advance(state, decision(B, crash_trigger_date=DAYS[0], crash_code=B), view, quotes, DAYS[0])
        self.assertEqual(state, result)

    def test_later_revision_of_booked_ex_action_stops_instead_of_silently_drifting(self):
        state = self.start()
        quotes, view = fixtures(DAYS[1])
        second = advance(state, decision(B), view, quotes, DAYS[1])
        # Even an initially zero distribution on the sold holding must be
        # checked once the provisional day becomes a finalized source row.
        quotes, view = fixtures(DAYS[2])
        view["actions"][A][DAYS[1]]["cash_per_old_share"] = .2
        saved = deepcopy(second)
        with self.assertRaisesRegex(ValueError, "previously booked provisional"):
            advance(second, decision(B), view, quotes, DAYS[2])
        self.assertEqual(second, saved)

    def test_missing_verified_action_aborts_without_mutating_state(self):
        state = self.start()
        saved = deepcopy(state)
        quotes, view = fixtures(DAYS[1])
        del view["actions"][A][DAYS[1]]
        with self.assertRaisesRegex(ValueError, "unverified corporate action"):
            advance(state, decision(), view, quotes, DAYS[1])
        self.assertEqual(state, saved)

    def test_older_foreign_and_nonexecutable_inputs_are_rejected(self):
        state = self.start()
        quotes, view = fixtures(DAYS[1])
        state["candidate_id"] = "another version"
        with self.assertRaises(ValueError):
            advance(state, decision(), view, quotes, DAYS[1])
        with self.assertRaises(ValueError):
            advance(None, decision(executable=False), view, quotes, DAYS[1])
        with self.assertRaisesRegex(ValueError, "backdated"):
            advance(self.start(), decision(), view, quotes, "2026-09-23")


class RuntimeTests(unittest.TestCase):
    def test_one_atomic_state_and_same_day_reply_avoids_all_recalculation(self):
        quotes, view = fixtures(DAYS[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.json"
            now = datetime(2026, 9, 24, 14, 50)
            first = run(quotes, DAYS[0], path, now, lambda *a, **k: view, lambda *a, **k: decision())
            saved = path.read_bytes()
            never = Mock(side_effect=AssertionError("same-day recomputation"))
            second = run({}, DAYS[0], path, now, never, never)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), saved)
            self.assertIn("当日QVIX缺失，渠道已停用", "\n".join(first))
            self.assertEqual(len(json.loads(saved)["events"]), 1)

    def test_invalid_historical_or_unavailable_data_never_creates_account(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.json"
            with self.assertRaisesRegex(ValueError, "historical"):
                run({}, DAYS[0], path, now=datetime(2026, 9, 27))
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, "missing data"):
                run(fixtures(DAYS[0])[0], DAYS[0], path, now=datetime(2026, 9, 24, 14, 50),
                    build_view=Mock(side_effect=ValueError("missing data")))
            self.assertFalse(path.exists())

    def test_failed_commit_leaves_previous_account_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.json"
            quotes, view = fixtures(DAYS[0])
            run(quotes, DAYS[0], path, datetime(2026, 9, 24, 14, 50), lambda *a, **k: view, lambda *a, **k: decision())
            before = path.read_bytes()
            quotes, view = fixtures(DAYS[1])
            with patch("v10_live.runtime.atomic_json", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    run(quotes, DAYS[1], path, datetime(2026, 9, 28, 14, 50), lambda *a, **k: view, lambda *a, **k: decision(B))
            self.assertEqual(path.read_bytes(), before)

    def test_concurrent_account_lock_fails_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.json"
            with locked(path):
                with self.assertRaises(BlockingIOError):
                    with locked(path):
                        self.fail("second lock succeeded")

    def test_timeout_only_disables_v10_section(self):
        with patch.object(shadow_v10.subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 60)), \
                patch.object(shadow_v10, "saved_block", return_value=None):
            text = "\n".join(shadow_v10.block({}, DAYS[0]))
        self.assertIn("【影子 V10-H】", text)
        self.assertIn("计算失败", text)
        self.assertIn("无有效建议", text)
        self.assertNotIn("建议:", text)

    def test_commit_then_worker_timeout_recovers_actual_saved_card(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.json"
            quotes, view = fixtures(DAYS[0])
            expected = run(quotes, DAYS[0], path, datetime(2026, 9, 24, 14, 50),
                           lambda *a, **k: view, lambda *a, **k: decision())
            before = path.read_bytes()
            with patch.object(shadow_v10, "STATE_FILE", str(path)), \
                    patch.object(shadow_v10.subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 60)):
                actual = shadow_v10.block(quotes, DAYS[0])
                self.assertEqual(expected, actual)
                self.assertEqual(path.read_bytes(), before)
                self.assertIsNone(shadow_v10.saved_block(DAYS[1]))


class RuntimeFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime_guard_")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "shadow.json"
        self.date = DAYS[1]
        self.now = datetime(2026, 9, 28, 14, 50, 5)
        self.clock = patch.object(runtime, "runtime_clock", return_value=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_cached_view_cannot_book_quote_that_expired_after_validation(self):
        quotes, view = fixtures(self.date)
        for quote in quotes.values():
            quote["timestamp"] = self.date + " 14:47:10"  # 175 seconds at first guard.
        def cached_view(*args, **kwargs):
            self.now = datetime(2026, 9, 28, 14, 50, 31)  # 201 seconds before valuation.
            return view
        with patch.object(runtime, "advance") as account, patch.object(runtime, "atomic_json") as commit:
            with self.assertRaisesRegex(ValueError, "陈旧"):
                run(quotes, self.date, self.path, self.now, cached_view, lambda *a, **k: decision())
            account.assert_not_called()
            commit.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_clock_past_window_after_policy_stops_before_valuation(self):
        quotes, view = fixtures(self.date)
        def slow_policy(*args, **kwargs):
            self.now = datetime(2026, 9, 28, 14, 55)
            return decision()
        with patch.object(runtime, "advance") as account:
            with self.assertRaisesRegex(ValueError, "14:50"):
                run(quotes, self.date, self.path, self.now, lambda *a, **k: view, slow_policy)
            account.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_render_crossing_deadline_does_not_replace_previous_account(self):
        first_quotes, first_view = fixtures(DAYS[0])
        self.now = datetime(2026, 9, 24, 14, 50)
        run(first_quotes, DAYS[0], self.path, self.now, lambda *a, **k: first_view, lambda *a, **k: decision())
        before = self.path.read_bytes()
        quotes, view = fixtures(self.date)
        for quote in quotes.values():
            quote["timestamp"] = self.date + " 14:54:50"
        self.now = datetime(2026, 9, 28, 14, 54, 58)
        def slow_render(*args):
            self.now = datetime(2026, 9, 28, 14, 55)
            return ["prepared but expired card"]
        with patch.object(runtime, "render", side_effect=slow_render), patch.object(runtime, "atomic_json") as commit:
            with self.assertRaisesRegex(ValueError, "14:50"):
                run(quotes, self.date, self.path, self.now, lambda *a, **k: view, lambda *a, **k: decision(B))
            commit.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_fsync_delay_is_rechecked_before_atomic_replace_and_temp_removed(self):
        self.path.write_text('{"previous":"account"}')
        before = self.path.read_bytes()
        quotes, unused = fixtures(self.date)
        self.now = datetime(2026, 9, 28, 14, 50)
        def slow_sync(unused_fd):
            self.now = datetime(2026, 9, 28, 14, 55)
        with patch.object(runtime.os, "fsync", side_effect=slow_sync):
            with self.assertRaisesRegex(ValueError, "14:50"):
                runtime.atomic_json(self.path, {"new":"expired"},
                    validator=lambda: runtime.validate_snapshot(quotes, self.date, self.now))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("shadow.json.tmp.*")), [])

    def test_same_day_saved_card_is_readonly_even_after_execution_window(self):
        quotes, view = fixtures(self.date)
        first = run(quotes, self.date, self.path, self.now, lambda *a, **k: view, lambda *a, **k: decision())
        before = self.path.read_bytes()
        self.now = datetime(2026, 9, 28, 16)
        with patch.object(runtime, "validate_snapshot", side_effect=AssertionError("No new transaction")), \
                patch.object(runtime, "advance", side_effect=AssertionError("No new valuation")):
            self.assertEqual(run({}, self.date, self.path, self.now), first)
        self.assertEqual(self.path.read_bytes(), before)

    def test_closed_day_or_outside_window_never_starts_data_or_policy_work(self):
        for day, now in (("2026-09-25", datetime(2026, 9, 25, 14, 50)),
                         (self.date, datetime(2026, 9, 28, 14, 49, 59))):
            self.now = now
            quotes, view = fixtures(day)
            never = Mock(side_effect=AssertionError("Outside-window work"))
            with self.assertRaisesRegex(ValueError, "14:50"):
                run(quotes, day, self.path, now, never, never)
            never.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_explicit_staging_path_does_not_write_default_production_account(self):
        quotes, view = fixtures(self.date)
        production = self.path.parent / "production.json"
        production.write_text('{"unrelated":"unchanged"}')
        before = production.read_bytes()
        with patch.object(runtime, "STATE", production):
            run(quotes, self.date, self.path, self.now, lambda *a, **k: view, lambda *a, **k: decision())
        self.assertEqual(production.read_bytes(), before)
        self.assertEqual(json.loads(self.path.read_text())["nav"], 1.)


class RuntimeClockTests(unittest.TestCase):
    def test_injected_clock_still_advances_by_monotonic_elapsed_work(self):
        now = datetime(2026, 9, 28, 14, 50)
        with patch.object(runtime.time, "monotonic", side_effect=[100., 160.]):
            clock = runtime.runtime_clock(now)
            current = clock()
        self.assertIsNone(current.tzinfo)
        self.assertEqual(current, now + timedelta(seconds=60))
