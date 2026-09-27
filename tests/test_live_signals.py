"""Offline regressions for dated signals, crash holding periods and real-time shadow NAV."""
import csv
import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import live_state
import shadow_0906
import shadow_v92
import signal_daily
from market_data import CASH


def weekdays(n=30):
    day, out = date(2026, 1, 1), []
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


DATES = weekdays()
MODULES = (shadow_0906, shadow_v92)


def history(last_index, prices=None):
    prices = prices or {"510300": 100.0, "510500": 100.0, CASH: 100.0}
    return {code: [(d, p, p, 100.0) for d in DATES[:last_index + 1]]
            for code, p in prices.items()}


def call(module, table, histories, prices, signal_date=None, actions=None):
    """Explicit verified raw/action fixtures; never infer actions from QFQ ratios."""
    day = signal_date or histories["510300"][-1][0]
    quotes = {code: dict(date=day, timestamp=day + " 14:50:00", price=price)
              for code, price in prices.items()}
    action_rows = {code: {row[0]: dict(split_ratio=1., cash_per_old_share=0.) for row in rows}
                   for code, rows in histories.items()}
    for code, values in (actions or {}).items():
        action_rows[code].update(deepcopy(values))
    view = dict(calendar=[row[0] for row in histories["510300"]], raw_histories=deepcopy(histories),
                actions=action_rows, metadata=dict(signal_date=day))
    return module.block(table, histories, prices, signal_date=day, quotes=quotes, action_view=view)


@contextmanager
def isolated(module, state=None):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        for attr, name in (("STATE_FILE", "state.json"), ("TRADES_FILE", "trades.csv"),
                           ("PORTFOLIO", "portfolio.json")):
            stack.enter_context(patch.object(module, attr, str(root / name)))
        if state is not None:
            (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
        stack.enter_context(patch.object(shadow_0906, "qvix_state",
                                        side_effect=lambda d: (0, 15, d, False, "")))
        yield root


class LiveSignalTests(unittest.TestCase):
    def test_crash_lock_holds_through_four_following_sessions_releases_on_fifth(self):
        crash = [("510300", {"mom5": -0.09, "dist_ma250": -0.25})]
        prices = {"510300": 100.0, "510500": 100.0, CASH: 100.0}
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module):
                with patch.object(module.strategy, "decide", return_value=(CASH, "ordinary exit")):
                    call(module, crash, history(20), prices, signal_date=DATES[20])
                    self.assertEqual(module._load_state()["lock_trigger_date"], DATES[20])
                    for i in range(21, 25):
                        call(module, [], history(i), prices, signal_date=DATES[i])
                        self.assertEqual(module._load_state()["holding"], "510300")
                    call(module, [], history(25), prices, signal_date=DATES[25])
                    self.assertEqual(module._load_state()["holding"], CASH)

    def test_nav_uses_saved_execution_price_and_does_not_backdate_entry(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module):
                with patch.object(module.strategy, "decide", return_value=("510300", "buy")):
                    prices = {"510300": 120.0, "510500": 100.0, CASH: 100.0}
                    call(module, [], history(20), prices, signal_date=DATES[20])
                    st = module._load_state()
                    self.assertEqual(st["nav"], 1.0)
                    self.assertEqual(st["mark_price"], 120.0)
                    self.assertAlmostEqual(st["units"], 1 / 120.)
                    # Yesterday's cache later says 105; entry was still 120, not 105.
                    hs = history(21)
                    hs["510300"][-2] = (DATES[20], 105, 105, 100)
                    prices["510300"] = 132.0
                    call(module, [], hs, prices, signal_date=DATES[21])
                    self.assertAlmostEqual(module._load_state()["nav"], 1.1)

    def test_raw_units_handle_cash_dividend_split_and_no_action(self):
        for module in MODULES:
            for ratio, cash, price, expected in ((2., 0., 50., 1.), (1., 10., 100., 1.1), (1., 0., 100., 1.)):
                with self.subTest(module=module.__name__, ratio=ratio, cash=cash), isolated(module):
                    with patch.object(module.strategy, "decide", return_value=("510300", "keep")):
                        call(module, [], history(20), {"510300": 100.0}, signal_date=DATES[20])
                        rebased = history(21)
                        # Arbitrary additive QFQ rebasing must not enter raw-unit NAV.
                        rebased["510300"] = [(d, o - 10., c - 10., v)
                                               for d, o, c, v in rebased["510300"]]
                        actions = {"510300": {DATES[21]: dict(split_ratio=ratio, cash_per_old_share=cash)}}
                        call(module, [], rebased, {"510300": price}, signal_date=DATES[21], actions=actions)
                        self.assertAlmostEqual(module._load_state()["nav"], expected)
                        call(module, [], history(22), {"510300": price * 1.1}, signal_date=DATES[22], actions=actions)
                        self.assertAlmostEqual(module._load_state()["nav"], expected * 1.1)

    def test_missing_explicit_raw_action_view_does_not_write_state(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module) as root:
                with patch.object(module.strategy, "decide", return_value=("510300", "buy")):
                    with self.assertRaisesRegex(ValueError, "原始行情/公司行动"):
                        module.block([], history(0), {"510300": 100.0}, signal_date=DATES[0])
                self.assertFalse((root / "state.json").exists())

    def test_version2_rebaseline_preserves_older_archive(self):
        older = {"nav": 0.8, "last_date": DATES[10]}
        old = {"holding": "510300", "nav": 1.2, "last_date": DATES[19],
               "start_date": DATES[11], "lock_code": None, "valuation_version": 2,
               "mark_price": 100.0, "mark_code": "510300", "legacy_performance": older}
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module, old):
                with patch.object(module.strategy, "decide", return_value=("510300", "keep")):
                    lines = call(module, [], history(20), {"510300": 50.0}, signal_date=DATES[20])
                st = module._load_state()
                self.assertEqual(st["valuation_version"], 4)
                self.assertEqual(st["nav"], 1.0)
                self.assertEqual(st["legacy_performance"]["nav"], 1.2)
                self.assertEqual(st["legacy_performance_history"], [older])
                self.assertIn("旧口径净值", "\n".join(lines))

    def test_same_day_repeat_does_not_mark_again_or_trade(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module) as root:
                prices = {"510300": 100.0, "510500": 100.0, CASH: 100.0}
                with patch.object(module.strategy, "decide", return_value=("510300", "buy")):
                    call(module, [], history(20), prices, signal_date=DATES[20])
                with patch.object(module.strategy, "decide", return_value=("510500", "switch")):
                    call(module, [], history(21), prices, signal_date=DATES[21])
                saved = (root / "state.json").read_bytes()
                trades = (root / "trades.csv").read_bytes()
                prices["510300"] = 150.0
                with patch.object(module.strategy, "decide", side_effect=AssertionError("same-day decide")):
                    call(module, [], history(21), prices, signal_date=DATES[21])
                self.assertEqual((root / "state.json").read_bytes(), saved)
                self.assertEqual((root / "trades.csv").read_bytes(), trades)

    def test_switch_marks_old_holding_then_starts_new_holding_at_execution_price(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module):
                prices = {"510300": 100.0, "510500": 100.0, CASH: 100.0}
                with patch.object(module.strategy, "decide", return_value=("510300", "buy")):
                    call(module, [], history(20), prices, signal_date=DATES[20])
                prices.update({"510300": 110.0, "510500": 200.0})
                with patch.object(module.strategy, "decide", return_value=("510500", "switch")):
                    call(module, [], history(21), prices, signal_date=DATES[21])
                    self.assertAlmostEqual(module._load_state()["nav"], 1.1 * (1 - module.FEE))
                    prices["510500"] = 220.0
                    call(module, [], history(22), prices, signal_date=DATES[22])
                self.assertAlmostEqual(module._load_state()["nav"], 1.21 * (1 - module.FEE))

    def test_stale_history_rejected_before_qvix_or_state_write(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module) as root:
                with patch.object(shadow_0906, "qvix_state", side_effect=AssertionError("stale signal")):
                    with self.assertRaisesRegex(ValueError, "交易日历|行情末日"):
                        call(module, [], history(20), {}, signal_date=DATES[21])
                self.assertFalse((root / "state.json").exists())

    def test_legacy_nav_is_archived_and_new_baseline_is_explicit(self):
        old = {"holding": "510300", "nav": 1.23, "last_date": DATES[19],
               "start_date": DATES[0], "lock_code": None, "lock_until": None}
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module, old):
                with patch.object(module.strategy, "decide", return_value=("510300", "keep")):
                    lines = call(module, [], history(20), {"510300": 150.0}, signal_date=DATES[20])
                st = module._load_state()
                self.assertEqual(st["legacy_performance"]["nav"], 1.23)
                self.assertEqual(st["nav"], 1.0)
                self.assertEqual(st["start_date"], DATES[20])
                self.assertIn("旧口径净值", "\n".join(lines))

    def test_legacy_lock_requires_a_recorded_trigger(self):
        old = {"holding": "510300", "nav": 1.2, "last_date": DATES[20],
               "lock_code": "510300", "lock_until": DATES[20]}
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module, old) as root:
                saved = (root / "state.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "没有触发日"):
                    call(module, [], history(21), {"510300": 100}, signal_date=DATES[21])
                self.assertEqual((root / "state.json").read_bytes(), saved)
        # Main v9 locks already recorded trigger_date, which can be migrated exactly.
        self.assertTrue(live_state.lock_active("510300", DATES[20], DATES[:25], DATES[24]))
        self.assertFalse(live_state.lock_active("510300", DATES[20], DATES[:26], DATES[25]))

    def test_held_crash_candidate_does_not_hide_second_candidate(self):
        old = {"holding": "510300", "nav": 1.0, "last_date": None, "lock_code": None}
        crash = [(code, {"mom5": -0.09, "dist_ma250": -0.25})
                 for code in ("510300", "510500")]
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module, old):
                call(module, crash, history(20), {"510300": 100, "510500": 100},
                             signal_date=DATES[20])
                self.assertEqual(module._load_state()["holding"], "510500")

    def test_missing_today_qvix_disables_channel_with_visible_date(self):
        dates = weekdays(125)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qvix.csv"
            with path.open("w", newline="") as f:
                csv.writer(f).writerows((d, 10 + i % 5) for i, d in enumerate(dates[:-1]))
            with patch.object(shadow_0906, "QVIX_CSV", str(path)), patch.object(shadow_0906, "_fetch_qvix"):
                _, _, latest, fear, note = shadow_0906.qvix_state(dates[-1])
            self.assertFalse(fear)
            self.assertEqual(latest, dates[-2])
            self.assertIn(dates[-1], note)
            self.assertIn("停用", note)

    def test_main_rejects_stale_quotes_without_generating_a_signal(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(signal_daily, "SIGNAL_DIR", directory), \
                patch.object(signal_daily, "fetch_histories", return_value=history(20)), \
                patch.object(signal_daily, "fetch_realtime", return_value={}), \
                patch.object(signal_daily, "prepare_live_histories", side_effect=ValueError("stale quote")), \
                patch.object(signal_daily.signal_store, "publish") as write:
            with self.assertRaisesRegex(ValueError, "stale quote"):
                signal_daily.main(now=datetime(2026, 9, 24, 14, 50))
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
