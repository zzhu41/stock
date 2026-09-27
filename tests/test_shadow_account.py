"""Temporary-account regression tests; no market requests, old state or messages."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import shadow_account
import shadow_0906
import shadow_v92
from test_live_signals import DATES, MODULES, call, history, isolated

A, B = "510300", "510500"


def actions(day, code=A, cash=0., split=1.):
    return {code: {day: dict(split_ratio=split, cash_per_old_share=cash)}}


class ShadowAccountTests(unittest.TestCase):
    def test_old_holder_receives_split_and_cash_before_switch_new_target_does_not(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module):
                with patch.object(module.strategy, "decide", return_value=(A, "entry")):
                    call(module, [], history(20), {A: 100., B: 10.}, DATES[20])
                events = actions(DATES[21], cash=10., split=2.)
                events.update(actions(DATES[21], B, cash=50.))
                with patch.object(module.strategy, "decide", return_value=(B, "switch")):
                    call(module, [], history(21), {A: 50., B: 10.}, DATES[21], actions=events)
                state = module._load_state()
                self.assertAlmostEqual(state["nav"], 1.1 * (1 - module.FEE))
                self.assertAlmostEqual(state["units"], state["nav"] / 10.)
                self.assertEqual(state["last_settlement_action"]["code"], A)
                self.assertEqual(state["events"][-1]["actions"][0]["cash_received"], .1)
                self.assertEqual(len(state["events"][-1]["actions"]), 1)

    def test_cold_entry_does_not_collect_that_dates_dividend(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module):
                with patch.object(module.strategy, "decide", return_value=(A, "entry")):
                    call(module, [], history(20), {A: 90.}, DATES[20], actions=actions(DATES[20], cash=10.))
                state = module._load_state()
                self.assertEqual(state["nav"], 1.)
                self.assertEqual(state["events"][-1]["actions"], [])
                self.assertIsNone(state["last_settlement_action"])

    def test_skipped_completed_dividend_days_reinvest_without_inventing_trades(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module):
                with patch.object(module.strategy, "decide", return_value=(A, "hold")):
                    call(module, [], history(20), {A: 100.}, DATES[20])
                    hs = history(23)
                    hs[A][21] = (DATES[21], 90., 90., 100.)
                    hs[A][22] = (DATES[22], 99., 99., 100.)
                    event = actions(DATES[21], cash=10.)
                    call(module, [], hs, {A: 99.}, DATES[23], actions=event)
                state = module._load_state()
                self.assertAlmostEqual(state["nav"], 1.1)
                self.assertEqual([e["date"] for e in state["events"]], [DATES[20], DATES[23]])
                self.assertFalse(any(e["trade"] for e in state["events"]))

    def test_bracketed_missing_day_carries_units_but_unknown_action_blocks(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module) as root:
                with patch.object(module.strategy, "decide", return_value=(B, "hold")):
                    call(module, [], history(20), {B: 100.}, DATES[20])
                    hs = history(22)
                    hs[B] = [r for r in hs[B] if r[0] != DATES[21]]
                    before = (root / "state.json").read_bytes()
                    with self.assertRaisesRegex(ValueError, "公司行动"):
                        call(module, [], hs, {B: 110.}, DATES[22])
                    self.assertEqual((root / "state.json").read_bytes(), before)
                    marker = dict(not_observed=True, split_ratio=1., cash_per_old_share=0.,
                                  verification="bracketed_no_action_interval",
                                  previous_quote_date=DATES[20], next_quote_date=DATES[22])
                    call(module, [], hs, {B: 110.}, DATES[22], actions={B: {DATES[21]: marker}})
                self.assertAlmostEqual(module._load_state()["nav"], 1.1)
                self.assertEqual(module._load_state()["events"][-1]["actions"], [])

    def test_previous_provisional_action_of_sold_asset_is_rechecked(self):
        for module in MODULES:
            for original_cash in (0., 1.):
                with self.subTest(module=module.__name__, cash=original_cash), isolated(module) as root:
                    with patch.object(module.strategy, "decide", return_value=(A, "entry")):
                        call(module, [], history(20), {A: 100., B: 100.}, DATES[20])
                    with patch.object(module.strategy, "decide", return_value=(B, "switch")):
                        call(module, [], history(21), {A: 100., B: 100.}, DATES[21],
                             actions=actions(DATES[21], cash=original_cash))
                    before = (root / "state.json").read_bytes()
                    changed = actions(DATES[21], cash=original_cash + .001)
                    with self.assertRaisesRegex(ValueError, "被修订"):
                        call(module, [], history(22), {A: 100., B: 100.}, DATES[22], actions=changed)
                    self.assertEqual((root / "state.json").read_bytes(), before)

    def test_json_failure_never_appends_csv_and_retry_has_one_trade(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module) as root:
                with patch.object(module.strategy, "decide", return_value=(A, "entry")):
                    call(module, [], history(20), {A: 100., B: 100.}, DATES[20])
                before = {name: (root / name).read_bytes() for name in ("state.json", "trades.csv")}
                with patch.object(module.strategy, "decide", return_value=(B, "switch")):
                    with patch.object(module, "_save_state", side_effect=OSError("disk full")):
                        with self.assertRaises(OSError):
                            call(module, [], history(21), {A: 100., B: 100.}, DATES[21])
                    for name, raw in before.items():
                        self.assertEqual((root / name).read_bytes(), raw)
                    call(module, [], history(21), {A: 100., B: 100.}, DATES[21])
                with (root / "trades.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(module._load_state()["nav"], 1 - module.FEE)

    def test_csv_failure_is_visible_but_committed_json_and_same_day_repair_are_idempotent(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module) as root:
                with patch.object(module.strategy, "decide", return_value=(A, "entry")):
                    call(module, [], history(20), {A: 100., B: 100.}, DATES[20])
                with patch.object(module.strategy, "decide", return_value=(B, "switch")), \
                        patch.object(shadow_account, "project_csv", side_effect=OSError("CSV unavailable")):
                    lines = call(module, [], history(21), {A: 100., B: 100.}, DATES[21])
                self.assertIn("JSON已入账", "\n".join(lines))
                from push_signal import build_markdown
                card = build_markdown("\n".join(lines), query=True)[0]
                if module is shadow_v92:
                    self.assertIn("CSV导出失败", card)
                else:
                    self.assertNotIn("0906", card)
                    self.assertNotIn("CSV导出失败", card)
                saved = (root / "state.json").read_bytes()
                with patch.object(module.strategy, "decide", side_effect=AssertionError("duplicate decide")), \
                        patch.object(shadow_0906, "qvix_state", side_effect=AssertionError("duplicate QVIX")):
                    repaired = module.block([], {}, {}, DATES[21])
                self.assertEqual((root / "state.json").read_bytes(), saved)
                self.assertNotIn("导出失败", "\n".join(repaired))
                with (root / "trades.csv").open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_account_lock_excludes_another_process_before_it_reads_state(self):
        module = shadow_0906
        with isolated(module) as root:
            code = (
                "import sys,shadow_0906; shadow_0906.STATE_FILE=sys.argv[1]; "
                "shadow_0906.TRADES_FILE=sys.argv[2]; "
                "shadow_0906.block([],{}, {}, signal_date='2026-01-29')")
            with shadow_account.account_lock(module.STATE_FILE):
                process = subprocess.run([sys.executable, "-B", "-c", code,
                                          str(root / "state.json"), str(root / "trades.csv")],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         universal_newlines=True, timeout=5)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("拒绝并发记账", process.stderr)
            self.assertFalse((root / "state.json").exists())
            self.assertFalse((root / "trades.csv").exists())

    def test_day_card_freezes_qvix_and_timeout_recovery_is_read_only(self):
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module) as root:
                with patch.object(module.strategy, "decide", return_value=(A, "entry")):
                    first = call(module, [], history(20), {A: 100.}, DATES[20])
                before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
                changed = (3., 99., DATES[20], True, "new reading")
                with patch.object(shadow_0906, "qvix_state", side_effect=AssertionError("refresh")):
                    self.assertEqual(module.block([], {}, {}, DATES[20], qvix=changed), first)
                    self.assertEqual(module.cached_block(DATES[20]), first)
                    self.assertIsNone(module.cached_block(DATES[21]))
                for name, raw in before.items():
                    self.assertEqual((root / name).read_bytes(), raw)

    def test_migration_archives_entire_state_and_csv_before_new_baseline(self):
        old = dict(holding=A, nav=1.234, last_date=DATES[19], start_date=DATES[0],
                   valuation_version=3, lock_code=None, mark_anchor_close=88., extra_record={"preserve": True})
        legacy_csv = "date,from,to,price,nav\n2025-01-01,510500,510300,88.0,1.1\n"
        for module in MODULES:
            with self.subTest(module=module.__name__), isolated(module, old) as root:
                (root / "trades.csv").write_text(legacy_csv)
                with patch.object(module.strategy, "decide", return_value=(A, "hold")):
                    call(module, [], history(20), {A: 100.}, DATES[20])
                state = module._load_state()
                archived = state["legacy_performance"]
                self.assertEqual(archived["state"], old)
                self.assertEqual(archived["trades_csv"], legacy_csv)
                self.assertEqual(archived["trades_sha256"], hashlib.sha256(legacy_csv.encode()).hexdigest())
                self.assertEqual(state["nav"], 1.)
                self.assertEqual(state["start_date"], DATES[20])
                self.assertEqual(len(state["events"]), 1)
                self.assertIn("旧口径净值", "\n".join(state["saved_lines"]))

    def test_wrong_v4_identity_or_corrupt_journal_cannot_silently_rebaseline(self):
        with isolated(shadow_0906) as root:
            with patch.object(shadow_0906.strategy, "decide", return_value=(A, "entry")):
                call(shadow_0906, [], history(20), {A: 100.}, DATES[20])
            state = shadow_0906._load_state()
            for key, value in (("account_id", "v9.2"), ("events", [])):
                bad = deepcopy(state)
                bad[key] = value
                path = root / "state.json"
                path.write_text(json.dumps(bad))
                before = path.read_bytes()
                self.assertIsNone(shadow_0906.cached_block(DATES[20]))
                with self.assertRaises(ValueError):
                    call(shadow_0906, [], history(21), {A: 100.}, DATES[21])
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
