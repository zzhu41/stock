"""Read-only dashboard regressions: metric clocks, comparisons and empty state."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import web_app
import web_signals
import signal_store
from v10_live.ledger import CANDIDATE_ID, advance


def version(daily, basis="corrected_tr_same_close"):
    return dict(label="synthetic research", daily=daily,
                trades=[[daily[0][0], None, daily[0][2], daily[0][1]]],
                crash_buys=[], metrics={}, metadata=dict(basis=basis, kind="research"))


class WebDashboardTests(unittest.TestCase):
    def test_corrected_interval_includes_first_return_and_previous_year_close(self):
        ver = version([["2025-12-31", 1., "510300"], ["2026-01-05", 1.1, "510300"],
                       ["2026-01-06", 1.21, "510300"]])
        metrics, years, base = web_app.range_summary(ver, ver["daily"][1:])
        self.assertEqual(base, 1)
        self.assertAlmostEqual(metrics["total_ret"], 21)
        self.assertEqual(metrics["days"], 2)
        self.assertAlmostEqual(years[0][1], 21)
        self.assertAlmostEqual(metrics["ann"], (1.21 ** (244 / 2) - 1) * 100, delta=1)

    def test_corrected_first_day_loss_counts_in_drawdown_and_single_day_is_valid(self):
        ver = version([["2026-01-05", 1., "510300"], ["2026-01-06", .9, "510300"]])
        metrics, years, base = web_app.range_summary(ver, ver["daily"][1:])
        self.assertAlmostEqual(metrics["max_dd"], -10)
        self.assertAlmostEqual(metrics["total_ret"], -10)
        self.assertEqual(metrics["days"], 1)

    def test_legacy_metric_convention_is_preserved(self):
        ver = version([["2026-01-05", 1., "510300"], ["2026-01-06", 1.1, "510300"],
                       ["2026-01-07", 1.21, "510300"]], "legacy_qfq_same_close")
        metrics, _, base = web_app.range_summary(ver, ver["daily"][1:])
        self.assertEqual(base, 1.1)
        self.assertEqual(metrics, web_app.range_metrics(ver["daily"][1:]))

    def test_comparison_is_clipped_to_actual_main_range_and_basis_warning_is_explicit(self):
        main = version([["2026-09-23", 1., "510300"], ["2026-09-24", 1.1, "510300"]])
        other = version(main["daily"] + [["2026-09-28", 2., "510300"]], "legacy_qfq_same_close")
        data = dict(start="2014-01-01", names={"510300": "沪深300ETF"}, benchmarks={},
                    versions={"v10-h": main, "old": other})
        with patch.object(web_app, "v10_signal", return_value={"status": "not_started"}):
            response = web_app.series_response(data, {"version": ["v10-h"], "compare": ["old"]})
        self.assertEqual(response["compare"]["range"], ["2026-09-23", "2026-09-24"])
        self.assertEqual(len(response["compare"]["daily"]), 2)
        self.assertIn("口径不同", response["comparison_warning"])
        self.assertAlmostEqual(response["compare"]["metrics"]["total_ret"], 10)

    def test_corrected_benchmark_overrides_old_global_benchmark(self):
        main = version([["2026-09-23", 1., "510300"], ["2026-09-24", 1.1, "510300"]])
        corrected_benchmark = deepcopy(main)
        corrected_benchmark["label"] = "corrected benchmark"
        main["benchmarks"] = {"510300": corrected_benchmark}
        data = dict(start="2014-01-01", names={}, versions={"v10-h": main},
                    benchmarks={"510300": version([["2026-09-23", 1., "510300"], ["2026-09-24", 4., "510300"]], "legacy_qfq_same_close")})
        with patch.object(web_app, "v10_signal", return_value={}):
            response = web_app.series_response(data, {"compare": ["510300"]})
        self.assertEqual(response["compare"]["label"], "corrected benchmark")
        self.assertIsNone(response["comparison_warning"])
        self.assertAlmostEqual(response["compare"]["metrics"]["total_ret"], 10)

    def test_historical_holding_is_not_taken_from_after_selected_endpoint(self):
        ver = version([["2026-09-23", 1., "510300"], ["2026-09-24", 1.1, "510300"],
                       ["2026-09-28", 1.2, "518880"]])
        ver["trades"].append(["2026-09-28", "510300", "518880", 1.2])
        data = dict(start="2014-01-01", names={}, versions={"v10-h": ver}, benchmarks={})
        with patch.object(web_app, "v10_signal", return_value={}):
            response = web_app.series_response(data, {"end": ["2026-09-24"]})
        self.assertEqual(response["holding"]["code"], "510300")
        self.assertEqual(response["holding"]["date"], "2026-09-24")

    def test_frozen_preview_never_initializes_forward_nav_or_creates_files(self):
        with tempfile.TemporaryDirectory() as directory:
            ver = version([["2026-09-24", 226.669, "513100"]])
            result = web_signals.v10_signal(ver, directory, datetime(2026, 9, 27, 20))
            self.assertEqual(result["status"], "historical_preview")
            self.assertIsNone(result["nav"])
            self.assertIsNone(result["target"])
            self.assertEqual(result["preview"]["code"], "513100")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_readonly_live_card_obeys_signal_freshness_and_account_integrity(self):
        day = "2026-09-28"
        quotes = {"513100": dict(price=10, date=day, timestamp=day + " 14:50:00")}
        view = dict(calendar=[day])
        decision = dict(candidate_id=CANDIDATE_ID, executable=True, target="513100", reason="synthetic signal")
        state = advance(None, decision, view, quotes, day)
        text = "\n".join(["动量轮动信号 | 生成 2026-09-28 14:50:10 | 数据截止 2026-09-28",
            "行情时间: 2026-09-28 14:50:00", "★ 建议: 买入 513100 纳指ETF",
            "【影子 V10-H】", "信号状态: 盘中影子试算"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow_v10.json"
            signal_store.atomic_json(path, state)
            signal_store.publish(text, day, "513100", None, directory)
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            live = web_signals.v10_signal(None, directory, datetime(2026, 9, 28, 14, 51))
            late = web_signals.v10_signal(None, directory, datetime(2026, 9, 28, 15))
            self.assertTrue(live["actionable"])
            self.assertEqual(live["status"], "active")
            self.assertEqual(live["nav"], 1)
            self.assertEqual(late["status"], "stale")
            self.assertFalse(late["actionable"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})
            state["nav"] = 226.669  # does not match actual units/mark
            signal_store.atomic_json(path, state)
            invalid = web_signals.v10_signal(None, directory, datetime(2026, 9, 28, 14, 51))
            self.assertEqual(invalid["status"], "failed")
            self.assertIsNone(invalid["nav"])
