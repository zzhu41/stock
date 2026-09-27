from contextlib import ExitStack
from datetime import datetime
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import signal_daily
import signal_store

NOW = datetime(2026, 9, 28, 14, 50, 30)
DAY = "2026-09-28"


class V10DailyIntegrationTests(unittest.TestCase):
    def setup_job(self, stack, directory, crash=False):
        quotes = {c: {"price": 10, "date": DAY, "timestamp": DAY + " 14:50:00",
                      "open": 10, "volume": 100} for c in signal_daily.UNIVERSE}
        histories = {c: [("2026-09-24", 10, 10, 100)] for c in signal_daily.UNIVERSE}
        stack.enter_context(patch.object(signal_daily, "SIGNAL_DIR", directory))
        stack.enter_context(patch.object(signal_daily, "CRASH_LOCK", str(Path(directory) / "crash_lock.json")))
        stack.enter_context(patch.object(signal_daily, "load_portfolio", return_value={"holding": None}))
        hist = stack.enter_context(patch.object(signal_daily, "fetch_histories", return_value=histories))
        stack.enter_context(patch.object(signal_daily, "fetch_realtime", return_value=quotes))
        table = [("510500", {"mom5": -.09, "dist_ma250": -.25})] if crash else []
        stack.enter_context(patch.object(signal_daily.strategy, "rank", return_value=table))
        stack.enter_context(patch.object(signal_daily.strategy, "advice", return_value=(
            "513100", "旧版原因", "买入 513100 纳指ETF", "旧版排名")))
        extra = stack.enter_context(patch.object(signal_daily, "optional_blocks", return_value=[
            "原0906展示", "原v9.2展示", "⚠️ 影子 V10-H 计算失败: H unavailable"]))
        stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
        return quotes, hist, extra

    def test_optional_failure_preserves_main_signal_and_day_is_frozen_once(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            quotes, hist, extra = self.setup_job(stack, directory)
            self.assertEqual(signal_daily.main(now=NOW), "513100")
            saved, journal = signal_store.load_saved(directory)
            self.assertIn("原0906展示", saved)
            self.assertIn("原v9.2展示", saved)
            self.assertIn("影子 V10-H 计算失败", saved)
            self.assertIn("行情时间: 2026-09-28 14:50:00", saved)
            self.assertEqual(saved, (Path(directory) / (DAY + ".txt")).read_text())
            self.assertEqual(extra.call_args[0][2], quotes)
            hist.side_effect = AssertionError("same-day rerun must not fetch/trade again")
            extra.side_effect = AssertionError("same-day rerun must not update accounts")
            self.assertEqual(signal_daily.main(now=NOW), "513100")
            self.assertEqual(signal_store.load_saved(directory)[1], journal)

    def test_crash_candidate_with_failed_signal_commit_never_persists_a_lock(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.setup_job(stack, directory, crash=True)
            stack.enter_context(patch.object(signal_store, "atomic_json", side_effect=OSError("disk full")))
            with self.assertRaises(OSError):
                signal_daily.main(now=NOW)
            self.assertFalse((Path(directory) / "crash_lock.json").exists())
            self.assertFalse((Path(directory) / "daily_state.json").exists())

    def test_outside_execution_window_cannot_fetch_or_commit(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, hist, extra = self.setup_job(stack, directory)
            with self.assertRaises(ValueError):
                signal_daily.main(now=NOW.replace(hour=15))
            hist.assert_not_called()
            extra.assert_not_called()
