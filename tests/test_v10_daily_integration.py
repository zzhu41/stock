from contextlib import ExitStack
from datetime import datetime
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import premium
import shadow_0906
import shadow_v92
import shadow_v10
import signal_daily


class V10DailyIntegrationTests(unittest.TestCase):
    def test_v10_failure_preserves_old_blocks_and_publishes_daily_error(self):
        day = datetime.now().strftime("%Y-%m-%d")
        quotes = {c: {"price": 10, "date": day, "timestamp": day + " 14:50:00",
                      "open": 10, "volume": 100} for c in signal_daily.UNIVERSE}
        histories = {c: [(day, 10, 10, 100)] for c in signal_daily.UNIVERSE}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(signal_daily, "SIGNAL_DIR", directory))
            stack.enter_context(patch.object(signal_daily, "CRASH_LOCK", str(Path(directory) / "lock.json")))
            stack.enter_context(patch.object(signal_daily, "load_portfolio", return_value={"holding": None}))
            stack.enter_context(patch.object(signal_daily, "fetch_history", side_effect=lambda c: histories[c]))
            stack.enter_context(patch.object(signal_daily, "fetch_realtime", return_value=quotes))
            stack.enter_context(patch.object(signal_daily, "prepare_live_histories", return_value=histories))
            stack.enter_context(patch.object(signal_daily.time, "sleep"))
            stack.enter_context(patch.object(signal_daily.strategy, "rank", return_value=[]))
            stack.enter_context(patch.object(signal_daily.strategy, "advice", return_value=(
                "513100", "旧版原因", "买入 513100 纳指ETF", "旧版排名")))
            stack.enter_context(patch.object(premium, "signal_block", return_value=[]))
            old_0906 = stack.enter_context(patch.object(shadow_0906, "block", return_value=["原0906展示"]))
            old_v92 = stack.enter_context(patch.object(shadow_v92, "block", return_value=["原v9.2展示"]))
            new_v10 = stack.enter_context(patch.object(shadow_v10, "block", side_effect=ValueError("H unavailable")))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            result = signal_daily.main()
            saved = (Path(directory) / "latest.txt").read_text()
            self.assertEqual(result, "513100")
            self.assertIn("原0906展示", saved)
            self.assertIn("原v9.2展示", saved)
            self.assertIn("影子 V10-H 计算失败", saved)
            self.assertEqual(saved, (Path(directory) / (day + ".txt")).read_text())
            self.assertEqual(old_0906.call_args[0][1], histories)
            self.assertEqual(old_v92.call_args[0][1], histories)
            new_v10.assert_called_once_with(quotes, signal_date=day)
            self.assertFalse((Path(directory) / "shadow_v10.json").exists())
