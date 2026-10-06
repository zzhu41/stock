"""14:30 preview: pushes a labeled live estimate without touching authority."""
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import daily_preview
import signal_store


NOW = datetime(2026, 10, 8, 14, 30, 5)
PAYLOAD = dict(ok=True, mode="live", date="2026-10-08",
               quote_time="2026-10-08 14:29:58", markdown="📊 动量实时估算\n表体")


class DailyPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.signals = Path(self.tmp.name)
        patches = [patch.object(daily_preview.signal_store, "SIGNALS", self.signals),
                   patch.object(daily_preview.signal_store, "trading_day", return_value=True),
                   patch.object(daily_preview.momentum_live, "_load_journal", return_value=None),
                   patch.object(daily_preview.momentum_live, "live_payload", return_value=dict(PAYLOAD)),
                   patch("sys.stdout", new_callable=io.StringIO)]
        started = [p.start() for p in patches]
        self.patches = patches
        self.live = started[3]
        for p in patches:
            self.addCleanup(p.stop)
        self.sent = []

    def send(self, text, title=None):
        self.sent.append((title, text))
        return True

    def status(self):
        return json.loads((self.signals / "preview_status.json").read_text(encoding="utf-8"))

    def test_trading_day_in_window_pushes_labeled_estimate_and_records(self):
        self.assertEqual(daily_preview.main(now=NOW, send=self.send), 0)
        self.assertEqual(len(self.sent), 1)
        title, text = self.sent[0]
        self.assertEqual(title, "动量预推送")
        self.assertIn("预推送", text)
        self.assertIn("以 14:50 为准", text)
        self.assertIn("表体", text)
        record = self.status()
        self.assertEqual(record["status"], "sent")
        self.assertEqual(record["date"], "2026-10-08")
        self.assertEqual(record["quote_time"], "2026-10-08 14:29:58")
        # 权威链路文件绝不被预览触碰。
        self.assertFalse((self.signals / "generation_status.json").exists())
        self.assertFalse((self.signals / "daily_state.json").exists())
        self.assertFalse((self.signals / "delivery.json").exists())

    def test_holiday_skips_without_push(self):
        self.patches[1].stop()
        with patch.object(daily_preview.signal_store, "trading_day", return_value=False):
            self.assertEqual(daily_preview.main(now=NOW, send=self.send), 0)
        self.assertEqual(self.sent, [])

    def test_outside_window_skips_without_push(self):
        for moment in (datetime(2026, 10, 8, 14, 29, 59), datetime(2026, 10, 8, 14, 40, 0),
                       datetime(2026, 10, 8, 14, 50, 0)):
            with self.subTest(moment=moment):
                self.assertEqual(daily_preview.main(now=moment, send=self.send), 0)
        self.assertEqual(self.sent, [])

    def test_saved_bundle_covering_today_skips_push(self):
        self.live.return_value = None
        self.assertEqual(daily_preview.main(now=NOW, send=self.send), 0)
        self.assertEqual(self.sent, [])
        self.assertFalse((self.signals / "preview_status.json").exists())

    def test_send_failure_records_failed_and_returns_one(self):
        self.assertEqual(daily_preview.main(now=NOW, send=lambda text, title=None: False), 1)
        self.assertEqual(self.status()["status"], "failed")

    def test_previous_preview_still_running_skips(self):
        with signal_store.file_lock(self.signals / "preview.lock"):
            self.assertEqual(daily_preview.main(now=NOW, send=self.send), 0)
        self.assertEqual(self.sent, [])

    def test_payload_failure_is_contained_and_never_blocks_the_real_chain(self):
        self.live.side_effect = RuntimeError("mock quote outage")
        self.assertEqual(daily_preview.main(now=NOW, send=self.send), 1)
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
