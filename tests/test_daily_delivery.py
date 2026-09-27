"""Failure-path tests use temporary journals and fake HTTP, never group sends."""
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch, Mock
from contextlib import redirect_stdout

import daily_job
import market_data
import push_signal
import signal_store as store

NOW = datetime(2026, 9, 28, 14, 50, 30)
DATE = "2026-09-28"
TEXT = "\n".join([
    "动量轮动信号 | 生成 2026-09-28 14:50:10 | 数据截止 2026-09-28",
    "行情时间: 2026-09-28 14:50:00",
    "当前持仓: 空仓",
    "★ 建议: 买入 513120 港股创新药ETF",
    "依据: 测试目标",
    "【影子 V9.2】虚拟跟踪不下单",
    "数据截止: 2026-09-28",
    "影子持仓: 513120 | 虚拟净值 1.0 | 建议: 买入 513120 港股创新药ETF | 测试目标",
    "513100 纳指ETF 溢价 +0.10%",
    "513120 港股创新药ETF 溢价 +8.00% ⚠️ >2% 勿追!",
])


class Response(io.BytesIO):
    def __init__(self, code):
        super().__init__(json.dumps({"errcode": code}).encode())


class DailyDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def publish(self):
        return store.publish(TEXT, DATE, "513120", {"code": "513120", "trigger_date": DATE}, self.directory)[0]

    def test_primary_same_day_morning_future_and_cross_asset_stale_quotes_rejected(self):
        histories = {"510300": [("2026-09-24", 10, 10, 10)]}
        for stamp in (DATE + " 09:31:00", DATE + " 14:51:00"):
            quote = dict(date=DATE, timestamp=stamp, price=10, open=10, volume=10)
            with self.assertRaises(ValueError):
                market_data.prepare_live_histories(histories, {"510300": quote}, DATE, now=NOW)
        quotes = {code: dict(date=DATE, timestamp=DATE + stamp, price=10, open=10, volume=10)
                  for code, stamp in (("510300", " 14:50:00"), ("510500", " 14:48:59"))}
        histories["510500"] = histories["510300"]
        with self.assertRaisesRegex(ValueError, "时间差"):
            market_data.prepare_live_histories(histories, quotes, DATE, now=NOW)

    def test_journal_failure_never_creates_or_changes_primary_crash_lock(self):
        legacy = self.directory / "crash_lock.json"
        legacy.write_text("null")
        with patch.object(store, "atomic_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual(legacy.read_text(), "null")
        self.assertFalse((self.directory / "daily_state.json").exists())

    def test_projection_failure_preserves_recoverable_signal_and_lock_together(self):
        real_write = store.atomic_text
        def write(path, content):
            if str(path).endswith("latest.txt"):
                raise OSError("text mirror unavailable")
            real_write(path, content)
        with patch.object(store, "atomic_text", side_effect=write):
            record, errors = store.publish(TEXT, DATE, "513120", {"code": "513120"}, self.directory)
        self.assertTrue(errors)
        loaded, journal = store.load_saved(self.directory)
        self.assertEqual(loaded, TEXT)
        self.assertEqual(journal["crash_lock"], {"code": "513120"})
        self.assertEqual(store.repair_projections(record, self.directory), [])
        self.assertEqual((self.directory / "latest.txt").read_text(), TEXT)
        journal["crash_lock"]["code"] = "510300"
        (self.directory / "daily_state.json").write_text(json.dumps(journal))
        with self.assertRaises(ValueError):
            store.load_saved(self.directory)

    def test_old_misaligned_missing_quote_and_after_window_cards_never_show_buy_instructions(self):
        variants = [(TEXT, datetime(2026, 9, 29, 14, 50)),
                    (TEXT.replace("数据截止 2026-09-28", "数据截止 2026-09-24"), NOW),
                    (TEXT.replace("行情时间:", "缺失时间:"), NOW),
                    (TEXT, datetime(2026, 9, 28, 15, 1)), ("", NOW)]
        for text, now in variants:
            with self.subTest(now=now, text=text[:30]):
                card, date = push_signal.build_markdown(text, query=True, now=now)
                self.assertNotIn("买入 **", card)
                self.assertNotIn("## 今日操作", card)
                self.assertIn("⚠️", card)

    def test_both_premiums_and_missing_premium_notice_are_visible(self):
        card, _ = push_signal.build_markdown(TEXT + "\nQDII 溢价: 获取失败", now=NOW)
        self.assertIn("买入 **513120 港股创新药ETF**", card)
        self.assertIn("8.00%", card)
        self.assertIn("0.10%", card)
        self.assertIn("QDII 溢价: 获取失败", card)

    def test_three_scheduled_attempts_retry_later_and_confirmed_send_is_deduplicated(self):
        record = self.publish()
        with patch.object(push_signal, "load_file", side_effect=lambda p: "https://example.invalid/send" if p == push_signal.WEBHOOK_FILE else ""), \
                patch.object(push_signal.urllib.request, "urlopen", side_effect=[Response(130101), Response(0)]) as send, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(push_signal.main(str(self.directory), now=NOW), 1)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(push_signal.main(str(self.directory), now=datetime(2026, 9, 28, 14, 52)), 0)
            self.assertEqual(send.call_count, 2)
            self.assertEqual(push_signal.main(str(self.directory), now=datetime(2026, 9, 28, 14, 54)), 0)
            self.assertEqual(send.call_count, 2)
            second_body = json.loads(send.call_args_list[-1].args[0].data)
            self.assertIn("请勿重复下单", second_body["markdown"]["text"])
        receipt = store.read_json(self.directory / "delivery.json")
        self.assertEqual(receipt["status"], "sent")
        self.assertEqual(receipt["signal_id"], record["signal_id"])
        self.assertEqual(len(receipt["attempts"]), 2)

    def test_timeout_is_recorded_uncertain_and_visible_in_readonly_query(self):
        self.publish()
        with patch.object(push_signal, "load_file", return_value="https://example.invalid"), \
                patch.object(push_signal.urllib.request, "urlopen", side_effect=TimeoutError("private URL should never be logged")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(push_signal.main(str(self.directory), now=NOW), 1)
        self.assertNotIn("private URL", output.getvalue())
        receipt_path = self.directory / "delivery.json"
        before = receipt_path.read_bytes()
        self.assertEqual(json.loads(before)["status"], "uncertain")
        card = push_signal.render_saved_query(str(self.directory / "latest.txt"), now=NOW)
        self.assertIn("尚未确认送达", card)
        self.assertEqual(before, receipt_path.read_bytes())

    def test_expired_outbound_never_makes_http_request(self):
        self.publish()
        with patch.object(push_signal, "load_file", return_value="unused"), \
                patch.object(push_signal.urllib.request, "urlopen") as send, redirect_stdout(io.StringIO()):
            self.assertEqual(push_signal.main(str(self.directory), now=datetime(2026, 9, 28, 15)), 1)
        send.assert_not_called()

    def test_missing_webhook_is_a_durable_failure_without_consuming_network_attempt(self):
        self.publish()
        with patch.object(push_signal, "load_file", return_value=""), \
                patch.object(push_signal.urllib.request, "urlopen") as send, redirect_stdout(io.StringIO()):
            self.assertEqual(push_signal.main(str(self.directory), now=NOW), 1)
        send.assert_not_called()
        receipt = store.read_json(self.directory / "delivery.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["attempts"], [])

    def test_job_failed_generation_is_visible_and_does_not_send(self):
        (self.directory / "latest.txt").write_text(TEXT.replace("2026-09-28", "2026-09-24"))
        with redirect_stdout(io.StringIO()):
            result = daily_job.run_job(self.directory, now=NOW,
                generate=lambda: (_ for _ in ()).throw(TimeoutError()),
                deliver=lambda: self.fail("must not send"))
        self.assertEqual(result, 1)
        card = push_signal.render_saved_query(str(self.directory / "latest.txt"), now=NOW)
        self.assertIn("今日信号生成失败", card)
        self.assertNotIn("买入 **", card)

    def test_last_seconds_before_deadline_cannot_start_new_send(self):
        self.publish()
        with patch.object(push_signal, "load_file", return_value="unused"), \
                patch.object(push_signal.urllib.request, "urlopen") as send, redirect_stdout(io.StringIO()):
            self.assertEqual(push_signal.main(str(self.directory), now=datetime(2026, 9, 28, 14, 54, 59)), 1)
        send.assert_not_called()

    def test_job_recovers_committed_child_and_does_not_regenerate_same_day(self):
        def generate():
            self.publish()
            raise TimeoutError("committed but lost reply")
        self.assertEqual(daily_job.run_job(self.directory, now=NOW, generate=generate, deliver=lambda: 0), 0)
        self.assertEqual(daily_job.run_job(self.directory, now=NOW,
            generate=lambda: self.fail("duplicate generation"), deliver=lambda: 0), 0)

    def test_job_delivery_hard_timeout_preserves_unknown_ack_and_later_recovery(self):
        record = self.publish()
        receipt = dict(date=DATE, signal_id=record["signal_id"], status="sending",
                       attempts=[dict(status="sending")])
        store.atomic_json(self.directory / "delivery.json", receipt)
        with patch.object(daily_job, "run_child", side_effect=TimeoutError()) as child, redirect_stdout(io.StringIO()):
            self.assertEqual(daily_job.run_job(self.directory, now=NOW), 1)
        self.assertEqual(child.call_args.args[1], 10.)
        self.assertEqual(store.read_json(self.directory / "delivery.json")["status"], "uncertain")
        receipt.update(status="sent")
        receipt["attempts"] = [dict(status="sent", errcode=0)]
        store.atomic_json(self.directory / "delivery.json", receipt)
        with patch.object(daily_job, "run_child", side_effect=TimeoutError()), redirect_stdout(io.StringIO()):
            self.assertEqual(daily_job.run_job(self.directory, now=NOW), 0)

    def test_process_wall_deadline_terminates_the_entire_group(self):
        child = Mock(pid=123456, returncode=0)
        child.communicate.side_effect = [subprocess.TimeoutExpired("synthetic", 10), ("", "")]
        with patch.object(daily_job.subprocess, "Popen", return_value=child) as spawn, \
                patch.object(daily_job.os, "killpg") as kill:
            with self.assertRaises(TimeoutError):
                daily_job.run_child(["synthetic-worker"], 10)
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(123456, daily_job.signal.SIGKILL)
        self.assertEqual(child.communicate.call_count, 2)

    def test_published_holidays_skip_but_unknown_year_is_not_assumed_closed(self):
        self.assertFalse(store.trading_day("2026-09-25"))
        self.assertTrue(store.trading_day("2026-09-28"))
        self.assertIsNone(store.trading_day("2027-01-04"))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(daily_job.run_job(self.directory, now=datetime(2026, 9, 25, 14, 50),
                generate=lambda: self.fail("holiday generation"), deliver=lambda: self.fail("holiday send")), 0)
