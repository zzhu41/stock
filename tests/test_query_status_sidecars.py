"""Query status degradation is read-only; canonical integrity stays mandatory."""
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess
from unittest.mock import patch

import push_signal
import signal_store

NOW = datetime(2026, 9, 28, 14, 50, 30)
DATE = "2026-09-28"
TEXT = "\n".join([
    "动量轮动信号 | 生成 2026-09-28 14:50:10 | 数据截止 2026-09-28",
    "行情时间: 2026-09-28 14:50:00",
    "策略版本: V9.2 | V9.2+ | V10-H",
    "【影子 V9.2】",
    "数据截止: 2026-09-28",
    "影子持仓: 513100 | 虚拟净值 1.0 | 建议: 买入 513100 纳指ETF | synthetic",
])


class QuerySidecarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.journal = signal_store.publish(TEXT, DATE, "513100", None, self.root)[0]

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def query(self, now=NOW):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with ExitStack() as stack:
            for name in ("atomic_json", "atomic_text", "publish", "repair_projections"):
                stack.enter_context(patch.object(push_signal.store, name, side_effect=AssertionError("Query cannot write")))
            stack.enter_context(patch.object(push_signal.urllib.request, "urlopen", side_effect=AssertionError("Query cannot send")))
            card = push_signal.render_saved_query(str(self.root / "latest.txt"), now=now)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        return card

    def test_broken_delivery_json_warns_without_hiding_current_verified_signal(self):
        (self.root / "delivery.json").write_text("{broken")
        card = self.query()
        self.assertIn("推送回执记录损坏或无法读取", card)
        self.assertIn("无法确认自动推送是否送达", card)
        self.assertIn("影子建议标的: 买入 **513100 纳指ETF**", card)
        self.assertNotIn("信号记录缺失或校验失败", card)

    def test_bad_generation_does_not_hide_good_delivery_or_signal(self):
        (self.root / "generation_status.json").write_text("{broken")
        self.write("delivery.json", dict(signal_id=self.journal["signal_id"], status="sent"))
        card = self.query()
        self.assertIn("生成状态记录损坏或无法读取", card)
        self.assertIn("影子建议标的:", card)
        self.assertNotIn("推送回执记录损坏", card)
        self.assertNotIn("尚未确认送达", card)

    def test_two_sidecar_failures_produce_independent_warnings(self):
        self.write("generation_status.json", ["wrong", "shape"])
        self.write("delivery.json", dict(signal_id=self.journal["signal_id"], status="sent", attempts="broken"))
        card = self.query()
        self.assertIn("生成状态记录损坏或无法读取", card)
        self.assertIn("推送回执记录损坏或无法读取", card)
        self.assertIn("无法确认自动推送是否送达", card)
        self.assertIn("影子建议标的:", card)

    def test_generation_failure_notice_survives_broken_delivery(self):
        self.write("generation_status.json", dict(date=DATE, status="failed"))
        (self.root / "delivery.json").write_bytes(b"\xff")
        card = self.query()
        self.assertIn("今日信号生成失败", card)
        self.assertIn("推送回执记录损坏或无法读取", card)

    def test_good_uncertain_receipt_survives_generation_read_error_without_secret_leak(self):
        self.write("delivery.json", dict(signal_id=self.journal["signal_id"], status="uncertain"))
        original = push_signal.store.read_json
        def fail(path, default=None):
            if Path(path).name == "generation_status.json":
                raise PermissionError("private URL/access_token must never be shown")
            return original(path, default)
        with patch.object(push_signal.store, "read_json", side_effect=fail):
            card = self.query()
        self.assertIn("生成状态记录损坏或无法读取", card)
        self.assertIn("尚未确认送达（uncertain）", card)
        self.assertNotIn("access_token", card)

    def test_missing_sidecars_remain_unknown_and_do_not_invent_corruption(self):
        card = self.query()
        self.assertIn("尚无自动推送成功记录", card)
        self.assertNotIn("记录损坏或无法读取", card)
        self.assertIn("影子建议标的:", card)

    def test_bad_sidecar_cannot_make_stale_saved_target_actionable(self):
        (self.root / "delivery.json").write_text("{broken")
        for now in (NOW.replace(hour=15), NOW.replace(day=29)):
            card = self.query(now)
            self.assertIn("历史影子目标:", card)
            self.assertNotIn("影子建议标的:", card)
            self.assertNotIn("买入 **", card)
            self.assertIn("推送回执记录损坏或无法读取", card)

    def test_corrupt_canonical_still_fails_closed_even_with_good_latest_and_sent_receipt(self):
        self.write("delivery.json", dict(signal_id=self.journal["signal_id"], status="sent"))
        for damaged in ("{broken", json.dumps(dict(self.journal, target="changed without checksum"))):
            (self.root / "daily_state.json").write_text(damaged)
            card = self.query()
            self.assertIn("信号记录缺失或校验失败", card)
            self.assertNotIn("513100", card)
            self.assertNotIn("影子建议标的", card)

    def test_missing_canonical_still_shows_one_sanitized_primary_failure_without_instructions(self):
        (self.root / "daily_state.json").unlink()
        (self.root / "latest.txt").unlink()
        self.write("generation_status.json",dict(date=DATE,status="failed",diagnostic={
            "phase":"prepare_versions","error_type":"PreparationError","message":"三个版本均失败",
            "causes":[{"phase":"input.premium","message":"optional pricing failed"},
                      {"phase":"input.view","message":"510300 当前配对缺失 https://example.invalid/?secret=private-secret token=private-value",
                       "frames":[{"file":"private_implementation.py","line":300,"function":"internal_secret"}]}]}))
        card=self.query()
        self.assertIn("信号记录缺失或校验失败",card)
        self.assertIn("生成诊断 [input.view]: 510300 当前配对缺失",card)
        self.assertEqual(card.count("生成诊断"),1)
        self.assertNotIn("optional pricing",card)
        for forbidden in ("private-secret","private-value","private_implementation","internal_secret","买入 **","影子建议标的"):
            self.assertNotIn(forbidden,card)

    def test_old_generation_diagnostic_is_not_presented_as_current_failure(self):
        self.write("generation_status.json",dict(date="2026-09-24",status="failed",
            diagnostic=dict(phase="input.view",message="old historical failure")))
        card=self.query()
        self.assertNotIn("old historical failure",card)
        self.assertNotIn("生成诊断",card)
        self.assertIn("影子建议标的:",card)

    def test_python36_absolute_import_from_foreign_directory_when_available(self):
        binary=shutil.which("python3.6")
        if not binary:self.skipTest("Python3.6 unavailable")
        script=("import importlib.util\\n"
                "spec=importlib.util.spec_from_file_location('formatter',%r)\\n"
                "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\\n"
                "from datetime import datetime\\n"
                "text=module.render_saved_query(path=%r,now=datetime(2026,9,28,14,50,30))\\n"
                "assert '513100' in text and '影子建议标的:' in text\\n") % (
                    str(Path(push_signal.__file__).resolve()),str(self.root/"latest.txt"))
        # Actual newline source, not literal escaped Python tokens.
        script=script.replace("\\n","\n")
        done=subprocess.run([binary,"-B","-c",script],cwd=str(self.root),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
        self.assertEqual(done.returncode,0,done.stderr)


if __name__ == "__main__": unittest.main()
