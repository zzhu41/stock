"""Temporary monitoring fixtures; sent comparison cards cannot hide a failed primary."""
from contextlib import ExitStack
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import watchdog


class PrimaryWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(watchdog.store, "trading_day", return_value=True))
        self.stack.enter_context(patch.object(watchdog, "_dual_source",
            return_value=dict(status="OK", external_has_today=True)))
        self.stack.enter_context(patch.object(watchdog, "_qvix_fresh", return_value=True))
        self.finish = self.stack.enter_context(patch.object(watchdog, "_finish"))
        self.stack.enter_context(patch.object(watchdog.live_utils, "send_dingtalk",
            side_effect=AssertionError("Test cannot send an alert")))
        self.stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))

    def save(self, day="2026-09-29", legacy=False, primary=True, stale_checkpoint=False,
             quote="14:50:00", failed_card=False, omit_card=False, sent=True, card_target="518880"):
        header = "V9.2 | V9.2+ | V10-H" if legacy else "V12-R2 | V9.2 | V9.2+ | V10-H"
        lines = ["动量轮动信号 | 生成 %s 14:50:30 | 数据截止 %s" % (day, day),
                 "行情时间: %s 14:50:00" % day, "策略版本: " + header]
        if not legacy and not omit_card:
            lines += ["【影子 V12-R2】",
                      "信号状态: 计算失败，本次无有效建议" if failed_card else "信号状态: 盘中影子试算"]
            if card_target is not None:
                lines += ["数据截止: "+day+" | 行情时间: "+day+" "+quote,
                          "影子持仓: "+card_target+" | 建议: "+card_target+" syntheticETF"]
        lines += ["【影子 V9.2】", "影子持仓: 513100 | 建议: 513100 纳指ETF"]
        accounts = {"shadow_v92.json": dict(last_date=day, holding="513100")}
        updated = ["shadow_v92.json"]
        target = "513100" if legacy else None
        if primary or stale_checkpoint:
            accounts[watchdog.store.PRIMARY_ACCOUNT] = dict(
                last_date="2026-09-28" if stale_checkpoint else day, holding="518880",
                quote_timestamp=day+" "+quote)
        if primary and not stale_checkpoint:
            updated.append(watchdog.store.PRIMARY_ACCOUNT)
            if not legacy: target = "518880"
        text = "\n".join(lines)
        record = dict(schema=2, date=day, signal_id=watchdog.store.digest(text), text=text,
                      committed_at=day+"T14:50:30", target=target, crash_lock=None,
                      account_states=accounts, updated_accounts=updated)
        if not legacy: record["primary_version"] = watchdog.store.PRIMARY_VERSION
        record["checksum"] = watchdog.store.record_digest(record)
        for name, value in (
            ("daily_state.json", record),
            ("generation_status.json", dict(date=day,status="ready",signal_id=record["signal_id"])),
            ("delivery.json", dict(date=day,status="sent" if sent else "uncertain",signal_id=record["signal_id"],
                                  attempts=[dict(status="sent",errcode=0)] if sent else []))):
            (self.root/name).write_text(json.dumps(value))
        return record

    def inspect(self, day="2026-09-29"):
        return watchdog.main(directory=self.root,now=datetime.strptime(day+" 15:20:00","%Y-%m-%d %H:%M:%S"))

    def test_sent_comparisons_with_failed_main_are_degraded(self):
        self.save(primary=False,failed_card=True)
        result=self.inspect()
        self.assertEqual(result["status"],"DEGRADED")
        self.assertTrue(any("主推送未生成" in item for item in result["alerts"]))

    def test_old_primary_checkpoint_is_not_todays_primary(self):
        self.save(primary=False,stale_checkpoint=True,failed_card=True)
        self.assertEqual(self.inspect()["status"],"DEGRADED")

    def test_successful_new_main_and_independent_quote_can_be_ok(self):
        self.save()
        self.assertEqual(self.inspect()["status"],"OK")

    def test_primary_quote_and_card_must_be_usable_independently_of_bundle(self):
        for change in (dict(quote="14:45:00"),dict(failed_card=True),dict(omit_card=True),
                       dict(card_target=None),dict(card_target="513100")):
            self.save(**change)
            result=self.inspect()
            self.assertEqual(result["status"],"DEGRADED")
            self.assertTrue(any("主推送" in item for item in result["alerts"]))

    def test_legacy_record_before_start_keeps_old_semantics_but_after_start_is_degraded(self):
        self.save(day="2026-09-28",legacy=True,primary=False)
        self.assertEqual(self.inspect("2026-09-28")["status"],"OK")
        self.save(legacy=True,primary=False)
        result=self.inspect()
        self.assertEqual(result["status"],"DEGRADED")
        self.assertTrue(any("旧部署版本" in item for item in result["alerts"]))

    def test_delivery_failure_remains_failed_not_masked_by_primary_degradation(self):
        self.save(primary=False,failed_card=True,sent=False)
        self.assertEqual(self.inspect()["status"],"FAILED")


if __name__=="__main__":unittest.main()
