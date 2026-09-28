# -*- coding: utf-8 -*-
"""momentum_live rendering/decision-table unit tests; no network, no bot app."""
import json
import unittest
from unittest.mock import patch

import momentum_live


class RowTests(unittest.TestCase):
    def test_holding_same_as_previous_shows_hold(self):
        new = dict(holding="513100", last_date="2026-09-28", start_date="2026-09-04",
                   last_advice="513100 纳指ETF | 避险池最强接盘(永不空仓)")
        label, action, target, reason = momentum_live._row("V9.2", new, dict(holding="513100"))
        self.assertEqual(action, "持有")
        self.assertEqual(target, "513100 纳指ETF")
        self.assertIn("避险池", reason)

    def test_switch_shows_sell_leg(self):
        new = dict(holding="518880", last_date="2026-09-28", start_date="2026-09-01",
                   last_decision={"reason": "黄金score居首", "previous_holding": "513100"})
        label, action, target, reason = momentum_live._row("V12-R2(主)", new, dict(holding="513100"))
        self.assertEqual(action, "换仓 卖纳指ETF")
        self.assertEqual(target, "518880 黄金ETF")
        self.assertEqual(reason, "黄金score居首")

    def test_first_day_state_shows_entry(self):
        new = dict(holding="513100", last_date="2026-09-29", start_date="2026-09-29",
                   last_decision={"reason": "首次建仓"})
        label, action, target, reason = momentum_live._row("V10-H", new, None)
        self.assertEqual(action, "建仓")

    def test_missing_state_marks_unavailable(self):
        label, action, target, reason = momentum_live._row("V12-R2(主)", None)
        self.assertIsNone(action)
        self.assertEqual(reason, "等待首次有效信号")

    def test_pipe_and_newline_never_break_table_cells(self):
        new = dict(holding="513100", last_date="2026-09-28", start_date="2026-09-04",
                   last_advice="513100 | 原因含|竖线\n和换行")
        _label, _action, _target, reason = momentum_live._row("V9.2", new, dict(holding="513100"))
        self.assertNotIn("|", reason)
        self.assertNotIn("\n", reason)


class RenderTests(unittest.TestCase):
    ROWS = [("V12-R2(主)", "持有", "513100 纳指ETF", "WLS25居首"),
            ("V9.2", None, None, "等待首次有效信号")]

    def test_live_table_marks_estimate_and_never_commits(self):
        md = momentum_live.render_table(self.ROWS, "live", "2026-09-29", "行情 2026-09-29 10:30:00")
        self.assertIn("实时估算", md)
        self.assertIn("| 版本 | 建议 | 买什么 | 原因 |", md)
        self.assertIn("| V12-R2(主) | 持有 | 513100 纳指ETF | WLS25居首 |", md)
        self.assertIn("| V9.2 | — | — | 等待首次有效信号 |", md)
        self.assertIn("不写入任何账户", md)
        self.assertIn("14:50", md)

    def test_saved_table_has_no_estimate_warning(self):
        md = momentum_live.render_table(self.ROWS, "saved", "2026-09-24", "14:50收盘信号")
        self.assertIn("最新信号", md)
        self.assertNotIn("实时估算，不写入", md)


class PremiumBriefTests(unittest.TestCase):
    def test_extracts_code_and_rate_with_warning_flag(self):
        lines = ["  513100 纳指ETF 溢价 +12.85% (现价 2.312 / 净值 2.0487@2026-09-23) ⚠️ >2% 勿追!",
                 "  513120 港股创新药ETF 溢价 -1.28% (现价 1.314 / 净值 1.3310@2026-09-24)",
                 "  无关行"]
        brief = momentum_live._premium_brief(lines)
        self.assertEqual(brief, "QDII 溢价: 513100 +12.85%⚠️ / 513120 -1.28%")

    def test_no_premium_lines_returns_none(self):
        self.assertIsNone(momentum_live._premium_brief(["没有溢价信息"]))
        self.assertIsNone(momentum_live._premium_brief(None))


class SavedDateTests(unittest.TestCase):
    def test_journal_date_wins(self):
        journal = {"date": "2026-09-29", "text": "数据截止 2026-09-24"}
        self.assertEqual(momentum_live._saved_date(journal), "2026-09-29")

    def test_legacy_latest_txt_header_fallback(self):
        with patch.object(momentum_live, "_saved_text",
                          return_value="动量轮动信号 | 生成 2026-09-25 14:50 | 数据截止 2026-09-24\n"):
            self.assertEqual(momentum_live._saved_date(None), "2026-09-24")

    def test_unparseable_returns_none(self):
        with patch.object(momentum_live, "_saved_text", return_value=""):
            self.assertIsNone(momentum_live._saved_date(None))


class LiveFreshnessGateTests(unittest.TestCase):
    """live_payload 的入口判定：非交易日/已有当日信号时不重算。报价拉取被替身。"""

    class _FakeDateTime:
        pass

    def _quotes(self, date):
        return {code: {"name": code, "price": 1.0, "date": date,
                       "timestamp": date + " 10:30:00", "prev_close": 1.0,
                       "open": 1.0, "volume": 100} for code in momentum_live.UNIVERSE}

    def test_non_trading_day_returns_none_without_collect(self):
        with patch.object(momentum_live, "fetch_realtime", return_value=self._quotes("2026-09-27")), \
             patch.object(momentum_live.signal_store, "trading_day", return_value=True):
            import datetime
            now = datetime.datetime(2026, 9, 28, 10, 30)  # 报价日期≠今天
            self.assertIsNone(momentum_live.live_payload(None, now))

    def test_same_day_saved_bundle_returns_none(self):
        with patch.object(momentum_live, "fetch_realtime", return_value=self._quotes("2026-09-28")), \
             patch.object(momentum_live.signal_store, "trading_day", return_value=True), \
             patch.object(momentum_live, "_saved_date", return_value="2026-09-28"):
            import datetime
            now = datetime.datetime(2026, 9, 28, 10, 30)
            self.assertIsNone(momentum_live.live_payload({"date": "2026-09-28"}, now))


if __name__ == "__main__":
    unittest.main()
