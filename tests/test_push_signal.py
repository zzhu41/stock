import unittest

from push_signal import build_markdown, is_current_signal


class PushSignalTests(unittest.TestCase):
    def test_old_or_misaligned_signal_is_not_current(self):
        header = "动量轮动信号 | 生成 2026-09-24 14:50 | 数据截止 "
        self.assertTrue(is_current_signal(header + "2026-09-24", "2026-09-24"))
        self.assertFalse(is_current_signal(header + "2026-09-23", "2026-09-24"))
        self.assertFalse(is_current_signal(header + "2026-09-24", "2026-09-25"))
        self.assertFalse(is_current_signal("missing header", "2026-09-24"))

    def test_forward_nav_migration_notice_is_visible_in_card(self):
        note = "⚠️ 旧口径净值 1.1 已归档，2026-09-24 起重新从1记账"
        card, _ = build_markdown("【影子 v9.2】\n影子持仓: 513100\n" + note)
        self.assertIn(note, card)
