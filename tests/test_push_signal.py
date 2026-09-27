import unittest

from push_signal import build_markdown, is_current_signal


class PushSignalTests(unittest.TestCase):
    def test_v10_saved_preview_keeps_date_status_and_bold_target_in_both_modes(self):
        signal = "\n".join([
            "动量轮动信号 | 生成 2026-09-25 14:50 | 数据截止 2026-09-24",
            "【影子 V10-H】研究观察 · 虚拟跟踪不下单",
            "信号状态: 历史预览；前向跟踪尚未启动",
            "数据截止: 2026-09-24",
            "影子持仓: 未启动 | 建议: 513100 纳指ETF | 冻结研究历史收盘目标，非今日买入指令",
            "策略规则: WLS20评分取3日均值 / MA180 / 波动急跌退出",
        ])
        for query in (False, True):
            card, _ = build_markdown(signal, query=query)
            self.assertIn("**影子 V10-H**", card)
            self.assertIn("影子建议标的: **513100 纳指ETF**", card)
            self.assertIn("历史预览；前向跟踪尚未启动", card)
            self.assertIn("非今日买入指令", card)
            self.assertIn("数据截止: 2026-09-24", card)
            self.assertIn("MA180", card)
        self.assertFalse(is_current_signal(signal, "2026-09-27"))

    def test_v10_failure_is_visible_without_inventing_a_target(self):
        signal = "【影子 V10-H】\n信号状态: 计算失败，本次无有效建议\n数据说明: 缺当日报价"
        card, _ = build_markdown(signal, query=True)
        self.assertIn("本次无有效建议", card)
        self.assertIn("缺当日报价", card)
        self.assertNotIn("影子建议标的:", card)

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

    def test_saved_signal_query_shares_targets_but_keeps_its_historical_context(self):
        signal = "\n".join([
            "动量轮动信号 | 生成 2026-09-24 14:50 | 数据截止 2026-09-24",
            "当前持仓: 518880 黄金ETF",
            "★ 建议: 卖出 518880 黄金ETF -> 买入 513100 纳指ETF",
            "  依据: 示例轮动条件",
            "【影子 v9.2】虚拟跟踪不下单",
            "影子持仓: 513120 | 虚拟净值 1.0 | 建议: 513120 港股创新药ETF | 示例原因",
        ])
        daily, daily_date = build_markdown(signal)
        query, query_date = build_markdown(signal, query=True)
        for reply in (daily, query):
            self.assertIn("卖出 518880 黄金ETF -> 买入 **513100 纳指ETF**", reply)
            self.assertIn("影子建议标的: **513120 港股创新药ETF**", reply)
            self.assertIn("虚拟跟踪不下单", reply)
        self.assertEqual((daily_date, query_date), ("2026-09-24", "2026-09-24"))
        self.assertIn("最近保存信号", query)
        self.assertIn("以每日 14:50 正式推送为准", query)
        self.assertIn("动量跟踪网页", query)
        self.assertNotIn("今日操作", query)
        self.assertNotIn("操作后回报", query)
        self.assertIn("今日操作", daily)
        self.assertIn("操作后回报", daily)
