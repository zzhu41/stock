"""牢国策略每日推送: 消息格式/幂等/引擎指纹/失败路径测试。全程不触网。"""
import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import laoguo_data
import laoguo_signal

ENGINE_EXISTS = os.path.exists(laoguo_signal.ENGINE_PATH)


def make_result(action_type="无", position="创业", target=None, old=None):
    return {
        "position": position,
        "entry_date": "2026-10-14",
        "entry_reason": "恐慌抄底",
        "momentums": {"纳指": 1.33, "黄金": 10.83, "创业": -15.56},
        "sh_momentum": -3.87,
        "action": {"type": action_type, "target": target, "old": old, "reason": "测试原因"},
    }


class MessageFormatTests(unittest.TestCase):
    """消息格式: 双消息、牢国标记、五种动作映射、代码显示。"""

    def build(self, **kw):
        return laoguo_signal.build_messages(make_result(**kw), ["纳指", "黄金", "创业"],
                                            "2026-10-20", "14:49 盘中价")

    def test_full_message_has_brand_rank_table_and_position(self):
        full, _big = self.build()
        self.assertIn("牢国策略", full)
        self.assertIn("🥇", full)
        self.assertIn("+10.83%", full)
        self.assertIn("← 持有", full)
        self.assertIn("自2026-10-14", full)

    def test_big_message_buy_sell_switch_hold_cash(self):
        self.assertEqual(self.build(action_type="买入", target="纳指")[1],
                         "# 🟧 买入 513100 纳指")
        self.assertEqual(self.build(action_type="换仓", target="黄金")[1],
                         "# 🔄 换仓 → 518880 黄金")
        self.assertEqual(self.build(action_type="卖出", old="创业")[1],
                         "# ⏸ 卖出 159915 创业, 空仓等待")
        self.assertEqual(self.build(position="创业")[1], "# 🟦 持有 159915 创业")
        self.assertEqual(self.build(position="")[1], "# ⏸ 空仓等待")

    def test_full_message_action_lines(self):
        self.assertIn("**买入 纳指**", self.build(action_type="买入", target="纳指")[0])
        self.assertIn("**卖出 创业 → 买入 黄金**",
                      self.build(action_type="换仓", target="黄金", old="创业")[0])
        self.assertIn("**卖出 创业 → 空仓**", self.build(action_type="卖出", old="创业")[0])
        self.assertIn("继续持有", self.build()[0])
        self.assertIn("继续空仓等待", self.build(position="")[0])


class SendBothTests(unittest.TestCase):
    """发送: 两群×两条、先完整版后大字版、间隔5秒、按群认证。"""

    def test_order_gap_and_per_group_auth(self):
        calls = []

        class Resp:
            def read(self):
                return json.dumps({"errcode": 0}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        targets = [("https://example.invalid/send?token=primary", "SEC-x"),
                   ("https://example.invalid/send?token=extra", None)]
        sleeps = []
        with patch.object(laoguo_signal.push_signal, "load_targets", return_value=targets), \
                patch.object(laoguo_signal.push_signal, "signed_url",
                             side_effect=lambda w, s: w + "&signed"), \
                patch.object(laoguo_signal.urllib.request, "urlopen",
                             side_effect=lambda req, timeout=None: calls.append(req) or Resp()):
            results = laoguo_signal.send_both("完整版内容", "# 大字版内容", "2026-10-20",
                                              sleeper=sleeps.append)
        self.assertEqual(sleeps, [5, 5])
        self.assertEqual(len(calls), 4)
        bodies = [json.loads(c.data) for c in calls]
        # 每群内: 先完整版后大字版; 标题均带【牢国】
        self.assertEqual(bodies[0]["markdown"]["text"], "完整版内容")
        self.assertEqual(bodies[1]["markdown"]["text"], "# 大字版内容")
        self.assertTrue(all("【牢国】" in b["markdown"]["title"] for b in bodies))
        # 关键词群(无 secret)兜底关键词, 加签群不动
        self.assertEqual(bodies[0]["markdown"]["text"], "完整版内容")  # primary 有 secret
        self.assertIn("股票", bodies[2]["markdown"]["text"])  # extra 无 secret
        self.assertTrue(all(code == 0 for *_x, code in results))

    def test_partial_failure_reported_per_group(self):
        class Resp:
            def __init__(self, code):
                self.code = code

            def read(self):
                return json.dumps({"errcode": self.code}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        seq = [Resp(0), Resp(0), Resp(310000), Resp(0)]
        targets = [("https://a.invalid", "SEC-x"), ("https://b.invalid", None)]
        with patch.object(laoguo_signal.push_signal, "load_targets", return_value=targets), \
                patch.object(laoguo_signal.push_signal, "signed_url", side_effect=lambda w, s: w), \
                patch.object(laoguo_signal.urllib.request, "urlopen",
                             side_effect=lambda req, timeout=None: seq.pop(0)):
            results = laoguo_signal.send_both("f", "b", "2026-10-20", sleeper=lambda s: None)
        self.assertEqual([code for _g, _p, code in results], [0, 0, 310000, 0])


@unittest.skipUnless(ENGINE_EXISTS, "引擎文件不存在")
class RunSignalTests(unittest.TestCase):
    """端到端(离线数据): 幂等、引擎指纹漂移拒绝、失败落状态。"""

    QUOTES = {"纳指": 2.340, "黄金": 9.836, "创业": 2.79, "上证": 3735.0, "创业指": 2750.0}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = os.path.join(self.temp.name, "state.json")

    def run_once(self, sender=None):
        return laoguo_signal.run_signal(
            today="2026-10-20", offline=True, live_quote=dict(self.QUOTES),
            sender=sender or (lambda f, b, d, sleeper=None: [(1, 1, 0), (1, 2, 0)]),
            sleeper=lambda s: None, state_file=self.state)

    def test_same_day_same_signal_is_deduplicated(self):
        sent = []
        ok, _ = self.run_once(sender=lambda f, b, d, sleeper=None: sent.append(1) or [(1, 1, 0)])
        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)
        ok, summary = self.run_once()
        self.assertIn("跳过重复发送", summary)
        self.assertEqual(len(sent), 1)  # 第二次未发
        state = json.load(open(self.state))
        self.assertEqual(state["status"], "sent")
        self.assertEqual(state["date"], "2026-10-20")
        self.assertTrue(state.get("engine_sha256"))

    def test_engine_drift_is_rejected(self):
        self.run_once()
        with patch.object(laoguo_signal, "engine_sha256", return_value="0" * 64):
            with self.assertRaises(ValueError):
                self.run_once()

    def test_failed_send_marks_state_failed_and_retries_next_run(self):
        ok, _ = self.run_once(sender=lambda f, b, d, sleeper=None: [(1, 1, 310000)])
        self.assertFalse(ok)
        state = json.load(open(self.state))
        self.assertEqual(state["status"], "failed")
        ok, _ = self.run_once()  # failed 状态不触发去重, 重发
        self.assertTrue(ok)

    def test_non_trading_day_skips_without_state(self):
        with patch.object(laoguo_signal.store, "trading_day", return_value=False):
            ok, summary = self.run_once()
        self.assertTrue(ok)
        self.assertIn("非交易日", summary)
        self.assertFalse(os.path.exists(self.state))


if __name__ == "__main__":
    unittest.main()
