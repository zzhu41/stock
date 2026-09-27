"""Real three-shadow integration with frozen seeds and a synthetic next session.

September 28 quotes below are artificial test observations, not market data.
All accounts, CSV projections and the live QVIX input live in TemporaryDirectory.
No real-time request, subprocess worker, production account or message is used.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import daily_extras
import premium
import shadow_0906
import shadow_v92
import strategy
from v10_live import data, policy, runtime


ROOT = Path(__file__).resolve().parent.parent
DAY = "2026-09-28"
NEXT_DAY = "2026-09-29"
NOW = datetime(2026, 9, 28, 14, 50, 30)
OLD = (shadow_0906, shadow_v92)


def strategy_globals():
    return {name: deepcopy(value) for name, value in vars(strategy).items()
            if name.isupper() or name == "_state_bull"}


class DailyExtrasIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # _load_seed and _build are read-only/pure helpers. No live cache builder
        # is called here, so neither a network request nor cache write can occur.
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")):
            seed = data._load_seed()
            pairs, histories, quotes = {}, {}, {}
            for code in data.CODES:
                with (data.SEED_DIR / "snapshots" / (code + ".csv")).open(newline="") as stream:
                    historical_qfq = [(r[0], float(r[1]), float(r[2]), float(r[3]))
                                      for r in csv.reader(stream) if r]
                raw = seed["raw"][code][-22:]
                prior_close, prior_volume = raw[-1][2], raw[-1][3]
                price = float(prior_close)  # An explicitly hypothetical flat-price session.
                volume = float(prior_volume) * .8
                row = (DAY, price, price, volume)
                qfq_by_date = {r[0]: r for r in historical_qfq}
                pairs[code] = dict(raw=raw + [row],
                    qfq=[qfq_by_date[r[0]] for r in raw] + [row],
                    sources={"raw": "synthetic integration fixture", "qfq": "same-vintage fixture"})
                histories[code] = historical_qfq + [row]
                quotes[code] = dict(date=DAY, timestamp=DAY + " 14:50:00", price=price,
                                    open=price, volume=volume, prev_close=price)
            checked = data._quotes(quotes, DAY, NOW)
            view, unused_completed_state = data._build(seed, data._empty_state(seed), pairs, checked, DAY, None)
            cls.view = view
            cls.payload = dict(histories=histories, quotes=quotes, date=DAY,
                               table=strategy.rank(histories, on_date=DAY))
            cls.qvix_bytes = (data.SEED_DIR / "snapshots/qvix50.csv").read_bytes()

    @contextmanager
    def isolated_collect(self):
        with tempfile.TemporaryDirectory(prefix="daily_extras_integration_") as directory, ExitStack() as stack:
            root = Path(directory)
            paths = {}
            for module, label in ((shadow_0906, "0906"), (shadow_v92, "v92")):
                paths[label] = root / (label + ".json")
                for attribute, path in (("STATE_FILE", paths[label]),
                                        ("TRADES_FILE", root / (label + ".csv")),
                                        ("PORTFOLIO", root / (label + "_actual_portfolio.json"))):
                    stack.enter_context(patch.object(module, attribute, str(path)))
            paths["H"] = root / "H.json"
            qvix_path = root / "qvix50.csv"
            qvix_path.write_bytes(self.qvix_bytes)
            stack.enter_context(patch.object(policy, "QVIX_PATH", qvix_path))
            stack.enter_context(patch("urllib.request.urlopen", side_effect=AssertionError("Network/message forbidden")))
            view_builder = stack.enter_context(patch.object(data, "build_live_view", return_value=self.view))
            qvix_fetch = stack.enter_context(patch.object(shadow_0906, "qvix_state", return_value=(
                None, None, "2026-09-24", False, "当日QVIX缺失，恐慌通道停用")))
            premium_fetch = stack.enter_context(patch.object(premium, "signal_block", return_value=[
                "  513100 纳指ETF 溢价 +0.00%（离线测试样例）"]))

            # Real runtime and policy, with only the account path/clock redirected.
            real_run = runtime.run
            real_decide = policy.decide
            h_decisions = []

            def h_decide(*args, **kwargs):
                decision = real_decide(*args, **kwargs)
                h_decisions.append(decision)
                return decision

            def routed_run(quotes, signal_date, **kwargs):
                return real_run(quotes, signal_date, state_path=paths["H"],
                    now=datetime.fromisoformat(signal_date + "T14:50:30"), decide=h_decide, **kwargs)

            runtime_calls = stack.enter_context(patch.object(runtime, "run", side_effect=routed_run))
            old_calls = {module.__name__: stack.enter_context(patch.object(module, "block", wraps=module.block))
                         for module in OLD}
            # Main generation has already made its ordinary advice before extras.
            # Preserve that normal regime initialization, rather than mistaking
            # _state_bull None -> bool for a cross-strategy parameter overwrite.
            stack.enter_context(patch.object(strategy, "_state_bull", strategy._state_bull))
            stack.enter_context(patch.object(strategy, "BULL_HYST_PENDING", strategy.BULL_HYST_PENDING))
            strategy.advice(self.payload["table"], None)
            yield dict(root=root, paths=paths, view_builder=view_builder, qvix_fetch=qvix_fetch,
                       premium_fetch=premium_fetch, runtime_calls=runtime_calls, old_calls=old_calls,
                       h_decisions=h_decisions, globals=strategy_globals())

    def assert_shared_inputs_and_first_accounts(self, context, lines):
        context["view_builder"].assert_called_once_with(self.payload["quotes"], DAY)
        context["qvix_fetch"].assert_called_once_with(DAY)
        context["premium_fetch"].assert_called_once_with(quotes=self.payload["quotes"])
        self.assertEqual(context["runtime_calls"].call_count, 1)
        self.assertEqual(len(context["h_decisions"]), 1)
        text = "\n".join(lines)
        for label in ("v9.1-0906", "v9.2", "V10-H"):
            self.assertIn("【影子 " + label + "】", text)
        self.assertNotIn("计算失败", text)
        self.assertIn("离线测试样例", text)
        self.assertEqual(text.count("虚拟净值 1.0000"), 3)
        for name, module in (("0906", shadow_0906), ("v92", shadow_v92)):
            call = context["old_calls"][module.__name__]
            call.assert_called_once()
            self.assertIs(call.call_args.kwargs["action_view"], self.view)
            self.assertIs(call.call_args.kwargs["quotes"], self.payload["quotes"])
            self.assertEqual(call.call_args.kwargs["qvix"], context["qvix_fetch"].return_value)
            state = json.loads(context["paths"][name].read_text())
            self.assertEqual(state["account_id"], "v9.1-0906" if name == "0906" else "v9.2")
            self.assertEqual(state["valuation_version"], 4)
            self.assertEqual(state["holding"], state["last_advice"].split()[0])
            self.assertEqual(state["holding"], state["events"][-1]["to"])
            self.assertEqual(state["nav"], 1.)
            self.assertEqual(state["last_date"], DAY)
            self.assertEqual(state["start_date"], DAY)
            self.assertEqual(len(state["events"]), 1)
            self.assertFalse(state["events"][0]["trade"])
            self.assertAlmostEqual(state["units"], 1 / self.payload["quotes"][state["holding"]]["price"])
        h = json.loads(context["paths"]["H"].read_text())
        self.assertEqual(h["candidate_id"], policy.CANDIDATE_ID)
        self.assertEqual(h["holding"], context["h_decisions"][0]["target"])
        self.assertEqual(h["holding"], h["last_decision"]["target"])
        self.assertEqual(h["nav"], 1.)
        self.assertEqual(h["last_date"], DAY)
        self.assertEqual(len(h["events"]), 1)
        self.assertEqual(h["switches"], 0)
        self.assertEqual(h["events"][0]["fee_fraction"], 0.)
        self.assertAlmostEqual(h["units"], 1 / self.payload["quotes"][h["holding"]]["price"])
        self.assertEqual(strategy_globals(), context["globals"])

    def test_real_three_strategy_blocks_share_inputs_and_repeat_without_rebooking(self):
        payload_before, view_before = deepcopy(self.payload), deepcopy(self.view)
        with self.isolated_collect() as context:
            first = daily_extras.collect(self.payload)
            self.assert_shared_inputs_and_first_accounts(context, first)
            saved = {path: path.read_bytes() for path in context["paths"].values()}
            csv_saved = {path: path.read_bytes() for path in context["root"].glob("*.csv")}
            with patch.object(strategy, "decide", side_effect=AssertionError("Duplicate old decision")), \
                    patch.object(policy, "decide", side_effect=AssertionError("Duplicate H decision")):
                second = daily_extras.collect(self.payload)
            self.assertEqual(first, second)
            for path, content in saved.items():
                self.assertEqual(path.read_bytes(), content)
            for path, content in csv_saved.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(context["view_builder"].call_count, 2)  # once per collect, never once per model
            self.assertEqual(context["qvix_fetch"].call_count, 2)
            self.assertEqual(len(context["h_decisions"]), 1)
            self.assertEqual(strategy_globals(), context["globals"])
        self.assertEqual(self.payload, payload_before)
        self.assertEqual(self.view, view_before)

    def test_new_day_shared_view_failure_returns_three_visible_failures_without_state_changes(self):
        with self.isolated_collect() as context:
            first = daily_extras.collect(self.payload)
            self.assert_shared_inputs_and_first_accounts(context, first)
            saved = {path: path.read_bytes() for path in context["paths"].values()}
            csv_saved = {path: path.read_bytes() for path in context["root"].glob("*.csv")}
            payload = deepcopy(self.payload)
            payload["date"] = NEXT_DAY
            for code, quote in payload["quotes"].items():
                quote.update(date=NEXT_DAY, timestamp=NEXT_DAY + " 14:50:00")
                payload["histories"][code].append((NEXT_DAY, quote["open"], quote["price"], quote["volume"]))
            payload["table"] = strategy.rank(payload["histories"], on_date=NEXT_DAY)
            context["view_builder"].side_effect = ValueError("unverified corporate action fixture")
            with patch.object(strategy, "decide", side_effect=AssertionError("Old decision without verified view")), \
                    patch.object(policy, "decide", side_effect=AssertionError("H decision without verified view")):
                lines = daily_extras.collect(payload)  # failures remain optional-block results, not a raised exception
            text = "\n".join(lines)
            self.assertIn("影子 v9.1-0906 计算失败", text)
            self.assertIn("影子 v9.2 计算失败", text)
            self.assertIn("【影子 V10-H】", text)
            self.assertIn("本次无有效建议", text)
            self.assertNotIn("影子持仓:", text)
            self.assertIn("离线测试样例", text)
            for path, content in saved.items():
                self.assertEqual(path.read_bytes(), content)
            for path, content in csv_saved.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(context["view_builder"].call_count, 2)
            self.assertEqual(context["qvix_fetch"].call_count, 2)
            self.assertEqual(len(context["h_decisions"]), 1)
            self.assertEqual(strategy_globals(), context["globals"])


if __name__ == "__main__":
    unittest.main()
