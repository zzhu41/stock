"""Integration checks for the explicitly ideal same-close diagnostic clock."""
import copy
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from v10_next.candidates import get_candidate
from v10_next.data import Features, load_histories
from v10_next.frozen.metadata import CASH, GLOBAL_POOL, STOCK_POOL
from v10_next.legacy import LegacyPolicy
from v10_round2.timing import for_same_close, run_same_close
from v10_round2.v92 import V92Policy, load_qvix


def dates(n):
    return [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]


class NoFear:
    def state(self, d):
        return dict(active=False, available=True, date=d, z=0.0, note="")


def make_policy(kind, features, calendar, qvix=None):
    config = get_candidate("control_v91")
    if kind == "v92":
        config.update(id="control_v92", kind="v92", channels=("deep", "qvix", "volume"))
    before = copy.deepcopy(config)
    adapted = for_same_close(config)
    if config != before:
        raise AssertionError("Clock adaptation mutated a registered candidate")
    if adapted["overrides"]["crash_lock"] != 6:
        raise AssertionError("Same-close diagnostic must offset the next-open decision clock once")
    return (V92Policy(adapted, features, calendar, qvix or NoFear()) if kind == "v92"
            else LegacyPolicy(adapted, features, calendar))


class SameCloseIntegrationTests(unittest.TestCase):
    def test_both_policies_lock_exactly_five_close_to_close_returns(self):
        ds = dates(8)
        h = {"510300": [(d, 100 * 1.1 ** i, 100 * 1.1 ** i, 100) for i, d in enumerate(ds)],
             CASH: [(d, 100, 100, 100) for d in ds]}
        trigger = [("510300", dict(mom5=-.09, dist_ma250=-.25))]
        for kind in ("legacy", "v92"):
            with self.subTest(kind=kind):
                features = Features(h)
                policy = make_policy(kind, features, ds)
                with patch.object(features, "table", side_effect=lambda d, p, w: trigger if d == ds[0] else []), \
                        patch("v10_next.legacy.strategy.decide", return_value=(CASH, "ordinary exit")):
                    result = run_same_close(h, ds, policy, ds[0], ds[-1], fee=0, slippage=0)
                self.assertEqual([t["date"] for t in result["trades"]], [ds[0], ds[5]])
                self.assertAlmostEqual(result["nav"], 1.1 ** 5)
                self.assertEqual(policy.metadata["crash_events"][0]["fill_date"], ds[0])
                signals = {r["date"]: r["target"] for r in policy.metadata["trace"]}
                self.assertEqual(signals[ds[4]], "510300")
                self.assertEqual(signals[ds[5]], CASH)

    def test_both_policies_pay_first_entry_and_sell_buy_commissions(self):
        ds, fee = dates(8), .0001
        h = {"510300": [(d, 100, 100 * 1.1 ** i, 100) for i, d in enumerate(ds)],
             CASH: [(d, 100, 100, 100) for d in ds]}
        trigger = [("510300", dict(mom5=-.09, dist_ma250=-.25))]
        for kind in ("legacy", "v92"):
            with self.subTest(kind=kind):
                features = Features(h)
                policy = make_policy(kind, features, ds)
                with patch.object(features, "table", side_effect=lambda d, p, w: trigger if d == ds[0] else []), \
                        patch("v10_next.legacy.strategy.decide", return_value=(CASH, "ordinary exit")):
                    result = run_same_close(h, ds, policy, ds[0], ds[-1], fee=fee, slippage=0)
                self.assertAlmostEqual(result["daily"][0][1], 1 / (1 + fee))
                self.assertAlmostEqual(result["nav"], 1.1 ** 5 * (1 - fee) / (1 + fee) ** 2)
                self.assertEqual(len(result["trades"][0]["fills"]), 1)
                self.assertEqual(len(result["trades"][1]["fills"]), 2)
                self.assertTrue(all(t["cash_after"] >= 0 for t in result["trades"]))

    def test_missing_held_bar_defers_crash_and_confirms_fill_before_lock(self):
        ds = dates(12)
        h = {
            "510300": [(d, 100, 100 if i < 4 else 120, 100)
                       for i, d in enumerate(ds) if i not in (2, 3)],
            "510500": [(d, 100, 100 * 1.1 ** max(0, i - 4), 100) for i, d in enumerate(ds)],
            CASH: [(d, 100, 100, 100) for d in ds],
        }
        trigger = [("510500", dict(mom5=-.09, dist_ma250=-.25))]
        for kind in ("legacy", "v92"):
            with self.subTest(kind=kind):
                features = Features(h)
                policy = make_policy(kind, features, ds)
                clock = {"date": None}

                def table(d, pool, windows):
                    clock["date"] = d
                    return trigger if d == ds[2] else []

                def decide(observed_table, holding, age):
                    return ("510300" if clock["date"] < ds[2] else CASH), "ordinary"

                with patch.object(features, "table", side_effect=table), \
                        patch("v10_next.legacy.strategy.decide", side_effect=decide):
                    result = run_same_close(h, ds, policy, ds[0], ds[-1], fee=0, slippage=0)
                self.assertEqual([t["date"] for t in result["trades"]], [ds[0], ds[4], ds[9]])
                self.assertEqual(result["trades"][1]["signal_date"], ds[2])
                self.assertEqual(result["diagnostics"]["deferred_count"], 2)
                self.assertEqual([d["date"] for d in result["diagnostics"]["deferred_rebalances"]], [ds[2], ds[3]])
                event = policy.metadata["crash_events"][0]
                self.assertEqual((event["signal_date"], event["fill_date"]), (ds[2], ds[4]))
                self.assertAlmostEqual(result["daily"][3][1], 1.0)
                self.assertAlmostEqual(result["nav"], 1.2 * 1.1 ** 5)

    def test_real_no_gap_close_targets_match_old_lab_for_legacy_and_v92(self):
        import v10.lab as lab
        h = load_histories()
        calendar = [r[0] for r in h["510300"]]
        qvix = load_qvix()
        features = Features(h)
        fear = dict(qd=[], qz=[], qv=[], rd=[], r5=[])
        for d, value in qvix.rows:
            observation = qvix.state(d)
            if observation["available"]:
                fear["qd"].append(d)
                fear["qz"].append(observation["z"])
                fear["qv"].append(value)
        original_histories = {c: h[c] for c in list(STOCK_POOL) + list(GLOBAL_POOL) + ["518880", CASH]}
        start, end = "2022-03-01", "2022-04-01"
        for kind in ("legacy", "v92"):
            with self.subTest(kind=kind):
                cfg = (dict(fear_qz=2.5, fear_m5=-.04, crash_volu=2.0, cv_m5=-.04, cv_dep=.10)
                       if kind == "v92" else {})
                with patch.dict(lab.CFG, cfg, clear=True), patch.dict(lab.FEAR, fear, clear=True), \
                        patch.object(lab.strategy, "STOCK_POOL", list(STOCK_POOL)), \
                        patch.object(lab.strategy, "GLOBAL_POOL", list(GLOBAL_POOL)):
                    original = lab.backtest_v10(original_histories, calendar, start, end)
                policy = make_policy(kind, features, calendar, qvix)
                executed = run_same_close(h, calendar, policy, start, end, fee=.0001, slippage=0)
                self.assertEqual(executed["diagnostics"]["deferred_count"], 0)
                self.assertEqual([(d, target) for d, _, target in original["daily"]],
                                 [(r["date"], r["target"]) for r in policy.metadata["trace"]])
                self.assertEqual(original["crash_buys"], [(e["signal_date"], e["code"])
                                                        for e in policy.metadata["crash_events"]])
                self.assertEqual(executed["trades"][0]["date"], executed["daily"][0][0])
                self.assertLess(executed["daily"][0][1], original["daily"][0][1])


if __name__ == "__main__":
    unittest.main()
