"""Offline v9.2 channel, data-timing, fill-lock and original-lab controls."""
import copy
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from v10_next.data import Features, load_histories
from v10_next.execution import run
from v10_next.frozen.metadata import CASH, GLOBAL_POOL, STOCK_POOL
from v10_round2.v92 import (ALL_CHANNELS, END, QvixSeries, V92Policy,
                           load_qvix, trigger_channels)


def days(n):
    return [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]


class FixedFear:
    def __init__(self, active):
        self.active = active

    def state(self, d):
        return dict(active=self.active, available=True, date=d,
                    z=3.0 if self.active else 0.0, note="")


def synthetic(n=30):
    calendar = days(n)
    return {code: [(d, 100.0, 100.0, 100.0) for d in calendar]
            for code in ("510300", "510500", CASH)}, calendar


class V92Tests(unittest.TestCase):
    def test_qvix_uses_previous_250_excludes_today_and_future(self):
        ds = days(303)
        values = [1000.0] * 50 + [10.0, 20.0] * 125 + [30.0]
        rows = list(zip(ds, values))
        current = QvixSeries(rows).state(ds[300])
        extended = QvixSeries(rows + [(ds[301], 1e9), (ds[302], 1e-6)]).state(ds[300])
        self.assertEqual(current, extended)
        self.assertEqual(current["previous_samples"], 250)
        self.assertEqual(current["z"], 3.0)
        self.assertTrue(current["active"])

    def test_qvix_requires_same_day_and_120_previous_rows(self):
        ds = days(123)
        series = QvixSeries([(d, 10 + i % 2) for i, d in enumerate(ds[:121])])
        self.assertFalse(series.state(ds[119])["available"])
        self.assertTrue(series.state(ds[120])["available"])
        missing = series.state(ds[121])
        self.assertFalse(missing["active"])
        self.assertEqual(missing["note"], "missing_same_day_qvix")
        self.assertEqual(missing["date"], ds[120])

    def test_volume_ratio_uses_strict_prior_20_and_current_day(self):
        h, ds = synthetic()
        h["510300"][20] = (ds[20], 100, 100, 200)
        feature = Features(h)
        policy = V92Policy({}, feature, ds, FixedFear(False))
        self.assertEqual(policy.volume_ratio(ds[20], "510300"), 2.0)
        self.assertEqual(policy.volume_ratio(ds[19], "510300"), 0.0)
        altered = copy.deepcopy(h)
        altered["510300"] = [(d, o, c, 1e9 if d > ds[20] else v) for d, o, c, v in h["510300"]]
        other = V92Policy({}, Features(altered), ds, FixedFear(False))
        self.assertEqual(other.volume_ratio(ds[20], "510300"), 2.0)

    def test_exact_channel_edges_and_qvix_only_keeps_original_deep_drop(self):
        self.assertEqual(trigger_channels({"mom5": -.08, "dist_ma250": -.20}, 0, False), [])
        self.assertEqual(trigger_channels({"mom5": -.08, "dist_ma250": -.201}, 0, False), ["deep"])
        self.assertEqual(trigger_channels({"mom5": -.04, "dist_ma250": -.201}, 2, True), ["qvix", "volume"])
        self.assertEqual(trigger_channels({"mom5": -.04, "dist_ma250": -.10}, 2, False), [])
        self.assertEqual(trigger_channels({"mom5": -.04, "dist_ma250": -.101}, 1.999, False), [])
        self.assertEqual(trigger_channels({"mom5": -.04, "dist_ma250": -.101}, 2, False), ["volume"])
        self.assertEqual(trigger_channels({"mom5": -.05, "dist_ma250": -.15}, 3, True,
                                          ("deep", "qvix")), [])
        self.assertEqual(trigger_channels({"mom5": -.09, "dist_ma250": -.25}, 0, False,
                                          ("deep", "qvix")), ["deep"])

    def test_current_holding_is_excluded_before_selecting_best_crash(self):
        h, ds = synthetic()
        h["510500"][20] = (ds[20], 100, 100, 200)
        features = Features(h)
        policy = V92Policy({}, features, ds, FixedFear(False))
        table = [("510300", {"mom5": -.09, "dist_ma250": -.25}),
                 ("510500", {"mom5": -.05, "dist_ma250": -.15})]
        state = dict(weights={"510300": 1}, holding_since={"510300": ds[19]},
                     pending_target=None, execution_deferred=False)
        with patch.object(features, "table", return_value=table), \
                patch("v10_round2.v92.strategy.decide", return_value=(CASH, "ordinary")):
            target = policy(ds[20], {}, state)
        self.assertEqual(target, {"510500": 1.0})
        self.assertEqual(policy.pending_crash["channels"], ["volume"])

    def test_deferred_entry_starts_five_day_lock_from_actual_fill(self):
        h, ds = synthetic()
        h["510300"] = [r for r in h["510300"] if r[0] != ds[21]]
        features = Features(h)
        policy = V92Policy({}, features, ds, FixedFear(False))
        trigger = [("510300", {"mom5": -.09, "dist_ma250": -.25})]
        with patch.object(features, "table", side_effect=lambda d, p, w: trigger if d == ds[20] else []), \
                patch("v10_round2.v92.strategy.decide", return_value=(CASH, "ordinary")):
            result = run(h, ds, policy, ds[20], ds[28], fee=0, slippage=0)
        event = policy.metadata["crash_events"][0]
        self.assertEqual(event["signal_date"], ds[20])
        self.assertEqual(event["fill_date"], ds[22])
        targets = {row["date"]: row["target"] for row in policy.metadata["trace"]}
        self.assertEqual(targets[ds[25]], "510300")
        self.assertEqual(targets[ds[26]], CASH)
        self.assertEqual([t["date"] for t in result["trades"]], [ds[22], ds[27]])
        self.assertEqual(result["diagnostics"]["deferred_count"], 1)

    def test_original_lab_engine_selects_identical_same_state_channel_union(self):
        # This invokes the original engine's actual candidate-selection block,
        # not a second copy of the channel expression as a test oracle.
        import v10.lab as lab
        h, ds = synthetic(3)
        cases = [(-.08, -.20, 0, False), (-.08, -.201, 0, False),
                 (-.04, -.201, 0, True), (-.04, -.101, 2, False),
                 (-.04, -.10, 2, False), (-.0399, -.25, 3, True),
                 (-.05, -.15, 1.999, False)]
        for channels in (ALL_CHANNELS, ("deep", "qvix")):
            for mom5, depth, ratio, fear in cases:
                with self.subTest(channels=channels, mom5=mom5, depth=depth, ratio=ratio, fear=fear):
                    ind = dict(mom5=mom5, dist_ma250=depth, vol_ratio20=ratio)
                    cfg = dict(fear_qz=2.5, fear_m5=-.04)
                    if "volume" in channels:
                        cfg.update(crash_volu=2.0, cv_m5=-.04, cv_dep=.10)
                    with patch.dict(lab.CFG, cfg, clear=True), \
                            patch.object(lab, "_reset_strategy"), \
                            patch.object(lab, "_fear_active", return_value=fear), \
                            patch.object(lab.strategy, "rank", side_effect=lambda hs, on_date: [("510300", ind)] if on_date == ds[1] else []), \
                            patch.object(lab.strategy, "decide", return_value=(CASH, "ordinary")), \
                            patch.object(lab.strategy, "STOCK_POOL", ["510300", "510500"]), \
                            patch.object(lab.strategy, "GLOBAL_POOL", []):
                        outcome = lab.backtest_v10(h, ds, ds[0], ds[-1])
                    expected = [(ds[1], "510300")] if trigger_channels(ind, ratio, fear, channels) else []
                    self.assertEqual(outcome["crash_buys"], expected)

    def test_real_no_gap_window_matches_original_lab_daily_targets(self):
        import v10.lab as lab
        h = load_histories()
        calendar = [row[0] for row in h["510300"]]
        qvix = load_qvix()
        feature = Features(h)
        fear = dict(qd=[], qz=[], qv=[], rd=[], r5=[])
        for d, value in qvix.rows:
            s = qvix.state(d)
            if s["available"]:
                fear["qd"].append(d)
                fear["qz"].append(s["z"])
                fear["qv"].append(value)
        start, end = "2022-03-01", "2022-04-01"
        old_histories = {c: h[c] for c in list(STOCK_POOL) + list(GLOBAL_POOL) + ["518880", CASH]}
        for channels in (ALL_CHANNELS, ("deep", "qvix")):
            with self.subTest(channels=channels):
                cfg = dict(fear_qz=2.5, fear_m5=-.04)
                if "volume" in channels:
                    cfg.update(crash_volu=2.0, cv_m5=-.04, cv_dep=.10)
                with patch.dict(lab.CFG, cfg, clear=True), patch.dict(lab.FEAR, fear, clear=True), \
                        patch.object(lab.strategy, "STOCK_POOL", list(STOCK_POOL)), \
                        patch.object(lab.strategy, "GLOBAL_POOL", list(GLOBAL_POOL)):
                    old = lab.backtest_v10(old_histories, calendar, start, end)
                policy = V92Policy({"channels": channels}, feature, calendar, qvix)
                new = run(h, calendar, policy, start, end)
                self.assertEqual(new["diagnostics"]["deferred_count"], 0)
                self.assertEqual([(d, target) for d, _, target in old["daily"]],
                                 [(r["date"], r["target"]) for r in policy.metadata["trace"]])
                self.assertEqual(old["crash_buys"], [(e["signal_date"], e["code"])
                                                   for e in policy.metadata["crash_events"]])

    def test_frozen_qvix_cutoff_does_not_fill_missing_latest_sessions(self):
        qvix = load_qvix()
        self.assertLessEqual(qvix.rows[-1][0], END)
        self.assertFalse(qvix.state("2026-09-10")["active"])
        self.assertFalse(qvix.state("2026-09-11")["available"])


if __name__ == "__main__":
    unittest.main()
