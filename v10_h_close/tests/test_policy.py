"""Focused close-policy coverage independent of candidate-return ranking."""
from copy import deepcopy
from itertools import combinations
import json
import unittest
from unittest.mock import patch

from v10_next.data import configure
from v10_next.frozen import strategy as native_strategy
from v10_search.features import CONSTANTS
from v10_search.registry import _candidate
from v10_h_close.data import BASE, BENCHMARK, CASH, GOLD
from v10_h_close.policy import ClosePolicy


def ind(m5=.01, depth=.05, ratio=1, mom20=-.10, score=10, bull=True):
    return dict(mom5=m5, mom20=mom20, mom60=mom20, vol=.02, score=score, above_ma=bull,
                ret1=.001, mom20_max=mom20, pos_frac=1, max_ret=.02, dvol=.02,
                ma_rising=True, dist_ma250=depth, volume_ratio20=ratio)


class SmallBank:
    def __init__(self, rows, n=10):
        self.calendar = tuple("2022-01-%02d" % (i + 1) for i in range(n))
        self.rows = [deepcopy(rows) for _ in self.calendar]
        self.dates = {CASH: list(self.calendar)}
        self.presets = json.loads((BASE.parent / "v10_next/frozen/presets.json").read_text())

    def ranked(self, mode, windows):
        return self.rows


def candidate(bank, channels, buffer=.02, split=False):
    c = _candidate(bank.presets["v9.1"], "fixed_close_policy_test", ("159915",), ("513100",),
                   crash="deep" in channels,
                   overrides=dict(never_empty=False, buffer=buffer,
                                  pool_buffer={"stock": .02, "global": .03, "gold": .03} if split else {}))
    c["channels"] = list(channels)
    return c


def row_with(candidate_ind):
    return [(BENCHMARK, ind(score=100, mom20=-.10, bull=False)),
            ("159915", candidate_ind), ("513100", ind(score=1)), (GOLD, ind(score=0))]


class ClosePolicyTests(unittest.TestCase):
    def test_all_eight_channel_subsets_have_independent_or_semantics(self):
        channels = ("deep", "qvix", "volume")
        subsets = [s for n in range(4) for s in combinations(channels, n)]
        # Expected channel names are specified independently, including strict
        # depth boundaries and external triggers which cannot pass the deep gate.
        samples = [
            (ind(-.09, -.25, 3), True, {"deep", "qvix", "volume"}),
            (ind(-.09, -.25, 1), False, {"deep"}),
            (ind(-.05, -.25, 1), True, {"qvix"}),
            (ind(-.05, -.15, 2), False, {"volume"}),
            (ind(-.08, -.20, 1), True, set()),
            (ind(-.04, -.1001, 2), False, {"volume"}),
            (ind(-.04, -.10, 2), False, set()),
            (ind(-.0399, -.25, 3), True, set()),
        ]
        for subset in subsets:
            for observation, fear, eligible in samples:
                with self.subTest(channels=subset, observation=observation, fear=fear):
                    bank = SmallBank(row_with(observation))
                    config = candidate(bank, subset)
                    if "deep" not in subset:
                        self.assertEqual(config["params"]["crash_mom5"], 0.0)
                    policy = ClosePolicy(config, bank, fear=[fear] * 10, trace=True)
                    hit = bool(set(subset) & eligible)
                    self.assertEqual(policy(1, CASH, 1, True), "159915" if hit else CASH)
                    self.assertEqual(policy.lock_until, 6 if hit else -1)
                    self.assertEqual(len(policy.metadata["crash_buys"]), int(hit))

    def test_external_channels_do_not_start_a_lock_without_sellable_holding(self):
        bank = SmallBank(row_with(ind(-.09, -.25, 3)))
        for channels in (("qvix",), ("volume",), ("qvix", "volume"), ("deep", "qvix", "volume")):
            with self.subTest(channels=channels):
                policy = ClosePolicy(candidate(bank, channels), bank, fear=[True] * 10, trace=True)
                self.assertEqual(policy(1, GOLD, 1, False), CASH)
                self.assertEqual(policy.lock_until, -1)
                self.assertEqual(policy.metadata["crash_buys"], [])

    def test_lock_uses_original_i_plus_five_and_excludes_current_holding(self):
        bank = SmallBank(row_with(ind(-.09, -.25, 3)))
        policy = ClosePolicy(candidate(bank, ("deep", "qvix", "volume")), bank, fear=[True] * 10, trace=True)
        self.assertEqual(policy(0, CASH, 5, True), "159915")
        for i in range(1, 5):
            self.assertEqual(policy(i, "159915", i, True), "159915")
        self.assertEqual(policy(5, "159915", 5, True), CASH)
        self.assertEqual(policy.metadata["crash_buys"], [(bank.calendar[0], "159915")])

    def test_benchmark_only_asset_cannot_be_bought_by_any_crash_channel(self):
        rows = [(BENCHMARK, ind(-.10, -.30, 3, score=100, bull=False)),
                ("159915", ind(mom20=.12, score=20)), ("513100", ind(score=1)), (GOLD, ind(score=0))]
        bank = SmallBank(rows)
        policy = ClosePolicy(candidate(bank, ("deep", "qvix", "volume")), bank, fear=[True] * 10, trace=True)
        self.assertIn(BENCHMARK, policy.feature_pool)
        self.assertNotIn(BENCHMARK, policy.trade_pool)
        self.assertEqual(policy(0, None, 0, True), CASH)
        self.assertEqual(policy.metadata["crash_buys"], [])

    def test_uniform_and_split_buffers_match_complete_native_decide(self):
        row = [("513100", ind(mom20=.125, score=30)), ("159915", ind(mom20=.10, score=20)),
               (BENCHMARK, ind(mom20=.01, score=1, bull=True)), (GOLD, ind(score=0))]
        bank = SmallBank(row)
        for buffer, split, expected in ((.01, False, "513100"), (.02, False, "513100"),
                                        (.03, False, "159915"), (.04, False, "159915"),
                                        (.02, True, "159915")):
            with self.subTest(buffer=buffer, split=split):
                config = candidate(bank, (), buffer, split)
                policy = ClosePolicy(config, bank)
                target = policy(1, "159915", 5, True)
                configure(config["params"], config["stock_pool"], config["global_pool"])
                complete_table = [(c, dict(CONSTANTS, **data, mom20_peak20=data["mom20_max"])) for c, data in row]
                native_target, _ = native_strategy.decide(complete_table, "159915", 5)
                self.assertEqual(target, native_target)
                self.assertEqual(target, expected)

    def test_full_union_boundary_cases_match_old_lab_actual_trigger_block(self):
        import v10.lab as lab
        observations = [(ind(-.08, -.20, 1), True), (ind(-.08, -.201, 1), False),
                        (ind(-.04, -.201, 1), True), (ind(-.04, -.101, 2), False),
                        (ind(-.0399, -.30, 3), True)]
        for observation, fear in observations:
            with self.subTest(observation=observation, fear=fear):
                bank = SmallBank(row_with(observation), 3)
                policy = ClosePolicy(candidate(bank, ("deep", "qvix", "volume")), bank,
                                     fear=[fear] * 3, trace=True)
                wanted = policy(1, CASH, 1, True)
                old_ind = dict(observation, vol_ratio20=observation["volume_ratio20"])
                histories = {code: [(d, 100, 100, 100) for d in bank.calendar] for code in ("159915", CASH)}
                cfg = dict(fear_qz=2.5, fear_m5=-.04, crash_volu=2.0, cv_m5=-.04, cv_dep=.10)
                with patch.dict(lab.CFG, cfg, clear=True), patch.object(lab, "_reset_strategy"), \
                        patch.object(lab, "_fear_active", return_value=fear), \
                        patch.object(lab.strategy, "rank", side_effect=lambda h, on_date: [("159915", old_ind)] if on_date == bank.calendar[1] else []), \
                        patch.object(lab.strategy, "decide", return_value=(CASH, "ordinary")), \
                        patch.object(lab.strategy, "STOCK_POOL", ["159915"]), \
                        patch.object(lab.strategy, "GLOBAL_POOL", []):
                    reference = lab.backtest_v10(histories, bank.calendar, bank.calendar[0], bank.calendar[-1])
                self.assertEqual(reference["daily"][1][2], wanted)
                self.assertEqual(reference["crash_buys"], policy.metadata["crash_buys"])


if __name__ == "__main__":
    unittest.main()
