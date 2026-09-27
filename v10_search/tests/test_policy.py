"""Independent indicator checks and bounded decision-oracle regressions."""
from copy import deepcopy
from datetime import date, timedelta
import json
import math
import unittest

from v10_next.frozen import strategy as frozen_strategy
from v10_search.data import BASE, BENCHMARK, CASH, GOLD
from v10_search.fast_execution import prepare, run
from v10_search.features import CONSTANTS, FeatureBank
from v10_search.policy import SearchPolicy
from v10_search.registry import _candidate
from v10_search.verify_fidelity import check_features, path_comparison


def indicator(score, mom20, mom5=.01, below=.05, bull=True, ret1=.001):
    return dict(mom5=mom5, mom20=mom20, mom60=mom20, vol=.02, score=score,
                above_ma=bull, ret1=ret1, mom20_max=mom20, pos_frac=1.0, max_ret=.02,
                dvol=.02, ma_rising=True, dist_ma250=below, volume_ratio20=1.0)


class SmallBank:
    def __init__(self, rows, calendar):
        self.rows, self.calendar = rows, tuple(calendar)
        self.dates = {CASH: list(calendar)}
        self.presets = json.loads((BASE.parent / "v10_next/frozen/presets.json").read_text())

    def ranked(self, mode, windows):
        return self.rows

    def table(self, i, mode, windows, codes, oracle=False):
        row = [(c, ind) for c, ind in self.rows[i] if c in codes]
        if oracle:
            return [(c, dict(CONSTANTS, **ind, mom20_peak20=ind["mom20_max"])) for c, ind in row]
        return row


def config(bank, regime="ma250", crash=True):
    return _candidate(bank.presets["v9.1"], "synthetic_fidelity_only", ("159915",), ("513100",),
                      crash=crash, regime_mode=regime, overrides={"never_empty": False})


class SearchPolicyTests(unittest.TestCase):
    def test_benchmark_excluded_from_trading_still_controls_bear_regime(self):
        row = [(BENCHMARK, indicator(100, -.20, mom5=-.10, below=-.30, bull=False)),
               ("159915", indicator(20, .10)), ("513100", indicator(10, -.05)),
               (GOLD, indicator(5, -.02))]
        bank = SmallBank([row], ["2022-03-15"])
        for oracle in (False, True):
            with self.subTest(oracle=oracle):
                policy = SearchPolicy(config(bank), bank, oracle=oracle, trace=True)
                self.assertIn(BENCHMARK, policy.feature_pool)
                self.assertNotIn(BENCHMARK, policy.trade_pool)
                self.assertEqual(policy(0, None, None, False), CASH)
                self.assertIsNone(policy.pending)  # The benchmark's deep drop is never buyable.

    def test_three_regime_modes_preserve_their_distinct_registered_meanings(self):
        row = [(BENCHMARK, indicator(100, -.20, mom5=-.10, below=-.30, bull=False)),
               (GOLD, indicator(30, .06)), ("159915", indicator(20, .05)),
               ("513100", indicator(10, .09))]
        bank = SmallBank([row], ["2022-03-15"])
        expected = dict(ma250="513100", open_stock="513100", always_bull="159915", all_assets=GOLD)
        for mode, target in expected.items():
            for oracle in (False, True):
                with self.subTest(mode=mode, oracle=oracle):
                    c = config(bank, mode)
                    policy = SearchPolicy(c, bank, oracle=oracle, trace=True)
                    self.assertEqual(policy(0, None, None, False), target)
                    self.assertNotEqual(target, BENCHMARK)
                    self.assertIsNone(policy.pending)
                    if oracle:
                        self.assertEqual(frozen_strategy.GLOBAL_POOL, c["global_pool"])
        # Once the stock clears the retained 7% bear floor, open_stock admits
        # it while normal ma250 continues to exclude the domestic stock pool.
        changed = deepcopy(row)
        next(ind for code, ind in changed if code == "159915")["mom20"] = .08
        bank = SmallBank([changed], ["2022-03-15"])
        self.assertEqual(SearchPolicy(config(bank, "open_stock"), bank)(0, None, None, False), "159915")
        self.assertEqual(SearchPolicy(config(bank, "ma250"), bank)(0, None, None, False), "513100")

    def test_crash_lock_uses_confirmed_fill_after_missing_bar_deferral(self):
        calendar = [(date(2022, 1, 3) + timedelta(days=i)).isoformat() for i in range(9)]
        regular = [(BENCHMARK, indicator(100, -.20, bull=False)),
                   ("159915", indicator(20, -.10)), ("513100", indicator(10, -.10)),
                   (GOLD, indicator(5, -.10))]
        first = deepcopy(regular)
        crash = next(ind for code, ind in first if code == "159915")
        crash.update(mom5=-.09, dist_ma250=-.25)
        bank = SmallBank([first] + [deepcopy(regular) for _ in calendar[1:]], calendar)
        histories = {code: [(d, 100, 100, 100) for j, d in enumerate(calendar)
                            if not (code == "159915" and j == 1)]
                     for code in (BENCHMARK, "159915", "513100", GOLD, CASH)}
        frame = prepare(histories, calendar)
        fast_policy = SearchPolicy(config(bank), bank, frame, trace=True)
        fast = run(frame, fast_policy, 0, 8, capture_daily=True, capture_trades=True)
        native_policy = SearchPolicy(config(bank), bank, frame, oracle=True, trace=True)
        native = run(frame, native_policy, 0, 8, capture_daily=True, capture_trades=True)
        path_comparison(fast, native)
        self.assertEqual(fast["deferred_count"], 1)
        self.assertEqual([t["date"] for t in fast["trades"]], [calendar[2], calendar[7]])
        self.assertEqual(fast_policy.metadata["crash_events"],
                         [dict(signal_date=calendar[0], code="159915", fill_date=calendar[2])])
        self.assertEqual(fast_policy.metadata, native_policy.metadata)

    def test_panic_exit_overrides_buffer_even_when_the_holding_ranks_first(self):
        row = [(BENCHMARK, indicator(1, .05)),
               ("159915", indicator(100, .25, ret1=-.05)),
               ("513100", indicator(90, .10)), (GOLD, indicator(5, .05))]
        bank = SmallBank([row], ["2024-10-09"])
        for oracle in (False, True):
            policy = SearchPolicy(config(bank, crash=False), bank, oracle=oracle)
            self.assertEqual(policy(0, "159915", 0, False), "513100")

    def test_bank_matches_independent_original_prices_and_has_no_future_leak(self):
        calendar = [(date(2013, 1, 1) + timedelta(days=i)).isoformat() for i in range(460)]
        histories = {}
        for k, code in enumerate((BENCHMARK, "159915", GOLD, CASH)):
            histories[code] = [(d, (10 + k) * (1 + .001 * i + .025 * math.sin(i / 11)),
                                (10 + k) * (1 + .001 * i + .03 * math.sin(i / 11)), 100 + i % 13)
                               for i, d in enumerate(calendar)]
        bank = FeatureBank(histories, calendar)
        checked = check_features(bank, histories, [calendar[370], calendar[420]])
        self.assertTrue(checked["passed"])
        altered = deepcopy(histories)
        for code, rows in altered.items():
            altered[code] = [(d, o * 100, c * 200, v * 1000) if d > calendar[420] else (d, o, c, v)
                             for d, o, c, v in rows]
        later = FeatureBank(altered, calendar)
        for mode, windows in (("wls", (20,)), ("wls", (40,)), ("mom60_vol", ()), ("mom20", ())):
            self.assertEqual(bank.ranked(mode, windows)[420], later.ranked(mode, windows)[420])

    def test_unsupported_changes_are_rejected_before_deciding(self):
        bank = SmallBank([[]], ["2022-03-15"])
        c = config(bank)
        c["params"]["min_hold"] = 3
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            SearchPolicy(c, bank)


if __name__ == "__main__":
    unittest.main()
