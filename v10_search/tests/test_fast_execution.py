"""Fast-kernel parity against the general causal units/cash execution engine."""
import copy
import random
import unittest
from datetime import date, timedelta

from evaluate_versions import metrics
from v10_next.execution import run as general_run
from v10_search.fast_execution import FLAT, prepare, run


def make_random_histories(seed, n=85):
    rng = random.Random(seed)
    calendar = [(date(2020, 12, 1) + timedelta(days=i)).isoformat() for i in range(n)]
    histories = {}
    for code in ("A", "B", "511880"):
        close, rows = 100.0, []
        for i, d in enumerate(calendar):
            opening = close * (1 + rng.uniform(-.08, .08))
            close = opening * (1 + rng.uniform(-.06, .06))
            if i not in (0, n - 1) and rng.random() < .14:
                continue
            rows.append((d, opening, close, 100.0))
        histories[code] = rows
    choices = [None, None, "A", "B", "511880", FLAT]
    instructions = [rng.choice(choices) for _ in calendar]
    return histories, calendar, instructions


def generic_bridge(frame, compact_policy):
    def policy(d, observed, state):
        holding = state["holding"]
        entry = frame.index[state["holding_since"][holding]] if holding is not None else None
        target = compact_policy(state["trading_index"], holding, entry, state["execution_deferred"])
        return None if target is None else ({} if target == FLAT else {target: 1.0})
    return policy


def legacy_bridge(frame, legacy_policy):
    """Only adapts state representation for an existing test policy, not its rules."""
    pending = [None]

    def policy(i, holding, entry, deferred):
        if not deferred:
            pending[0] = None
        state = dict(weights={holding: 1.0} if holding else {},
                     holding_since={holding: frame.dates[entry]} if holding else {},
                     pending_target=({pending[0]: 1.0} if pending[0] else {}) if pending[0] is not None else None,
                     execution_deferred=deferred)
        target = legacy_policy(frame.dates[i], {}, state)
        if target is None:
            return None
        if not target:
            pending[0] = FLAT
        elif len(target) == 1 and next(iter(target.values())) == 1:
            pending[0] = next(iter(target))
        else:
            raise AssertionError("Test bridge only supports full single-asset targets")
        return pending[0]
    return policy


class FastExecutionTests(unittest.TestCase):
    def assert_parity(self, fast, general):
        self.assertEqual(len(fast["daily"]), len(general["daily"]))
        for f, g in zip(fast["daily"], general["daily"]):
            self.assertEqual(f[0], g[0])
            self.assertAlmostEqual(f[1], g[1], delta=1e-10 * max(1, abs(g[1])))
            self.assertEqual(set(f[2]), set(g[2]))
            for code in g[2]:
                self.assertAlmostEqual(f[2][code], g[2][code], places=10)
        self.assertEqual(fast["signal_target"], general["signal_target"])
        self.assertEqual(len(fast["trades"]), len(general["trades"]))
        for f, g in zip(fast["trades"], general["trades"]):
            self.assertEqual((f["date"], f["signal_date"], f["target"]),
                             (g["date"], g["signal_date"], g["target"]))
            for key in ("nav_before", "nav_after", "cash_after", "commission", "slippage_cost", "turnover"):
                self.assertAlmostEqual(f[key], g[key], delta=1e-10 * max(1, abs(g[key])))
            self.assertEqual([(x["code"], x["side"]) for x in f["fills"]],
                             [(x["code"], x["side"]) for x in g["fills"]])
        for key in ("deferred_count", "rebalance_count", "fill_count", "policy_calls", "completed_noop_orders"):
            self.assertEqual(fast["diagnostics"][key], general["diagnostics"][key])
        for key in ("total_commission", "total_slippage_cost", "total_turnover"):
            self.assertAlmostEqual(fast["diagnostics"][key], general["diagnostics"][key], places=9)
        self.assertEqual(fast["diagnostics"]["deferred_rebalances"], general["diagnostics"]["deferred_rebalances"])
        self.assertEqual(fast["diagnostics"]["missing_held_bars"], general["diagnostics"]["missing_held_bars"])
        reference = metrics(general["daily"])
        for key in ("ann", "max_dd", "sharpe"):
            self.assertAlmostEqual(fast[key], reference[key], places=9)
        self.assertEqual(set(fast["yearly"]), set(reference["yearly"]))
        for year in reference["yearly"]:
            self.assertAlmostEqual(fast["yearly"][year], reference["yearly"][year], places=10)

    def test_random_paths_gaps_costs_and_actual_state_feedback_match_general_engine(self):
        for seed in range(18):
            h, calendar, instructions = make_random_histories(seed)
            frame = prepare(h, calendar)
            fee, slip = ((0, 0), (.0001, .001), (.001, .01))[seed % 3]
            fast_trace, general_trace = [], []

            def make_policy(trace):
                def policy(i, holding, entry, deferred):
                    trace.append((i, holding, entry, deferred))
                    # Change orders from actual execution feedback, so equality
                    # cannot be obtained merely by replaying a fixed target path.
                    return FLAT if deferred and holding == "A" else instructions[i]
                return policy

            with self.subTest(seed=seed, fee=fee, slippage=slip):
                start, end = 4, len(calendar) - 2
                fast = run(frame, make_policy(fast_trace), start, end, fee, slip,
                           capture_daily=True, capture_trades=True)
                general = general_run(h, calendar, generic_bridge(frame, make_policy(general_trace)),
                                      calendar[start], calendar[end], fee, slip)
                self.assertEqual(fast_trace, general_trace)
                self.assertEqual(fast["navs"][0], 1.0)
                self.assert_parity(fast, general)

    def test_array_capture_matches_full_and_streaming_without_extra_daily_dicts(self):
        h, calendar, targets = make_random_histories(42)
        frame = prepare(h, calendar)
        policy = lambda i, holding, entry, deferred: targets[i]
        full = run(frame, policy, calendar[0], calendar[-1], capture_daily=True, capture_trades=True)
        arrays = run(frame, policy, 0, len(calendar) - 1, capture_daily="arrays")
        stream = run(frame, policy, 0, len(calendar) - 1)
        self.assertEqual(full["navs"], arrays["navs"])
        self.assertEqual(full["holdings"], arrays["holdings"])
        self.assertIsNone(arrays["daily"])
        self.assertIsNone(arrays["signal_target"])
        self.assertIsNone(arrays["trades"])
        self.assertIsNone(stream["navs"])
        for key in ("nav", "ann", "max_dd", "sharpe", "yearly", "trade_count", "deferred_count"):
            self.assertEqual(full[key], arrays[key])
            self.assertEqual(full[key], stream[key])

    def test_first_fee_and_overnight_intraday_split(self):
        ds = ["2026-01-01", "2026-01-02", "2026-01-05"]
        h = {"A": [(ds[0], 100, 100), (ds[1], 100, 110), (ds[2], 121, 500)],
             "B": [(ds[0], 200, 200), (ds[1], 200, 200), (ds[2], 200, 220)]}
        f, s = .0001, .001
        policy = lambda i, holding, entry, deferred: "A" if i == 0 else "B"
        result = run(prepare(h, ds), policy, 0, 2, f, s, capture_daily=True, capture_trades=True)
        buy = (1 + f) * (1 + s)
        sell = (1 - f) * (1 - s)
        self.assertEqual(result["navs"][0], 1.0)
        self.assertAlmostEqual(result["navs"][1], 1.1 / buy)
        self.assertAlmostEqual(result["navs"][2], 1.331 * sell / buy ** 2)
        self.assertEqual(result["trade_count"], 2)
        self.assertEqual(result["fill_count"], 3)

    def test_held_suspension_postpones_whole_switch_and_entry_clock(self):
        ds = ["2026-01-%02d" % i for i in range(1, 6)]
        h = {"A": [(ds[0], 100, 100), (ds[1], 100, 110), (ds[3], 132, 132), (ds[4], 132, 132)],
             "B": [(d, 200, 210) for d in ds]}
        trace = []

        def policy(i, holding, entry, deferred):
            trace.append((i, holding, entry, deferred))
            return "A" if i == 0 else "B" if i == 1 else None

        result = run(prepare(h, ds), policy, 0, 4, fee=0, slippage=0,
                     capture_daily=True, capture_trades=True)
        self.assertEqual(trace[2], (2, "A", 1, True))
        self.assertEqual(trace[3], (3, "B", 3, False))
        self.assertAlmostEqual(result["navs"][2], 1.1)
        self.assertAlmostEqual(result["navs"][3], 1.386)
        self.assertEqual(result["trades"][1]["signal_date"], ds[1])

    def test_preparation_preserves_inputs_and_future_changes_do_not_affect_past(self):
        h, calendar, targets = make_random_histories(7)
        original = copy.deepcopy(h)
        frame = prepare(h, calendar)
        policy = lambda i, holding, entry, deferred: targets[i]
        past = run(frame, policy, 0, 30, capture_daily=True)
        self.assertEqual(h, original)
        for code, rows in h.items():
            h[code] = [(d, o * 100, c * 300, v) if d > calendar[30] else (d, o, c, v)
                       for d, o, c, v in rows]
        changed = run(prepare(h, calendar), policy, 0, 30, capture_daily=True)
        self.assertEqual(past["daily"], changed["daily"])
        with self.assertRaises(TypeError):
            frame.opens["A"][0] = 999

    def test_invalid_target_fractional_allocation_and_short_tail_rejected(self):
        h, calendar, _ = make_random_histories(6)
        frame = prepare(h, calendar)
        for bad in ({"A": .5}, "UNKNOWN", 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run(frame, lambda i, holding, entry, deferred: bad, 0, 5)
        h["A"] = h["A"][:-2]
        with self.assertRaisesRegex(ValueError, "coverage"):
            run(prepare(h, calendar), lambda i, holding, entry, deferred: None, 0, len(calendar) - 1)

    def test_real_legacy_and_v92_policies_match_general_engine_every_day(self):
        from v10_next.candidates import get_candidate
        from v10_next.data import Features, load_histories
        from v10_next.legacy import LegacyPolicy
        from v10_round2.v92 import V92Policy, load_qvix
        h = load_histories()
        calendar = [row[0] for row in h["510300"]]
        frame, features = prepare(h, calendar), Features(h)
        qvix = load_qvix()
        candidate = get_candidate("control_v91")
        for kind, start, end in (("legacy", "2015-01-05", "2016-06-30"),
                                 ("v92", "2022-01-04", "2022-06-30")):
            with self.subTest(kind=kind):
                def make_policy():
                    return (LegacyPolicy(candidate, features, calendar) if kind == "legacy"
                            else V92Policy(candidate, features, calendar, qvix))
                original_policy = make_policy()
                general = general_run(h, calendar, original_policy, start, end)
                adapted_policy = make_policy()
                fast = run(frame, legacy_bridge(frame, adapted_policy), start, end,
                           capture_daily=True, capture_trades=True)
                self.assert_parity(fast, general)
                self.assertEqual(original_policy.metadata["trace"], adapted_policy.metadata["trace"])
                self.assertEqual(original_policy.metadata["crash_events"], adapted_policy.metadata["crash_events"])


if __name__ == "__main__":
    unittest.main()
