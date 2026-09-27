"""Literal old-close accounting parity; no production files or network I/O."""
import random
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from v10_h_close.engine import prepare, run


def dates(n):
    return [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]


class OldCloseEngineTests(unittest.TestCase):
    def test_first_session_free_then_linear_double_fee_and_close_prices_only(self):
        ds = dates(3)
        h = {"A": [(ds[0], 1, 100), (ds[1], 999, 110), (ds[2], 1, 95)],
             "B": [(ds[0], 999, 200), (ds[1], 1, 200), (ds[2], 999, 220)]}
        result = run(prepare(h, ds), lambda i, h, age, sell: "A" if i == 0 else "B",
                     0, 2, capture_daily=True, capture_trades=True)
        self.assertEqual(result["navs"][0], 1.0)
        self.assertAlmostEqual(result["navs"][1], 1.1 * .9998)
        self.assertAlmostEqual(result["navs"][2], 1.21 * .9998)
        self.assertEqual(result["switches"], 1)
        self.assertEqual(result["entry_count"], 2)
        self.assertEqual(result["trades"], [(ds[1], "A", "B", result["navs"][1])])
        self.assertEqual(result["returns"][0], 0.0)

    def test_missing_held_price_carries_mark_and_discards_unfilled_instruction(self):
        ds = dates(4)
        h = {"A": [(ds[0], 100, 100), (ds[2], 130, 130), (ds[3], 140, 140)],
             "B": [(d, 100, 100) for d in ds]}
        seen = []

        def policy(i, holding, age, can_sell):
            seen.append((i, holding, age, can_sell))
            return "B" if i == 1 else "A"

        result = run(prepare(h, ds), policy, ds[0], ds[-1], capture_daily=True, capture_trades=True)
        self.assertEqual(seen[1], (1, "A", 1, False))
        self.assertEqual(seen[2], (2, "A", 2, True))
        self.assertEqual(result["holdings"], ["A"] * 4)
        self.assertEqual(result["navs"], [1.0, 1.0, 1.3, 1.4])
        self.assertEqual(result["switches"], 0)
        self.assertEqual(result["diagnostics"]["blocked_switch_days"], 1)
        self.assertFalse(result["diagnostics"]["order_carryover"])

    def test_missing_target_does_not_fill_and_holding_age_resets_only_on_real_switch(self):
        ds = dates(4)
        h = {"A": [(d, 100, 100 + 10 * i) for i, d in enumerate(ds)],
             "B": [(ds[0], 100, 100), (ds[2], 100, 100), (ds[3], 100, 100)]}
        seen = []

        def policy(i, holding, age, can_sell):
            seen.append((holding, age))
            return "A" if i == 0 else "B"

        result = run(prepare(h, ds), policy, 0, 3, capture_trades=True)
        self.assertEqual(result["holdings"], ["A", "A", "B", "B"])
        self.assertEqual(seen, [(None, 0), ("A", 1), ("A", 2), ("B", 1)])
        self.assertEqual(result["trades"][0][:3], (ds[2], "A", "B"))

    def test_crash_callback_can_lock_only_when_held_asset_is_sellable(self):
        ds = dates(9)
        h = {"A": [(d, 100, 100) for i, d in enumerate(ds) if i != 1],
             "B": [(d, 100, 100 * 1.1 ** max(0, i - 2)) for i, d in enumerate(ds)],
             "511880": [(d, 100, 100) for d in ds]}
        locks = []
        state = {"until": -1}

        def policy(i, holding, age, can_sell):
            if i < state["until"]:
                return holding
            if i in (1, 2) and can_sell:
                state["until"] = i + 5
                locks.append(i)
                return "B"
            return "A" if i < 2 else "511880"

        result = run(prepare(h, ds), policy, 0, 8, fee=0, capture_trades=True)
        self.assertEqual(locks, [2])
        self.assertEqual([t[0] for t in result["trades"]], [ds[2], ds[7]])
        self.assertAlmostEqual(result["nav"], 1.1 ** 5)

    def test_array_and_streaming_modes_share_identical_accounting(self):
        ds = dates(5)
        h = {"A": [(d, 100, 100 + i) for i, d in enumerate(ds)]}
        frame = prepare(h, ds)
        policy = lambda i, holding, age, can_sell: "A"
        full = run(frame, policy, 0, 4, capture_daily=True)
        array = run(frame, policy, 0, 4)
        stream = run(frame, policy, 0, 4, capture_daily=False)
        self.assertEqual(array["navs"], full["navs"])
        self.assertEqual(array["returns"], full["returns"])
        self.assertIsNone(array["daily"])
        self.assertIsNone(stream["navs"])
        for key in ("nav", "ann", "max_dd", "sharpe", "switches", "yearly"):
            self.assertEqual(full[key], stream[key])

    def test_random_scripted_targets_match_original_root_backtest(self):
        import backtest
        import strategy
        from strategy_versions import backtest_kwargs
        for seed in range(10):
            rng, ds = random.Random(seed), dates(80)
            h = {"511880": [(d, 100, 100) for d in ds]}
            for code in ("A", "B"):
                price, rows = 100.0, []
                for i, d in enumerate(ds):
                    price *= 1 + rng.uniform(-.08, .08)
                    if i not in (0, len(ds) - 1) and rng.random() < .14:
                        continue
                    rows.append((d, price * 1.2, price))
                h[code] = rows
            choices = [rng.choice(("A", "B", "511880")) for _ in ds]
            low, high = 3, 76
            choices[low] = "511880"  # Ensure the original engine can establish its first holding.
            positions = {d: i for i, d in enumerate(ds)}
            present = {c: {r[0] for r in rows} for c, rows in h.items()}
            clock, native_calls, own_calls = {}, [], []

            def rank(hs, on_date=None):
                clock["i"] = positions[on_date]
                return []

            def decide(table, holding, age):
                i = clock["i"]
                native_calls.append((i, holding, age, holding is None or ds[i] in present[holding]))
                return choices[i], "scripted execution check"

            def policy(i, holding, age, can_sell):
                own_calls.append((i, holding, age, can_sell))
                return choices[i]

            fee = (0, .0001, .001)[seed % 3]
            with self.subTest(seed=seed), patch.object(strategy, "rank", side_effect=rank), \
                    patch.object(strategy, "decide", side_effect=decide), patch.object(backtest, "FEE", fee):
                expected = backtest.backtest(h, ds, start=ds[low], end=ds[high],
                                             **backtest_kwargs("v9.1", crash_mom5=0))
                actual = run(prepare(h, ds), policy, low, high, fee=fee,
                             capture_daily=True, capture_trades=True)
            self.assertEqual(actual["daily"], expected["daily"])
            self.assertEqual(actual["trades"], expected["trades"])
            self.assertEqual(actual["switches"], expected["switches"])
            self.assertEqual(own_calls, native_calls)

    def test_real_full_period_matches_root_v9_v91_and_old_lab_v92(self):
        import backtest
        import strategy
        import v10.lab as lab
        from strategy_versions import backtest_kwargs
        from v10_next.data import load_histories
        from v10_next.frozen.metadata import STOCK_POOL, GLOBAL_POOL, CASH, GOLD
        from v10_round2.v92 import load_qvix
        h = load_histories()
        h = {code: h[code] for code in list(STOCK_POOL) + list(GLOBAL_POOL) + [GOLD, CASH]}
        calendar = [r[0] for r in h["510300"]]
        frame, native_rank, cache = prepare(h, calendar), strategy.rank, {}
        qvix = load_qvix()
        fear = dict(qd=[], qz=[], qv=[], rd=[], r5=[])
        for d, value in qvix.rows:
            observation = qvix.state(d)
            if observation["available"]:
                fear["qd"].append(d)
                fear["qz"].append(observation["z"])
                fear["qv"].append(value)

        def cached_rank(hs, on_date=None, live_prices=None):
            lab.STATE["last_date"] = on_date
            if on_date not in cache:
                saved = lab.CFG.get("crash_volu")
                lab.CFG["crash_volu"] = 2.0  # Cache a continuous ratio, never a crash decision.
                try:
                    cache[on_date] = native_rank(hs, on_date=on_date)
                finally:
                    if saved is None:
                        lab.CFG.pop("crash_volu", None)
                    else:
                        lab.CFG["crash_volu"] = saved
            return [(c, dict(ind)) for c, ind in cache[on_date]]

        start, end = "2014-01-01", "2026-09-11"
        with patch.object(strategy, "rank", side_effect=cached_rank), \
                patch.object(strategy, "STOCK_POOL", list(STOCK_POOL)), \
                patch.object(strategy, "GLOBAL_POOL", list(GLOBAL_POOL)), \
                patch.dict(lab.FEAR, fear, clear=True), patch.object(backtest, "FEE", .0001):
            for version in ("v9", "v9.1", "v9.2"):
                cfg = (dict(fear_qz=2.5, fear_m5=-.04, crash_volu=2.0, cv_m5=-.04, cv_dep=.10)
                       if version == "v9.2" else {})
                with self.subTest(version=version), patch.dict(lab.CFG, cfg, clear=True):
                    expected = (lab.backtest_v10(h, calendar, start, end) if version == "v9.2" else
                                backtest.backtest(h, calendar, start=start, end=end, **backtest_kwargs(version)))
                    strategy._state_bull = None
                    strategy.BULL_HYST_PENDING = None
                    lock, events = {"until": -1}, []

                    def policy(i, holding, age, can_sell):
                        d = calendar[i]
                        table = cache[d]
                        target, _ = strategy.decide(table, holding, age)
                        if i < lock["until"]:
                            return holding
                        fear_today = version == "v9.2" and qvix.state(d)["active"]
                        for code, ind in table:
                            if code == holding or code not in set(STOCK_POOL) | set(GLOBAL_POOL) | {GOLD}:
                                continue
                            deep = ind["mom5"] <= -.08 and ind["dist_ma250"] < -.20
                            extra = version == "v9.2" and (
                                (fear_today and ind["mom5"] <= -.04 and ind["dist_ma250"] < -.20)
                                or (ind["vol_ratio20"] >= 2 and ind["mom5"] <= -.04 and ind["dist_ma250"] < -.10))
                            if (deep or extra) and can_sell:
                                target, lock["until"] = code, i + 5
                                events.append((d, code))
                                break
                        return target

                    actual = run(frame, policy, start, end, capture_daily=True, capture_trades=True)
                    self.assertEqual(actual["daily"], expected["daily"])
                    self.assertEqual(actual["trades"], expected["trades"])
                    self.assertEqual(events, expected["crash_buys"])
                    for key in ("nav", "ann", "max_dd", "switches"):
                        self.assertEqual(actual[key], expected[key])
                    self.assertAlmostEqual(actual["sharpe"], expected["sharpe"], places=12)


if __name__ == "__main__":
    unittest.main()
