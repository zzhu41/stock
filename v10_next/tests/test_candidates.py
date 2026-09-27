"""Offline policy regressions: monthly timing, dated features and fixed slots."""
import copy
from datetime import date, timedelta
import unittest

from v10_next.candidates import (CASH, ROBUST_POOL, CANDIDATES,
                                 build_policy, get_candidate, registry)
from v10_next.data import Features
from v10_next.execution import run


class Reader:
    """Small deterministic reader for allocation-rule checks."""
    def __init__(self, values, cash=True):
        self.values, self.cash = values, cash
        self.calls = []

    def __call__(self, day, code, lookbacks):
        self.calls.append((day, code, lookbacks))
        value = self.values.get(code)
        if value is None:
            return None
        returns = value if isinstance(value, dict) else {n: value for n in lookbacks}
        return dict(available=True, returns=returns,
                    score=sum(returns.values()) / len(returns))

    def cash_available(self, day):
        return self.cash


def synthetic_histories():
    """Enough weekday observations to exercise the real 244-bar feature reader."""
    days, day = [], date(2022, 1, 3)
    while day <= date(2024, 2, 5):
        if day.weekday() < 5 and day.isoformat() != "2024-01-01":
            days.append(day.isoformat())
        day += timedelta(days=1)
    current = {"510300": 120.0, "510500": 110.0, "513100": 90.0,
               "518880": 90.0, "511010": 90.0, CASH: 100.0}
    histories = {
        code: [(d, current[code] if d >= "2024-01-02" else 100.0,
                current[code] if d >= "2024-01-02" else 100.0, 1000.0)
               for d in days]
        for code in tuple(ROBUST_POOL) + (CASH,)
    }
    return days, histories


class CandidateTests(unittest.TestCase):
    def policy(self, candidate, reader):
        return build_policy(candidate, legacy_factory=None, momentum_reader=reader)

    def test_first_session_after_month_change_and_no_midmonth_rebalance(self):
        reader = Reader({"510300": .12, "510500": .20, "511010": .03})
        policy = self.policy("r_m12_top2", reader)
        target = policy("2026-08-03", {}, {"previous_date": "2026-07-31"})
        self.assertEqual(target, {"510500": .5, "510300": .5})
        count = len(reader.calls)
        self.assertIsNone(policy("2026-08-04", {}, {
            "previous_date": "2026-08-03", "weights": {"510500": .6, "510300": .4}}))
        self.assertEqual(len(reader.calls), count)
        self.assertIsNone(policy("2026-08-20", {}, {"previous_date": "2026-08-19"}))
        with self.assertRaises(ValueError):
            policy("2026-08-20", {}, {"previous_date": "2026-08-20"})

    def test_one_positive_keeps_one_half_slot_in_cash(self):
        policy = self.policy("r_m12_top2", Reader({"510300": .1, "510500": 0, "511010": -.1}))
        self.assertEqual(policy("2026-08-03", {}, {"previous_date": "2026-07-31"}),
                         {"510300": .5, CASH: .5})

    def test_zero_positive_allocates_cash_and_unavailable_cash_stays_uninvested(self):
        for available, expected in ((True, {CASH: 1.0}), (False, {})):
            with self.subTest(cash_available=available):
                policy = self.policy("r_m12_top2", Reader({"510300": 0, "510500": -.1}, available))
                self.assertEqual(policy("2026-08-03", {}, {"previous_date": "2026-07-31"}), expected)
                event = policy.metadata["rebalance_events"][0]
                self.assertEqual(event["uninvested_cash_weight"], 0.0 if available else 1.0)

    def test_missing_cash_does_not_expand_or_discard_selected_risk_slot(self):
        policy = self.policy("r_m12_top2", Reader({"510300": .1}, cash=False))
        self.assertEqual(policy("2026-08-03", {}, {"previous_date": "2026-07-31"}), {"510300": .5})
        self.assertEqual(policy.metadata["missing_cash_dates"], ["2026-08-03"])
        self.assertEqual(policy.metadata["rebalance_events"][0]["uninvested_cash_weight"], .5)

    def test_broad_sleeves_stay_at_one_fifth_with_cash_residual(self):
        reader = Reader({"510300": .12, "510500": .20, "513100": -.01, "518880": 0, "511010": .03})
        policy = self.policy("r_m12_broad", reader)
        self.assertEqual(policy("2026-08-03", {}, {"previous_date": "2026-07-31"}),
                         {"510500": .2, "510300": .2, "511010": .2, CASH: .4})

    def test_ties_are_deterministic_and_multihorizon_uses_fixed_mean(self):
        policy = self.policy("r_m12_top2", Reader({code: .1 for code in ROBUST_POOL}))
        self.assertEqual(policy("2026-08-03", {}, {"previous_date": "2026-07-31"}),
                         {"510300": .5, "510500": .5})
        reader = Reader({"510300": {61: -.1, 122: .1, 244: .3},
                         "510500": {61: -.3, 122: .1, 244: .1}})
        policy = self.policy("r_multi_top2", reader)
        self.assertEqual(policy("2026-08-03", {}, {"previous_date": "2026-07-31"}),
                         {"510300": .5, CASH: .5})
        self.assertTrue(all(call[2] == (61, 122, 244) for call in reader.calls))

    def test_real_reader_requires_current_bar_and_complete_warmup(self):
        _, histories = synthetic_histories()
        histories["510500"] = [r for r in histories["510500"] if r[0] != "2024-01-02"]
        histories["510300"] = [r for r in histories["510300"] if r[0] <= "2024-01-02"][-244:]
        reader = Features(histories)
        self.assertIsNone(reader("2024-01-02", "510500", (244,)))
        self.assertIsNone(reader("2024-01-02", "510300", (244,)))
        policy = self.policy("r_m12_top2", reader)
        self.assertEqual(policy("2024-01-02", {}, {"previous_date": "2023-12-29"}), {CASH: 1.0})
        self.assertIn("510500", policy.metadata["rebalance_events"][0]["unavailable_assets"])

    def test_future_prices_cannot_change_current_real_features_or_target(self):
        _, histories = synthetic_histories()
        changed = copy.deepcopy(histories)
        for code, rows in changed.items():
            changed[code] = [(d, 1e7, 1e7, v) if d > "2024-01-02" else (d, o, c, v)
                             for d, o, c, v in rows]
        first, second = Features(histories), Features(changed)
        for code in ROBUST_POOL:
            self.assertEqual(first("2024-01-02", code, (61, 122, 244)),
                             second("2024-01-02", code, (61, 122, 244)))
        p1, p2 = self.policy("r_multi_top2", first), self.policy("r_multi_top2", second)
        state = {"previous_date": "2023-12-29"}
        self.assertEqual(p1("2024-01-02", {}, state), p2("2024-01-02", {}, state))

    def test_monthly_signal_executes_next_open_and_does_not_rebalance_daily(self):
        calendar, histories = synthetic_histories()
        reader = Features(histories)
        policy = self.policy("r_m12_top2", reader)
        result = run(histories, calendar, policy, "2024-01-02", "2024-01-05", fee=0, slippage=0)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["signal_date"], "2024-01-02")
        self.assertEqual(result["trades"][0]["date"], "2024-01-03")
        self.assertEqual(result["trades"][0]["target"], {"510300": .5, "510500": .5})

    def test_registration_copies_and_legacy_factory_do_not_mutate_the_protocol(self):
        registered = registry()
        self.assertEqual(len(CANDIDATES), 10)
        registered["candidates"][0]["overrides"]["crash_mom5"] = 99
        self.assertEqual(get_candidate("control_v91")["overrides"], {})
        observed = []

        def factory(candidate):
            observed.append(candidate)
            return lambda day, histories, state: {"513100": candidate["risk_weight"]}

        policy = build_policy("control_v91_half", factory, None)
        self.assertEqual(observed[0]["risk_weight"], .5)
        self.assertEqual(policy("2024-01-02", {}, {}), {"513100": .5})


if __name__ == "__main__":
    unittest.main()
