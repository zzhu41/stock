"""Synthetic-clock and accounting checks; no candidate performance is inspected."""
from copy import deepcopy
from datetime import date, timedelta
import math
import unittest

from v11 import portfolio_controls as controls


def weekdays(count=430):
    current, result = date(2011, 1, 3), []
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


DATES = weekdays()
A, B = controls.RISK_ASSETS[:2]


def histories(rate_a=.004, rate_b=.002):
    rates = {A: rate_a, B: rate_b}
    return {code: [(day, "unused open placeholder", 100 * math.exp(rates.get(code, 0.) * i), 100.)
                   for i, day in enumerate(DATES)] for code in controls.ASSETS}


def config(topk=2, frequency="weekly", horizons=(244,)):
    return next(c for c in controls.registry()["candidates"]
                if c["topk"] == topk and c["frequency"] == frequency and tuple(c["horizons"]) == horizons)


def weekday_index(weekday, start=300):
    return next(i for i in range(start, len(DATES)) if date.fromisoformat(DATES[i]).weekday() == weekday)


class LongTrendControlTests(unittest.TestCase):
    def test_registry_is_exactly_eight_fixed_original_pool_controls(self):
        first = controls.registry()
        self.assertEqual(first["count"], 8)
        self.assertEqual(len({c["hash"] for c in first["candidates"]}), 8)
        self.assertEqual({tuple(c["horizons"]) for c in first["candidates"]}, {(244,), (61, 122, 244)})
        self.assertTrue(all(c["minimum_observations"] == 270 and not c["crash_channels"]
                            and not c["leverage"] for c in first["candidates"]))
        first["candidates"][0]["risk_assets"].clear()
        self.assertEqual(len(controls.registry()["candidates"][0]["risk_assets"]), 10)

    def test_own_quote_momentum_has_exact_exponential_rate_and_270_observation_gate(self):
        hs = histories()
        own_dates = [day for i, day in enumerate(DATES) if i != 100]
        hs[B] = [(day, None, 100 * math.exp(.002 * i), 1.) for i, day in enumerate(own_dates)]
        prepared = controls.prepare(hs, DATES)
        for horizons in controls.HORIZONS:
            self.assertTrue(math.isnan(prepared["scores"][horizons][A][268]))
            self.assertAlmostEqual(prepared["scores"][horizons][A][269], 244 * .004, places=12)
            self.assertTrue(math.isnan(prepared["scores"][horizons][B][269]))
            self.assertAlmostEqual(prepared["scores"][horizons][B][270], 244 * .002, places=12)
        self.assertTrue(math.isnan(prepared["prices"][B][100]))

    def test_top2_unfilled_slot_uses_cash_etf_not_leverage_or_implicit_cash(self):
        prepared = controls.prepare(histories(rate_b=-.002), DATES)
        result = controls.run_control(config(), prepared, DATES[300], DATES[300], fee=.0011)
        self.assertEqual(result["signal_target"][0][1], {A: .5, controls.CASH: .5})
        self.assertEqual(result["nav"], 1.)
        self.assertEqual(result["final_state"]["cash"], 0.)
        self.assertAlmostEqual(result["final_state"]["weights"][A], .5)
        flat = controls.prepare(histories(rate_a=0., rate_b=0.), DATES)
        cash = controls.run_control(config(), flat, DATES[300], DATES[300])
        self.assertEqual(cash["signal_target"][0][1], {controls.CASH: 1.})

    def test_initial_exception_then_nonclock_days_keep_units_and_drift(self):
        prepared = controls.prepare(histories(), DATES)
        start = weekday_index(1)  # Tuesday is deliberately not the week's first observation.
        result = controls.run_control(config(), prepared, start, start + 2, fee=.0011)
        self.assertTrue(result["policy_trace"][0]["initial_allocation"])
        self.assertEqual([item[1] for item in result["signal_target"]][1:], [None, None])
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["diagnostics"]["total_model_fee"], 0.)
        self.assertGreater(result["final_state"]["weights"][A], .5)
        self.assertLess(result["final_state"]["weights"][B], .5)
        self.assertAlmostEqual(result["final_state"]["units"][A], .5 / prepared["prices"][A][start])

    def test_clock_rebalance_charges_actual_drifted_weight_turnover(self):
        prepared = controls.prepare(histories(), DATES)
        start = weekday_index(3)  # Thursday, then Friday and Monday.
        end = weekday_index(0, start + 1)
        fee = .0011
        result = controls.run_control(config(), prepared, start, end, fee=fee)
        self.assertEqual(len(result["trades"]), 2)
        last = result["trades"][-1]
        self.assertEqual(last["date"], DATES[end])
        ratios = {c: prepared["prices"][c][end] / prepared["prices"][c][start] for c in (A, B)}
        pre_nav = .5 * (ratios[A] + ratios[B])
        weights = {c: .5 * ratios[c] / pre_nav for c in (A, B)}
        turnover = sum(abs(.5 - weights[c]) for c in (A, B))
        self.assertGreater(last["model_fee"], 0.)
        self.assertAlmostEqual(last["turnover"], turnover, places=12)
        self.assertAlmostEqual(result["nav"], pre_nav * (1 - fee * turnover), places=12)

    def test_lag1_uses_previous_signal_but_current_execution_price(self):
        hs = histories(rate_b=0.)
        today = weekday_index(1)
        hs[A][today] = (DATES[today], "unusable", 1., 100.)
        prepared = controls.prepare(hs, DATES)
        delayed = controls.run_control(config(topk=1), prepared, today, today, lag=1)
        immediate = controls.run_control(config(topk=1), prepared, today, today, lag=0)
        self.assertEqual(delayed["signal_target"][0][1], {A: 1.})
        self.assertEqual(delayed["final_state"]["units"][A], 1.)
        self.assertEqual(immediate["signal_target"][0][1], {controls.CASH: 1.})
        self.assertEqual(delayed["policy_trace"][0]["signal_date"], DATES[today - 1])

    def test_lag1_delays_the_clock_and_iso_week_boundary_uses_no_future_day(self):
        prepared = controls.prepare(histories(), DATES)
        start = weekday_index(1)
        monday = weekday_index(0, start + 1)
        result = controls.run_control(config(), prepared, start, monday + 1, lag=1)
        trace = {row["date"]: row for row in result["policy_trace"]}
        self.assertFalse(trace[DATES[monday]]["rebalance"])
        self.assertTrue(trace[DATES[monday + 1]]["rebalance"])
        self.assertEqual(trace[DATES[monday + 1]]["signal_date"], DATES[monday])
        dates = ["2019-12-30", "2020-01-02", "2020-01-06"]
        hs = {c: [(d, 0, 100., 1.) for d in dates] for c in controls.ASSETS}
        frame = controls.prepare(hs, dates)
        weekly = controls.make_policy(config(), frame, 0)
        self.assertIsNotNone(weekly(0, {}))
        self.assertIsNone(weekly(1, {}))  # Same ISO week despite the changed calendar year.
        self.assertIsNotNone(weekly(2, {}))
        monthly = controls.make_policy(config(frequency="monthly"), frame, 0)
        self.assertIsNotNone(monthly(0, {}))
        self.assertIsNotNone(monthly(1, {}))
        self.assertIsNone(monthly(2, {}))

    def test_missing_cash_quote_atomically_blocks_slot_allocation(self):
        start = weekday_index(2)
        hs = histories(rate_b=0.)
        hs[controls.CASH] = [row for row in hs[controls.CASH] if row[0] != DATES[start]]
        prepared = controls.prepare(hs, DATES)
        result = controls.run_control(config(), prepared, start, start + 1)
        self.assertEqual(result["diagnostics"]["blocked_rebalance_count"], 1)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["final_state"]["units"], {})
        self.assertEqual(result["final_state"]["cash"], 1.)

    def test_future_prices_cannot_revise_any_score_or_target_prefix(self):
        hs = histories()
        altered = deepcopy(hs)
        cut = 310
        for code in controls.ASSETS:
            altered[code] = [row if row[0] <= DATES[cut] else (row[0], row[1], row[2] * 100, row[3])
                             for row in altered[code]]
        first, second = controls.prepare(hs, DATES), controls.prepare(altered, DATES)
        for horizons in controls.HORIZONS:
            for code in controls.RISK_ASSETS:
                for a, b in zip(first["scores"][horizons][code][:cut + 1], second["scores"][horizons][code][:cut + 1]):
                    if math.isnan(a):
                        self.assertTrue(math.isnan(b))
                    else:
                        self.assertEqual(a, b)
        a = controls.run_control(config(horizons=(61, 122, 244)), first, 300, cut, lag=1)
        b = controls.run_control(config(horizons=(61, 122, 244)), second, 300, cut, lag=1)
        self.assertEqual(a["daily"], b["daily"])
        self.assertEqual(a["policy_trace"], b["policy_trace"])


if __name__ == "__main__":
    unittest.main()
