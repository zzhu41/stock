"""Synthetic selection/fee regressions; no registered-market-data search."""
from copy import deepcopy
import unittest

import numpy as np

from v10_h_close import scan
from v10_h_close.engine import prepare, run
from v10_h_close.registry import controls


def row(name, cagr=.60, dd=-.32, recent=.38, fee=.53, switches=10):
    return dict(id=name, metrics={"selection": {"cagr": cagr, "max_dd": dd},
                "recent_2022_2025": {"cagr": recent}, "full": {"cagr": 999},
                "report_only_2026": {"cagr": 999}},
                fee5={"selection": {"cagr": fee}, "full": {"cagr": 999}},
                selection_switches=switches, switches=999999)


def baseline():
    return row("c_v92", cagr=.50, dd=-.30, recent=.40, fee=.46)


def config(name, external=0, rules=4):
    return dict(id=name, channels=["deep"] + ["qvix", "volume"][:external],
                complexity={"active_optional_rule_count": rules})


class Guarded(dict):
    def __init__(self, values, allowed):
        super().__init__(values)
        self.allowed = set(allowed)

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError("Selection accessed an excluded field: " + key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self.allowed:
            raise AssertionError("Selection accessed an excluded field: " + key)
        return super().get(key, default)


def protect_selection_fields(value):
    value = deepcopy(value)
    value["metrics"] = Guarded(value["metrics"], ("selection", "recent_2022_2025"))
    value["fee5"] = Guarded(value["fee5"], ("selection",))
    return Guarded(value, ("id", "metrics", "fee5", "selection_switches"))


D = ("2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06")


def synthetic_frame():
    closes = {"A": (100, 110, 121, 133.1, 146.41, 161.051),
              "B": (100, 100, 112, 120, 134.4, 140),
              "C": (100, 100, 100, 100, 100, 100)}
    histories = {code: [(d, values[i], values[i], 100.) for i, d in enumerate(D)
                         if not (code == "C" and i == 2) and not (code == "B" and i == 3)]
                 for code, values in closes.items()}
    return prepare(histories, D)


def execute(frame, targets, fee=.0001, start=None):
    return run(frame, lambda i, holding, age, can_sell: targets.get(i),
               start or D[0], D[-1], fee=fee, capture_daily=True, capture_trades=True)


class SelectionTests(unittest.TestCase):
    def test_primary_is_maximum_cagr_even_when_risk_guards_fail(self):
        risky = row("highest", cagr=.90, dd=-.95, recent=-.5, fee=.10)
        safe = row("guarded", cagr=.60)
        selected = scan.select([safe, risky], [config("highest"), config("guarded")], baseline())
        self.assertEqual(selected["primary_return_champion"], "highest")
        self.assertTrue(selected["primary_above_v92"])
        self.assertEqual(selected["risk_guarded_candidate"], "guarded")
        self.assertFalse(selected["risk_checks"]["highest"]["drawdown"])

    def test_guarded_has_five_independent_tests_and_no_eligible_means_none(self):
        cases = {
            "above_v92": row("fail", cagr=.50, dd=-.30, recent=.40, fee=.46),
            "drawdown": row("fail", dd=-.36),
            "recent": row("fail", recent=.34),
            "fee_retention": row("fail", fee=.47),
            "fee_relative": row("fail", cagr=.52, fee=.42),
        }
        for failed, value in cases.items():
            with self.subTest(failed=failed):
                selected = scan.select([value], [config("fail")], baseline())
                self.assertEqual({key for key, passed in selected["risk_checks"]["fail"].items() if not passed}, {failed})
                self.assertIsNone(selected["risk_guarded_candidate"])
                self.assertEqual(selected["risk_eligible_count"], 0)
                self.assertEqual(selected["primary_return_champion"], "fail")
        selected = scan.select([row("below", cagr=.49, fee=.45)], [config("below")], baseline())
        self.assertFalse(selected["primary_above_v92"])

    def test_guard_boundaries_are_inclusive_except_strictly_above_reference(self):
        reference = baseline()
        reference["fee5"]["selection"]["cagr"] = .50
        boundary = row("boundary", cagr=.60, dd=-.35, recent=.35, fee=.48)
        selected = scan.select([boundary], [config("boundary")], reference)
        self.assertTrue(all(selected["risk_checks"]["boundary"].values()))
        self.assertEqual(selected["risk_guarded_candidate"], "boundary")

    def test_ties_use_external_channels_then_rule_count_then_pre2026_switches_then_id(self):
        specs = [config("external_two", 2, 1), config("external_one", 1, 1),
                 config("many_rules", 0, 6), config("many_switches", 0, 4),
                 config("id_b", 0, 4), config("id_a", 0, 4)]
        rows = [row(c["id"], switches=1 if c["id"].startswith("external") or c["id"] == "many_rules"
                    else 10 if c["id"] == "many_switches" else 3) for c in specs]
        selected = scan.select(rows, specs, baseline())
        expected = ["id_a", "id_b", "many_switches", "many_rules", "external_one", "external_two"]
        self.assertEqual(selected["top20_return_ids"], expected)
        self.assertEqual(selected["primary_return_champion"], "id_a")
        self.assertEqual(selected["risk_guarded_candidate"], "id_a")
        for c in controls():
            self.assertIn("active_optional_rule_count", c["complexity"])

    def test_selection_does_not_read_full_2026_or_all_period_switches(self):
        values = [row("first", cagr=.61, switches=5), row("second", cagr=.60, switches=1)]
        specs = [config(c["id"]) for c in values]
        guarded = [protect_selection_fields(c) for c in values]
        selected = scan.select(guarded, specs, protect_selection_fields(baseline()))
        plain = scan.select(values, specs, baseline())
        self.assertEqual(selected, plain)
        for i, value in enumerate(values):
            value["metrics"]["full"] = {"cagr": (-1 if i == 0 else 1e10)}
            value["metrics"]["report_only_2026"] = {"cagr": (1e10 if i == 0 else -1)}
            value["fee5"]["full"] = {"cagr": 1e20}
            value["switches"] = 0 if i else 10**9
        self.assertEqual(scan.select(values, specs, baseline()), selected)

    def test_top20_keeps_return_ranking_without_filtering_to_guarded_candidates(self):
        values = [row("candidate_%02d" % i, cagr=.60 + i * .001, dd=-.99) for i in range(25)]
        selected = scan.select(list(reversed(values)), [config(c["id"]) for c in values], baseline())
        self.assertEqual(selected["top20_return_ids"], ["candidate_%02d" % i for i in range(24, 4, -1)])
        self.assertIsNone(selected["risk_guarded_candidate"])


class FeeAndPeriodTests(unittest.TestCase):
    def test_fee_reprice_matches_fresh_fee5_execution_with_blocked_switches_and_missing_bars(self):
        frame = synthetic_frame()
        targets = {0: "A", 1: "B", 2: "C", 3: "A", 4: "A"}
        base = execute(frame, targets)
        fresh = execute(frame, targets, fee=.0005)
        self.assertEqual(base["holdings"], ["A", "B", "B", "B", "A", "A"])
        self.assertEqual(base["holdings"], fresh["holdings"])
        self.assertEqual(base["switches"], 2)
        self.assertEqual(base["diagnostics"]["blocked_switch_days"], 2)
        self.assertEqual(base["diagnostics"]["missing_held_days"], 1)
        self.assertEqual(base["navs"][0], 1.0)
        self.assertEqual(fresh["navs"][0], 1.0)
        np.testing.assert_allclose(scan.fee_reprice(base), fresh["navs"], rtol=1e-13, atol=1e-13)
        ratio = (1 - 2 * .0005) / (1 - 2 * .0001)
        np.testing.assert_allclose(np.asarray(fresh["navs"]) / base["navs"],
                                   [1., ratio, ratio, ratio, ratio**2, ratio**2], rtol=1e-13)

    def test_free_first_evaluation_session_is_not_a_free_later_first_entry(self):
        frame = synthetic_frame()
        later = execute(frame, {1: "B"})
        high = execute(frame, {1: "B"}, fee=.0005)
        self.assertEqual(later["navs"][0], 1.0)
        self.assertEqual(later["navs"][1], .9998)
        self.assertEqual(high["navs"][1], .999)
        self.assertEqual(later["switches"], 1)
        np.testing.assert_allclose(scan.fee_reprice(later), high["navs"], rtol=1e-13)
        middle = execute(frame, {2: "A", 4: "B"}, start=D[2])
        middle_high = execute(frame, {2: "A", 4: "B"}, fee=.0005, start=D[2])
        self.assertEqual(middle["navs"][0], 1.0)  # Global index 2, but local first session is free.
        self.assertEqual(middle["switches"], 1)
        np.testing.assert_allclose(scan.fee_reprice(middle), middle_high["navs"], rtol=1e-13)

    def test_fee_reprice_rejects_inconsistent_actual_switch_count(self):
        result = execute(synthetic_frame(), {0: "A", 1: "B"})
        result["switches"] += 1
        with self.assertRaisesRegex(AssertionError, "switch count"):
            scan.fee_reprice(result)

    def test_collect_switch_tie_counts_stop_at_2025_and_ignore_blocked_targets(self):
        frame = synthetic_frame()
        first = execute(frame, {0: "A", 1: "B", 2: "C", 3: "A", 4: "A"})
        changed = execute(frame, {0: "A", 1: "B", 2: "C", 3: "A", 4: "C", 5: "A"})
        candidate = dict(config("fixture"), candidate_hash="synthetic", stages=["fixture"])
        before = scan.collect(candidate, first, D)[0]
        after = scan.collect(candidate, changed, D)[0]
        self.assertEqual(before["selection_switches"], 1)
        self.assertEqual(after["selection_switches"], 1)
        self.assertEqual((before["switches"], after["switches"]), (2, 3))
        self.assertEqual(before["metrics"]["selection"], after["metrics"]["selection"])
        self.assertEqual(before["fee5"]["selection"], after["fee5"]["selection"])
        self.assertEqual(scan.rank_key(before, candidate), scan.rank_key(after, candidate))

    def test_yearly_and_recent_period_keep_the_first_day_return_and_cache_only_dates(self):
        dates = ("2021-12-31", "2022-01-04", "2025-12-31", "2026-01-02", "2026-01-05")
        nav = np.asarray([1.1, 1.21, 1.452, .9, 1.8])
        scan.date_slices.cache_clear()
        first, returns = scan.summarize_nav(nav, dates)
        self.assertAlmostEqual(first["selection"]["nav_factor"], 1.452)
        self.assertAlmostEqual(first["recent_2022_2025"]["nav_factor"], 1.32)
        self.assertAlmostEqual(first["yearly"]["2022"], .10)
        self.assertAlmostEqual(first["yearly"]["2025"], .20)
        self.assertAlmostEqual(first["yearly"]["2026"], 1.8 / 1.452 - 1)
        self.assertAlmostEqual(float(np.prod([1 + value for value in first["yearly"].values()])), nav[-1])
        hits = scan.date_slices.cache_info().hits
        altered = nav.copy()
        altered[3:] = [10., 20.]
        later, _ = scan.summarize_nav(altered, dates)
        self.assertGreater(scan.date_slices.cache_info().hits, hits)
        self.assertEqual(first["selection"], later["selection"])
        self.assertEqual(first["recent_2022_2025"], later["recent_2022_2025"])
        self.assertNotEqual(first["report_only_2026"], later["report_only_2026"])
        shortened, _ = scan.summarize_nav(nav[:-1], dates[:-1])
        self.assertAlmostEqual(shortened["yearly"]["2026"], .9 / 1.452 - 1)
        self.assertEqual(scan.date_slices.cache_info().currsize, 2)


if __name__ == "__main__":
    unittest.main()
