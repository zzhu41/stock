"""Selection must depend only on frozen development data and constraints."""
from copy import deepcopy
import unittest

from v10_next.candidates import CANDIDATES
from v10_next.research import selection


def synthetic_path(pattern, cash=False):
    nav, daily = 1.0, []
    for year in range(2014, 2022):
        for suffix, ret in zip(("01-02", "04-01", "07-01", "10-01"), pattern):
            nav *= 1 + ret
            daily.append(("%d-%s" % (year, suffix), nav,
                          {"511880" if cash else "510300": 1.0}))
    return dict(daily=daily, nav=nav, trades=[],
                diagnostics=dict(deferred_rebalances=[], missing_held_bars=[]))


def fixture():
    base = synthetic_path((.002, -.001, .002, .001))
    results = {c["id"]: deepcopy(base) for c in CANDIDATES}
    results["h_no_crash"] = synthetic_path((.003, -.001, .003, .001))
    results["h_no_bear_gate"] = synthetic_path((.0021, -.001, .0021, .001))
    results["h_ensemble"] = synthetic_path((.0022, -.001, .0022, .001))
    results["h_core_pool"] = synthetic_path((.0018, -.001, .0018, .001))
    results["r_m12_top2"] = synthetic_path((.001, -.0005, .0015, .0005))
    results["r_multi_top2"] = synthetic_path((.0011, -.0005, .0016, .0006))
    results["r_m12_broad"] = synthetic_path((.0008, -.0004, .0012, .0004))
    cash = synthetic_path((.0002,) * 4, cash=True)
    return results, cash


def append_future(result, first_factor, later_factor):
    out = deepcopy(result)
    nav = out["daily"][-1][1]
    first = True
    for year in range(2022, 2027):
        for suffix in ("01-04", "04-01", "07-01", "09-11"):
            nav *= first_factor if first else later_factor
            first = False
            date = "%d-%s" % (year, suffix)
            out["daily"].append((date, nav, {"510300": 1.0}))
            # Future turnover and execution problems also must not change eligibility.
            out["trades"].append(dict(date=date, fills=[{}] * 6, turnover=8.0))
            out["diagnostics"]["deferred_rebalances"].append(dict(date=date))
            out["diagnostics"]["missing_held_bars"].append(dict(date=date))
    out["nav"] = nav
    return out


class SelectionProtocolTests(unittest.TestCase):
    def test_selection_has_eligible_development_winners(self):
        results, cash = fixture()
        picked = selection(results, cash)
        self.assertEqual(picked["high_return"], "h_no_crash")
        self.assertEqual(picked["robust"], "r_multi_top2")

    def test_future_returns_turnover_and_failures_cannot_change_selection(self):
        results, cash = fixture()
        expected = selection(results, cash)
        for reverse in (False, True):
            changed = {}
            for i, (name, path) in enumerate(results.items()):
                losing = (i % 2 == 0) != reverse
                changed[name] = append_future(path, .1 if losing else 4.0,
                                              .98 if losing else 1.08)
            changed_cash = append_future(cash, 20.0 if reverse else .01, 1.10)
            with self.subTest(reverse=reverse):
                # Compare all reported eligibility metrics, not just the winner IDs.
                self.assertEqual(selection(changed, changed_cash), expected)

    def test_no_robust_candidate_above_cash_returns_none(self):
        results, cash = fixture()
        for candidate in CANDIDATES:
            if candidate["family"] == "robust":
                results[candidate["id"]] = deepcopy(cash)
        picked = selection(results, cash)
        self.assertIsNone(picked["robust"])
        self.assertTrue(all(not row["checks"]["above_cash_both_segments"]
                            for row in picked["robust_constraints"]))

    def test_zero_drawdown_cannot_win_with_infinite_calmar(self):
        results, cash = fixture()
        for candidate in CANDIDATES:
            if candidate["family"] == "robust":
                results[candidate["id"]] = synthetic_path((.003,) * 4)
        picked = selection(results, cash)
        self.assertIsNone(picked["robust"])
        self.assertTrue(all(row["checks"]["above_cash_both_segments"]
                            for row in picked["robust_constraints"]))
        self.assertTrue(all(not row["checks"]["nonzero_drawdowns"]
                            and row["score"] is None for row in picked["robust_constraints"]))

    def test_no_better_high_candidate_falls_back_to_v91(self):
        results, cash = fixture()
        for candidate in CANDIDATES:
            if candidate["family"] == "high_return":
                results[candidate["id"]] = deepcopy(cash)
        picked = selection(results, cash)
        self.assertEqual(picked["high_return"], "control_v91")
        self.assertFalse(picked["high_return_improved_development"])


if __name__ == "__main__":
    unittest.main()
