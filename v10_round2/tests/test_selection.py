"""Round-two objective boundaries and separation of selection from validation."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from v10_round2 import research
from v10_round2.registry import CANDIDATES, HIGH_RETURN_IDS, STABILITY_IDS


def metric_fixture():
    results = {c["id"]: {"metric": dict(cagr=.20, max_dd=-.35, turnover=10.0)}
               for c in CANDIDATES}
    results["c_v92"]["metric"] = dict(cagr=.40, max_dd=-.30, turnover=10.0)
    return results


def path(pattern):
    nav, daily = 1.0, []
    for year in range(2014, 2022):
        for suffix, ret in zip(("01-02", "04-01", "07-01", "10-01"), pattern):
            nav *= 1 + ret
            daily.append(("%d-%s" % (year, suffix), nav, {"510300": 1.0}))
    return dict(daily=daily, nav=nav, trades=[],
                diagnostics=dict(deferred_rebalances=[], missing_held_bars=[]))


def paths_fixture():
    results = {c["id"]: path((.002, -.001, .002, .001)) for c in CANDIDATES}
    results["h_qvix_only"] = path((.003, -.001, .003, .001))
    results["h_v91_no_overheat"] = path((.0024, -.001, .0024, .001))
    for i, name in enumerate(STABILITY_IDS):
        results[name] = path((.0025, -.001 + i * .0001, .0025, .0015))
    return results


def append_future(result, winner):
    result = deepcopy(result)
    nav = result["nav"]
    first = True
    for year in range(2022, 2027):
        for suffix in ("01-04", "04-01", "07-01", "09-11"):
            nav *= (4.0 if winner else .1) if first else (1.08 if winner else .98)
            first = False
            date = "%d-%s" % (year, suffix)
            result["daily"].append((date, nav, {"510300": 1.0}))
            result["trades"].append(dict(date=date, fills=[{}] * 4, turnover=12.0))
            result["diagnostics"]["deferred_rebalances"].append(dict(date=date))
            result["diagnostics"]["missing_held_bars"].append(dict(date=date))
    result["nav"] = nav
    return result


class RoundTwoSelectionTests(unittest.TestCase):
    def select_metrics(self, results):
        calls = []
        def summarize(result, start, end):
            calls.append((start, end))
            return deepcopy(result["metric"])
        with patch.object(research, "summarize", side_effect=summarize):
            picked = research.select(results)
        self.assertTrue(calls)
        self.assertTrue(all(bounds == (research.START, research.TRAIN_END) for bounds in calls))
        return picked

    def test_h_must_strictly_beat_v92_and_no_eligible_means_none(self):
        results = metric_fixture()
        for name in HIGH_RETURN_IDS:
            results[name]["metric"] = dict(cagr=.40, max_dd=-.30, turnover=1.0)
        picked = self.select_metrics(results)
        self.assertIsNone(picked["high_return"])
        self.assertIsNone(picked["robust"])
        self.assertTrue(all(not r["checks"]["cagr_above_v92"] for r in picked["high_rows"]))

    def test_h_drawdown_allows_two_percentage_points_but_not_more(self):
        results = metric_fixture()
        results["h_qvix_only"]["metric"] = dict(cagr=.41, max_dd=-.32, turnover=10.0)
        results["h_v91_no_overheat"]["metric"] = dict(cagr=.60, max_dd=-.320001, turnover=1.0)
        picked = self.select_metrics(results)
        self.assertEqual(picked["high_return"], "h_qvix_only")
        rows = {r["id"]: r for r in picked["high_rows"]}
        self.assertTrue(rows["h_qvix_only"]["checks"]["drawdown"])
        self.assertFalse(rows["h_v91_no_overheat"]["checks"]["drawdown"])

    def test_h_tie_prefers_fewer_external_channels_then_lower_turnover(self):
        results = metric_fixture()
        for name in HIGH_RETURN_IDS:
            results[name]["metric"] = dict(cagr=.41, max_dd=-.30, turnover=10.0)
        results["h_qvix_only"]["metric"]["turnover"] = .1
        results["h_v91_no_overheat"]["metric"]["turnover"] = 5.0
        self.assertEqual(self.select_metrics(results)["high_return"], "h_v91_no_overheat")

    def test_s_thirty_percent_is_inclusive_and_drawdown_cannot_worsen(self):
        results = metric_fixture()
        results["s_v91_wls30"]["metric"] = dict(cagr=.30, max_dd=-.30, turnover=10.0)
        results["s_v91_wls30_no_crash"]["metric"] = dict(cagr=.60, max_dd=-.300001, turnover=1.0)
        results["s_accounts25_30"]["metric"] = dict(cagr=.299999, max_dd=-.10, turnover=1.0)
        picked = self.select_metrics(results)
        self.assertEqual(picked["robust"], "s_v91_wls30")
        rows = {r["id"]: r for r in picked["robust_rows"]}
        self.assertFalse(rows["s_v91_wls30_no_crash"]["checks"]["drawdown"])
        self.assertFalse(rows["s_accounts25_30"]["checks"]["cagr_at_least_30"])

    def test_s_prefers_lower_drawdown_then_turnover_not_highest_cagr(self):
        results = metric_fixture()
        results["s_v91_wls30"]["metric"] = dict(cagr=.55, max_dd=-.29, turnover=1.0)
        results["s_v91_wls30_no_crash"]["metric"] = dict(cagr=.32, max_dd=-.20, turnover=8.0)
        results["s_accounts25_30"]["metric"] = dict(cagr=.30, max_dd=-.20, turnover=4.0)
        self.assertEqual(self.select_metrics(results)["robust"], "s_accounts25_30")

    def test_future_paths_costs_and_failures_do_not_change_full_selection_output(self):
        results = paths_fixture()
        expected = research.select(results)
        self.assertIsNotNone(expected["high_return"])
        self.assertIsNotNone(expected["robust"])
        for reverse in (False, True):
            changed = {name: append_future(result, (i % 2 == 0) != reverse)
                       for i, (name, result) in enumerate(results.items())}
            with self.subTest(reverse=reverse):
                self.assertEqual(research.select(changed), expected)

    def test_acceptance_reports_validation_failure_without_reselecting(self):
        picked = dict(high_return="h_qvix_only", robust="s_accounts25_30")
        original = deepcopy(picked)
        summaries = {
            "c_v92": dict(full=dict(cagr=.40), validation=dict(cagr=.42)),
            "h_qvix_only": dict(full=dict(cagr=.41), validation=dict(cagr=.41)),
            "s_accounts25_30": dict(full=dict(cagr=.30), validation=dict(cagr=.29)),
            # A much better unselected result cannot replace a frozen choice.
            "h_v91_no_overheat": dict(full=dict(cagr=.99), validation=dict(cagr=.99)),
            "s_v91_wls30": dict(full=dict(cagr=.99), validation=dict(cagr=.99)),
        }
        result = research.gates(summaries, picked)
        self.assertTrue(result["full"]["high_pass"])
        self.assertTrue(result["full"]["robust_pass"])
        self.assertFalse(result["validation"]["high_pass"])
        self.assertFalse(result["validation"]["robust_pass"])
        self.assertEqual(set(result), {"full", "validation"})
        self.assertEqual(picked, original)

    def test_acceptance_ties_fail_for_h_and_empty_choices_remain_unselected(self):
        summaries = {name: {p: dict(cagr=.40) for p in ("full", "validation")}
                     for name in ("c_v92", "c_v91")}
        tied = research.gates(summaries, dict(high_return="c_v91", robust=None))
        self.assertFalse(tied["full"]["high_pass"])
        absent = research.gates(summaries, dict(high_return=None, robust=None))
        self.assertTrue(all(not row["high_pass"] and not row["robust_pass"] for row in absent.values()))


if __name__ == "__main__":
    unittest.main()
