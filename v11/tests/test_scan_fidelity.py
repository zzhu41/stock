"""Development selection/provenance and independent-audit safety regressions."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from v11 import fidelity
from v11.scan import select


CONTROL_IDS = {"h": "base", "v92": "v92", "simple": "simple"}


def config(candidate_id, family="test", changed=0):
    return dict(id=candidate_id, hash=candidate_id, families=[family], score="base_score",
                risk_context="current20", mechanism=changed)


def row(candidate_id, improvement=0.02, switches=20):
    scenarios = {}
    for name, cagr, drawdown, early, late in (
        ("close_1bp", .40, -.20, .45, .35), ("close_11bp", .36, -.22, .40, .30),
        ("lag1_1bp", .30, -.24, .34, .24), ("lag1_11bp", .28, -.25, .32, .20)):
        delta = 0. if candidate_id in CONTROL_IDS.values() else improvement
        scenarios[name] = dict(full=dict(cagr=cagr + delta, max_dd=drawdown), switches=switches,
                               blocks={"early": dict(cagr=early + delta), "late": dict(cagr=late + delta)})
    return dict(id=candidate_id, families=["control" if candidate_id in CONTROL_IDS.values() else "test"], scenarios=scenarios)


def fixture(candidate_ids=("a",)):
    ids = list(CONTROL_IDS.values()) + list(candidate_ids)
    return [row(value) for value in ids], [config(value) for value in ids]


def deterministic(selection):
    return {key: value for key, value in selection.items() if key != "frozen_at"}


class DevelopmentSelectionTests(unittest.TestCase):
    def test_every_protocol_gate_is_inclusive_and_fails_below_its_boundary(self):
        rows, configs = fixture()
        baseline, candidate = rows[0], rows[-1]
        candidate["scenarios"]["close_1bp"]["full"].update(cagr=max(.30, .9*.40), max_dd=-.20-.02)
        for scenario in ("close_11bp", "lag1_11bp"):
            candidate["scenarios"][scenario]["full"] = dict(
                cagr=baseline["scenarios"][scenario]["full"]["cagr"],
                max_dd=baseline["scenarios"][scenario]["full"]["max_dd"]-.02)
        outcome = select(rows, configs, CONTROL_IDS)
        self.assertEqual(outcome["primary"], "a")
        self.assertTrue(all(outcome["checks"]["a"].values()))
        gates = (("close_1bp", "cagr", "return_floor"), ("close_1bp", "max_dd", "main_drawdown"),
                 ("close_11bp", "cagr", "close_11bp_growth"), ("close_11bp", "max_dd", "close_11bp_drawdown"),
                 ("lag1_11bp", "cagr", "lag1_11bp_growth"), ("lag1_11bp", "max_dd", "lag1_11bp_drawdown"))
        for scenario, metric, gate in gates:
            changed = deepcopy(rows)
            changed[-1]["scenarios"][scenario]["full"][metric] -= .00001
            outcome = select(changed, configs, CONTROL_IDS)
            self.assertFalse(outcome["checks"]["a"][gate])
            self.assertIsNone(outcome["primary"])
            self.assertEqual(outcome["parents"], [])

    def test_absolute_30_percent_floor_remains_when_h_ninety_percent_is_lower(self):
        rows, configs = fixture()
        rows[0]["scenarios"]["close_1bp"]["full"]["cagr"] = .20
        rows[-1]["scenarios"]["close_1bp"]["full"]["cagr"] = .299
        self.assertFalse(select(rows, configs, CONTROL_IDS)["checks"]["a"]["return_floor"])

    def test_worst_block_before_median_before_switches_complexity_and_id(self):
        rows, configs = fixture(("a", "b"))
        # Mean/high ideal-close return cannot rescue a weaker worst high-fee block.
        a, b = rows[-2:]
        a["scenarios"]["close_1bp"]["full"]["cagr"] = 9.
        a["scenarios"]["close_11bp"]["blocks"]["early"]["cagr"] = .405
        self.assertEqual(select(rows, configs, CONTROL_IDS)["primary"], "b")
        # Equal worst, improved median wins even with more switches.
        rows, configs = fixture(("a", "b"))
        rows[-1]["scenarios"]["close_11bp"]["blocks"]["late"]["cagr"] += .04
        rows[-1]["scenarios"]["lag1_11bp"]["blocks"]["late"]["cagr"] += .04
        rows[-1]["scenarios"]["close_1bp"]["switches"] = 100
        self.assertEqual(select(rows, configs, CONTROL_IDS)["primary"], "b")
        rows, configs = fixture(("a", "b"))
        rows[-1]["scenarios"]["close_1bp"]["switches"] = 19
        self.assertEqual(select(rows, configs, CONTROL_IDS)["primary"], "b")
        rows, configs = fixture(("a", "b"))
        configs[-2]["mechanism"] = 1
        self.assertEqual(select(rows, configs, CONTROL_IDS)["primary"], "b")
        rows, configs = fixture(("a", "b"))
        self.assertEqual(select(rows, configs, CONTROL_IDS)["primary"], "a")

    def test_controls_do_not_become_discoveries_and_high_return_reject_cannot_replace_primary(self):
        rows, configs = fixture(("a", "b"))
        rows[-1]["scenarios"]["close_1bp"]["full"]["cagr"] = 100.
        rows[-1]["scenarios"]["lag1_11bp"]["full"]["cagr"] = -.50
        result = select(rows, configs, CONTROL_IDS)
        self.assertEqual(result["primary"], "a")
        self.assertEqual(result["top_return"], "b")
        self.assertNotIn("base", result["parents"])
        self.assertNotIn("b", result["parents"])

    def test_confirmation_2026_and_reporting_metrics_are_not_consulted(self):
        rows, configs = fixture(("a", "b"))
        before = deterministic(select(rows, configs, CONTROL_IDS))
        for i, value in enumerate(rows):
            value["confirmation"] = {"cagr": 1000 * i}
            value["report_only_2026"] = {"cagr": -999999 * i}
            value["full_2014_2026"] = {"cagr": 999999 * i}
            value["scenarios"]["confirmation_extra"] = {"full": {"cagr": 123456}}
        self.assertEqual(before, deterministic(select(rows, configs, CONTROL_IDS)))


class FidelityAuditTests(unittest.TestCase):
    def registry(self):
        candidates = []
        specs = []
        for index, context in enumerate(fidelity.CONTEXTS):
            score = "rank" if index % 2 else "ordinary"
            candidates.append(dict(id="id_%d" % index, families=["family"], risk_context=context, score=score))
        candidates.append(dict(id="id_z", families=["other"], risk_context="current20", score="robust"))
        specs = [dict(name="ordinary", estimator="wls", aggregate="mean"),
                 dict(name="rank", estimator="logmom", aggregate="rank"),
                 dict(name="robust", estimator="huberlog", aggregate="mean")]
        return dict(candidates=candidates, score_specs=specs), dict(required_observations_by_score={"ordinary": 270, "rank": 272, "robust": 270})

    def test_sample_uses_configuration_ids_only_with_context_and_rank_coverage(self):
        registry, meta = self.registry()
        chosen, reasons, unused = fidelity.choose_cases(registry, meta)
        ids = [c["id"] for c in chosen]
        self.assertIn("id_0", ids)
        self.assertIn("id_1", ids)
        self.assertEqual({c["risk_context"] for c in chosen}, set(fidelity.CONTEXTS))
        self.assertTrue(any(c["score"] == "rank" for c in chosen))
        modified = deepcopy(registry)
        modified["candidates"].reverse()
        for i, c in enumerate(modified["candidates"]):
            c["all_time_cagr"] = 1e10 * i
            c["development_cagr"] = -1e10 * i
        again, again_reasons, unused = fidelity.choose_cases(modified, meta)
        self.assertEqual(ids, [c["id"] for c in again])
        self.assertEqual(reasons, again_reasons)

    def test_grouping_audit_rejects_equal_validity_with_other_feature_difference(self):
        candidates = [dict(id="a", score="first", risk_context="current20"), dict(id="b", score="second", risk_context="current20")]
        arrays = dict(features=np.ones((3, 2, 2)), scores=np.ones((2, 3, 2)), orders=np.zeros((2, 3, 2), dtype=np.int32))
        def bad_view(arrays, meta, context, score):
            out = dict(arrays, features=arrays["features"].copy())
            out["features"][:, :, 1] = 2. if score == "second" else 1.
            return out
        with patch.object(fidelity, "risk_view", side_effect=bad_view):
            with self.assertRaisesRegex(AssertionError, "aliases"):
                fidelity.audit_grouping(candidates, arrays, {"feature_names": ["valid", "vol20"]})

    def test_oracle_can_only_see_development_prefix(self):
        arrays = dict(features=np.ones((3, 1, 2)), scores=np.ones((2, 3, 1)), orders=np.zeros((2, 3, 1)), fear=np.ones(3), observations=np.ones((3, 1)))
        meta = dict(dates=["2021-12-30", "2021-12-31", "2022-01-04"])
        shortened, short_meta = fidelity.development_prefix(arrays, meta)
        self.assertEqual(short_meta["dates"], ["2021-12-30", "2021-12-31"])
        self.assertEqual(shortened["features"].shape[0], 2)
        self.assertEqual(shortened["scores"].shape[1], 2)
        self.assertFalse(np.shares_memory(shortened["scores"], arrays["scores"]))
        with self.assertRaisesRegex(ValueError, "2014-2021"):
            fidelity.validate_calendar(["2021-12-31", "2022-01-04"])
        with self.assertRaises(ValueError):
            fidelity.validate_calendar(["2014-01-03", "2014-01-02"])

    def test_any_exact_path_or_summary_mismatch_fails_instead_of_becoming_tolerance_pass(self):
        dates = ["2014-01-02", "2014-01-03"]
        returns, holdings, summary = np.asarray([0., .01]), np.asarray([0, 1]), np.zeros(10)
        reference = dict(dates=dates, returns=returns.copy(), holdings=holdings.copy(), summary=summary.copy())
        self.assertTrue(fidelity.compare_path(reference, returns, holdings, summary, dates)["exact_summary"])
        reference["returns"][1] += 1e-15
        with self.assertRaises(AssertionError):
            fidelity.compare_path(reference, returns, holdings, summary, dates)
        reference["returns"] = returns
        reference["summary"][3] = 1
        with self.assertRaisesRegex(AssertionError, "switches"):
            fidelity.compare_path(reference, returns, holdings, summary, dates)


if __name__ == "__main__":
    unittest.main()
