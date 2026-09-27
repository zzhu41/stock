"""Synthetic pool-exclusion accounting and independent execution checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v11 import leave_one_out as loo
from v10_deep.schema import baseline, identifier, semantic, STOCK, GLOBAL
from v10_deep.tests.test_native import inputs, prices, put, finish, AI, A, B, X, BENCH, GOLD, CASH


def configs():
    output = {}
    for role, hold, score in (("a", 2, "wls25_v20"), ("b", 3, "wls30_v20"), ("h", 0, "wls25_v20")):
        c = baseline()
        c.update(min_hold=hold, score=score, risk_context="current20", buffer=0., global_buffer=0., gold_buffer=0.)
        digest = identifier(c)
        output[role] = dict(c, id="v11_" + digest[:20], hash=digest, families=["synthetic"], parents=[], stage="fixture")
    return output


def fixture(days=9):
    arrays, meta = inputs(days)
    arrays["scores"][:, :, AI[BENCH]] = -10.
    arrays["observations"] = np.full(arrays["features"].shape[:2], 300, dtype=np.int32)
    meta["required_observations_by_score"] = {name: 270 for name in meta["score_names"]}
    return arrays, meta


class PoolDefinitionTests(unittest.TestCase):
    def test_plan_pairs_all_nine_exclusions_and_preserves_inputs(self):
        original = configs()
        before = deepcopy(original)
        plan = loo.make_plan(original)
        self.assertEqual(original, before)
        self.assertEqual(plan["excluded_assets"], list(STOCK + GLOBAL))
        self.assertEqual(len(plan["cases"]), 10)
        self.assertEqual(len(plan["candidates"]), 30)
        candidates = {c["id"]: c for c in plan["candidates"]}
        for case in plan["cases"]:
            self.assertEqual(set(case["roles"]), {"a", "b", "h"})
            for role, cid in case["roles"].items():
                actual = semantic(candidates[cid])
                expected = semantic(original[role])
                if case["excluded"]:
                    key = "stock_pool" if case["excluded"] in expected["stock_pool"] else "global_pool"
                    expected[key].remove(case["excluded"])
                self.assertEqual(actual, expected)
                self.assertEqual(candidates[cid]["hash"], identifier(actual))
        self.assertEqual(plan["cases"][0]["roles"], plan["frozen_ids"])

    def test_reject_gold_cash_nonmember_and_mismatched_starting_pools(self):
        values = configs()
        for code in (GOLD, CASH, "999999"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                loo.exclude_asset(values["a"], code)
        changed = deepcopy(values)
        changed["a"]["stock_pool"].remove(A)
        with self.assertRaises(ValueError):
            loo.make_plan(changed)

    def test_registration_is_before_execution_and_detects_input_changes(self):
        with tempfile.TemporaryDirectory(prefix="v11-loo-test-") as directory:
            root = Path(directory)
            source = root / "fixture.json"
            source.write_text('{"fixed": true}')
            context = dict(plan=loo.make_plan(configs()), files={"fixture": source}, score_specs=[],
                           feature_fingerprints={"synthetic": True}, feature_array_sha256="0" * 64)
            with patch.object(loo, "frozen_inputs", return_value=context), \
                    patch.object(loo, "build", side_effect=AssertionError("Registration cannot build prices")) as build, \
                    patch.object(loo, "run_candidates", side_effect=AssertionError("Registration cannot evaluate returns")) as run:
                design = loo.register(root / "output")
                self.assertEqual(design["path_count"], 120)
                self.assertFalse(design["candidate_selection_performed"])
                self.assertFalse(design["confirmation_failure_overturned"])
                self.assertEqual(loo.register(root / "output"), design)
                build.assert_not_called()
                run.assert_not_called()
                source.write_text('{"fixed": false}')
                with self.assertRaises(ValueError):
                    loo.register(root / "output")


class SyntheticNativeReferenceTests(unittest.TestCase):
    def test_exclusions_match_independent_reference_and_do_not_touch_features(self):
        arrays, meta = fixture()
        prices(arrays, A, [100., 105., np.nan, 110., 90., 93., 96., 100., 101.])
        prices(arrays, B, [100., 101., 102., 104., 103., 102., 104., 106., 107.])
        prices(arrays, X, [100., 100., 101., 102., 103., 105., 106., 105., 107.])
        arrays["scores"][:, 3:, AI[B]] = 80.
        put(arrays, X, "mom5", [.01, .01, -.1, .01, .01, .01, .01, .01, .01])
        put(arrays, X, "ma250", [.1, .1, -.3, .1, .1, .1, .1, .1, .1])
        put(arrays, BENCH, "ma250", [.1, .1, .1, .1, -.1, -.1, .1, .1, .1])
        arrays = finish(arrays)
        before = {name: value.copy() for name, value in arrays.items()}
        plan = loo.make_plan(configs())
        result, dates = loo.run_candidates(plan["candidates"], arrays, meta,
            start=meta["dates"][0], end=meta["dates"][-1], scenarios=loo.SCENARIOS, workers=1)
        proofs = loo.validate_reference(plan, arrays, meta, result, dates)
        self.assertEqual(len(proofs), 6)
        self.assertEqual({p["role"] for p in proofs}, {"a", "b", "h"})
        self.assertEqual({p["scenario"] for p in proofs}, {"close_1bp", "lag1_11bp"})
        positions = {c["id"]: i for i, c in enumerate(plan["candidates"])}
        for case in plan["cases"][1:]:
            for payload in result.values():
                for cid in case["roles"].values():
                    self.assertFalse(np.any(payload["holdings"][positions[cid]] == AI[case["excluded"]]))
        for name, value in arrays.items():
            np.testing.assert_array_equal(value, before[name])

    def test_deleted_benchmark_still_controls_bear_regime(self):
        arrays, meta = fixture(5)
        put(arrays, BENCH, "ma250", [.1, .1, -.1, -.1, -.1])
        arrays = finish(arrays)
        c = loo.exclude_asset(configs()["h"], BENCH)
        result, dates = loo.run_candidates([c], arrays, meta, start=meta["dates"][0],
            end=meta["dates"][-1], scenarios=loo.SCENARIOS, workers=1)
        names = [meta["assets"][i] for i in result["close_1bp"]["holdings"][0]]
        self.assertEqual(names, [A, A, X, X, X])
        # Clearing the excluded benchmark's information would turn this bearish
        # decision into the engine's default-bull case; we expressly retain it.
        self.assertTrue(np.all(arrays["features"][2:, AI[BENCH], meta["feature_names"].index("ma250")] < 0))

    def test_full_pool_report_comparison_checks_all_fields_and_row_order(self):
        plan = loo.make_plan(configs())
        ids = [c["id"] for c in plan["candidates"]]
        dates = ["2020-01-02", "2020-01-03"]
        result = {name: dict(returns=np.zeros((len(ids), 2)), holdings=np.zeros((len(ids), 2), dtype=np.int32),
                            summary=np.zeros((len(ids), 10))) for name, _, _ in loo.SCENARIOS}
        original_ids = list(reversed(list(plan["frozen_ids"].values())))
        archive = {}
        for name, payload in result.items():
            for field, values in payload.items():
                archive[name + "__" + field] = np.asarray([values[ids.index(cid)] for cid in original_ids])
        metadata = dict(ids=original_ids, dates=dates)
        self.assertEqual(len(loo.validate_full_pool(plan, result, dates, metadata, archive)), 12)
        for field in ("returns", "holdings", "summary"):
            altered = {key: value.copy() for key, value in archive.items()}
            altered["lag1_11bp__" + field][0, 0] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                loo.validate_full_pool(plan, result, dates, metadata, altered)


class PairedAccountingTests(unittest.TestCase):
    def test_period_slicing_retains_first_return_fee_and_2026_is_not_annualized(self):
        dates = ["2021-12-31", "2022-01-04", "2025-12-31", "2026-01-05"]
        daily = np.asarray([0., 1.10 * .9978 - 1, -.02, -.03])
        holdings = np.asarray([0, 1, 1, 2])
        result = loo.period_metrics(daily, holdings, dates)
        self.assertAlmostEqual(result["confirmation"]["total_return"], 1.10 * .9978 * .98 - 1)
        self.assertEqual(result["confirmation"]["switches"], 1)
        self.assertEqual(result["report_only_2026"]["switches"], 1)
        self.assertAlmostEqual(result["report_only_2026"]["max_dd"], -.03)
        self.assertIsNone(result["report_only_2026"]["cagr"])
        self.assertAlmostEqual(result["report_only_2026"]["total_return"], -.03)

    def test_pair_uses_h_under_same_exclusion_and_keeps_own_full_pool_separate(self):
        plan = loo.make_plan(configs())
        ids = [c["id"] for c in plan["candidates"]]
        dates = ["2020-01-02", "2020-01-03", "2020-01-04"]
        payload = dict(returns=np.zeros((len(ids), 3)), holdings=np.zeros((len(ids), 3), dtype=np.int32))
        full, deleted = plan["cases"][0], plan["cases"][1]
        for case, role, value in ((full, "a", .03), (full, "h", .01), (deleted, "a", .02), (deleted, "h", -.02)):
            payload["returns"][ids.index(case["roles"][role]), -1] = value
        result = loo.summarize(plan, {"close_1bp": payload}, dates, periods=(("full", dates[0], dates[-1]),))
        actual = result[1]["roles"]["a"]["scenarios"]["close_1bp"]["full"]
        self.assertAlmostEqual(actual["versus_own_full_pool"]["total_return_pp"], -1.)
        self.assertAlmostEqual(actual["versus_paired_h"]["total_return_pp"], 4.)
        self.assertAlmostEqual(actual["change_in_excess_vs_full_pool"]["total_return_pp"], 2.)
        self.assertAlmostEqual(actual["versus_paired_h"]["net_log_growth"], np.log(1.02) - np.log(.98))
        self.assertGreater(actual["versus_paired_h"]["max_dd_pp"], 0)
        self.assertFalse(any("winner" in key or "selected" in key for row in result for key in row))


if __name__ == "__main__":
    unittest.main()
