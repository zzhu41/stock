"""Stress workflow tests use synthetic arrays and mocked simulations only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v10_deep import stress
from v10_deep.schema import baseline, identifier


DATES = ["2014-01-02", "2021-12-31", "2022-01-04", "2025-12-31", "2026-01-05", "2026-09-24"]


def config(**changes):
    c = baseline()
    c.update(changes)
    h = identifier(c)
    c.update(id="vd_" + h[:20], hash=h, families=["synthetic"], stage="synthetic")
    return c


def fixture(root, guarded_same=True):
    cache = root / "cache"
    cache.mkdir(parents=True)
    (cache / "features.npz").write_bytes(b"test-cache-not-a-real-npz")
    (cache / "features.json").write_text('{"fixture":true}')
    controls = [config(crash_mask=1, global_buffer=.02, gold_buffer=.02),
                config(crash_mask=1), config()]
    primary, first_primary = config(mom=40), config(min_hold=3)
    configs = controls + [primary, first_primary]
    control_ids = dict(zip(("v9", "v9.1", "v9.2"), (c["id"] for c in controls)))
    meta = dict(dates=list(DATES), fingerprints={"data": "frozen-base-data", "source": "frozen-feature-code"},
                assets=["510300"], feature_names=["close"], score_names=["synthetic"])
    arrays = {"fear": np.zeros(len(DATES), dtype=np.int32)}
    for stage, chosen in (("mechanisms", first_primary), ("refinements", primary)):
        output = root / "results" / stage
        stress.dump(output / "registered_candidates.json", dict(candidates=configs))
        stress.dump(output / "registration.json", dict(registry_sha256=stress.sha(output / "registered_candidates.json"),
                    fingerprints=dict(features=meta["fingerprints"], feature_cache_sha256=stress.sha(cache / "features.npz"))))
        stress.dump(output / "selection.json", dict(primary=chosen["id"], guarded=chosen["id"] if guarded_same else None,
                    controls=control_ids, registration_sha256=stress.sha(output / "registration.json")))
    return configs, primary, first_primary, arrays, meta


def precision_case(name):
    return {"synthetic_scenario": name}, dict(scenario=name, calendar=list(DATES),
                    histories_content_sha256="histories-" + name,
                    input_manifest_content_sha256="manifest-content", source_file_sha256={"frozen.csv": "file-sha"},
                    constructor_source_sha256="precision-code", data_view="cash_precision_" + name)


class StressTests(unittest.TestCase):
    def test_fixed_budget_has_fee_and_lag_cross_product_plus_three_cash_views(self):
        cases = stress.scenario_specs()
        self.assertEqual(len(cases), 11)
        self.assertEqual(len({c["name"] for c in cases}), 11)
        base = [c for c in cases if c["data_scenario"] == "base"]
        self.assertEqual({(c["force_lag"], c["fee"]) for c in base},
                         {(lag, fee) for lag in (None, 1) for fee in stress.FEES})
        cash = [c for c in cases if c["data_scenario"] != "base"]
        self.assertTrue(all(c["fee"] == .0001 and c["force_lag"] is None for c in cash))

    def test_finalist_plan_deduplicates_roles_without_losing_three_controls(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(stress, "BASE", Path(directory)):
            root = Path(directory)
            configs, primary, first, _, _ = fixture(root)
            plan = stress.finalist_plan("refinements")
            self.assertEqual(len(plan["configs"]), 5)
            self.assertEqual(plan["roles"]["primary"], plan["roles"]["guarded"])
            self.assertEqual(plan["roles"]["stage1_primary"], first["id"])
            self.assertEqual({c["id"] for c in plan["configs"]}, {c["id"] for c in configs})
            selection = root / "results/refinements/selection.json"
            d = json.loads(selection.read_text())
            d["guarded"] = None
            stress.dump(selection, d)
            self.assertIsNone(stress.finalist_plan("refinements")["roles"]["guarded"])

    def test_lag_changes_are_separately_identified_and_do_not_mutate_registered_configs(self):
        original = [config()]
        saved = deepcopy(original)
        delayed = stress.effective_configs(original, dict(force_lag=1))
        self.assertEqual(original, saved)
        self.assertEqual(delayed[0]["lag"], 1)
        self.assertNotEqual(delayed[0]["id"], original[0]["id"])
        self.assertEqual(delayed[0]["hash"], identifier(delayed[0]))

    def test_precision_build_cannot_save_cache_and_replaces_effective_data_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(stress.features, "BASE", Path(directory)):
            root = Path(directory)
            _, _, _, arrays, base_meta = fixture(root)
            histories, detail = precision_case("cash_lower")
            before = stress._cache_hashes()
            old_writer, old_dump, old_inputs = stress.features.np.savez_compressed, stress.features.dump, stress.features.inputs

            def fake_build(force=False):
                self.assertTrue(force)
                self.assertEqual(stress.features.inputs()[0], histories)
                stress.features.np.savez_compressed(root / "cache/features.npz", dummy=np.ones(2))
                stress.features.dump(root / "cache/features.json", {"should_not_be_saved": True})
                return arrays, base_meta

            with patch.object(stress.features, "build", fake_build):
                actual, meta = stress.precision_features(histories, detail, arrays["fear"], DATES)
            self.assertIs(actual, arrays)
            self.assertEqual(before, stress._cache_hashes())
            self.assertIs(stress.features.np.savez_compressed, old_writer)
            self.assertIs(stress.features.dump, old_dump)
            self.assertIs(stress.features.inputs, old_inputs)
            self.assertEqual(meta["fingerprints"]["data"], detail["histories_content_sha256"])
            self.assertEqual(meta["fingerprints"]["base_data_manifest_sha256"], "frozen-base-data")
            self.assertEqual(base_meta["fingerprints"]["data"], "frozen-base-data")
            self.assertFalse(meta["cache_persisted"])
            self.assertFalse(meta["usable_for_next_open"])

    def test_stale_base_cache_cannot_be_written_implicitly(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(stress.features, "BASE", Path(directory)):
            root = Path(directory)
            fixture(root)
            before = stress._cache_hashes()

            def stale_build():
                stress.features.np.savez_compressed(root / "cache/features.npz", bad=np.ones(3))

            with patch.object(stress.features, "build", stale_build), self.assertRaisesRegex(RuntimeError, "missing/stale"):
                stress.cached_base_features()
            self.assertEqual(before, stress._cache_hashes())

    def test_unexpected_direct_cache_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(stress.features, "BASE", Path(directory)):
            root = Path(directory)
            _, _, _, arrays, meta = fixture(root)
            histories, detail = precision_case("cash_upper")

            def bad_builder(force=False):
                (root / "cache/features.npz").write_bytes(b"unexpected-direct-writer")
                return arrays, meta

            with patch.object(stress.features, "build", bad_builder), self.assertRaisesRegex(AssertionError, "cache changed"):
                stress.precision_features(histories, detail, arrays["fear"], DATES)

    def test_full_workflow_registers_before_any_simulation_and_does_not_reselect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs, primary, first, arrays, meta = fixture(root)
            selection_files = [root / "results" / s / "selection.json" for s in ("mechanisms", "refinements")]
            before = [stress.sha(p) for p in selection_files]
            calls = []
            testcase = self

            class FakeSimulator:
                def __init__(self, values, metadata):
                    self.meta = metadata
                    self.build_fingerprint = {"synthetic": True}

                def run(self, cases, start, end, fee, workers, capture):
                    testcase.assertTrue((root / "results/refinements/stress/registration.json").is_file())
                    calls.append((fee, [c["lag"] for c in cases], self.meta["fingerprints"]["data"]))
                    r = np.tile([0., .001 - 2 * fee, .002, .003, .01, .02], (len(cases), 1))
                    summary = np.zeros((len(cases), 10))
                    summary[:, 0] = np.prod(1 + r, axis=1)
                    summary[:, 3] = 2
                    return dict(returns=r, summary=summary, dates=list(DATES))

            def fake_build(force=False):
                if force:
                    _, calendar, fear = stress.features.inputs()
                    self.assertEqual(calendar, DATES)
                    self.assertEqual(len(fear), len(DATES))
                    stress.features.np.savez_compressed(root / "cache/features.npz", synthetic=arrays["fear"])
                    stress.features.dump(root / "cache/features.json", {})
                return deepcopy(arrays), deepcopy(meta)

            with patch.object(stress, "BASE", root), patch.object(stress.features, "BASE", root), \
                    patch.object(stress.features, "build", fake_build), \
                    patch.object(stress, "build_scenario", side_effect=precision_case), \
                    patch.object(stress, "Simulator", FakeSimulator), \
                    patch.object(stress, "protect", return_value=289), \
                    patch.object(stress, "_source_fingerprints", return_value={"stress.py": "test-source"}):
                result = stress.run_stress("refinements", workers=1)
            self.assertEqual(len(calls), 11)
            self.assertTrue(all(lag == 1 for call in calls[4:8] for lag in call[1]))
            self.assertEqual([call[2] for call in calls[-3:]], ["histories-" + n for n in stress.SCENARIOS])
            self.assertEqual(before, [stress.sha(p) for p in selection_files])
            self.assertFalse(result["selection_feedback_permitted"])
            self.assertEqual(len(result["scenarios"]), 11)
            for scenario in result["scenarios"]:
                self.assertEqual(len(scenario["rows"]), 5)
                for row in scenario["rows"]:
                    metrics = row["metrics"]
                    self.assertAlmostEqual(metrics["report_only_2026"]["total_return"], 1.01 * 1.02 - 1)
                    self.assertAlmostEqual(metrics["yearly"]["2026"], metrics["report_only_2026"]["total_return"])
                    self.assertIn("selection", metrics)
                    self.assertIn("recent_2022_2025", metrics)
            registration = json.loads((root / "results/refinements/stress/registration.json").read_text())
            self.assertEqual(registration["distinct_source_config_count"], 5)
            self.assertEqual(registration["scenario_count"], 11)
            self.assertEqual(registration["roles"]["primary"], primary["id"])
            self.assertEqual(registration["roles"]["stage1_primary"], first["id"])


if __name__ == "__main__":
    unittest.main()
