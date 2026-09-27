"""Preregistration/diagnostic tests; no confirmation or report-only returns read."""
from copy import deepcopy
import json
import unittest

import numpy as np

from v11 import neighborhood as n
from v10_deep.schema import identifier


class NeighborhoodDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads(n.PARENT_REGISTRY.read_text())
        cls.selection = json.loads((n.DEVELOPMENT / "selection.json").read_text())
        cls.record = n.generate(cls.original, cls.selection)
        cls.configs = {c["id"]: c for c in cls.record["candidates"]}
        cls.specs = {s["name"]: s for s in cls.record["score_specs"]}

    def proposal(self, parent, label):
        row = next(x for x in self.record["parent_map"][parent]["proposals"] if x["label"] == label)
        return row, self.configs[row["candidate_id"]]

    def test_fixed_parent_family_has_expected_deduplicated_counts_and_no_new_primary(self):
        self.assertEqual(self.record["primary_unchanged"], self.selection["primary"])
        self.assertFalse(self.record["reselect_primary"])
        self.assertFalse(self.record["heldout_performance_accessed"])
        self.assertEqual(self.record["development"], ["2014-01-02", "2021-12-31"])
        self.assertEqual(self.record["candidate_count"], 58)
        self.assertEqual(len(self.record["extra_score_specs"]), 19)
        self.assertEqual(len(self.record["score_specs"]), len(self.original["score_specs"]) + 19)
        self.assertEqual([len(self.record["parent_map"][p]["neighbors"]) for p in self.record["parents"]],
                         [11, 9, 14, 9])
        self.assertEqual(len({c["hash"] for c in self.record["candidates"]}), 58)
        self.assertTrue(all(identifier(c) == c["hash"] for c in self.record["candidates"]))
        old = {c["id"]: c for c in self.original["candidates"]}
        for pid in self.record["parents"] + list(self.record["controls"].values()):
            self.assertEqual(n.normalize(self.configs[pid]), n.normalize(old[pid]))

    def test_exact_frozen_score_aliases_never_become_recomputed_centers(self):
        catalog = {}
        for name in ("wls20_smooth3", "wls25_v20"):
            spec = n.spec_for(name, catalog)
            self.assertEqual(n.materialize_spec(spec, catalog), name)
        self.assertEqual(catalog, {})
        self.assertNotIn("wls20_smooth3", self.specs)
        self.assertNotIn("wls25_v20", self.specs)
        h = self.record["controls"]["h"]
        self.assertEqual(self.configs[h]["score"], "wls20_smooth3")
        v92 = self.record["controls"]["v92"]
        self.assertEqual(self.configs[v92]["score"], "wls25_v20")

    def test_individual_and_group_window_changes_are_separate_not_cartesian(self):
        parent = self.record["parents"][2]
        source = n.spec_for(self.configs[parent]["score"], self.specs)
        self.assertEqual(source["windows"], [10, 20, 40])
        for index in range(3):
            for scale in (.8, 1.2):
                row, config = self.proposal(parent, "score_window_%d_x%.1f" % (index, scale))
                actual = n.spec_for(config["score"], self.specs)["windows"]
                expected = list(source["windows"])
                expected[index] = max(2, round(expected[index] * scale))
                self.assertEqual(actual, sorted(set(expected)))
                self.assertFalse(row["no_op"])
        for scale in (.8, 1.2):
            unused, config = self.proposal(parent, "score_all_windows_x%.1f" % scale)
            self.assertEqual(n.spec_for(config["score"], self.specs)["windows"],
                             [round(w * scale) for w in source["windows"]])
        single = self.record["parents"][0]
        for scale in (.8, 1.2):
            a, unused = self.proposal(single, "score_window_0_x%.1f" % scale)
            b, unused = self.proposal(single, "score_all_windows_x%.1f" % scale)
            self.assertEqual(a["candidate_id"], b["candidate_id"])
            self.assertTrue(b["deduplicated"])

    def test_buffer_shift_preserves_spacing_and_ablations_restore_only_the_named_group(self):
        h = self.configs[self.record["controls"]["h"]]
        for pid in self.record["parents"]:
            parent = self.configs[pid]
            for delta in (-.005, .005):
                unused, changed = self.proposal(pid, "momentum_buffers_%+.3f" % delta)
                for field in ("buffer", "global_buffer", "gold_buffer"):
                    self.assertAlmostEqual(changed[field] - parent[field], delta)
                self.assertAlmostEqual(changed["global_buffer"] - changed["buffer"],
                                       parent["global_buffer"] - parent["buffer"])
        primary = self.record["parents"][0]
        unused, restored = self.proposal(primary, "restore_H_buffer_group")
        self.assertEqual(restored["min_hold"], 3)
        self.assertEqual([restored[k] for k in ("buffer", "global_buffer", "gold_buffer")],
                         [h[k] for k in ("buffer", "global_buffer", "gold_buffer")])
        unused, restored = self.proposal(primary, "restore_H_minimum_hold")
        self.assertEqual(restored["min_hold"], h["min_hold"])
        self.assertEqual([restored[k] for k in ("buffer", "global_buffer", "gold_buffer")], [.01] * 3)

    def test_risk_and_score_ablations_do_not_reoptimize_other_fields(self):
        h = self.configs[self.record["controls"]["h"]]
        risk = self.record["parents"][1]
        unused, context = self.proposal(risk, "restore_H_risk_context")
        self.assertEqual((context["risk_context"], context["panic"]), ("current20", 1.75))
        unused, magnitude = self.proposal(risk, "restore_H_panic_magnitude")
        self.assertEqual((magnitude["risk_context"], magnitude["panic"]), ("prior60", h["panic"]))
        row, unused = self.proposal(risk, "restore_H_whole_risk")
        self.assertEqual(row["candidate_id"], h["id"])
        multi = self.record["parents"][2]
        unused, single = self.proposal(multi, "restore_single_20_window")
        self.assertEqual(n.spec_for(single["score"], self.specs)["windows"], [20])
        self.assertEqual(n.spec_for(single["score"], self.specs)["smooth"], 1)
        robust = self.record["parents"][3]
        unused, estimator = self.proposal(robust, "restore_WLS_estimator")
        self.assertEqual(n.spec_for(estimator["score"], self.specs)["estimator"], "wls")
        self.assertEqual(n.spec_for(estimator["score"], self.specs)["smooth"], 5)

    def test_hold_one_is_semantic_noop_and_fixed_panic_keeps_fixed_mode(self):
        parent = self.configs[self.record["parents"][0]]
        zero, one = deepcopy(parent), deepcopy(parent)
        zero["min_hold"], one["min_hold"] = 0, 1
        self.assertEqual(identifier(n.normalize(zero)), identifier(n.normalize(one)))
        for pid in self.record["parents"][1:]:
            for step in (-1, 1):
                row, unused = self.proposal(pid, "minimum_hold_%+d" % step)
                self.assertTrue(row["no_op"])
        fixed = n.normalize(dict(parent, panic_mode="fixed", panic=.04))
        digest = identifier(fixed)
        fixed.update(id="v11_" + digest[:20], hash=digest, families=["synthetic_test"], parents=[], stage="test")
        original = deepcopy(self.original)
        original["candidates"].append(fixed)
        selection = dict(self.selection, primary=fixed["id"], parents=[fixed["id"]])
        generated = n.generate(original, selection)
        candidates = {c["id"]: c for c in generated["candidates"]}
        for row in generated["parent_map"][fixed["id"]]["proposals"]:
            if row["label"].startswith("panic_x"):
                c = candidates[row["candidate_id"]]
                self.assertEqual(c["panic_mode"], "fixed")
                self.assertIn(c["panic"], (.036, .044))

    def test_flatness_excludes_center_and_ablation_from_pass_rate_and_quartile(self):
        registry = dict(parents=["center"], parent_map={"center": dict(
            neighbors=["n1", "n2", "n3", "n4"], ablations=["ablation"])})
        ids = ["center", "n1", "n2", "n3", "n4", "ablation"]
        checks = {cid: {"gate": cid not in ("n2", "n4")} for cid in ids}
        scores = {cid: dict(worst_block_excess=value) for cid, value in
                  zip(ids, [99., .01, -.02, .03, 0., 99.])}
        result = n.stability(registry, checks, scores)["center"]
        self.assertEqual(result["neighbor_count"], 4)
        self.assertEqual(result["passed_count"], 2)
        self.assertEqual(result["pass_rate"], .5)
        self.assertAlmostEqual(result["worst_block_excess_median"], .005)
        self.assertAlmostEqual(result["worst_block_excess_q25"], -.005)

    def test_execution_views_physically_exclude_confirmation_and_2026_axes(self):
        dates = ["2020-12-31", "2021-12-31", "2022-01-04", "2026-01-05"]
        arrays = dict(features=np.ones((4, 2, 3)), scores=np.ones((5, 4, 2)),
                      orders=np.zeros((5, 4, 2), dtype=np.int32), fear=np.zeros(4),
                      observations=np.ones((4, 2)), years=np.arange(4))
        meta = dict(dates=dates, shape=[4, 2, 3])
        truncated, metadata = n.development_arrays(arrays, meta)
        self.assertEqual(metadata["dates"], dates[:2])
        self.assertEqual(truncated["features"].shape[0], 2)
        self.assertEqual(truncated["scores"].shape[1], 2)
        self.assertEqual(truncated["orders"].shape[1], 2)
        self.assertEqual(len(truncated["fear"]), 2)
        self.assertEqual(meta["dates"], dates)
        self.assertEqual(meta["shape"][0], 4)


if __name__ == "__main__":
    unittest.main()
