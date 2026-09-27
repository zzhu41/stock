"""B/fixed-H preregistration and paired-center diagnostics, development inputs only."""
from copy import deepcopy
import unittest

from v11 import extended_neighborhoods as extended
from v11 import neighborhood as original


class ExtendedNeighborhoodDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = extended.load_sources()
        cls.records, cls.override, cls.matched = extended.prepare_records(cls.sources)

    def test_b_uses_the_real_frozen_combination_choice_and_h_is_only_a_fixed_control(self):
        b = self.records["b"]
        h = self.records["h"]
        actual = self.sources["b"]["selection"]["primary"]
        self.assertEqual(b["parents"], [actual])
        self.assertEqual(b["primary_unchanged"], actual)
        self.assertEqual(actual, "v11_e2a10e8c6af1c0803ba4")
        self.assertEqual(h["parents"], [self.sources["a"]["registry"]["controls"]["h"]])
        self.assertEqual(h["subject_kind"], "fixed_H_control")
        self.assertNotIn("primary_unchanged", h)
        self.assertFalse(h["selection_performed_here"])
        self.assertFalse(self.override["selected_as_new_A_or_B_candidate"])
        self.assertFalse(self.override["adaptive_selection"])
        self.assertEqual(h["A_primary_unchanged"], self.sources["a"]["selection"]["primary"])
        self.assertEqual(h["B_primary_unchanged"], actual)

    def test_complete_and_matched_sets_are_fixed_before_any_new_performance(self):
        counts = {key: (record["candidate_count"], len(record["parent_map"][record["parents"][0]]["neighbors"]))
                  for key, record in self.records.items()}
        self.assertEqual(counts, {"b": (25, 16), "h": (12, 9)})
        self.assertEqual(self.matched["operation_labels"], [
            "momentum_buffers_+0.005", "momentum_buffers_-0.005", "panic_x0.9", "panic_x1.1",
            "score_all_windows_x0.8", "score_all_windows_x1.2", "smooth_+1", "switch_confirmation_+1"])
        for subject in self.matched["subjects"].values():
            self.assertEqual(len(subject["operation_to_id"]), 8)
            self.assertEqual(len(set(subject["operation_to_id"].values())), 8)
        for record in self.records.values():
            self.assertTrue(record["complete_neighborhood_is_primary"])
            self.assertEqual(record["development"], ["2014-01-02", "2021-12-31"])
            self.assertFalse(record["read_confirmation_results"])
            self.assertFalse(record["read_2026_results"])

    def test_wrapper_keeps_the_frozen_a_generator_constants_and_original_inputs_intact(self):
        constants = (original.OUT, original.PARENT_REGISTRY, original.DEVELOPMENT, original.SOURCE_NAMES)
        snapshots = deepcopy((self.sources["a"]["registry"], self.sources["a"]["selection"], self.sources["a_neighbors"]))
        extended.prepare_records(self.sources)
        self.assertEqual(constants, (original.OUT, original.PARENT_REGISTRY, original.DEVELOPMENT, original.SOURCE_NAMES))
        self.assertEqual(snapshots, (self.sources["a"]["registry"], self.sources["a"]["selection"], self.sources["a_neighbors"]))
        fingerprint = extended.fingerprints()
        self.assertIn("v11/extended_neighborhoods.py", fingerprint["sources"])
        self.assertIn("v11/COMBINATION_PROTOCOL.md", fingerprint["sources"])
        self.assertIn("B/registered_candidates.json", fingerprint["original_artifacts"])
        self.assertIn("A_neighborhood/evaluation.json", fingerprint["original_artifacts"])

    def test_b_score_ablation_recovers_original_a_without_reselecting_it(self):
        record = self.records["b"]
        parent = record["parents"][0]
        restore = next(p for p in record["parent_map"][parent]["proposals"] if p["label"] == "restore_H_whole_score")
        self.assertEqual(restore["candidate_id"], self.sources["a"]["selection"]["primary"])
        self.assertEqual(record["primary_unchanged"], self.sources["b"]["selection"]["primary"])


class PairedCenterDiagnosticTests(unittest.TestCase):
    def test_pair_blocks_before_minimum_and_keep_full_gate_distinct_from_pressure_only(self):
        def row(cid, whole, dd, blocks):
            result = dict(id=cid, scenarios={})
            for name, unused_lag, unused_fee in extended.SCENARIOS:
                values = blocks[:2] if name.startswith("close") else blocks[2:]
                result["scenarios"][name] = dict(full=dict(cagr=whole, max_dd=dd),
                    blocks={key: dict(cagr=value) for key, value in zip(("early", "late"), values)})
            return result
        rows = [row("center", .4, -.2, [.4, .2, .1, .3]),
                row("one", .35, -.25, [.3, .3, .2, .35]),
                row("two", .4, -.2, [.4, .2, .1, .3])]
        checks = {cid: dict(return_floor=True, main_drawdown=True,
                           **{key: True for key in extended.PRESSURE_KEYS}) for cid in ("center", "one", "two")}
        checks["two"]["return_floor"] = False
        result = extended.compare_to_center("center", ["one", "two"], rows, checks)
        # min(neighbor)-min(center) would be +0.10, incorrectly suggesting an
        # improvement. The minimum paired difference is actually -0.10.
        self.assertAlmostEqual(result["per_neighbor_worst_paired_delta"]["one"], -.1)
        self.assertAlmostEqual(result["worst_paired_pressure_block_delta_to_center"]["median"], -.05)
        self.assertAlmostEqual(result["worst_paired_pressure_block_delta_to_center"]["q25"], -.075)
        self.assertEqual(result["existing_H_gate_rate"], .5)
        self.assertEqual(result["pressure_only_rate"], 1.)
        main = result["scenarios"]["close_1bp"]
        self.assertAlmostEqual(main["cagr_delta_to_center"]["median"], -.025)
        self.assertAlmostEqual(main["cagr_drop_from_center"]["q75"], .0375)
        self.assertAlmostEqual(main["drawdown_worsening"]["median"], .025)
        self.assertTrue(result["no_selection_or_new_gate"])


if __name__ == "__main__":
    unittest.main()
