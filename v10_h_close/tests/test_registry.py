"""Configuration-only tests for the fixed same-close H search; no backtests."""
from copy import deepcopy
import json
import unittest

from v10_h_close.registry import (
    BUFFERS, CHANNEL_SUBSETS, EXECUTION_SPEC, PRESETS, SCORE_WINDOWS,
    build_registry, candidate_hash, controls, required_codes, semantic_payload,
)
from v10_search.registry import build_registry as build_parent_registry


class CloseRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = build_registry()
        cls.candidates = cls.registered["candidates"]
        cls.by_hash = {c["candidate_hash"]: c for c in cls.candidates}
        cls.parent = build_parent_registry()
        cls.parent_by_id = {c["id"]: c for c in cls.parent["candidates"]}
        cls.base_params = json.loads(PRESETS.read_text())["v9.1"]

    def test_registered_counts_and_cross_stage_deduplication(self):
        self.assertEqual(self.registered["raw_count"], 6596)
        self.assertEqual(self.registered["unique_count"], 6544)
        self.assertEqual(len(self.registered["duplicates"]), 52)
        self.assertEqual(self.registered["stage_counts"]["wide"],
                         {"raw": 5156, "first_unique": 5156, "memberships": 5156})
        self.assertEqual(self.registered["stage_counts"]["focused"],
                         {"raw": 1440, "first_unique": 1388, "memberships": 1440})
        self.assertEqual(len(self.by_hash), 6544)
        self.assertEqual(len({c["id"] for c in self.candidates}), 6544)
        for duplicate in self.registered["duplicates"]:
            kept = self.by_hash[duplicate["candidate_hash"]]
            self.assertEqual(duplicate["stage"], "focused")
            self.assertEqual(kept["stage"], "wide")
            self.assertEqual(set(kept["stages"]), {"wide", "focused"})
            self.assertEqual(duplicate["kept_id"], kept["id"])

    def test_wide_reuses_every_old_semantic_config_under_an_explicit_new_id(self):
        wide = [c for c in self.candidates if c["stage"] == "wide"]
        self.assertEqual({c["parent_candidate_id"] for c in wide}, set(self.parent_by_id))
        self.assertEqual(self.registered["parent"]["registry_sha256"], self.parent["registry_sha256"])
        for candidate in wide:
            original = self.parent_by_id[candidate["parent_candidate_id"]]
            self.assertEqual(semantic_payload(candidate), semantic_payload(original))
            self.assertEqual(candidate["candidate_hash"], original["candidate_hash"])
            self.assertNotEqual(candidate["id"], original["id"])
            self.assertTrue(candidate["id"].startswith("hc_"))
            self.assertTrue(candidate["previously_seen"])
            self.assertFalse(candidate["new_configuration_vs_parent_registry"])
            self.assertEqual(candidate["parent_registration_sources"], original["registration_sources"])

    def test_focused_cartesian_axes_are_complete_and_coordinates_are_retained(self):
        focused = [c for c in self.candidates if "focused" in c["stages"]]
        self.assertEqual(len(focused), 3 * 2 * 6 * 5 * 8)
        coordinates = [source["coordinates"] for c in focused for source in c["registration_sources"]
                       if source["stage"] == "focused"]
        self.assertEqual(len(coordinates), 1440)
        self.assertEqual(len({json.dumps(row, sort_keys=True) for row in coordinates}), 1440)
        self.assertEqual({tuple(c["score_windows"]) for c in focused}, set(SCORE_WINDOWS))
        self.assertEqual({tuple(c["channels"]) for c in focused}, set(CHANNEL_SUBSETS))
        self.assertEqual({row["buffer"] for row in coordinates}, {name for name, _ in BUFFERS})
        self.assertEqual(len({tuple(c["stock_pool"]) for c in focused}), 3)
        self.assertEqual({tuple(c["global_pool"]) for c in focused}, {("513100",), ("513100", "513120")})

    def test_uniform_buffers_are_effective_and_other_focused_rules_stay_fixed(self):
        for candidate in (c for c in self.candidates if "focused" in c["stages"]):
            params = candidate["params"]
            source = next(s for s in candidate["registration_sources"] if s["stage"] == "focused")
            setting = source["coordinates"]["buffer"]
            if setting == "original_split_2_3_3":
                self.assertEqual(params["buffer"], .02)
                self.assertEqual(params["pool_buffer"], {"stock": .02, "global": .03, "gold": .03})
            else:
                self.assertEqual(params["pool_buffer"], {})
                self.assertEqual(params["buffer"], dict(BUFFERS)[setting])
            self.assertEqual(candidate["regime_mode"], "ma250")
            self.assertFalse(params["bear_open_stock"])
            self.assertEqual(params["panic_drop"], .04)
            self.assertEqual(params["bear_enter_mom"], .07)
            self.assertEqual(params["overheat"], .4)
            self.assertTrue(params["never_empty"])
            self.assertEqual(params["crash_lock"], 5)
            self.assertEqual(params["crash_below_ma"], .2)

    def test_external_channels_remain_distinct_when_the_deep_trigger_is_off(self):
        for candidate in (c for c in self.candidates if "focused" in c["stages"]):
            channels = set(candidate["channels"])
            self.assertEqual(candidate["params"]["crash_mom5"], -.08 if "deep" in channels else 0.0)
            self.assertEqual(candidate["kind"], "v92" if channels & {"qvix", "volume"} else "legacy")
            self.assertEqual(candidate["score_mode"], "wls")
            self.assertTrue(candidate["params"]["score_wls"])
            self.assertEqual(candidate["complexity"]["score_inputs"], len(candidate["score_windows"]))
        fixed = self.registered["channel_definitions"]
        self.assertEqual((fixed["qvix"]["z_threshold"], fixed["qvix"]["previous_window"]), (2.5, 250))
        self.assertEqual((fixed["volume"]["ratio_threshold"], fixed["volume"]["previous_volume_window"]), (2., 20))

    def test_complete_params_hashes_pools_and_novelty_scope_are_explicit(self):
        for candidate in self.candidates:
            self.assertEqual(set(candidate["params"]), set(self.base_params))
            self.assertEqual(candidate_hash(candidate), candidate["candidate_hash"])
            self.assertEqual(candidate["risk_weight"], 1.0)
            self.assertIn("510300", required_codes(candidate))
            self.assertIn("511880", required_codes(candidate))
            self.assertIn("not an audit of all older research", candidate["previously_seen_scope"])
        altered = deepcopy(self.candidates[0])
        altered["stage"] = "another_label"
        altered["parent_candidate_id"] = "another_label"
        self.assertEqual(candidate_hash(altered), self.candidates[0]["candidate_hash"])
        self.assertEqual(self.registered["candidate_id_to_hash"][altered["id"]], altered["candidate_hash"])

    def test_controls_are_fixed_references_and_known_v92_is_not_claimed_novel(self):
        references = controls()
        self.assertEqual([c["id"] for c in references], ["c_v9", "c_v91", "c_v92"])
        for reference in references:
            self.assertTrue(reference["reference_only"])
            self.assertTrue(reference["previously_seen"])
            self.assertEqual(candidate_hash(reference), reference["candidate_hash"])
            matched = self.by_hash[reference["candidate_hash"]]
            self.assertIn(reference["id"], matched["prior_control_ids"])
            self.assertTrue(matched["previously_seen"])
        matched_v92 = self.by_hash[references[-1]["candidate_hash"]]
        self.assertIsNone(matched_v92["parent_candidate_id"])
        self.assertEqual(matched_v92["stage"], "focused")

    def test_original_fee_clock_is_explicit_and_no_cutoff_is_chosen_by_generator(self):
        self.assertEqual(self.registered["execution_spec"], EXECUTION_SPEC)
        self.assertEqual(EXECUTION_SPEC["clock"], "original_same_close")
        self.assertEqual(EXECUTION_SPEC["commission_per_side"], .0001)
        self.assertEqual(EXECUTION_SPEC["switch_nav_multiplier"], .9998)
        self.assertEqual(EXECUTION_SPEC["initial_entry_fee"], 0.0)
        self.assertEqual(EXECUTION_SPEC["slippage_per_side"], 0.0)
        self.assertEqual(EXECUTION_SPEC["annualization_trading_days"], 244)
        self.assertNotIn("end", self.registered)
        self.assertNotIn("end", EXECUTION_SPEC)
        json.dumps(self.registered, allow_nan=False)

    def test_returned_configs_do_not_share_nested_mutable_parameter_state(self):
        a, b = deepcopy(self.candidates[:2])
        a["params"]["pool_buffer"]["stock"] = .99
        self.assertEqual(b["params"]["pool_buffer"]["stock"], .02)
        self.assertEqual(self.candidates[0]["params"]["pool_buffer"]["stock"], .02)
        first, second = controls(), controls()
        first[1]["params"]["pool_buffer"]["stock"] = .88
        self.assertEqual(second[1]["params"]["pool_buffer"]["stock"], .02)


if __name__ == "__main__":
    unittest.main()
