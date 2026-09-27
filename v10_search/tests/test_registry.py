"""Enumeration/configuration tests only; no prices, backtests or selection."""
from copy import deepcopy
import json
import unittest

from v10_search.registry import (
    ASSET_CATALOG, BASE_STOCK, CASH, EXTRA_GLOBAL, FIXED_INPUT_ASSET_ORDER, GLOBAL_SUBSETS, PRESETS,
    build_registry, candidate_hash, controls, required_codes,
)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = build_registry()
        cls.candidates = cls.registered["candidates"]
        cls.presets = json.loads(PRESETS.read_text(encoding="utf-8"))

    def test_bounded_size_family_counts_and_duplicate_provenance(self):
        self.assertEqual(self.registered["raw_count"], 5164)
        self.assertEqual(self.registered["unique_count"], 5156)
        self.assertEqual(len(self.registered["duplicates"]), 8)
        expected = dict(pool_subsets=3048, rule_grid=1016, extra_stock_single=432,
                        extra_global_single=192, fixed_baskets=144, ranking_ablation=36,
                        regime_ablation=288)
        self.assertEqual({f: n["first_unique"] for f, n in self.registered["family_counts"].items()}, expected)
        self.assertEqual(self.registered["family_counts"]["rule_grid"]["raw"], 1024)
        by_id = {c["id"]: c for c in self.candidates}
        for duplicate in self.registered["duplicates"]:
            kept = by_id[duplicate["kept_id"]]
            self.assertIn("pool_subsets", kept["families"])
            self.assertIn("rule_grid", kept["families"])
            self.assertEqual(kept["candidate_hash"], duplicate["candidate_hash"])

    def test_all_parameters_pools_roles_and_channels_are_explicit(self):
        keys = set(self.presets["v9.1"])
        self.assertEqual(len(ASSET_CATALOG), 24)
        for candidate in self.candidates:
            self.assertEqual(set(candidate["params"]), keys)
            self.assertTrue(candidate["stock_pool"])
            self.assertTrue(candidate["global_pool"])
            self.assertFalse(set(candidate["stock_pool"]) & set(candidate["global_pool"]))
            self.assertEqual(candidate["risk_weight"], 1.0)
            self.assertEqual(candidate["params"]["leverage"], 1.0)
            self.assertEqual(candidate["cash"], CASH)
            self.assertEqual(candidate["benchmark_code"], "510300")
            self.assertEqual(candidate["tie_break"], "fixed_input_asset_order")
            self.assertEqual(tuple(candidate["input_asset_order"]), FIXED_INPUT_ASSET_ORDER)
            self.assertEqual(candidate["channels"], ["deep"] if candidate["params"]["crash_mom5"] < 0 else [])
            self.assertEqual(set(candidate["asset_roles"]), set(required_codes(candidate)))
        # The public registration is strict JSON, including disabled parameters.
        self.assertEqual(json.loads(json.dumps(self.registered, allow_nan=False))["unique_count"], 5156)

    def test_regime_ablation_distinguishes_open_stock_from_no_regime_decision(self):
        rows = [c for c in self.candidates if c["family"] == "regime_ablation"]
        self.assertEqual(len(rows), 3 * 2 * 3 * 4 * 2 * 2)
        self.assertEqual({c["regime_mode"] for c in rows}, {"open_stock", "always_bull", "all_assets"})
        self.assertEqual({c["params"]["never_empty"] for c in rows}, {False, True})
        for candidate in rows:
            mode = candidate["regime_mode"]
            self.assertEqual(candidate["params"]["bear_open_stock"], mode == "open_stock")
            self.assertEqual(candidate["params"]["bear_enter_mom"], .07)
            self.assertEqual(candidate["regime_uses_benchmark"], mode == "open_stock")
            self.assertEqual(candidate["benchmark_purpose"],
                             "regime_and_calendar" if mode == "open_stock" else "calendar_only")
            self.assertIn("510300", required_codes(candidate))
        for candidate in self.candidates:
            if candidate["family"] != "regime_ablation":
                self.assertEqual(candidate["regime_mode"], "ma250")
        for reference in controls():
            self.assertEqual(reference["regime_mode"], "ma250")

    def test_power_set_family_is_exhaustive_for_the_registered_axes(self):
        rows = [c for c in self.candidates if c["family"] == "pool_subsets"]
        stocks = {tuple(c["stock_pool"]) for c in rows}
        self.assertEqual(len(stocks), 127)
        self.assertTrue(all(set(pool) <= set(BASE_STOCK) and pool for pool in stocks))
        self.assertEqual({tuple(c["global_pool"]) for c in rows}, {tuple(sorted(x)) for x in GLOBAL_SUBSETS})
        self.assertEqual({tuple(c["score_windows"]) for c in rows}, {(20,), (25,), (30,), (40,)})
        self.assertEqual({c["params"]["crash_mom5"] for c in rows}, {0.0, -.08})
        tuples = {(tuple(c["stock_pool"]), tuple(c["global_pool"]), tuple(c["score_windows"]),
                   c["params"]["crash_mom5"]) for c in rows}
        self.assertEqual(len(tuples), 127 * 3 * 4 * 2)

    def test_benchmark_remains_observable_after_removal_from_trading_pool(self):
        candidate = next(c for c in self.candidates if c["stock_pool"] == ["159915"])
        self.assertNotIn("510300", candidate["stock_pool"] + candidate["global_pool"])
        self.assertIn("510300", required_codes(candidate))
        self.assertEqual(candidate["asset_roles"]["510300"], "benchmark_only")
        self.assertIn("not eligible", self.registered["role_descriptions"]["benchmark_only"])

    def test_ranking_ablation_keeps_entry_momentum_and_disables_wls(self):
        for candidate in (c for c in self.candidates if c["family"] == "ranking_ablation"):
            mode, params = candidate["score_mode"], candidate["params"]
            self.assertEqual(candidate["score_windows"], [])
            self.assertFalse(params["score_wls"])
            self.assertEqual(params["mom_main"], 20)
            self.assertEqual(params["score_mom"], 60 if mode == "mom60_vol" else 0)
            self.assertEqual(params["score_plain_mom"], mode == "mom20")

    def test_unconstrained_global_role_does_not_mislabel_bonds_or_commodities(self):
        self.assertEqual(ASSET_CATALOG["511010"]["asset_class"], "government_bond")
        self.assertEqual(ASSET_CATALOG["159985"]["asset_class"], "commodity")
        self.assertTrue(all(ASSET_CATALOG[c]["role"] == "global" for c in EXTRA_GLOBAL))
        rows = [c for c in self.candidates if c["family"] == "extra_global_single"]
        self.assertTrue(any("511010" in c["global_pool"] for c in rows))
        self.assertIn("not geographical", " ".join(self.registered["caveats"]))

    def test_hashes_are_semantic_unique_and_ignore_metadata_and_pool_order(self):
        self.assertEqual(len({c["candidate_hash"] for c in self.candidates}), 5156)
        self.assertEqual(len({c["id"] for c in self.candidates}), 5156)
        original = next(c for c in self.candidates if len(c["stock_pool"]) == 7)
        edited = deepcopy(original)
        edited["id"] = "different-label"
        edited["family"] = "different-family"
        edited["complexity"] = {"note": "metadata only"}
        edited["stock_pool"].reverse()
        self.assertEqual(candidate_hash(edited), original["candidate_hash"])
        edited["params"]["panic_drop"] = .031
        self.assertNotEqual(candidate_hash(edited), original["candidate_hash"])
        regime_changed = deepcopy(original)
        regime_changed["regime_mode"] = "always_bull"
        self.assertNotEqual(candidate_hash(regime_changed), original["candidate_hash"])
        for candidate in self.candidates:
            self.assertEqual(candidate_hash(candidate), candidate["candidate_hash"])
            self.assertEqual(len(candidate["trade_pool_hash"]), 64)

    def test_controls_share_schema_and_duplicates_do_not_add_trials(self):
        references = controls()
        self.assertEqual([c["id"] for c in references], ["c_v9", "c_v91", "c_v92"])
        search_hashes = {c["candidate_hash"] for c in self.candidates}
        self.assertIn(references[0]["candidate_hash"], search_hashes)
        self.assertIn(references[1]["candidate_hash"], search_hashes)
        self.assertNotIn(references[2]["candidate_hash"], search_hashes)
        self.assertEqual(references[0]["params"]["pool_buffer"], {})
        self.assertEqual(references[2]["channels"], ["deep", "qvix", "volume"])
        for reference in references:
            self.assertTrue(reference["is_control"])
            self.assertFalse(reference["counts_as_independent_extra_control_trial"])
            self.assertEqual(set(reference["params"]), set(self.presets["v9.1"]))

    def test_regeneration_is_deterministic_and_has_no_mutable_parameter_sharing(self):
        other = build_registry()
        self.assertEqual(other["registry_sha256"], self.registered["registry_sha256"])
        self.assertEqual([c["id"] for c in other["candidates"]], [c["id"] for c in self.candidates])
        other["candidates"][0]["params"]["pool_buffer"]["stock"] = .99
        self.assertEqual(other["candidates"][1]["params"]["pool_buffer"]["stock"], .02)
        self.assertEqual(self.candidates[0]["params"]["pool_buffer"]["stock"], .02)


if __name__ == "__main__":
    unittest.main()
