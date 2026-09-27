"""Public research profiles: frozen semantics, honest labels and dated signals.

These tests use small temporary manifests and mocks, never a return matrix or
the registered market-data search. Existing profiles and sources are untouched.
"""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from v10_search import cli, report
from v10_search.registry import BASE_STOCK, PRESETS, _candidate, candidate_hash, required_codes


METHODS = {
    "growth": "development_2014_2021_only",
    "exploratory": "full_history_2014_2026_hindsight",
    "regime": "post_search_best_open_stock_mode_on_full_history",
}


def configurations():
    base = json.loads(PRESETS.read_text())["v9.1"]
    return {
        "growth": _candidate(base, "synthetic", BASE_STOCK + ("512200",), ("513100", "513120")),
        "exploratory": _candidate(base, "synthetic", ("159915",), ("513100",),
                                   crash=False, mode="mom60_vol"),
        "regime": _candidate(base, "synthetic", ("159915", "510500"), ("513100",), window=40,
                              regime_mode="open_stock", overrides={"panic_drop": .05, "overheat": 9.9,
                                                                   "never_empty": False, "pool_buffer": {}}),
    }


def profiles_for(configs):
    return {variant: dict(name="fixture_" + variant, selected_by=METHODS[variant],
                          status="research-only_" + variant, config=deepcopy(config),
                          candidate_hash=candidate_hash(config), not_deployed=True, clean_oos=False)
            for variant, config in configs.items()}


class PublicTests(unittest.TestCase):
    def fixture(self, directory):
        parent = Path(directory)
        root = parent / "v10_search"
        (root / "results").mkdir(parents=True)
        (parent / "v10_next").mkdir()
        (root / "api.py").write_text("API = 1\n")
        (parent / "v10_next/core.py").write_text("CORE = 1\n")
        for name in ("data_manifest.json", "results/selection.json", "results/evaluation.json"):
            (root / name).write_text("{}\n")
        frozen = dict(
            variants=profiles_for(configurations()),
            source_sha256={"api.py": cli.sha(root / "api.py")},
            dependencies_sha256={"v10_next/core.py": cli.sha(parent / "v10_next/core.py")},
            artifact_sha256={name: cli.sha(root / name) for name in
                             ("data_manifest.json", "results/selection.json", "results/evaluation.json")})
        (root / "profiles.json").write_text(json.dumps(frozen))
        return root, parent, frozen

    def test_load_profile_preserves_three_distinct_frozen_labels_and_returns_a_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, frozen = self.fixture(directory)
            with patch.object(cli, "BASE", root):
                for variant in METHODS:
                    loaded = cli.load_profile(variant)
                    self.assertEqual(loaded["selected_by"], METHODS[variant])
                    self.assertEqual(loaded["config"]["id"], frozen["variants"][variant]["config"]["id"])
                    self.assertTrue(loaded["not_deployed"])
                    self.assertFalse(loaded["clean_oos"])
                loaded = cli.load_profile("regime")
                loaded["config"]["params"]["panic_drop"] = .99
                self.assertEqual(cli.load_profile("regime")["config"]["params"]["panic_drop"], .05)
                with self.assertRaisesRegex(ValueError, "No frozen research profile"):
                    cli.load_profile("unknown")

    def test_source_dependency_artifact_and_full_semantic_config_hashes_are_enforced(self):
        cases = (("own", "Search implementation changed"), ("dependency", "Frozen dependency changed"),
                 ("selection", "Search result/snapshot manifest changed"),
                 ("evaluation", "Search result/snapshot manifest changed"),
                 ("manifest", "Search result/snapshot manifest changed"),
                 ("params", "Frozen strategy configuration changed"),
                 ("regime", "Frozen strategy configuration changed"),
                 ("score", "Frozen strategy configuration changed"))
        for target, message in cases:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root, parent, frozen = self.fixture(directory)
                if target in ("params", "regime", "score"):
                    config = frozen["variants"]["regime"]["config"]
                    if target == "params":
                        config["params"]["never_empty"] = True
                    elif target == "regime":
                        config["regime_mode"] = "always_bull"
                    else:
                        config["score_windows"] = [25]
                    (root / "profiles.json").write_text(json.dumps(frozen))
                else:
                    changed = {"own": root / "api.py", "dependency": parent / "v10_next/core.py",
                               "selection": root / "results/selection.json",
                               "evaluation": root / "results/evaluation.json",
                               "manifest": root / "data_manifest.json"}[target]
                    changed.write_text("changed\n")
                with patch.object(cli, "BASE", root), self.assertRaisesRegex(ValueError, message):
                    cli.load_profile("regime")

    def test_profile_generation_does_not_promote_hindsight_or_regime_to_growth(self):
        configs = configurations()
        by_id = {config["id"]: config for config in configs.values()}
        fingerprints = dict(source_sha256={}, dependencies_sha256={})
        evaluation = dict(fingerprints=fingerprints, selection={"champion": configs["growth"]["id"]},
                          hindsight_winner={"id": configs["exploratory"]["id"]})
        followup = {"candidate": {"id": configs["regime"]["id"]}}
        with patch.object(report, "fingerprints", return_value=fingerprints), \
                patch.object(report, "sha", return_value="fixture-hash"):
            generated = report.profiles(evaluation, by_id, followup)
            for variant, method in METHODS.items():
                self.assertEqual(generated["variants"][variant]["config"]["id"], configs[variant]["id"])
                self.assertEqual(generated["variants"][variant]["selected_by"], method)
            self.assertIn("failed later", generated["variants"]["growth"]["status"])
            self.assertIn("no clean OOS", generated["variants"]["exploratory"]["status"])
            self.assertIn("not the frozen development champion", generated["variants"]["regime"]["status"])
            evaluation["selection"]["champion"] = None
            no_champion = report.profiles(evaluation, by_id, followup)
            self.assertNotIn("growth", no_champion["variants"])
            self.assertEqual(set(no_champion["variants"]), {"exploratory", "regime"})

    def test_required_observations_include_added_asset_cash_and_excluded_benchmark(self):
        configs = configurations()
        self.assertIn("512200", required_codes(configs["growth"]))
        for config in configs.values():
            codes = required_codes(config)
            self.assertIn("511880", codes)
            self.assertIn("518880", codes)
            self.assertIn("510300", codes)
            self.assertEqual(len(codes), len(set(codes)))
        exploratory = configs["exploratory"]
        self.assertNotIn("510300", exploratory["stock_pool"])
        self.assertEqual(exploratory["asset_roles"]["510300"], "benchmark_only")

    def test_evaluate_passes_complete_frozen_params_mode_and_score_without_defaults(self):
        for variant, profile in profiles_for(configurations()).items():
            config = profile["config"]
            histories = {code: [("2026-09-11", 1., 1., 100.)] for code in required_codes(config)}
            bank, frame = object(), object()
            policy = MagicMock(metadata={"fixture": variant})
            result = {"final_state": {"date": "2026-09-11"}}
            with self.subTest(variant=variant), \
                    patch.object(cli, "load_profile", return_value=profile), \
                    patch.object(cli, "read_prices", return_value=histories) as prices, \
                    patch.object(cli, "load_histories", side_effect=AssertionError("must use explicit snapshots")), \
                    patch.object(cli, "load_qvix", side_effect=AssertionError("must not load old fear feed")), \
                    patch.object(cli, "FeatureBank", return_value=bank), \
                    patch.object(cli, "prepare", return_value=frame), \
                    patch.object(cli, "SearchPolicy", return_value=policy) as factory, \
                    patch.object(cli, "run", return_value=result) as execute:
                returned, actual = cli.evaluate(variant, end="2026-09-11", data_dir="explicit-fixture")
            self.assertEqual(returned, profile)
            self.assertEqual(factory.call_args[0][0], config)
            self.assertEqual(factory.call_args[0][0]["params"], config["params"])
            self.assertEqual(factory.call_args[0][0]["regime_mode"], config["regime_mode"])
            self.assertEqual(factory.call_args[0][0]["score_mode"], config["score_mode"])
            self.assertEqual(factory.call_args[0][0]["score_windows"], config["score_windows"])
            self.assertEqual(set(prices.call_args[0][1]), set(required_codes(config)))
            self.assertEqual(execute.call_args[1]["fee"], .0001)
            self.assertEqual(execute.call_args[1]["slippage"], .001)
            self.assertEqual(actual["policy_metadata"], {"fixture": variant})

    def test_future_dates_and_unregistered_external_fear_channels_are_rejected(self):
        profile = profiles_for(configurations())["growth"]
        histories = {c: [("2026-09-11", 1., 1., 100.)] for c in required_codes(profile["config"])}
        with patch.object(cli, "load_profile", return_value=profile), \
                patch.object(cli, "read_prices", return_value=histories), \
                patch.object(cli, "FeatureBank") as bank, \
                patch.object(cli, "prepare"), patch.object(cli, "SearchPolicy") as policy:
            with self.assertRaisesRegex(ValueError, "dated observation"):
                cli.evaluate("growth", end="2026-09-25", data_dir="fixture")
            bank.assert_not_called()
            profile["config"]["channels"] = ["deep", "qvix"]
            with self.assertRaisesRegex(ValueError, "old external fear feed"):
                cli.evaluate("growth", end="2026-09-11", data_dir="fixture")
            policy.assert_not_called()

    def test_signals_keep_historical_date_non_deployment_and_selection_labels(self):
        result = {"final_state": dict(date="2026-09-11", weights={"513100": 1.},
                                      pending_target=None, pending_signal_date=None)}
        for variant, profile in profiles_for(configurations()).items():
            output = io.StringIO()
            with self.subTest(variant=variant), \
                    patch.object(cli, "evaluate", return_value=(profile, result)), \
                    patch("sys.argv", ["v10_search.cli", "signal", variant, "--end", "2026-09-11"]), \
                    contextlib.redirect_stdout(output):
                cli.main()
            message = json.loads(output.getvalue())
            self.assertEqual(message["date"], "2026-09-11")
            self.assertEqual(message["selected_by"], METHODS[variant])
            self.assertEqual(message["candidate_id"], profile["config"]["id"])
            self.assertTrue(message["not_deployed"])
            self.assertFalse(message["clean_oos"])
            self.assertIn("not real-time quotes or actual brokerage positions", message["source"])


if __name__ == "__main__":
    unittest.main()
