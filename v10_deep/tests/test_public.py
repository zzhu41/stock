"""Frozen public routing/integrity checks; no real cache build or backtest."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from v10_deep import cli
from v10_deep.data import sha
from v10_deep.schema import baseline, identifier, semantic


class PublicEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "v10_deep"
        self.base.mkdir()
        for relative, contents in {
            "v10_deep/frozen_source.py": "source\n",
            "dependency.py": "dependency\n",
            "v10_deep/frozen_selection.json": "selection\n",
            "v10_h_close/corrected_manifest.json": "corrected\n",
            "v10_h_close/qvix_manifest.json": "qvix\n",
            "v10_deep/cache/features.npz": "fixed arrays\n",
            "v10_deep/cache/features.json": json.dumps(dict(dates=[cli.START], score_names=["wls25_v20"])),
        }.items():
            path = self.root / relative
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(contents)
        config = baseline()
        config.update(hash=identifier(config), id="vd_" + identifier(config)[:20])
        self.profile = dict(name="test historical reference", config=config,
                            selected_by="registered reference, not a qualified S")
        self.profiles = dict(variants={name: deepcopy(self.profile) for name in ("growth", "simple", "v92")},
            source_sha256={"frozen_source.py": sha(self.base / "frozen_source.py")},
            dependency_sha256={"dependency.py": sha(self.root / "dependency.py")},
            artifact_sha256={"frozen_selection.json": sha(self.base / "frozen_selection.json")},
            input_manifests={name: sha(self.root / "v10_h_close" / filename) for name, filename in
                             (("corrected", "corrected_manifest.json"), ("qvix", "qvix_manifest.json"))},
            feature_cache_sha256=sha(self.base / "cache/features.npz"),
            feature_metadata_sha256=sha(self.base / "cache/features.json"), selected_s=None)
        self.write_profiles()
        self.inputs = Mock(return_value=({}, [], []))
        for name, value in (("BASE", self.base), ("ROOT", self.root), ("inputs", self.inputs)):
            handle = patch.object(cli, name, value)
            handle.start()
            self.addCleanup(handle.stop)

    def write_profiles(self):
        (self.base / "profiles.json").write_text(json.dumps(self.profiles))

    def test_each_source_dependency_artifact_and_input_manifest_is_checked(self):
        for relative in ("v10_deep/frozen_source.py", "dependency.py", "v10_deep/frozen_selection.json",
                         "v10_h_close/corrected_manifest.json", "v10_h_close/qvix_manifest.json"):
            with self.subTest(path=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                try:
                    with self.assertRaisesRegex(ValueError, "changed"):
                        cli.load_profile("growth")
                finally:
                    path.write_bytes(original)
        self.inputs.assert_not_called()

    def test_actual_csv_validation_still_runs_with_cached_arrays(self):
        self.inputs.side_effect = ValueError("Corrected data changed: 510300")
        with patch.object(cli, "build") as build:
            with self.assertRaisesRegex(ValueError, "Corrected data changed"):
                cli.evaluate("growth", cli.START)
            build.assert_not_called()

    def test_config_change_without_matching_semantic_hash_is_rejected(self):
        self.profiles["variants"]["growth"]["config"]["panic"] = .09
        self.write_profiles()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            cli.load_profile("growth")
        self.inputs.assert_not_called()

    def test_unknown_variant_fails_without_data_reads(self):
        with self.assertRaisesRegex(ValueError, "Unknown frozen variant"):
            cli.load_profile("qualified_s")
        self.inputs.assert_not_called()

    def test_changed_array_cache_never_reaches_simulator(self):
        (self.base / "cache/features.npz").write_bytes(b"changed arrays")
        with patch.object(cli, "build", return_value=({}, {"dates": [cli.START]})), \
             patch.object(cli, "Simulator") as simulator:
            with self.assertRaisesRegex(ValueError, "cache"):
                cli.evaluate("growth", cli.START)
            simulator.assert_not_called()

    def test_changed_metadata_cannot_relabel_pinned_arrays(self):
        # The NPZ remains byte-identical: relabeling score/date axes alone must
        # fail, rather than route a frozen configuration to another array row.
        original_npz = sha(self.base / "cache/features.npz")
        changed = {"dates": [cli.START], "score_names": ["another_score"]}
        (self.base / "cache/features.json").write_text(json.dumps(changed))
        self.assertEqual(original_npz, self.profiles["feature_cache_sha256"])
        with patch.object(cli, "build", return_value=({}, changed)), \
             patch.object(cli, "Simulator") as simulator:
            with self.assertRaisesRegex(ValueError, "[Mm]etadata"):
                cli.evaluate("growth", cli.START)
            simulator.assert_not_called()

    def test_changed_metadata_dates_are_rejected_before_build(self):
        expected = {"dates": [cli.START, "2014-01-03"], "score_names": ["wls25_v20"]}
        path = self.base / "cache/features.json"
        path.write_text(json.dumps(expected))
        self.profiles["feature_metadata_sha256"] = sha(path)
        self.write_profiles()
        expected["dates"].reverse()
        path.write_text(json.dumps(expected))
        with patch.object(cli, "build") as build, patch.object(cli, "Simulator") as simulator:
            with self.assertRaisesRegex(ValueError, "[Mm]etadata"):
                cli.evaluate("growth", cli.START)
            build.assert_not_called()
            simulator.assert_not_called()

    def test_cold_cache_may_build_only_the_exact_pinned_bytes(self):
        paths = [self.base / "cache/features.npz", self.base / "cache/features.json"]
        original = {path: path.read_bytes() for path in paths}
        for path in paths:
            path.unlink()  # These are disposable fixtures in TemporaryDirectory.

        def build():
            for path, content in original.items():
                path.write_bytes(content)
            return {}, {"dates": [cli.START]}

        with patch.object(cli, "build", side_effect=build) as rebuilt, patch.object(cli, "Simulator") as simulator:
            cli.evaluate("growth", cli.START)
            rebuilt.assert_called_once_with()
            simulator.assert_called_once()

    def test_cold_cache_rebuild_with_changed_metadata_is_rejected(self):
        paths = [self.base / "cache/features.npz", self.base / "cache/features.json"]
        original = {path: path.read_bytes() for path in paths}
        for path in paths:
            path.unlink()

        def build():
            for path, content in original.items():
                path.write_bytes(content)
            paths[1].write_text(json.dumps({"dates": [cli.START], "score_names": ["wrong"]}))
            return {}, {"dates": [cli.START]}

        with patch.object(cli, "build", side_effect=build), patch.object(cli, "Simulator") as simulator:
            with self.assertRaisesRegex(ValueError, "[Mm]etadata"):
                cli.evaluate("growth", cli.START)
            simulator.assert_not_called()

    def test_valid_route_passes_exact_frozen_configuration_and_observed_end(self):
        arrays, meta = {"dummy": "arrays"}, {"dates": [cli.START]}
        expected = {"dummy": "result"}
        with patch.object(cli, "build", return_value=(arrays, meta)), \
             patch.object(cli, "Simulator") as simulator:
            simulator.return_value.run.return_value = expected
            profile, result, actual_meta = cli.evaluate("simple", cli.START)
            self.assertEqual(result, expected)
            self.assertEqual(actual_meta, meta)
            simulator.assert_called_once_with(arrays, meta)
            simulator.return_value.run.assert_called_once_with([profile["config"]], end=cli.START, workers=1)
            self.inputs.assert_called_once_with()

    def test_nonobserved_or_future_end_never_runs_native(self):
        with patch.object(cli, "build", return_value=({}, {"dates": [cli.START]})), \
             patch.object(cli, "Simulator") as simulator:
            for end in ("2013-12-31", "2099-01-01", "2014-01-04"):
                with self.subTest(end=end), self.assertRaisesRegex(ValueError, "observed date"):
                    cli.evaluate("growth", end)
            simulator.assert_not_called()

    def test_public_signal_is_historical_and_simple_is_not_renamed_s(self):
        fake_result = dict(dates=[cli.START], returns=np.zeros((1, 1)),
                           holdings=np.array([[0]]), summary=np.zeros((1, 10)))
        meta = dict(assets=["510300"])
        self.profile["name"] = "单项简化研究对照"
        self.profile["selected_by"] = "Registered min-hold2 reference; not an independently validated S"
        self.profile["research_status"] = "Explanatory single-change comparison; not a qualified new S"
        with patch.object(cli, "evaluate", return_value=(self.profile, fake_result, meta)) as evaluate:
            for command in ("signal", "backtest"):
                output = io.StringIO()
                with patch("sys.argv", ["cli", command, "simple", "--end", cli.START]), redirect_stdout(output):
                    cli.main()
                result = json.loads(output.getvalue())
                self.assertEqual(result["version"], "单项简化研究对照")
                self.assertIn("not an independently validated S", result["selected_by"])
                self.assertEqual(result["research_status"], self.profile["research_status"])
                self.assertTrue(result["not_deployed"])
                self.assertFalse(result["order_submission"])
                self.assertFalse(result["clean_oos"])
                self.assertIn("same-close", result["clock"])
                if command == "signal":
                    self.assertEqual(result["historical_close_target"], "510300")
                    self.assertIn("not a current intraday order", result["warning"])
                else:
                    self.assertIn("metrics", result)
                evaluate.assert_called_with("simple", cli.START)


class ActualProfileSemantics(unittest.TestCase):
    def test_published_growth_and_simple_rules_match_their_descriptions(self):
        # Small JSON read only: no native engine, matrix or feature cache load.
        path = Path(__file__).resolve().parents[1] / "profiles.json"
        profiles = json.loads(path.read_text())
        self.assertIsNone(profiles["selected_s"])
        self.assertFalse(profiles["live_enabled"])
        self.assertFalse(profiles["clean_oos"])
        self.assertIn("not a qualified new S", profiles["variants"]["simple"]["research_status"])
        self.assertEqual(profiles["variants"]["v92"]["research_status"], "Existing v9.2 reference")
        base = semantic(profiles["variants"]["v92"]["config"])
        growth = semantic(profiles["variants"]["growth"]["config"])
        simple = semantic(profiles["variants"]["simple"]["config"])
        self.assertEqual({k: v for k, v in growth.items() if v != base[k]},
                         dict(score="wls20_smooth3", ma="ma180", panic_mode="volatility", panic=1.5))
        self.assertEqual({k: v for k, v in simple.items() if v != base[k]}, dict(min_hold=2))
        for profile in profiles["variants"].values():
            self.assertEqual(profile["config"]["lag"], 0)
            self.assertEqual(identifier(profile["config"]), profile["config"]["hash"])


if __name__ == "__main__":
    unittest.main()
